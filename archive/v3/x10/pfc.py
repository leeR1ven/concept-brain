"""PFC [P1-P12]: L1 = one copy per chain source; L2, L3 = fixed random local middle layers; L4 = last layer.
L4(t) -> L1(t+1): birth loop (msg 84) + acquired prediction links (local, two-layer weights).
L4 <-> L4: birth lateral + acquired lateral. Fixed per-layer thresholds, calibrated once at birth for count
preservation; optional same-tick inhibitory neuron per patch (P1). An internally lit copy counts fully."""
import torch

import geom
from common import DEV
from learning import topk_pos, ByPost, ByPre, Pop


class PFC:
    def __init__(self, c, pos1, gam, seed):
        self.c = c
        n1 = len(pos1)
        n = c['rho'] * n1                                   # [P2] rho = 1: L2 = L3 = L4 = L1
        self.n1, self.n = n1, n
        g = torch.Generator(device=DEV)
        g.manual_seed(seed + 500)
        self.pos = [pos1] + [torch.rand((n, 2), generator=g, device=DEV) for _ in range(3)]
        F, r = c['ff_fan'], c['r_ff']
        self.ff = []
        for l in range(3):                                  # L1->L2, L2->L3, L3->L4
            idx, w = geom.local_fanin(self.pos[l], self.pos[l + 1], F, r, seed + 510 + l)
            self.ff.append(ByPost(idx, w, len(self.pos[l]), n, f'PFC ff L{l + 1}->L{l + 2}'))
        # [P6] birth loop L4 -> L1
        if c['loop'] == 'recip':
            li, lw = geom.reciprocal_loop(self.ff[2].idx, self.ff[2].w, self.ff[1].idx, self.ff[1].w,
                                          self.ff[0].idx, self.ff[0].w, n1, c['loop_fan'], seed + 520)
        else:
            li, lw = geom.random_local_loop(self.pos[3], pos1, c['loop_fan'], c['r_learn'], seed + 520)
        self.loop = ByPre(li, lw * c['g_loop'], n, n1, 'PFC birth loop L4->L1', birth=True)
        # [P8] birth lateral L4 <-> L4 (random local, no self)
        cand = geom.radius_lists(self.pos[3], self.pos[3], c['r_learn'], exclude_self=True)
        key = torch.rand(cand.shape, generator=g, device=DEV) + (cand >= n).float() * 2.0
        pick = torch.topk(key, min(c['lat_fan'], cand.shape[1]), dim=1, largest=False).indices
        lat_i = torch.gather(cand, 1, pick)
        lat_w = (0.5 + torch.rand(lat_i.shape, generator=g, device=DEV)) * (lat_i < n)
        lat_w = lat_w / lat_w.sum(1, keepdim=True).clamp_min(1e-9)
        self.lat = ByPre(lat_i, lat_w * c['w44'], n, n, 'PFC birth lateral L4<->L4', birth=True)
        # [P7, P8] acquired tables: every unit within r_learn is a potential synapse (count grows with s)
        self.L41 = ByPre(geom.radius_lists(self.pos[3], pos1, c['r_learn']), 0.0, n, n1, 'PFC L4->L1 acquired',
                         birth=False, eta=c['eta'], eta_m=c['eta_m'], gain=c['g31'], gamma=gam['pfc'], c=c)
        self.L44 = ByPre(cand, 0.0, n, n, 'PFC L4<->L4 acquired', birth=False, eta=c['eta'], eta_m=c['eta_m'],
                         gain=c['g44'], gamma=gam['pfc'], c=c)
        # [P4] patches (one non-learning inhibitory neuron each)
        self.patch = [geom.patches(p, c['patch'], seed + 530 + l) for l, p in enumerate(self.pos)]
        self.theta = [c['theta1'], None, None, None]
        self.g_l = [c['g_pm'], None, None, None]                 # patch-neuron gain per layer (L1: internal drive)
        self.G1P = None                                     # gen-0 speech wiring, set by g1p.py (acquired part)
        self.L1, self.L1ext = Pop(n1), Pop(n1)
        self.L2, self.L3, self.L4 = Pop(n), Pop(n), Pop(n)
        self.x_int = torch.zeros(n1, dtype=torch.bool, device=DEV)

    def tables(self):
        return self.ff + [self.loop, self.lat, self.L41, self.L44] + ([self.G1P] if self.G1P is not None else [])

    def plastic(self):
        return [self.L41, self.L44] + ([self.G1P] if self.G1P is not None else [])

    # ------------------------------------------------------------------ inhibition [P4]
    def inhib(self, d, l, g=None):
        g = self.g_l[l] if g is None else g
        if self.c['inhib'] == 'P0' or not g:
            return d
        pid, n_p = self.patch[l]
        if d.dim() == 1:
            s = torch.zeros(n_p, device=DEV).index_add_(0, pid, d)
            cnt = torch.zeros(n_p, device=DEV).index_add_(0, pid, torch.ones_like(d))
            return d - g * (s / cnt.clamp_min(1))[pid]
        s = torch.zeros((d.shape[0], n_p), device=DEV).index_add_(1, pid, d)
        cnt = torch.zeros(n_p, device=DEV).index_add_(0, pid, torch.ones(d.shape[1], device=DEV))
        return d - g * (s / cnt.clamp_min(1))[:, pid]

    # ------------------------------------------------------------------ birth calibration [P3]
    def calibrate(self, X):
        """X: bool [T, n1] chain input stream of the birth period (loop, lateral and learning off).
        Per layer, a fixed threshold theta_l and (P1) the patch neuron's gain g_l, chosen once so that the layer's
        count tracks the count of the layer below tick by tick (least squares over the birth ticks; P0: g = 0).
        (Arm rho4: L2..L4 are rho x larger at the same count, i.e. sparser.) Returns mean counts."""
        cnt_in = X.float().sum(1)
        counts = [cnt_in.mean().item()]
        x = X
        gains = [0.0] if self.c['inhib'] == 'P0' else [0.0, 0.25, 0.5, 0.75, 1.0]
        for l in range(3):
            T = x.shape[0]
            Draw = torch.empty((T, self.n), device=DEV)
            for a in range(0, T, 64):
                Draw[a:a + 64] = self.ff[l].drive_batch(x[a:a + 64].float())
            best = None
            for g in gains:
                D = torch.empty_like(Draw)
                for a in range(0, T, 256):
                    D[a:a + 256] = self.inhib(Draw[a:a + 256], l + 1, g)
                lo, hi = float(D.min()), float(D.max())
                tgt = cnt_in.mean().item()
                for _ in range(40):
                    mid = 0.5 * (lo + hi)
                    if (D > mid).float().sum(1).mean().item() > tgt:
                        lo = mid
                    else:
                        hi = mid
                out = (D > hi).float().sum(1)
                err = (((out - cnt_in) ** 2).mean() / max(1.0, tgt) ** 2).item()
                if best is None or err < best[0]:
                    best = (err, g, hi, D > hi)
                del D
            _, g_best, self.theta[l + 1], x = best
            self.g_l[l + 1] = g_best + self.c['pm_extra']        # remedy step 2 only (default 0)
            cnt_in = x.float().sum(1)
            counts.append(cnt_in.mean().item())
            del Draw
        return counts

    # ------------------------------------------------------------------ one tick
    def step(self, x_ext, force_int=None):
        """x_ext: bool [n1], the chain input S(t-1). Returns the internally lit copies (they send returns)."""
        a4 = self.L4.on                                                     # L4(t-1), as a mask
        d1 = self.loop.drive(a4) + self.L41.drive(a4)
        if self.G1P is not None:
            d1 = d1 + self.G1P.drive(a4)
        x_int = self.inhib(d1, 0) > self.theta[0]
        if force_int is not None:                                           # DIAGNOSTIC hook only
            x_int = x_int | force_int
        x = x_ext | x_int
        n2 = self.inhib(self.ff[0].drive(self.L1.on.float()), 1) > self.theta[1]
        n3 = self.inhib(self.ff[1].drive(self.L2.on.float()), 2) > self.theta[2]
        d4 = self.ff[2].drive(self.L3.on.float()) + self.lat.drive(a4) + self.L44.drive(a4)
        if self.c['pfc_count'] == 'k':      # [fix A] L4 (the only layer fed by learned links) lights |L3(t-1)| units
            n4 = topk_pos(self.inhib(d4, 3), self.L3.on.sum())
        else:                               # birth-calibrated fixed threshold
            n4 = self.inhib(d4, 3) > self.theta[3]
        self.L1ext.set(x_ext)
        self.L1.set(x)
        self.L2.set(n2)
        self.L3.set(n3)
        self.L4.set(n4)
        self.x_int = x_int
        self.d1 = d1                                                        # kept for diagnostic d3
        return x_int

    # ------------------------------------------------------------------ learning [R1, P7, P8]
    def l1_post(self):
        fw = self.c['fall_win']
        if self.c['post'] == 'chain':                                       # onset of the chain-driven input
            ev = self.L1ext.rose() & ~self.L1.prev
            return ev, ev, self.L1ext.age_prev <= fw
        ev = self.L1.rose()                                                 # arm POST-any
        return ev, ev, self.L1.age_prev <= fw

    def learn(self):
        fw = self.c['fall_win']
        el = self.L4.elig_prev(fw)
        ro = self.L4.rose()
        ev, rs, rc = self.l1_post()
        self.L41.learn(el, ro, ev, rs, rc)
        if self.G1P is not None:                                            # the same rule and post event as L41
            self.G1P.learn(el, ro, ev, rs, rc)
        r4 = self.L4.rose()
        self.L44.learn(el, ro, r4, r4, self.L4.age_prev <= fw)

    def pops(self):
        return [self.L1, self.L1ext, self.L2, self.L3, self.L4]
