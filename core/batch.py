"""Many children, one set of connections (a development speed-up, ARCH 五 "开发时可以用等价的程序加速，要登记").
B copies of the neuron states run side by side on one brain's connections; each copy hears its own dialogue; every
tick the learned changes of all copies are added into the one set of connections. Per heard tick each change is the
same as in a single life; decay and each unit's usual level keep their rates per heard tick (every decay_every / B
ticks; usual level moved B / tau_usual of the way per tick). The brain's own single
state is left alone, so saving, probes and talking work on it as before. Speech wiring (G1P, pools) is not batched:
the child only overhears here."""
import math

import numpy as np
import torch

from .net import DEV


class BPop:
    def __init__(self, B, n):
        self.on = torch.zeros((B, n), dtype=torch.bool, device=DEV)
        self.prev = self.on.clone()
        self.age = torch.full((B, n), 99, dtype=torch.int16, device=DEV)
        self.age_prev = self.age.clone()

    def set(self, new):
        self.prev, self.age_prev = self.on, self.age
        self.on = new
        self.age = torch.where(new, torch.zeros_like(self.age), (self.age + 1).clamp_max(99))

    def rose(self):
        return self.on & ~self.prev

    def held(self, win):
        return self.age_prev <= win


def fdrive(F, X):
    """fixed fan-in links, batched: X bool [B, n_pre] -> [B, n_post]"""
    return (X.float()[:, F.idx] * F.w).sum(-1)


def ldrive(L, act):
    """by-pre links, batched: act bool [B, n_pre] -> [B, n_post]"""
    B, n = act.shape[0], L.n_post + 1
    out = torch.zeros(B * n, device=DEV)
    bi, ai = torch.nonzero(act, as_tuple=True)
    if ai.numel():
        flat = (bi[:, None] * n + L.idx[ai]).reshape(-1)
        out.index_add_(0, flat, (L.gain * L.w[ai]).reshape(-1))
    return out.view(B, n)[:, :-1]


def topk_rows(d, k):
    """per row, the k[b] most driven with positive drive"""
    out = torch.zeros_like(d, dtype=torch.bool)
    kmax = int(k.max())
    if kmax > 0:
        v, i = torch.topk(d, min(kmax, d.shape[1]), dim=1)
        keep = (torch.arange(v.shape[1], device=DEV)[None, :] < k[:, None]) & (v > 0)
        out.scatter_(1, i, keep)
    return out


def llearn(L, pre, post_a, post_b, eta, rule):
    """the rules of net.Links.learn, batched; the changes of all copies are summed into the one table.
    rule 'main': grow when post rises while pre is on/held, weaken when pre rises while post is on and did not rise;
    'plain' (hippocampus): grow by the full step when post rises; 'coactive' (concept area): grow while both are on."""
    if not L.learned:
        return
    B, n = pre.on.shape[0], L.n_post + 1
    pad = torch.zeros((B, 1), dtype=torch.bool, device=DEV)
    pa = torch.cat([post_a, pad], 1).reshape(-1)
    bi, ai = torch.nonzero(pre.held(L.c['fall_win']) | pre.rose() if rule != 'coactive'
                           else pre.on | pre.held(L.c['fall_win']), as_tuple=True)
    if ai.numel() == 0:
        return
    flat = bi[:, None] * n + L.idx[ai]
    grow = pa[flat]
    if rule == 'plain':
        dw = eta * grow.float()
    elif rule == 'coactive':
        dw = eta * L.plast(L.w[ai]) * grow.float()
    else:
        pb = torch.cat([post_b, pad], 1).reshape(-1)
        weaken = pre.rose()[bi, ai][:, None] & pb[flat] & ~grow
        dw = eta * L.plast(L.w[ai]) * (grow.float() - weaken.float())
    L.w.index_add_(0, ai, dw)
    rows = torch.unique(ai)
    L.w[rows] = L.w[rows].clamp_min(0.0)


class Batch:
    def __init__(self, life, worlds):
        self.life, self.worlds = life, worlds
        b, B = life.brain, len(worlds)
        self.B = B
        self.sound = [torch.zeros((B, n), dtype=torch.bool, device=DEV) for n in b.sound.n]
        self.touch = [s[None].expand(B, -1).clone() for s in b.touch.state]
        self.rs = [torch.zeros((B, n), device=DEV) for n in b.sound.n]
        self.rt = [torch.zeros((B, n), device=DEV) for n in b.touch.n]
        m = b.motor
        self.force, self.stop = BPop(B, m.nf), BPop(B, m.ns)
        self.rf = torch.zeros((B, m.nf), device=DEV)
        self.rp = torch.zeros((B, m.ns), dtype=torch.bool, device=DEV)
        n1 = b.n1
        self.L1, self.L1ext, self.L2, self.L3, self.L4 = (BPop(B, n1) for _ in range(5))
        self.H1, self.H1ext, self.Hs = BPop(B, n1), BPop(B, n1), BPop(B, b.hip.nh)
        cc = getattr(b, 'concept', None)
        if cc is not None:
            self.cpop, self.csrc = BPop(B, cc.n), BPop(B, cc.n_src)
            self.rc = torch.zeros((B, cc.n), device=DEV)
        self.muscles = torch.zeros((B, 24), dtype=torch.bool, device=DEV)
        self.t = 0
        self.decay_every = max(1, round(b.c['decay_every'] / B))
        self.adapt = min(1.0, B / b.c['tau_usual'])          # each unit's usual level also moves per heard tick

    def n_ex(self):
        return sum(w.n_ex for w in self.worlds)

    def tick(self, learning=True):
        b, c, B = self.life.brain, self.life.c, self.B
        mus = self.muscles.cpu().numpy()
        a0 = torch.as_tensor(np.stack([w.step(mus[k]) for k, w in enumerate(self.worlds)]), device=DEV)
        ch, m, p, h = b.chains, b.motor, b.pfc, b.hip
        cc = getattr(b, 'concept', None)
        parts = [self.sound[l][:, ix] for l, ix in enumerate(ch.sound_idx)] + \
                [self.touch[l][:, ix] for l, ix in enumerate(ch.touch_idx)] + [self.force.on, self.stop.on]
        if cc is not None:
            parts.append(self.cpop.on)
        x = torch.cat(parts, 1)                                                   # chain sources at t-1
        if cc is not None:                                                        # the shared concept area
            src = x[:, :cc.n_src]
            e = fdrive(cc.birth, src) + ldrive(cc.learned, src) - cc.usual
            if learning:
                cc.usual += e.mean(0) * self.adapt
            e = e + c['g_ret'] * cc.win * self.rc
            on = topk_rows(e, torch.full((B,), cc.k, device=DEV))
            if learning and on.any():
                cc.win += (e[on].mean() - cc.win) * self.adapt
            self.csrc.set(src)
            self.cpop.set(on)
        # sound area: layer 1 reads this tick's input, layer l the layer below one tick earlier; returns from t-1
        prev = [a0] + self.sound[:-1]
        self.sound = [fdrive(ly, pv) + r * rw > th for ly, pv, r, rw, th in
                      zip(b.sound.layers, prev, self.rs, b.sound.ret_w, b.sound.thr)]
        # motor: the brake from the input layer, returns into force neurons, blocking
        stop = ((a0[:, m.brake_lines].float().sum(1, keepdim=True) / m.n_brake) >= m.thr_s[None]) | self.rp
        lit = self.rf > 0
        d = lit.float() * m.ret_drive
        P = d.sum(1, keepdim=True)
        nblock = m.n_tonic + (P >= m.ladder[None]).sum(1, keepdim=True)
        inh = c['b_block'] * nblock / (8.0 * m.m) + c['c_stop'] * stop.float().sum(1, keepdim=True) / (8.0 * m.m)
        force = d - inh > m.thr_f
        self.force.set(force)
        self.stop.set(stop)
        cnt = torch.zeros((B, 24), device=DEV).index_add_(1, m.muscle_of, force.float())
        self.muscles = cnt >= math.ceil(m.m / 2)
        f = (cnt >= b.skin.half).float() * cnt / b.skin.m                        # the body's force, felt now
        up = f[:, :, None] >= b.skin.th[None, None, :]
        tin = torch.stack([up, ~up], 3).reshape(B, -1)
        prev = [tin] + self.touch[:-1]
        self.touch = [fdrive(ly, pv) + r * rw > th for ly, pv, r, rw, th in
                      zip(b.touch.layers, prev, self.rt, b.touch.ret_w, b.touch.thr)]
        # PFC
        a4 = self.L4.on
        d1 = ldrive(p.loop, a4) + ldrive(p.L41, a4)
        e1 = d1 - p.usual
        if learning:
            p.usual += (d1 - p.usual).mean(0) * self.adapt
        n_ext = x.sum(1).float()
        if getattr(p, 'ext_usual', None) is None:
            p.ext_usual = 0.026 * x.shape[1]
        if learning:
            p.ext_usual += float(n_ext.mean() - p.ext_usual) * self.adapt
        budget = torch.full_like(n_ext, c['context'] * p.ext_usual)       # the usual external count, as in brain.py
        x_int = (e1 > c['theta1']) & topk_rows(e1, budget)
        n2 = fdrive(p.ff[0], self.L1.on) > p.thr2
        n3 = fdrive(p.ff[1], self.L2.on) > p.thr3
        if getattr(p, 'usual4', None) is None:
            p.usual4 = torch.zeros(p.ff[2].idx.shape[0], device=DEV)
        d4 = fdrive(p.ff[2], self.L3.on) + ldrive(p.L44, a4) - p.usual4
        if learning:
            p.usual4 += d4.mean(0) * self.adapt
        n4 = topk_rows(d4, self.L1.on.sum(1))
        self.L1ext.set(x)
        self.L1.set(x | x_int)
        self.L2.set(n2)
        self.L3.set(n3)
        self.L4.set(n4)
        # hippocampus
        ah = self.Hs.on
        eh = ldrive(h.H1t, ah) - h.usual
        if learning:
            h.usual += eh.mean(0) * self.adapt
        h_int = topk_rows(eh, budget)
        if getattr(h, 'usual_s', None) is None:
            h.usual_s = torch.zeros(h.nh, device=DEV)
        ds = fdrive(h.rec, self.H1.on) + ldrive(h.HH, ah) - h.usual_s
        if learning:
            h.usual_s += ds.mean(0) * self.adapt
        hs = topk_rows(ds, torch.full((B,), c['kh'], device=DEV))
        self.H1ext.set(x)
        self.H1.set(x | h_int)
        self.Hs.set(hs)
        # returns, 1:1 to their sources
        R = x_int | h_int
        for kind, l, a, z in ch.segs:
            if kind == 'sound':
                self.rs[l] = torch.zeros_like(self.rs[l])
                self.rs[l][:, ch.sound_idx[l]] = R[:, a:z].float()
            elif kind == 'touch':
                self.rt[l] = torch.zeros_like(self.rt[l])
                self.rt[l][:, ch.touch_idx[l]] = R[:, a:z].float()
            elif kind == 'force':
                self.rf = R[:, a:z].float()
            elif kind == 'stop':
                self.rp = R[:, a:z]
            else:
                self.rc = R[:, a:z].float()
        if learning:
            fw, eta = c['fall_win'], c['eta']
            onset = self.L1ext.rose() & ~self.L1.prev
            llearn(p.L41, self.L4, onset, self.L1ext.held(fw), eta, 'main')
            llearn(p.L44, self.L4, self.L4.rose(), self.L4.held(fw), eta, 'main')
            llearn(h.HH, self.Hs, self.Hs.rose(), self.Hs.held(fw), c['eta_h'], 'main')
            llearn(h.H1t, self.Hs, self.H1ext.rose() & ~self.H1.prev, self.H1ext.held(fw), c['eta_h'], 'plain')
            if cc is not None:
                llearn(cc.learned, self.csrc, self.cpop.on, None, eta, 'coactive')
            self.t += 1
            if self.t % self.decay_every == 0:
                for tb in b.plastic():
                    tb.decay()

    def counts(self):
        v = [s.sum(1) for s in self.sound] + [self.L1ext.on.sum(1), (self.L1.on & ~self.L1ext.on).sum(1),
                                               self.L1.on.sum(1), self.L2.on.sum(1), self.L3.on.sum(1),
                                               self.L4.on.sum(1), (self.H1.on & ~self.H1ext.on).sum(1),
                                               self.Hs.on.sum(1), self.force.on.sum(1), self.stop.on.sum(1)]
        if getattr(self.life.brain, 'concept', None) is not None:
            v.append(self.cpop.on.sum(1))
        return torch.stack(v).float().mean(1)
