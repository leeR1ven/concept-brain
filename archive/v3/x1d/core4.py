"""X1d core = X1c core3.Brain (unchanged code, copied) + two options the coordinator's 11-01 notes ask for:

rule='onset' (designer 11-01: held PFC neurons DO learn; "always-on grow nothing" is only for shallow sensory units)
    top->L1 prediction links L3(t-1) -> L1(t), by onset order:
      * an L1 unit B comes on at t: every L3 unit A that was on at t-1 (held or new) or fell within fell_win
        strengthens A->B  (A came on before B's onset, even if A is still held);
      * an L3 unit A comes on at t: every L1 unit B that came on earlier and is still on (on at t-1 and t)
        weakens A->B  (B's onset came before A's onset);
      * nothing happens at a fall (X1c's "held across the fall -> weaken" is removed; a held unit is not "B then A").
    concave growth, 1 % floor, two-layer weights and the one global decay are unchanged.
rule='x1c'  : X1c's form (for comparison).

babble=N  : the direct auditory->motor link is learned by babbling instead of being given at birth
    (ARCH sec. 2.7; char-neuron-voice speak_dev.babble_once): N babbles, each a random set of 3-5 muscles held
    3 ticks; the own voice is heard 1 tick later; co-active (heard content unit u, active muscle m) strengthen u->m,
    minus the chance level (speak_dev's mean subtraction), concave growth.  Only physics and hearing are used.
"""
import numpy as np
import os as _os, sys as _sys   # repo: core3.py, world.py and run.py from ../x1c
_d = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', 'x1c')
if _d not in _sys.path:
    _sys.path.insert(1, _d)
from core3 import Brain as Brain3, DEF as DEF3
from world import render, OWN

DEF = dict(DEF3)
DEF.update(rule='x1c', babble=0, eta_bab=0.05, bab_gain=1.0, nrm31='mass', g31=1.0, birth31='random', post='l1',
           dgain=False, dg_tau=3000.0, dg_lo=0.5, dg_hi=2.0, loc_same=False, pm=0.0, n_pm=12)


class Brain(Brain3):
    def __init__(self, phys, seed, rates=None, **kw):
        c = dict(DEF)
        c.update(kw)
        super().__init__(phys, seed, rates=rates, **c)
        self.c = c
        if c['birth31'] == 'recip' and c['g_loop'] > 0:
            # designer msg 84: "A activates G, G comes back and activates A" -- the birth loop top->L1 is wired
            # back to the L1 units that drive each top unit through L1->L2->L3 (same local mask, same gain, same fan)
            R = (self.W12 @ self.W23).T * self.M31                 # (n3, n1): how strongly L1 j drives L3 i
            B = np.zeros_like(self.B31)
            for j in range(self.n1):
                col = R[:, j]
                pick = np.argsort(-col)[:c['fan31']]
                pick = pick[col[pick] > 0]
                if len(pick):
                    B[pick, j] = col[pick] / col[pick].sum() * c['g_loop']
            self.B31 = B.astype(np.float32)
            self._norm()
        if c['babble']:
            self._babble(c['babble'])
        self.m2 = np.ones(c['n2'], np.float32)
        self.m3 = np.ones(c['n3'], np.float32)
        if c['pm'] > 0:                                        # patches of a few dozen neurons (by position)
            def patches(pos, n):
                ppos = np.random.default_rng(seed + 99).random((n, 2))
                lab = np.argmin(((pos[:, None, :] - ppos[None, :, :]) ** 2).sum(2), 1)
                M = np.zeros((len(pos), n), np.float32)
                M[np.arange(len(pos)), lab] = 1.0
                return M / np.sqrt(np.maximum(M.sum(0, keepdims=True), 1.0))   # PM @ PM.T @ x = patch mean
            self.PM1 = patches(self.pos1, max(2, self.n1 // 30))
            self.PM3 = patches(self.pos3, c['n_pm'] if c['n3'] <= 400 else c['n3'] // 33)

    # ------------------------------------------------------------------ two-layer weights for the top->L1 links
    def _norm(self):
        """nrm31='mass': X1c (propagation re-scaled to keep the total learned mass, so growth passes into drive);
        nrm31='arch': ARCH sec. 2.4 / X3 form, propagation = g31 * w / (out^a in^(1-a)) with no re-scaling, so the
        drive is scale-free and the learned top->L1 drive cannot grow without bound."""
        super()._norm()
        if self.c.get('nrm31', 'mass') == 'arch':
            a = self.c['alpha']
            W = np.maximum(self._pred_mat() + self.B31, 0) * self.M31
            rs, cs = W.sum(1), W.sum(0)
            if W.sum() > 0:
                rs_f = np.maximum(rs, 0.3 * np.median(rs[rs > 0]))
                cs_f = np.maximum(cs, 0.3 * np.median(cs[cs > 0]))
                self.P31 = (self.c['g31'] * W / (rs_f[:, None] ** a * cs_f[None, :] ** (1 - a))).astype(np.float32)
            self.in1 = self.P31.sum(0)

    # ------------------------------------------------------------------ babbling-learned auditory -> motor link
    def _babble(self, n):
        ph, c = self.ph, self.c
        rng = np.random.default_rng(int(self.rng.integers(1 << 30)))
        nm = ph.n_mus
        W = np.zeros((ph.n_c, nm), np.float32)                 # heard content feature -> muscle
        heard_prev = np.zeros(ph.n_a, bool)
        for _ in range(n):
            k = int(rng.integers(3, 6))
            mus = np.zeros(nm, bool)
            mus[rng.choice(nm, k, replace=False)] = True
            m = ph.decode(mus)
            for tick in range(4):                              # 3 ticks of the gesture + 1 tick rest
                act = mus if tick < 3 else np.zeros(nm, bool)
                snd = render(ph, [(m[0], m[1], OWN, 1.0)], rng) if (tick >= 1 and m is not None) else render(ph, [], rng)
                u = snd[:ph.n_c]
                a = act if tick >= 1 else np.zeros(nm, bool)   # heard at tick t comes from the muscles of t-1
                if u.any() and a.any():
                    ui = np.nonzero(u)[0]
                    ix = np.ix_(ui, np.nonzero(a)[0])
                    W[ix] += c['eta_bab'] * self._s(W[ix])
                    W[ui] -= c['eta_bab'] * a.sum() / nm       # chance level (mean subtraction)
        W = np.maximum(W, 0.0)
        cont = np.zeros((self.n1, ph.n_c), np.float32)
        for k in range(self.na1):
            if self.l1_content[k]:
                cont[k, self.l1_feat[k]] = 1.0
        M = cont @ W
        M *= self.Mmap.sum() / max(M.sum(), 1e-9) * c['bab_gain']   # same total strength as the stand-in (a gene)
        self.Mmap_given = self.Mmap
        self.Mmap = M.astype(np.float32)
        self.Wbab = W

    # ------------------------------------------------------------------ the one rule, onset-order form
    def _rule_onset(self, L, pre_prev, pre_now, pre_fell, post_prev, post_now, eta, etm, mask, track=None):
        rise = np.nonzero(post_now & ~post_prev)[0]
        if len(rise):
            earlier = np.nonzero(pre_prev | (pre_fell <= self.c['fell_win']))[0]
            if len(earlier):
                ix = np.ix_(earlier, rise)
                d = eta * self._s(L[ix]) * mask[ix]
                L[ix] += d
                if track is not None:
                    track(earlier, rise, d)
        arise = np.nonzero(pre_now & ~pre_prev)[0]
        older = np.nonzero(post_prev & post_now)[0]
        if len(arise) and len(older) and etm > 0:
            ix = np.ix_(arise, older)
            d = etm * self._s(L[ix]) * mask[ix]
            L[ix] = np.maximum(L[ix] - d, 0.0)

    def _learn(self, l1p, l1, l3p, l3, ext):
        if self.c['rule'] != 'onset' or self.R31 is not None:
            return super()._learn(l1p, l1, l3p, l3, ext)
        c = self.c
        sm = self.sl_mus

        def track(pre, post, d):
            m = (post >= sm.start) & (post < sm.stop)
            if m.any():
                self.E31[np.ix_(pre, post[m] - sm.start)] += d[:, m]
        self.E31 *= c['lam_r']
        self.Ectx *= c['lam_r']
        if c['post'] == 'ext':
            # designer: "最后一层在这一刻的活动，和下一刻第一层的输入做赫布连接" -- the post side is the next INPUT
            # (forward-chain activity: heard sound incl. own voice, actual muscles, stop neurons), not imagery
            ep = self.ext_prev if hasattr(self, 'ext_prev') else np.zeros_like(ext)
            self._rule_onset(self.L31, l3p, l3, self.fell3, ep, ext, c['eta'], c['eta_minus'], self.M31, track=track)
        else:
            self._rule_onset(self.L31, l3p, l3, self.fell3, l1p, l1, c['eta'], c['eta_minus'], self.M31, track=track)
        if c['eta33'] > 0:
            self._rule(self.L33, self.B33, l3p, l3, self.fell3, l3p, l3, c['eta33'], c['eta33'] * 0.5, lateral=True)
        if self.V is not None and c['act'] not in ('frac', 'own'):
            R = np.nonzero(l3 & ~l3p)[0]
            if len(R):
                off = np.nonzero(~l3)[0]
                ix = np.ix_(R, off)
                self.V[ix] += c['eta_v'] * self._s(self.V[ix])
                ix2 = np.ix_(np.nonzero(l3)[0], R)
                self.V[ix2] = np.maximum(self.V[ix2] - c['eta_v_co'], 0.0)

    # ------------------------------------------------------------------ hand-wired DIAGNOSTIC hook (probes only)
    _force_ok = True

    def step(self, feat, r=0.0, learn=None):
        """if self._force (L1 line indices) is set, those lines are made 'predicted' (inner hearing) on this tick by a
        temporary strong top->L1 link from the active top units.  Used only by the diagnostic probe, never in learning."""
        f = getattr(self, '_force', None)
        if f is None or not len(f):
            return self._step4(feat, r, learn)
        self._force = None
        a3 = np.nonzero(self.l3)[0]
        save, save_in = self.P31, self.in1
        P = self.P31.copy()
        if len(a3):
            P[np.ix_(a3, f)] += 100.0
        self.P31 = P
        try:
            return self._step4(feat, r, learn)
        finally:
            self.P31, self.in1 = save, save_in

    def _step4(self, feat, r=0.0, learn=None):
        c = self.c
        learn = c['learn'] if learn is None else learn
        a0 = np.concatenate([feat, ~feat])
        ext = np.zeros(self.n1, bool)                          # forward chains
        ext[:self.na1] = a0[self.send]
        ext[self.sl_mus] = self.mus
        ext[self.sl_blk] = self.stops                          # phasic stop neurons have chain pairs
        heard = ext[:self.na1]
        l1p, l2p, l3p = self.l1, self.l2, self.l3
        a1, a2, a3 = np.nonzero(l1p)[0], np.nonzero(l2p)[0], np.nonzero(l3p)[0]
        # ---- L1 internal activation: predicted by the top layer, recalled by the hippocampus
        x1 = self.P31[a3].sum(0) if len(a3) else np.zeros(self.n1, np.float32)
        self.recall = np.zeros(self.n1, bool)
        if c['hip']:                                           # recall re-lights S in its own area (return chains);
            ah = np.nonzero(self.h)[0]                          # S then reaches L1 by its forward chain [D]
            if len(ah):
                xr = c['g_h'] * self.Wh1[ah].sum(0) / (1.0 + 0.05 * len(ah))
                self.recall = xr >= np.maximum(c['f_h'] * self.Wh1.sum(0), c['floor_h'])
                self.recall[self.sl_blk] = False
                ext = ext | self.recall
        if c['pm'] > 0 and len(a3):
            x1 = x1 - c['pm'] * (self.PM1 @ (self.PM1.T @ x1))   # patch mean of the prediction drive, same tick
        if c['act'] not in ('frac', 'own'):
            x1 = x1 / (1.0 + c['gain'] * len(a3))
        if learn:
            self.mu1 += (x1 - self.mu1) / c['tau_mu']
        p1 = x1 - self.mu1 - self.h1
        if c['act'] in ('frac', 'own') or c['pred_rel']:
            pred = x1 >= np.maximum(c['f1'] * self.in1, c['floor_in'])
        else:
            if c['act'] == 'global':
                self.th_in = self._qthr(p1, c['k_in'], self.th_in, c['th_lr'], floor=c['floor_in'], bounded=False)
            else:
                self.th_in = c['floor_in']
            pred = p1 > self.th_in
            self._homeo(self.h1, pred, c['k_in'] * 0.2, self.n1, learn)
        l1 = ext | pred
        # ---- L2
        x2 = self.W12[a1].sum(0) if len(a1) else np.zeros(c['n2'], np.float32)
        ai = np.nonzero(l1p & ~self.ext_prev)[0] if hasattr(self, 'ext_prev') else []
        if len(ai):                                            # internal (returned) activity only modulates:
            x2 = x2 - (1.0 - c['g_int']) * self.W12[ai].sum(0)  # <= half the forward strength [D sec. 5]
        if c['dgain']:                                         # slow bounded divisive gain (own long-run mean drive)
            if learn:
                self.m2 += (x2 - self.m2) / c['dg_tau']
            x2 = x2 / np.clip(self.m2 / max(self.m2.mean(), 1e-6), c['dg_lo'], c['dg_hi'])
        if c['act'] == 'own':
            self.mu2 += (x2 - self.mu2) / c['tau_own'] if learn else 0
            l2 = x2 > np.maximum(c['c2m'] * self.mu2, 0.3)
        elif c['act'] == 'frac':
            l2 = x2 >= c['f2'] * self.in2
        elif c['act'] == 'global':
            x2 = x2 / (1.0 + c['gain'] * len(a1))
            p2 = x2 - self.h2
            self.th2 = self._qthr(p2, c['k2'], self.th2, c['th_lr'], floor=c['floor2'], bounded=False)
            l2 = p2 > self.th2
        else:
            x2 = x2 / (1.0 + c['gain'] * len(a1))
            p2 = x2 - self.h2
            if c['loc_same']:                                  # same-tick feedback inhibition (provisional pass)
                l0 = p2 > c['base2']
                self.inh2 = self.I2b @ (self.I2.T @ l0.astype(np.float32)) * 4.0
            l2 = p2 - c['g_loc'] * self.inh2 > c['base2']
            self.inh2 = self.I2b @ (self.I2.T @ l2.astype(np.float32)) * 4.0
        if c['act'] not in ('frac', 'own'):
            self._homeo(self.h2, l2, c['k2'], c['n2'], learn)
        # ---- L3 (top, thinking range)
        f3 = np.asarray(self.W23[a2].sum(0) if len(a2) else np.zeros(c['n3']), np.float32)
        if c['dgain']:
            if learn:
                self.m3 += (f3 - self.m3) / c['dg_tau']
            f3 = f3 / np.clip(self.m3 / max(self.m3.mean(), 1e-6), c['dg_lo'], c['dg_hi'])
        lat = np.asarray(self.P33[a3].sum(0) if len(a3) else np.zeros(c['n3']), np.float32)
        if c['pm'] > 0 and len(a3):
            lat = lat - c['pm'] * (self.PM3 @ (self.PM3.T @ lat))
        x3 = f3 + lat
        if c['act'] == 'own':
            self.mu3 += (x3 - self.mu3) / c['tau_own'] if learn else 0
            l3 = x3 + c['kappa_c'] * l3p > np.maximum(c['c3m'] * self.mu3, 0.3)
        elif c['act'] == 'frac':
            kc = c['kappa_c']
            l3 = x3 + kc * l3p >= c['f3'] * (self.in3 + kc)
        else:
            x3 = x3 / (1.0 + c['gain'] * (len(a2) + len(a3)))
        p3 = x3 - self.h3 + c['kappa'] * l3p
        if c['act'] not in ('frac', 'own') and self.V is not None and len(a3):
            p3 = p3 - c['g_inh'] * self.V[a3].sum(0) * (~l3p)    # partner I_j acts only while j is off
        if c['act'] == 'global':
            self.th3 = self._qthr(p3, c['k3'], self.th3, c['th_lr'], floor=c['floor3'])
            l3 = p3 > self.th3
        elif c['act'] == 'local':
            if c['loc_same']:
                l0 = p3 > c['base3']
                self.inh3 = self.I3b @ (self.I3.T @ l0.astype(np.float32)) * 4.0
            l3 = p3 - c['g_loc'] * self.inh3 > c['base3']
            self.inh3 = self.I3b @ (self.I3.T @ l3.astype(np.float32)) * 4.0
        if c['act'] not in ('frac', 'own'):
            self._homeo(self.h3, l3, c['k3'], c['n3'], learn)
        oc = getattr(self, '_oracle', None)                    # hand-wired DIAGNOSTIC: oracle reproducible path
        if oc is not None:
            l3 = oc.copy()
            self._oracle = None
        if learn:
            self._learn(l1p, l1, l3p, l3, ext)
        self.fell3 = np.where(l3p & ~l3, 0, np.minimum(self.fell3 + 1, 99)).astype(np.int16)
        self.l1, self.l2, self.l3, self.pred = l1, l2, l3, pred
        self.ext_prev = ext
        mus = self._motor(heard, pred, l3, r, learn)
        if c['hip']:
            self._hip(l1, l3, learn)
        self.t += 1
        if self.t % 10 == 0:
            for k, v in (('n1', l1), ('n2', l2), ('n3', l3)):
                self.cnt[k].append(int(v.sum()))
        if learn:
            if self.t % c['dec_every'] == 0:
                self._decay()
            if self.t % c['norm_every'] == 0:
                self._norm()
            if self.F31 is not None and self.t % 10 == 0:
                self.F31 *= c['dec_f'] ** 10
                self.F33 *= c['dec_f'] ** 10
        return mus

