"""X1e core = X1d core4.Brain (copied, unchanged) + the top layer's internal links (L3 -> L3) learning by the one rule
(ARCH sec. 4 item 11, corrected 11-01: "有关的互相激活", like the designer's char network).

New options (all off by default, so defaults reproduce core4 exactly):

rule33='onset'  the L3 -> L3 links learn with the same onset / held form already used for top -> L1:
    * B rises at t:  every A on at t-1 (just risen OR held) or fallen within fell_win  ->  A->B += eta33 * s(w)
      ("A then B strengthens A->B"; held PFC units do learn, designer 11-01);
    * A and B rise in the same tick  ->  A->B and B->A += eta33 * s(w)  ("co-rising strengthen each other");
    * A rises at t while B was already on (on at t-1 and still on at t)  ->  A->B -= eta33_minus * s(w)
      ("B first, A after weakens A->B");
    * nothing happens at a fall (ARCH sec. 2.8: "B 熄灭不减弱任何连接").
    s(w) = max(1 / (1 + w / w_ref)^pw, plast_floor): concave growth, no ceiling, ~1 % plasticity kept.
    The learned part may cancel the birth part but the total stays excitatory (W33 = B33 + L33 >= 0).
    Decay: the one global multiplicative decay (dec_33 every dec_every ticks, prune below `prune`) acts on L33 only;
    birth links B33 are exempt (ARCH sec. 2.8).
rule33='x1c'    X1c's lateral form (includes the fall clause) -- for comparison only.

nrm33='arch'    two-layer weights for L3 -> L3 in ARCH form: propagation P33 = g33 * W / (out^a33 * in^(1-a33)),
    no re-scaling of the total (so learned growth changes WHICH units are driven, not HOW MUCH in total).
    nrm33='mass' is X1c's form (total learned mass passes into drive).
a31 / a33       exponent split of the two-layer weights (0.5 = sqrt(out * in), the ARCH / X3 form).

No new module, rule, timer, refractoriness, fatigue or self-hold.
"""
import numpy as np
import os as _os, sys as _sys
_d = _os.path.dirname(_os.path.abspath(__file__))
if _d not in _sys.path:
    _sys.path.insert(0, _d)
for _x in ('x1d', 'x1c'):   # repo: core4/phase1/run2/regimes from ../x1d, core3/world/run from ../x1c
    _p = __import__('os').path.join(__import__('os').path.dirname(__import__('os').path.abspath(__file__)), '..', _x)
    if _p not in __import__('sys').path:
        __import__('sys').path.append(_p)
from core4 import Brain as Brain4, DEF as DEF4

DEF = dict(DEF4)
DEF.update(rule33='onset', nrm33='mass', g33=2.0, a33=0.5, a31=None, eta33_minus=None, pm_ff=0.0, homeo3=False)


class Brain(Brain4):
    def __init__(self, phys, seed, rates=None, **kw):
        c = dict(DEF)
        c.update(kw)
        super().__init__(phys, seed, rates=rates, **c)
        self.c = c
        self.n33_ev = dict(grow=0, co=0, weak=0)
        self._norm()

    # ------------------------------------------------------------------ two-layer weights
    @staticmethod
    def _arch(W, g, a):
        rs, cs = W.sum(1), W.sum(0)
        if W.sum() <= 0:
            return np.zeros_like(W)
        rs_f = np.maximum(rs, 0.3 * np.median(rs[rs > 0]))
        cs_f = np.maximum(cs, 0.3 * np.median(cs[cs > 0]))
        return (g * W / (rs_f[:, None] ** a * cs_f[None, :] ** (1 - a))).astype(np.float32)

    def _norm(self):
        super()._norm()                                    # core4: P31 (mass or arch), P33 (mass), in1/in2/in3
        c = self.c
        if 'rule33' not in c:                              # called from the parent constructor
            return
        if c.get('nrm31') == 'arch' and c.get('a31') is not None:
            W = np.maximum(self._pred_mat() + self.B31, 0) * self.M31
            self.P31 = self._arch(W, c['g31'], c['a31'])
            self.in1 = self.P31.sum(0)
        if c['nrm33'] == 'arch':
            W33 = np.maximum(self.B33 + self.L33, 0)
            self.P33 = self._arch(W33, c['g33'], c['a33'])
            self.in3 = self.W23.sum(0) + self.P33.sum(0)

    # ------------------------------------------------------------------ the one rule on L3 -> L3, onset / held form
    def _rule33_onset(self, l3p, l3):
        c = self.c
        L, B = self.L33, self.B33
        eta = c['eta33']
        etm = c['eta33'] * 0.5 if c['eta33_minus'] is None else c['eta33_minus']
        rise = np.nonzero(l3 & ~l3p)[0]
        if not len(rise):
            return
        earlier = np.nonzero(l3p | (self.fell3 <= c['fell_win']))[0]     # on at t-1 (new or held) or just fell
        if len(earlier):                                                  # A then B: A->B grows
            ix = np.ix_(earlier, rise)
            d = eta * self._s(L[ix] + B[ix])
            L[ix] += d
            self.n33_ev['grow'] += d.size
        if len(rise) > 1:                                                 # co-rising: both directions grow
            ix = np.ix_(rise, rise)
            d = eta * self._s(L[ix] + B[ix])
            np.fill_diagonal(d, 0.0)
            L[ix] += d
            self.n33_ev['co'] += d.size - len(rise)
        older = np.nonzero(l3p & l3)[0]                                   # B on before A's onset and still on
        if len(older) and etm > 0:                                        # A (rising now) -> B weakens
            ix = np.ix_(rise, older)
            d = etm * self._s(L[ix] + B[ix])
            L[ix] = np.maximum(L[ix] - d, -B[ix])
            self.n33_ev['weak'] += d.size

    def _learn(self, l1p, l1, l3p, l3, ext):
        c = self.c
        if c['eta33'] <= 0 or c['rule33'] != 'onset':
            return super()._learn(l1p, l1, l3p, l3, ext)
        e33 = c['eta33']
        c['eta33'] = 0.0                                   # parent: top -> L1 (+ partners) only
        try:
            super()._learn(l1p, l1, l3p, l3, ext)
        finally:
            c['eta33'] = e33
        self._rule33_onset(l3p, l3)

    # ------------------------------------------------------------------ one tick: core4._step4 copied, two options added
    #   pm_ff  : the patch-mean subtraction (X3 gain control, ARCH 4.10) also acts on the feed-forward L3 drive
    #   homeo3 : 'frac' thresholds also carry the slow homeostatic offset h3 (target k3 / n3, tau homeo_tau)
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
        if c['pm_ff'] > 0 and c['pm'] > 0:                   # the patch's inhibitory unit sees the whole drive
            f3 = f3 - c['pm_ff'] * (self.PM3 @ (self.PM3.T @ f3))
        x3 = f3 + lat
        if c['act'] == 'own':
            self.mu3 += (x3 - self.mu3) / c['tau_own'] if learn else 0
            l3 = x3 + c['kappa_c'] * l3p > np.maximum(c['c3m'] * self.mu3, 0.3)
        elif c['act'] == 'frac':
            kc = c['kappa_c']
            l3 = x3 + kc * l3p - (self.h3 if c['homeo3'] else 0.0) >= c['f3'] * (self.in3 + kc)
            if c['homeo3']:
                self._homeo(self.h3, l3, c['k3'], c['n3'], learn)
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

    # ------------------------------------------------------------------ diagnostics (read only)
    def hub_stats(self, ph):
        """in-strength of every syllable's L1 content code under the learned top->L1 links (connection strength L31)
        and under the propagation weights P31; and the in-strength spread of L3 -> L3."""
        W = np.maximum(self._pred_mat(), 0) * self.M31
        cs_w = W.sum(0)
        cs_p = self.P31.sum(0)
        rows = []
        for s in range(ph.content.shape[0]):
            lines = np.nonzero(self.l1_content & np.isin(self.l1_feat, np.nonzero(ph.content[s])[0]))[0]
            rows.append((float(cs_w[lines].mean()) if len(lines) else 0.0, float(cs_p[lines].mean()) if len(lines) else 0.0))
        rows = np.array(rows)
        W33 = np.maximum(self.B33 + self.L33, 0)
        in33 = W33.sum(0)
        out33 = W33.sum(1)
        p_in33 = self.P33.sum(0)
        q = lambda v: [round(float(x), 4) for x in np.percentile(v, [50, 90, 99, 100])]
        return dict(syl_in_w=np.round(rows[:, 0], 4).tolist(), syl_in_p=np.round(rows[:, 1], 4).tolist(),
                    L31_in_w_pct=q(cs_w[self.l1_content]), L31_in_p_pct=q(cs_p[self.l1_content]),
                    L33_in_w_pct=q(in33), L33_out_w_pct=q(out33), P33_in_pct=q(p_in33),
                    L33_learned_max=round(float(self.L33.max()), 4), L33_learned_mean=round(float(self.L33.mean()), 5),
                    L33_neg_frac=round(float((self.L33 < 0).mean()), 4), L33_pos_frac=round(float((self.L33 > 1e-3).mean()), 4),
                    n33_ev=dict(self.n33_ev))
