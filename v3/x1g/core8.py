"""X1g core = X1f core6.Brain (copied unchanged) + options for the sparse / large PFC test (ARCH sec. 4 item 20).
All options are off by default, so the defaults reproduce core6 exactly (eqtest8.py).

lat_fly=True   the same ARCH two-layer L3 -> L3 propagation, computed in a form that scales to n3 = 6,400:
    * the drive is computed every tick from the CURRENT connection strengths,
          lat_j = g33 * c_j * sum_{i active} (B33_ij + L33_ij) * r_i,   r_i = 1 / out_i^a33,  c_j = 1 / in_j^(1-a33)
      (out / in = the row / column sums, with core5's 0.3 x median floors); core5 instead freezes P33 = g33 W /
      (out^a in^(1-a)) every norm_every ticks.  r, c and the frac threshold's own in-weight sum in3 are refreshed every
      norm_every ticks, as before.
    * the one global decay of the learned L33 is kept as one scale factor (L33_true = Ls * L33_stored): identical
      arithmetic, pruning below `prune` every prune_every ticks instead of every decay step.
    * the X3 patch mean is computed from patch labels (identical to PM @ PM.T @ x).
    No new rule: only the bookkeeping changes.

a31 / a33      exponent split of the two-layer weights (core5): propagation = g w / (out^a in^(1-a)); a = 1 is
               out-dominant (a unit linked to many answers spreads its drive thin; P31 row sums = g31).

rel1=True      release of the inner hearing by sharpness, relative not absolute (ARCH 2.7 "想法越集中、肌肉信号越锐利";
               the same two forms the motor area already uses):
    divisive gain   xn_j = x1_j / (eps1 + sum_k x1_k)     (x1 = top -> L1 drive, sum over all L1 lines, before the
                    patch mean)  -- the share of the prediction that goes to line j;
    max-relative    pred_j = (xn_j >= th1) and (x1_j >= rho1 * max_k x1_k)
    instead of the fixed absolute floor pred_j = x1_j >= floor_in (0.35), which scale-free weights cannot reach.
"""
import numpy as np
import os as _os, sys as _sys
_d = _os.path.dirname(_os.path.abspath(__file__))
if _d not in _sys.path:
    _sys.path.insert(0, _d)
for _x in ('x1f', 'x1d', 'x1c'):   # repo: core5/core6/common6/world6 from ../x1f, core4/phase1 from ../x1d, core3/world/run from ../x1c
    _p = __import__('os').path.join(__import__('os').path.dirname(__import__('os').path.abspath(__file__)), '..', _x)
    if _p not in __import__('sys').path:
        __import__('sys').path.append(_p)
from core6 import Brain as Brain6, DEF as DEF6

DEF = dict(DEF6)
DEF.update(lat_fly=False, prune_every=1000, rel1=False, eps1=1.0, th1=0.015, rho1=0.6, homeo2=False, fell_win31=None)


class Brain(Brain6):
    def __init__(self, phys, seed, rates=None, **kw):
        c = dict(DEF)
        c.update(kw)
        super().__init__(phys, seed, rates=rates, **c)
        self.c = c
        if c['lat_fly'] or c['rel1']:
            assert c['act'] == 'frac' and not c['hip'] and not c['dgain'] and c['pm'] > 0
        if c['lat_fly']:
            self.lab3 = self.PM3.argmax(1)
            self.sz3 = np.maximum(np.bincount(self.lab3, minlength=self.PM3.shape[1]), 1).astype(np.float32)
            self.lab1 = self.PM1.argmax(1)
            self.sz1 = np.maximum(np.bincount(self.lab1, minlength=self.PM1.shape[1]), 1).astype(np.float32)
            self.Ls = 1.0
            self.refresh_birth()
            self.P33 = None                                   # never materialised
            self._norm()

    # ------------------------------------------------------------------ bookkeeping for lat_fly
    def refresh_birth(self):
        """exact row / column sums of the birth part and of the stored learned part (call after any hand edit)"""
        self.B_out = self.B33.sum(1).astype(np.float64)
        self.B_in = self.B33.sum(0).astype(np.float64)
        self.L_out = self.L33.sum(1).astype(np.float64)
        self.L_in = self.L33.sum(0).astype(np.float64)
        r, c = np.nonzero(self.B33)
        self.B_nz = (r, c, self.B33[r, c].astype(np.float64))

    def L33_true(self):
        return self.L33 * self.Ls if self.c['lat_fly'] else self.L33

    def wipe33(self):
        self.L33[:] = 0
        if self.c['lat_fly']:
            self.Ls = 1.0
            self.refresh_birth()
        self._norm()

    def _pmean3(self, x):
        return (np.bincount(self.lab3, weights=x, minlength=len(self.sz3)) / self.sz3)[self.lab3].astype(np.float32)

    def _pmean1(self, x):
        return (np.bincount(self.lab1, weights=x, minlength=len(self.sz1)) / self.sz1)[self.lab1].astype(np.float32)

    def _norm(self):
        c = self.c
        if not c.get('lat_fly') or not hasattr(self, 'Ls'):
            return super()._norm()
        a = c['a31'] if c.get('a31') is not None else c['alpha']
        W = np.maximum(self._pred_mat() + self.B31, 0) * self.M31
        self.P31 = self._arch(W, c['g31'], a)
        self.in1 = self.P31.sum(0)
        self.in2 = self.W12.sum(0)
        a3 = c['a33']
        out = self.B_out + self.Ls * self.L_out
        inn = self.B_in + self.Ls * self.L_in
        rs_f = np.maximum(out, 0.3 * np.median(out[out > 0])) if (out > 0).any() else np.ones_like(out)
        cs_f = np.maximum(inn, 0.3 * np.median(inn[inn > 0])) if (inn > 0).any() else np.ones_like(inn)
        self.r33 = (1.0 / rs_f ** a3).astype(np.float32)
        self.c33 = (c['g33'] / cs_f ** (1 - a3)).astype(np.float32)
        r, cc, v = self.B_nz
        colw = np.bincount(cc, weights=v * self.r33[r], minlength=len(self.r33)) + self.Ls * (self.L33.T @ self.r33)
        self.in3 = self.W23.sum(0) + colw * self.c33

    def _decay(self):
        c = self.c
        if not c['lat_fly']:
            return super()._decay()
        A = self._pred_mat()
        A *= c['dec_31']
        A[np.abs(A) < c['prune']] = 0.0
        self.Ls *= c['dec_33']
        if self.t % c['prune_every'] == 0 or self.Ls < 1e-2:
            L = self.L33
            L *= np.float32(self.Ls)
            L[np.abs(L) < c['prune']] = 0.0
            self.Ls = 1.0
            self.L_out = L.sum(1).astype(np.float64)
            self.L_in = L.sum(0).astype(np.float64)

    def _rule33_onset(self, l3p, l3):
        c = self.c
        if not c['lat_fly']:
            return super()._rule33_onset(l3p, l3)
        L, B, s = self.L33, self.B33, np.float32(self.Ls)
        eta = c['eta33']
        etm = c['eta33'] * 0.5 if c['eta33_minus'] is None else c['eta33_minus']
        rise = np.nonzero(l3 & ~l3p)[0]
        if not len(rise):
            return
        earlier = np.nonzero(l3p | (self.fell3 <= c['fell_win']))[0]

        def add(ix, d, r_idx, c_idx):
            L[ix] += d
            self.L_out[r_idx] += d.sum(1)
            self.L_in[c_idx] += d.sum(0)
        if len(earlier):
            ix = np.ix_(earlier, rise)
            d = eta * self._s(L[ix] * s + B[ix]) / s
            add(ix, d, earlier, rise)
            self.n33_ev['grow'] += d.size
        if len(rise) > 1:
            ix = np.ix_(rise, rise)
            d = eta * self._s(L[ix] * s + B[ix]) / s
            np.fill_diagonal(d, 0.0)
            add(ix, d, rise, rise)
            self.n33_ev['co'] += d.size - len(rise)
        older = np.nonzero(l3p & l3)[0]
        if len(older) and etm > 0:
            ix = np.ix_(rise, older)
            d = etm * self._s(L[ix] * s + B[ix]) / s
            new = np.maximum(L[ix] - d, -B[ix] / s)
            add(ix, new - L[ix], rise, older)
            self.n33_ev['weak'] += d.size

    # ------------------------------------------------------------------ top -> L1 rule with its own fall window
    def _rule_onset(self, L, pre_prev, pre_now, pre_fell, post_prev, post_now, eta, etm, mask, track=None):
        """core4's onset rule (used only for top -> L1); fell_win31 (if set) replaces fell_win for this rule only.
        fell_win31 = -1: only top units that were on at the previous tick strengthen A -> B ('A on, then B'),
        not units that fell 1-2 ticks before B rose."""
        fw = self.c.get('fell_win31')
        if fw is None:
            return super()._rule_onset(L, pre_prev, pre_now, pre_fell, post_prev, post_now, eta, etm, mask, track)
        keep = self.c['fell_win']
        self.c['fell_win'] = fw
        try:
            return super()._rule_onset(L, pre_prev, pre_now, pre_fell, post_prev, post_now, eta, etm, mask, track)
        finally:
            self.c['fell_win'] = keep

    # ------------------------------------------------------------------ one tick: core6._step4 copied; options marked [X1g]
    def _step4(self, feat, r=0.0, learn=None):
        c = self.c
        if not c['lat_fly'] and not c['rel1']:
            return super()._step4(feat, r, learn)
        fly = c['lat_fly']
        learn = c['learn'] if learn is None else learn
        a0 = np.concatenate([feat, ~feat])
        ext = np.zeros(self.n1, bool)
        ext[:self.na1] = a0[self.send]
        ext[self.sl_mus] = self.mus
        ext[self.sl_blk] = self.stops
        heard = ext[:self.na1]
        l1p, l2p, l3p = self.l1, self.l2, self.l3
        a1, a2, a3 = np.nonzero(l1p)[0], np.nonzero(l2p)[0], np.nonzero(l3p)[0]
        # ---- L1 internal activation (prediction = inner hearing)
        x1 = self.P31[a3].sum(0) if len(a3) else np.zeros(self.n1, np.float32)
        if self.H31 is not None and len(a3):
            x1 = x1 + self.H31[a3].sum(0)
        tot1 = float(x1.sum())
        self.recall = np.zeros(self.n1, bool)
        if c['pm'] > 0 and len(a3):
            x1 = x1 - c['pm'] * (self._pmean1(x1) if fly else self.PM1 @ (self.PM1.T @ x1))
        if c['rel1']:                                          # [X1g] release by sharpness (divisive gain + max-relative)
            mx = float(x1.max()) if len(a3) else 0.0
            if mx > 0:
                xn = x1 / (c['eps1'] + tot1)
                pred = (xn >= c['th1']) & (x1 >= c['rho1'] * mx)
            else:
                pred = np.zeros(self.n1, bool)
        else:
            pred = x1 >= np.maximum(c['f1'] * self.in1, c['floor_in'])
        self._x1 = x1
        self._tot1 = tot1
        fl = getattr(self, '_force_lines', None)
        if fl is not None:
            pred = pred.copy()
            pred[fl] = True
            self._force_lines = None
        l1 = ext | pred
        # ---- L2 (frac threshold, as core6)
        x2 = self.W12[a1].sum(0) if len(a1) else np.zeros(c['n2'], np.float32)
        ai = np.nonzero(l1p & ~self.ext_prev)[0] if hasattr(self, 'ext_prev') else []
        if len(ai):
            x2 = x2 - (1.0 - c['g_int']) * self.W12[ai].sum(0)
        if c['homeo2']:                                       # [X1g] threshold also follows the unit's own long-run
            l2 = x2 - self.h2 >= c['f2'] * self.in2            #   activity (target k2 / n2), as homeo3 does in L3
            self._homeo(self.h2, l2, c['k2'], c['n2'], learn)
        else:
            l2 = x2 >= c['f2'] * self.in2
        # ---- L3 (top, thinking range)
        f3 = np.asarray(self.W23[a2].sum(0) if len(a2) else np.zeros(c['n3']), np.float32)
        if len(a3):
            if fly:                                            # [X1g] same propagation, current weights
                W = self.B33[a3] + np.float32(self.Ls) * self.L33[a3]
                lat = (self.r33[a3] @ W) * self.c33
            else:
                lat = self.P33[a3].sum(0)
            lat = np.asarray(lat, np.float32)
        else:
            lat = np.zeros(c['n3'], np.float32)
        if self.H33 is not None and len(a3):
            lat = lat + self.H33[a3].sum(0)
        x_wm_raw = f3 + lat
        H13 = getattr(self, 'H13', None)
        if H13 is not None and hasattr(self, 'ext_prev'):
            ae = np.nonzero(self.ext_prev)[0]
            if len(ae):
                x_wm_raw = x_wm_raw + H13[ae].sum(0)
        pm3 = self._pmean3 if fly else (lambda v: self.PM3 @ (self.PM3.T @ v))
        if c['pm'] > 0 and len(a3):
            lat = lat - c['pm'] * pm3(lat)
        if c['pm_ff'] > 0 and c['pm'] > 0:
            f3 = f3 - c['pm_ff'] * pm3(f3)
        x3 = f3 + lat
        self._x3 = x3
        kc = c['kappa_c']
        l3 = x3 + kc * l3p - (self.h3 if c['homeo3'] else 0.0) >= c['f3'] * (self.in3 + kc)
        if c['homeo3']:
            self._homeo(self.h3, l3, c['k3'], c['n3'], learn)
        wm = getattr(self, 'wm', None)
        if wm is not None:
            l3 = l3.copy()
            l3[wm] = x_wm_raw[wm] >= self.th_wm
        if learn:
            self._learn(l1p, l1, l3p, l3, ext)
        self.fell3 = np.where(l3p & ~l3, 0, np.minimum(self.fell3 + 1, 99)).astype(np.int16)
        self.l1, self.l2, self.l3, self.pred = l1, l2, l3, pred
        self.ext_prev = ext
        mus = self._motor(heard, pred, l3, r, learn)
        self.t += 1
        if self.t % 10 == 0:
            for k, v in (('n1', l1), ('n2', l2), ('n3', l3)):
                self.cnt[k].append(int(v.sum()))
        if learn:
            if self.t % c['dec_every'] == 0:
                self._decay()
            if self.t % c['norm_every'] == 0:
                self._norm()
        return mus
