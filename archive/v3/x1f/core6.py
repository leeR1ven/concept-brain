"""X1f core = X1e core5.Brain (copied unchanged) + generation-0 hand-added connections (all off by default, so the
defaults reproduce core5 exactly).

Hand-added connection sets (see REGISTRY.md; built by wire.py, never changed by learning or by the global decay,
which spares birth links, ARCH sec. 2.8):
  H33 (n3 x n3)  hand-added L3 -> L3 links, given directly as drive strengths and added to the lateral drive before the
                 patch-mean (X3 gain control).  Not counted in the frac threshold's own in-weight sum.
  H31 (n3 x n1)  hand-added top -> L1 links (inner hearing of answer syllables), given directly as drive strengths and
                 added to the prediction drive before the patch-mean.
Speech basics (a) use the existing sound -> muscle map Mmap (X1c stand-in, ARCH sec. 2.7 / 2.11 step 1) and the existing
motor knobs only (c_mb, gamma_m, theta_m, g_ret_mot, rho).

A2 (speech basics, hand-added blocking links, gate_lo / gate_hi): two groups of blocking neurons see the muscle neurons
that would be released this tick (same-tick feed-forward): one blocks when fewer than gate_lo are about to fire, the
other when more than gate_hi are.  So only a syllable-shaped command (3-5 muscles) wins against the blocking neurons;
a mixed inner hearing ("unsure") gives silence, not garble (ARCH 2.7: release-by-sharpness; "没把握时结果是不说").

Diagnostic only (never used in learning runs): `_force_lines` = L1 lines switched on as predicted (inner hearing) on
this tick, directly (core4's force hook does nothing when L3 is empty, so it is replaced here).
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
from core5 import Brain as Brain5, DEF as DEF5

DEF = dict(DEF5)
DEF.update(hw=False, gate_lo=0, gate_hi=99)


class Brain(Brain5):
    def __init__(self, phys, seed, rates=None, **kw):
        c = dict(DEF)
        c.update(kw)
        super().__init__(phys, seed, rates=rates, **c)
        self.c = c
        self.H33 = None
        self.H31 = None

    def snapshot(self):
        d = super().snapshot()
        if hasattr(self, 'mu1'):
            d['mu1'] = self.mu1.copy()
        return d

    def step(self, feat, r=0.0, learn=None):
        return self._step4(feat, r, learn)

    # ------------------------------------------------------------------ one tick: core5._step4 copied; three lines added
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
        if self.H31 is not None and len(a3):                   # [X1f] hand-added top -> L1 links
            x1 = x1 + self.H31[a3].sum(0)
        self.recall = np.zeros(self.n1, bool)
        if c['hip']:
            ah = np.nonzero(self.h)[0]
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
        fl = getattr(self, '_force_lines', None)               # [X1f] DIAGNOSTIC forced inner hearing
        if fl is not None:
            pred = pred.copy()
            pred[fl] = True
            self._force_lines = None
        l1 = ext | pred
        # ---- L2
        x2 = self.W12[a1].sum(0) if len(a1) else np.zeros(c['n2'], np.float32)
        ai = np.nonzero(l1p & ~self.ext_prev)[0] if hasattr(self, 'ext_prev') else []
        if len(ai):
            x2 = x2 - (1.0 - c['g_int']) * self.W12[ai].sum(0)
        if c['dgain']:
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
        if self.H33 is not None and len(a3):                   # [X1f] hand-added L3 -> L3 links
            lat = lat + self.H33[a3].sum(0)
        x_wm_raw = f3 + lat                                    # [X1f] raw drive (the WM part has no patch-mean)
        H13 = getattr(self, 'H13', None)                       # [X1f, diagnostic] hand-added heard L1 copies -> WM
        if H13 is not None and hasattr(self, 'ext_prev'):
            ae = np.nonzero(self.ext_prev)[0]
            if len(ae):
                x_wm_raw = x_wm_raw + H13[ae].sum(0)
        if c['pm'] > 0 and len(a3):
            lat = lat - c['pm'] * (self.PM3 @ (self.PM3.T @ lat))
        if c['pm_ff'] > 0 and c['pm'] > 0:
            f3 = f3 - c['pm_ff'] * (self.PM3 @ (self.PM3.T @ f3))
        x3 = f3 + lat
        self._x3 = x3
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
            p3 = p3 - c['g_inh'] * self.V[a3].sum(0) * (~l3p)
        if c['act'] == 'global':
            self.th3 = self._qthr(p3, c['k3'], self.th3, c['th_lr'], floor=c['floor3'])
            l3 = p3 > self.th3
        elif c['act'] == 'local':
            l3 = p3 - c['g_loc'] * self.inh3 > c['base3']
            self.inh3 = self.I3b @ (self.I3.T @ l3.astype(np.float32)) * 4.0
        if c['act'] not in ('frac', 'own'):
            self._homeo(self.h3, l3, c['k3'], c['n3'], learn)
        wm = getattr(self, 'wm', None)                         # [X1f, diagnostic arm] hand-wired working-memory part:
        if wm is not None:                                     # driven laterally only, own fixed threshold
            l3 = l3.copy()
            l3[wm] = x_wm_raw[wm] >= self.th_wm
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
        return mus

    # ------------------------------------------------------------------ motor area: core3._motor copied + A2 gate
    def _motor(self, heard, pred, l3, r, learn):
        c = self.c
        rc = self.recall if hasattr(self, 'recall') else np.zeros(self.n1, bool)
        cont = self.l1_content
        d_heard = c['g_dir'] * ((heard & cont[:self.na1]).astype(np.float32) @ self.Mmap[:self.na1])
        d_inner = c['g_ret_aud'] * ((pred & ~rc & cont).astype(np.float32) @ self.Mmap)
        d_rec = c['g_ret_aud'] * ((rc & cont).astype(np.float32) @ self.Mmap)
        d_mot = c['g_ret_mot'] * (pred | rc)[self.sl_mus].astype(np.float32)
        intent_s = (pred | rc)[self.sl_blk].astype(np.float32)
        d = d_heard + d_inner + d_rec + d_mot
        dn = d / (1.0 + c['gamma_m'] * d.sum())
        brake = 0.0
        for vl in self.voice_lines:
            if len(vl) and heard[vl].mean() >= c['near']:
                brake = c['g_brake']
        noise = c['noise_b'] * self.rng.standard_normal(self.nbk)
        Bc = c['tonic'] - c['c_mb'] * self.mus.sum() / 4.0 + noise
        blocks = Bc > self.th_b
        stops = brake + c['w_stop'] * c['g_ret_mot'] * intent_s.mean() * 2 + noise > self.th_s
        nb = blocks.mean()
        thr = c['theta_m'] + c['c_bm'] * nb + c['c_sm'] * stops.mean()
        mus = (dn > thr) & (dn >= c['rho'] * dn.max()) if dn.max() > 0 else np.zeros(self.nm, bool)
        k = int(mus.sum())
        if k and (k < c['gate_lo'] or k > c['gate_hi']):        # [X1f] A2 blocking links
            mus = np.zeros(self.nm, bool)
            self.route['gated'] = self.route.get('gated', 0) + 1
        if mus.any() and not self.mus.any():
            m = mus.astype(np.float32)
            parts = dict(heard=float(d_heard @ m), inner=float(d_inner @ m), recall=float(d_rec @ m),
                         pfc_motor=float(d_mot @ m))
            tot = sum(parts.values()) or 1.0
            for kk, v in parts.items():
                self.route[kk] += v / tot
            self.route['onsets'] += 1
        if learn:
            if mus.any() and not self.mus.any():
                self.Ectx += l3.astype(np.float32)
            if r != 0:
                P = self._pred_mat()
                sm, sb = self.sl_mus, self.sl_blk
                if r > 0:
                    P[:, sm] += c['eta_r'] * r * self.E31
                else:
                    P[:, sm] = np.maximum(P[:, sm] - c['eta_r'] * (-r) * self.E31, 0.0)
                    ctx = self.Ectx[:, None] * self.M31[:, sb]
                    P[:, sb] += c['eta_r'] * (-r) * 0.05 * ctx * self._s(P[:, sb])
        self.blocks, self.stops, self.mus, self.nb = blocks, stops, mus, nb
        return mus
