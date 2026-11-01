"""X1c continuous designer core, v3 (follows ARCHITECTURE.md as updated 2026-11-01).

Areas (binary neurons, every tick):
  auditory A0 : thermometer on/off pair per sound feature (1 level) -> exactly half on.
  motor       : muscle neurons (24) and blocking (inhibitory) neurons (8).
  chain pairs : every A0 unit that is not always on, every muscle and every block neuron S sends a forward chain to one
                PFC input neuron X and gets a return chain from that same X.  X is S's copy.
  PFC         : L1 = the copies (forward chain OR internally activated: predicted by the top layer / recalled by the
                hippocampus); L2 = local random feed-forward from L1 (birth), sparser; L3 = top / thinking range: local
                random feed-forward from L2 (birth) + lateral mutual excitation (weak random at birth, learned) +
                persistence; L3 at t -> L1 at t+1 by local Hebbian links (next-input prediction; also holds a thought by
                the loop).
  activity    : arm 'global' = smoothed quantile threshold per layer (development shortcut);
                arm 'local'  = local feedback inhibition (neighbourhood inhibitory pools driven by nearby excitatory
                units, inhibiting them back) + fixed base threshold.  Both: per-neuron homeostatic thresholds that
                slowly track each neuron's own long-run activity, divisive gain control, learned inhibitory partners.
  output      : internally activated X re-lights its S through the return chain (return gain = gene, weaker than the
                links that drive the next item): auditory return = inner hearing -> direct auditory->motor link (the
                babbling result, given at birth here); motor return = speech intention.  Muscle neurons vs blocking
                neurons decide.  Innate brake: a loud partner voice drives the blocking neurons.
  learning    : one rule on all learned links (L3->L1, L3<->L3, hippocampus): j rises -> units already on / just fallen
                strengthen ->j (lateral: j->them weakens); j falls -> units held across j's fall weaken ->j; co-rising
                strengthen each other.  Concave growth, 1 % floor, one global multiplicative decay by depth + pruning.
  values      : innate only (pleased +, upset -, hunger relief +).  A value amplifies the change of the links that were
                just active: + strengthens the just-made top->motor-copy increments; - weakens them and links the
                release context to the blocking copies.
"""
import numpy as np

DEF = dict(
    n2=400, n3=400, k2=30, k3=40, k_in=10, r12=0.2, fan12=24, r23=0.3, fan23=30, r31=0.3, fan33=20, w33=0.1,
    s12=12.0, s23=16.0, act='global', th_lr=1.0, floor2=1.5, floor3=1.0, floor_in=0.35, kappa=2.0, gain=0.01,
    homeo_tau=3000.0, homeo_gain=2.0, homeo_lo=0.3, homeo_hi=3.0, tau_mu=1000.0,
    # arm C: threshold = fraction of each neuron's own in-weight sum, rising with depth
    pred_rel=True, g_int=0.5, block_chain=False, f1=0.35, f2=0.15, f3=0.2, kappa_c=0.0, c2m=2.0, c3m=2.0, tau_own=500.0,
    # local inhibition (arm A)
    n_inh2=40, n_inh3=40, r_inh=0.25, g_loc=0.6, base2=1.5, base3=1.6,
    # learning
    eta=0.02, eta_minus=0.01, eta33=0.0, w_ref=0.5, pw=1.0, plast_floor=0.01, fell_win=2, concave=True,
    dec_every=100, dec_31=0.998, dec_33=0.999, prune=1e-3, norm_every=25, alpha=0.5,
    partners=True, eta_v=0.004, eta_v_co=0.01, g_inh=1.0,
    fast=False, eta_f=0.1, dec_f=0.9995, nlms=False, eta_n=0.3,
    g_loop=0.0, fan31=30,
    f_h=0.6, floor_h=1.0, g_dir=0.5, w_stop=1.5, c_sm=1.0,
    hip=False, nh=300, kh=15, fanh=24, eta_h=0.3, dec_h=0.9995, g_h=1.0,
    # motor / output
    n_b=8, g_ret_aud=1.0, g_ret_mot=1.0, gamma_m=0.1, theta_m=0.22, c_bm=0.5, rho=0.6, tonic=1.0, c_mb=1.2,
    g_brake=3.0, near=0.6, noise_b=0.15, w_bk=0.6,
    eta_r=0.5, lam_r=0.85, rate_cut=0.6,
    learn=True,
)


def _local(rng, pos_pre, pos_post, radius, fan):
    W = np.zeros((len(pos_pre), len(pos_post)), np.float32)
    for j, p in enumerate(pos_post):
        d = np.sqrt(((pos_pre - p) ** 2).sum(1))
        near = np.nonzero(d < radius)[0]
        if len(near) == 0:
            near = np.argsort(d)[:3]
        pick = rng.choice(near, min(fan, len(near)), replace=False)
        W[pick, j] = rng.uniform(0.3, 1.7, len(pick))
    return W


class Brain:
    def __init__(self, phys, seed, rates=None, **kw):
        c = dict(DEF)
        c.update(kw)
        self.c = c
        self.ph = phys
        rng = np.random.default_rng(seed)
        self.rng = rng
        prng = np.random.default_rng(4321)                     # lineage-fixed layout of sensory/motor sheets
        nf, nm, nb = phys.n_a, phys.n_mus, c['n_b']
        self.nf, self.nc, self.nm, self.nbk = nf, phys.n_c, nm, nb
        fpos = prng.random((nf, 2))
        a0pos = np.vstack([fpos, fpos])
        send = np.ones(2 * nf, bool) if rates is None else rates <= c['rate_cut']
        self.send = np.nonzero(send)[0]                        # A0 units with a chain pair
        na1 = len(self.send)
        self.na1 = na1
        n1 = na1 + nm + nb
        self.n1 = n1
        self.sl_mus = slice(na1, na1 + nm)
        self.sl_blk = slice(na1 + nm, n1)
        self.l1_content = np.zeros(n1, bool)
        self.l1_feat = np.full(n1, -1)
        for k, u in enumerate(self.send):
            if u < nf:
                self.l1_feat[k] = u
                self.l1_content[k] = u < phys.n_c
        pos1 = np.vstack([a0pos[self.send], prng.random((nm + nb, 2))])
        pos2 = rng.random((c['n2'], 2))
        pos3 = rng.random((c['n3'], 2))
        self.pos1, self.pos2, self.pos3 = pos1, pos2, pos3
        W12 = _local(rng, pos1, pos2, c['r12'], c['fan12'])
        W23 = _local(rng, pos2, pos3, c['r23'], c['fan23'])
        self.W12 = W12 / np.maximum(W12.sum(0, keepdims=True), 1e-6) * c['s12']
        self.W23 = W23 / np.maximum(W23.sum(0, keepdims=True), 1e-6) * c['s23']
        n3 = c['n3']
        B33 = np.zeros((n3, n3), np.float32)
        for j in range(n3):
            B33[rng.choice(n3, c['fan33'], replace=False), j] = c['w33'] * rng.uniform(0.3, 1.7, c['fan33'])
        np.fill_diagonal(B33, 0)
        self.B33 = B33
        self.L33 = np.zeros((n3, n3), np.float32)
        d31 = np.sqrt(((pos3[:, None, :] - pos1[None, :, :]) ** 2).sum(2))
        self.M31 = (d31 < c['r31']).astype(np.float32)        # top -> L1 links only to nearby L1 neurons
        self.L31 = np.zeros((n3, n1), np.float32)
        self.B31 = np.zeros((n3, n1), np.float32)              # random birth part of the loop (exempt from decay)
        if c['g_loop'] > 0:
            for j in range(n1):
                near = np.nonzero(self.M31[:, j])[0]
                if len(near):
                    pick = rng.choice(near, min(c['fan31'], len(near)), replace=False)
                    w = rng.uniform(0.3, 1.7, len(pick)).astype(np.float32)
                    self.B31[pick, j] = w / w.sum() * c['g_loop']
        self.F31 = np.zeros((n3, n1), np.float32) if c['fast'] else None
        self.F33 = np.zeros((n3, n3), np.float32) if c['fast'] else None
        self.R31 = np.zeros((n3, n1), np.float32) if c['nlms'] else None
        self.V = np.zeros((n3, n3), np.float32) if c['partners'] else None
        # local inhibitory pools (arm A): pool k at a random position, driven by excitatory units within r_inh
        if c['act'] == 'local':
            def pools(pos, n):
                ppos = rng.random((n, 2))
                d = np.sqrt(((pos[:, None, :] - ppos[None, :, :]) ** 2).sum(2))
                Mm = (d < c['r_inh']).astype(np.float32)        # unit x pool membership (drives and receives)
                return Mm / np.maximum(Mm.sum(0, keepdims=True), 1.0)
            self.I2 = pools(pos2, c['n_inh2'])
            self.I3 = pools(pos3, c['n_inh3'])
            self.I2b = (self.I2 > 0).astype(np.float32)
            self.I3b = (self.I3 > 0).astype(np.float32)
        # hippocampus
        if c['hip']:
            nh = c['nh']
            src = n1 + n3
            P = np.zeros((src, nh), np.float32)
            for j in range(nh):
                P[rng.choice(src, c['fanh'], replace=False), j] = rng.uniform(0.3, 1.7, c['fanh'])
            self.Ph = P / P.sum(0, keepdims=True) * 4.0
            self.Whh = np.zeros((nh, nh), np.float32)
            self.Wh1 = np.zeros((nh, n1), np.float32)
            self.h = np.zeros(nh, bool)
            self.h_prev = np.zeros(nh, bool)
            self.th_h = 1.0
            self.fellh = np.zeros(nh, np.int16) + 99
        # auditory -> motor direct link (the babbling result; given at birth here)
        cont = np.zeros((n1, phys.n_c), np.float32)
        for k in range(na1):
            if self.l1_content[k]:
                cont[k, self.l1_feat[k]] = 1.0
        self.Mmap = cont @ (phys.content.T.astype(np.float32) @ phys.need_mat.astype(np.float32)) / phys.k_syl
        self.th_b = c['tonic'] + rng.uniform(-0.45, 0.45, nb)
        self.voice_lines = []
        for v in range(1, 5):
            lo, hi = phys.n_c + v * phys.n_vu, phys.n_c + (v + 1) * phys.n_vu
            self.voice_lines.append(np.nonzero((self.l1_feat >= lo) & (self.l1_feat < hi))[0])
        # state
        self.l1 = np.zeros(n1, bool)
        self.l2 = np.zeros(c['n2'], bool)
        self.l3 = np.zeros(n3, bool)
        self.fell3 = np.zeros(n3, np.int16) + 99
        self.mu1 = np.zeros(n1, np.float32)
        self.mu2 = np.zeros(c['n2'], np.float32) + 0.5
        self.mu3 = np.zeros(c['n3'], np.float32) + 0.5
        self.h1 = np.zeros(n1, np.float32)
        self.h2 = np.zeros(c['n2'], np.float32)
        self.h3 = np.zeros(n3, np.float32)
        self.th2, self.th3, self.th_in = 1.0, 1.0, 1.0
        self.inh2 = np.zeros(c['n2'], np.float32)
        self.inh3 = np.zeros(n3, np.float32)
        self.pred = np.zeros(n1, bool)
        self.mus = np.zeros(nm, bool)
        self.blocks = np.ones(nb, bool)
        self.stops = np.zeros(nb, bool)
        self.th_s = 1.0 + rng.uniform(-0.2, 0.2, nb)
        self.route = dict(heard=0.0, inner=0.0, recall=0.0, pfc_motor=0.0, onsets=0)
        self.nb = 1.0
        self.E31 = np.zeros((n3, nm), np.float32)              # eligibility of top->muscle-copy increments
        self.Ectx = np.zeros(n3, np.float32)                    # release context
        self.t = 0
        self.cnt = dict(n1=[], n2=[], n3=[])
        self._norm()
        if c['act'] in ('frac', 'own'):
            self.V = None                                      # arm C: no inhibitory neurons in the PFC

    # ------------------------------------------------------------------ helpers
    def _s(self, w):
        c = self.c
        if not c['concave']:
            return np.ones_like(np.asarray(w, np.float32))      # linear growth control: no firm-memory factor
        return np.maximum(1.0 / (1.0 + np.abs(w) / c['w_ref']) ** c['pw'], c['plast_floor'])

    def _nrm(self, W):
        """two-layer weights [D char-net]: propagation = strength / (out^a in^(1-a)); total mass kept."""
        a = self.c['alpha']
        if W.sum() <= 0:
            return W.copy()
        rs, cs = W.sum(1), W.sum(0)
        rs_f = np.maximum(rs, 0.3 * np.median(rs[rs > 0]))
        cs_f = np.maximum(cs, 0.3 * np.median(cs[cs > 0]))
        P = W / (rs_f[:, None] ** a * cs_f[None, :] ** (1 - a))
        return P * (W.sum() / max(P.sum(), 1e-9))

    def _pred_mat(self):
        return self.R31 if self.R31 is not None else self.L31

    def _norm(self):
        W31 = self._pred_mat() + self.B31 + (self.F31 if self.F31 is not None else 0)
        self.P31 = self._nrm(np.maximum(W31, 0) * self.M31)
        W33 = np.maximum(self.B33 + self.L33 + (self.F33 if self.F33 is not None else 0), 0)
        self.P33 = self._nrm(W33)
        self.in1 = self.P31.sum(0)                             # each neuron's own in-weight sums (arm C thresholds)
        self.in2 = self.W12.sum(0)
        self.in3 = self.W23.sum(0) + self.P33.sum(0)

    def snapshot(self):
        keys = ['l1', 'l2', 'l3', 'fell3', 'pred', 'mus', 'blocks', 'stops', 'E31', 'Ectx', 'inh2', 'inh3', 'ext_prev']
        d = {k: getattr(self, k).copy() for k in keys}
        d.update(th2=self.th2, th3=self.th3, th_in=self.th_in, nb=self.nb, rng=self.rng.bit_generator.state)
        if self.c['hip']:
            d.update(h=self.h.copy(), h_prev=self.h_prev.copy(), th_h=self.th_h, fellh=self.fellh.copy())
        return d

    def restore(self, d):
        for k, v in d.items():
            if k == 'rng':
                self.rng.bit_generator.state = v
            elif isinstance(v, np.ndarray):
                setattr(self, k, v.copy())
            else:
                setattr(self, k, v)

    @staticmethod
    def _qthr(pot, k, prev, lr, floor=None, bounded=True):
        n = len(pot)
        srt = np.sort(pot)
        th = (1 - lr) * prev + lr * srt[n - k]
        if bounded:
            th = float(np.clip(th, srt[n - 2 * k], srt[n - max(1, k // 2)]))
        if floor is not None:
            th = max(floor, th)
        return th

    def _homeo(self, h, a, k, n, learn):
        if learn and self.c['homeo_gain'] > 0:
            h += self.c['homeo_gain'] * (a.astype(np.float32) - k / n) / self.c['homeo_tau']
            np.clip(h, -self.c['homeo_lo'], self.c['homeo_hi'], out=h)   # bounded homeostatic offset

    # ------------------------------------------------------------------ one tick
    def step(self, feat, r=0.0, learn=None):
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
        x3 = (self.W23[a2].sum(0) if len(a2) else 0) + (self.P33[a3].sum(0) if len(a3) else 0)
        x3 = np.asarray(x3, np.float32)
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
            l3 = p3 - c['g_loc'] * self.inh3 > c['base3']
            self.inh3 = self.I3b @ (self.I3.T @ l3.astype(np.float32)) * 4.0
        if c['act'] not in ('frac', 'own'):
            self._homeo(self.h3, l3, c['k3'], c['n3'], learn)
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

    # ------------------------------------------------------------------ the one learning rule
    def _rule(self, L, Bw, pre_prev, pre_now, pre_fell, post_prev, post_now, eta, etm, mask=None, lateral=False,
              fast=None, track=None):
        rise = np.nonzero(post_now & ~post_prev)[0]
        fall = np.nonzero(post_prev & ~post_now)[0]
        earlier = np.nonzero(pre_prev | (pre_fell <= self.c['fell_win']))[0]
        held = np.nonzero(pre_prev & pre_now)[0]
        ef = self.c['eta_f']

        def wt(ix):
            return L[ix] + (Bw[ix] if Bw is not None else 0)

        def lo(ix):
            return -(Bw[ix] if Bw is not None else 0)
        if len(rise) and len(earlier):
            ix = np.ix_(earlier, rise)
            d = eta * self._s(wt(ix))
            if lateral:
                d = d * (earlier[:, None] != rise[None, :])
            if mask is not None:
                d = d * mask[ix]
            L[ix] += d
            if fast is not None:
                fast[ix] += ef * (d > 0)
            if track is not None:
                track(earlier, rise, d)
            if lateral:
                ixr = np.ix_(rise, earlier)
                dr = etm * self._s(wt(ixr)) * (rise[:, None] != earlier[None, :])
                L[ixr] = np.maximum(L[ixr] - dr, lo(ixr))
                if fast is not None:
                    fast[ixr] -= ef * 0.5 * (dr > 0)
        if len(fall) and len(held):
            ix = np.ix_(held, fall)
            d = eta * self._s(wt(ix))
            if lateral:
                d = d * (held[:, None] != fall[None, :])
            if mask is not None:
                d = d * mask[ix]
            L[ix] = np.maximum(L[ix] - d, lo(ix))
            if fast is not None:
                fast[ix] -= ef * (d > 0)
        if lateral and len(rise) > 1:
            ix = np.ix_(rise, rise)
            d = eta * self._s(wt(ix))
            np.fill_diagonal(d, 0)
            L[ix] += d
            if fast is not None:
                fast[ix] += ef * (d > 0)

    def _learn(self, l1p, l1, l3p, l3, ext):
        c = self.c
        sm = self.sl_mus

        def track(pre, post, d):                               # eligibility of top -> muscle-copy increments
            m = (post >= sm.start) & (post < sm.stop)
            if m.any():
                self.E31[np.ix_(pre, post[m] - sm.start)] += d[:, m]
        self.E31 *= c['lam_r']
        self.Ectx *= c['lam_r']
        if self.R31 is None:
            self._rule(self.L31, None, l3p, l3, self.fell3, l1p, l1, c['eta'], c['eta_minus'], mask=self.M31,
                       fast=self.F31, track=track)
        else:                                                  # NLMS readout: error-driven next-input prediction
            if len(np.nonzero(l3p)[0]):
                a3 = np.nonzero(l3p)[0]
                y = self.R31[a3].sum(0)
                err = ext.astype(np.float32) - y
                self.R31[a3] += c['eta_n'] * err[None, :] * self.M31[a3] / (len(a3) + 1.0)
        self._rule(self.L33, self.B33, l3p, l3, self.fell3, l3p, l3, c['eta33'], c['eta33'] * 0.5, lateral=True,
                   fast=self.F33)
        if self.V is not None and c['act'] not in ('frac', 'own'):
            R = np.nonzero(l3 & ~l3p)[0]
            if len(R):
                off = np.nonzero(~l3)[0]
                ix = np.ix_(R, off)
                self.V[ix] += c['eta_v'] * self._s(self.V[ix])
                ix2 = np.ix_(np.nonzero(l3)[0], R)
                self.V[ix2] = np.maximum(self.V[ix2] - c['eta_v_co'], 0.0)

    def _decay(self):
        c = self.c
        for A, g in ((self._pred_mat(), c['dec_31']), (self.L33, c['dec_33'])):
            A *= g
            A[np.abs(A) < c['prune']] = 0.0
        if self.V is not None:
            self.V *= c['dec_33']
            self.V[self.V < c['prune']] = 0.0

    # ------------------------------------------------------------------ motor area (output via return chains)
    def _motor(self, heard, pred, l3, r, learn):
        c = self.c
        rc = self.recall if hasattr(self, 'recall') else np.zeros(self.n1, bool)
        cont = self.l1_content
        d_heard = c['g_dir'] * ((heard & cont[:self.na1]).astype(np.float32) @ self.Mmap[:self.na1])   # direct link
        d_inner = c['g_ret_aud'] * ((pred & ~rc & cont).astype(np.float32) @ self.Mmap)               # PFC aud. return
        d_rec = c['g_ret_aud'] * ((rc & cont).astype(np.float32) @ self.Mmap)                         # recall
        d_mot = c['g_ret_mot'] * (pred | rc)[self.sl_mus].astype(np.float32)                           # PFC motor return
        intent_s = (pred | rc)[self.sl_blk].astype(np.float32)                                         # PFC -> stop neurons
        d = d_heard + d_inner + d_rec + d_mot
        dn = d / (1.0 + c['gamma_m'] * d.sum())
        brake = 0.0
        for vl in self.voice_lines:                            # innate brake (birth reflex link)
            if len(vl) and heard[vl].mean() >= c['near']:
                brake = c['g_brake']
        noise = c['noise_b'] * self.rng.standard_normal(self.nbk)
        Bc = c['tonic'] - c['c_mb'] * self.mus.sum() / 4.0 + noise
        blocks = Bc > self.th_b                                 # tonic blocking neurons: win by default
        stops = brake + c['w_stop'] * c['g_ret_mot'] * intent_s.mean() * 2 + noise > self.th_s   # phasic stop neurons
        nb = blocks.mean()
        thr = c['theta_m'] + c['c_bm'] * nb + c['c_sm'] * stops.mean()
        mus = (dn > thr) & (dn >= c['rho'] * dn.max()) if dn.max() > 0 else np.zeros(self.nm, bool)
        if mus.any() and not self.mus.any():                   # which route drove this onset
            m = mus.astype(np.float32)
            parts = dict(heard=float(d_heard @ m), inner=float(d_inner @ m), recall=float(d_rec @ m),
                         pfc_motor=float(d_mot @ m))
            tot = sum(parts.values()) or 1.0
            for k, v in parts.items():
                self.route[k] += v / tot
            self.route['onsets'] += 1
        if learn:
            if mus.any() and not self.mus.any():
                self.Ectx += l3.astype(np.float32)             # release context
            if r != 0:
                P = self._pred_mat()
                sm, sb = self.sl_mus, self.sl_blk
                if r > 0:                                      # amplify the just-made increments
                    P[:, sm] += c['eta_r'] * r * self.E31
                else:                                          # weaken them; the release context -> blocking copies
                    P[:, sm] = np.maximum(P[:, sm] - c['eta_r'] * (-r) * self.E31, 0.0)
                    ctx = self.Ectx[:, None] * self.M31[:, sb]
                    P[:, sb] += c['eta_r'] * (-r) * 0.05 * ctx * self._s(P[:, sb])
        self.blocks, self.stops, self.mus, self.nb = blocks, stops, mus, nb
        return mus

    # ------------------------------------------------------------------ hippocampus
    def _hip(self, l1, l3, learn):
        c = self.c
        src = np.concatenate([l1, l3])
        a = np.nonzero(src)[0]
        hp = self.h
        ah = np.nonzero(hp)[0]
        pot = self.Ph[a].sum(0) / (1.0 + c['gain'] * len(a)) if len(a) else np.zeros(c['nh'], np.float32)
        if len(ah):
            pot = pot + self.Whh[ah].sum(0) / (1.0 + 0.05 * len(ah))
        self.th_h = self._qthr(pot, c['kh'], self.th_h, c['th_lr'], floor=0.3, bounded=False)
        h = pot > self.th_h
        if learn:
            self._rule(self.Whh, None, hp, h, self.fellh, hp, h, c['eta_h'], c['eta_h'] * 0.5, lateral=True)
            self._rule(self.Wh1, None, hp, h, self.fellh, self._l1h_prev if hasattr(self, '_l1h_prev') else l1, l1,
                       c['eta_h'], 0.0)
            if self.t % 10 == 0:
                self.Whh *= c['dec_h'] ** 10
                self.Wh1 *= c['dec_h'] ** 10
        self.fellh = np.where(hp & ~h, 0, np.minimum(self.fellh + 1, 99)).astype(np.int16)
        self._l1h_prev = l1.copy()
        self.h_prev = hp
        self.h = h
