# The three local rule families of ARCH_v2-brainlike.  Every synapse sees only its own pre/post activity, its own
# trace or tag, and at most one broadcast gate M.  Weights are float tables (pre x post); w0 = birth weight.
#
# R1  ORDER rule (sequence areas: ECHO phonological loop, SYL->SYL statistics, CA3 sequences).  The designer's rule
#     "A then B -> A->B stronger; B then A -> A->B weaker", made onset-based (skeptic S5) and given an eligibility
#     window so that events 1..W ticks apart can be linked:
#        e_i(t) = lam^age_i  if age_i <= W else 0      (age_i = ticks since i's last onset, counted before t)
#        forward:  on_j(t)            -> w_ij <- min(w_ij + eta_f * M * e_i(t), wmax)
#        back:     on_i(t)            -> w_ij -= eta_b * M * e_j(t) * w_ij     (multiplicative, floor w0 = 0)
#        same-tick onsets never link (age >= 1).  Optional normalisation AT LEARNING TIME:
#        'pre'  : row sums <= B  (forward transition probability P(j | i), presynaptic resource)
#        'post' : column sums <= B (heterosynaptic, the designer's per-target budget, but immediate)
# R2  ASSOCIATION rule (assembly areas: SYL templates, LEX<->VIS binding, PFC-SUP recurrent assemblies, CA3, CA1):
#        co-activity in either order, held states count:  w_ij += eta * M * x_i * y_j      (x, y in [0,1])
#        then immediately, for every post j that learned:  if sum_i w_ij > B_j: w_.j *= B_j / sum_i w_ij
#     'template' form (competitive learning = Hebb + exact normalisation): winner j:
#        w_.j <- (1 - eta*M) w_.j + eta*M * B * x / |x|
# R3  DECISION rule (premotor readout, copy gate, END): eligibility tag x three-factor, predictive (delta) form,
#     signed weights (the negative part = learned inhibition through an inhibitory pool, Dale-compliant):
#        tag:      e_i <- x_i at the prediction event (or a decaying trace e <- lam*e + x)
#        predict:  d_j = sum_i w_ij e_i ;  p_j = clip(d_j, 0, 1)
#        learn:    w_ij += (eta / max(|e|^2, 1)) * M * e_i * (target_j - p_j)      (normalised LMS)
#     "already predicted -> no more strengthening; predicted but absent -> weaken".
import numpy as np


class OrderPath:
    def __init__(self, n_src, n_dst=None, eta_f=1.0, eta_b=1.0, W=4, lam=1.0, wmax=1.0, norm=None, B=1.0):
        same = n_dst is None
        n_dst = n_src if n_dst is None else n_dst
        self.w = np.zeros((n_src, n_dst), np.float32)
        self.eta_f, self.eta_b, self.W, self.lam, self.wmax, self.norm, self.B = eta_f, eta_b, W, lam, wmax, norm, B
        self.same = same
        self.age_s = np.full(n_src, 10 ** 6)
        self.age_d = self.age_s if same else np.full(n_dst, 10 ** 6)

    def grow(self, n_src, n_dst=None):
        n_dst = n_src if n_dst is None else n_dst
        s0, d0 = self.w.shape
        if n_src > s0 or n_dst > d0:
            w = np.zeros((max(n_src, s0), max(n_dst, d0)), np.float32); w[:s0, :d0] = self.w; self.w = w
            self.age_s = np.r_[self.age_s, np.full(max(n_src - s0, 0), 10 ** 6)]
            self.age_d = self.age_s if self.same else np.r_[self.age_d, np.full(max(n_dst - d0, 0), 10 ** 6)]

    def elig(self, age):
        return np.where(age <= self.W, self.lam ** np.minimum(age, 60), 0.0).astype(np.float32)

    def step(self, on_src, on_dst=None, M=1.0, learn=True, dt=1):
        """on_src / on_dst: index arrays of onsets at this tick; dt = ticks elapsed since the previous call (the
        path may be stepped only at event ticks).  Ages are counted before t (so same-tick = no link)."""
        if dt > 1:
            self.age_s += dt - 1
            if not self.same:
                self.age_d += dt - 1
        on_src = np.asarray(on_src, int)
        on_dst = on_src if on_dst is None else np.asarray(on_dst, int)
        if learn and M > 0:
            es, ed = self.elig(self.age_s), self.elig(self.age_d)
            if len(on_dst) and es.any():
                i = np.nonzero(es)[0]
                sub = self.w[np.ix_(i, on_dst)]
                self.w[np.ix_(i, on_dst)] = np.minimum(sub + self.eta_f * M * es[i, None], self.wmax)
            if len(on_src) and ed.any():
                j = np.nonzero(ed)[0]
                sub = self.w[np.ix_(on_src, j)]
                self.w[np.ix_(on_src, j)] = np.maximum(sub - self.eta_b * M * ed[None, j] * sub, 0.0)
            if self.norm == 'pre' and len(on_dst) and es.any():
                i = np.nonzero(es)[0]; s = self.w[i].sum(1)
                self.w[i] *= np.minimum(1.0, self.B / np.maximum(s, 1e-12))[:, None]
            elif self.norm == 'post' and len(on_dst):
                s = self.w[:, on_dst].sum(0)
                self.w[:, on_dst] *= np.minimum(1.0, self.B / np.maximum(s, 1e-12))[None, :]
        self.age_s += 1
        if not self.same:
            self.age_d += 1
            self.age_d[on_dst] = 0
        self.age_s[on_src] = 0

    def succ(self, i):
        return self.w[i]

    def decay(self, g):
        self.w *= g


class AssocPath:
    """R2.  Effective weight w_ij = raw_ij * s_j (s_j = per-target scale): the immediate per-target normalisation
    'if sum_i w_ij > B: w_.j *= B / sum_i w_ij' is applied exactly, by updating s_j (O(1) per target), so learning
    costs O(active pre x active post).  .w gives the effective table."""

    def __init__(self, n_pre, n_post, eta=0.1, B=1.0, mask=None, w_init=0.0, rng=None):
        self.eta, self.B = eta, B
        self.mask = mask
        if w_init > 0:
            rng = rng or np.random.default_rng(0)
            self.raw = (rng.uniform(0, w_init, (n_pre, n_post))).astype(np.float32)
        else:
            self.raw = np.zeros((n_pre, n_post), np.float32)
        if mask is not None:
            self.raw *= mask
        self.s = np.ones(n_post, np.float32)
        self.rawsum = self.raw.sum(0).astype(np.float64)

    @property
    def w(self):
        return self.raw * self.s[None, :]

    def col(self, j):
        return self.raw[:, j] * self.s[j]

    def current(self, x):
        if x.dtype == bool:
            idx = np.nonzero(x)[0]
            return self.raw[idx].sum(0) * self.s
        return (x @ self.raw) * self.s

    def learn(self, x, y, M=1.0):
        """co-activity: x pre (bool or float), y post (bool or float); immediate per-post normalisation."""
        if M <= 0:
            return
        xi = np.nonzero(x)[0]; yj = np.nonzero(y)[0]
        if not len(xi) or not len(yj):
            return
        xv = (x[xi].astype(np.float32))[:, None]; yv = (y[yj].astype(np.float32))[None, :]
        d = self.eta * M * xv * yv
        if self.mask is not None:
            d = d * self.mask[np.ix_(xi, yj)]
        sj = self.s[yj]
        self.raw[np.ix_(xi, yj)] += d / sj[None, :]
        self.rawsum[yj] += d.sum(0) / sj
        tot = self.rawsum[yj] * sj
        over = tot > self.B
        if over.any():
            jj = yj[over]
            self.s[jj] = (self.s[jj] * (self.B / tot[over])).astype(np.float32)
            small = jj[self.s[jj] < 1e-3]
            for j in small:                                              # re-materialise tiny scales (numerics)
                self.raw[:, j] *= self.s[j]; self.rawsum[j] *= self.s[j]; self.s[j] = 1.0

    def learn_template(self, x, j, M=1.0):
        """competitive form: winner j's weights move toward the normalised input; sum stays B."""
        xv = x.astype(np.float32); n = xv.sum()
        if n <= 0 or M <= 0:
            return
        a = min(1.0, self.eta * M)
        wj = (1 - a) * self.col(j) + a * self.B * xv / n
        self.raw[:, j] = wj; self.s[j] = 1.0; self.rawsum[j] = wj.sum()

    def grow_post(self, n):
        if n > self.raw.shape[1]:
            k = n - self.raw.shape[1]
            self.raw = np.concatenate([self.raw, np.zeros((self.raw.shape[0], k), np.float32)], 1)
            self.s = np.r_[self.s, np.ones(k, np.float32)]; self.rawsum = np.r_[self.rawsum, np.zeros(k)]

    def grow_pre(self, n):
        if n > self.raw.shape[0]:
            self.raw = np.concatenate([self.raw, np.zeros((n - self.raw.shape[0], self.raw.shape[1]), np.float32)], 0)


class DecisionPath:
    def __init__(self, n_pre, n_post, eta=0.3, lam=0.0, clip=True):
        self.w = np.zeros((n_pre, n_post), np.float32)
        self.eta, self.lam, self.clip = eta, lam, clip
        self.e = np.zeros(n_pre, np.float32)

    def tag(self, x):
        """snapshot eligibility at a prediction event (lam = 0) or decaying trace (lam > 0)."""
        x = x.astype(np.float32)
        self.e = self.lam * self.e + x if self.lam > 0 else x.copy()

    def drive(self, x=None):
        x = self.e if x is None else x.astype(np.float32)
        idx = np.nonzero(x)[0]
        return (x[idx, None] * self.w[idx]).sum(0) if len(idx) < 0.3 * len(x) else x @ self.w

    def learn(self, target, M=1.0, extra=None, cols=None):
        """target: (n_post,) in [0,1]; extra: an additive current (e.g. the copy route) that is part of the
        prediction (so the readout learns only the residual); cols: restrict learning to these posts."""
        if M <= 0:
            return
        idx = np.nonzero(self.e)[0]
        if not len(idx):
            return
        ev = self.e[idx]
        d = (ev[:, None] * self.w[idx]).sum(0)
        if extra is not None:
            d = d + extra
        p = np.clip(d, 0.0, 1.0) if self.clip else d
        err = np.asarray(target, np.float32) - p
        if cols is not None:
            m = np.zeros_like(err); m[cols] = 1; err = err * m
        self.w[idx] += (self.eta * M / max(float((ev * ev).sum()), 1.0)) * ev[:, None] * err[None, :]   # NLMS step
        return err

    def grow_post(self, n):
        if n > self.w.shape[1]:
            w = np.zeros((self.w.shape[0], n), np.float32); w[:, :self.w.shape[1]] = self.w; self.w = w

    def grow_pre(self, n):
        if n > self.w.shape[0]:
            w = np.zeros((n, self.w.shape[1]), np.float32); w[:self.w.shape[0]] = self.w; self.w = w
            self.e = np.r_[self.e, np.zeros(n - len(self.e), np.float32)]
