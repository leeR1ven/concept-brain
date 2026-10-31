# -*- coding: utf-8 -*-
"""ARCH_v2-P core: ONE local learning rule (the predictive rule, PR) and the unit / area dynamics.

PR -- for every learned synapse i -> j of a path P from source area S into target group D:

    dw_ij = eta_D * theta_D * M_D(t) * pi_P * (g_S x_i(t-1)) * (y_j(t) - p_j(t)) / n_j(t-1)

  x_i(t)  presynaptic output trace:  x_i(t) = max(s_i(t), lam_S * x_i(t-1))      (lam_S = 0: the spike itself)
  g_S     fixed input gain of source area S (default 1/sqrt(k_S): every population brings about unit power)
  n_j     sum over ALL synapses onto j of pi_P (g_S x_i)^2 -> normalisation at learning time (heterosynaptic);
          one error is shared by all paths onto j, so a fast path (large pi, e.g. hippocampal) takes most of it
  pi_P    plasticity factor of path P (1 = cortical default)
  y_j(t)  the target = what j actually did, set by j's DRIVER input (sensory input, efference copy, own hold):
            event groups : y_j = 1 iff j is the event at an event tick (e.g. 'char c heard now'); learning only at
                           event ticks (the group's event clock is a factor of M)
            state groups : y_j = s_j(t)   (held activity counts -> co-activity is a target -> assemblies)
  p_j(t)  j's own prediction  p_j = clip(d_j / theta_D, 0, 1),  d_j = sum_paths sum_i g_S x_i(t-1) w_ij + b_j
  w       signed.  w = w+ - w-: the negative part is the disynaptic path through j's local inhibitory
          interneuron (learned inhibition; Dale's law kept: every real synapse keeps one sign)
  M_D(t)  broadcast third factor of the group: mode gate (listening / own voice / sleep) x novelty, or the reward
          prediction error delta for self-produced targets.  Default 1.

Properties (unit-tested in test_rule.py):
  unpredicted target -> strengthen; predicted and present -> no change; predicted but absent -> weaken (can go
  below 0 = learned inhibition); weights estimate P(target | context) (no budget needed); held sources count
  through the trace; order is respected (a source that comes after the target gets no credit); M = 0 freezes.

Everything else in the model is wiring (who drives whom, which paths are learned) and parameters per area:
size n, active count k (k-WTA) or threshold, leak of the potential lam_v (persistence), trace lam_x, eta, theta.
"""
from __future__ import annotations

import numpy as np

F32 = np.float32


def topk_mask(v, k):
    """boolean mask of the k largest entries of v (k-WTA)."""
    s = np.zeros(v.shape[0], bool)
    if k <= 0:
        return s
    if k >= v.shape[0]:
        s[:] = True
        return s
    s[np.argpartition(-v, k)[:k]] = True
    return s


class Area:
    """binary units, leaky potential, k-WTA.  v(t) = lam_v * v(t-1) + I(t) (+ fixed tie-break);  s = top-k(v).
    hold=True freezes v and s (BG-gated working memory: the input gate is closed).  x = presynaptic trace."""

    def __init__(self, n, k, lam_v=0.0, lam_x=0.0, gain=None, seed=0, tie=1e-3):
        rng = np.random.default_rng(seed)
        self.n, self.k, self.lam_v, self.lam_x = n, k, lam_v, lam_x
        self.gain = F32(1.0 / np.sqrt(max(k, 1))) if gain is None else F32(gain)
        self.tie = (rng.random(n) * tie).astype(F32)
        self.v = np.zeros(n, F32)
        self.s = np.zeros(n, bool)
        self.prev = np.zeros(n, bool)
        self.x = np.zeros(n, F32)

    def step(self, I, hold=False):
        self.prev = self.s
        if not hold:
            self.v = (self.lam_v * self.v + I).astype(F32)
            self.s = topk_mask(self.v + self.tie, self.k)
        self.x = np.maximum(self.s.astype(F32), F32(self.lam_x) * self.x)
        return self.s

    def set(self, s):
        """externally clamped state (e.g. a localist detector)."""
        self.prev = self.s
        self.s = s.astype(bool)
        self.x = np.maximum(self.s.astype(F32), F32(self.lam_x) * self.x)

    def reset(self):
        self.v[:] = 0
        self.s = np.zeros(self.n, bool)
        self.prev = np.zeros(self.n, bool)
        self.x[:] = 0

    @property
    def out(self):
        """gained presynaptic signal sent along learned paths."""
        return self.gain * self.x


class Trace:
    """presynaptic trace of an arbitrary binary/graded signal (for sources that are not Areas)."""

    def __init__(self, n, lam, gain=1.0):
        self.n, self.lam, self.gain = n, F32(lam), F32(gain)
        self.x = np.zeros(n, F32)

    def step(self, s):
        self.x = np.maximum(np.asarray(s, F32), self.lam * self.x)
        return self.x

    def reset(self):
        self.x[:] = 0

    @property
    def out(self):
        return self.gain * self.x


class Path:
    """learned weights from one source population into a Predictor group.  Dense (n_src x n_dst) float32; an
    optional fixed fan-in mask (synapses that exist).  Only rows of active sources are touched."""

    def __init__(self, n_src, n_dst, mask=None, w_scale=0.0, seed=0, pi=1.0):
        rng = np.random.default_rng(seed)
        self.pi = np.float32(pi)                 # plasticity factor of this path (fast hippocampal vs slow cortical)
        self.W = np.zeros((n_src, n_dst), F32)
        if w_scale > 0:
            self.W[:] = (rng.random((n_src, n_dst)) * w_scale).astype(F32)
        self.mask = mask
        if mask is not None:
            self.W *= mask

    def drive(self, xs):
        a = np.flatnonzero(xs)
        if not len(a):
            return np.zeros(self.W.shape[1], F32)
        return xs[a] @ self.W[a]

    def update(self, xs, err, scale):
        a = np.flatnonzero(xs)
        c = np.flatnonzero(err)
        if not len(a) or not len(c):
            return
        ix = np.ix_(a, c)
        d = scale * np.outer(xs[a], err[c]).astype(F32)
        if self.mask is not None:
            d *= self.mask[ix]
        self.W[ix] += d

    def grow(self, n_new):
        """add target columns (new units recruited into the group)."""
        self.W = np.concatenate([self.W, np.zeros((self.W.shape[0], n_new), F32)], 1)
        if self.mask is not None:
            self.mask = np.concatenate([self.mask, np.ones((self.mask.shape[0], n_new), bool)], 1)


class Predictor:
    """a group of target units that predicts its own targets y from several source populations with PR.
    d_j = b_j + sum_paths drive;  p_j = clip(d_j / theta, 0, 1).  learn(): one PR step from the stored inputs."""

    def __init__(self, n, eta, theta=1.0, bias=True, name=''):
        self.n, self.eta, self.theta, self.name = n, F32(eta), F32(theta), name
        self.paths = {}
        self.b = np.zeros(n, F32)
        self.use_bias = bias
        self.d = np.zeros(n, F32)
        self.p = np.zeros(n, F32)
        self._xs = None
        self.n_updates = 0

    def add(self, name, n_src, mask=None, pi=1.0):
        self.paths[name] = Path(n_src, self.n, mask, pi=pi)
        return self.paths[name]

    def predict(self, xs: dict, extra=None):
        """xs: name -> gained source signal at t-1 (missing name = silent source).  Stores xs for learn().
        extra: a fixed (non-learned) additive input to d (e.g. a gated one-to-one mirror path)."""
        d = self.b.copy() if self.use_bias else np.zeros(self.n, F32)
        if extra is not None:
            d += extra
        for nm, P in self.paths.items():
            x = xs.get(nm)
            if x is not None:
                d += P.drive(x)
        self.d = d
        self.p = np.clip(d / self.theta, 0.0, 1.0).astype(F32)
        self._xs = xs
        return self.p

    def learn(self, y, M=1.0, xs=None, p=None, cols=None, eta=None):
        """PR step with target y (n,) in {0,1} (or graded), third factor M (scalar or (n,)).  Uses the inputs of the
        last predict() unless xs/p are given.  cols: restrict learning to these target units (others untouched)."""
        xs = self._xs if xs is None else xs
        p = self.p if p is None else p
        if xs is None:
            return
        err = (np.asarray(y, F32) - p) * np.asarray(M, F32)
        if cols is not None:
            m = np.zeros(self.n, F32); m[cols] = 1; err = err * m
        if not np.any(err):
            return
        # per-target input power n_j over the synapses that exist (normalisation at learning time)
        n_j = np.full(self.n, 1.0 if self.use_bias else 0.0, F32)   # the bias = an always-on input of power 1
        for nm, P in self.paths.items():
            x = xs.get(nm)
            if x is None:
                continue
            a = np.flatnonzero(x)
            if not len(a):
                continue
            x2 = (P.pi * x[a] ** 2).astype(F32)
            n_j += (x2 @ P.mask[a]) if P.mask is not None else F32(x2.sum())
        e = np.where(n_j > 0, err * F32(self.eta if eta is None else eta) * self.theta / np.maximum(n_j, 1e-9), 0)
        e = e.astype(F32)
        for nm, P in self.paths.items():
            x = xs.get(nm)
            if x is not None:
                P.update(x, e, P.pi)
        if self.use_bias:
            self.b += e
        self.n_updates += 1

    def grow(self, n_new):
        for P in self.paths.values():
            P.grow(n_new)
        self.b = np.r_[self.b, np.zeros(n_new, F32)]
        self.d = np.r_[self.d, np.zeros(n_new, F32)]
        self.p = np.r_[self.p, np.zeros(n_new, F32)]
        self.n += n_new


class Recurrent:
    """a state area with learned recurrent PR synapses (assemblies): fixed driver input + learned recurrence.
    s(t) = top-k( lam_v v(t-1) + drive(t) + c_rec * d_rec(t) ),  d_rec = W^T x(t-1)  (the learned prediction of
    s(t)).  learn(): target y = s(t) (state target), prediction p = clip(d_rec/theta, 0, 1).  Fan-in mask C per unit."""

    def __init__(self, n, k, fanin, eta, theta=1.0, lam_v=0.0, lam_x=0.0, c_rec=1.0, seed=0, tie=1e-3,
                 sp_beta=0.0, sp_tau=500.0):
        rng = np.random.default_rng(seed)
        self.sp_beta, self.sp_tau = F32(sp_beta), F32(sp_tau)       # homeostatic set point (framework: net.py sp_beta)
        self.u = np.full(n, k / n, F32)
        self.A = Area(n, k, lam_v=lam_v, lam_x=lam_x, gain=1.0, seed=seed, tie=tie)
        self.n, self.k = n, k
        mask = np.zeros((n, n), bool)
        for i in range(n):
            mask[i, rng.choice(n, min(fanin, n - 1), replace=False)] = True
        np.fill_diagonal(mask, False)
        self.P = Predictor(n, eta, theta, bias=False)
        self.P.add('rec', n, mask)
        self.c_rec = F32(c_rec)
        self.d = np.zeros(n, F32)

    def step(self, drive, learn=False, M=1.0, hold=False):
        x_prev = self.A.x.copy()
        p = self.P.predict({'rec': x_prev})
        self.d = self.P.d
        base = 0.0
        if self.sp_beta > 0:
            base = -self.sp_beta * np.maximum(self.u / (self.k / self.n) - 1.0, 0.0)
        s = self.A.step(drive + self.c_rec * self.P.d + base, hold=hold)
        if self.sp_beta > 0:
            self.u += (s.astype(F32) - self.u) / self.sp_tau
        if learn:
            self.P.learn(s.astype(F32), M=M)
        return s
