# -*- coding: utf-8 -*-
"""ARCH_v2-develop: the learning rules and memory primitives (all local).

R1  OneShotChain      one-shot sequence binding inside a line (hippocampal / PFC fast chain; the designer's
                      "earlier -> later" order rule at eta = 1, cleared at the next line onset).
R2  Delta             signed error-driven readout = predictive form of the order rule:
                      dW_ij = eta * g * m_i * x_i * (t_j - clip(d_j, 0, 1)) / |A|
                      x_i: source on at the previous event; t_j: the target unit's own driven activity (e.g. its
                      auditory mirror fired because the char was HEARD); d_j: the target's total dendritic drive
                      (all routes); g: broadcast third factor; m_i: per-source maturation (plasticity falls with use).
                      "already predicted -> no more strengthening"; "predicted but absent -> weaken" (negative part =
                      learned inhibition through interneurons; Dale's law kept by an E and an I pathway).
R3  MixDict           conjunctive (mixed-selectivity) units: one unit per observed pair of input features, recruited
                      on first co-occurrence ("fresh units join"); least-recently-used units are deleted when the pool
                      is full (the designer's delete of never-active units).
R4  Hippo             episodic index: fixed DG sparse expansion (k-WTA) + recruited one-shot episode cells; recall by
                      overlap (WTA over cells); strength decays slowly; replay sampling for sleep.
R5  Templates         auditory mirror Z: per-program sound templates learned from the own voice (babbling) and
                      from the tutor's imitation; detection = WTA of normalised template match at a local peak.
R6  Critic            value V(context) and reward-prediction error delta = r - V (dopamine-like third factor).
"""
from __future__ import annotations

import numpy as np


# ----------------------------------------------------------------------------------------------- R1
class OneShotChain:
    """Ordered memory of one line: items in order; successor query with 'inhibition of return' on used positions."""

    def __init__(self, items=()):
        self.items = list(items)

    def add(self, c):
        self.items.append(c)

    def clear(self):
        self.items = []

    def candidate(self, last, used):
        """first successor of `last` (None = line start) at a position not yet used in the current reply."""
        it = self.items
        if not it:
            return None, -1
        if last is None:
            return (it[0], 0) if 0 not in used else (None, -1)
        for k in range(len(it) - 1):
            if it[k] == last and (k + 1) not in used:
                return it[k + 1], k + 1
        return None, -1


# ----------------------------------------------------------------------------------------------- R2
class Delta:
    """Signed error-driven readout over a growing set of binary sources."""

    def __init__(self, n_out, cap=4096, eta=0.4, mature=0.0, m_min=0.0, dtype=np.float32, nu=0.0):
        self.W = np.zeros((cap, n_out), dtype)
        self.use = np.zeros(cap, np.float32)
        self.eta, self.mature, self.m_min = eta, mature, m_min
        self.nu = nu                  # tune: normalisation |A|^(1-nu) * (sum m)^nu  (nu=1: plastic units share the step)
        self.n_out = n_out
        self.last_err = 0.0           # tune: max |t - p| of the last learning event (for split)
        self.U = None                 # x2: per-synapse use trace (None = the stability rule is off; v2.2 bit for bit)
        self.stab, self.stab_ref, self.stab_min = 0.0, 1.0, 0.0

    def enable_stab(self, beta, w_ref, f_min, track=False):
        """x2 (the designer's weight-stability rule, 2026-11-01): each synapse keeps a use trace u_ij = the number of
        learning events it took part in; its step (both signs) is scaled by
            f_ij = max(1 / (1 + beta * u_ij * |w_ij| / w_ref), f_min)
        i.e. eta_eff = eta * f_ij: a strong, often-used synapse is hard to raise AND hard to lower; a zero-weight synapse
        is fully plastic whatever its use (so a new output or a fresh unit learns at full rate: no per-row block).
        track=True (beta 0): keep u only (diagnostics; the steps are unchanged)."""
        self.U = np.zeros(self.W.shape, np.float32)
        self.stab, self.stab_ref, self.stab_min = float(beta), float(w_ref), float(f_min)
        self.stab_track = bool(track)

    def _stab_f(self, u, w):
        f = 1.0 / (1.0 + (self.stab / self.stab_ref) * u * np.abs(w))
        return np.maximum(f, self.stab_min).astype(np.float32)

    def nudge(self, A, y, v):
        """a direct (third-factor / actor) step v on the synapses (A, y); the stability rule applies to it as well."""
        if self.U is None:
            self.W[A, y] += v
            return
        if self.stab > 0:
            self.W[A, y] += (v * self._stab_f(self.U[A, y], self.W[A, y])).astype(self.W.dtype)
        else:
            self.W[A, y] += v
        self.U[A, y] += 1.0

    def plasticity(self, A):
        """per-source plasticity m_i = max(1 / (1 + use_i / M), m_min) (1 when maturation is off)"""
        if self.mature > 0:
            return np.maximum(1.0 / (1.0 + self.use[A] / self.mature), self.m_min).astype(np.float32)
        return np.ones(len(A), np.float32)

    def grow(self, n_in):
        if n_in > self.W.shape[0]:
            new = max(n_in, int(1.25 * self.W.shape[0]) + 1)      # tune: 1.25x growth (memory); capacity only
            W = np.zeros((new, self.n_out), self.W.dtype); W[:self.W.shape[0]] = self.W; self.W = W
            u = np.zeros(new, np.float32); u[:len(self.use)] = self.use; self.use = u
            if self.U is not None:
                U = np.zeros(W.shape, np.float32); U[:self.U.shape[0]] = self.U; self.U = U

    def add_out(self, n_out):
        if n_out > self.n_out:
            W = np.zeros((self.W.shape[0], n_out), self.W.dtype); W[:, :self.n_out] = self.W
            if self.U is not None:
                U = np.zeros(W.shape, np.float32); U[:, :self.n_out] = self.U; self.U = U
            self.W, self.n_out = W, n_out

    def drive(self, A):
        if len(A) == 0:
            return np.zeros(self.n_out, np.float32)
        return self.W[A].sum(0)

    def learn(self, A, target, total, J=None, g=1.0, eta=None, grp=None):
        """A: active source ids; target/total: arrays over J (or all outputs); total = summed drive of all routes.
        grp: pathway (dendritic branch) id of each source; the step is shared equally between the active branches and
        within a branch by its active count (branch-level normalisation).  grp=None: one branch (plain NLMS)."""
        if len(A) == 0 or g == 0:
            return 0.0
        eta = self.eta if eta is None else eta
        p = np.clip(total, 0.0, 1.0)
        err = (target - p).astype(np.float32)
        if J is None:
            J = np.nonzero(np.abs(err) > 1e-4)[0]; err = err[J]
        else:
            J = np.asarray(J); keep = np.abs(err) > 1e-4; J, err = J[keep], err[keep]
        if len(J) == 0:
            self.last_err = 0.0
            return 0.0
        self.last_err = float(np.abs(err).max())
        m = np.maximum(1.0 / (1.0 + self.use[A] / self.mature), self.m_min) if self.mature > 0 else np.ones(len(A), np.float32)
        if grp is None and self.nu > 0 and self.mature > 0:
            sm = float(m.sum())
            norm = np.full(len(A), 1.0 / (len(A) ** (1.0 - self.nu) * max(sm, 1e-6) ** self.nu), np.float32)
        elif grp is None:
            norm = np.full(len(A), 1.0 / len(A), np.float32)
        else:
            u, inv, cnt = np.unique(grp, return_inverse=True, return_counts=True)
            norm = (1.0 / (len(u) * cnt[inv])).astype(np.float32)
        step = (eta * g) * (norm * m)[:, None] * err[None, :]
        if self.U is None:
            self.W[np.ix_(A, J)] += step.astype(self.W.dtype)
        else:                                                   # x2: per-synapse stability (both signs)
            ix = np.ix_(A, J)
            u = self.U[ix]
            if self.stab > 0:
                step = step * self._stab_f(u, self.W[ix])
            self.W[ix] += step.astype(self.W.dtype)
            self.U[ix] = u + 1.0
        self.use[A] += 1.0
        return float(np.abs(err).sum())

    def scale(self, f):
        self.W *= f


# ----------------------------------------------------------------------------------------------- R7
class Outstar:
    """each active source i learns the conditional frequency of the target pattern (Grossberg outstar):
    w_i <- w_i + m_i * (t - w_i),  m_i = max(1 / n_i, m_min)  (plasticity falls with use = maturation).
    Local: the synapse i->b sees only its source's activity, its target's activity and its own weight."""

    def __init__(self, n_src, n_out, m_min=0.02):
        self.W = np.zeros((n_src, n_out), np.float32)
        self.n = np.zeros(n_src, np.float32)
        self.m_min = m_min

    def learn(self, A, t, J=None):
        if len(A) == 0:
            return
        self.n[A] += 1
        m = np.maximum(1.0 / self.n[A], self.m_min)[:, None]
        if J is None:
            self.W[A] += m * (t[None, :] - self.W[A])
        else:
            sub = self.W[np.ix_(A, J)]
            self.W[np.ix_(A, J)] = sub + m * (t[None, :] - sub)

    def conf(self, A, thr, nmin):
        """(len(A), n_out) bool: confident expectations of each source with enough evidence (else all False)"""
        ok = (self.n[A] >= nmin)[:, None]
        return (self.W[A] >= thr) & ok


# ----------------------------------------------------------------------------------------------- R3
class MixDict:
    """Recruited conjunctive units.  key -> unit id.  Direct features are units too (first-order mixed units)."""

    def __init__(self, cap=200000, recruit_p=1.0, seed=0):
        self.recruit_p = recruit_p    # tune: a new pair gets a unit with this probability per co-occurrence
        self.rng = np.random.default_rng(seed + 991)
        self.key_of = {}              # unit -> key
        self.ids = {}
        self.last = np.zeros(cap, np.int64)
        self.grp = np.zeros(cap, np.int32)
        self.gid = {}
        self.cap = cap
        self.n = 0
        self.clock = 0
        self.free = []
        self.deleted = 0

    def units(self, keys, create=True, tick=True):
        if tick:
            self.clock += 1
        out = []
        for k in keys:
            u = self.ids.get(k)
            if u is None:
                if not create:
                    continue
                if self.recruit_p < 1.0 and self.rng.random() >= self.recruit_p:
                    continue
                if self.free:
                    u = self.free.pop()
                elif self.n < self.cap:
                    u = self.n; self.n += 1
                else:
                    continue
                self.ids[k] = u
                self.key_of[u] = k
                gk = k[0] if len(k) == 2 else ((k[0], k[2]) if len(k) == 4 else (k[1], k[3]))
                self.grp[u] = self.gid.setdefault(gk, len(self.gid))
            self.last[u] = self.clock
            out.append(u)
        return np.array(sorted(set(out)), np.int64)

    def prune(self, older_than, readouts=()):
        """delete units not used for `older_than` events; their readout rows are zeroed (synapses removed)."""
        dead = [k for k, u in self.ids.items() if self.clock - self.last[u] > older_than]
        for k in dead:
            u = self.ids.pop(k)
            self.key_of.pop(u, None)
            for r in readouts:
                if hasattr(r, 'use'):
                    if u < r.W.shape[0]:
                        r.W[u] = 0; r.use[u] = 0
                        if getattr(r, 'U', None) is not None:
                            r.U[u] = 0
                elif u < len(r.v):
                    r.v[u] = 0
            self.free.append(u)
        self.deleted += len(dead)
        return len(dead)


def pair_keys(groups, pairs):
    """groups: dict name -> list of feature values; direct keys (name, v) + pair keys for the listed group pairs."""
    keys = [(g, v) for g, vs in groups.items() for v in vs]
    for a, b in pairs:
        for va in groups.get(a, ()):
            for vb in groups.get(b, ()):
                keys.append((a, va, b, vb))
    return keys


# ----------------------------------------------------------------------------------------------- R4
class Hippo:
    """Episodic index.  DG: fixed random fan-in from several input groups, k-WTA (pattern separation).
    Episode cells: recruited one-shot; key = DG code; content = arbitrary record; strength sigma decays slowly."""

    def __init__(self, group_sizes, n_dg=3000, k_dg=60, fan=(12,), weights=None, cap=4000, seed=0,
                 gamma=0.9995, theta=0.25):
        rng = np.random.default_rng(seed)
        self.groups = list(group_sizes)
        self.n_dg, self.k_dg = n_dg, k_dg
        self.idx = {}
        fan = dict(zip(self.groups, fan if len(fan) == len(self.groups) else fan * len(self.groups)))
        for g, n in group_sizes.items():
            self.idx[g] = rng.integers(0, n, size=(n_dg, fan[g]))
        self.wg = weights or {g: 1.0 for g in self.groups}
        self.tie = rng.random(n_dg).astype(np.float32) * 1e-3
        self.keys = np.zeros((cap, k_dg), np.int32)        # each cell's key as the list of its k DG units
        self.sig = np.zeros(cap, np.float32)
        self.content = [None] * cap
        self.born = np.zeros(cap, np.int64)
        self.cap, self.n = cap, 0
        self.gamma, self.theta = gamma, theta
        self.clock = 0

    def dg(self, inputs):
        """inputs: dict group -> bool vector; returns the DG code (bool n_dg)."""
        d = self.tie.copy()
        for g, x in inputs.items():
            if g in self.idx and x is not None and x.any():
                d += self.wg[g] * x[self.idx[g]].sum(1)
        code = np.zeros(self.n_dg, bool)
        # ARCH_v2.2 canonical tie rule (gpuport P0): exact ties at the k-th place are broken by unit index (stable
        # sort), so the DG code no longer depends on the numpy version (argpartition breaks ties differently on 1.26 / 2.x)
        code[np.argsort(-d, kind='stable')[:self.k_dg]] = True
        return code

    def store(self, code, content):
        if self.n < self.cap:
            e = self.n; self.n += 1
        else:
            e = int(np.argmin(self.sig[:self.n]))          # the weakest trace is overwritten
        idx = np.flatnonzero(code)[:self.k_dg]
        self.keys[e, :len(idx)] = idx
        self.sig[e] = 1.0
        self.content[e] = content
        self.born[e] = self.clock
        return e

    def overlaps(self, code):
        if self.n == 0:
            return np.zeros(0, np.float32)
        return code[self.keys[:self.n]].sum(1).astype(np.float32) / self.k_dg

    def recall(self, code, exclude=None):
        ov = self.overlaps(code)
        if len(ov) == 0:
            return -1, 0.0
        sc = ov * np.sqrt(np.maximum(self.sig[:self.n], 0))
        if exclude is not None and exclude >= 0 and exclude < self.n:
            sc[exclude] = -1
        e = int(np.argmax(sc))
        if ov[e] < self.theta or self.sig[e] <= 0.05:
            return -1, float(ov[e])
        return e, float(ov[e])

    def tick(self, n=1):
        self.clock += n
        self.sig[:self.n] *= self.gamma ** n

    def sample(self, k, rng, prio=None):
        if self.n == 0:
            return []
        w = np.maximum(self.sig[:self.n], 0).astype(np.float64)
        if prio is not None:
            w = w * prio[:self.n]
        if w.sum() <= 0:
            return []
        return list(rng.choice(self.n, size=min(k, int((w > 0).sum())), replace=False, p=w / w.sum()))


# ----------------------------------------------------------------------------------------------- R5
class Templates:
    """Z: sound template per program (FEAT0 space).  Detection: normalised match, WTA, threshold, local peak."""

    def __init__(self, n_feat, cap=512, theta=0.8, rate=0.5):
        self.W = np.zeros((n_feat, cap), np.float32)
        self.n = 0
        self.theta, self.rate = theta, rate
        self.count = np.zeros(cap, np.int32)

    def new(self, f):
        j = self.n; self.n += 1
        if j >= self.W.shape[1]:
            W = np.zeros((self.W.shape[0], 2 * self.W.shape[1]), np.float32); W[:, :j] = self.W[:, :j]; self.W = W
            c = np.zeros(2 * len(self.count), np.int32); c[:j] = self.count[:j]; self.count = c
        self.W[:, j] = f
        self.count[j] = 1
        return j

    def learn(self, j, f, rate=None):
        r = self.rate if rate is None else rate
        self.W[:, j] = (1 - r) * self.W[:, j] + r * f
        self.count[j] += 1

    def match(self, f):
        if self.n == 0:
            return np.zeros(0, np.float32)
        W = self.W[:, :self.n]
        num = W[f].sum(0)
        den = np.maximum(W.sum(0), 1e-6)
        # normalised overlap: fraction of the template's mass present, penalised by extra active bits
        return num / den - 0.5 * np.maximum(f.sum() - den, 0) / np.maximum(den, 1e-6) * 0.0

    def best(self, f):
        m = self.match(f)
        if len(m) == 0:
            return -1, 0.0
        j = int(np.argmax(m))
        return j, float(m[j])


# ----------------------------------------------------------------------------------------------- R6
class Critic:
    def __init__(self, cap=200000, eta=0.2):
        self.v = np.zeros(cap, np.float32)
        self.eta = eta

    def grow(self, n):
        if n > len(self.v):
            v = np.zeros(max(n, 2 * len(self.v)), np.float32); v[:len(self.v)] = self.v; self.v = v

    def value(self, A):
        return float(self.v[A].sum()) if len(A) else 0.0

    def update(self, A, r):
        V = self.value(A)
        delta = r - V
        if len(A):
            self.v[A] += self.eta * delta / len(A)
        return delta


def jaccard(a, b):
    a = np.asarray(a, bool); b = np.asarray(b, bool)
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else 1.0
