# -*- coding: utf-8 -*-
"""RandMix: PFC-2 as a RANDOM layer generated at birth (the designer's idea, 2026-10-31), replacing MixDict (R3).

Drop-in for rules.MixDict: units(keys, create, tick) -> sorted ids of the active units; n = number of units (fixed
from birth, so GEN / gates / critic get all their rows at once); grp; key_of; prune (no-op for the old API).

INPUT LINES (one bundle per source group; the groups are exactly the groups MIX keys use today)
  item groups P (held thought items, a bag), P1, PL (first / last held item), L (last item of the current line),
      CP, CN, CC (the copy routes' candidates), VR (items recognised at the attended slot):
      one bundle of n_c lines per group; an item enters as a sparse code of k_c lines (the same code in every bundle).
      code='acou': the code is a fly-hash of the item's perceptual template (the Z category's FEAT0 template
      binarised at 0.5, a fixed random projection with r_f = 24 features per line, top-k_c lines): items that sound
      alike get overlapping codes.  code='rand': an arbitrary random code.  BOS / NONE get random codes.
      An item's code is fixed the first time it enters (its category is formed by then).
  small groups N (position 0-6), FAM (0-3), A (slot 0-2 / none), M (0-2), NOP: k_s dedicated lines per value.
  F: the 240 slot-relative eye lines of the attended block (as they are).
UNITS
  One population per MIX key type (the 14 direct groups and the 40 pairs of the pair schema, innate #18).
  Population size N_t = max(min_units, round(scale * c_t)), c_t = MIX's number of keys of that type at the end of a
  staged run (ARCH_v2 seed 0, 47.9k in total), so scale 1 = as many units as MIX recruits, 4 = four times
  (alloc='flat': the same total split equally between the populations: a birth that knows nothing of later use).
  A pair unit has fan lines from each of its two bundles (a direct unit: fan lines from its one bundle), drawn at
  random at birth; weights 1.
ACTIVITY (dendritic AND + k-WTA at a fixed active fraction)
  s_b = sum_j w_j x_j over branch b; a unit is eligible iff every branch has s_b >= theta (1);
  among the eligible units of each population the k_t = max(1, round(alpha * N_t)) with the largest s_a + s_b
  (+ a fixed tiny tie-break) fire.  Populations whose groups are absent are silent (as MIX has no key then).
LEARNING OF THE FAN-IN (variants; all local: synapse = its pre line, its post unit, its own weight / age)
  train=0  (a) fixed.
  train=1  (b) instar with normalisation: at every learning event (when the readout learns), for an active unit u
           and each synapse j of u: w_j <- w_j + eta (x_j - w_j); then each branch is rescaled to sum = fan
           (heterosynaptic competition: the lines that fire with the unit grow, the others shrink).  Applied in
           batches of `every` learning events (exact for constant x; one batch = ~3 dialogues).
  prune=1  (c) use it or lose it: a synapse whose line has not been active while its unit fired for `age` learning
           events is removed; regrow=1: a new random line of the same bundle replaces it (total fixed);
           regrow=0: it is gone (total declining; birth = overproduction factor of units).
"""
from __future__ import annotations

import numpy as np

ITEM_G = ('P', 'P1', 'PL', 'L', 'CP', 'CN', 'CC', 'VR')
SMALL = {'N': [0, 1, 2, 3, 4, 5, 6], 'FAM': [0, 1, 2, 3], 'A': [0, 1, 2, -2], 'M': [0, 1, 2], 'NOP': [1]}
DIRECT = ('P', 'L', 'N', 'FAM', 'P1', 'PL', 'NOP', 'CP', 'CN', 'CC', 'A', 'F', 'VR', 'M')
PAIRS = [('P', 'M'), ('L', 'M'), ('P', 'L'), ('P', 'N'), ('L', 'N'), ('P1', 'N'), ('PL', 'N'),
         ('P', 'CP'), ('L', 'CP'), ('N', 'CP'), ('P', 'CN'), ('L', 'CN'), ('FAM', 'CN'),
         ('P', 'CC'), ('L', 'CC'), ('FAM', 'CC'), ('P', 'FAM'), ('N', 'FAM'),
         ('P', 'A'), ('L', 'A'), ('N', 'A'), ('P', 'F'), ('L', 'F'), ('P', 'VR'), ('L', 'VR'), ('N', 'VR'),
         ('NOP', 'VR'), ('NOP', 'F'), ('NOP', 'A'), ('NOP', 'L'), ('NOP', 'CC')]
# MIX keys per type at the end of a staged run (ARCH_v2 seed 0, out/mix_keys_final_s0.json); others -> min_units
MIX_COUNT = {'P|F': 11427, 'L|F': 10681, 'P|L': 5159, 'P|CN': 4295, 'P|CC': 4248, 'P|VR': 1974, 'L|VR': 1945,
             'P|CP': 1711, 'P|N': 845, 'L|CC': 436, 'P|FAM': 420, 'P|A': 365, 'PL|N': 359, 'L|CN': 356, 'L|CP': 351,
             'L|A': 351, 'L|M': 278, 'P1|N': 266, 'L|N': 260, 'P|M': 247, 'N|CP': 241, 'FAM|CN': 240, 'FAM|CC': 234,
             'N|VR': 219, 'F': 196, 'L': 139, 'P': 138, 'CP': 116, 'CC': 116, 'CN': 104, 'PL': 58, 'P1': 38,
             'VR': 32, 'N|FAM': 28, 'N|A': 28, 'N': 7, 'FAM': 4, 'A': 4, 'M': 3}


class RandMix:
    def __init__(self, scale=1.0, fan=3, alpha=0.01, n_c=512, k_c=16, k_s=16, n_f=240, code='acou', r_f=24,
                 train=0, eta=0.02, every=200, prune=0, age=20000, regrow=1, birth=1.0, min_units=32, theta=1.0,
                 cache_cap=20000, alloc='mix', seed=0, stab=0.0, stab_min=0.05):
        rng = np.random.default_rng(seed + 4242)
        self.rng = np.random.default_rng(seed + 4243)
        self.fan, self.alpha, self.n_c, self.k_c, self.k_s, self.code = fan, alpha, n_c, k_c, k_s, code
        self.r_f, self.train, self.eta, self.every = r_f, train, eta, every
        self.prune_on, self.age, self.regrow, self.theta = prune, age, regrow, theta
        self.cache_cap = cache_cap
        self.stab, self.stab_min = float(stab), float(stab_min)   # x2: instar stability (0 = off, v2.2 bit for bit)
        self.feat_src = None              # set by the agent: the Z templates (perceptual codes of the items)
        # ---- input bundles
        self.boff, self.bsize = {}, {}
        off = 0
        for g in ITEM_G:
            self.boff[g], self.bsize[g] = off, n_c; off += n_c
        for g, vals in SMALL.items():
            self.boff[g], self.bsize[g] = off, len(vals) * k_s; off += len(vals) * k_s
        self.boff['F'], self.bsize['F'] = off, n_f; off += n_f
        self.D = off
        self.small_idx = {g: {v: i for i, v in enumerate(vals)} for g, vals in SMALL.items()}
        self.codes = {}
        self.proj = None
        self.code_rng = np.random.default_rng(seed + 4244)
        # ---- populations
        types = [(g,) for g in DIRECT] + [tuple(p) for p in PAIRS]
        self.types = types
        sizes = []
        tot = sum(MIX_COUNT.values())
        cnt = np.array([MIX_COUNT.get('|'.join(t), 0) for t in types], np.float64)
        if alloc.startswith('pow:'):          # v2.2: N_t proportional to c_t ** p, same total (p 0 = flat, 1 = MIX's)
            pw = float(alloc[4:])
            wt = np.where(cnt > 0, cnt ** pw, 0.0)
            share = tot * wt / wt.sum()
        for ti, t in enumerate(types):
            if alloc == 'mix':
                c = MIX_COUNT.get('|'.join(t), 0)
            elif alloc.startswith('pow:'):
                c = share[ti]
            else:
                c = tot / len(types)                                                      # 'flat': equal per population
            sizes.append(max(min_units, int(round(scale * birth * c))))
        self.Nt = np.array(sizes, np.int64)
        self.N = int(self.Nt.sum())
        self.toff = np.concatenate([[0], np.cumsum(self.Nt)[:-1]]).astype(np.int64)
        self.kt = np.maximum(1, np.round(alpha * self.Nt / birth)).astype(np.int64)
        self.type_of = np.repeat(np.arange(len(types)), self.Nt).astype(np.int32)
        self.direct = np.repeat(np.array([len(t) == 1 for t in types]), self.Nt)
        f = fan
        self.lines = np.zeros((self.N, 2, f), np.int32)
        for ti, t in enumerate(types):
            o, n = self.toff[ti], self.Nt[ti]
            for b, g in enumerate(t):
                self.lines[o:o + n, b, :] = self.boff[g] + self._draw(rng, n, f, self.bsize[g])
            if len(t) == 1:
                self.lines[o:o + n, 1, :] = -1
        self.w = np.where(self.lines >= 0, 1.0, 0.0).astype(np.float32)
        self.last_used = np.zeros((self.N, 2, f), np.int64)          # learning-event clock of the last co-activity
        self.su = np.zeros((self.N, 2, f), np.float32) if self.stab > 0 else None   # x2: per-synapse use (post fired)
        self.tie = (rng.random(self.N) * 1e-3).astype(np.float32)
        self.alive_unit = np.ones(self.N, bool)
        self.unit_w = not train                                  # all live weights are 1 (fast path) until trained
        self._build_csr()
        # ---- MixDict interface
        self.n = self.N
        self.grp = np.zeros(self.N, np.int32)
        self.key_of = {}
        self.ids = {}
        self.clock = 0
        self.free, self.deleted = [], 0
        self.cap = self.N
        # ---- bookkeeping
        self.cache = {}
        self.ev = 0                       # learning events seen by the layer
        self.batch = []                   # (A, X) of learning events not yet applied (train / prune)
        self.ever = np.zeros(self.N, bool)   # units that have ever fired at a learning event
        self.fires = np.zeros(self.N, np.int64)
        self.last_A, self.last_X = None, None
        self.stats = dict(calls=0, hits=0, computed=0, active_sum=0, hebb_batches=0, pruned=0, regrown=0,
                          dead_units=0)
        self.history = []

    # ------------------------------------------------------------------ construction helpers
    @staticmethod
    def _draw(rng, n, f, size):
        """n rows of f distinct line indices in [0, size) (redraw rows with a repeated line)"""
        f = min(f, size)
        out = rng.integers(0, size, size=(n, f))
        while True:
            srt = np.sort(out, 1)
            bad = np.flatnonzero((srt[:, 1:] == srt[:, :-1]).any(1)) if f > 1 else np.zeros(0, np.int64)
            if len(bad) == 0:
                return out
            out[bad] = rng.integers(0, size, size=(len(bad), f))

    def _build_csr(self):
        L = self.lines.ravel()
        live = np.flatnonzero(L >= 0)
        order = live[np.argsort(L[live], kind='stable')]
        self.csr_syn = order.astype(np.int64)
        self.csr_ptr = np.searchsorted(L[order], np.arange(self.D + 1)).astype(np.int64)

    # ------------------------------------------------------------------ item codes
    def _item_code(self, v):
        c = self.codes.get(v)
        if c is not None:
            return c
        k, n = self.k_c, self.n_c
        if self.code == 'acou' and v >= 0 and self.feat_src is not None and v < self.feat_src.n:
            t = self.feat_src.W[:, v] >= 0.5
            if self.proj is None:
                self.proj = self.code_rng.integers(0, len(t), size=(n, self.r_f))
                self.jit = self.code_rng.random(n).astype(np.float32) * 1e-3
            s = t[self.proj].sum(1).astype(np.float32) + self.jit
            c = np.sort(np.argpartition(-s, k)[:k]).astype(np.int64)
        else:
            r = np.random.default_rng(1000003 + int(v) * 7919)
            c = np.sort(r.choice(n, k, replace=False)).astype(np.int64)
        self.codes[v] = c
        return c

    def lines_of(self, direct_keys):
        X = []
        for g, v in direct_keys:
            if g in self.small_idx:
                i = self.small_idx[g].get(v)
                if i is None:
                    continue
                o = self.boff[g] + i * self.k_s
                X.append(np.arange(o, o + self.k_s))
            elif g == 'F':
                b = v[1] if isinstance(v, tuple) else v
                X.append(np.array([self.boff['F'] + int(b)]))
            elif g in self.boff:
                X.append(self.boff[g] + self._item_code(int(v)))
        if not X:
            return np.zeros(0, np.int64)
        return np.unique(np.concatenate(X))

    # ------------------------------------------------------------------ activity
    def activate(self, X):
        if len(X) == 0:
            return np.zeros(0, np.int64)
        st, en = self.csr_ptr[X], self.csr_ptr[X + 1]
        cnt = en - st
        tot = int(cnt.sum())
        if tot == 0:
            return np.zeros(0, np.int64)
        cs = np.cumsum(cnt)
        idx = np.arange(tot) - np.repeat(cs - cnt, cnt) + np.repeat(st, cnt)
        syn = self.csr_syn[idx]
        f = self.fan
        if self.unit_w:                                          # untrained weights are all 1: plain counts
            s = np.bincount(syn // f, minlength=2 * self.N).reshape(self.N, 2)
        else:
            s = np.bincount(syn // f, weights=self.w.ravel()[syn], minlength=2 * self.N).reshape(self.N, 2)
        el = (s[:, 0] >= self.theta) & ((s[:, 1] >= self.theta) | self.direct)
        E = np.flatnonzero(el)
        if len(E) == 0:
            return E.astype(np.int64)
        tE = self.type_of[E]
        key = tE * 1000.0 - (s[E, 0] + s[E, 1]) - self.tie[E]    # by population, then by drive (descending)
        o = np.argsort(key, kind='stable')
        ts = tE[o]
        first = np.searchsorted(ts, ts, side='left')
        rank = np.arange(len(ts)) - first
        keep = rank < self.kt[ts]
        return np.sort(E[o[keep]]).astype(np.int64)

    def units(self, keys, create=True, tick=True):
        if tick:
            self.clock += 1
        dk = tuple(k for k in keys if len(k) == 2)
        self.stats['calls'] += 1
        hit = self.cache.get(dk)
        if hit is not None:
            self.stats['hits'] += 1
            A, X = hit
        else:
            X = self.lines_of(dk)
            A = self.activate(X)
            self.stats['computed'] += 1
            if len(self.cache) >= self.cache_cap:
                self.cache = {}
            self.cache[dk] = (A.astype(np.int32), X.astype(np.int16))
        A = A.astype(np.int64)
        X = X.astype(np.int64)
        self.last_A, self.last_X = A, X
        self.stats['active_sum'] += len(A)
        return A

    # ------------------------------------------------------------------ learning events (called by the core)
    def learn_event(self):
        """the readout just learned at this context: the layer's own plasticity sees (A, X) of that event."""
        A, X = self.last_A, self.last_X
        if A is None:
            return
        self.ev += 1
        if len(A):
            self.ever[A] = True
            self.fires[A] += 1
        if self.train or self.prune_on:
            self.batch.append((A, X))
            if len(self.batch) >= self.every:
                self._apply()

    def _apply(self):
        batch, self.batch = self.batch, []
        if not batch:
            return
        f = self.fan
        rows, xs = [], []
        for A, X in batch:
            if len(A) == 0:
                continue
            ind = np.zeros(self.D + 1, bool)
            ind[X] = True
            Lr = self.lines[A]                                   # (|A|, 2, f), -1 = no synapse
            x = ind[np.where(Lr >= 0, Lr, self.D)]               # line active at this event
            rows.append(A); xs.append(x)
        if not rows:
            return
        R = np.concatenate(rows)
        Xa = np.concatenate(xs).astype(np.float32)               # (rows, 2, f)
        U, inv, n_u = np.unique(R, return_inverse=True, return_counts=True)
        sx = np.zeros((len(U), 2, f), np.float32)
        np.add.at(sx, inv, Xa)
        used = sx > 0
        if self.prune_on:
            lu = self.last_used[U]
            lu[used] = self.ev
            self.last_used[U] = lu
        if self.train:
            xbar = sx / n_u[:, None, None]
            w = self.w[U]
            if self.su is None:
                keep = (1.0 - self.eta) ** n_u[:, None, None]
            else:                                                # x2: eta_eff = eta * max(1/(1+b u |w|), f_min)
                su = self.su[U]
                fs = np.maximum(1.0 / (1.0 + self.stab * su * np.abs(w)), self.stab_min)
                keep = (1.0 - self.eta * fs) ** n_u[:, None, None]
                self.su[U] = su + n_u[:, None, None]
            live = self.lines[U] >= 0
            w = np.where(live, xbar + keep * (w - xbar), 0.0).astype(np.float32)
            sm = w.sum(2, keepdims=True)
            nl = live.sum(2, keepdims=True)
            w = np.where(sm > 0, w * nl / np.maximum(sm, 1e-9), np.where(live, 1.0, 0.0)).astype(np.float32)
            self.w[U] = w
            self.stats['hebb_batches'] += 1
        if self.prune_on:
            self._prune_syn()
        self.cache = {}

    def _prune_syn(self):
        old = (self.lines >= 0) & ((self.ev - self.last_used) > self.age)
        k = int(old.sum())
        if k == 0:
            return
        u, b, i = np.nonzero(old)
        self.stats['pruned'] += k
        if self.regrow:
            for ti in np.unique(self.type_of[u]):
                sel = self.type_of[u] == ti
                t = self.types[ti]
                for br in (0, 1):
                    m = sel & (b == br)
                    if not m.any() or br >= len(t):
                        continue
                    g = t[br]
                    self.lines[u[m], br, i[m]] = self.boff[g] + self.rng.integers(0, self.bsize[g], int(m.sum()))
            self.w[u, b, i] = 1.0
            self.last_used[u, b, i] = self.ev
            if self.su is not None:
                self.su[u, b, i] = 0.0
            self.stats['regrown'] += k
            self.unit_w = self.unit_w and not self.train
        else:
            self.lines[u, b, i] = -1
            self.w[u, b, i] = 0.0
            dead = ~((self.lines[:, 0, :] >= 0).any(1) & ((self.lines[:, 1, :] >= 0).any(1) | self.direct))
            self.stats['dead_units'] = int(dead.sum())
            self.alive_unit = ~dead
        self._build_csr()

    # ------------------------------------------------------------------ copies: the lookup cache is not state
    def __getstate__(self):
        d = dict(self.__dict__)
        d['cache'] = {}
        return d

    # ------------------------------------------------------------------ reporting
    def synapses(self):
        return int((self.lines >= 0).sum())

    def snapshot(self, tag=''):
        s = dict(tag=tag, units=int(self.N), alive_units=int(self.alive_unit.sum()), synapses=self.synapses(),
                 ever_active=int(self.ever.sum()), learn_events=int(self.ev),
                 mean_active=round(self.stats['active_sum'] / max(self.stats['calls'], 1), 1),
                 cache_hit=round(self.stats['hits'] / max(self.stats['calls'], 1), 3),
                 pruned=int(self.stats['pruned']), regrown=int(self.stats['regrown']), items_coded=len(self.codes))
        self.history.append(s)
        return s

    def prune(self, older_than, readouts=()):
        return 0
