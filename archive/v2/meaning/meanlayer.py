# -*- coding: utf-8 -*-
"""The learned meaning layer (Step 4; design.md) and the random layer that reads it.

MeanCodes: one sparse meaning code per item (Z category), learned online from heard lines.
  Meaning lines: D binary units.  Fixed at birth: every item j (and BOS / EOS) projects to kx random lines as a LEFT
  neighbour (idxL[j]) and to kx random lines as a RIGHT neighbour (idxR[j]).
  mode 'ri'  (A, chosen): at a heard line's end, for every position p with item i:
               x = idxL[i_{p-1} or BOS] u idxR[i_{p+1} or EOS]
               M[i] <- M[i] + eta_i (x - M[i]),  eta_i = max(1 / n_i, eta_min)          (instar, maturation)
               b    <- b + (x - b) / tau_b                                             (each line's own base rate)
             code c_i = the km lines with the largest M[i,j] / (b_j + eps) among M[i,j] >= m_min; incumbent lines x (1+h)
  mode 'tc'  (C): as ri, but x = alpha * idx(neighbours, both sides merged) + (1 - alpha) * code(neighbours)  (bootstrapped)
  mode 'cl'  (B): competitive k-WTA: units u with prototypes C[u] over the context space; winners = top-kcl of
             C x - theta (theta_u rises with u's own win rate); winners: C[u] <- C[u] + eta_c (x - C[u]);
             W[i] <- W[i] + eta_i (win - W[i]); code = top-km of W[i]
  mode 'hip' (D): codes from awake recall: a replayed episode e and its NEXT line n: items of e learn x = idxN[items of n],
             items of n learn x = idxP[items of e] (same instar, base rate and code rule as ri)
  mode 'oracle': the meaning half is given: one shared random code per class (cls: item -> class), else a unique one
  mode 'rand2':  a fixed random unique meaning code per item (no learning): controls for the code doubling
A code exists once n_i >= n_min (oracle / rand2: always).  Codes are recomputed every U observed lines; `version`
increments whenever any code changed (the random layer then clears its lookup cache).

MeanMix(RandMix): item bundles have ns (sound, 512) + D (meaning) lines; an item enters as its fixed sound code (16 of
ns: randmix's acoustic fly-hash, or a random code without templates) plus its meaning code (km of D, offset ns).
mode None / 'none': bundle = ns lines and the sound code only (== randmix's RandMix with the same arguments).
Lesions (test time): lesion='off' (meaning half removed), 'shuffle' (meaning codes permuted between items).
"""
from __future__ import annotations

import numpy as np

from randlayer import RandMix

BOS_I, EOS_I = -1, -2
ITEM_ALL = {'P', 'P1', 'PL', 'L', 'CP', 'CN', 'CC', 'VR'}


class MeanCodes:
    def __init__(self, V=400, D=512, kx=6, km=16, eta_min=0.002, tau_b=20000.0, h=0.2, U=20, m_min=0.02, n_min=3,
                 mode='ri', alpha=0.5, kcl=16, eta_c=0.02, tau_cl=500.0, cls=None, eps=1e-3, bfloor=-1.0,
                 init='rand', max_swap=2, kcls=6, seed=0):
        self.V, self.D, self.kx, self.km = V, D, kx, km
        self.eta_min, self.tau_b, self.h, self.U, self.m_min, self.n_min = eta_min, tau_b, h, U, m_min, n_min
        self.mode, self.alpha, self.eps = mode, alpha, eps
        self.bfloor = 2.0 * kx / D if bfloor < 0 else bfloor      # prior rate of a context line (no PMI spikes)
        self.init, self.max_swap, self.kcls = init, max_swap, kcls
        self.lesion, self.lmap = None, None
        r = np.random.default_rng(seed + 777)
        nidx = V + 2                                  # items + BOS (V) + EOS (V+1)
        self.idxL = np.stack([r.choice(D, kx, replace=False) for _ in range(nidx)])
        self.idxR = np.stack([r.choice(D, kx, replace=False) for _ in range(nidx)])
        self.M = np.zeros((V, D), np.float32)
        self.n = np.zeros(V, np.int64)
        self.b = np.full(D, 2.0 * kx / D, np.float32)
        self.code = {}                                # item -> sorted int64 array of meaning lines
        self.dirty = set()
        self.version = 0
        self.lines = 0
        self.stats = dict(lines=0, occ=0, recomputes=0, changes=0, lines_changed=0, placeholders=0)
        if mode == 'cl':
            self.kcl, self.eta_c, self.tau_cl = kcl, eta_c, tau_cl
            self.C = (r.random((D, D)) * 0.1).astype(np.float32)        # unit prototypes over the context space
            self.C /= self.C.sum(1, keepdims=True)
            self.theta = np.zeros(D, np.float32)
            self.use = np.full(D, kcl / D, np.float32)
        self.cls = None
        if mode in ('oracle', 'oracle2'):
            self.set_classes(cls or {}, kcls if mode == 'oracle2' else km)
        if mode == 'rand2':
            for i in range(V):
                self.code[i] = self._placeholder(i)

    def _placeholder(self, i):
        return np.sort(np.random.default_rng(70001 + 11 * int(i)).choice(self.D, self.km, replace=False)).astype(np.int64)

    def set_classes(self, cls, kcls):
        """oracle codes: items of class c share kcls class lines; with kcls < km the other km - kcls lines are shared
        by every classed item (as the learned codes share the frame contexts of all fillers); unclassed items keep a
        unique random code."""
        self.cls = dict(cls)
        shared = np.random.default_rng(80000).choice(self.D, self.km, replace=False)
        pat = {}
        for i in range(self.V):
            c = self.cls.get(i)
            if c is None:
                new = self._placeholder(i)
            else:
                if c not in pat:
                    rr = np.random.default_rng(80001 + 17 * len(pat))
                    cand = np.setdiff1d(np.arange(self.D), shared[:self.km - kcls])
                    pat[c] = rr.choice(cand, kcls, replace=False)
                new = np.sort(np.concatenate([pat[c], shared[:self.km - kcls]])).astype(np.int64)
            old = self.code.get(i)
            if old is None or len(old) != len(new) or np.any(old != new):
                self.code[i] = new
                self.version += 1

    # ------------------------------------------------------------------ context drives
    def _ix(self, j):
        return self.V if j == BOS_I else (self.V + 1 if j == EOS_I else j)

    def _ctx(self, left, right):
        """0/1 (ri, cl, hip) or graded (tc) context drive over the D meaning lines"""
        x = np.zeros(self.D, np.float32)
        if self.mode == 'tc':
            a = self.alpha
            x[self.idxL[self._ix(left)]] += a
            x[self.idxR[self._ix(right)]] += a
            for j in (left, right):
                c = self.code.get(j) if j >= 0 else None
                if c is not None:
                    x[c] += (1.0 - a)
            return np.minimum(x, 1.0)
        x[self.idxL[self._ix(left)]] = 1.0
        x[self.idxR[self._ix(right)]] = 1.0
        return x

    def _learn(self, i, x):
        self.n[i] += 1
        eta = max(1.0 / self.n[i], self.eta_min)
        if self.mode == 'cl':
            d = self.C @ x - self.theta
            win = np.argpartition(-d, self.kcl)[:self.kcl]
            w = np.zeros(self.D, np.float32); w[win] = 1.0
            self.C[win] += self.eta_c * (x[None, :] / max(x.sum(), 1.0) - self.C[win])
            self.use += (w - self.use) / self.tau_cl
            self.theta = (self.use / (self.kcl / self.D) - 1.0) * 0.05 * float(d.max() if d.max() > 0 else 1.0)
            self.M[i] += eta * (w - self.M[i])
        else:
            self.M[i] += eta * (x - self.M[i])
            self.b += (x - self.b) / self.tau_b
        self.dirty.add(i)
        self.stats['occ'] += 1

    # ------------------------------------------------------------------ observations
    def observe_line(self, items):
        """a heard line (item ids in order) ended: learn every item's code from its neighbours"""
        if self.mode not in ('ri', 'tc', 'cl'):
            return
        it = [i for i in items if 0 <= i < self.V]
        if not it:
            return
        seq = [BOS_I] + it + [EOS_I]
        for p in range(1, len(seq) - 1):
            self._learn(seq[p], self._ctx(seq[p - 1], seq[p + 1]))
        self._tick()

    def observe_pair(self, items_e, items_n):
        """awake recall (mode hip): a replayed line and its NEXT line teach each other's items"""
        if self.mode != 'hip':
            return
        a = [i for i in items_e if 0 <= i < self.V]
        b = [i for i in items_n if 0 <= i < self.V]
        if not a or not b:
            return
        xn = np.zeros(self.D, np.float32); xn[self.idxR[b].ravel()] = 1.0    # 'what follows my line'
        xp = np.zeros(self.D, np.float32); xp[self.idxL[a].ravel()] = 1.0    # 'what my line follows'
        for i in a:
            self._learn(i, xn)
        for i in b:
            self._learn(i, xp)
        self._tick()

    def _tick(self):
        self.lines += 1
        self.stats['lines'] += 1
        if self.lines % self.U == 0:
            self.recompute()

    def recompute(self):
        """codes follow M / max(b, bfloor) (positive-PMI): the km best lines with M >= m_min.  max_swap > 0: at most
        max_swap incumbent lines are replaced per recompute, each only by a challenger scoring > (1 + h) x the weakest
        incumbent (gradual, hysteretic code change; an item starts from a random placeholder code with init='rand')."""
        if self.mode in ('oracle', 'oracle2', 'rand2'):
            self.dirty.clear()
            return False
        changed = False
        self.stats['recomputes'] += 1
        for i in sorted(self.dirty):
            old = self.code.get(i)
            if old is None and self.init == 'rand':
                old = self._placeholder(i)
                self.code[i] = old
                self.stats['placeholders'] += 1
                changed = True
            if self.n[i] < self.n_min:
                continue
            m = self.M[i]
            s = m / np.maximum(self.b, self.bfloor) if self.mode != 'cl' else m.copy()
            s = np.where(m >= self.m_min, s, 0.0)
            if old is None or self.max_swap <= 0:
                if old is not None and self.h > 0:
                    s = s.copy(); s[old] *= (1.0 + self.h)
                pos = np.flatnonzero(s > 0)
                if len(pos) == 0:
                    continue
                k = min(self.km, len(pos))
                new = np.sort(pos[np.argsort(-s[pos], kind='stable')[:k]]).astype(np.int64)
            else:
                cur = set(old.tolist())
                for _ in range(self.max_swap):
                    inc = sorted(cur, key=lambda j: (s[j], j))
                    w = inc[0]
                    ch = np.flatnonzero(s > 0)
                    ch = [j for j in ch[np.argsort(-s[ch], kind='stable')] if j not in cur][:1]
                    if not ch or s[ch[0]] <= (1.0 + self.h) * s[w]:
                        break
                    cur.discard(w); cur.add(int(ch[0]))
                new = np.array(sorted(cur), np.int64)
            if old is None or len(old) != len(new) or np.any(old != new):
                self.stats['changes'] += 1
                self.stats['lines_changed'] += len(new) if old is None else len(set(new.tolist()) - set(old.tolist()))
                self.code[i] = new
                changed = True
        self.dirty.clear()
        if changed:
            self.version += 1
        return changed

    def get(self, i):
        if self.lesion == 'off':
            return None
        if self.lmap is not None:
            i = self.lmap.get(i, i)
        return self.code.get(i)

    def set_lesion(self, mode, pairs=None):
        """test-time lesions: 'off' (no meaning half), 'shuffle' (codes permuted between items), 'swap' (codes of the
        given item pairs exchanged), None (intact).  Bumps the version so the layer's cache is cleared."""
        self.lesion, self.lmap = mode, None
        if mode == 'shuffle':
            ks = sorted(self.code)
            p = np.random.default_rng(5).permutation(len(ks))
            self.lmap = {ks[a]: ks[p[a]] for a in range(len(ks))}
        elif mode == 'swap':
            self.lmap = {}
            for a, b in pairs or []:
                self.lmap[a], self.lmap[b] = b, a
        self.version += 1

    # ------------------------------------------------------------------ diagnostics
    def snapshot(self):
        return {int(i): c.copy() for i, c in self.code.items()}


def code_stats(codes, groups, km=16):
    """codes: item -> lines; groups: dict name -> list of items.  Within / between Jaccard and NN class purity."""
    def jac(a, b):
        a, b = set(a.tolist()), set(b.tolist())
        return len(a & b) / max(1, len(a | b))
    items = [i for g in groups.values() for i in g if i in codes]
    lab = {i: g for g, its in groups.items() for i in its}
    win, btw, pur = [], [], []
    for x in items:
        best, bl = -1, None
        for y in items:
            if y == x:
                continue
            j = jac(codes[x], codes[y])
            (win if lab[x] == lab[y] else btw).append(j)
            if j > best:
                best, bl = j, lab[y]
        pur.append(bl == lab[x])
    allc = list(codes.values())
    allj = [jac(allc[a], allc[b]) for a in range(len(allc)) for b in range(a + 1, len(allc))]
    return dict(within=round(float(np.mean(win)), 3) if win else None, between=round(float(np.mean(btw)), 3) if btw else None,
                purity=round(float(np.mean(pur)), 3) if pur else None, all_pairs=round(float(np.mean(allj)), 3) if allj else None,
                n=len(items), n_codes=len(codes))


def code_drift(a, b):
    """fraction of meaning lines changed between two snapshots (items present in both)"""
    ch, tot = 0, 0
    for i, c in a.items():
        d = b.get(i)
        if d is None:
            continue
        ch += len(set(c.tolist()) - set(d.tolist()))
        tot += len(c)
    return round(ch / max(1, tot), 4)


class MeanMix(RandMix):
    """RandMix whose item bundles carry the sound code + the learned meaning code."""

    def __init__(self, mc=None, ns=512, mgroups='P,P1,PL,L,CP,CN,CC,VR', **kw):
        self.mc = mc
        self.ns = ns
        self.mgroups = set(mgroups.split(',')) if isinstance(mgroups, str) else set(mgroups)
        self.lesion = None
        self._perm = None
        D = mc.D if mc is not None else 0
        kw = dict(kw)
        kw['n_c'] = ns + D
        super().__init__(**kw)
        self._ver = mc.version if mc is not None else 0
        self.mstats = dict(clears=0)

    def _sound(self, v):
        c = self.codes.get(v)
        if c is not None:
            return c
        k, n = self.k_c, self.ns
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

    def _meaning(self, v):
        if self.mc is None or self.lesion == 'off' or v < 0:
            return None
        return self.mc.get(v)

    def _item_code(self, v):
        s = self._sound(v)
        m = self._meaning(int(v))
        if m is None:
            return s
        return np.concatenate([s, self.ns + m])

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
                X.append(self.boff[g] + (self._item_code(int(v)) if g in getattr(self, 'mgroups', ITEM_ALL) else self._sound(int(v))))
        if not X:
            return np.zeros(0, np.int64)
        return np.unique(np.concatenate(X))

    def set_lesion(self, mode):
        self.lesion = mode
        self._perm = None
        self.cache = {}

    def units(self, keys, create=True, tick=True):
        if self.mc is not None and self.mc.version != self._ver:
            self._ver = self.mc.version
            self.cache = {}
            self.mstats['clears'] += 1
        return super().units(keys, create=create, tick=tick)


def make_mix(mc, rm, seed):
    rm = dict(rm)
    use_torch, mf = rm.pop('torch', 0), rm.pop('memfrac', 0.0)
    if use_torch:
        from meantorch import TorchMeanMix
        return TorchMeanMix(mc=mc, seed=seed, device='cuda', mem_frac=float(mf), **rm)
    return MeanMix(mc=mc, seed=seed, **rm)
