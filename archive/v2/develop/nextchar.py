# -*- coding: utf-8 -*-
"""Next-char prediction on the adult corpus with the ARCH_v2-develop readout rule (R2: signed error-driven, NLMS step,
prediction = clip(summed drive)), over recruited context units.  Same split as coord/ngram.py (rng 1 shuffle, last
1,000 dialogues test, first 5,000 train; lines joined by 。).  Context variants:
  L      unit per previous char                                     (bigram)
  L+L2   + unit per char two back + recruited pair unit (L2, L)     (trigram, conjunctive)
  L+R    + recurrent top state R(t) = kWTA(W_in c(t) + W_rec R(t-1)), N 2000, k 40, fixed random (the designer's
         "top layer fed its own previous state"; G(t-1) -> B(t) learned)
  R      recurrent state alone
Outputs: the 1,500 most frequent training chars; pair units only for pairs seen >= 2 times (rarer targets count as misses).
usage: python3 nextchar.py [ntrain]"""
import json, sys, time
import numpy as np

ntr = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
lines = [json.loads(l) for l in open("/root/work/corpus/dialogs.jsonl", encoding="utf-8")]
rs = np.random.default_rng(1); rs.shuffle(lines)
text = lambda d: "。".join(d) + "。"
test = [text(d) for d in lines[-1000:]]
train = [text(d) for d in lines[:ntr]]
from collections import Counter
cnt = Counter(ch for s in train for ch in s)
outs = [c for c, _ in cnt.most_common(1500)]
pairs = Counter(s[j - 2:j] for s in train for j in range(2, len(s)))
oid = {c: i for i, c in enumerate(outs)}
allc = sorted(set("".join(train + test)))
cid = {c: i for i, c in enumerate(allc)}
NC, NO = len(allc), len(outs)

N, K = 2000, 40
r = np.random.default_rng(7)
W_in = (r.random((NC, N)) < 0.02).astype(np.float32) * r.random((NC, N)).astype(np.float32)
W_rec = (r.random((N, N)) < 0.02).astype(np.float32) * r.random((N, N)).astype(np.float32)


def rstep(prev, c):
    d = W_in[cid[c]] + 0.6 * W_rec[prev].sum(0) + r.random(N).astype(np.float32) * 1e-3
    return np.argpartition(-d, K)[:K]


MATURE = len(sys.argv) > 2 and sys.argv[2] == "mature"
MODES = sys.argv[3].split(",") if len(sys.argv) > 3 else ["L", "L+L2", "L+R", "R"]


class Readout:
    def __init__(self, eta=0.3):
        self.rows = {}; self.eta = eta
        self.W = np.zeros((1024, NO), np.float32); self.n = 0
        self.use = np.zeros(1024, np.float32)

    def ids(self, keys, create):
        out = []
        for k in keys:
            u = self.rows.get(k)
            if u is None and create:
                if self.n == len(self.W):
                    self.W = np.concatenate([self.W, np.zeros((len(self.W) // 2, NO), np.float32)])
                    self.use = np.concatenate([self.use, np.zeros(len(self.W) - len(self.use), np.float32)])
                u = self.rows[k] = self.n; self.n += 1
            if u is not None:
                out.append(u)
        return np.array(out, np.int64)

    def drive(self, A):
        return self.W[A].sum(0) if len(A) else np.zeros(NO, np.float32)

    def learn(self, A, tgt):
        d = self.drive(A); p = np.clip(d, 0, 1)
        t = np.zeros(NO, np.float32)
        if tgt is not None:
            t[tgt] = 1
        err = t - p
        J = np.nonzero(np.abs(err) > 1e-4)[0]
        if len(J):
            if MATURE:                                   # plasticity falls with use: 1/n, floor 0.02 (maturation)
                self.use[A] += 1
                m = np.maximum(1.0 / self.use[A], 0.02)[:, None]
                self.W[np.ix_(A, J)] += (m / len(A)) * err[J][None, :]
            else:
                self.W[np.ix_(A, J)] += (self.eta / len(A)) * err[J][None, :]


def keys(s, j, R, mode):
    k = [("L", s[j - 1])]
    if "L2" in mode and j >= 2:
        k += [("L2", s[j - 2])] + ([("LL", s[j - 2], s[j - 1])] if pairs[s[j - 2:j]] >= 2 else [])
    if "R" in mode and R is not None:
        k += [("R", int(i)) for i in R]
    if mode == "R":
        k = [("R", int(i)) for i in R]
    return k


for mode in MODES:
    t0 = time.time()
    ro = Readout()
    for s in train:
        R = rstep(np.zeros(0, np.int64), s[0]) if "R" in mode else None
        for j in range(1, len(s)):
            A = ro.ids(keys(s, j, R, mode), True)
            ro.learn(A, oid.get(s[j]))
            if "R" in mode:
                R = rstep(R, s[j])
    hit = n = 0
    for s in test:
        R = rstep(np.zeros(0, np.int64), s[0]) if "R" in mode else None
        for j in range(1, len(s)):
            A = ro.ids(keys(s, j, R, mode), False)
            if len(A):
                hit += int(outs[int(np.argmax(ro.drive(A)))] == s[j])
            n += 1
            if "R" in mode:
                R = rstep(R, s[j])
    print(f"{ntr} dialogues, {'maturing' if MATURE else 'eta 0.3'} readout, context {mode:5s}: next-char top-1 {hit / n:.3f} (n={n}, units {ro.n}, {time.time() - t0:.0f}s)", flush=True)
