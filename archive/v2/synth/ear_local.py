# -*- coding: utf-8 -*-
"""Module test: learned LOCAL receptive fields in the ear's upper layers (designer's local connectivity; coordinator's
synthesis item).  Layer 0 = the bench's exact fixed ear layer (Ear0 / FEAT0, 1,153 units, a near-copy of the input
lines).  Layers 1-2 are 3D grids over the same (band, slice) map; each unit sees only layer-(l-1) units within a local
neighbourhood (|df| <= r, |dt| <= r, all depths) and learns with a competitive Hebbian rule:
  in every (band, slice) column, the k units with the largest normalised match w.x win (k-WTA, lateral inhibition);
  winners: w <- w + eta (x/|x| - w)   (instar; weights stay normalised; local: pre, post and own weight only).
Training stream: aligned and misaligned windows of the bench's own chars in the partner's voice (other) only.
Tests (SIMULATED): syllable identity decoded by nearest centroid (centroids from voice 'other', clean) on the other
two voices (other2, self) and on noisy windows (bench noise sigma 1, 2), per layer; unit selectivity (how many of the
chars a unit responds to).  The body's F detector reads the input energy only, so it is unaffected by construction.
usage: python3 ear_local.py [seed=0] [D=12] [k=2] [r=1] [epochs=3]"""
import os, sys, time, json
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'bench_copy'))
import numpy as np
from frontend import Ear0, feat0
from physics import Stream
from world import World

A = dict(a.split('=', 1) for a in sys.argv[1:])
seed, D, K, R, EP = int(A.get('seed', 0)), int(A.get('D', 12)), int(A.get('k', 2)), int(A.get('r', 1)), int(A.get('epochs', 3))
rng = np.random.default_rng(seed)
w = World('C', seed=seed)
voice = w.voice
ear = Ear0(seed=w.seed)
F0 = feat0(Ear0(seed=w.seed), w.warmup_windows())
ear.set_feat0(F0)
# 3D coordinates of the FEAT0 units: Ear0 builds its 8 x 6 x 40 grid in meshgrid(ij) order
F, T, Z0 = 8, 6, 40
coord = np.stack(np.meshgrid(np.arange(F), np.arange(T), np.arange(Z0), indexing='ij'), -1).reshape(-1, 3)
c0 = coord[F0]


def windows(ch, who, noise=0.0, nseed=0, align=2):
    st = Stream(voice.F, voice.T, voice.T // 3, 3, voice.levels, noise=noise, noise_seed=nseed)
    e = Ear0.__new__(Ear0); e.__dict__.update(ear.__dict__); e.reset()
    g = voice.hear(ch, who)
    none = (None, None, False)
    out = []
    for k in range(3):
        src = [none, none, none]
        src[{'other': 0, 'other2': 1, 'self': 2}[who]] = (ch, g, k == 0)
        win = st.tick(tuple(src))
        s = e.step(win)
        out.append(s[F0].copy())
    return out[align]


def lexicon():
    L = set()
    for st in 'ABCD':
        ww = World(st, seed=seed)
        ww.run(type('S', (), {'step': lambda self, o: np.zeros(32)})(), 60)
        for r in ww.log:
            L |= set(r.get('prompt') or '') | set(r.get('model_reply') or '') | set(''.join(r.get('acceptable') or []))
    return sorted(c for c in L if c in voice.sound)


chars = lexicon()
print(f"chars {len(chars)}, FEAT0 {len(F0)}", flush=True)


class Layer:
    def __init__(self, in_coord, D, k, r, eta=0.05, seed=0):
        rg = np.random.default_rng(seed)
        self.cols = [(f, t) for f in range(F) for t in range(T)]
        self.D, self.k = D, k
        self.rf = []
        for (f, t) in self.cols:
            m = (np.abs(in_coord[:, 0] - f) <= r) & (np.abs(in_coord[:, 1] - t) <= r)
            self.rf.append(np.flatnonzero(m))
        self.W = [rg.random((D, len(ix))).astype(np.float32) for ix in self.rf]
        for Wc in self.W:
            Wc /= np.linalg.norm(Wc, axis=1, keepdims=True)
        self.eta = eta
        self.coord = np.array([(f, t, d) for (f, t) in self.cols for d in range(D)])

    def __call__(self, x, learn=False):
        out = np.zeros(len(self.cols) * self.D, bool)
        for c, ix in enumerate(self.rf):
            xi = x[ix].astype(np.float32)
            nx = np.linalg.norm(xi)
            if nx == 0:
                continue
            s = self.W[c] @ (xi / nx)
            win = np.argpartition(-s, self.k)[:self.k]
            out[c * self.D + win] = True
            if learn:
                self.W[c][win] += self.eta * (xi / nx - self.W[c][win])
                self.W[c][win] /= np.linalg.norm(self.W[c][win], axis=1, keepdims=True)
        return out


t0 = time.time()
train = []
for ch in chars:
    for al in (2, 1, 0):
        train.append(windows(ch, 'other', align=al))
L1 = Layer(c0, D, K, R, seed=seed)
for ep in range(EP):
    for i in rng.permutation(len(train)):
        L1(train[i], learn=True)
h1 = [L1(x) for x in train]
L2 = Layer(L1.coord, D, K, R + 1, seed=seed + 1)
for ep in range(EP):
    for i in rng.permutation(len(h1)):
        L2(h1[i], learn=True)
print(f"trained in {time.time() - t0:.0f}s", flush=True)


def codes(layer, X):
    if layer == 0:
        return np.array(X)
    H = [L1(x) for x in X]
    return np.array(H) if layer == 1 else np.array([L2(h) for h in H])


def nc_acc(Ctr, Cte):
    P = Ctr.astype(np.float32); Q = Cte.astype(np.float32)
    Pn = P / np.maximum(np.linalg.norm(P, axis=1, keepdims=True), 1e-6)
    Qn = Q / np.maximum(np.linalg.norm(Q, axis=1, keepdims=True), 1e-6)
    return float(np.mean(np.argmax(Qn @ Pn.T, 1) == np.arange(len(Q))))


res = {}
ref = [windows(ch, 'other') for ch in chars]
tests = {'other2': [windows(ch, 'other2') for ch in chars], 'self': [windows(ch, 'self') for ch in chars], 'misalign': [windows(ch, 'other', align=1) for ch in chars]}
for sg in (1.0, 2.0, 4.0, 6.0):
    tests[f'noise{sg:g}'] = [windows(ch, 'other', noise=sg, nseed=i + 7) for i, ch in enumerate(chars)]
for layer in (0, 1, 2):
    Cr = codes(layer, ref)
    row = {k: round(nc_acc(Cr, codes(layer, X)), 3) for k, X in tests.items()}
    act = Cr.mean(0)
    resp = (Cr.astype(np.float32).sum(0))                     # how many chars each unit responds to
    used = resp > 0
    row['units'] = int(Cr.shape[1]); row['active_frac'] = round(float(Cr.mean()), 3)
    row['selective<=5'] = round(float(np.mean(resp[used] <= 5)), 3) if used.any() else None
    row['median_chars_per_unit'] = float(np.median(resp[used])) if used.any() else None
    res[layer] = row
    print(f"layer {layer}: {row}", flush=True)
json.dump(dict(seed=seed, D=D, k=K, r=R, epochs=EP, chars=len(chars), res=res, secs=round(time.time() - t0)),
          open(os.path.join(HERE, 'out', f'ear_local_s{seed}.json'), 'w'))
