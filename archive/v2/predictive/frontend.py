# -*- coding: utf-8 -*-
"""Fixed ear layer 0 + FEAT0 (exact replica of ARCH_v1 / arch/content/fe.py: Net3D layer-0 wiring and bias drawn with
the same constructor and seed, encode, top-k 40 %, output reserve U 0.5 tau 6).  FEAT0 = layer-0 units used <= 60 % of
ticks during a birth warm-up of 60 random chars (each followed by 4 silent ticks).  Nothing here learns."""
from __future__ import annotations

import numpy as np


class Ear0:
    def __init__(self, F=8, T=6, levels=10, zs=(40, 60, 120, 240), frac=(.4, .3, .2, .1), alpha=.01, bias=.5,
                 spare=.25, seed=0, U=0.5, tau=6.0):
        rng = np.random.default_rng(seed)
        self.F, self.T, self.levels = F, T, levels
        z0 = levels * 2
        prev, out = (F, T, z0), (F, T, zs[0])
        x, y, z = np.meshgrid(np.arange(out[0]), np.arange(out[1]), np.arange(out[2]), indexing='ij')
        oc = np.stack([x, y, z], -1).reshape(-1, 3)
        n = len(oc); cap = n + int(round(n * spare))
        selfc = oc.copy(); selfc[:, 2] = np.round(oc[:, 2] * (prev[2] - 1) / max(out[2] - 1, 1)).astype(int)
        idx = np.zeros((cap, 4), np.int64); w = np.zeros((cap, 4), np.float32); b = np.zeros(cap, np.float32)
        idx[:n, 0] = np.ravel_multi_index(selfc.T, prev); w[:n, 0] = 1.0
        for j in range(1, 4):
            nb = selfc.copy()
            nb[:, 0] = np.clip(nb[:, 0] + rng.integers(-1, 2, n), 0, prev[0] - 1)
            nb[:, 1] = np.clip(nb[:, 1] + rng.integers(-1, 2, n), 0, prev[1] - 1)
            nb[:, 2] = np.clip(nb[:, 2] + rng.integers(-2, 3, n), 0, prev[2] - 1)
            idx[:n, j] = np.ravel_multi_index(nb.T, prev)
            w[:n, j] = (rng.random(n) * .5 + .5) * alpha
        b[:n] = rng.random(n) * bias
        self.idx, self.w, self.b = idx, w, b
        self.alive = np.zeros(cap, bool); self.alive[:n] = True
        self.n = cap
        self.k = int(round(frac[0] * self.alive.sum()))
        self.U, self.tau = U, tau
        self.feat = None
        self.reset()

    def reset(self):
        self.x = np.ones(self.n, np.float32)
        self.s = np.zeros(self.n, bool)

    def encode(self, win):
        img = np.asarray(win).reshape(self.F, self.T, -1)
        up = img[..., None] > np.arange(self.levels)
        return np.stack([up, ~up], -1).reshape(-1)

    def respond(self, lines):
        h = lines.astype(np.float32)
        s = (h[self.idx] * self.w).sum(1) + self.b
        s = np.where(self.alive, s, -np.inf)
        on = np.zeros(self.n, bool)
        on[np.argpartition(-s, self.k)[:self.k]] = True
        return on & self.alive

    def step(self, win):
        """one tick: returns layer-0 state s(t) (bool) and the reserve x used for currents at t."""
        prev = self.s
        self.s = self.respond(self.encode(win))
        self.x = np.where(prev, self.x * (1 - self.U), self.x)
        self.x = self.x + (1 - self.x) / self.tau
        return self.s

    def warmup(self, windows, thr=0.6):
        """birth calibration: FEAT0 = alive layer-0 units on in <= thr of the warm-up ticks."""
        use = np.zeros(self.n)
        for wv in windows:
            use += self.step(wv)
        use /= max(len(windows), 1)
        self.feat = np.flatnonzero(self.alive & (use <= thr))
        self.reset()
        return len(self.feat)

    def feat0(self):
        return self.s[self.feat]
