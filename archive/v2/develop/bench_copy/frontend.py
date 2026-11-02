# -*- coding: utf-8 -*-
"""The fixed ear front end of ARCH_v1, exact (from arch/synth/archv1.py, arch/content/fe.py and
arch/verify_learnability_dynamics/ear0.py):

  Ear0   the ear's first 3D layer ("layer 0") of brain/net3d.py: 1,920 born neurons (+25 % spare slots) reading the 960
         thermometer input lines (8 bands x 6 slices x 10 levels x up/down), fan-in 4 (self weight 1 + 3 neighbours
         U(.5,1)*.01), bias U(0,.5), top-k 40 %; output reserve x (deplete 0.5 when on, recover (1-x)/6).
         Wiring and bias are bit-identical to Net3D(8, 6, seed) layer 0 (same rng draws, selftest checks).
         Layer 0 receives no learned or returned current in ARCH_v1, so it is exact for any v1 agent.
  feat0  FEAT0 = layer-0 neurons on in <= 60 % of the ticks of the birth warm-up (60 random chars, 4 silent ticks each;
         World.warmup_windows()).  1,152 neurons with random physics, every seed.
  T      ARCH_v1's PFC working memory: top-K of [sum of 30 fixed random FEAT0 inputs x reserve at t-1 + 3.0 self-hold
         - 0.3 max(0, u/(K/N) - 1) + eps], u += (T - u)/1000.  v1_tie() replays archv1's rng so T is tick-identical
         to the integrated toy.

    ear = Ear0(seed); ear.set_feat0(feat0(Ear0(seed), world.warmup_windows()))
    s = ear.step(obs['ear'])        # layer-0 state at t; ear.s_prev / ear.x_prev = state and reserve at t-1
"""
from __future__ import annotations

import numpy as np


class Ear0:
    def __init__(self, seed=0, F=8, T=6, levels=10, zs0=40, frac0=0.4, alpha=0.01, bias=0.5, spare=0.25, U=0.5,
                 tau=6.0, x_dtype=np.float32):
        """x_dtype: float32 = archv1.py (the integrated toy, default); float64 = ear0.py / fe.py."""
        rng = np.random.default_rng(seed)
        self.F, self.T, self.levels = F, T, levels
        prev, out = (F, T, levels * 2), (F, T, zs0)
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
        self.n = cap
        self.alive = np.zeros(cap, bool); self.alive[:n] = True
        self.k = int(round(frac0 * n))
        self.U, self.tau, self.x_dtype = U, tau, x_dtype
        self._lv = np.arange(levels)
        self.feat0 = None
        self.reset()

    def reset(self):
        self.s = np.zeros(self.n, bool)                  # state at t (after step)
        self.x = np.ones(self.n, self.x_dtype)           # reserve after the update of tick t
        self.s_prev = np.zeros(self.n, bool)             # state at t-1
        self.x_prev = np.ones(self.n, self.x_dtype)      # reserve used for currents at t (= x after tick t-1)

    def encode(self, window):
        img = np.asarray(window).reshape(self.F, self.T, 1)
        up = img > self._lv
        return np.stack([up, ~up], -1).reshape(-1)

    def respond(self, lines):
        s = (lines.astype(np.float32)[self.idx] * self.w).sum(1) + self.b
        s[~self.alive] = -np.inf
        on = np.zeros(self.n, bool)
        on[np.argpartition(-s, self.k)[:self.k]] = True
        return on

    def step(self, window):
        """one tick: layer 0 responds to the window at t.  Afterwards s_prev/x_prev hold the state/reserve at t-1
        (what an ARCH_v1 unit reads at t) and s the state at t; x is depleted by the state at t-1 (Region order)."""
        self.s_prev, self.x_prev = self.s, self.x.copy()
        self.s = self.respond(self.encode(window))
        self.x = np.where(self.s_prev, self.x * (1 - self.U), self.x)
        self.x += (1 - self.x) / self.tau
        return self.s

    def set_feat0(self, feat0_idx):
        self.feat0 = np.asarray(feat0_idx, np.int64)


def feat0(ear: Ear0, windows, thr=0.6):
    """FEAT0 from warm-up windows (World.warmup_windows()): alive layer-0 neurons with usage <= thr.  The ear is reset."""
    ear.reset()
    use = np.zeros(ear.n)
    for wdw in windows:
        use += ear.step(wdw)
    ear.reset()
    u = use / max(len(windows), 1)
    return np.flatnonzero(ear.alive & (u <= thr))


def feat0_v1(ear: Ear0, windows):
    """FEAT0 as archv1.py's warm-up effectively computes it (its usage loop adds the state after the 4 silent ticks
    7 times per char): alive layer-0 neurons that are off in silence = 1,920 - 768 = 1,152 every seed.  Use this for
    tick-exact equivalence with the integrated ARCH_v1 toy; feat0() is the intended usage rule (ear0.py, fe.py)."""
    ear.reset()
    for wdw in windows:
        ear.step(wdw)
    silent = np.zeros_like(np.asarray(windows[0]))
    for _ in range(4):
        ear.step(silent)
    on_in_silence = ear.s.copy()
    ear.reset()
    return np.flatnonzero(ear.alive & ~on_in_silence)


# ------------------------------------------------------------------------------------------ T (v1 working memory)
def v1_tie(seed, NT=4000):
    """the tie-breaker eps of archv1.Brain (drawn from the brain's rng after its sparse paths): replays those draws."""
    from archv1_copy import Net3D
    rng = np.random.default_rng(seed)
    E = Net3D(8, 6, seed=seed)
    cand_e = np.flatnonzero(E.layer >= 1)
    M = Net3D(8, 4, levels=3, zs=(20, 40, 80), frac=(.4, .3, .2), seed=seed + 5)
    cand_c = np.flatnonzero(M.layer >= 1)
    ne, nc = E.n, M.n
    hid = np.r_[cand_e, ne + cand_c, ne + nc + np.arange(32 * 3 + 32)]
    for src, dst, k in ((cand_e, cand_e, 96), (cand_c, cand_c, 96), (cand_e, cand_c, 32), (cand_c, cand_e, 32),
                        (hid, hid, 96)):
        rng.integers(0, len(dst), (len(src), k))
        rng.random((len(src), k))
    return (rng.random(NT) * 0.01).astype(np.float32)


class WorkingMemoryT:
    """ARCH_v1's T: fixed random expansion of FEAT0 x reserve (fan-in 30) with a one-tick self-hold and a set point.
    step(ear) uses the ear's state/reserve at t-1 (ear.s_prev, ear.x_prev) -> T at t."""

    def __init__(self, n_feat0, NT=4000, KT=200, fanin=30, hold=3.0, sp_beta=0.3, sp_tau=1000.0, seed=0, tie=None):
        rng = np.random.default_rng(seed + 77)
        self.T_in = rng.integers(0, n_feat0, (NT, fanin))
        self.tie = tie if tie is not None else (np.random.default_rng(seed + 78).random(NT) * 0.01).astype(np.float32)
        self.NT, self.KT, self.hold, self.sp_beta, self.sp_tau = NT, KT, hold, sp_beta, sp_tau
        self.T = np.zeros(NT, bool)
        self.u = np.full(NT, KT / NT)
        self.learning = True

    def step(self, f_prev, x_prev):
        """f_prev: FEAT0 state at t-1 (bool), x_prev: its reserve at t-1.  Returns T(t) (does not commit u)."""
        if f_prev.any():
            a = (f_prev * x_prev).astype(np.float32)
            drive = a[self.T_in].sum(1)
        else:                                            # silence: exact shortcut (all inputs 0)
            drive = np.zeros(self.NT, np.float32)
        tsc = drive + self.hold * self.T - self.sp_beta * np.maximum(self.u / (self.KT / self.NT) - 1.0, 0) + self.tie
        Tn = np.zeros(self.NT, bool)
        Tn[np.argpartition(-tsc, self.KT)[:self.KT]] = True
        return Tn

    def commit(self, Tn):
        if self.learning:
            self.u += (Tn - self.u) / self.sp_tau
        self.T = Tn
