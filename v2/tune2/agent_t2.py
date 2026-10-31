# -*- coding: utf-8 -*-
"""tune2 agent = ARCH_v2's SynthAgent with the T2Core (wiring locality, context and wandering records).
With every tune2 key at its default this agent is SynthAgent exactly (the core is rebuilt from the same cfg and seed
before anything has run, so every RNG stream starts where SynthAgent's would)."""
from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
from agent_synth import SynthAgent           # noqa: E402
from core_t2 import T2Core, T2Cfg            # noqa: E402

T2_KEYS = set(T2Cfg.EXTRA2)


class T2Agent(SynthAgent):
    def __init__(self, world, seed=0, **kw):
        t2 = {k: kw.pop(k) for k in list(kw) if k in T2_KEYS}
        super().__init__(world, seed=seed, **kw)
        cfg = T2Cfg()
        cfg.__dict__.update(self.core.c.__dict__)        # every resolved value (seg, rel, retino, ...) carried over
        for k, v in t2.items():
            setattr(cfg, k, v)
        self.core = T2Core(cfg)
        # sheet coordinates of the ear's layer-0 units (band, slice, depth) scaled to [0, 1]: the innate layout
        F, T, Z = 8, 6, 40
        x, y, z = np.meshgrid(np.arange(F), np.arange(T), np.arange(Z), indexing='ij')
        oc = np.stack([x / (F - 1), y / (T - 1), z / (Z - 1)], -1).reshape(-1, 3).astype(np.float32)
        self.f_coord = oc[np.asarray(self.F0)]           # FEAT0 feature -> its unit's position

    def _sheet_pos(self, f):
        on = np.asarray(f) > 0.5
        if not on.any():
            return np.full(3, 0.5, np.float32)
        return self.f_coord[on].mean(0)

    def _detect(self, f):
        n0 = self.Z.n
        j = SynthAgent._detect(self, f)
        if self.Z.n > n0:
            for jj in range(n0, self.Z.n):
                self.core.set_item_pos(jj, self._sheet_pos(self.Z.W[:, jj]))
        return j

