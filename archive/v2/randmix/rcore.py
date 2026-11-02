# -*- coding: utf-8 -*-
"""ARCH_v2 with PFC-2 replaced by the random-at-birth layer (randlayer.RandMix).  Everything else is ARCH_v2
(final_noctx) unchanged: GEN, the BG copy gates and the critic read the random layer's active units exactly as they read
MIX's units (same NLMS rule R2, same 1/|A| normalisation, same actor / critic).

RandCore._learn_step = SynthCore._learn_step + one call into the layer's own plasticity (the layer sees the context
the readout learned at: fan-in instar / use-it-or-lose-it bookkeeping, variants b / c).
RandAgent = SynthAgent whose core is a RandCore; the layer reads the item codes from the agent's own Z templates.
rm=None gives the plain SynthAgent (variant d, the ARCH_v2 dict MIX)."""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
_SYN = os.path.join(HERE, '..', 'synth')   # repo: the unchanged ARCH_v2 core and its bench_copy
if _SYN not in sys.path:
    sys.path.insert(1, _SYN)
from agent_synth import SynthAgent           # noqa: E402
from core_synth import SynthCore             # noqa: E402
from randlayer import RandMix                # noqa: E402


class RandCore(SynthCore):
    def __init__(self, cfg, rm):
        super().__init__(cfg)
        rm = dict(rm)
        use_torch, mf = rm.pop('torch', 0), rm.pop('memfrac', 0.0)
        if use_torch:                                   # the same layer, activity computed in torch on CUDA
            from randtorch import TorchRandMix
            self.mix = TorchRandMix(seed=cfg.seed, mem_frac=float(mf), **rm)
        else:
            self.mix = RandMix(seed=cfg.seed, **rm)

    def _learn_step(self, st, target, g=1.0, eta=None):
        cands = super()._learn_step(st, target, g=g, eta=eta)
        self.mix.learn_event()
        return cands


class RandAgent(SynthAgent):
    def __init__(self, world, seed=0, rm=None, **kw):
        super().__init__(world, seed=seed, **kw)
        self.rm = rm
        if rm is not None:
            self.core = RandCore(self.core.c, rm)
            self.core.mix.feat_src = self.Z
