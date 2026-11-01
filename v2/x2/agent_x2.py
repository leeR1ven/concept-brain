# -*- coding: utf-8 -*-
"""x2 agent = ARCH_v2.2's AgentV22 on the X2Core.  New keys (all OFF by default; then AgentV22 bit for bit):
  stab=beta        the weight-stability rule on GEN (per synapse, both signs; rules.Delta.enable_stab)
  stab_ref=w_ref   weight scale of the rule (GEN)        stab_min=f_min   floor of the step factor (keeps learning)
  stab_g=beta_g    the same rule on the copy gates and gateR (w_ref_g = stab_refg)
  stab_track=1     keep the use traces only (beta 0: diagnostics, steps unchanged)
  (the random layer's instar stability is rm_stab=beta_rm, rm_stab_min=f_min: RandMix arguments)"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
from agent_v22 import AgentV22, V22_AGENT_KEYS      # noqa: E402
from core_x2 import X2Core                          # noqa: E402

X2_AGENT_KEYS = dict(V22_AGENT_KEYS, stab=float, stab_ref=float, stab_min=float, stab_g=float, stab_refg=float,
                     stab_track=int)


class AgentX2(AgentV22):
    def __init__(self, world, seed=0, stab=0.0, stab_ref=0.01, stab_min=0.05, stab_g=0.0, stab_refg=0.1,
                 stab_track=0, **kw):
        on = stab > 0 or stab_g > 0 or stab_track
        self._stab = dict(beta=stab, w_ref=stab_ref, f_min=stab_min, beta_g=stab_g, w_ref_g=stab_refg,
                          track=stab_track) if on else None
        super().__init__(world, seed=seed, **kw)

    def _make_core(self, cfg):
        return X2Core(cfg, rm=self._rm, stab=self._stab)
