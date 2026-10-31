# -*- coding: utf-8 -*-
"""ARCH_v2.1 core (RoleCore) with PFC-2 = randmix's random-at-birth layer reading learned meaning codes.

MeanCore = RoleCore (v2.1: role-aware listening, say routes, value loop; unchanged) with
  * self.mix = MeanMix (meanlayer.py): randmix's RandMix whose item bundles carry sound + meaning lines;
  * _learn_step -> the layer's own plasticity sees each learning event (randmix's RandCore hook, unchanged);
  * line_end of a HEARD line (learning on, not a replay) -> MeanCodes.observe_line(items) (mode ri / tc / cl);
  * awake recall (mode hip) -> MeanCodes.observe_pair(replayed line, its NEXT line).
mean=None: the meaning layer is absent and the layer is randmix's RandMix (sound codes only) = the v2.1 + c_flat base.
MeanAgent = AgentV21 whose core is a MeanCore; the layer's sound codes read the agent's own Z templates."""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
for _d in ('synth', 'randmix', 'v21'):   # repo: the unchanged ARCH_v2, randmix and v2.1 code
    if os.path.join(HERE, '..', _d) not in sys.path:
        sys.path.insert(1, os.path.join(HERE, '..', _d))
from core_v21 import RoleCore                # noqa: E402
from meanlayer import MeanCodes, make_mix    # noqa: E402
from rules import Hippo                       # noqa: E402
import numpy as np                           # noqa: E402


class MeanCore(RoleCore):
    def __init__(self, cfg, rm, mean=None):
        super().__init__(cfg)
        self.mean_cfg = dict(mean) if mean is not None else None
        mean = dict(mean) if mean is not None else None
        self.hipcue = float(mean.pop('hipcue', 0.0)) if mean else 0.0      # meaning in the hippocampal DG cue
        hipnorm = int(mean.pop('hipnorm', 0)) if mean else 0
        self.mcodes = MeanCodes(V=cfg.V, seed=cfg.seed, **mean) if mean is not None else None
        self.mix = make_mix(self.mcodes, rm, cfg.seed)
        if self.hipcue > 0 and self.mcodes is not None:
            c, D = self.c, self.mcodes.D
            groups = {'bag': c.V, 'first': c.V, 'last': c.V, 'scene': c.n_eye, 'ctx': self.ctx.n, 'mbag': D, 'mfirst': D}
            fan = [10, 4, 4, 10, 10, 10, 10]
            # hipnorm=1: weight the meaning groups so that their expected drive per DG unit equals hipcue x the
            # identity groups' (an item is 1 of V identity lines but km of D meaning lines: ratio km V / D)
            wm = self.hipcue * (D / (self.mcodes.km * c.V) if hipnorm else 1.0)
            self.hip_wm = wm
            wts = {'bag': 1.0, 'first': 1.0, 'last': 1.0, 'scene': c.w_scene, 'ctx': c.w_ctx,
                   'mbag': wm, 'mfirst': wm}
            assert not c.hip_bigram
            self.hip = Hippo(groups, n_dg=c.hip_ndg, k_dg=c.hip_kdg, fan=tuple(fan), weights=wts,
                             cap=c.hip_cap, seed=c.seed + 7, gamma=c.hip_gamma, theta=c.hip_theta)

    def _cue(self, items, eye):
        out = super()._cue(items, eye)
        if self.hipcue > 0 and self.mcodes is not None:
            D = self.mcodes.D
            mb = np.zeros(D, bool); mf = np.zeros(D, bool)
            for p, i in enumerate(items or []):
                cd = self.mcodes.get(i) if 0 <= i < self.c.V else None
                if cd is not None:
                    mb[cd] = True
                    if p == 0:
                        mf[cd] = True
            out['mbag'], out['mfirst'] = mb, mf
        return out

    def _learn_step(self, st, target, g=1.0, eta=None):
        cands = super()._learn_step(st, target, g=g, eta=eta)
        self.mix.learn_event()
        return cands

    def line_end(self, st, learn=True, store=True):
        items = list(st['items'])
        out = super().line_end(st, learn=learn, store=store)
        if self.mcodes is not None and learn and self.c.learn_listen and self.storing and not self._replaying:
            self.mcodes.observe_line(items)
        return out

    def _replay(self, e, learn, eta):
        out = super()._replay(e, learn, eta)
        if self.mcodes is not None and self.mcodes.mode == 'hip' and learn and self.storing:
            rec = self.hip.content[e]
            nxt = self.hip.content[rec['next']] if rec is not None and rec.get('next') is not None else None
            if nxt is not None:
                self.mcodes.observe_pair(rec['items'], nxt['items'])
        return out


def make_agent(world, seed=0, rm=None, mean=None, **kw):
    from agent_v21 import AgentV21
    ag = AgentV21(world, seed=seed, **kw)
    if rm is not None:
        ag.core = MeanCore(ag.core.c, rm, mean)
        ag.core.mix.feat_src = ag.Z
    return ag
