# -*- coding: utf-8 -*-
"""ARCH_v2.2 agent = ARCH_v2.1's AgentV21 on the V22Core (core_v22.py).  Every new key is OFF by default: with
rm=None, veto_R1=0, dip_g=1, dip_scope='all' it is ARCH_v2.1 (v21final semantics) bit for bit.

New keys
  rm_<k>=v      RandMix arguments (randlayer.py); rm=1 switches PFC-2 to the random-at-birth layer (step 2).
  veto_R1=1     step 3: the learned NoGo veto (V_go < val_veto) also holds a REPLY thought when the line it replies to
                was itself a reaction to my own utterance (R = 1: someone spoke right after me: feedback, a recast, a
                greeting returned).  No reply is owed there; speaking is kept only where it has been welcome.
                (u_R1 < 1, an existing ThinkAgent key, removes the innate urgency there.)
                veto_R1=2: the veto holds every reply thought (speak only where speaking has been welcome).
                veto_W: the veto holds only replies to lines that started <= veto_W ticks after my own utterance
                ended (the reaction window: a line that soon after me is a reaction to me; 40 ticks = 3 s).
                veto_rel: the reply veto's threshold is veto_rel * rbar (rbar = the agent's running rate of rewarded
                reactions, ThinkAgent's own baseline) instead of the fixed val_veto: a reply is withheld only where it is
                less welcome than usual, so a world without any feedback (untutored use) does not silence the agent.
                veto_gate: the reply veto is active only while rbar >= veto_gate: hold back only in a world that rewards
                at all (rbar decays to 0 in untutored use, where a learned veto would otherwise silence every reply).
                veto_fast: the gate reads a fast running reward rate (time constant 1 / veto_fast reactions) so that it
                closes within a few dozen unrewarded reactions.
                veto_eps: a vetoed reply thought escapes the veto with this probability (exploration, so a context
                whose value fell below the threshold can still be re-learned: no absorbing silence).
  dip_scope     'all' (ARCH_v2.1), 'scene' (self-initiated only), 'pred' (step 4: a dip only where a reaction was
                expected: P_react(context of the released thought) >= dip_pthr; P_react is a local predictor of
                "another line starts within react_T ticks after my utterance", learned like V_go).
  dip_g         step 4: dip step size (critic and actor), 1 = ARCH_v2.1.
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import _paths  # noqa: E402,F401  (repo: v2.2's own agent_v21 / rules / randlayer before ../v21 and ../synth)
from agent_v21 import AgentV21, V21_AGENT_KEYS      # noqa: E402
from agent_think import SparsePR                    # noqa: E402
from core_v22 import V22Core                        # noqa: E402

V22_AGENT_KEYS = dict(V21_AGENT_KEYS, veto_R1=int, dip_scope=str, dip_g=float, dip_pthr=float, veto_eps=float, veto_W=int, veto_rel=float, veto_gate=float, veto_fast=float, gain_reply=int)


class AgentV22(AgentV21):
    def __init__(self, world, seed=0, rm=None, veto_R1=0, dip_scope='all', dip_g=1.0, dip_pthr=0.5, veto_eps=0.0, veto_W=0, veto_rel=0.0, veto_gate=0.0, veto_fast=0.0, gain_reply=0, **kw):
        self._rm = rm
        super().__init__(world, seed=seed, **kw)
        if rm:
            self.core.mix.feat_src = self.Z          # item codes from the agent's own perceptual templates
        self.veto_R1, self.dip_scope, self.dip_g, self.dip_pthr = veto_R1, dip_scope, dip_g, dip_pthr
        self.veto_eps = veto_eps                  # exploration: a reply thought escapes the veto with this probability
        self.veto_W = veto_W                      # reaction window: the veto holds only replies to lines that started
        self._own_end = -10 ** 9                  # <= veto_W ticks after my own utterance ended (0 = no window)
        self._line_gap = None
        self.veto_rel = veto_rel                  # > 0: the reply veto's threshold is veto_rel * rbar (the running rate
                                                  # of rewarded reactions) instead of val_veto: "less welcome than usual"
        self.veto_gate = veto_gate                # > 0: the reply veto is used only while rbar >= veto_gate (a partner
                                                  # who rewards anything at all is around; none does in untutored use)
        self.gain_reply = gain_reply              # 1: reply thoughts keep ARCH_v2's learned eagerness G = g0 + beta (V - rbar)
        self.veto_fast = veto_fast                # > 0: the gate reads a fast running reward rate rr (rate veto_fast per
        self.rr = 0.5                             # reaction, same samples as rbar) instead of rbar
        self._veto_rng = __import__('numpy').random.default_rng(seed + 977)
        self.P_react = SparsePR(eta=0.2, init=0.5) if dip_scope == 'pred' else None
        self._dip_keys = None
        self.tstats['blocked_R1'] = 0
        self.dip_stats = dict(armed=0, fired=0, skipped_pred=0)

    def _make_core(self, cfg):
        return V22Core(cfg, rm=self._rm)

    # ------------------------------------------------------------------ step 3: no compulsive reply to a reaction
    def _intend(self):
        th = self.th
        AgentV21._intend(self)
        if self.veto_R1 and self.val_veto > 0 and self.REL and th is not None and self.th is th \
                and th['kind'] == 'reply' and (self.R_line or self.veto_R1 == 2) and not th['released'] \
                and (not self.veto_W or (self._line_gap is not None and self._line_gap <= self.veto_W)) \
                and (self.rr if self.veto_fast > 0 else self.rbar) >= self.veto_gate \
                and th.get('V', 1.0) < (self.veto_rel * self.rbar if self.veto_rel > 0 else self.val_veto):
            if self.veto_eps > 0:                  # decided once per thought (no per-tick re-draw)
                if 'escape' not in th:
                    th['escape'] = bool(self._veto_rng.random() < self.veto_eps)
                if th['escape']:
                    return
            self.REL = False
            self.tstats['blocked_R1'] += 1

    def _gain(self, th):
        if getattr(self, 'gain_reply', 0) and th['kind'] == 'reply':
            from agent_synth import SynthAgent
            return SynthAgent._gain(self, th)       # sets th['V'] too
        return AgentV21._gain(self, th)

    # ------------------------------------------------------------------ step 4: dips
    def _other_onset(self):
        self._line_gap = (self.t - self._own_end) if self.spoke_since_line else None   # efference timing
        if self.P_react is not None and self._dip_t is not None and self._dip_keys and self.learning \
                and self.t - self._dip_t <= self.react_T:
            self.P_react.learn(self._dip_keys, 1.0)          # a reaction came after my utterance
        return AgentV21._other_onset(self)

    def step(self, obs):
        n0, rb0 = self.rbar_n, self.rbar
        force = AgentV21.step(self, obs)
        if self.veto_fast > 0 and self.rbar_n > n0:   # the same reaction samples rbar just took (r > 0 or not)
            m = min(self.rbar_n, 200)
            y = 1.0 if (rb0 + m * (self.rbar - rb0)) > 0.5 else 0.0
            self.rr += (y - self.rr) * self.veto_fast
        if self.core.actor_t == self.t - 1:      # own_finish ran this step: my utterance just ended
            self._own_end = self.t
        return force

    def _dip_fire(self):
        self.dip_stats['fired'] += 1
        if self.P_react is not None and self._dip_keys:
            p = self.P_react.p(self._dip_keys)
            self.P_react.learn(self._dip_keys, 0.0, p)
            if p < self.dip_pthr:                             # no reaction was expected: no disappointment
                self.dip_stats['skipped_pred'] += 1
                return
        self.core.dip_scale = self.dip_g
        try:
            self.core.reward(0.0)
        finally:
            self.core.dip_scale = 1.0
