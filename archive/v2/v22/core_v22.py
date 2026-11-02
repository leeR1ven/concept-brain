# -*- coding: utf-8 -*-
"""ARCH_v2.2 core = ARCH_v2.1's RoleCore (core_v21.py) + two optional parts.  Every switch is OFF by default, so with
rm=None and dip_g=1 this is RoleCore bit for bit (test_equiv_v22.py).

1. PFC-2 born random (step 2; randmix's RandMix, randlayer.py).  rm = dict of RandMix arguments.  The layer replaces
   MixDict: GEN, the three copy gates, gateR (the "say what I recognise" gate) and the critic read its active units
   exactly as they read MIX units.  Its own plasticity (instar, use-it-or-lose-it with regrowth) sees the GEN context
   of every learning event (as in randmix; the say route's extra lookup does not change what the layer learns from).
2. Dip size (step 4).  A dip (r = 0 after an own utterance that got no reaction) is applied with its critic and actor
   steps scaled by dip_scale (set by the agent for that call; 1 = ARCH_v2.1).

Diagnostics: ral_log = (t_on, t_end, outcome, after_own, contingent, n_items) per resolved heard line (to match the
world's ground-truth line types afterwards; never read by the agent)."""
from __future__ import annotations

from core_v21 import RoleCore, RCfg      # noqa: F401


class V22Core(RoleCore):
    def __init__(self, cfg, rm=None):
        super().__init__(cfg)
        self.rmix = bool(rm)
        if rm:
            from randlayer import RandMix
            self.mix = RandMix(seed=cfg.seed, **dict(rm))
        self.dip_scale = 1.0
        self.ral_log = []

    # ------------------------------------------------------------------ the random layer's own plasticity
    def _learn_step(self, st, target, g=1.0, eta=None):
        cands = super()._learn_step(st, target, g=g, eta=eta)
        if self.rmix:
            self.mix.learn_event()
        return cands

    def _say_route(self, st, total, create):
        if not self.rmix:
            return super()._say_route(st, total, create)
        la, lx = self.mix.last_A, self.mix.last_X        # the layer learns from the GEN context, not the route's lookup
        out = super()._say_route(st, total, create)
        self.mix.last_A, self.mix.last_X = la, lx
        return out

    # ------------------------------------------------------------------ dips with a smaller step
    def reward(self, r):
        g = self.dip_scale
        if r or g == 1.0:
            return super().reward(r)
        c = self.c
        e0, ea = self.critic.eta, c.eta_actor
        self.critic.eta, c.eta_actor = e0 * g, ea * g
        try:
            return super().reward(r)
        finally:
            self.critic.eta, c.eta_actor = e0, ea

    # ------------------------------------------------------------------ diagnostics of role resolution
    def _ral_commit(self, p):
        super()._ral_commit(p)
        self._ral_note(p, 'commit')

    def _ral_cancel(self, p):
        super()._ral_cancel(p)
        self._ral_note(p, 'cancel')

    def _ral_note(self, p, what):
        if len(self.ral_log) < 300000:
            self.ral_log.append((int(p.get('t_on', -1)), int(p.get('t_end', -1)), what, bool(p.get('after_own')),
                                 bool(p.get('cont')), len(p.get('items') or [])))
