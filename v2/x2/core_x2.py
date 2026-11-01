# -*- coding: utf-8 -*-
"""x2 core = ARCH_v2.2's V22Core + the designer's weight-stability rule (2026-11-01) as an option of the existing
plasticity (rules.Delta.enable_stab; randlayer.RandMix(stab=)).  Every key is OFF by default (stab = None): then this is
V22Core bit for bit (test_equiv_x2.py).

stab = dict(beta, w_ref, f_min, track, beta_g):
  GEN (the readout, per synapse):  eta_eff = eta * max(1 / (1 + beta * u_ij * |w_ij| / w_ref), f_min)
  copy gates + gateR (beta_g > 0): the same rule with beta_g.
  u_ij = learning events the synapse took part in (pre active, post in the error set); never decays (lifelong)."""
from __future__ import annotations

from core_v22 import V22Core


class X2Core(V22Core):
    def __init__(self, cfg, rm=None, stab=None):
        super().__init__(cfg, rm=rm)
        self.stab_cfg = stab
        if stab:
            b, wr, fm = float(stab.get('beta', 0.0)), float(stab.get('w_ref', 0.01)), float(stab.get('f_min', 0.05))
            track = bool(stab.get('track', 0)) or b <= 0
            self.gen.enable_stab(b, wr, fm, track=track)
            bg = float(stab.get('beta_g', 0.0))
            if bg > 0 or track:
                for r in (self.gate, self.gateR):
                    if r is not None:
                        r.enable_stab(bg, float(stab.get('w_ref_g', 0.1)), fm, track=bg <= 0)
