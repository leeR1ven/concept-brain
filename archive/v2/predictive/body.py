# -*- coding: utf-8 -*-
"""The innate body (ARCH_v1 §1.3, copied logic; kept as a reflex scaffold).  Synchronous, all inputs from t-1:
  F   = [0.01 E(t-1) - 10 sumCD(t-1) >= 0.1]          someone else is talking (ear INPUT window energy)
  GO  = F(t-2) & !F(t-1) & !SPK(t-1) & !CD(t-1)        one pulse at the offset of the other's speech
        (+ a spontaneous start SP(t-1) & !F & !SPK & !CD: the babbling / play CPG; innate, off by default)
  CD  = 4-unit delay line driven by GO | SPK           corollary discharge: own voice hidden from F
  R1  = GO(t-1);  boundary = (R1(t-1) & !F(t-1)) | (SPK(t-1) & phase(t-1) == 2)     the 3-tick beat
  at a boundary the MIND gives one score per program and one END score; the body applies (first boundary = none held):
      not first and s_END >= s_best          -> stop (END)
      not first and n >= 12                  -> stop (breath)
      s_best < theta                         -> stop (floor) / stay silent (first)
      else release argmax: Hp = y for 3 ticks, D = its muscle set, force ramp 1-2-3 (D/I/M slaved, same tick)
Programs are muscle sets owned by the mind (created by babbling).  Nothing here learns."""
from __future__ import annotations

import numpy as np


class Body:
    def __init__(self, n_muscles=32, cd_len=4, breath=12, theta=0.15, f_w=0.01, f_thr=0.1):
        self.M = n_muscles
        self.cd_len, self.breath, self.theta, self.f_w, self.f_thr = cd_len, breath, theta, f_w, f_thr
        self.reset()

    def reset(self):
        self.F = False; self.F2 = False; self.GO = False; self.R1 = False
        self.CD = np.zeros(self.cd_len, bool)
        self.SPK = False; self.phase = 0; self.Hp = -1; self.nsyl = 0
        self.E_prev = 0.0
        self.SP = False
        self.force = np.zeros(self.M, np.float32)
        self.t = 0

    def step(self, E_now, scores_fn, prog_sets, spont=False, babble_fn=None):
        """E_now: input-window energy at t (read by F at t+1).  scores_fn(first) -> (scores per program (array),
        s_end) computed by the mind from states at t-1.  prog_sets: list of muscle index arrays.  babble_fn():
        returns a muscle set to release at this boundary instead of the mind's choice (or None).
        Returns (force (M,), ev dict)."""
        ev = {}
        ncd = int(self.CD.sum())
        F_new = (self.f_w * self.E_prev - 10.0 * ncd) >= self.f_thr
        F2_new = self.F
        GO_new = ((1.0 * self.F2 - 2.0 * self.F - 2.0 * self.SPK - 2.0 * ncd) >= 1.0) or \
                 (self.SP and not self.F and not self.SPK and ncd == 0 and not self.GO and not self.R1)
        CD_new = np.r_[self.GO or self.SPK, self.CD[:-1]]
        R1_new = self.GO
        bnd = (self.R1 and not self.F) or (self.SPK and self.phase == 2)
        if bnd:
            first = not self.SPK
            ev['bnd'] = True; ev['first'] = first
            if first:
                self.nsyl = 0
            choice = None
            bab = babble_fn() if babble_fn is not None else None
            if isinstance(bab, str) and bab == 'STOP':
                ev['stop'] = 'babble_end'
            elif bab is not None:
                choice = ('babble', tuple(int(m) for m in bab))
            else:
                sc, s_end = scores_fn(first)
                ev['s_end'] = float(s_end)
                best = int(np.argmax(sc)) if len(sc) else -1
                s_best = float(sc[best]) if best >= 0 else 0.0
                ev['s_best'] = s_best
                if not first and s_end >= s_best:
                    ev['stop'] = 'END'
                elif not first and self.nsyl >= self.breath:
                    ev['stop'] = 'breath'
                elif best < 0 or s_best < self.theta:
                    ev['stop'] = 'floor' if not first else 'silent'
                else:
                    choice = ('prog', best)
            if choice is not None:
                self.Hp = choice; self.phase = 0; self.nsyl += 1
                ev['rel'] = choice
            else:
                self.Hp = -1; self.phase = 0
        elif self.SPK:
            self.phase += 1
        SPK_new = self.Hp != -1
        force = np.zeros(self.M, np.float32)
        if SPK_new:
            kind, val = self.Hp
            mus = prog_sets[val] if kind == 'prog' else val
            force[list(mus)] = self.phase + 1
        # commit
        self.F, self.F2, self.GO, self.R1, self.CD = bool(F_new), bool(F2_new), bool(GO_new), bool(R1_new), CD_new
        self.SPK = SPK_new
        self.SP = bool(spont)
        self.E_prev = float(E_now)
        self.force = force
        self.t += 1
        ev.update(F=self.F, GO=self.GO, CD=int(self.CD.sum()), SPK=self.SPK, phase=self.phase)
        return force, ev
