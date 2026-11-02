# -*- coding: utf-8 -*-
"""ThinkBody = the bench body (ARCH_v1) with ONE change: where GO comes from.

  v1 (go_mode='v1'):   GO(t) = [F2 - 2F - 2SPK - 2 sum CD >= 1]  = the offset of F: a fixed pulse 4 ticks after any line.
  gate (go_mode='gate'): GO(t) = REL(t-1) & !F(t-1) & !SPK(t-1) & !CD(t-1) & !GO(t-1) & !R1(t-1) & idle mouth
      REL = the mind's BG-gate release of the speech intention (set by the mind before step()).
      The innate brake is kept: F ("other is talking") inhibits GO; so do the own voice (SPK), the corollary discharge
      and a GO/R1 already in flight.  Everything downstream is unchanged: CD, R1, R2, boundary = R1(t-1) & !F(t-1)
      (the first syllable is still blocked if the other starts within the 2-tick GO->boundary pipeline), the 3-tick beat,
      program WTA, END rule, breath counter, floor, D/I/M with the 1-2-3 ramp.
The rest of the file is Body.step copied verbatim except the GO line (marked)."""
from __future__ import annotations

import os
import sys

import numpy as np

BENCH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bench_copy")
if BENCH not in sys.path:
    sys.path.insert(0, BENCH)
from body import Body  # noqa: E402


class ThinkBody(Body):
    def __init__(self, *a, go_mode='gate', brake_now=False, **kw):
        super().__init__(*a, **kw)
        self.go_mode = go_mode
        self.brake_now = brake_now   # the FIRST syllable also needs a silent ear now (E(t) < F threshold): no blind window
        self.now_blocked = 0
        self.REL = False
        self.go_blocked = 0          # releases refused by the brake (F / own voice / CD)
        self.first_blocked = 0       # GO fired but the first boundary was blocked by F (other started)

    def step(self, E_now, scorer=None, cue=None, endpush=False, reward=0.0):
        ev = {}
        ncd = int(self.CD.sum())
        F_new = (self.f_w * self.E_prev - 10.0 * ncd) >= self.f_thr
        F2_new = self.F
        # ------------------------------------------------------------------ the one change: GO
        if self.go_mode == 'v1':
            GO_new = (1.0 * self.F2 - 2.0 * self.F - 2.0 * self.SPK - 2.0 * ncd) >= 1.0
        else:
            free = (not self.F) and (not self.SPK) and ncd == 0 and (not self.GO) and (not self.R1) and self.Hp == -1
            GO_new = bool(self.REL) and free
            if self.REL and not free:
                self.go_blocked += 1
        # ------------------------------------------------------------------ unchanged below
        CD_new = np.r_[self.GO or self.SPK, self.CD[:-1]]
        R1_new = self.GO
        R2_new = self.GO or (self.SPK and self.phase == 1)
        bnd = (self.R1 and not self.F) or (self.SPK and self.phase == 2)
        if bnd and self.brake_now and not self.SPK and self.f_w * E_now >= self.f_thr:
            bnd = False; self.now_blocked += 1          # another voice in the ear right now: do not start
        if self.R1 and self.F and not self.SPK:
            self.first_blocked += 1
        Hc_prev = self._hc_at(1)
        Hc_now = self._hc_at(0)
        Y_new, END_new, Yext, ENDext = -1, False, False, False
        first = bnd and not self.SPK
        self.stop = None; self.recruited = None
        if bnd and self.Hp == -2:
            self.stop = 'babble'; self.Hp = -1; self.phase = 0
            self.last_babble = self.trial; self.babble_end_t = self.t; self.trial = None
        elif bnd:
            ev['bnd'] = True
            if first:
                self.nsyl = 0
            com = np.flatnonzero(self.committed)
            if scorer is not None and len(com):
                sc, s_end = scorer(self, com, first)
                sc = np.asarray(sc, np.float64).copy()
                s_end = float(s_end)
            else:
                sc, s_end = np.zeros(len(com)), 0.0
            if endpush:
                s_end += self.end_push
            cue_ov = None
            if cue is not None:
                cv = np.zeros(self.M_N, bool); cv[list(cue)] = True
                ov = (self.prog[com] & cv).sum(1) if len(com) else np.zeros(0, int)
                if not (ov == 4).any() and self.recruit in ('cue', 'both'):
                    y = self.commit(np.flatnonzero(cv))
                    if y is not None:
                        com = np.r_[com, y]; sc = np.r_[sc, 0.0]; ov = np.r_[ov, 4]
                        ev['recruit'] = y; self.recruited = y
                    else:
                        ev['pool_full'] = True
                sc = sc + self.cue_w * ov
                cue_ov = ov
            sc_learn = sc if cue is None else sc - self.cue_w * cue_ov
            if self.rep > 0 and not first and len(com):
                sc = sc - self.rep * self.said_turn[com]
            best = int(np.argmax(sc)) if len(com) else -1
            s_best = float(sc[best]) if best >= 0 else 0.0
            ev['s_best'] = s_best; ev['s_end'] = s_end
            if self.end_rule == 'sum':
                comp = (1 - self.Ub) ** self.nsyl * float(np.maximum(sc_learn, 0).sum()) + \
                    (self.cue_w * float(cue_ov.max()) if cue is not None and len(cue_ov) else 0.0)
            else:
                comp = self.rho * s_best
            if not first and (endpush or (s_end > 0 and s_end >= comp)):
                self.stop = 'END'; END_new = True; ENDext = bool(endpush)
            elif not first and self.nsyl >= self.breath:
                self.stop = 'breath'
            elif best < 0 or s_best < self.theta:
                self.stop = 'floor' if not first else 'silent'
            else:
                Y_new = int(com[best]); Yext = cue is not None and bool(cue_ov[best] == 4)
                self.Hp = Y_new; self.phase = 0; self.nsyl += 1
                if first:
                    self.said_turn[:] = 0
                self.said_turn[Y_new] = 1
                ev['rel'] = Y_new; ev['first'] = first
            if self.stop is not None:
                self.Hp = -1; self.phase = 0
        elif self.SPK:
            self.phase += 1
        if self.Hp == -1 and not bnd and self.idle and not F_new and not GO_new and not R1_new and not CD_new.any():
            if self.pending_babble is None and self.babble_rate > 0 and self.rng.random() < self.babble_rate:
                self.babble()
            if self.pending_babble is not None:
                self.trial = self.pending_babble; self.pending_babble = None
                self.Hp = -2; self.phase = 0; self.n_babbles += 1
                ev['babble'] = tuple(self.trial)
        SPK_new = self.Hp != -1
        D = np.zeros(self.M_N, bool)
        if self.Hp >= 0:
            D = self.prog[self.Hp].copy()
        elif self.Hp == -2:
            D[self.trial] = True
        Mlv = D[:, None] & (np.arange(self.LV)[None, :] <= self.phase)
        force = Mlv.sum(1)
        if self.keep == 'reward' and self.last_babble is not None and reward > 0 and \
                self.t - self.babble_end_t <= self.keep_window:
            y = self.commit(self.last_babble)
            if y is not None:
                self.n_kept += 1; ev['kept'] = y
            self.last_babble = None
        self.Y_prev, self.Yext_prev, self.END_prev, self.ENDext_prev = self.Y, self.Yext, self.END, self.ENDext
        self.GO_prev = self.GO
        self.F, self.F2, self.GO, self.R1, self.CD = bool(F_new), bool(F2_new), bool(GO_new), bool(R1_new), CD_new
        self.R2 = bool(R2_new)
        self.bnd, self.first = bool(bnd), bool(first)
        self.Yh = [Y_new, self.Yh[0], self.Yh[1], self.Yh[2]]
        self.Y, self.END, self.Yext, self.ENDext = Y_new, END_new, Yext, ENDext
        self.Hc_prev, self.Hc = Hc_prev, Hc_now
        self.SPK = SPK_new
        self.D = D
        self.force = force
        self.E_prev = float(E_now)
        self.t += 1
        ev.update(F=self.F, GO=self.GO, CD=int(self.CD.sum()), SPK=SPK_new, phase=self.phase, R2=self.R2)
        if self.stop:
            ev['stop'] = self.stop
        self.last_ev = ev
        return force
