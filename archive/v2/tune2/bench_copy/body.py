# -*- coding: utf-8 -*-
"""ARCH_v1's body as a reusable module (extracted from arch/synth/archv1.py Brain.step; same tick algebra).

Turn units (fixed weights, inputs from t-1):
  F   = [0.01 * E(t-1) - 10 * sum CD(t-1) >= 0.1]          other is talking (E = energy of the ear input window)
  F2  = F(t-1);  GO = [F2 - 2F - 2SPK - 2 sum CD >= 1]      = offset of F: one pulse 4 ticks after a line
  CD  = 4-unit delay line driven by GO | SPK               corollary discharge (own voice hidden from F)
  R1  = GO(t-1);  R2 = GO(t-1) | (SPK(t-1) & phase(t-1) = 1)   R2 = the tick before every boundary (read-out clock)
  boundary = (R1(t-1) & !F(t-1)) | (SPK(t-1) & phase(t-1) = 2) first boundary = GO+2 (blocked if the other talks)
Programs (syllable units): prog[y] = a 4-muscle set.  At a boundary the MIND gives a score per committed program and
for END; the body adds the teacher cue (cue_w per matching muscle), recruits (v1: a cue no program fully matches), applies
the use penalty `rep`, then: END (teacher push, or s_end > 0 and s_end >= comp) -> stop; breath counter >= 12 -> stop;
best < theta -> stop ('floor' / 'silent' at the first boundary); else release the best: Y pulse, hold Hp 3 ticks,
D = prog[Hp], I = !D, M[m,l] = D[m] & l <= phase  ->  force ramp 1-2-3 (same tick).  Hc(t) = {Y(t-2), Y(t-3)} (late
one-hot copy, the source for next-syllable / END learning).

    body = Body(seed=0)
    force = body.step(E_now, scorer=mind_scores, cue=cue_set, endpush=flag, reward=r)
        scorer(body, com, first) -> (scores over com, s_end)    called only at boundary ticks
    after step: body.F, F2, GO, R1, R2, CD, SPK, phase, Hp, nsyl, bnd, first, Y (pulse id at t or -1), Yext (released
    under full cue drive), Y_prev / Yext_prev (t-1), END, ENDext, END_prev, ENDext_prev, Hc (ids at t), Hc_prev,
    stop (reason at t or None), recruited (id or None), GO_prev (GO at t-1, for gates like P = T & GO).

Teacher-free program creation (babbling):  body.babble_rate (per idle tick) or body.babble(set) starts a one-syllable
babble with a random (or given) 4-muscle set; it uses the same D/I/M/CD path (3 ticks, force 1-2-3).  With
keep='reward' the set becomes a program if a reward > 0 arrives within keep_window ticks after the babble (the tutor
imitated it = it was heard as a char); keep='manual': the mind calls body.commit(set).  recruit: 'cue' (v1), 'babble'
(no cue recruitment) or 'both'.
"""
from __future__ import annotations

import numpy as np


class Body:
    M_N, LV = 32, 3

    def __init__(self, n_prog=2048, cue_w=10.0, end_push=100.0, theta=0.02, rho=1.0, end_rule='max', Ub=0.1,
                 rep=0.0, breath=12, f_w=0.01, f_thr=0.1, cd_len=4, recruit='cue', babble_rate=0.0, keep='reward',
                 keep_window=15, seed=0):
        self.NP, self.cue_w, self.end_push, self.theta, self.rho = n_prog, cue_w, end_push, theta, rho
        self.end_rule, self.Ub, self.rep, self.breath = end_rule, Ub, rep, breath
        self.f_w, self.f_thr, self.cd_len = f_w, f_thr, cd_len
        self.recruit, self.babble_rate, self.keep, self.keep_window = recruit, babble_rate, keep, keep_window
        self.rng = np.random.default_rng(seed + 991)
        self.prog = np.zeros((n_prog, self.M_N), bool)
        self.committed = np.zeros(n_prog, bool)
        self.n_prog = 0
        self.set2prog = {}
        # turn units
        self.F = False; self.F2 = False; self.GO = False; self.R1 = False; self.R2 = False
        self.CD = np.zeros(cd_len, bool)
        self.E_prev = 0.0
        # motor
        self.Hp = -1; self.phase = 0; self.SPK = False; self.nsyl = 0
        self.said_turn = np.zeros(n_prog, np.float32)
        self.Yh = [-1, -1, -1, -1]            # Y(t-1..t-4) during step; after step Yh[0] = Y(t)
        self.Y = -1; self.END = False; self.Yext = False; self.ENDext = False
        self.Y_prev = -1; self.Yext_prev = False; self.END_prev = False; self.ENDext_prev = False
        self.Hc = np.zeros(0, np.int64); self.Hc_prev = np.zeros(0, np.int64)
        self.force = np.zeros(self.M_N, int)
        self.bnd = False; self.first = False; self.stop = None; self.recruited = None; self.GO_prev = False
        self.t = 0
        # babbling
        self.trial = None; self.pending_babble = None; self.last_babble = None; self.babble_end_t = None
        self.n_babbles = 0; self.n_kept = 0
        self.last_ev = {}

    # ------------------------------------------------------------------ programs
    def commit(self, muscles):
        """make a muscle set a program (returns its id; existing id if already committed)."""
        s = frozenset(int(m) for m in muscles)
        y = self.set2prog.get(s)
        if y is not None:
            return y
        if self.n_prog >= self.NP:
            return None
        y = self.n_prog; self.n_prog += 1
        self.prog[y, list(s)] = True; self.committed[y] = True
        self.set2prog[s] = y
        return y

    def babble(self, muscles=None):
        """request a one-syllable babble at the next idle tick (random 4-muscle set if None)."""
        if muscles is None:
            muscles = self.rng.choice(self.M_N, 4, replace=False)
        self.pending_babble = np.asarray(sorted(int(m) for m in muscles))

    def _hc_at(self, lag):
        return np.array(sorted({y for y in self.Yh[1 + lag:3 + lag] if y >= 0}), dtype=np.int64)

    @property
    def idle(self):
        return (not self.SPK) and (not self.F) and (not self.CD.any()) and (not self.GO) and (not self.R1) \
            and self.Hp == -1

    # ------------------------------------------------------------------ one tick
    def step(self, E_now, scorer=None, cue=None, endpush=False, reward=0.0):
        """E_now: energy of the ear window at t (read by F at t+1); cue: teacher muscle set (iterable of 4) or None;
        endpush: teacher END push.  Returns force (32,) int (0..3) at t."""
        ev = {}
        ncd = int(self.CD.sum())
        F_new = (self.f_w * self.E_prev - 10.0 * ncd) >= self.f_thr
        F2_new = self.F
        GO_new = (1.0 * self.F2 - 2.0 * self.F - 2.0 * self.SPK - 2.0 * ncd) >= 1.0
        CD_new = np.r_[self.GO or self.SPK, self.CD[:-1]]
        R1_new = self.GO
        R2_new = self.GO or (self.SPK and self.phase == 1)
        bnd = (self.R1 and not self.F) or (self.SPK and self.phase == 2)
        Hc_prev = self._hc_at(1)
        Hc_now = self._hc_at(0)
        Y_new, END_new, Yext, ENDext = -1, False, False, False
        first = bnd and not self.SPK
        self.stop = None; self.recruited = None
        if bnd and self.Hp == -2:                          # end of a babble: no selection, just stop
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
        # babble start (idle only, never inside a turn: no other speech, no corollary discharge, no pending GO)
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
        # keep a babble that the partner answered (reward) -> program
        if self.keep == 'reward' and self.last_babble is not None and reward > 0 and \
                self.t - self.babble_end_t <= self.keep_window:
            y = self.commit(self.last_babble)
            if y is not None:
                self.n_kept += 1; ev['kept'] = y
            self.last_babble = None
        # ---------------- commit (state t)
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
