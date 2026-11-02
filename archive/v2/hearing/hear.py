# -*- coding: utf-8 -*-
"""Open-world hearing front end for ARCH_v2.1 (behind config keys; every key off = v2.1 bit for bit).

SoundGate: an adaptive noise floor + a constant-false-alarm sound detector on the FRESH ear slices.
  e2(t)   = energy of the 2 newest slices of the ear window (16 cells; the window = the last 3 fresh pairs)
  mu, nu  = running mean and mean absolute deviation of e2 over BACKGROUND ticks (no detected sound in the window,
            and e2 <= mu + k_trim*nu: a trimmed estimate, so faint far speech below the threshold cannot inflate it),
            leaky integrators with time constant tau; initialised once at birth from the warm-up stream's silent
            windows (the world's only allowed calibration input); if sound seems to last > stuck ticks without a
            break (no line is that long) the floor also adapts during sound (a rising background).
  s(t)    = e2(t) > mu + k_on * nu   (k_off instead while the previous pair was sound: hysteresis)
  S(t)    = s(t) or s(t-1) or s(t-2)  ("the window holds sound")
  gated window / energy: win_eff = win if S else 0, E_eff = sum(win_eff).  Every v2.1 trigger (line onset E>0 after
  E==0, syllable clock, line end E==0, quiet moment E==0, wandering, the body's brake F = E>=10, the eye parser's
  silence test) reads E_eff, so no trigger code changes.
  In a silent background (the bench: warm-up silence is exactly 0) mu = nu = 0, the threshold is 0 and
  s(t) <=> e2(t) > 0, S(t) <=> E(t) > 0, win_eff == win: the bench is reproduced bit for bit (PROVED; test_equiv_h.py).
  conf    = all 3 pairs of the window exceed the ONSET threshold (clearly heard; used to allow new categories).
            In a silent background every sample window of a line is conf (each pair holds speech, e2 >= 21 > 0).
  onset back-tracking (backtrack=1): a run that starts at t whose previous pair already exceeded the low threshold
  mu + k_off*nu started at t-1 (the syllable clock is phase-reset there).  Never fires in a silent background.

Other keys are documented where they are implemented (agent_h.py).
"""
from __future__ import annotations

import numpy as np


class SoundGate:
    def __init__(self, k_on=7.0, k_off=4.0, tau=500.0, stuck=600, backtrack=1, nu_min=0.0, k_trim=4.0, k_back=None):
        self.k_on, self.k_off, self.alpha, self.stuck = float(k_on), float(k_off), 1.0 / float(tau), int(stuck)
        self.backtrack = int(backtrack)
        self.k_back = float(k_off if k_back is None else k_back)   # back-track if e2(t-1) > mu + k_back*nu
        self.nu_min = float(nu_min)
        self.k_trim = float(k_trim)     # background updates ignore pairs above mu + k_trim*nu (faint far speech must not
                                        # inflate the floor: without trimming, crowds + noise ran away to mu ~ 15-20)
        self.mu = 0.0
        self.nu = 0.0
        self.s1 = self.s2 = False          # s(t-1), s(t-2)
        self.c1 = self.c2 = False          # e2 above the ONSET threshold at t-1, t-2 (no hysteresis): 'clearly heard'
        self.conf = False                  # all 3 pairs of the current window clearly heard
        self.e2_prev = 0.0
        self.run = 0                       # consecutive ticks with S
        self.n_bg = 0
        self.stats = dict(ticks=0, S=0, onsets=0, backtracked=0, stuck_updates=0)
        self.last = dict(s=False, S=False, onset_shift=0, e2=0.0, thr=0.0)

    # ---------------------------------------------------------------- calibration at birth
    def calibrate(self, windows):
        """birth calibration from the warm-up stream: background = windows with no char sound in them (the 4 silent
        ticks after each 3-tick warm-up char are silent from the voice; in a noisy world they hold only noise).  Uses
        the low half of the fresh-pair energies (a pair holding a char is far louder: e2 >= 21 for every char)."""
        e2 = np.array([float(np.asarray(w)[:, -2:].sum()) for w in windows])
        if len(e2) == 0:
            return
        bg = e2[e2 <= np.percentile(e2, 50)]            # 4 of every 7 warm-up pairs are silent (or noise only)
        if len(bg) == 0 or not bg.any():
            self.mu = self.nu = 0.0
            return
        self.mu = float(bg.mean())
        self.nu = float(np.abs(bg - self.mu).mean())

    def thresholds(self):
        nu = max(self.nu, self.nu_min) if self.mu > 0 else self.nu
        return self.mu + self.k_on * nu, self.mu + self.k_off * nu

    # ---------------------------------------------------------------- one tick
    def step(self, win):
        w = np.asarray(win)
        e2 = float(w[:, -2:].sum())
        thr_on, thr_off = self.thresholds()
        s = e2 > (thr_off if self.s1 else thr_on)
        S = s or self.s1 or self.s2
        shift = 0
        thr_back = self.mu + self.k_back * self.nu
        if s and not self.s1 and not self.s2 and self.backtrack and self.e2_prev > thr_back and self.e2_prev > 0:
            shift = 1                                   # the run began one pair earlier, below the onset threshold
            self.stats['backtracked'] += 1
        if s and not self.s1:
            self.stats['onsets'] += 1
        self.run = self.run + 1 if S else 0
        if not S:
            if self.nu == 0.0 or e2 <= self.mu + self.k_trim * self.nu:
                self._update(e2)
        elif self.run > self.stuck:                     # a background that rose: adapt during 'sound' as well
            self._update(e2)
            self.stats['stuck_updates'] += 1
        c = e2 > thr_on
        self.conf = c and self.c1 and self.c2
        self.c2, self.c1 = self.c1, c
        self.s2, self.s1 = self.s1, s
        self.e2_prev = e2
        self.stats['ticks'] += 1
        self.stats['S'] += int(S)
        self.last = dict(s=s, S=S, onset_shift=shift, e2=e2, thr=thr_on)
        return S, shift

    def _update(self, e2):
        a = self.alpha
        if self.mu == 0.0 and self.nu == 0.0 and e2 == 0.0:
            return                                      # exact silence keeps an exact zero floor
        self.mu += a * (e2 - self.mu)
        self.nu += a * (abs(e2 - self.mu) - self.nu)
        self.n_bg += 1

    def gate(self, win):
        """-> (window the rest of the agent reads, E_eff, onset_shift)"""
        S, shift = self.step(win)
        if S:
            w = np.asarray(win)
            return w, float(w.sum()), shift
        return None, 0.0, 0


def cell_gate(gate):
    """per-cell background level: a cell counts as signal above ceil(5 x the mean background level per cell)
    (0 in a silent background; 1 at sigma 0.5, 2 at sigma 1, 4 at sigma 2)."""
    if gate is None or gate.mu <= 0:
        return 0
    return int(np.ceil(5.0 * gate.mu / 16.0))


def level_shifts(win, g, kmax=3, levels=10):
    """the window heard k = 1..kmax levels louder (signal cells only): world2 attenuation subtracts levels, so this is
    the inverse of distance (a gain change in a log-level code)."""
    w = np.asarray(win)
    sig = w > g
    for k in range(1, kmax + 1):
        yield k, np.where(sig, np.minimum(w + k, levels), w)
