# -*- coding: utf-8 -*-
"""AgentH = ARCH_v2.1 (AgentV21) + the open-world hearing front end (hear.py).  Every h_* key at its default = v2.1
bit for bit (the class then only forwards to AgentV21.step).  Only the front end and its triggers change; the PFC side
(core_v21) is untouched, so this merges as a wrapper over whatever AgentV21 becomes (v22).

Keys (agent level):
  h_gate=1     SoundGate: adaptive noise floor + constant-false-alarm detector on the fresh ear slices; the agent reads
               the window only while the gate says "sound" (else zeros).  All existing triggers (line onset/end,
               syllable clock, quiet moment, wandering, the brake F, the eye parser's silence test) become
               floor-relative without code changes.  Params h_kon 7, h_koff 4, h_tau 500, h_trim 4, h_stuck 600.
  h_back=1     onset back-tracking: a run whose previous pair already exceeded mu + h_kback*nu (5.5) restarts the
               syllable clock one tick earlier (post-step clk correction).
  h_relock=1   the syllable clock follows recognition: at a sample tick whose window is not recognised (< theta_z),
               if one of the two previous in-line windows (the other phases) was recognised with m >= h_lock (0.85),
               that window is the item and the clock relocks to its phase.
  h_vbound=1   (with h_relock) a relock ends the current line (another speaker's rhythm): the line ends, a new line
               starts at once with the relocked item as its first item.
  h_nocreate=1 (with h_relock) no new syllable categories in a line that showed a second rhythm (match only).
  h_level=1    level-invariant recognition: a window that matches no category is compared again 1-3 levels louder
               (signal cells only); a match >= h_lvth (0.85) is that category (a farther instance of a known syllable:
               world2's distance subtracts levels).  No template update from shifted matches.
  h_denoise=1  (with h_gate) syllable patterns (detection, relock, level matching) are computed from the window with
               background-level cells silenced (cells <= ceil(5 x the mean background per cell); 0 = nothing on the
               bench); the triggers still read the gated raw window.
  h_conf=1     (with h_gate) a new syllable category is recruited only from a window whose 3 fresh pairs were all
               clearly heard (above the onset threshold); faint (far, noisy) windows are matched only.  On the bench
               every sample window qualifies (threshold 0, every speech pair e2 >= 21).
Perception only; nothing here reads world state (warm-up windows at birth = the allowed calibration).
"""
from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
V21 = os.path.join(HERE, '..', 'v21')
SYN = os.path.join(HERE, '..', 'synth')   # repo: v21copy = ../v21 + the unchanged ARCH_v2 core in ../synth
for p in (SYN, V21, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)
from agent_v21 import AgentV21, V21_AGENT_KEYS          # noqa: E402
from agent_synth import SynthAgent                        # noqa: E402
from hear import SoundGate, cell_gate, level_shifts      # noqa: E402
from physics import Stream                                # noqa: E402  (../synth/bench_copy, on the path via agent)

H_KEYS = {'h_gate': int, 'h_back': int, 'h_relock': int, 'h_vbound': int, 'h_nocreate': int, 'h_conf': int,
          'h_level': int, 'h_denoise': int, 'h_kon': float, 'h_koff': float, 'h_kback': float, 'h_tau': float, 'h_trim': float,
          'h_stuck': int, 'h_lock': float, 'h_lvth': float}


class AgentH(AgentV21):
    def __init__(self, world, seed=0, h_gate=0, h_back=0, h_relock=0, h_vbound=0, h_nocreate=0, h_conf=0, h_level=0,
                 h_denoise=0,
                 h_kon=7.0, h_koff=4.0, h_kback=5.5, h_tau=500.0, h_trim=4.0, h_stuck=600, h_lock=0.85, h_lvth=0.85,
                 **kw):
        super().__init__(world, seed=seed, **kw)
        self.gate = None
        if h_gate:
            self.gate = SoundGate(k_on=h_kon, k_off=h_koff, tau=h_tau, stuck=h_stuck, backtrack=h_back, k_trim=h_trim,
                                  k_back=h_kback)
            self.gate.calibrate(world.warmup_windows())
        self.h_back, self.h_relock, self.h_vbound, self.h_nocreate, self.h_lock = \
            int(h_back), int(h_relock), int(h_vbound), int(h_nocreate), float(h_lock)
        self.h_conf, self.h_level, self.h_lvth = int(h_conf), int(h_level), float(h_lvth)
        self.h_denoise = int(h_denoise)
        self.h_on = bool(h_gate or h_relock)
        self.hstats = dict(relocks=0, vbounds=0, nocreate_skips=0, unclear_skips=0, level_matches=0, backtracked=0,
                           gated_ticks=0, max_run=0)
        self._hist = []                 # (t, cat, match, FEAT0) of the last 2 in-line windows (the other two phases)
        self._mixed = False             # this line showed a second rhythm
        self._force = None              # (cat, FEAT0) the relocked item, consumed by _detect
        self._zero = None

    # ---------------------------------------------------------------- detection hooks
    def _g(self):
        return cell_gate(self.gate) if (self.h_denoise and self.gate is not None) else 0

    def _feat(self, win):
        """FEAT0 of the window; with h_denoise, cells at background level (<= the cell gate) are silenced first."""
        g = self._g()
        if g > 0:
            win = np.where(np.asarray(win) > g, win, 0)
        return self.ear.respond(self.ear.encode(win))[self.F0]

    def _detect(self, f):
        if self._force is None and self._g() > 0 and self._win is not None:
            f = self._feat(self._win)                   # denoised pattern (the triggers still read the gated window)
        if self._force is not None:                     # the relocked window is the item of this sample
            j, fb = self._force
            self._force = None
            if self.learning and self.learn_cats:
                self.Z.learn(j, fb.astype(np.float32))
            if self.rebind and j not in self.cat2prog and len(self.bab_set) and self.learning:
                self._bind_from_memory(j, fb)
                if j in self.cat2prog:
                    self.stats['rebound'] += 1
            return j
        j, m = self.Z.best(f)
        if j >= 0 and m >= self.theta_z:
            return AgentV21._detect(self, f)            # recognised: the v2.1 path
        if self.h_level and self.Z.n and self._win is not None:
            jl = self._level_match()                    # a quieter (farther) instance of a known syllable
            if jl is not None:
                return jl
        unclear = self.h_conf and self.gate is not None and not self.gate.conf
        if (self._mixed and self.h_nocreate) or unclear:   # overlapped or faint speech: match only, never recruit
            self.hstats['unclear_skips' if unclear else 'nocreate_skips'] += 1
            return None
        return AgentV21._detect(self, f)                # unrecognised: v2.1 recruits a new category

    # ---------------------------------------------------------------- the inverse map under denoising
    def babble(self, voice, n, rng=None, log_every=0):
        hist = AgentV21.babble(self, voice, n, rng=rng, log_every=log_every)
        if self.h_denoise:                       # keep the own-voice windows (the raw auditory memory of babbling)
            self._bab_win = np.stack([self._own_window(voice, voice.decode(m)) for m in self.bab_set]) \
                if self.bab_set else None
            self._bab_g = {}
        return hist

    def _own_window(self, voice, ch):
        st = Stream(voice.F, voice.T, voice.T // 3, 3, voice.levels)
        none = (None, None, False)
        g = voice.hear(ch, 'self')
        win = None
        for k in range(3):
            win = st.tick((none, none, (ch, g, k == 0)))
        return np.array(win, np.int64)

    def _bind_from_memory(self, j, f):
        """with denoising, a heard syllable is compared with the own-voice memories denoised the same way (compare like
        with like: a denoised heard pattern lacks the background bits a clean own pattern has)."""
        g = self._g()
        W = getattr(self, '_bab_win', None)
        if g <= 0 or W is None:
            return AgentV21._bind_from_memory(self, j, f)
        P = self._bab_g.get(g)
        if P is None:
            P = np.stack([self.ear.respond(self.ear.encode(np.where(w > g, w, 0)))[self.F0] for w in W])
            self._bab_g = {g: P}
        m = (P & f[None, :]).sum(1) / np.maximum(np.maximum(P.sum(1), f.sum()), 1)
        k = int(np.argmax(m))
        if m[k] >= self.theta_bind and j not in self.cat2prog:
            y = self.body.commit(self.bab_set[k])
            if y is not None:
                self.cat2prog[j] = y; self.prog2cat[y] = j
                self.stats['imitated'] = self.stats.get('imitated', 0) + 1
                self.hstats['bound_denoised'] = self.hstats.get('bound_denoised', 0) + 1

    def _level_match(self):
        best = (-1, 0.0)
        win = self._win
        g = cell_gate(self.gate)
        if self._g() > 0:
            win = np.where(np.asarray(win) > g, win, 0)
        for k, w2 in level_shifts(win, g):
            jj, mm = self.Z.best(self.ear.respond(self.ear.encode(w2))[self.F0])
            if mm > best[1]:
                best = (jj, mm)
        if best[1] >= self.h_lvth and best[0] >= 0:
            self.hstats['level_matches'] += 1
            return best[0]
        return None

    # ---------------------------------------------------------------- one tick
    def step(self, obs):
        if not self.h_on:
            return AgentV21.step(self, obs)
        win = obs['ear']
        shift = 0
        if self.gate is not None:
            S, shift = self.gate.step(win)
            self.hstats['max_run'] = max(self.hstats['max_run'], self.gate.run)
            if not S:
                if self._zero is None or self._zero.shape != np.shape(win):
                    self._zero = np.zeros(np.shape(win), np.asarray(win).dtype)
                obs = dict(obs); obs['ear'] = self._zero
                self.hstats['gated_ticks'] += int(float(np.asarray(win).sum()) > 0)
        E = float(np.asarray(obs['ear']).sum())
        was_line = self.in_line
        post_clk, item_after = None, None
        if self.h_relock and not self._own_voice():
            if self.in_line and E > 0:
                f = self._feat(obs['ear'])
                j, m = self.Z.best(f)
                relocked = False
                if (self.clk + 1) % 3 == 2 and m < self.theta_z and self._hist:
                    tb, jb, mb, fb = max(self._hist, key=lambda h: h[2])
                    if mb >= self.h_lock and jb >= 0:            # syllables are recognised at another phase
                        d = self.t - tb
                        self.hstats['relocks'] += 1
                        if self.h_vbound:                         # a new speaker: end this line, start a new one now
                            self._end_line_now()
                            self.E_prev = 0.0                    # -> the step below sees an onset at this tick
                            item_after = (jb, fb)
                            post_clk = 0 if d == 1 else 1
                            self.hstats['vbounds'] += 1
                        else:
                            self._force = (jb, fb)
                            post_clk = 0 if d == 1 else 1
                        self._mixed = True
                        relocked = True
                self._hist = [] if relocked else (self._hist + [(self.t, j, m, f)])[-2:]
            elif not self.in_line:
                self._hist = []
        mixed_keep = self._mixed
        force = AgentV21.step(self, obs)
        onset_now = self.in_line and not was_line
        if onset_now:
            self._mixed = mixed_keep if item_after is not None else False
            if item_after is None:
                self._hist = []
        if item_after is not None and self.in_line and self.st is not None:
            jb, fb = item_after
            self._force = (jb, fb)
            c = self._detect(fb)
            if c is not None:
                self.stats['items'] += 1
                if self.st['att_tutor'] is None and self.att is not None:
                    self.st['att_tutor'] = self.att
                self.core.line_item(self.st, c, learn=self.learning)
        if post_clk is not None and self.in_line:
            self.clk = post_clk
        if shift and onset_now and item_after is None and self.clk == 0:
            self.clk = shift                                     # back-tracked onset: the clock started one tick earlier
            self.hstats['backtracked'] += 1
        self._force = None
        if self.t % 500 == 0:                           # export the hearing counters with the core's (run JSONs)
            self.core.stats.update({'h_' + k: v for k, v in self.hstats.items()})
            if self.gate is not None:
                self.core.stats.update(h_mu=round(self.gate.mu, 3), h_nu=round(self.gate.nu, 3),
                                       h_stuck=self.gate.stats['stuck_updates'])
        return force

    def _end_line_now(self):
        """the line-end block of ThinkAgent.step (another speaker's line ended: learn, store, open a reply thought)."""
        core = self.core
        self.in_line = False
        core.line_end(self.st, learn=self.learning, store=self.learning)
        self.stats['lines'] += 1
        self.st = None
        if self.voice is not None:
            vs = [v for v in self._line_voices if v >= 0]
            maj = max(set(vs), key=vs.count) if vs else -1
            if len(self.voice_log) < 200000:
                self.voice_log.append((self.t, maj, len(self._line_voices)))
            if core.last_rec is not None:
                core.last_rec['voice'] = maj
            self._line_voices = []
        self._open('reply')


def typed_h(k, v, base_typed):
    if k in H_KEYS:
        return H_KEYS[k](v)
    return base_typed(k, v)
