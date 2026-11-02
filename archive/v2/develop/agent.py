# -*- coding: utf-8 -*-
"""ARCH_v2-develop agent on the shared bench: ear (exact layer 0, FEAT0) -> perceptual syllable categories (Z, formed
without a teacher) -> cognitive core (core.py) -> the kept body (bench body.py: F/GO/CD, beat, WTA, D/I/M, END, counter).

Per tick:
  1. ear: FEAT0 pattern of the window; E = window energy.
  2. other's speech (not while the own voice can be in the ear: SPK or CD): a syllable clock is phase-reset by the
     speech onset (E 0 -> >0) and samples every 3rd tick from onset+2 (the aligned window = exactly one char).
     Sample -> best category match >= theta (0.75) -> item; below -> a new category is recruited (vigilance).
     Line end = the first silent window.  Items / line ends drive the core (learning by listening).
  3. eye: the scene baseline is the eye at scene onset; extra units = the tutor's pointing code, mapped to one of
     3 attention units by competitive code learning (innate orienting to a pointing gesture).
  4. body.step(E, scorer): at each boundary the core scores items; items map 1:1 to programs bound in babbling.
Babbling (play period, offline): random 4-muscle sets; the own voice at the aligned tick is matched to the heard
categories; a match >= theta binds the set as the program of that category (inverse map), others are dropped.
"""
from __future__ import annotations

import sys

import numpy as np

import os
BENCH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bench_copy")
if BENCH not in sys.path:
    sys.path.insert(0, BENCH)
from body import Body          # noqa: E402
from frontend import Ear0, feat0   # noqa: E402
from physics import Stream     # noqa: E402

from core import Core, Cfg     # noqa: E402
from rules import Templates    # noqa: E402


class DevAgent:
    privileged = False

    def __init__(self, world, seed=0, V=256, theta_z=0.75, z_rate=0.1, learn_cats=True, **cfgkw):
        self.seed = seed
        self.ear = Ear0(seed=world.seed)
        self.F0 = feat0(Ear0(seed=world.seed), world.warmup_windows())
        self.ear.set_feat0(self.F0)
        self.NF = len(self.F0)
        self.Z = Templates(self.NF, cap=V, theta=theta_z, rate=z_rate)
        self.V = V
        self.theta_z = theta_z
        self.learn_cats = learn_cats
        cfg = Cfg(V=V, n_eye=len(world.eye), retino=False, seed=seed, **cfgkw)
        self.core = Core(cfg)
        self.body = Body(n_prog=V + 64, theta=cfg.floor, end_rule='max', rho=1.0, recruit='babble', keep='manual',
                         seed=seed)
        self.cat2prog, self.prog2cat = {}, {}
        self.bab_pat = np.zeros((0, self.NF), bool)        # babbling memory: own-voice sound -> muscle set
        self.bab_set = []
        self.imitate = True                                # new heard syllables get programs from the babbling memory
        self.theta_bind = 0.85
        # line segmentation
        self.E_prev = 0.0
        self.in_line = False
        self.clk = 0
        self.st = None
        # eye parsing
        self.base = None
        self.att_codes = []
        self.scene, self.att = None, None
        self.own_st = None
        self.learning = True
        self.stats = dict(items=0, new_cats=0, lines=0, own=0, babbles=0, bound=0)
        self.det_log = []          # (tick, category) of detections (for audits)
        self.t = 0
        self.mute = False

    # ---------------------------------------------------------------- eye
    def _parse_eye(self, eye):
        e = np.asarray(eye).astype(bool)
        if not e.any():
            self.base = None
            return None, None
        if self.base is None:
            self.base = e.copy()
        extra = e & ~self.base
        missing = self.base & ~e
        if missing.sum() > 2:                    # the scene itself changed: new baseline
            self.base = e.copy(); extra[:] = False
        att = None
        if extra.sum() >= 4:
            best, bj = 0.0, -1
            for j, c in enumerate(self.att_codes):
                jac = (c & extra).sum() / max((c | extra).sum(), 1)
                if jac > best:
                    best, bj = jac, j
            if best >= 0.6:
                att = bj
            elif len(self.att_codes) < 3:
                self.att_codes.append(extra.copy()); att = len(self.att_codes) - 1
        return self.base, att

    # ---------------------------------------------------------------- perception
    def _detect(self, f):
        j, m = self.Z.best(f)
        if j >= 0 and m >= self.theta_z:
            if self.learning and self.learn_cats:
                self.Z.learn(j, f.astype(np.float32))
            return j
        if self.learning and self.learn_cats and self.Z.n < self.V and f.sum() > 40:
            self.stats['new_cats'] += 1
            j = self.Z.new(f.astype(np.float32))
            if self.imitate and len(self.bab_set):
                self._bind_from_memory(j, f)
            return j
        return None

    def _bind_from_memory(self, j, f):
        """imitation through the inverse map: the babbled set whose own voice matched this heard syllable"""
        m = (self.bab_pat & f[None, :]).sum(1) / np.maximum(np.maximum(self.bab_pat.sum(1), f.sum()), 1)
        k = int(np.argmax(m))
        if m[k] >= self.theta_bind and j not in self.cat2prog:
            y = self.body.commit(self.bab_set[k])
            if y is not None:
                self.cat2prog[j] = y; self.prog2cat[y] = j; self.stats['imitated'] = self.stats.get('imitated', 0) + 1

    def _own_voice(self):
        return self.body.SPK or bool(self.body.CD.any())

    # ---------------------------------------------------------------- scorer for the body
    def _scorer(self, body, com, first):
        if self.mute:
            return np.full(len(com), -1.0), 0.0
        if first or self.own_st is None:
            self.own_st = self.core.own_begin(self.scene, self.att)
        prog, s_end = self.core.own_scores(self.own_st)
        sc = np.array([prog[self.prog2cat[y]] if y in self.prog2cat else -1.0 for y in com])
        return sc, s_end

    # ---------------------------------------------------------------- one tick
    def step(self, obs):
        win = obs['ear']
        E = float(np.asarray(win).sum())
        s = self.ear.step(win)
        f = s[self.F0]
        self.scene, self.att = self._parse_eye(obs['eye'])
        core = self.core
        if not self._own_voice():
            if E > 0 and self.E_prev == 0 and not self.in_line:
                self.in_line = True; self.clk = 0
                core.heard_onset(t_now=self.t)
                self.st = core.begin_line(self.scene, self.att)
            elif self.in_line and E > 0:
                self.clk += 1
            if self.in_line and E > 0 and self.clk % 3 == 2:
                c = self._detect(f)
                if c is not None:
                    self.stats['items'] += 1
                    self.det_log.append((self.t, c))
                    if self.st['att_tutor'] is None and self.att is not None:
                        self.st['att_tutor'] = self.att      # pointing started after the onset tick
                    core.line_item(self.st, c, learn=self.learning)
            if self.in_line and E == 0:
                self.in_line = False
                core.line_end(self.st, learn=self.learning, store=self.learning)
                self.stats['lines'] += 1
                self.st = None
        elif self.in_line:                                   # the agent started speaking over a heard line: close it
            self.in_line = False
            core.line_end(self.st, learn=self.learning, store=self.learning)
            self.st = None
        force = self.body.step(E, scorer=self._scorer, reward=obs.get('reward', 0.0))
        b = self.body
        if b.Y >= 0 and self.own_st is not None:
            c = self.prog2cat.get(b.Y)
            if c is not None:
                core.own_advance(self.own_st, c); self.stats['own'] += 1
        if b.stop is not None and b.stop != 'babble':
            if self.own_st is not None:
                core.own_stop(self.own_st, b.stop == 'END')
                core.own_finish(self.own_st, t_end=self.t)
            self.own_st = None
        r = obs.get('reward', 0.0)
        if r:
            core.reward(r)
        self.E_prev = E
        self.t += 1
        return force.astype(np.float32)

    # ---------------------------------------------------------------- babbling (play period)
    def babble(self, voice, n, rng=None, log_every=0):
        """n random 4-muscle babbles; the own voice at the aligned tick is matched to the heard categories."""
        rng = rng or np.random.default_rng(self.seed + 31)
        cache = {}
        hist = []
        pats, sets = [], []
        seen = set()
        import itertools
        allsets = list(itertools.combinations(range(32), 4))
        order = rng.permutation(len(allsets))                  # each muscle set tried once (no repeats)
        for i in range(1, n + 1):
            m = allsets[order[(i - 1) % len(allsets)]]
            said = voice.decode(m)
            self.stats['babbles'] += 1
            if said is not None and said != '~':
                f = cache.get(said)
                if f is None:
                    f = cache[said] = self.own_pattern(voice, said)
                if said not in seen:                           # remember every producible sound once (inverse map)
                    seen.add(said); pats.append(f); sets.append(m)
            if log_every and i % log_every == 0:
                P = np.array(pats, bool)
                cov = 0
                if len(P):
                    for jj in range(self.Z.n):
                        t = self.Z.W[:, jj] >= 0.5
                        cov += int(((P & t[None, :]).sum(1) / np.maximum(P.sum(1), 1)).max() >= self.theta_bind)
                hist.append((i, cov))
        if pats:
            self.bab_pat = np.concatenate([self.bab_pat, np.array(pats, bool)], 0)
            self.bab_set += sets
        self.stats['babble_sounds'] = len(self.bab_set)
        for j in range(self.Z.n):                              # bind every heard category to its best babbled sound
            if j not in self.cat2prog:
                self._bind_from_memory(j, self.Z.W[:, j] >= 0.5)
        self.stats['bound'] = len(self.cat2prog)
        return hist

    def own_pattern(self, voice, ch):
        """the exact FEAT0 pattern of the own voice saying ch, at its aligned tick (fresh stream, as heard)."""
        st = Stream(voice.F, voice.T, voice.T // 3, 3, voice.levels)
        e = self._ear_copy()
        none = (None, None, False)
        g = voice.hear(ch, 'self')
        out = None
        for k in range(3):
            win = st.tick((none, none, (ch, g, k == 0)))
            out = e.step(win)
        return out[self.F0]

    def _ear_copy(self):
        e = Ear0.__new__(Ear0)
        e.__dict__.update(self.ear.__dict__)
        e.reset()
        return e
