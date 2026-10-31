# -*- coding: utf-8 -*-
"""Bench physics: sound and mouth of the real model (world/voice.py, copied), plus the bench options.

Copied from /root/work/concept-brain/world/voice.py (verbatim copy kept as voice_orig.py) and from ARCH_v1's patched
copy (arch/synth/voice_copy.py = voice_v1copy.py: own-voice re-articulation onset `new_self`).

Exactness (selftest.py checks it): with physics='random' the draws for every char's sound, the 'self' offset and every
char's 4-muscle set are the original ones (same rng, same order), so a Voice built over the sorted corpus chars with
seed s is the real model's Voice(chars, seed=s).  The mouth rule (which char a muscle set produces) is the original one,
vectorised and cached.  Stream reproduces voice_v1copy.Stream bit for bit when only 'other' and 'self' speak.

Bench additions (do not change the original draws; they use their own rngs):
  * voice 'other2': a third speaker with its own fixed offset (overheard dialogue, model replies).
  * physics='lawful': each muscle m has a fixed spectro-temporal pattern P_m (peaks in 3 of the 6 slices); the sound of
    ANY active muscle set S is clip(sum_{m in S} P_m, 0, levels).  A char's sound = sound(need[c]).  Similar gestures
    sound similar.  The own voice is the sound of the set the mouth actually makes (mumbles and garbles included).
  * noise=sigma: background noise mixed (max) into every new slice column: clip(rint(N(0, sigma)), 0, levels).
"""
from __future__ import annotations

import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SOUND = 3            # ticks per char (step = T // SOUND = 2 slices per tick)
PAUSE = 5            # silent ticks after an END before the next line (API: PAUSE >= 5)
WHO = ('other', 'other2', 'self')


# ------------------------------------------------------------------------------------------ corpus
def corpus_path():
    for p in (os.environ.get('BENCH_CORPUS'), '/root/work/corpus/dialogs.jsonl',
              os.path.expanduser('~/v2lab/data/dialogs.jsonl')):
        if p and os.path.exists(p):
            return p
    return None


_POOL = None


def load_corpus():
    """dialogues with >= 2 lines, in file order (as talk.py / archv1.py)."""
    global _POOL
    if _POOL is None:
        p = corpus_path()
        if p is None:
            raise FileNotFoundError('corpus not found: set BENCH_CORPUS')
        pool = [json.loads(line) for line in open(p, encoding='utf-8')]
        _POOL = [c for c in pool if len(c) >= 2]
    return _POOL


def corpus_chars():
    """sorted char set of the corpus = the real model's Voice char list (cached in corpus_chars.json)."""
    cache = os.path.join(HERE, 'corpus_chars.json')
    if os.path.exists(cache):
        return json.load(open(cache, encoding='utf-8'))
    chars = sorted({ch for c in load_corpus() for u in c for ch in u})
    with open(cache, 'w', encoding='utf-8') as f:
        json.dump(chars, f, ensure_ascii=False)
    return chars


# ------------------------------------------------------------------------------------------ voice
class Voice:
    def __init__(self, chars, F: int = 8, T: int = 6, levels: int = 10, n_muscles: int = 32, k: int = 4,
                 peaks: int = 2, seed: int = 0, physics: str = 'random', lawful_slices: int = 3):
        # ---------------- verbatim part of world/voice.py (same draws, same order) ----------------
        rng = np.random.default_rng(seed)
        self.chars = list(chars)
        self.F, self.T, self.levels = F, T, levels
        self.n_muscles, self.k = n_muscles, k
        self.sound = {}
        for c in self.chars:
            g = rng.integers(0, 2, size=(F, T))
            for t in range(T):
                g[rng.choice(F, size=peaks, replace=False), t] = rng.integers(levels // 2, levels, size=peaks)
            self.sound[c] = g
        self.voice = {"other": np.zeros((F, T), dtype=int),
                      "self": rng.choice([-1, 0, 1], p=[0.15, 0.7, 0.15], size=(F, T))}
        self.need: dict[str, frozenset] = {}
        used = set()
        for c in self.chars:
            while True:
                s = frozenset(int(m) for m in rng.choice(n_muscles, k, replace=False))
                if s not in used:
                    used.add(s)
                    self.need[c] = s
                    break
        self._noise_rng = np.random.default_rng(seed + 1)
        self.silence = np.zeros((F, T), dtype=int)
        # ---------------- bench additions ----------------
        self.seed, self.physics = seed, physics
        r2 = np.random.default_rng(seed + 1000)
        self.voice['other2'] = r2.choice([-1, 0, 1], p=[0.15, 0.7, 0.15], size=(F, T))
        self.cid = {c: i for i, c in enumerate(self.chars)}
        self.NEED = np.zeros((len(self.chars), n_muscles), np.int16)
        for c, s in self.need.items():
            self.NEED[self.cid[c], list(s)] = 1
        self.set2char = {s: c for c, s in self.need.items()}
        if physics == 'lawful':
            rl = np.random.default_rng(seed + 2000)
            self.mpat = np.zeros((n_muscles, F, T), int)
            for m in range(n_muscles):
                for t in rl.choice(T, size=lawful_slices, replace=False):
                    self.mpat[m, rl.integers(F), t] = rl.integers(levels // 2, levels)
            self.random_sound = self.sound
            self.sound = {c: self.set_sound(self.need[c]) for c in self.chars}
        elif physics != 'random':
            raise ValueError(physics)
        self._mouth_cache = {}
        self._hear_cache = {}

    # ---------------------------------------------------------------- sounds
    def set_sound(self, active) -> np.ndarray:
        """lawful physics: the sound of any muscle set (clipped sum of the muscles' patterns)."""
        a = list(active)
        if not a:
            return self.silence.copy()
        return np.clip(self.mpat[a].sum(0), 0, self.levels)

    def hear(self, c, who: str) -> np.ndarray:
        """as world/voice.py: the grid heard when `who` says char c; None = silence; '~' = garble (fresh noise)."""
        if c is None:
            return self.silence.copy()
        if c == "~":
            return self._noise_rng.integers(0, 4, size=(self.F, self.T))
        key = (c, who)
        g = self._hear_cache.get(key)
        if g is None:
            g = np.clip(self.sound[c] + self.voice[who], 0, self.levels)
            g.setflags(write=False)
            self._hear_cache[key] = g
        return g

    def hear_set(self, active: tuple) -> np.ndarray:
        """lawful physics: own voice of an arbitrary muscle set."""
        key = (active, 'set')
        g = self._hear_cache.get(key)
        if g is None:
            g = np.clip(self.set_sound(active) + self.voice['self'], 0, self.levels)
            g.setflags(write=False)
            self._hear_cache[key] = g
        return g

    # ---------------------------------------------------------------- mouth
    def decode(self, active: tuple):
        """the original mouth rule for a sorted tuple of active muscles: None (silent), a char (clear: all k muscles of
        the char; mumbled: k-1 of them; first char in list order with the most hits), or '~' (garble)."""
        r = self._mouth_cache.get(active, 0)
        if r == 0:
            if not active:
                r = None
            elif len(active) <= self.k + 2:
                hits = self.NEED[:, list(active)].sum(1)
                b = int(np.argmax(hits))
                r = self.chars[b] if hits[b] >= self.k - 1 else '~'
            else:
                r = '~'
            self._mouth_cache[active] = r
        return r

    def mouth(self, force: dict) -> str | None:
        """original signature (dict 'm<i>' -> force)."""
        return self.decode(tuple(sorted(int(n[1:]) for n, f in force.items() if f > 0)))


# ------------------------------------------------------------------------------------------ stream
class Stream:
    """the ear window (F x T): every tick advances `step` slices; sources mix by taking the maximum.
    Sources are given per tick as (token, grid, fresh): the phase of a source restarts when its token changes or
    `fresh` is set (a new char of a partner; the own voice's re-articulation onset).  grid=None for silence.
    With sources other+self only this is voice_v1copy.Stream bit for bit."""

    def __init__(self, F, T, step, n_src=3, levels=10, noise=0.0, noise_seed=0):
        self.F, self.T, self.step, self.levels = F, T, step, levels
        self.win = np.zeros((F, T), dtype=int)
        self.tok = [None] * n_src
        self.ph = [0] * n_src
        self.noise = float(noise)
        self.nrng = np.random.default_rng(noise_seed)

    def tick(self, sources) -> np.ndarray:
        T, step = self.T, self.step
        new = None
        for i, (tok, g, fresh) in enumerate(sources):
            ph = self.ph[i]
            if tok != self.tok[i] or fresh:
                ph = 0
            if tok is not None:
                if ph + step <= T:
                    cols = g[:, ph:ph + step]
                else:                                           # held longer than the sound: stay on the last slice
                    cols = g[:, [min(ph + j, T - 1) for j in range(step)]]
                new = cols.copy() if new is None else np.maximum(new, cols)
            self.tok[i] = tok
            self.ph[i] = ph + step
        if new is None:
            new = np.zeros((self.F, step), dtype=int)
        if self.noise > 0:
            nz = np.clip(np.rint(self.nrng.normal(0.0, self.noise, (self.F, step))), 0, self.levels).astype(int)
            new = np.maximum(new, nz)
        self.win = np.concatenate([self.win[:, step:], new], axis=1)
        self.win.setflags(write=False)                  # agents get this array: read-only, a new one every tick
        return self.win
