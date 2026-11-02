# -*- coding: utf-8 -*-
"""Runner helpers: a diagnostic World (records the true char per tick for MEASUREMENT only; the agent never sees it),
the babbling phase (stage A play without a tutor, inside the world's own tick), and reports."""
from __future__ import annotations

import collections
import sys
import time

import numpy as np

import os
BENCH = os.environ.get('PRED_BENCH', __import__('os').path.join(__import__('os').path.dirname(__import__('os').path.abspath(__file__)), 'bench_snapshot'))
HERE = os.environ.get('PRED_HERE', __import__('os').path.dirname(__import__('os').path.abspath(__file__)))
for p in (BENCH, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)
import world as W  # noqa: E402
from agent import PredAgent, Cfg  # noqa: E402


class DiagWorld(W.World):
    """the bench World + a record of what each other speaker said at each tick (diagnostics only)."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.truth = {}

    def _tick(self, other=None, other2=None, cue=None, reward=0.0):
        if other is not None:
            self.truth[self.t] = ('other', other[0])
        elif other2 is not None:
            self.truth[self.t] = ('other2', other2[0])
        return super()._tick(other=other, other2=other2, cue=cue, reward=reward)

    def babble(self, agent, n_ticks, bout=(1, 4)):
        """stage A play without a tutor: silence from others, the agent babbles on its own beat."""
        self.agent = agent
        agent.babbling = True
        agent.bab_bout = bout
        for _ in range(n_ticks):
            self._tick()
        agent.babbling = False
        agent.bab_plan = []
        for _ in range(12):
            self._tick()


def new_agent(world, cfg=None):
    a = PredAgent(cfg or Cfg(seed=world.seed))
    a.attach(world.warmup_windows())
    return a


def z_report(agent, world, lexicon=None):
    """purity of the agent's perceptual units against the true chars (sampled event ticks)."""
    tab = collections.defaultdict(collections.Counter)
    n = 0
    for t, who, z in (agent.log_events or []):
        if who != 'other' or z == 'END':
            continue
        tr = world.truth.get(t)
        if tr is None:
            continue
        tab[z][tr[1]] += 1; n += 1
    maj = {z: cnt.most_common(1)[0] for z, cnt in tab.items()}
    pure = sum(v for _, v in maj.values()) / max(n, 1)
    char2z = collections.defaultdict(set)
    for z, (ch, v) in maj.items():
        char2z[ch].add(z)
    lex = sorted(lexicon) if lexicon is not None else sorted(char2z)
    covered = [ch for ch in lex if ch in char2z]
    with_prog = [ch for ch in lex if any(agent.z_prog[z] is not None for z in char2z.get(ch, ()))]
    # does each program say its unit's char? (mouth decode of the program's muscle set)
    prog_ok = 0; prog_n = 0
    for z in range(agent.nz):
        pg = agent.z_prog[z]
        if pg is None or z not in maj:
            continue
        prog_n += 1
        prog_ok += int(world.voice.decode(tuple(sorted(pg))) == maj[z][0])
    return dict(events=n, units=agent.nz, purity=round(pure, 4), chars_seen=len(char2z),
                lex=len(lex), lex_covered=len(covered), lex_with_program=len(with_prog),
                programs=prog_n, programs_say_their_char=prog_ok,
                split_chars=sum(1 for ch in char2z if len(char2z[ch]) > 1))


def lexicon_chars():
    s = set()
    for x in W.OBJECTS + W.COLOURS + W.COUNTS + W.SLOTS + W.NAMES:
        s |= set(x)
    for q, acc in W.RITUALS:
        s |= set(q)
        for a in acc:
            s |= set(a)
    for frag in ('这是什么', '是什么', '在哪里', '在', '你要吗', '要我', '好不好吃', '很好吃', '是什么颜色', '的',
                 '有几个', '有', '是不是', '我叫什么', '刚才'):
        s |= set(frag)
    return s


class Timer:
    def __init__(self):
        self.t0 = time.time()

    def __call__(self):
        return f'{time.time() - self.t0:.0f}s'
