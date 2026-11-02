# -*- coding: utf-8 -*-
"""Reference policies, run through the SAME world, mouth, timing and evaluate() as any agent.

They are 'privileged' agents: the world tells them the prompt text etc. through hooks that ordinary agents never get
(bench_attach / bench_prompt / bench_taught).  Each speaks its answer through the muscles with ARCH_v1 timing (first
syllable 6 ticks after the prompt, 3 ticks per char, force ramp 1-2-3 so repeated chars are re-articulated), and only
on free-reply exchanges (probe, recast).  They learn online from the replies they heard taught, as an agent would.

    baselines('B', seed=0)  -> {'echo': evaluate(log), 'last': ..., 'prior': ..., 'lookup': ..., 'ngram': ..., 'oracle': ...}
"""
from __future__ import annotations

import collections
import hashlib
import json
import os
import time

import numpy as np

from metrics import evaluate
from world import World

HERE = os.path.dirname(os.path.abspath(__file__))
LATENCY = 6


class Scripted:
    """says a planned text through the mouth (exact muscle sets, force 1-2-3 per char)."""
    privileged = True
    answer_kinds = ('probe', 'recast')

    def __init__(self):
        self.plan = collections.deque()
        self.zero = np.zeros(32, np.float32)
        self.need = None

    def bench_attach(self, world):
        self.need = {c: sorted(s) for c, s in world.voice.need.items()}

    def say(self, text, delay=LATENCY):
        self.plan.clear()
        for _ in range(delay):
            self.plan.append(self.zero)
        for ch in text:
            if ch not in self.need:
                continue
            for lv in (1.0, 2.0, 3.0):
                m = np.zeros(32, np.float32); m[self.need[ch]] = lv
                self.plan.append(m)

    def bench_prompt(self, info):
        if info['kind'] in self.answer_kinds:
            txt = self.answer(info)
            if txt:
                self.say(txt)

    def bench_taught(self, info, reply):
        pass

    def answer(self, info):
        return ''

    def step(self, obs):
        return self.plan.popleft() if self.plan else self.zero


class Silent:
    """the trivial agent (never moves)."""
    def __init__(self):
        self.zero = np.zeros(32, np.float32)

    def step(self, obs):
        return self.zero


class Echo(Scripted):
    def answer(self, info):
        return info['prompt']


class LastChar(Scripted):
    def answer(self, info):
        return info['prompt'][-1:]


class Prior(Scripted):
    """the most frequent reply heard taught for this task type."""
    def __init__(self):
        super().__init__()
        self.cnt = collections.defaultdict(collections.Counter)

    def bench_taught(self, info, reply):
        self.cnt[info['task']][reply] += 1

    def answer(self, info):
        c = self.cnt.get(info['task'])
        return c.most_common(1)[0][0] if c else ''


class Lookup(Scripted):
    """exact lookup: (prompt, scene, attended slot) -> the reply heard taught for exactly that input."""
    def __init__(self):
        super().__init__()
        self.tab = {}

    @staticmethod
    def key(info):
        return (info['prompt'], info.get('scene'), info.get('attend'))

    def bench_taught(self, info, reply):
        self.tab[self.key(info)] = reply

    def answer(self, info):
        return self.tab.get(self.key(info), '')


class Ngram(Scripted):
    """char n-gram of the reply given the prompt: next reply char from (prompt head+tail, previous reply char) with
    back-off to (prompt tail, prev), (prompt head, prev), (prev); greedy, <= 8 chars.  No copy mechanism."""
    def __init__(self):
        super().__init__()
        self.tabs = [collections.defaultdict(collections.Counter) for _ in range(4)]

    @staticmethod
    def ctx(p, prev):
        return [(p[:2], p[-2:], prev), (p[-2:], prev), (p[:2], prev), prev]

    def bench_taught(self, info, reply):
        p = info['prompt']
        seq = ['^'] + list(reply) + ['$']
        for a, b in zip(seq, seq[1:]):
            for tab, k in zip(self.tabs, self.ctx(p, a)):
                tab[k][b] += 1

    def answer(self, info):
        p, prev, out = info['prompt'], '^', []
        for _ in range(8):
            nxt = None
            for tab, k in zip(self.tabs, self.ctx(p, prev)):
                if tab.get(k):
                    nxt = tab[k].most_common(1)[0][0]
                    break
            if nxt is None or nxt == '$':
                break
            out.append(nxt); prev = nxt
        return ''.join(out)


class Oracle(Scripted):
    def answer(self, info):
        return info['acceptable'][0] if info.get('acceptable') else ''


BASELINES = {'echo': Echo, 'last': LastChar, 'prior': Prior, 'lookup': Lookup, 'ngram': Ngram, 'oracle': Oracle}


def run_agent(agent, stage, seed=0, n_dialogues=2000, tutor='model', **world_kw):
    w = World(stage, seed=seed, tutor=tutor, **world_kw)
    log = w.run(agent, n_dialogues)
    return w, log


def baselines(stage, seed=0, n_dialogues=2000, tutor='model', names=None, cache=True, **world_kw):
    """scores of every baseline on World(stage, seed, tutor, **world_kw) with n_dialogues; cached in cache/."""
    names = list(names or BASELINES)
    key = json.dumps(dict(stage=stage, seed=seed, n=n_dialogues, tutor=tutor, names=names, kw=world_kw), sort_keys=True)
    path = os.path.join(HERE, 'cache', 'baselines_' + hashlib.md5(key.encode()).hexdigest()[:12] + '.json')
    if cache and os.path.exists(path):
        d = json.load(open(path, encoding='utf-8'))
        if d.get('key') == key:
            return d['res']
    res = {}
    for nm in names:
        t0 = time.time()
        w, log = run_agent(BASELINES[nm](), stage, seed, n_dialogues, tutor, **world_kw)
        res[nm] = evaluate(log)
        res[nm]['_secs'] = round(time.time() - t0, 1)
        res[nm]['_ticks'] = w.t
    if cache:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        json.dump(dict(key=key, res=res), open(path, 'w', encoding='utf-8'), ensure_ascii=False)
    return res
