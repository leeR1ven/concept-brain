# -*- coding: utf-8 -*-
"""Adapters around the batched torch brain.

TorchAgent      one bench agent (World.run calls agent.step(obs) -> (32,) muscles); B = 1 brain.  Same birth as the
                numpy SynthAgent: ear layer 0 + FEAT0 from the world's warm-up, DG wiring from the agent seed (numpy
                rng, identical draws), babbling = DevAgent.babble's offline pass (numpy ear; it is not part of the
                learned core), then the categories bind their babbled sounds on the device.
"""
from __future__ import annotations

import copy
import itertools
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'synth'))
sys.path.insert(0, os.path.join(HERE, '..', 'synth', 'bench_copy'))
from frontend import Ear0, feat0          # noqa: E402
from physics import Stream                # noqa: E402
from rules import Hippo                   # noqa: E402
from brain_torch import Brain, Cfg        # noqa: E402


def birth_init(world, seed, cfg):
    """the fixed random wiring of one agent, drawn exactly as the numpy reference draws it."""
    ear = Ear0(seed=world.seed)
    F0 = feat0(Ear0(seed=world.seed), world.warmup_windows())
    n_eye = len(world.eye)
    V = cfg.V
    groups = {'bag': V, 'first': V, 'last': V, 'scene': n_eye, 'ctx': 800}
    hip = Hippo(groups, n_dg=cfg.hip_ndg, k_dg=cfg.hip_kdg, fan=(10, 4, 4, 10, 10),
                weights={'bag': 1.0, 'first': 1.0, 'last': 1.0, 'scene': 1.0, 'ctx': 1.0}, cap=1, seed=seed + 7,
                gamma=cfg.hip_gamma, theta=cfg.hip_theta)
    d = dict(F0=F0, ear_idx=ear.idx, ear_w=ear.w, ear_b=ear.b, ear_alive=ear.alive, hip_tie=hip.tie, seed=seed,
             n_eye=n_eye)
    for g in ('bag', 'first', 'last', 'scene'):
        d['hip_idx_' + g] = hip.idx[g]
    return d, ear, F0


def babble_patterns(ear, F0, voice, n, seed):
    """DevAgent.babble's offline pass: every producible own sound once (its FEAT0 pattern) + its muscle set."""
    rng = np.random.default_rng(seed + 31)
    cache, pats, sets, seen = {}, [], [], set()
    allsets = list(itertools.combinations(range(32), 4))
    order = rng.permutation(len(allsets))
    for i in range(1, n + 1):
        m = allsets[order[(i - 1) % len(allsets)]]
        said = voice.decode(m)
        if said is not None and said != '~':
            f = cache.get(said)
            if f is None:
                f = cache[said] = own_pattern(ear, F0, voice, said)
            if said not in seen:
                seen.add(said); pats.append(f); sets.append(m)
    return np.array(pats, bool).reshape(-1, len(F0)), np.array(sets, np.int64).reshape(-1, 4)


def own_pattern(ear, F0, voice, ch):
    st = Stream(voice.F, voice.T, voice.T // 3, 3, voice.levels)
    e = Ear0.__new__(Ear0)
    e.__dict__.update(ear.__dict__)
    e.reset()
    none = (None, None, False)
    g = voice.hear(ch, 'self')
    out = None
    for k in range(3):
        win = st.tick((none, none, (ch, g, k == 0)))
        out = e.step(win)
    return out[F0]


class TorchAgent:
    privileged = False

    def __init__(self, world, seed=0, cfg=None, device='cuda'):
        self.cfg = cfg or Cfg()
        init, self.ear, self.F0 = birth_init(world, seed, self.cfg)
        self.seed = seed
        self.brain = Brain(1, [init], self.cfg, device)
        self._z = np.array([0])

    # ---- run_j.py's view of an agent
    @property
    def learning(self):
        return bool(self.brain.learning[0])

    @learning.setter
    def learning(self, v):
        self.brain.learning[0] = bool(v)

    def set_learn_listen(self, v):
        self.brain.learn_listen[0] = bool(v)

    def step(self, obs):
        f = self.brain.step(np.asarray(obs['ear'])[None], np.asarray(obs['eye']).astype(bool)[None],
                            np.array([float(obs.get('reward', 0.0))]))
        return f[0].astype(np.float32)

    def babble(self, voice, n):
        pats, sets = babble_patterns(self.ear, self.F0, voice, n, self.seed)
        self.brain.load_babble(self._z, pats, sets)

    def __deepcopy__(self, memo):
        new = copy.copy(self)
        new.brain = self.brain.clone()
        return new

    def summary(self):
        b = self.brain
        return dict(categories=int(b.zn[0]), programs=int((b.cat2prog[0] >= 0).sum()), mix=int(b.conj.n[0].item()),
                    hip=int(b.hn[0]), recalls=int(b.stats['awake_recalls']), released=int(b.stats['released']))
