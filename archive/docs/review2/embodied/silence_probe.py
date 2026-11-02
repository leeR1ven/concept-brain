# -*- coding: utf-8 -*-
"""Embodied review probe (read-only use of world2): does world2 leave ARCH_v2 the exact silences it depends on?

ARCH_v2 code facts being tested (agent_think.py / agent_synth.py):
  line onset  = E > 0 after E == 0;  line end = E == 0       (E = ear-window energy sum)
  quiet moment for awake recall = 12 consecutive ticks with E == 0, not speaking, not in a line
  innate brake F = 0.01 * E >= 0.1  (E >= 10): no GO while F
Speakers are reference "chatter" bodies: each agent, after its ear has been silent >= GAP ticks, starts with
probability p per tick an utterance of 2-4 random lexicon chars (force ramp 1-2-3 per char, like world2's Scripted).
The rate is calibrated to ARCH_v2's own initiative rates (life: ~2 per 1,000 quiet ticks; world2 smoke: ~8 per 1,000
ticks).  Measured per agent from obs['ear'] only (what the agent would see), daytime ticks only.
usage: python3 silence_probe.py N p noise days
"""
import os
import sys
import time
import json
import collections

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'bench'))
from world2 import World2                    # noqa: E402
from world2.agents import Scripted           # noqa: E402
from world2.core import N_MOUTH              # noqa: E402

GAP = 10


class Chatter(Scripted):
    def __init__(self, n, p, seed=0):
        super().__init__(n)
        self.p = p
        self.rng = np.random.default_rng(seed)
        self.quiet = np.zeros(n, np.int64)
        self.N = n
        # measurement
        self.m = collections.defaultdict(float)
        self.qrun = np.zeros(n, np.int64)
        self.seg = np.zeros(n, np.int64)
        self.segs = []
        self.t = 0
        self.mouth_ago = np.full(n, 99)

    def bench_attach(self, world):
        super().bench_attach(world)
        self.chars = [c for c in self.need]

    def bench_prompt(self, j, info):
        pass

    def step(self, obs):
        E = obs['ear'].reshape(self.N, -1).sum(1)
        day = (self.w.t % 30000) < 24000
        own = self.mouth_ago <= 2
        if day:
            self.m['ticks'] += self.N
            self.m['silent'] += float(((E == 0) & ~own).sum())
            self.m['F_on_not_own'] += float(((E >= 10) & ~own).sum())
            self.m['own'] += float(own.sum())
            # quiet moments (ARCH_v2 rule: 12 consecutive silent ticks, not speaking)
            q = (E == 0) & ~own
            self.qrun = np.where(q, self.qrun + 1, 0)
            self.m['quiet_moments'] += float((self.qrun == 12).sum())
            self.m['long_silence_200'] += float((self.qrun == 200).sum())
            # heard segments (E > 0 runs while not speaking): ARCH_v2's "lines"
            heard = (E > 0) & ~own
            ended = (~heard) & (self.seg > 0)
            for j in np.flatnonzero(ended):
                self.segs.append(int(self.seg[j]))
            self.seg = np.where(heard, self.seg + 1, 0)
        # chatter
        self.quiet = np.where(E == 0, self.quiet + 1, 0)
        for j in np.flatnonzero((self.quiet >= GAP) & (self.rng.random(self.N) < self.p)):
            if j not in self.active:
                L = int(self.rng.integers(2, 5))
                txt = ''.join(self.chars[int(i)] for i in self.rng.integers(len(self.chars), size=L))
                self.say(j, txt, delay=0)
                self.m['utterances'] += 1
                self.quiet[j] = 0
        A = super().step(obs)
        self.mouth_ago = np.where(A[:, :N_MOUTH].max(1) > 0, 0, self.mouth_ago + 1)
        return A


def main():
    N, p, noise, days = int(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
    t0 = time.time()
    w = World2(n_agents=N, seed=0, noise=noise)
    ag = Chatter(N, p, seed=1)
    w.run(ag, ticks=int(days * 30000))
    m = ag.m
    dayticks = m['ticks'] / N
    says = [r for r in w.log if r.get('type') == 'say']
    qs = [r for r in w.log if r.get('type') == 'q']
    segs = np.array(ag.segs) if ag.segs else np.array([0])
    out = dict(N=N, p=p, noise=noise, days=days, secs=round(time.time() - t0, 1),
               utter_per_agent_per_1000=1000 * m['utterances'] / N / (days * 30000),
               silent_frac=m['silent'] / m['ticks'], F_on_frac=m['F_on_not_own'] / m['ticks'],
               quiet_moments_per_agent_day=m['quiet_moments'] / N / days,
               silences_200_per_agent_day=m['long_silence_200'] / N / days,
               heard_segments_per_agent_day=len(ag.segs) / N / days,
               seg_len_median=float(np.median(segs)), seg_len_p90=float(np.percentile(segs, 90)),
               seg_len_max=int(segs.max()),
               caregiver_lines_per_day=len(says) / days, questions_per_agent_day=len(qs) / N / days)
    print(json.dumps(out), flush=True)


if __name__ == '__main__':
    main()
