# -*- coding: utf-8 -*-
"""AgentV21 with every new key at its default == ARCH_v2's SynthAgent, bit for bit (mouth output, replies, weights).
usage: python3 test_equiv_v21.py [nB=60] [extra agent keys for BOTH runs]"""
import os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'synth'))  # repo: unchanged ARCH_v2 core + bench_copy
from agent_synth import SynthAgent
from agent_v21 import AgentV21
from agent import BENCH
sys.path.insert(0, BENCH)
from world import World
from run_j import TUNE_BEST, THINK_FINAL
A = dict(a.split('=', 1) for a in sys.argv[1:])
nB = int(A.pop('nB', 60)); v21x = A.pop('v21', '')
FINAL = dict(rebind=1, ct=True, init_min_tau=10, hip_rho=2.0, ctx=False)
def run(cls, extra):
    kw = dict(TUNE_BEST); kw.update(THINK_FINAL); kw.update(FINAL); kw.update(extra)
    wA = World('A', seed=0, tutor='recast', eye_layout='retino')
    ag = cls(wA, seed=0, **kw)
    wA.run(ag, 20); ag.babble(wA.voice, 35960)
    w = World('B', seed=0, tutor='recast', prev=wA, eye_layout='retino'); w.run(ag, nB)
    w2 = World('C', seed=0, tutor='recast', prev=w, eye_layout='retino'); w2.run(ag, nB)
    reps = [r.get('reply') for r in w.log + w2.log]
    return reps, ag.core.gen.W[:ag.core.mix.n].copy(), ag.core.mix.n, ag.core.hip.n, ag
t0 = time.time()
r1, W1, m1, h1, a1 = run(SynthAgent, {})
extra = {}
if v21x:
    for kv in v21x.split(','):
        k, v = kv.split(':'); extra[k] = type(getattr(__import__('core_v21').RCfg(), k))(v) if k != 'voice' else int(v)
r2, W2, m2, h2, a2 = run(AgentV21, extra)
same = r1 == r2 and m1 == m2 and h1 == h2 and W1.shape == W2.shape and np.array_equal(W1, W2)
print(f'replies {len(r1)} identical {r1 == r2} | mix {m1} {m2} | hip {h1} {h2} | GEN identical {W1.shape == W2.shape and np.array_equal(W1, W2)}')
print('EQUIVALENT' if same else 'DIFFERENT', f'{time.time() - t0:.0f}s', getattr(a2.core, 'ral_stats', None))
if a2.voice is not None:
    print('voice templates', len(a2.voice.V), 'line voices', __import__('collections').Counter(v for _, v, _ in a2.voice_log))
