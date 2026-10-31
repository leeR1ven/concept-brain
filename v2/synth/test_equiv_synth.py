# -*- coding: utf-8 -*-
"""Equivalence: SynthAgent in v1 mode (think_k=3, ior=0, every synthesis key at default) == tune's DevAgent, with
tune's recommended recall configuration.  Also: ct=True at the register limit (ct_lam 0.3 < theta, no fading) == the
same.  usage: python3 test_equiv_synth.py nA=100 nB=300 [ct=1]"""
import os, sys, copy, time
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np
from agent import DevAgent, BENCH
from agent_synth import SynthAgent
sys.path.insert(0, BENCH)
from world import World
A = dict(a.split("=", 1) for a in sys.argv[1:])
nA, nB = int(A.get('nA', 100)), int(A.get('nB', 300))
RN = int(A.pop('rn', 8))
extra = {k: (float(v) if '.' in v else int(v)) for k, v in A.items() if k not in ('nA', 'nB')}
CFG = dict(recall_n=RN, eta_recall=0.2, rec_pow=0.0, hip_cap=10000, hip_ndg=6000, hip_kdg=120, floor=0.3)
def run(kind):
    t0 = time.time()
    wA = World('A', seed=0, tutor='recast')
    if kind == 'dev':
        ag = DevAgent(wA, seed=0, **CFG)
    else:
        ag = SynthAgent(wA, seed=0, policy='v1', think_k=3, ior=0, **CFG, **extra)
    wA.run(ag, nA); ag.babble(wA.voice, 35960)
    wB = World('B', seed=0, tutor='recast', prev=wA)
    wB.run(ag, nB)
    recs = [(r['i'], r['kind'], r['prompt'], r['reply'], r.get('latency'), r.get('after_recast_reply')) for r in wB.log]
    W = ag.core.gen.W[:ag.core.mix.n]
    return recs, ag.core.mix.n, float(np.abs(W).sum()), ag.core.stats['awake_recalls'], time.time() - t0
a = run('dev'); b = run('synth')
same = sum(x == y for x, y in zip(a[0], b[0]))
print(f"records {len(a[0])} vs {len(b[0])}, identical {same}; mix {a[1]} vs {b[1]}; |W| {a[2]:.4f} vs {b[2]:.4f}; "
      f"recalls {a[3]} vs {b[3]}; secs {a[4]:.0f} vs {b[4]:.0f}")
for x, y in zip(a[0], b[0]):
    if x != y:
        print('first diff', x, y); break
