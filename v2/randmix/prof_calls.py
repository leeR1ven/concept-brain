# count MIX lookups and repeated contexts in a short ARCH_v2 run (A 100 + babble + B 150)
import os, sys, time, collections
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
sys.path.insert(1, os.path.join(HERE, '..', 'synth'))   # repo: the unchanged ARCH_v2 core and its bench_copy
import numpy as np
import rules
from agent_synth import SynthAgent
from agent import BENCH
sys.path.insert(0, BENCH)
from world import World
from run_j import TUNE_BEST, THINK_FINAL
cnt = collections.Counter(); seen = set(); tt = [0.0]; nk = []
orig = rules.MixDict.units
def units(self, keys, create=True, tick=True):
    t0 = time.time(); r = orig(self, keys, create, tick); tt[0] += time.time() - t0
    k = tuple(keys); cnt['calls'] += 1; cnt['create' if create else 'ro'] += 1
    if k in seen: cnt['repeat'] += 1
    else: seen.add(k)
    nk.append(len(keys))
    return r
rules.MixDict.units = units
kw = dict(TUNE_BEST); kw.update(THINK_FINAL); kw.update(rebind=1, ct=True, init_min_tau=10, hip_rho=2.0, ctx=False)
wA = World('A', seed=0, tutor='recast', eye_layout='retino')
ag = SynthAgent(wA, seed=0, **kw)
t0 = time.time(); wA.run(ag, 100); ag.babble(wA.voice, 35960); t1 = time.time()
w = World('B', seed=0, tutor='recast', prev=wA, eye_layout='retino'); w.run(ag, 150)
w = World('BC', seed=0, tutor='recast', prev=w, eye_layout='retino'); w.run(ag, 150)
t2 = time.time()
print('A+babble', round(t1-t0,1), 'B+BC', round(t2-t1,1), 'ticks', w.t, dict(cnt), 'mix secs', round(tt[0],1), 'keys/call', np.mean(nk), 'mix n', ag.core.mix.n, 'learn', ag.core.stats['learn_events'])
