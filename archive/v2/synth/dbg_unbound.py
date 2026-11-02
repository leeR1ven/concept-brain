import os, sys
sys.path.insert(0, '.')
import numpy as np
from agent_synth import SynthAgent
from agent import BENCH
sys.path.insert(0, BENCH)
from world import World
from run_j import TUNE_BEST, THINK_FINAL
seed = int(sys.argv[1]) if len(sys.argv) > 1 else 1
base = sys.argv[2] if len(sys.argv) > 2 else 'think'
kw = dict(TUNE_BEST); kw.update(THINK_FINAL if base == 'think' else dict(policy='v1', think_k=3, ior=0))
wA = World('A', seed=seed, tutor='recast')
ag = SynthAgent(wA, seed=seed, **kw)
wA.run(ag, 300); ag.babble(wA.voice, 35960)
print('cats', ag.Z.n, 'progs', len(ag.cat2prog), 'items', ag.stats['items'])
# which char does each category sound like (match the own-pattern of each lexicon char)
chars = sorted(set(''.join(r['prompt'] + (r.get('model_reply') or '') for r in wA.log)))
pats = {ch: ag.own_pattern(wA.voice, ch) for ch in chars}
for j in range(ag.Z.n):
    t = ag.Z.W[:, j]
    best = sorted(((float((t * p).sum() / max(t.sum(), 1e-6)), ch) for ch, p in pats.items()), reverse=True)[:3]
    tag = '' if j in ag.cat2prog else '  <-- UNBOUND'
    if tag or j < 0:
        print(j, 'count', ag.Z.count[j], best, tag)
dup = {}
for j in range(ag.Z.n):
    t = ag.Z.W[:, j]
    b = max(((float((t * p).sum() / max(t.sum(), 1e-6)), ch) for ch, p in pats.items()))
    dup.setdefault(b[1], []).append((j, round(b[0], 2), int(ag.Z.count[j]), j in ag.cat2prog))
print({k: v for k, v in dup.items() if len(v) > 1})
