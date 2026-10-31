# -*- coding: utf-8 -*-
"""Speaker identity check (step 1 hook): AgentV21 with voice=1 hears stage A (overheard routines: other2 answers) and
model-tutor B (70 % of model replies by other2); each heard line's voice category (agent's own, from the ear) is
matched by time to the world's speaker of that line (privileged, for scoring only).  Purity / confusion table.
usage: python3 voice_check.py [nA=300] [nB=200]"""
import collections, json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'synth'))  # repo: unchanged ARCH_v2 core + bench_copy
from agent_v21 import AgentV21
from agent import BENCH
sys.path.insert(0, BENCH)
from world import World
from run_v21 import TUNE_BEST, THINK_FINAL, ARCH_V2
A = dict(a.split('=', 1) for a in sys.argv[1:])
nA, nB = int(A.get('nA', 300)), int(A.get('nB', 200))
kw = dict(TUNE_BEST); kw.update(THINK_FINAL); kw.update(ARCH_V2); kw['voice'] = 1
t0 = time.time()
wA = World('A', seed=0, tutor='recast', eye_layout='retino')
ag = AgentV21(wA, seed=0, **kw)
wA.run(ag, nA); ag.babble(wA.voice, 35960)
wB = World('B', seed=0, tutor='model', prev=wA, eye_layout='retino')
wB.run(ag, nB)
spk = []                                   # (t_end of the line, speaker) from the world logs
for r in wA.log + wB.log:
    if r.get('t_prompt_end') is not None and r.get('prompt'):
        spk.append((r['t_prompt_end'], r.get('prompt_by', 'other')))
    if r.get('t_model_end') is not None:
        spk.append((r['t_model_end'], r.get('model_by', 'other2')))
spk.sort()
import bisect
ts = [t for t, _ in spk]
tab = collections.Counter()
for t, v, n in ag.voice_log:
    i = bisect.bisect_left(ts, t - 3)
    if i < len(ts) and abs(ts[i] - t) <= 3:
        tab[(spk[i][1], v)] += 1
print('voice templates', len(ag.voice.V), f'{time.time() - t0:.0f}s')
print('world speaker x agent voice category:', dict(tab))
for who in ('other', 'other2'):
    row = {v: c for (w, v), c in tab.items() if w == who}
    tot = sum(row.values())
    if tot:
        best = max(row, key=row.get)
        print(f'  {who}: n {tot}, majority category {best} share {row[best] / tot:.2f}, unclassified (-1) {row.get(-1, 0) / tot:.2f}')
json.dump({f'{w}|{v}': c for (w, v), c in tab.items()}, open(os.path.join(HERE, 'out', 'voice_check.json'), 'w'))
