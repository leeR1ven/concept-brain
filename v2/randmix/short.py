# short functional + speed test: A nA + babble + B nB + BC nB, then frozen B and C probes (40 dialogues each)
import os, sys, time, copy, json, resource
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
sys.path.insert(1, os.path.join(HERE, '..', 'synth'))   # repo: the unchanged ARCH_v2 core and its bench_copy
import numpy as np
from rcore import RandAgent
from agent import BENCH
sys.path.insert(0, BENCH)
from world import World
from metrics import evaluate
from run_j import TUNE_BEST, THINK_FINAL
A = dict(a.split('=', 1) for a in sys.argv[1:])
nA, nB, seed = int(A.pop('nA', 100)), int(A.pop('nB', 150)), int(A.pop('seed', 0))
tag = A.pop('tag', 'x')
rm = None
for k in list(A):
    if k.startswith('rm_') or k == 'rm':
        v = A.pop(k); rm = {} if rm is None else rm
        if k != 'rm': rm[k[3:]] = v if k[3:] in ('code', 'alloc') else (float(v) if '.' in v else int(v))
kw = dict(TUNE_BEST); kw.update(THINK_FINAL); kw.update(rebind=1, ct=True, init_min_tau=10, hip_rho=2.0, ctx=False)
t0 = time.time()
wA = World('A', seed=seed, tutor='recast', eye_layout='retino')
ag = RandAgent(wA, seed=seed, rm=rm, **kw)
wA.run(ag, nA); ag.babble(wA.voice, 35960); t1 = time.time()
w = World('B', seed=seed, tutor='recast', prev=wA, eye_layout='retino'); w.run(ag, nB)
w = World('BC', seed=seed, tutor='recast', prev=w, eye_layout='retino'); w.run(ag, nB)
t2 = time.time()
out = {}
for st in 'BC':
    a2 = copy.deepcopy(ag); a2.learning = False
    wp = World(st, seed=seed, tutor='recast', probe_frac=1.0, held_frac=0.3, eye_layout='retino'); wp.run(a2, 40)
    r = evaluate(wp.log)[st]; out[st] = round(r['ALL']['acc'], 3); out['h' + st] = r['ALL'].get('held', {}).get('acc')
    del a2
t3 = time.time()
m = ag.core.mix
info = dict(tag=tag, rm=rm, A=round(t1 - t0, 1), train=round(t2 - t1, 1), probe=round(t3 - t2, 1), res=out,
            rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024, n=m.n,
            stats=(m.stats if rm is not None else None), snap=(m.snapshot() if rm is not None else None))
print(json.dumps(info, default=str))
open(os.path.join(HERE, 'out', 'short.jsonl'), 'a').write(json.dumps(info, default=str) + '\n')
