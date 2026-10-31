# -*- coding: utf-8 -*-
"""Context-length diagnostic (separate from J): train A 300 + B 300 dialogues (the agent can name objects), then the
context probe of run_t2.ctx_probe (partner: X -> [gap fillers] -> 刚才说什么 ; 150 recast dialogues with learning on,
then 60 frozen tests), gaps 0 and 1, plus the thought's held-line statistics.
usage: python3 ctxprobe_run.py tag=x seed=0 [cfg keys]"""
import os, sys, json, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from run_t2 import T2Agent, World, TUNE_BEST, THINK_FINAL, ARCH_V2, typed, ctx_probe   # noqa: E402
import numpy as np                                                                     # noqa: E402
A = dict(a.split('=', 1) for a in sys.argv[1:])
tag, seed = A.pop('tag', 'cp'), int(A.pop('seed', 0))
ntr, nte, nB = int(A.pop('ntr', 150)), int(A.pop('nte', 60)), int(A.pop('nB', 300))
kw = dict(TUNE_BEST); kw.update(THINK_FINAL); kw.update(ARCH_V2)
for k, v in A.items():
    kw[k] = typed(k, v)
wkw = {'eye_layout': 'retino'}
t0 = time.time()
wA = World('A', seed=seed, tutor='recast', **wkw)
ag = T2Agent(wA, seed=seed, **kw); wA.run(ag, 300); ag.babble(wA.voice, 35960)
w = World('B', seed=seed, tutor='recast', prev=wA, **wkw); w.run(ag, nB)
cs = ag.core.ctx_stats
out = dict(tag=tag, seed=seed, cfg={k: str(v) for k, v in kw.items()}, mix=int(ag.core.mix.n),
           mean_held=round(cs['held_sum'] / max(cs['reply_views'], 1), 2),
           back=[round(b / max(cs['reply_views'], 1), 3) for b in cs['back']], probes=[])
for gap in (0, 1):
    r = ctx_probe(ag, seed, gap, ntr, nte, wkw)
    out['probes'].append(r)
    print(f"[{tag} s{seed}] gap {gap}: test {r['test_acc']} (const {r['const']}) spoke {r['spoke']} train_last {r['train_last']} says_obj {r['test_says_object']}", flush=True)
out['secs'] = round(time.time() - t0, 1)
os.makedirs(os.path.join(HERE, 'out', 'ctxprobe'), exist_ok=True)
json.dump(out, open(os.path.join(HERE, 'out', 'ctxprobe', f'{tag}_s{seed}.json'), 'w'), ensure_ascii=False)
print(f"[{tag} s{seed}] done {out['secs']}s mix {out['mix']} held {out['mean_held']} back {out['back'][:5]}", flush=True)
