# -*- coding: utf-8 -*-
"""Compare configurations: J and its parts (analyze.py's objective), per-task end scores, drift, mechanics.
usage: python3 table.py tag1 tag2 ... [--seeds 0,1] [--dir out/runs] [--tune]  (--tune: tune's base from its folder)"""
import sys, os, glob, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from analyze import load, score
a = sys.argv[1:]
seeds = [int(x) for x in a[a.index('--seeds') + 1].split(',')] if '--seeds' in a else None
tags = [t for i, t in enumerate(a) if not t.startswith('--') and (i == 0 or a[i - 1] not in ('--seeds',))]
def recs(tag):
    if tag.startswith('tune:'):
        pat = os.path.join(HERE, '..', 'tune', 'out', 'runs', tag[5:] + '_s*.json')
    else:
        pat = os.path.join(HERE, 'out', 'runs', tag + '_s*.json')
    R = load(pat)
    return [r for r in R if seeds is None or r['seed'] in seeds]
KEYS = ('J', 'F', 'RET', 'H', 'DR', 'B1', 'C2', 'B3', 'C3', 'D3')
TASKS = ('what', 'slot', 'where', 'offer', 'taste', 'colour', 'count', 'ritual', 'check', 'name', 'before', 'oneshot')
print(f"{'config':28s} seeds  " + " ".join(f"{k:>5s}" for k in KEYS) + "  | " + " ".join(f"{t[:5]:>5s}" for t in TASKS) + " | viol")
for tag in tags:
    R = recs(tag)
    if not R:
        print(f"{tag:28s} (no runs)"); continue
    S = [score(r) for r in R]
    m = lambda k: np.mean([s[k] for s in S if s.get(k) is not None]) if any(s.get(k) is not None for s in S) else float('nan')
    print(f"{tag:28s} {','.join(str(s['seed']) for s in S):5s}  " + " ".join(f"{m(k):5.2f}" for k in KEYS) + "  | " +
          " ".join(f"{m(t):5.2f}" for t in TASKS) + f" | {sum(s['viol'] for s in S)}")
    if '--per' in a:
        for s in S:
            print(f"   s{s['seed']}: " + " ".join(f"{k} {s[k]:.2f}" for k in KEYS if s.get(k) is not None))
