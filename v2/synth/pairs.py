# -*- coding: utf-8 -*-
"""Paired comparison of an addition against its reference, per seed: python3 pairs.py add=ref [add2=ref2 ...]"""
import sys, os
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from analyze import load, score
TASKS = ('what', 'slot', 'where', 'offer', 'taste', 'colour', 'count', 'ritual', 'check', 'name', 'before', 'oneshot')
def S(tag):
    pat = os.path.join(HERE, '..', 'tune', 'out', 'runs', tag[5:] + '_s*.json') if tag.startswith('tune:') else \
        os.path.join(HERE, 'out', 'runs', tag + '_s*.json')
    return {r['seed']: score(r) for r in load(pat)}
for a in sys.argv[1:]:
    add, ref = a.split('=')
    A, R = S(add), S(ref)
    seeds = sorted(set(A) & set(R))
    if not seeds:
        print(f"{add} vs {ref}: no common seeds ({sorted(A)} / {sorted(R)})"); continue
    dJ = [A[s]['J'] - R[s]['J'] for s in seeds]
    print(f"{add} vs {ref}: seeds {seeds}  J " + " ".join(f"{A[s]['J']:.2f}/{R[s]['J']:.2f}" for s in seeds) +
          f"  dJ mean {np.mean(dJ):+.3f}  viol {[A[s]['viol'] for s in seeds]}")
    for k in ('F', 'RET', 'H', 'DR', 'B3', 'C3', 'D3'):
        print(f"    {k:4s} " + " ".join(f"{A[s][k] - R[s][k]:+.2f}" for s in seeds), end='')
    print()
    print("    tasks " + " ".join(f"{t}{np.mean([A[s][t] - R[s][t] for s in seeds if A[s].get(t) is not None and R[s].get(t) is not None]):+.2f}" for t in TASKS))
