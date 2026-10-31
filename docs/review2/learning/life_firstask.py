# -*- coding: utf-8 -*-
"""Held-out pairs in ARCH_v2's life (seed 5): accuracy by how often the same pair was asked before (never followed
by help: held-out pairs are always probes).  Existing log synth/out/life/final_noctx_s5_log.pkl.  Also the seen
items asked as probes vs recast.  usage: python3 life_firstask.py"""
import os, pickle, sys, collections
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'synth', 'bench_copy'))
L = pickle.load(open(os.path.join(HERE, '..', '..', 'synth', 'out', 'life', 'final_noctx_s5_log.pkl'), 'rb'))
ex = [r for r in L if r.get('type') == 'ex' and r.get('kind') in ('probe', 'recast')]
for name, sel in [('held-out', lambda r: r['held_out']), ('seen, new situation', lambda r: not r['held_out'] and r.get('new_situation'))]:
    for lo, hi in [(0, 0), (1, 1), (2, 3), (4, 7), (8, 999)]:
        R = [r for r in ex if sel(r) and lo <= r.get('n_asked_before', 0) <= hi]
        if R:
            for half, (a, b) in [('1st half', (0, 500000)), ('2nd half', (500000, 9e9))]:
                S = [r for r in R if a <= r['life_t'] < b]
                if S:
                    print(f'{name:20s} asked before {lo}-{hi}: {half} n={len(S):4d} acc={sum(r["acc"] for r in S)/len(S):.2f}')
