# -*- coding: utf-8 -*-
"""Held-out vs all accuracy per task of ARCH_v2 (final_noctx, seeds 0-4) from the existing staged-run JSONs.
table entries are [acc, n, held_acc, held_n, ...] (run_j.py).  usage: python3 held_table.py"""
import json, os
import numpy as np
RUNS = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'synth', 'out', 'runs')
tasks = {'B': ['what', 'slot', 'where'], 'C': ['colour', 'count', 'offer', 'taste'], 'D': ['check']}
agg = {}
for s in range(5):
    T = json.load(open(os.path.join(RUNS, f'final_noctx_s{s}.json')))['table']['s3']
    for st, ts in tasks.items():
        for t in ts:
            agg.setdefault(t, []).append(T[st][t])
for t, l in agg.items():
    a = np.mean([x[0] for x in l]); h = np.nanmean([np.nan if x[2] is None else x[2] for x in l])
    print(f'{t:7s} all {a:.2f}  held-out {h:.2f}  (held n {sum(x[3] for x in l)})  per seed', [x[2] for x in l])
