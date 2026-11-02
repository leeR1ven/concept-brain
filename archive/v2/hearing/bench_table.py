# -*- coding: utf-8 -*-
"""Paired bench table: arm A (v2.1 final, canonical) vs arm B (+ hearing keys), and the noise arm if present.
Reads results/bench/<tag>_s*.json (copies of the A800 runner's hearing_<job>/result/out/runs/).  usage: python3 bench_table.py"""
import os, glob, json
HERE = os.path.dirname(os.path.abspath(__file__))
Q = os.path.join(HERE, 'results', 'bench')


def load(job, tag):
    out = {}
    for f in glob.glob(os.path.join(Q, f'{tag}_s*.json')):
        for l in open(f):
            d = json.loads(l)
            out[d['seed']] = d
    return out


A, B = load('benchA', 'hA'), load('benchB', 'hB')
print('| seed | J A | J B | ΔJ | J2 A | J2 B | ΔJ2 | H A / B | replies identical | hearing counters B | viol A/B |')
print('|---|---|---|---|---|---|---|---|---|---|---|')
dj, dj2 = [], []
for s in sorted(set(A) | set(B)):
    a, b = A.get(s), B.get(s)
    if a is None or b is None:
        print(f'| {s} | {a and a["score"]["J"]} | {b and b["score"]["J"]} | - | | | | | | | |'); continue
    same = [r['reply'] for r in a['replies']] == [r['reply'] for r in b['replies']]
    hc = {k: v for k, v in b['stats'].items() if k.startswith('h_') and v}
    dj.append(b['score']['J'] - a['score']['J']); dj2.append(b['score']['J2'] - a['score']['J2'])
    print(f"| {s} | {a['score']['J']} | {b['score']['J']} | {dj[-1]:+.4f} | {a['score']['J2']} | {b['score']['J2']} | "
          f"{dj2[-1]:+.4f} | {a['score']['H']} / {b['score']['H']} | {same} (mix {a['mix']}/{b['mix']}) | {hc} | "
          f"{a['viol']}/{b['viol']} |")
if dj:
    import numpy as np
    print(f'mean ΔJ {np.mean(dj):+.4f} ± {np.std(dj):.4f}, mean ΔJ2 {np.mean(dj2):+.4f}; '
          f'J A {np.mean([A[s]["score"]["J"] for s in A]):.4f} B {np.mean([B[s]["score"]["J"] for s in B]):.4f}')
for job, tag in (('benchN', 'hN'), ('benchN', 'hNv21')):
    N = load(job, tag)
    for s, d in sorted(N.items()):
        hs = {k: v for k, v in d['stats'].items() if k.startswith('h_') and v}
        print(f"noise σ1 {tag} s{s}: score {d['score']} viol {d['viol']} hearing {hs} mix {d['mix']} hip {d['hip']} "
              f"per-task { {t: v['acc'] for t, v in d['per_task'].items()} }")
