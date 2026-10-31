# -*- coding: utf-8 -*-
"""Markdown table of configurations: python3 mdtable.py "label=tag[@seeds]" ...  (tag 'tune:<tag>' = tune's runs)"""
import sys, os
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from analyze import load, score
KEYS = ('J', 'F', 'RET', 'H', 'DR', 'B3', 'C3', 'D3')
TASKS = ('what', 'slot', 'where', 'offer', 'taste', 'colour', 'count', 'ritual', 'check', 'name', 'before', 'oneshot')
def recs(tag):
    pat = os.path.join(HERE, '..', 'tune', 'out', 'runs', tag[5:] + '_s*.json') if tag.startswith('tune:') else \
        os.path.join(HERE, 'out', 'runs', tag + '_s*.json')
    return load(pat)
print('| configuration | seeds | ' + ' | '.join(KEYS) + ' | ' + ' | '.join(TASKS) + ' | viol |')
print('|' + '---|' * (3 + len(KEYS) + len(TASKS)))
for a in sys.argv[1:]:
    label, spec = a.split('=', 1)
    seeds = None
    if '@' in spec:
        spec, ss = spec.split('@'); seeds = [int(x) for x in ss.split(',')]
    tags = spec.split('+')
    R = [r for t in tags for r in recs(t) if seeds is None or r['seed'] in seeds]
    S = [score(r) for r in R]
    if not S:
        print(f'| {label} | – |'); continue
    def m(k):
        v = [s[k] for s in S if s.get(k) is not None]
        return np.mean(v) if v else float('nan')
    sd = np.std([s['J'] for s in S]) if len(S) > 1 else 0
    print(f"| {label} | {','.join(str(s['seed']) for s in S)} | {m('J'):.2f} ({sd:.2f}) | " + ' | '.join(f"{m(k):.2f}" for k in KEYS[1:]) +
          ' | ' + ' | '.join(f"{m(t):.2f}" for t in TASKS) + f" | {sum(s['viol'] for s in S)} |")
