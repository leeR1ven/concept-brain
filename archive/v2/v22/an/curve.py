# -*- coding: utf-8 -*-
"""Life free-reply accuracy per tenth of life, with the same-tenth echo-or-constant reference (constant chosen per task
on the whole life, as life_metrics).  usage: python3 an/curve.py tag [tag ...] (out/life/<tag>_s{5,6}_log.pkl)"""
import collections, pickle, sys, os
import numpy as np
HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
LIFE_FREE = ('probe', 'recast', 'social')
for tag in sys.argv[1:]:
    for s in (5, 6):
        p = os.path.join(HERE, 'out', 'life', f'{tag}_s{s}_log.pkl')
        if not os.path.exists(p):
            continue
        L = pickle.load(open(p, 'rb'))
        free = [r for r in L if r.get('type') == 'ex' and r.get('kind') in LIFE_FREE and r.get('acceptable')]
        T = max(r['t'] for r in L if 't' in r)
        by = collections.defaultdict(list)
        for r in free:
            by[r['task']].append(r)
        const = {t: collections.Counter(a for x in rs for a in set(x['acceptable'])).most_common(1)[0][0] for t, rs in by.items()}
        ag, ref = [], []
        for k in range(10):
            rs = [r for r in free if k * T / 10 <= r['t_prompt_start'] < (k + 1) * T / 10]
            ag.append(np.mean([r['reply'] in r['acceptable'] for r in rs]) if rs else np.nan)
            ref.append(np.mean([(r['prompt'] in r['acceptable']) or (const[r['task']] in r['acceptable']) for r in rs]) if rs else np.nan)
        print(f'{tag:10s} s{s} agent ' + ' '.join(f'{x:.2f}' for x in ag) + ' | e|c ' + ' '.join(f'{x:.2f}' for x in ref))
