# -*- coding: utf-8 -*-
"""summarise gen_probe_mean.py outputs: per variant, means over seeds; pooled held outcomes vs chance (binomial).
usage: python3 gp_table.py <jsonl files or dirs>"""
import collections, glob, json, os, sys
from math import comb
import numpy as np

files = []
for a in sys.argv[1:] or ['out/gp1']:
    files += sorted(glob.glob(os.path.join(a, '*.jsonl'))) if os.path.isdir(a) else [a]
rows = collections.defaultdict(list)
for f in files:
    for l in open(f):
        r = json.loads(l)
        key = (r.get('tag') or '') + ('w2' if r.get('w2') else '') + (f"late{r['late']}" if r.get('late') else '') + '|' + r['layer'] + ('' if r['layer'] == 'mix' else '/' + r['mean']) + \
            ('' if not r.get('mk') else ' ' + ','.join(f'{k}={v}' for k, v in sorted(r['mk'].items()) if k != 'cls')) + \
            ('' if not r.get('cfgx') else ' ' + ','.join(f'{k}={v}' for k, v in sorted(r['cfgx'].items()))) + \
            ('' if r.get('ral', {}).get('ral', 1) else ' noRAL')
        rows[key].append(r)


def m(v):
    v = [x for x in v if x is not None]
    return f'{np.mean(v):.2f}' if v else '–'


def pbin(k, n):
    """one-sided P(X >= k) for X ~ Bin(n, 0.5)"""
    return sum(comb(n, j) for j in range(k, n + 1)) / 2 ** n


def held_count(F):
    k = n = 0
    for f in F:
        for x, y in f['held_ex']:
            n += 1
    return n


print('| variant | seeds | held per seed | held | food / toy | pooled fillers correct (p vs 0.5) | seen_all | fault drop | '
      'no recall: held | meaning off (all): held | layer-only off: held | swap held F<->T: held | codes within / between / purity | class-only units F/T (held F in F / in T) | secs |')
print('|' + '---|' * 15)
for k, R in sorted(rows.items()):
    R = sorted(R, key=lambda r: r['seed'])
    F = [r['frozen'] for r in R]
    nc = sum(int(round(f['held'] * 4)) for f in F)       # 4 held fillers per seed; replies deterministic per filler
    nn = 4 * len(F)
    L = lambda name: [r.get('lesion_' + name, {}).get('held') for r in R]
    cs = [r.get('codes') for r in R if r.get('codes')]
    cu = [r['frozen'].get('class_units') for r in R if r['frozen'].get('class_units')]
    print(f"| {k} | {len(R)} | {'/'.join('%.2f' % f['held'] for f in F)} | **{m([f['held'] for f in F])}** | "
          f"{m([f['held_food'] for f in F])} / {m([f['held_toy'] for f in F])} | {nc}/{nn} (p={pbin(nc, nn):.3f}) | "
          f"{m([f['seen_all'] for f in F])} | {m([f['fault_drop'] for f in F])} | {m(L('norec'))} | {m(L('off'))} | "
          f"{m(L('lay_off'))} | {m(L('swap'))} | "
          f"{(m([c['within'] for c in cs]) + ' / ' + m([c['between'] for c in cs]) + ' / ' + m([c['purity'] for c in cs])) if cs else '–'} | "
          f"{(m([c['food_only'] for c in cu]) + '/' + m([c['toy_only'] for c in cu]) + ' (' + m([c['held_food_in_food'] for c in cu]) + ' / ' + m([c['held_food_in_toy'] for c in cu]) + ')') if cu else '–'} | "
          f"{m([r['secs'] for r in R])} |")
