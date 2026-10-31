# -*- coding: utf-8 -*-
"""Aggregate run_v21 results per tag (mean, SD over seeds): J, J2, F, RET, H, DR, drift loss (300 / 900), per-task
accuracy and held-out (pooled n-weighted), role confusion, same-run trivial references, comprehension probes.
usage: python3 agg_v21.py [glob ...] [--tags a,b] [--json out.json]"""
import collections, glob, json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
opt, pats, av = {}, [], sys.argv[1:]
i = 0
while i < len(av):
    if av[i].startswith('--'):
        opt[av[i]] = av[i + 1]; i += 2
    else:
        pats.append(av[i]); i += 1
pats = pats or [os.path.join(HERE, 'out', 'runs', '*.json')]
recs = {}
for p in pats:
    for f in sorted(glob.glob(p)):
        for l in open(f):
            if l.strip():
                r = json.loads(l)
                recs[(r['tag'], r['seed'])] = r            # last record per (tag, seed) wins
by = collections.defaultdict(list)
for (t, s), r in sorted(recs.items()):
    by[t].append(r)
want = opt.get('--tags')
tags = want.split(',') if want else sorted(by)
TASKS = ['what', 'slot', 'where', 'offer', 'taste', 'colour', 'count', 'ritual', 'check', 'name', 'before', 'oneshot']
ms = lambda xs: (float(np.mean(xs)), float(np.std(xs))) if xs else (None, None)
out = {}
for t in tags:
    R = by.get(t)
    if not R:
        continue
    o = dict(seeds=[r['seed'] for r in R], proto=R[0].get('proto'), viol=sum(r['viol'] for r in R))
    for k in ('J', 'J2', 'F', 'RET', 'H', 'DR'):
        o[k] = ms([r['score'][k] for r in R])
    d0 = [r['drift']['before']['ALL'][0] for r in R]; d1 = [r['drift']['after']['ALL'][0] for r in R]
    d2 = [r['drift']['after2']['ALL'][0] for r in R if r['drift'].get('after2')]
    o['drift_before'] = ms(d0); o['loss300'] = ms([a - b for a, b in zip(d0, d1)])
    o['loss900'] = ms([a - r['drift']['after2']['ALL'][0] for a, r in zip(d0, R) if r['drift'].get('after2')])
    pt = {}
    for task in TASKS:
        n = hn = 0; a = ha = 0.0
        for r in R:
            v = r['per_task'].get(task)
            if not v:
                continue
            if v['acc'] is not None and v['n']:
                a += v['acc'] * v['n']; n += v['n']
            if v['held'] is not None and v['n_held']:
                ha += v['held'] * v['n_held']; hn += v['n_held']
        pt[task] = dict(acc=round(a / n, 3) if n else None, n=n, held=round(ha / hn, 3) if hn else None, n_held=hn)
    o['per_task'] = pt
    o['role_conf'] = ms([r['role']['role_conf'] for r in R])
    o['role_conf_held'] = ms([r['role_held']['role_conf'] for r in R])
    o['role_conf_drift300'] = ms([r['drift']['role_after']['role_conf'] for r in R])
    o['role_conf_drift900'] = ms([r['drift']['role_after2']['role_conf'] for r in R if r['drift'].get('role_after2')])
    for g in ('all', 'held', 'seen'):
        o['refs_' + g] = {k: ms([r['refs'][g][k] for r in R if g in r['refs']]) for k in ('agent', 'echo', 'const_best', 'prior')}
    comp = collections.defaultdict(lambda: [0, 0, 0])
    for r in R:
        for k, v in (r.get('comp') or {}).items():
            for i in range(3):
                comp[k][i] += v[i]
    o['comp'] = {k: dict(n=v[0], comprehends=round(v[1] / v[0], 3) if v[0] else None, says=round(v[2] / v[0], 3) if v[0] else None)
                 for k, v in sorted(comp.items())}
    o['ral'] = R[0].get('ral')
    o['secs'] = ms([r['secs'] for r in R])
    out[t] = o
f2 = lambda m: '-' if m[0] is None else f'{m[0]:.3f}±{m[1]:.3f}'
for t, o in out.items():
    print(f"== {t} [{o['proto']}] seeds {o['seeds']} viol {o['viol']}  secs {o['secs'][0]:.0f}")
    print('   ' + '  '.join(f"{k} {f2(o[k])}" for k in ('J', 'J2', 'F', 'RET', 'H', 'DR')))
    print(f"   drift before {f2(o['drift_before'])} loss300 {f2(o['loss300'])} loss900 {f2(o['loss900'])} | role-conf {f2(o['role_conf'])} held {f2(o['role_conf_held'])} drift300 {f2(o['role_conf_drift300'])} drift900 {f2(o['role_conf_drift900'])}")
    print('   held-out by task: ' + '  '.join(f"{k} {v['held']}({v['n_held']})" for k, v in o['per_task'].items() if v['n_held']))
    print('   acc by task:      ' + '  '.join(f"{k} {v['acc']}" for k, v in o['per_task'].items() if v['n']))
    for g in ('all', 'held'):
        print(f"   refs {g}: " + '  '.join(f"{k} {f2(v)}" for k, v in o['refs_' + g].items()))
    print('   comp: ' + '  '.join(f"{k} {v['comprehends']}/{v['says']} (n{v['n']})" for k, v in o['comp'].items()))
if '--json' in opt:
    json.dump(out, open(opt['--json'], 'w'), ensure_ascii=False, indent=1)
