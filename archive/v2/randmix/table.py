# -*- coding: utf-8 -*-
"""Markdown tables from out/runs/*.json (J, analyze.py's objective unchanged) and out/post/*.json (held-out per task
with trivial references, fault tolerance, drift 300/900).  usage: python3 table.py [tag1,tag2,...] [seeds=0,1]"""
import glob, json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(1, os.path.join(HERE, '..', 'synth'))   # repo: the unchanged ARCH_v2 core and its bench_copy
from analyze import score   # noqa: E402

A = dict(a.split('=', 1) for a in sys.argv[1:] if '=' in a)
tags = [a for a in sys.argv[1:] if '=' not in a]
tags = tags[0].split(',') if tags else None
seeds = [int(s) for s in A.get('seeds', '0,1').split(',')]


def runs(tag):
    out = {}
    for s in seeds:
        f = os.path.join(HERE, 'out', 'runs', f'{tag}_s{s}.json')
        if os.path.exists(f):
            out[s] = json.loads(open(f).read().strip().splitlines()[-1])
    return out


def post(tag):
    out = {}
    for s in seeds:
        f = os.path.join(HERE, 'out', 'post', f'{tag}_s{s}.json')
        if os.path.exists(f):
            out[s] = json.load(open(f))
    return out


def m(xs):
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else None


def f2(x):
    return '—' if x is None else f'{x:.2f}'


if tags is None:
    tags = sorted(set(os.path.basename(f).rsplit('_s', 1)[0] for f in glob.glob(os.path.join(HERE, 'out', 'runs', '*.json'))))

cols = ['J', 'F', 'RET', 'H', 'DR', 'B3', 'C3', 'D3', 'what', 'slot', 'where', 'offer', 'taste', 'colour', 'count',
        'ritual', 'check', 'name']
print('### J (analyze.py), mean over seeds ' + ','.join(map(str, seeds)))
print('| tag | seeds | J per seed | ' + ' | '.join(cols) + ' | viol | units end | secs | RSS MB |')
print('|' + '---|' * (len(cols) + 7))
for t in tags:
    R = runs(t)
    if not R:
        continue
    S = {s: score(r) for s, r in R.items()}
    row = [t, ','.join(map(str, sorted(R))), ' / '.join(f"{S[s]['J']:.2f}" for s in sorted(S))]
    row += [f2(m([S[s].get(c) for s in S])) for c in cols]
    row += [str(sum(r['viol'] for r in R.values())), str(int(m([r['mix'] for r in R.values()]))),
            str(int(m([r['secs'] for r in R.values()]))), str(int(m([r.get('rss_mb') or 0 for r in R.values()])))]
    print('| ' + ' | '.join(row) + ' |')

tasks = ['taste', 'check', 'offer', 'slot', 'where', 'what', 'colour', 'count']
print('\n### Held-out per task (post.py, held_frac 0.5, 150 dialogues per stage): agent / echo / best constant (n)')
print('| tag | ' + ' | '.join(tasks) + ' | mean agent | mean const |')
print('|' + '---|' * (len(tasks) + 3))
for t in tags:
    P = post(t)
    P = {s: p for s, p in P.items() if 'heldout' in p}
    if not P:
        continue
    row, ag, co = [t], [], []
    for k in tasks:
        a, e, c, n = [], [], [], 0
        for p in P.values():
            for st in 'BCD':
                v = p['heldout'].get(st, {}).get(k, {}).get('held')
                if v:
                    a.append(v['acc'] * v['n']); e.append(v['echo'] * v['n']); c.append(v['const'] * v['n']); n += v['n']
        if n:
            row.append(f"{sum(a) / n:.2f} / {sum(e) / n:.2f} / {sum(c) / n:.2f} ({n})")
            ag.append(sum(a) / n); co.append(sum(c) / n)
        else:
            row.append('—')
    row += [f2(m(ag)), f2(m(co))]
    print('| ' + ' | '.join(row) + ' |')

print('\n### Fault tolerance (run_j final probe worlds, one prompt syllable replaced): accuracy B+C+D')
print('| tag | clean | random syllable | similar-sounding | drop random | drop similar | seen clean→rand |')
print('|---|---|---|---|---|---|---|')
for t in tags:
    P = {s: p for s, p in post(t).items() if 'fault' in p}
    if not P:
        continue
    c = m([p['fault']['clean']['ALL'] for p in P.values()])
    r = m([p['fault']['rand']['ALL'] for p in P.values()])
    si = m([p['fault']['sim']['ALL'] for p in P.values()])
    sc = m([m([p['fault']['clean'][st]['seen'] for st in 'BCD']) for p in P.values()])
    sr = m([m([p['fault']['rand'][st]['seen'] for st in 'BCD']) for p in P.values()])
    print(f"| {t} | {f2(c)} | {f2(r)} | {f2(si)} | {f2(c - r)} | {f2(c - si)} | {f2(sc)}→{f2(sr)} |")

print('\n### Drift (C probe, untutored C dialogues with learning on)')
print('| tag | before | after 300 | after 900 | loss 300 | loss 900 |')
print('|---|---|---|---|---|---|')
for t in tags:
    P = {s: p for s, p in post(t).items() if 'drift' in p}
    if not P:
        continue
    b = m([p['drift']['before']['all'] for p in P.values()])
    a3 = m([p['drift']['after300']['all'] for p in P.values()])
    a9 = m([p['drift'].get('after900', {}).get('all') for p in P.values()])
    print(f"| {t} | {f2(b)} | {f2(a3)} | {f2(a9)} | {f2(a3 - b)} | {f2(None if a9 is None else a9 - b)} |")

print('\n### Size over time (units / synapses / ever-active units; RSS)')
for t in tags:
    R = runs(t)
    for s, r in sorted(R.items()):
        print(t, s, [(z['stage'], z.get('units'), z.get('synapses'), z.get('ever_active'), z.get('rss_mb'), z.get('t'))
                     for z in r.get('sizes', [])], (r.get('mixstats') or {}).get('hits'), (r.get('mixstats') or {}).get('calls'))
