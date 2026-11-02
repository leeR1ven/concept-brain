# -*- coding: utf-8 -*-
"""Markdown table from ablate_synth.py's JSON: python3 lesion_md.py out/agent_final_s0_ablate.json"""
import sys, json
d = json.load(open(sys.argv[1]))
per, pooled = d['per'], d['pooled']
full = pooled['full'][2]
names = list(pooled)
def g(st, nm, k):
    r = per.get(f'{st}|{nm}', {})
    v = r.get(k)
    return v[0] if isinstance(v, list) and v and v[0] is not None else None
print('| part removed | pooled B+C+D | drop | B | C | D | key tasks | spoke / latency ok–wrong | verdict |')
print('|---|---|---|---|---|---|---|---|---|')
for nm in names:
    c, t, a = pooled[nm]
    drop = full - a
    B, C, D = (g(st, nm, 'ALL') for st in 'BCD')
    keys = []
    for st, task in (('B', 'what'), ('B', 'where'), ('C', 'offer'), ('C', 'colour'), ('C', 'count'), ('C', 'ritual'), ('D', 'check'), ('D', 'name')):
        v = g(st, nm, task)
        if v is not None:
            keys.append(f"{task} {v:.2f}")
    tim = [per.get(f'{st}|{nm}', {}).get('_timing', {}) for st in 'BCD']
    sp = '/'.join(f"{x.get('spoke', 0):.2f}" for x in tim)
    lat = '; '.join(f"{x.get('lat_ok')}–{x.get('lat_bad')}" for x in tim)
    verdict = '' if nm == 'full' else ('reaches behaviour' if drop >= 0.03 else ('weak' if drop >= 0.01 else 'passenger'))
    f = lambda x: '–' if x is None else f'{x:.2f}'
    print(f"| {nm} | {a:.3f} ({c}/{t}) | {drop:+.3f} | {f(B)} | {f(C)} | {f(D)} | {', '.join(keys)} | {sp}; {lat} | {verdict} |")
