# -*- coding: utf-8 -*-
"""Paired per-seed comparison of two run_v21 tags.  usage: python3 paired.py tagA tagB"""
import json, glob, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
def load(tag):
    out = {}
    for f in glob.glob(os.path.join(HERE, 'out', 'runs', f'{tag}_s*.json')):
        for l in open(f):
            r = json.loads(l)
            if r['tag'] == tag:
                out[r['seed']] = r
    return out
a, b = load(sys.argv[1]), load(sys.argv[2])
seeds = sorted(set(a) & set(b))
def m(r):
    d = r['drift']
    o = dict(r['score'])
    o['loss300'] = d['before']['ALL'][0] - d['after']['ALL'][0]
    o['loss900'] = d['before']['ALL'][0] - d['after2']['ALL'][0] if d.get('after2') else None
    o['role'] = r['role']['role_conf']
    for t in ('ritual', 'colour', 'what', 'slot', 'where', 'taste', 'offer', 'check'):
        o[t] = r['per_task'].get(t, {}).get('acc')
        o[t + '_h'] = r['per_task'].get(t, {}).get('held')
    return o
keys = ['J', 'J2', 'F', 'RET', 'H', 'DR', 'loss300', 'loss900', 'role', 'ritual', 'colour', 'what_h', 'slot_h', 'where_h']
print(f'{sys.argv[1]} -> {sys.argv[2]}, seeds {seeds}')
for k in keys:
    xa = [m(a[s])[k] for s in seeds]; xb = [m(b[s])[k] for s in seeds]
    if any(x is None for x in xa + xb):
        continue
    d = np.array(xb) - np.array(xa)
    d14 = [dd for s, dd in zip(seeds, d) if s >= 1]
    print(f'  {k:8s} ' + ' '.join(f'{x:.3f}->{y:.3f}' for x, y in zip(xa, xb)) +
          f' | delta {d.mean():+.3f}±{d.std():.3f} (seeds>=1 {np.mean(d14):+.3f}, all {"+" if (d > 0).all() else ("-" if (d < 0).all() else "mixed")})')
