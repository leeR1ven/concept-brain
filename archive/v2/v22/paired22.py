# -*- coding: utf-8 -*-
"""Paired per-seed comparison of two run_v22 tags (same machine: A800 / numpy 2.1).
usage: python3 paired22.py tagA tagB [seeds=0,1,2,3,4]"""
import json, glob, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))


def load(tag):
    out = {}
    for f in glob.glob(os.path.join(HERE, 'out', 'runs', f'{tag}_s*.json')):
        for l in open(f):
            if l.strip():
                r = json.loads(l)
                if r['tag'] == tag:
                    out[r['seed']] = r
    return out


def m(r):
    d = r['drift']
    o = dict(r['score'])
    o['loss300'] = d['before']['ALL'][0] - d['after']['ALL'][0]
    o['loss900'] = d['before']['ALL'][0] - d['after2']['ALL'][0] if d.get('after2') else None
    for k, v in (d.get('after3') or {}).items():
        o['loss' + k] = d['before']['ALL'][0] - v['acc']
    o['role'] = r['role']['role_conf']
    o['held_vs_const'] = r['refs'].get('held', {}).get('agent')
    for t in ('ritual', 'colour', 'count', 'what', 'slot', 'where', 'taste', 'offer', 'check'):
        o[t] = r['per_task'].get(t, {}).get('acc')
        o[t + '_h'] = r['per_task'].get(t, {}).get('held')
    for les, v in (r.get('lesions') or {}).items():
        o['les_' + les.split(':')[0]] = (v['B'][0] + v['C'][0] + v['D'][0]) / 3
    o['F_'] = o['F']
    return o


KEYS = ['J', 'J2', 'F', 'RET', 'H', 'DR', 'loss300', 'loss900', 'loss1500', 'loss2100', 'loss3000', 'role',
        'held_vs_const', 'ritual', 'colour', 'count', 'check', 'what_h', 'slot_h', 'where_h', 'offer_h', 'taste_h',
        'colour_h', 'count_h', 'check_h']


def main():
    A = dict(a.split('=') for a in sys.argv[3:] if '=' in a)
    a, b = load(sys.argv[1]), load(sys.argv[2])
    seeds = sorted(set(a) & set(b))
    if 'seeds' in A:
        seeds = [s for s in seeds if str(s) in A['seeds'].split(',')]
    print(f'{sys.argv[1]} -> {sys.argv[2]}, seeds {seeds}')
    ma = {s: m(a[s]) for s in seeds}; mb = {s: m(b[s]) for s in seeds}
    keys = KEYS + sorted(k for k in set(ma[seeds[0]]) & set(mb[seeds[0]]) if k.startswith('les_')) if seeds else []
    for k in keys:
        xa = [ma[s].get(k) for s in seeds]; xb = [mb[s].get(k) for s in seeds]
        if any(x is None for x in xa + xb):
            continue
        d = np.array(xb) - np.array(xa)
        sg = "+" if (d > 0).all() else ("-" if (d < 0).all() else "mixed")
        print(f'  {k:13s} {np.mean(xa):.3f} -> {np.mean(xb):.3f} | delta {d.mean():+.3f} ± {d.std():.3f} ({sg}) | '
              + ' '.join(f'{x:.2f}>{y:.2f}' for x, y in zip(xa, xb)))


if __name__ == '__main__':
    main()
