# -*- coding: utf-8 -*-
"""Paired per-seed comparison of two run_mean tags (staged run JSON + post JSON + meaning side file).
usage: python3 paired_mean.py tagA tagB [dir]   (dir defaults to out/)"""
import json, glob, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
D = sys.argv[3] if len(sys.argv) > 3 else os.path.join(HERE, 'out')
TASKS = ('what', 'slot', 'where', 'offer', 'taste', 'colour', 'count', 'check')


def load(tag):
    out = {}
    for f in glob.glob(os.path.join(D, 'runs', f'{tag}_s*.json')):
        if f.endswith('.meaning.json'):
            continue
        for l in open(f):
            r = json.loads(l)
            if r['tag'] == tag:
                out[r['seed']] = dict(run=r)
    for s in list(out):
        p = os.path.join(D, 'post', f'{tag}_s{s}.json')
        if os.path.exists(p):
            out[s]['post'] = json.load(open(p))
        m = os.path.join(D, 'runs', f'{tag}_s{s}.meaning.json')
        if os.path.exists(m):
            out[s]['mean'] = json.load(open(m))
    return out


def metrics(x):
    r = x['run']
    d = r['drift']
    o = dict(r['score'])
    o['loss300'] = d['before']['ALL'][0] - d['after']['ALL'][0]
    o['loss900'] = d['before']['ALL'][0] - d['after2']['ALL'][0] if d.get('after2') else None
    o['role'] = r['role']['role_conf']
    o['viol'] = r['viol']
    o['held_vs_const'] = r['refs'].get('held', {}).get('agent')
    for t in TASKS + ('ritual',):
        o[t] = r['per_task'].get(t, {}).get('acc')
        o[t + '_h'] = r['per_task'].get(t, {}).get('held')
    p = x.get('post')
    if p:
        H = p.get('heldout', {})
        allh = []
        for st in 'BCD':
            for t, v in H.get(st, {}).items():
                if t.startswith('_') or 'held' not in v:
                    continue
                o['P' + t + '_h'] = v['held']['acc']
                o['P' + t + '_const'] = v['held']['const']
                allh.append((v['held']['acc'], v['held']['n']))
        if allh:
            o['Pheld_mean'] = float(np.mean([a for a, _ in allh]))
        F = p.get('fault')
        if F:
            o['f_clean'] = F['clean']['ALL']
            o['f_rand'] = F['rand']['ALL']
            o['f_sim'] = F['sim']['ALL']
            o['drop_rand'] = F['clean']['ALL'] - F['rand']['ALL']
            o['drop_sim'] = F['clean']['ALL'] - F['sim']['ALL']
        L = p.get('lesion')
        if L:
            for les in [k for k in L if k != 'intact']:
                o[f'les_{les}_all'] = np.mean([L[les][st]['all'] for st in 'BCD']) - np.mean([L['intact'][st]['all'] for st in 'BCD'])
                hi = [L['intact'][st]['held'] for st in 'BCD' if L['intact'][st]['held'] is not None]
                hl = [L[les][st]['held'] for st in 'BCD' if L[les][st]['held'] is not None]
                o[f'les_{les}_held'] = np.mean(hl) - np.mean(hi) if hi and hl else None
                for t in ('taste', 'offer', 'what', 'where', 'check'):
                    a = [L['intact'][st]['held_tasks'].get(t) for st in 'BCD' if L['intact'][st]['held_tasks'].get(t) is not None]
                    b = [L[les][st]['held_tasks'].get(t) for st in 'BCD' if L[les][st]['held_tasks'].get(t) is not None]
                    if a and b:
                        o[f'les_{les}_{t}_h'] = np.mean(b) - np.mean(a)
    m = x.get('mean')
    if m and m.get('codes'):
        c = m['codes'][-1]['stats']
        o['c_within'], o['c_between'], o['c_purity'], o['c_all'] = c['within'], c['between'], c['purity'], c['all_pairs']
        dr = [s['drift_from_prev'] for s in m['codes'][1:] if s['drift_from_prev'] is not None]
        o['c_drift_stages'] = '/'.join(f'{v:.2f}' for v in dr)
    o['secs'] = r['secs']
    return o


def main():
    a, b = load(sys.argv[1]), load(sys.argv[2])
    seeds = sorted(set(a) & set(b))
    print(f'{sys.argv[1]} -> {sys.argv[2]}, seeds {seeds}')
    ma = {s: metrics(a[s]) for s in seeds}
    mb = {s: metrics(b[s]) for s in seeds}
    keys = ['J', 'J2', 'F', 'RET', 'H', 'DR', 'loss300', 'loss900', 'role', 'held_vs_const', 'viol', 'ritual', 'colour', 'check']
    keys += [t + '_h' for t in TASKS]
    keys += ['Pheld_mean'] + ['P' + t + '_h' for t in TASKS] + ['f_clean', 'f_rand', 'f_sim', 'drop_rand', 'drop_sim']
    keys += sorted(set(k for s in seeds for k in mb[s] if k.startswith('les_')))
    keys += ['c_within', 'c_between', 'c_purity', 'c_all', 'secs']
    for k in keys:
        xa = [ma[s].get(k) for s in seeds]
        xb = [mb[s].get(k) for s in seeds]
        if all(x is None for x in xa) and all(x is not None for x in xb):
            print(f'  {k:16s} B only: ' + ' '.join(f'{y:.3f}' for y in xb) + f' | mean {np.mean(xb):.3f}')
            continue
        if any(x is None for x in xa + xb):
            continue
        d = np.array(xb, float) - np.array(xa, float)
        sign = '+' if (d > 0).all() else ('-' if (d < 0).all() else 'mixed')
        print(f'  {k:16s} ' + ' '.join(f'{x:.3f}->{y:.3f}' for x, y in zip(xa, xb)) +
              f' | A {np.mean(xa):.3f} B {np.mean(xb):.3f} delta {d.mean():+.3f}±{d.std():.3f} ({sign})')
    for s in seeds:
        if 'c_drift_stages' in mb[s]:
            print(f'  code drift between stages s{s}: {mb[s]["c_drift_stages"]}')
    for k in ('Ptaste_const', 'Poffer_const', 'Pwhat_const', 'Pwhere_const', 'Pslot_const', 'Pcheck_const'):
        v = [mb[s].get(k) for s in seeds if mb[s].get(k) is not None]
        if v:
            print(f'  ref {k}: {np.mean(v):.3f}')


if __name__ == '__main__':
    main()
