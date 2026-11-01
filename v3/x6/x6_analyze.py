"""Summaries for X6. python3 -I x6_analyze.py <item> [out dir]. Reads out/<name>.json."""
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, 'out')
BINS = [0, 1000, 2000, 5000, 10000, 15000, 20000, 30000, 40000, 60000, 80000, 100000, 150000, 200000, 400000]


def load(prefix):
    R = {}
    for f in sorted(glob.glob(os.path.join(OUT, prefix + '*.json'))):
        d = json.load(open(f))
        R[d['spec']['name']] = d
    return R


PFX = ['']


def arm_seed(name):
    a, s = name[len(PFX[0]):].rsplit('_s', 1)
    return a, int(s)


def msd(v):
    v = [x for x in v if x is not None and not (isinstance(x, float) and np.isnan(x))]
    return (float(np.mean(v)), float(np.std(v)), len(v)) if v else (float('nan'), float('nan'), 0)


def fmt(v, p=2):
    m, s, n = msd(v)
    return f'{m:.{p}f}±{s:.{p}f}' + ('' if n >= 3 else f'(n{n})')


def mean_age(rows, key, lo, hi):
    v = [r[key] for r in rows if key in r and lo <= r['age'] < hi]
    return float(np.mean(v)) if v else float('nan')


def curve(rows, key):
    out = []
    for lo, hi in zip(BINS, BINS[1:]):
        v = [r[key] for r in rows if key in r and lo <= r['age'] < hi]
        if len(v) >= 3:
            out.append(((lo + hi) / 2, float(np.mean(v)), len(v)))
    return out


def cross_age(rows, key, level):
    """age at which the binned recall first falls below level (linear interpolation between bin centres)"""
    c = curve(rows, key)
    if not c:
        return float('nan')
    if c[0][1] < level:
        return c[0][0] * c[0][1] / level if level else 0.0
    for (a0, m0, _), (a1, m1, _) in zip(c, c[1:]):
        if m1 < level <= m0:
            return a0 + (m0 - level) / (m0 - m1) * (a1 - a0)
    return float('inf')


def paired(R, base, arm, fn):
    d, better = [], 0
    for s in range(10):
        a = next((v for n, v in R.items() if arm_seed(n) == (arm, s)), None)
        b = next((v for n, v in R.items() if arm_seed(n) == (base, s)), None)
        if a and b:
            x, y = fn(a), fn(b)
            if not (np.isnan(x) or np.isnan(y)):
                d.append(x - y); better += x > y
    return d, better


# ----------------------------------------------------------------------------------------------- items
def item17(prefix='q17_'):
    PFX[0] = prefix
    R = load(prefix)
    arms = [a for a in ('A0', 'A1', 'A1w', 'A2', 'R', 'A0p30', 'A1p30') if any(arm_seed(n)[0] == a for n in R)]
    V = ('F', 'P', 'SF', 'SP', 'IF', 'IP')

    def rate(d, v, what):
        q = d['qa']
        if not q:
            return float('nan')
        if what == 'right':
            return float(np.mean([r[v + '_first'] == r['ans'] for r in q]))
        if what == 'wrong':
            return float(np.mean([r[v + '_first'] not in (-1, r['ans']) for r in q]))
        if what == 'mate':
            return float(np.mean([r[v + '_first'] in r['mates'] for r in q]))
        return float(np.mean([r[v + '_hsA'] for r in q if v + '_hsA' in r]))

    side = {'i100<10k': lambda d: mean_age(d['all'], 'i100', 0, 10000),
            'r50<10k': lambda d: mean_age(d['all'], 'r50', 0, 10000)}
    print('item 17: right-A / wrong-A per recall variant (cue: re-hear Q | S = 50% stored code | I = ideal '
          're-hearing; continuation F = free run | P = pause input present again)')
    for a in arms:
        ds = [R[n] for n in sorted(R) if arm_seed(n)[0] == a]
        print(f'{a:6s} pairs {[len(d["qa"]) for d in ds]}  side i100<10k {fmt([side["i100<10k"](d) for d in ds])} '
              f'r50<10k {fmt([side["r50<10k"](d) for d in ds])}')
        for v in V:
            print(f'    {v:3s} right {fmt([rate(d, v, "right") for d in ds])}  wrong {fmt([rate(d, v, "wrong") for d in ds])}'
                  f'  mate {fmt([rate(d, v, "mate") for d in ds])}  hsA {fmt([rate(d, v, "hsA") for d in ds])}')
    for base, arm in (('A0', 'A1'), ('A0', 'A1w'), ('A0', 'A2'), ('A0', 'R'), ('A0p30', 'A1p30')):
        if arm not in arms:
            continue
        for v in V:
            d, b = paired(R, base, arm, lambda x: rate(x, v, 'right'))
            print(f'  paired right[{v}] {arm}-{base}: {fmt(d)}  better in {b}/{len(d)}')
        for k, f in side.items():
            d2, _ = paired(R, base, arm, f)
            print(f'      side {k} {arm}-{base}: {fmt(d2)}')


def item18(prefix='q18_'):
    PFX[0] = prefix
    R = load(prefix)
    arms = sorted({arm_seed(n)[0] for n in R}, key=lambda a: (a != 'B0', a))
    print('item 18: fresh recall without a switch')
    print(f'{"arm":12s} {"R":>8s} {"r50<10k":>11s} {"r50d<10k":>11s} {"i100<5k":>11s} {"i100d<5k":>11s} '
          f'{"i100x<5k":>11s} {"hs50":>11s} {"A90 r50":>9s}')
    for a in arms:
        ds = [R[n] for n in R if arm_seed(n)[0] == a]
        lam = ds[0]['spec']['cfg'].get('lam', 0.0)
        Rs = [np.mean([s['q_birth'] / (lam * s['q_rec']) for s in d['scale']]) if lam and d['scale'] else np.nan
              for d in ds]
        a90 = [cross_age(d['all'], 'r50', 0.9) for d in ds]
        print(f'{a:12s} {fmt(Rs, 2) if lam else "  -":>8s} {fmt([mean_age(d["all"], "r50", 0, 10000) for d in ds]):>11s} '
              f'{fmt([mean_age(d["all"], "r50d", 0, 10000) for d in ds]):>11s} '
              f'{fmt([mean_age(d["all"], "i100", 0, 5000) for d in ds]):>11s} '
              f'{fmt([mean_age(d["all"], "i100d", 0, 5000) for d in ds]):>11s} '
              f'{fmt([mean_age(d["all"], "i100x", 0, 5000) for d in ds]):>11s} '
              f'{fmt([mean_age(d["all"], "hs50", 0, 10000) for d in ds]):>11s} {np.nanmedian(a90):9.0f}')
    for a in arms:
        if a == 'B0':
            continue
        d, b = paired(R, 'B0', a, lambda x: mean_age(x['all'], 'r50', 0, 10000))
        print(f'  paired r50<10k {a}-B0: {fmt(d)}  better in {b}/{len(d)}')


def item19():
    R = load('q19_')
    keys = ('silent', 'single', 'real', 'chimera', 'anyheard', 'right')
    per = defaultdict(lambda: defaultdict(list))
    for n, d in R.items():
        for arm in ('random', 'cued'):
            tot = defaultdict(int)
            for k in d['kick']:
                for kk, v in k[arm].items():
                    tot[kk] += v
            N = tot['n']
            for kk in keys:
                per[arm][kk].append(tot[kk] / N)
            per[arm]['wrong_per_1000'].append(tot['wrong'] / N * 1000 / 12)
            per[arm]['wrong_in_per_1000'].append(tot['wrong_in'] / N * 1000 / 12)
            per[arm]['wrong_tc_per_1000'].append(tot['wrong_tc'] / N * 1000 / 12)
    print('item 19: kicks (fractions of kicks; wrong pairings per 1,000 quiet ticks)')
    for kk in keys + ('wrong_per_1000', 'wrong_in_per_1000', 'wrong_tc_per_1000'):
        r, c = per['random'][kk], per['cued'][kk]
        p = 0 if 'per' in kk else 3
        print(f'  {kk:18s} random {fmt(r, p if p else 1):>12s}  cued {fmt(c, p if p else 1):>12s}  '
              f'paired cued-random {fmt([b - a for a, b in zip(r, c)], p if p else 1)}')


def item21(prefix='q21_'):
    PFX[0] = prefix
    R = load(prefix)
    arms = sorted({arm_seed(n)[0] for n in R})
    print('item 21: horizon (ages in ticks where unique-probe recall crosses the level)')
    print(f'{"arm":16s} {"A50 r50":>14s} {"A90 r50":>14s} {"A50 i100":>14s} {"r50<5k":>10s} {"hs50<10k":>10s} secs rss')
    A = {}
    for a in arms:
        ds = [R[n] for n in R if arm_seed(n)[0] == a]
        a50 = [cross_age(d['all'], 'r50', 0.5) for d in ds]
        a90 = [cross_age(d['all'], 'r50', 0.9) for d in ds]
        i50 = [cross_age(d['all'], 'i100', 0.5) for d in ds]
        A[a] = a50
        print(f'{a:16s} {fmt(a50, 0):>14s} {fmt(a90, 0):>14s} {fmt(i50, 0):>14s} '
              f'{fmt([mean_age(d["all"], "r50", 0, 5000) for d in ds]):>10s} '
              f'{fmt([mean_age(d["all"], "hs50", 0, 10000) for d in ds]):>10s} '
              f'{int(np.mean([d["secs"] for d in ds]))} {max(d["rss_mb"] for d in ds)}')
    base = np.mean(A.get('s1', [np.nan]))
    for a in arms:
        print(f'  A50 ratio {a}/s1 = {np.mean(A[a]) / base:.2f}')
    for a in arms:
        print('  curve', a, [(int(x), round(m, 2)) for x, m, n in curve(sum([R[n]['all'] for n in R
                                                                          if arm_seed(n)[0] == a], []), 'r50')])


def item22(prefix='q22_'):
    PFX[0] = prefix
    R = load(prefix)
    print('item 22: one-shot fact (name recalled = step-1 decode is the name char)')
    for life in ('fresh', 'adult'):
        ds = [d for n, d in R.items() if n.startswith(f'{prefix}{life}')]
        if not ds:
            continue
        print(f' {life}: {len(ds)} lives')
        for tag in ('R1', 'R1s', 'R2', 'I1', 'S50'):
            per_seed = [np.mean([r[tag + '_ok'] for r in d['fact']]) for d in ds]
            byd = {dl: fmt([np.mean([r[tag + '_ok'] for r in d['fact'] if r['delay'] == dl]) for d in ds])
                   for dl in (1000, 2000, 5000, 10000)}
            full = [np.mean([r[tag + '_full'] for r in d['fact']]) for d in ds]
            print(f'   {tag:4s} name {fmt(per_seed)}  full {fmt(full)}  by delay {byd}')
        # intrusions at step 1 (R1)
        cnt = defaultdict(int); tot = 0
        for d in ds:
            names = {f['name'][0] for f in d['facts']}
            for r in d['fact']:
                tot += 1
                c = r['R1'][0]
                ch = None if c < 0 else d['chars'][c] if 'chars' in d else c
                if r['R1_ok']:
                    cnt['right'] += 1
                elif c < 0:
                    cnt['none'] += 1
                elif ch in names:
                    cnt['other new name'] += 1
                elif ch == '小':
                    cnt['old name (小)'] += 1
                else:
                    cnt['other'] += 1
        print('   R1 step-1 outcomes', {k: round(v / tot, 2) for k, v in cnt.items()})


def dsep():
    PFX[0] = 'dsep_'
    R = load('dsep_')
    for a in sorted({arm_seed(n)[0] for n in R}):
        ds = [R[n] for n in R if arm_seed(n)[0] == a]
        line = f'{a:6s} u r50<10k {fmt([mean_age([r for r in d["all"] if r["kind"] == "u"], "r50", 0, 10000) for d in ds])}'
        line += f' i100<10k {fmt([mean_age([r for r in d["all"] if r["kind"] == "u"], "i100", 0, 10000) for d in ds])}'
        for k in ('pS', 'pV', 'pD'):
            rr = [[r for r in d['all'] if r['kind'] == k and 'sep_right' in r and r['age'] < 20000] for d in ds]
            line += f' | {k} right {fmt([np.mean([r["sep_right"] for r in x]) if x else np.nan for x in rr])}'
            line += f' other {fmt([np.mean([r["sep_other"] for r in x]) if x else np.nan for x in rr])}'
        print(line)


if __name__ == '__main__':
    f = {'17': item17, '18': item18, '19': item19, '21': item21, '22': item22, 'sep': dsep}[sys.argv[1]]
    f(*sys.argv[3:4]) if len(sys.argv) > 3 else f()
