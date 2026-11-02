"""tables for x4b.json: day 0 (secondary) and life after birth (main). Run: python3 -I x4b_table.py [x4b.json]"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
D = json.load(open(sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, 'x4b.json')))
REPS = (0, 1, 2, 3, 5, 10)


def f2(x):
    return f'{x:.2f}'


def nsib(r):
    return len(r['parents'])


def boot_ci(d, n=4000, seed=0):
    rng = np.random.default_rng(seed)
    d = np.asarray(d, float)
    m = rng.choice(d, (n, len(d))).mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


for scen in dict.fromkeys(r['scenario'] for r in D):
    rs = [r for r in D if r['scenario'] == scen]
    k = nsib(rs[0])
    R = len(rs[0]['life']['fresh']['shared']['curve']) - 1
    print(f'\n=== {scen}  ({len(rs)} seeds, {k} parents, life {R} rounds)')
    par = np.mean([r['parents'][str(i)]['own']['right'] for r in rs for i in range(k)])
    print(f'parents, own lessons right: {f2(par)}')
    # ---- day 0
    print('\nDAY 0 | op | shared right | disjoint right | disjoint wrong | conflict none | per-parent disjoint right | code Jaccard')
    for op in rs[0]['day0']:
        v = [r['day0'][op] for r in rs]
        dr = [np.mean([x[f'only{i}']['right'] for x in v]) for i in range(k)]
        dn = [np.mean([x[f'only{i}']['none'] for x in v]) for i in range(k)]
        jac = np.mean([x['code_jaccard_vs_ref'] for x in v]) if 'code_jaccard_vs_ref' in v[0] else None
        print(f"| {op} | {f2(np.mean([x['shared']['right'] for x in v]))} | {f2(np.mean(dr))} | "
              f"{f2(1 - np.mean(dr) - np.mean(dn))} | {f2(np.mean([x['conflict']['none'] for x in v]))} | "
              f"{' / '.join(f2(x) for x in dr)} | {'' if jac is None else f2(jac)}")
    # ---- life
    print('\nLIFE | agent | target | right after ' + ' / '.join(map(str, REPS)) +
          ' reps | mean reps to first correct | savings vs fresh | paired diff vs fresh [95% CI] | reps to 0.8 | '
          'reps to stable | conflict none @R | shared @R')
    agents = list(rs[0]['life'])

    def sets_of(tgt, me):
        if tgt == 'all':
            return ['shared'] + [f'only{i}' for i in range(k)]
        if tgt == 'disjoint':
            return [f'only{i}' for i in range(k)]
        return [f'only{i}' for i in range(k) if i != me]           # a parent's new lessons

    def gather(ag, tgt, me):
        curves, firsts, stables, conf, shr, r08 = [], [], [], [], [], []
        for r in rs:
            L = r['life'][ag]
            sets = sets_of(tgt, me)
            n = np.array([len(L[s]['first']) for s in sets], float)
            c = np.array([L[s]['curve'] for s in sets]).T @ n / n.sum()
            curves.append(c)
            firsts.append(sum((L[s]['first'] for s in sets), []))
            stables.append(sum((L[s]['stable'] for s in sets), []))
            conf.append(L['conflict']['curve'][-1]); shr.append(L['shared']['curve'][-1])
            hit = np.flatnonzero(c >= 0.8)
            r08.append(int(hit[0]) if len(hit) else R + 1)
        return np.mean(curves, 0), np.concatenate(firsts), np.concatenate(stables), conf, shr, r08

    for ag in ['fresh'] + [a for a in agents if a != 'fresh']:
        me = int(ag[-1]) if ag.startswith('parent') else -1
        for tgt in (('all', 'new') if me >= 0 else ('all', 'disjoint')):
            c, fa, sa, conf, shr, r08 = gather(ag, tgt, me)
            sav = pd = ''
            if ag != 'fresh':
                ref = gather('fresh', tgt, me)[1]
                sav = f2(1 - fa.mean() / ref.mean())
                d = fa - ref
                lo, hi = boot_ci(d)
                pd = f'{d.mean():+.2f} [{lo:+.2f}, {hi:+.2f}]'
            print(f"| {ag} | {tgt} (n={len(fa)}) | {' / '.join(f2(c[i]) for i in REPS if i <= R)} | {f2(fa.mean())} | "
                  f"{sav} | {pd} | {np.mean(r08):.1f} | {f2(sa.mean())} | {f2(np.mean(conf))} | {f2(np.mean(shr))} |")
    # ---- which lessons are innate: the owning parent's margin
    print('\nINNATE vs owning parent margin (disjoint lessons; innate = right at day 0 in the child)')
    for ag in [a for a in agents if not a.startswith('parent') and a != 'fresh']:
        inn, mg, owner = [], [], []
        for r in rs:
            L = r['life'][ag]
            for i in range(k):
                f = np.array(L[f'only{i}']['first'])
                inn += list(f == 0); mg += r['margin'][str(i)]; owner += [i] * len(f)
        inn, mg, owner = np.array(inn), np.array(mg), np.array(owner)
        med = np.median(mg)
        known = mg > 1.25                                      # the parent itself releases (approximately)
        print(f"| {ag} | innate {f2(inn.mean())} | parent margin > median: {f2(inn[mg > med].mean())} | "
              f"<= median: {f2(inn[mg <= med].mean())} | parent margin < 1.25: {f2(inn[~known].mean()) if (~known).any() else '-'} | "
              f"by owner: {' / '.join(f2(inn[owner == i].mean()) for i in range(k))}")
