"""main (life) table from x4b.json + x4b_extra.json. Taught lessons = shared + every disjoint lesson.
Columns: innate share (right at day 0), right after 1 / 2 / 3 / 10 reps, mean reps to first correct (never within
R -> R + 1), savings vs fresh = 1 - reps/reps_fresh, paired per-lesson difference vs fresh with a bootstrap 95 % CI,
and the speed on the lessons NOT known at birth (paired with fresh on the same lessons).
Run: python3 -I x4b_main_table.py"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
M = json.load(open(os.path.join(HERE, 'x4b.json')))
E = json.load(open(os.path.join(HERE, 'x4b_extra.json')))


def boot(d, n=4000):
    rng = np.random.default_rng(0)
    m = rng.choice(np.asarray(d, float), (n, len(d))).mean(1)
    return np.percentile(m, 2.5), np.percentile(m, 97.5)


ORDER = ['fresh', 'mean', 'sign', 'cross', 'cross+noise', 'cross+noise_all', 'cross/learned', 'cross+noise x0.3',
         'cross+noise x0.1', 'sign+noise', 'sign+noise x0.3', 'sign+noise x0.1', 'parent0', 'parent1']
for scen in dict.fromkeys(r['scenario'] for r in M):
    rm = [r for r in M if r['scenario'] == scen]
    re_ = [r for r in E if r['scenario'] == scen]
    for a, b in zip(rm, re_):                                 # determinism check of the shared fresh baseline
        assert a['seed'] == b['seed'] and a['life']['fresh'] == b['life']['fresh'], (scen, a['seed'])
    k = len(rm[0]['parents'])
    sets = ['shared'] + [f'only{i}' for i in range(k)]
    R = len(rm[0]['life']['fresh']['shared']['curve']) - 1

    def get(ag, which):
        cur, first = [], []
        for a, b in zip(rm, re_):
            L = a['life'].get(ag) or b['life'].get(ag)
            ss = sets if which == 'all' else [s for s in sets if s != 'shared']
            n = np.array([len(L[s]['first']) for s in ss], float)
            cur.append(np.array([L[s]['curve'] for s in ss]).T @ n / n.sum())
            first.append(sum((L[s]['first'] for s in ss), []))
        return np.mean(cur, 0), np.concatenate(first)

    print(f'\n=== {scen}: life after birth, {len(rm)} seeds, {R} rounds, taught = shared + disjoint')
    print('| agent | innate all / disjoint | right after 1 / 2 / 3 / 10 reps | mean reps | savings vs fresh | '
          'paired diff [95% CI] | not known at birth: n, reps child vs fresh, savings |')
    cf, ff = get('fresh', 'all')
    agents = [a for a in ORDER if a in rm[0]['life'] or a in re_[0]['life']]
    for ag in agents:
        c, f = get(ag, 'all')
        cd, _ = get(ag, 'disjoint')
        if ag.startswith('parent'):
            sav = pd = nk = ''
        else:
            sav = f'{1 - f.mean() / ff.mean():+.2f}' if ag != 'fresh' else ''
            d = f - ff
            lo, hi = boot(d)
            pd = f'{d.mean():+.2f} [{lo:+.2f}, {hi:+.2f}]' if ag != 'fresh' else ''
            m = f > 0
            nk = (f'{m.sum()}, {f[m].mean():.2f} vs {ff[m].mean():.2f}, {1 - f[m].mean() / ff[m].mean():+.2f}'
                  if ag != 'fresh' else '')
        print(f"| {ag} | {c[0]:.2f} / {cd[0]:.2f} | {c[1]:.2f} / {c[2]:.2f} / {c[3]:.2f} / {c[R]:.2f} | {f.mean():.2f} | "
              f"{sav} | {pd} | {nk} |")
    print('day 0 of the extra rows (shared right, disjoint right, disjoint wrong, conflict none):')
    for key in re_[0]['day0']:
        v = [r['day0'][key] for r in re_]
        dr = np.mean([np.mean([x[f'only{i}']['right'] for i in range(k)]) for x in v])
        dn = np.mean([np.mean([x[f'only{i}']['none'] for i in range(k)]) for x in v])
        print(f"  {key}: {np.mean([x['shared']['right'] for x in v]):.2f}, {dr:.2f}, {1 - dr - dn:.2f}, "
              f"{np.mean([x['conflict']['none'] for x in v]):.2f}")
