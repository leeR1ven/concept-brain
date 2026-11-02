"""generate the X6 result specs (seeds 0-4; seed 9 was calibration only)"""
import json
import numpy as np
SEEDS5, SEEDS3 = range(5), range(3)
PETS = ['小狗', '小猫', '鸭子', '小鸟']
NAMES = ['豆豆', '花花', '球球', '星星', '白白', '月月', '熊熊', '果果']


def sp(name, seed, t_end, cfg, **kw):
    return dict(name=f'{name}_s{seed}', seed=seed, t_end=t_end, cfg=cfg, kw=kw)


def item17():
    arms = [('A0', {}), ('A1', dict(iv=0.25)), ('A1w', dict(iv=0.10)), ('A2', dict(win=15)),
            ('R', dict(step_pause=False)), ('A0p30', dict(pause=30)), ('A1p30', dict(pause=30, iv=0.25))]
    return [sp(f'q17_{a}', s, 100000, dict(kinds=['u', 'qa'], **c), test_every=10000, tests=['all', 'qa'],
               test_kw=dict(all=dict(n_old=10))) for a, c in arms for s in SEEDS5]


def item18():
    arms = [('B0', {})] + [(f'Bl{l}', dict(unified=True, lam=l)) for l in (1.5, 2.5, 4.0, 6.5, 10.0)] + \
           [('Bstale4', dict(unified=True, lam=4.0, stale=True))]
    return [sp(f'q18_{a}', s, 100000, c, test_every=5000, tests=['all'], scale_at=[50000, 100000],
               test_kw=dict(all=dict(n_old=10, quiets=['eye', 'dark']))) for a, c in arms for s in SEEDS5]


def item19():
    return [sp('q19_base', s, 200000, {}, test_every=50000, tests=['all'], kick_at=[50000, 100000, 150000, 200000],
               test_kw=dict(all=dict(n_old=10), kick=dict(n=100))) for s in SEEDS5]


def item22():
    out = []
    for s in SEEDS5:
        names = [NAMES[i] for i in np.random.default_rng(500 + s).choice(len(NAMES), 4, replace=False)]
        for life, tf, te in (('fresh', 3000, 5000), ('adult', 300000, 100000)):
            plan = [[tf, p, n] for p, n in zip(PETS, names)]
            out.append(sp(f'q22_{life}', s, tf + 16000, {}, test_every=te, tests=['all'], fact_plan=plan,
                          test_kw=dict(all=dict(n_old=10))))
    return out


def item21_cfg(sz, arm):
    k = 64 * sz if arm in 'bd' else 64
    C = 2000 * sz if arm in 'cd' else 2000
    return dict(N_H=10000 * sz, k=k, C=C, cB=int(400 * sz * 64 / k))


def item21():
    out = [sp('q21_s1', s, 200000, {}, test_every=10000, tests=['all'],
              test_kw=dict(all=dict(n_recent=60, n_old=40))) for s in SEEDS3]
    for sz in (2, 4):
        for arm in 'abcd':
            out += [sp(f'q21_s{sz}{arm}', s, 200000, item21_cfg(sz, arm), test_every=10000, tests=['all'],
                       test_kw=dict(all=dict(n_recent=60, n_old=40))) for s in SEEDS3]
    return out


if __name__ == '__main__':
    a17, a18, a19, a22, a21 = item17(), item18(), item19(), item22(), item21()
    # job 1: four processes of similar cost
    json.dump(a17[:18], open('specs/m1_p0.json', 'w')); json.dump(a17[18:], open('specs/m1_p1.json', 'w'))
    json.dump(a18[:18], open('specs/m1_p2.json', 'w'))
    json.dump(a18[18:] + a19 + a22, open('specs/m1_p3.json', 'w'))
    # job 2: item 21; the 3 heaviest (s4d) in separate processes
    heavy = [x for x in a21 if x['name'].startswith('q21_s4d')]
    rest = [x for x in a21 if not x['name'].startswith('q21_s4d')]
    cost = {'s1': 1, 's2a': 1.2, 's2b': 2, 's2c': 2, 's2d': 4, 's4a': 1.5, 's4b': 4, 's4c': 4}
    P = [[h] for h in heavy] + [[]]
    load = [16, 16, 16, 0]
    for x in sorted(rest, key=lambda x: -cost[x['name'].split('_')[1]]):
        i = int(np.argmin(load)); P[i].append(x); load[i] += cost[x['name'].split('_')[1]]
    for i, p in enumerate(P):
        json.dump(p, open(f'specs/m2_p{i}.json', 'w'))
    print('loads', load, [len(p) for p in P], len(a17), len(a18), len(a19), len(a22), len(a21))


def item21_decay(arms):
    """decay ∝ 1/s on the two arms with the largest A50 at s = 4; life 400 k; s = 1 base at 400 k too"""
    out = [sp('q21g_s1', s, 400000, {}, test_every=10000, tests=['all'],
              test_kw=dict(all=dict(n_recent=60, n_old=40))) for s in SEEDS3]
    for sz in (2, 4):
        for arm in arms:
            cfg = dict(item21_cfg(sz, arm), gamma=float(0.99 ** (1.0 / sz)))
            out += [sp(f'q21g_s{sz}{arm}', s, 400000, cfg, test_every=10000, tests=['all'],
                       test_kw=dict(all=dict(n_recent=60, n_old=40))) for s in SEEDS3]
    return out


def diag():
    """follow-up diagnostics (not pre-registered): item 18 split by side; birth-recurrence strength and replay"""
    out = []
    for L in (2.5, 6.5, 20.0):
        out += [sp(f'd18_D1r{L}', s, 100000, dict(unified=True, lam=0.0, lam_rec=L), test_every=5000, tests=['all'],
                   test_kw=dict(all=dict(n_old=10, quiets=['eye', 'dark']))) for s in SEEDS5]
    out += [sp('d18_D2e4', s, 100000, dict(lam=4.0), test_every=5000, tests=['all'],
               test_kw=dict(all=dict(n_old=10))) for s in SEEDS5]
    for rho in (0.5, 0.25):
        for x in item22():
            if True:
                y = json.loads(json.dumps(x)); y['cfg']['rho'] = rho
                y['name'] = y['name'].replace('q22_', f'dr{rho}_22').replace('22fresh', '22_fresh').replace('22adult', '22_adult')
                out.append(y)
        for x in item17():
            a = x['name'].split('_')[1]
            if a in ('A0', 'A1w'):
                y = json.loads(json.dumps(x)); y['cfg']['rho'] = rho
                y['name'] = y['name'].replace('q17_', f'dr{rho}_17_'); out.append(y)
    for rho in (1.0, 0.5, 0.25):
        out += [sp(f'dsep_r{rho}', s, 100000, dict(rho=rho, kinds=['u', 'u', 'pS', 'u', 'pV', 'rB', 'u', 'pD', 'rS']),
                   test_every=10000, tests=['all'], test_kw=dict(all=dict(n_old=20))) for s in SEEDS3]
    return out
