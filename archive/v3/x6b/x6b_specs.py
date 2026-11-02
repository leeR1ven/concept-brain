"""X6b specs. python3 x6b_specs.py cal   -> specs/cal_p*.json ;  python3 x6b_specs.py res <cal_choice.json> -> specs/r*_p*.json"""
import json
import math
import sys
import numpy as np

SEEDS = range(5)
PETS = ['小狗', '小猫', '鸭子', '小鸟']
NAMES = ['豆豆', '花花', '球球', '星星', '白白', '月月', '熊熊', '果果']
X3MIX = ['u', 'u', 'pS', 'u', 'pV', 'rB', 'u', 'pD', 'rS']


def sp(name, seed, t_end, cfg, **kw):
    return dict(name=f'{name}_s{seed}', seed=seed, t_end=t_end, cfg=cfg, kw=kw)


def eta(k):
    """eta x k on all hippocampal rates; decay keeps the age at which a one-exposure link reaches w_min"""
    if k == 1:
        return {}
    g = 0.99 ** (math.log(2.0 * k / 0.01) / math.log(2.0 / 0.01))
    return dict(eta_seq=2.0 * k, eta_co=0.2 * k, eta_rev=2.0 * k, eta_out=2.0 * k, gamma=round(g, 6))


def lives(arm, cfg, seeds=SEEDS):
    out = []
    for s in seeds:
        out.append(sp(f'qa_{arm}', s, 100000, dict(cfg, kinds=['u', 'qa'], iv=0.10), test_every=10000,
                      tests=['all', 'qa'], test_kw=dict(all=dict(n_old=10))))
        out.append(sp(f'uni_{arm}', s, 100000, dict(cfg), test_every=5000, tests=['all'], scale_at=[50000, 100000],
                      test_kw=dict(all=dict(n_old=10, quiets=['eye', 'dark']))))
        out.append(sp(f'sep_{arm}', s, 100000, dict(cfg, kinds=X3MIX), test_every=10000, tests=['all'],
                      test_kw=dict(all=dict(n_old=20))))
        names = [NAMES[i] for i in np.random.default_rng(500 + s).choice(len(NAMES), 4, replace=False)]
        for life, tf, te in (('fresh', 3000, 5000), ('adult', 300000, 100000)):
            plan = [[tf, p, n] for p, n in zip(PETS, names)]
            out.append(sp(f'{life}_{arm}', s, tf + 16000, dict(cfg), test_every=te, tests=['all'], fact_plan=plan,
                          test_kw=dict(all=dict(n_old=10))))
    return out


def ls_cfg(f, lam, alpha, rho=1.0):
    return dict(unified=True, ls=f, lam=lam, alpha=alpha, rho=rho)


LS_ARMS = {'L001': (0.01, 1.0), 'L03': (0.03, 1.0), 'L1': (0.1, 1.0), 'L3': (0.3, 1.0), 'L5': (0.5, 1.0),
           'L1r': (0.1, 0.5), 'L3r': (0.3, 0.5), 'L5r': (0.5, 0.5)}


def cal():
    out = []
    for arm, (f, rho) in LS_ARMS.items():
        for lam in (3.0, 6.0, 12.0, 24.0):
            for a in (0.3, 1.0):
                out.append(sp(f'cal_{arm}_l{lam}_a{a}', 9, 50000, ls_cfg(f, lam, a, rho), test_every=5000,
                              tests=['all'], test_kw=dict(all=dict(n_old=10))))
    for f in (0.03, 0.1, 0.3):
        for lam in (3.0, 6.0, 12.0, 24.0):
            for a in (0.3, 1.0):
                for k in (1, 10):
                    out.append(sp(f'cal_ZL_f{f}_l{lam}_a{a}_e{k}', 9, 50000, dict(ls_cfg(f, lam, a, 0.0), **eta(k)),
                                  test_every=5000, tests=['all'], test_kw=dict(all=dict(n_old=10))))
    return out


def result_arms(choice):
    arms = {'X3': {}, 'NS': dict(unified=True, lam=2.5)}
    for arm, (f, rho) in LS_ARMS.items():
        c = choice[arm]
        arms[arm] = ls_cfg(f, c['lam'], c['alpha'], rho)
    for k in (1, 4, 10):
        arms[f'Z{k}'] = dict(rho=0.0, **eta(k))
    c = choice['ZL']
    arms['ZL'] = dict(ls_cfg(c['f'], c['lam'], c['alpha'], 0.0), **eta(c['eta']))
    return arms


def split(specs, n):
    cost = {'qa': 1.0, 'uni': 1.6, 'sep': 0.9, 'fresh': 0.2, 'adult': 1.6}
    P = [[] for _ in range(n)]; load = [0.0] * n
    for x in sorted(specs, key=lambda x: -cost[x['name'].split('_')[0]]):
        i = int(np.argmin(load)); P[i].append(x); load[i] += cost[x['name'].split('_')[0]]
    return P, load


if __name__ == '__main__':
    if sys.argv[1] == 'cal':
        c = cal()
        P, load = split([dict(x, name='uni_' + x['name']) for x in c], 8)
        for i, p in enumerate(P):
            for x in p:
                x['name'] = x['name'][4:]
            json.dump(p, open(f'specs/cal_p{i}.json', 'w'))
        print(len(c), 'cal specs', [len(p) for p in P])
    else:
        choice = json.load(open(sys.argv[2]))
        arms = result_arms(choice)
        json.dump(arms, open('specs/arms.json', 'w'), indent=1)
        allspecs = sum([lives(a, c) for a, c in arms.items()], [])
        P, load = split(allspecs, 16)
        for i, p in enumerate(P):
            json.dump(p, open(f'specs/r_p{i}.json', 'w'))
        print(len(allspecs), 'result specs', [len(p) for p in P], [round(l, 1) for l in load])
