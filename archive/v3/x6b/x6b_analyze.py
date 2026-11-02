"""X6b tables. python3 -I x6b_analyze.py cal [dir]  |  python3 -I x6b_analyze.py res [dir] [arms...]"""
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BINS = [0, 1000, 2000, 5000, 10000, 15000, 20000, 30000, 40000, 60000, 80000, 100000]


def load(d, pat):
    R = {}
    for f in sorted(glob.glob(os.path.join(d, pat))):
        x = json.load(open(f))
        R[x['spec']['name']] = x
    return R


def mean_age(rows, key, lo, hi, kinds=None):
    v = [r[key] for r in rows if key in r and lo <= r['age'] < hi and (kinds is None or r['kind'] in kinds)]
    return float(np.mean(v)) if v else float('nan')


def cross_age(rows, key, level, kinds=('u',)):
    c = []
    for lo, hi in zip(BINS, BINS[1:]):
        v = [r[key] for r in rows if key in r and lo <= r['age'] < hi and r['kind'] in kinds]
        if len(v) >= 3:
            c.append(((lo + hi) / 2, float(np.mean(v))))
    if not c:
        return float('nan')
    if c[0][1] < level:
        return c[0][0] * c[0][1] / level
    for (a0, m0), (a1, m1) in zip(c, c[1:]):
        if m1 < level <= m0:
            return a0 + (m0 - level) / (m0 - m1) * (a1 - a0)
    return float('inf')


def ovstats(rows, key='ov', pos=None, cond=None):
    v = [r[key] for r in rows if key in r and r[key] and (cond is None or cond(r))]
    if not v:
        return float('nan')
    L = min(len(x) for x in v)
    a = np.array([x[:L] for x in v])
    return float(a.mean()) if pos is None else float(a[:, pos].mean())


# metrics per life: name -> fn(result) -> float
def m_uni(d):
    a = [r for r in d['all'] if r['kind'] == 'u']
    y = [r for r in a if r['age'] < 10000]
    return {'r50': mean_age(a, 'r50', 0, 10000), 'r50d': mean_age(a, 'r50d', 0, 10000),
            'i100': mean_age(a, 'i100', 0, 10000),
            'ov': ovstats(y), 'ov6': ovstats(y, pos=5), 'ov1': ovstats(y, pos=0), 'ovx': ovstats(y, 'ovx'),
            'ovn_other': ovstats(y, 'ovn', cond=lambda r: not r['same_scene']),
            'n_other': float(sum(not r['same_scene'] for r in y)),
            'rd': mean_age(a, 'rd', 0, 10000), 'r50x3': mean_age(a, 'r50x3', 0, 10000), 'persist': mean_age(a, 'persist', 0, 10000),
            'A50': cross_age(d['all'], 'r50', 0.5), 'live': float(d['live']),
            'on_sp': d['ls_stat']['on_speech'], 'on_si': d['ls_stat']['on_silence']}


def m_qa(d):
    q = d['qa']
    o = {}
    for v in ('P', 'F', 'IP'):
        o[f'qa{v}_right'] = float(np.mean([r[v + '_first'] == r['ans'] for r in q])) if q else float('nan')
        o[f'qa{v}_wrong'] = float(np.mean([r[v + '_first'] not in (-1, r['ans']) for r in q])) if q else float('nan')
    o['ovQ'] = ovstats(q, 'ovQ')
    a = [r for r in d['all'] if r['kind'] == 'u']
    o['qa_i100'] = mean_age(a, 'i100', 0, 10000); o['qa_r50'] = mean_age(a, 'r50', 0, 10000)
    return o


def m_sep(d):
    o = {}
    for k in ('pS', 'pV', 'pD'):
        rr = [r for r in d['all'] if r['kind'] == k and 'sep_right' in r and r['age'] < 20000]
        o[f'sep_{k}'] = float(np.mean([r['sep_right'] for r in rr])) if rr else float('nan')
        o[f'sepo_{k}'] = float(np.mean([r['sep_other'] for r in rr])) if rr else float('nan')
    a = [r for r in d['all'] if r['kind'] == 'u']
    o['sep_u_r50'] = mean_age(a, 'r50', 0, 10000)
    return o


def m_fact(d, life):
    f = d['fact']
    o = {}
    for tag in ('R1s', 'R1', 'I1', 'S50'):
        o[f'{life}_{tag}'] = float(np.mean([r[tag + '_ok'] for r in f]))
    o[f'{life}_ov7'] = ovstats(f, 'ov7')
    o[f'{life}_ov7last'] = ovstats(f, 'ov7', pos=6)
    o[f'{life}_ov7n_other'] = ovstats(f, 'ov7n', cond=lambda r: not r['same_scene'])
    return o


def per_arm(R, arm):
    seeds = {}
    for n, d in R.items():
        life, rest = n.split('_', 1)
        a, s = rest.rsplit('_s', 1)
        if a != arm:
            continue
        m = seeds.setdefault(int(s), {})
        if life == 'uni':
            m.update(m_uni(d))
        elif life == 'qa':
            m.update(m_qa(d))
        elif life == 'sep':
            m.update(m_sep(d))
        elif life in ('fresh', 'adult'):
            m.update(m_fact(d, life))
    return seeds


def fmt(v, p=2):
    v = [x for x in v if not (isinstance(x, float) and (np.isnan(x)))]
    if not v:
        return '  -  '
    if any(np.isinf(v)):
        return 'inf'
    return f'{np.mean(v):.{p}f}±{np.std(v):.{p}f}' + ('' if len(v) >= 5 else f'(n{len(v)})')


KEYS = [('r50', 'T1/T6 r50 (eye) <10k'), ('r50d', '    r50 dark'), ('i100', 'T6 i100 <10k'),
        ('ov', 'T2 ov same scene, mean c1-6'), ('ov1', '    ov char 1'), ('ov6', '    ov char 6'),
        ('ovx', '    ovx other-line control'), ('ovn_other', '    ov other scene (measure)'),
        ('fresh_ov7', '    fact line ov (7 chars, same scene)'), ('fresh_ov7n_other', '    fact line ov other scene'),
        ('ovQ', '    Q line ov (qa life)'),
        ('fresh_R1s', 'T3 fresh R1s same scene'), ('adult_R1s', 'T3 adult R1s same scene'),
        ('fresh_R1', '    fresh R1 test scene'), ('adult_R1', '    adult R1 test scene'),
        ('fresh_I1', '    fresh I1 ideal'), ('adult_I1', '    adult I1 ideal'),
        ('qaP_right', 'T4 qa natural right'), ('qaP_wrong', 'T4 qa natural wrong'), ('qaIP_right', '    qa ideal right'),
        ('qa_i100', '    qa-life i100'), ('qa_r50', '    qa-life r50'),
        ('sep_pS', 'T5 sep pS same scene+voice (report)'), ('sep_pV', 'T5 sep pV other voice'),
        ('sep_pD', 'T5 sep pD other scene'), ('sepo_pS', '    pS gives twin'), ('sep_u_r50', '    sep-life u r50'),
        ('A50', 'A50 horizon (ticks)'), ('rd', 'diag readout of stored codes'), ('r50x3', 'diag r50 with X3 recall dynamics'), ('persist', 'diag persistence'),
        ('live', 'diag live links'), ('on_sp', 'diag units on, speech'), ('on_si', 'diag units on, silence')]


def verdict(m, x3):
    def g(k):
        return np.nanmean([s.get(k, np.nan) for s in m.values()])
    t = {}
    t[1] = g('r50') >= 0.9
    t[2] = g('ov') >= 0.7 and g('ov6') >= 0.7 and g('ovx') <= 0.3
    t[3] = g('fresh_R1s') >= 0.6 and g('adult_R1s') >= 0.6
    t[4] = g('qaP_right') >= 0.8 and g('qaP_wrong') <= 0.05
    t[5] = g('sep_pV') >= 0.9 and g('sep_pD') >= 0.9
    t[6] = g('i100') >= x3['i100'] - 0.05 and g('r50') >= x3['r50'] - 0.05
    return t


def res(d, arms=None):
    R = load(d, '*.json')
    R = {n: v for n, v in R.items() if not n.startswith('cal_')}
    allarms = []
    for n in R:
        a = n.split('_', 1)[1].rsplit('_s', 1)[0]
        if a not in allarms:
            allarms.append(a)
    order = ['X3', 'NS', 'L001', 'L03', 'L1', 'L3', 'L5', 'L1r', 'L3r', 'L5r', 'Z1', 'Z4', 'Z10', 'ZL']
    arms = arms or [a for a in order if a in allarms]
    M = {a: per_arm(R, a) for a in arms}
    x3 = {k: np.nanmean([s.get(k, np.nan) for s in M['X3'].values()]) for k in ('i100', 'r50')} if 'X3' in M else None
    w = 16
    print(f'{"metric":38s}' + ''.join(f'{a:>{w}s}' for a in arms))
    for k, lab in KEYS:
        p = 0 if k in ('A50', 'live') else (0 if k.startswith('on_') else 2)
        print(f'{lab:38s}' + ''.join(f'{fmt([s.get(k, np.nan) for s in M[a].values()], p):>{w}s}' for a in arms))
    if x3:
        print('\npaired difference vs X3 (same seed), mean±SD, and seeds better')
        for k in ('r50', 'i100', 'ov', 'fresh_R1s', 'adult_R1s', 'qaP_right', 'sep_pS', 'sep_pV', 'sep_pD'):
            line = f'{k:14s}'
            for a in arms:
                if a == 'X3':
                    continue
                dd = [M[a][s][k] - M['X3'][s][k] for s in M[a] if s in M['X3'] and k in M[a][s] and k in M['X3'][s]
                      and not np.isnan(M[a][s][k]) and not np.isnan(M['X3'][s][k])]
                line += f' {a}:{fmt(dd)}({sum(x > 0 for x in dd)}/{len(dd)})'
            print(line)
        print('\nverdict per test (1 r50>=0.9 | 2 re-hear | 3 fact same scene | 4 Q-A bridge | 5 sep pV,pD | 6 no fall)')
        for a in arms:
            t = verdict(M[a], x3)
            print(f'  {a:5s} ' + ' '.join(f'T{i}:{"P" if t[i] else "-"}' for i in range(1, 7)))
    return M


def cal(d):
    R = load(d, 'cal_*.json')
    rows = []
    for n, x in R.items():
        m = m_uni(dict(x, all=[r for r in x['all']]))
        arm = n.split('_')[1]
        rows.append((arm, n, m))
    best = {}
    for arm in sorted({r[0] for r in rows}):
        rs = [r for r in rows if r[0] == arm]
        print(f'--- {arm}')
        for _, n, m in sorted(rs, key=lambda r: r[1]):
            print(f'  {n:34s} r50 {m["r50"]:.2f} i100 {m["i100"]:.2f} ov {m["ov"]:.2f} ov6 {m["ov6"]:.2f} ovx {m["ovx"]:.2f}'
                  f' rd {m["rd"]:.2f} live {int(m["live"])} on {m["on_sp"]:.0f}/{m["on_si"]:.0f}')
        # rule: highest r50; ties within 0.02 -> higher i100 -> higher (ov - ovx)
        top = max(m['r50'] for _, _, m in rs)
        cand = [r for r in rs if r[2]['r50'] >= top - 0.02]
        top_i = max(r[2]['i100'] for r in cand)
        cand = [r for r in cand if r[2]['i100'] >= top_i - 0.02]
        b = max(cand, key=lambda r: r[2]['ov'] - r[2]['ovx'])
        best[arm] = b[1]
        print(f'  CHOSEN {b[1]}')
    ch = {}
    for arm, n in best.items():
        cfg = R[n]['spec']['cfg']
        c = dict(lam=cfg['lam'], alpha=cfg['alpha'])
        if arm == 'ZL':
            c.update(f=cfg['ls'], eta=int(round(cfg.get('eta_seq', 2.0) / 2.0)))
        ch[arm] = c
    json.dump(ch, open(os.path.join(HERE, 'cal_choice.json'), 'w'), indent=1)
    print(json.dumps(ch))


if __name__ == '__main__':
    d = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, 'out')
    if sys.argv[1] == 'cal':
        cal(d)
    else:
        res(d, sys.argv[3:] or None)
