"""aggregate X9 runs: python3 -I agg.py dir [arm ...] -> per-arm table (final checkpoint; drift 2/3 -> end) and paired
differences against the bases.  File names: <arm>_s<seed>.json"""
import glob
import json
import os
import sys

import numpy as np

D = sys.argv[1]
ARMS = sys.argv[2:] or None
rows = {}
for p in sorted(glob.glob(os.path.join(D, '*.json'))):
    r = json.load(open(p))
    arm = os.path.basename(p)[:-5].rsplit('_s', 1)[0]
    if ARMS and arm not in ARMS:
        continue
    m = dict(r['res'][-1])
    m.update(r['drift'])
    m['seed'] = r['cfg']['seed']
    L2 = 'L2_c1_within' in m
    # primary class code: deep layer C2, first-char codes (single-layer arms: their only layer)
    pre = 'L2_c1_' if L2 else 'c1_'
    m['P_within'], m['P_between'] = m[pre + 'within'], m[pre + 'between']
    m['P_held'], m['P_margin'], m['P_seen'] = m[pre + 'held_dec'], m[pre + 'held_margin'], m[pre + 'seen_dec']
    he = m['L2_held_each_c1'] if L2 else None
    if he is None:          # single layer: recompute pooled count from held_dec (4 held each)
        m['P_held_ok'] = round(m['P_held'] * 4)
    else:
        m['P_held_ok'] = sum(he.values())
    m['P_ratio'] = m['P_within'] / max(1e-9, m['P_between'])
    m['P_wb'] = m['P_within'] - m['P_between']
    m['xm_diff'] = m['xm_match'] - m['xm_mis']
    m['P_drift'] = m['L2_drift_first'] if L2 else m['drift_first']
    m['P_collapse'] = m['L2_collapse'] if L2 else m['collapse']
    rows.setdefault(arm, {})[m['seed']] = m

KEYS = [('P_held', 'held decode (class code)'), ('P_margin', 'held margin'), ('P_seen', 'seen decode (LOO)'),
        ('P_within', 'within'), ('P_between', 'between'), ('P_ratio', 'within/between'),
        ('xm_match', 'cross-modal matched'), ('xm_mis', 'cross-modal mismatched'), ('xm_diff', 'cross-modal diff'),
        ('xm_top1', 'cross-modal 12-way top-1'), ('L2_xm_top1', 'cross-modal top-1, deep layer'),
        ('drift_words', 'drift whole area, words'), ('P_drift', 'drift class code'),
        ('L1_drift_words', 'drift input layer, words'), ('L2_drift_words', 'drift deep layer, words'),
        ('P_collapse', 'collapse index (class layer chars)'), ('collapse', 'collapse index (whole area)'),
        ('L2_within', 'deep layer word-union within'), ('L2_between', 'deep layer word-union between'),
        ('within', 'whole-area word within'), ('between', 'whole-area word between')]


def ms(v):
    v = np.asarray(v, float)
    return f'{v.mean():.3f} ± {v.std(ddof=1) if len(v) > 1 else 0:.3f}'


print(f'dir {D}')
for arm, S in rows.items():
    seeds = sorted(S)
    print(f'\n== {arm}  seeds {seeds}')
    ok = sum(S[s]['P_held_ok'] for s in seeds)
    print(f'  held decode pooled {ok}/{4 * len(seeds)} = {ok / max(1, 4 * len(seeds)):.3f}')
    print(f'  within > between on {sum(S[s]["P_wb"] > 0 for s in seeds)}/{len(seeds)};'
          f' cross-modal matched > mismatched on {sum(S[s]["xm_diff"] > 0 for s in seeds)}/{len(seeds)}')
    for k, lab in KEYS:
        v = [S[s][k] for s in seeds if k in S[s]]
        if v:
            print(f'  {lab:38s} {ms(v)}')
for bname in ('L20', 'B0'):
    base = rows.get(bname)
    if not base:
        continue
    for arm, S in rows.items():
        if arm == bname:
            continue
        seeds = sorted(set(S) & set(base))
        if not seeds:
            continue
        print(f'\n-- paired {arm} - {bname} ({len(seeds)} seeds)')
        for k in ('P_held', 'P_margin', 'P_within', 'P_between', 'P_wb', 'xm_diff', 'xm_top1', 'drift_words', 'P_drift',
                  'P_collapse'):
            dv = [S[s][k] - base[s][k] for s in seeds if k in S[s] and k in base[s]]
            if dv:
                print(f'  {k:14s} {ms(dv)}  (up on {sum(x > 0 for x in dv)}/{len(dv)})')
