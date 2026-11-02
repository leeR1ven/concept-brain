"""aggregate X8 run JSONs into the tables of the report.  python3 -I analyze.py <runs dir> [more dirs ...]"""
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

runs = []
for d in sys.argv[1:]:
    for f in sorted(glob.glob(os.path.join(d, '*.json'))):
        r = json.load(open(f))
        r['_name'] = os.path.basename(f)[:-5]
        runs.append(r)


def key(r):
    tag = r['_name'].split('_w')[0]
    return tag, r['w_own']


G = defaultdict(dict)
for r in runs:
    G[key(r)][r['seed']] = r


def m(v):
    v = [x for x in v if x is not None and not (isinstance(x, float) and np.isnan(x))]
    return (float(np.mean(v)), float(np.std(v))) if v else (float('nan'), float('nan'))


def fmt(ms, d=2):
    return f'{ms[0]:.{d}f} ± {ms[1]:.{d}f}'


def metrics(r):
    out = {}
    out['p1'] = r['p1']['ctx']
    out['ret_stretch'] = r['stretch']['ctx_ret']
    out['ret_beh'] = r['stretch']['beh_abs']
    out['ret_amn'] = r['amnesia']['ctx_ret'] if 'amnesia' in r else r['stretch']['ctx_ret']
    out['ret_life'] = r['life'][-1]['ctx_ret'] if 'life' in r else None
    out['ret_life300k'] = r['life'][1]['ctx_ret'] if 'life' in r else None
    per = r['corr']['per']
    out['first'] = [p['first'] for p in per]
    out['stable'] = [p['stable'] for p in per]
    out['sw5'] = float(np.mean([p['first'] <= 5 for p in per]))
    out['st5'] = float(np.mean([p['stable'] <= 5 for p in per]))
    out['mean_first'] = float(np.mean([p['first'] for p in per]))
    kn = [p for p in per if p['right_before']]
    out['n_known'] = len(kn)
    out['known_first'] = [p['first'] for p in kn]
    out['recasts'] = float(np.mean([p['total_recasts'] for p in per]))
    out['ch_ctx_wipe'] = r['corr']['ctx_after_wipe']
    out['ch_beh_end'] = r['corr']['rounds'][-1]['ch_beh']
    ub = r['corr']['rounds'][-1]['unch_beh']; cb = r['corr_control']['rounds'][-1]['unch_beh']
    uc = r['corr']['rounds'][-1]['unch_ctx']; cc = r['corr_control']['rounds'][-1]['unch_ctx']
    out['unch_end_start'] = ub
    out['unch_ctl'] = cb
    out['unch_vs_ctl'] = ub / cb if cb else None
    out['unch_ctx_end_start'] = uc
    out['unch_ctx_vs_ctl'] = uc / cc if cc else None
    if 'new_adult' in r:
        out['new_adult'] = float(np.mean(r['new_adult']['first']))
        out['new_fresh'] = float(np.mean(r['new_fresh']['first']))
        out['new_adult_end_ctx'] = r['new_adult']['curve'][-1]['ctx']
        out['new_adult_end_beh'] = r['new_adult']['curve'][-1]['beh']
        out['new_fresh_end_beh'] = r['new_fresh']['curve'][-1]['beh']
        out['new_adult_r1'] = r['new_adult']['curve'][0]['beh']
        out['new_fresh_r1'] = r['new_fresh']['curve'][0]['beh']
    return out


M = {k: {s: metrics(r) for s, r in v.items()} for k, v in G.items()}
order = [p + a for p in ('', 'C-') for a in ['A0', 'A3w', 'A3', 'A3R', 'A4H', 'A4', 'A34H', 'A34', 'A34U', 'A12', 'A12U']] or ['A0', 'A3w', 'A3', 'A3R', 'A4H', 'A4', 'A34H', 'A34', 'A34U', 'A12', 'A12U']


def col(k, name):
    return [M[k][s][name] for s in sorted(M[k])]


def paired(k, name, base='A0'):
    b = (('C-' if k[0].startswith('C-') else '') + base, k[1])
    if b not in M:
        return None
    ss = sorted(set(M[k]) & set(M[b]))
    d = [M[k][s][name] - M[b][s][name] for s in ss
         if M[k][s][name] is not None and M[b][s][name] is not None]
    return m(d) if d else None


print('## Retention and corrections, per arm and own-voice weight (mean ± SD over seeds; n = seeds)\n')
print('| arm | w_own | n | learned (p1) | retention, store kept | retention, cortex only (amnesia) | Δ vs A0 | '
      'life 1M ticks | switched ≤5 | stable ≤5 | mean re-teachings | Δ vs A0 | new answer in cortex | unchanged end/start | '
      'unchanged vs control |')
print('|' + '---|' * 15)
for arm in order:
    for w in (0.0, 0.25, 0.5, 1.0):
        k = (arm, w)
        if k not in M:
            continue
        n = len(M[k])
        dr = paired(k, 'ret_amn'); dm = paired(k, 'mean_first')
        life = col(k, 'ret_life')
        print(f'| {arm} | {w} | {n} | {fmt(m(col(k, "p1")))} | {fmt(m(col(k, "ret_stretch")))} | {fmt(m(col(k, "ret_amn")))} | '
              f'{fmt(dr) if dr else "–"} | {fmt(m(life)) if any(v is not None for v in life) else "–"} | '
              f'{m(col(k, "sw5"))[0]:.2f} | {m(col(k, "st5"))[0]:.2f} | {fmt(m(col(k, "mean_first")), 1)} | '
              f'{fmt(dm, 1) if dm else "–"} | {m(col(k, "ch_ctx_wipe"))[0]:.2f} | {fmt(m(col(k, "unch_end_start")))} | '
              f'{fmt(m(col(k, "unch_vs_ctl")))} |')

print('\n## Corrections of changed lessons that were still right before phase 3 (pooled over seeds)\n')
print('| arm | w_own | n known / n changed | switched <= 5 | mean re-teachings | never (21) |')
print('|---|---|---|---|---|---|')
for arm in order:
    for w in (0.0, 0.25, 0.5, 1.0):
        k = (arm, w)
        if k not in M:
            continue
        kf = sum((M[k][s]['known_first'] for s in M[k]), [])
        nt = 4 * len(M[k])
        if kf:
            print(f'| {arm} | {w} | {len(kf)} / {nt} | {np.mean([v <= 5 for v in kf]):.2f} | {np.mean(kf):.1f} | {np.mean([v == 21 for v in kf]):.2f} |')
        else:
            print(f'| {arm} | {w} | 0 / {nt} | – | – | – |')

print('\n## Adult learning of new lessons (teachings before the first right answer; never in 10 = 11)\n')
print('| arm | w_own | adult | fresh | right after 1 round adult / fresh | adult end beh / ctx | fresh end beh |')
print('|---|---|---|---|---|---|---|')
for arm in order:
    for w in (0.0, 0.25, 0.5, 1.0):
        k = (arm, w)
        if k not in M or 'new_adult' not in next(iter(M[k].values())):
            continue
        print(f'| {arm} | {w} | {fmt(m(col(k, "new_adult")), 1)} | {fmt(m(col(k, "new_fresh")), 1)} | '
              f'{m(col(k, "new_adult_r1"))[0]:.2f} / {m(col(k, "new_fresh_r1"))[0]:.2f} | '
              f'{m(col(k, "new_adult_end_beh"))[0]:.2f} / {m(col(k, "new_adult_end_ctx"))[0]:.2f} | '
              f'{m(col(k, "new_fresh_end_beh"))[0]:.2f} |')

print('\n## Retention curve over the stretch (cortex only, absolute; mean over seeds)\n')
print('| arm | w_own | branch | p1 | 1,500 | 3,000 | 4,500 | 6,000 lines |')
print('|---|---|---|---|---|---|---|---|')
for arm in order:
    for w in (0.0, 0.25, 0.5, 1.0):
        k = (arm, w)
        if k not in G:
            continue
        for br in ('stretch', 'amnesia'):
            rs = [G[k][s] for s in sorted(G[k]) if br in G[k][s]]
            if not rs:
                continue
            cur = np.mean([[c['ctx'] for c in r[br]['curve']] for r in rs], 0)
            p1 = np.mean([r['p1']['ctx'] for r in rs])
            print(f'| {arm} | {w} | {br} | {p1:.2f} | ' + ' | '.join(f'{v:.2f}' for v in cur) + ' |')

print('\n## Per-lesson re-teachings (first switch; 21 = never), all seeds\n')
for arm in order:
    for w in (0.0, 0.25, 0.5, 1.0):
        k = (arm, w)
        if k in M:
            print(f'- {arm} w {w}: ' + ' | '.join(','.join(str(v) for v in M[k][s]['first']) for s in sorted(M[k])))
