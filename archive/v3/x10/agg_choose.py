"""Step-2 table and the pre-registered rule. python3 agg_choose.py <folders with choose_*.json>"""
import glob
import json
import sys
from collections import defaultdict

import numpy as np


def cand(c):
    if c.get('pfc_size') == 'large':
        return f"large g_loop {c.get('g_loop')}"
    if c.get('eta') == 0.005:
        return 'C (w44 0 + R3)'
    return 'A (w44 0.5 + R2)' if c.get('w44') == 0.5 else 'B (w44 0 + R2)'


rows = defaultdict(list)
for d in sys.argv[1:]:
    for f in sorted(glob.glob(d + '/choose_*.json')):
        r = json.load(open(f))
        rows[cand(r['config'])].append(r)
print('| candidate | seed | a1 | a2 (rev.) | a3 | lag 0 / 12 | filler 12 | Hs J margin | Hs decode 0 | a1L windows ok | '
      'ext pass | L4 / Hs / A4: last100 ÷ at500 | slopes per 100 dlg (L4, Hs, A4) | last window L4/L3, Hs |')
print('|' + '---|' * 14)
summary = {}
for k in sorted(rows):
    for r in sorted(rows[k], key=lambda x: x['seed']):
        cd, st = r['codes'], r['stability']
        tr = st.get('trend', {})
        w = st['windows'][-1]
        print(f"| {k} | {r['seed']} | {'P' if r['a1_pass'] else '-'} | {'P' if cd['a2_pass'] else '-'} | "
              f"{'P' if cd['a3_pass'] else '-'} | {cd['L4_decode']['0']:.2f} / {cd['L4_decode']['12']:.2f} | "
              f"{cd['L4_filler_given_frame_lag12']:.2f} | {cd['Hs_J_margin']:.2f} | {cd['Hs_decode']['0']:.2f} | "
              f"{st['share_windows_ok']:.2f} | {'P' if st.get('pass_ext', st['pass_']) else '-'} | "
              + (', '.join(f"{tr[x]['ratio']:.2f}" for x in ('L4', 'Hs', 'A4')) if tr else 'n/a') + ' | '
              + (', '.join(f"{tr[x]['slope_per100']:.0f}" for x in ('L4', 'Hs', 'A4')) if tr else 'n/a') +
              f" | {w['L4_L3']:.2f}, {w['Hs']:.0f} |")
    R = rows[k]
    allpass = all(r['a1_pass'] and r['codes']['a2_pass'] and r['stability'].get('pass_ext', r['stability']['pass_'])
                  for r in R)
    summary[k] = dict(n=len(R), all_pass=allpass, lag12=float(np.mean([r['codes']['L4_decode']['12'] for r in R])),
                      a3=sum(r['codes']['a3_pass'] for r in R))
print()
print('| candidate | seeds | passes a1 + a2 + a1L on every seed | mean lag 12 | a3 passes |')
print('|---|---|---|---|---|')
for k, v in summary.items():
    print(f"| {k} | {v['n']} | {'yes' if v['all_pass'] else 'no'} | {v['lag12']:.3f} | {v['a3']}/{v['n']} |")
ok = {k: v for k, v in summary.items() if v['all_pass'] and v['n'] >= 4 and not k.startswith('large')}
if ok:
    best = sorted(ok.items(), key=lambda kv: -kv[1]['lag12'])
    top = best[0]
    if len(best) > 1 and top[0].startswith('C') and best[1][1]['lag12'] >= top[1]['lag12'] - 0.03:
        top, second = best[1], best[0]
    else:
        second = best[1] if len(best) > 1 else None
    print(f"\nrule: chosen {top[0]} (lag 12 {top[1]['lag12']:.3f}); runner-up {second[0] if second else None}")
else:
    print('\nrule: no medium candidate passes on all 4 seeds (yet)')
