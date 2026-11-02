"""Stage (a) table: python3 agg_a.py <json folders...>. Sorted by the decision rule: a1 and a2 pass first, then
lag-12 decoding (a3)."""
import glob
import json
import sys

rows = []
for d in sys.argv[1:]:
    for f in sorted(glob.glob(d + '/s*.json')):
        r = json.load(open(f))
        a1 = r['a1']
        rows.append(dict(cfg=r['config'], n1=r.get('n1'), a1=r['a1_pass'], a2=r['a2_pass'], a3=r['a3_pass'],
                         lag0=r['L4_decode']['0'], lag6=r['L4_decode']['6'], lag12=r['L4_decode']['12'],
                         ffr=r['L4_filler_given_frame_lag12'], mot=r['motion_J_t_t1'], jm=r['J_margin'],
                         hs0=r.get('Hs_decode', {}).get('0'),
                         l1=(round(a1['L1']['silence']), round(a1['L1']['speech'])),
                         l1int=round(a1['L1int']['speech']), l4=round(a1['L4']['speech']),
                         r21=round(a1['L2/L1 speech'], 2), r43=round(a1['L4/L3 speech'], 2),
                         r43s=round(a1['L4/L3 silence'], 2), hs=round(a1['Hs']['speech']) if 'Hs' in a1 else None,
                         l4max=round(a1['L4']['frac_max'], 2)))
rows.sort(key=lambda r: (not (r['a1'] and r['a2']), -r['lag12']))
print('| config | N1 | a1 | a2 | a3 | lag 0 / 6 / 12 | filler-given-frame 12 | motion J | J margin | Hs lag 0 | L1 sil/sp | '
      'L1int | L4 | L2/L1 | L4/L3 sp/sil | Hs | L4 max frac |')
print('|' + '---|' * 17)
for r in rows:
    c = ' '.join(f'{k}={v}' for k, v in r['cfg'].items())
    print(f"| {c} | {r['n1']} | {'P' if r['a1'] else '-'} | {'P' if r['a2'] else '-'} | {'P' if r['a3'] else '-'} | "
          f"{r['lag0']:.2f} / {r['lag6']:.2f} / {r['lag12']:.2f} | {r['ffr']:.2f} | {r['mot']:.2f} | {r['jm']:.2f} | "
          f"{r['hs0'] if r['hs0'] is None else round(r['hs0'], 2)} | {r['l1'][0]} / {r['l1'][1]} | {r['l1int']} | "
          f"{r['l4']} | {r['r21']} | {r['r43']} / {r['r43s']} | {r['hs']} | {r['l4max']} |")
