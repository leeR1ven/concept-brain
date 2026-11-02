"""Tables for a1L and the sensitivity sweep, with activity budgets. python3 agg_sens.py <results dir>"""
import glob
import json
import sys

d = sys.argv[1]
print('### sensitivity (one knob at a time around P1 / reciprocal / g_loop 1 / w44 0)')
print('| variation | N1 | L4 frac birth (speech) | a1 | a2 | a3 | lag 12 | Hs lag 0 | Hs J margin | a1L | last window: '
      'L2/L1, L3/L2, L4/L3, Hs, L4 frac |')
print('|' + '---|' * 11)
for f in sorted(glob.glob(d + '/sens/sens_*.json')):
    r = json.load(open(f))
    act, cod, st = r['birth_budget'], r['codes'], r['stability']
    w = st['windows'][-1] if st['windows'] else {}
    n4 = r['n1']
    print(f"| {r['label']} | {r['n1']} | {act['L4']['speech'] / n4:.3f} | {'P' if r['a1_pass'] else '-'} | "
          f"{'P' if cod['a2_pass'] else '-'} | {'P' if cod['a3_pass'] else '-'} | {cod['L4_decode']['12']:.2f} | "
          f"{cod.get('Hs_decode', {}).get('0', float('nan')):.2f} | {cod.get('Hs_J_margin', float('nan')):.2f} | "
          f"{'P' if st['pass_'] else '-'} | {w.get('L2_L1', 0):.2f}, {w.get('L3_L2', 0):.2f}, {w.get('L4_L3', 0):.2f}, "
          f"{w.get('Hs', 0):.0f}, {w.get('L4_frac', 0):.2f} |")
print()
print('### a1L remedy ladder')
print('| configuration | rung | a1L | last window: L2/L1, L3/L2, L4/L3, Hs, L4 frac | worst window L4/L3 | Hs range | max frac |')
print('|' + '---|' * 7)
for f in sorted(glob.glob(d + '/a1L/*.json')):
    r = json.load(open(f))
    st = r['stability']
    w = st['windows'][-1]
    c = r['config']
    print(f"| w44={c.get('w44')} | {r['rung']} | {'P' if st['pass_'] else '-'} | {w['L2_L1']:.2f}, {w['L3_L2']:.2f}, "
          f"{w['L4_L3']:.2f}, {w['Hs']:.0f}, {w['L4_frac']:.2f} | {max(x['L4_L3'] for x in st['windows']):.2f} | "
          f"{min(x['Hs'] for x in st['windows']):.0f}-{max(x['Hs'] for x in st['windows']):.0f} | "
          f"{max(st['frac_max'].values()):.2f} |")
print()
print('### activity budget (mean active units per tick, silence / speech): birth, after 100 and 300 learning dialogues')
names = ['A1', 'A2', 'A3', 'A4', 'L1ext', 'L1int', 'L2', 'L3', 'L4', 'H1', 'Hs', 'force', 'stop']
print('| configuration | when | ' + ' | '.join(names) + ' |')
print('|' + '---|' * (len(names) + 2))
for f in sorted(glob.glob(d + '/a1L/*.json')):
    r = json.load(open(f))
    lab = f"w44={r['config'].get('w44')} {r['rung'].split()[0]}"
    rows = [('birth', r['birth_budget'])] + [(f'{k} dlg', v) for k, v in r['stability']['budget'].items()]
    for when, b in rows:
        cells = []
        for n in names:
            x = b.get(n, {})
            s, p = x.get('silence'), x.get('speech')
            cells.append(f"{s:.0f} / {p:.0f}" if s is not None and p is not None else '-')
        print(f'| {lab} | {when} | ' + ' | '.join(cells) + ' |')
