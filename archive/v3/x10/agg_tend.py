"""Tendency tables. python3 agg_tend.py <json files>"""
import json
import sys

print('| run | ticks 10-16: x share / best rival / rivals >= half / x first / total lit | nothing forced: total lit | peak x first (tick) | units per syllable (min / median) |')
print('|---|---|---|---|---|')
for f in sys.argv[1:]:
    r = json.load(open(f))
    p = r['profile']
    T = p['tick']
    ix = [i for i, t in enumerate(T) if 10 <= t <= 16]
    m = lambda k: sum(p[k][i] for i in ix) / len(ix)
    base = sum(p['baseline_total'][i] for i in ix) / len(ix)
    pk = max(range(len(T)), key=lambda i: p['first'][i])
    g = r.get('registry', {}).get('G1P')
    us = f"{min(g['per_syllable'])} / {sorted(g['per_syllable'])[len(g['per_syllable']) // 2]}" if g else '-'
    print(f"| {r['arm']}{r.get('tag', '')} | {m('x'):.2f} / {m('best'):.2f} / {m('ge_half'):.1f} / {m('first'):.2f} / {m('total'):.2f} | "
          f"{base:.2f} | {p['first'][pk]:.2f} ({T[pk]}) | {us} |")
