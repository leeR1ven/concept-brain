# -*- coding: utf-8 -*-
"""Life comparison: headline numbers + initiative over life time (categories, appropriate share, latency of the
initiative after the visual event that opened it is not logged by the world; we report what the world logs).
usage: python3 life_summary.py name=out/life_<tag> ..."""
import sys, pickle, collections, json
import numpy as np
sys.path.insert(0, 'bench_copy')
from life import evaluate_life, headline_life   # noqa: E402

runs = dict(a.split('=', 1) for a in sys.argv[1:])
for name, pref in runs.items():
    L = pickle.load(open(pref + '_log.pkl', 'rb'))
    res = evaluate_life(L)
    print(f"== {name}: {headline_life(res)}")
    ini = [r for r in L if r.get('type') == 'init']
    t_max = max(r['t'] for r in L if r.get('type') == 'mon')
    edges = np.linspace(0, t_max, 6)
    rows = []
    for a, b in zip(edges[:-1], edges[1:]):
        rs = [r for r in ini if a <= r['t'] < b]
        c = collections.Counter(r['category'] for r in rs)
        rows.append(f"[{a/1000:.0f}k-{b/1000:.0f}k) n={len(rs)} appr={np.mean([r['appropriate'] for r in rs]) if rs else 0:.2f} "
                    f"reward={sum(r['reward'] for r in rs):.0f} {dict(c)}")
    print("   initiatives by life time:", " | ".join(rows))
    pres = [r for r in ini if r['partner']]
    print(f"   with partner present: n={len(pres)}, appropriate {np.mean([r['appropriate'] for r in pres]) if pres else None}, "
          f"examples {[ (r['category'], r['text']) for r in pres[-12:]]}")
    print(f"   alone/night: n={sum(1 for r in ini if not r['partner'])}, examples {[r['text'] for r in ini if not r['partner']][-8:]}")
    g = res['groups']
    print("   groups:", {k: (v['n'], None if v['acc'] is None else round(float(v['acc']), 3)) for k, v in g.items()})
    print("   tasks:", {k: (v.get('n'), None if v.get('acc') is None else round(v['acc'], 2)) for k, v in res['tasks'].items()
                         if isinstance(v, dict) and 'acc' in v})
    print("   overlap:", res['overlap'], "monologue:", res['monologue'], "latency:", res['latency'].get('median'),
          "flags:", res['stability_flags'])
    soc = [r for r in L if r.get('type') == 'ex' and r.get('task') == 'greet']
    print(f"   greetings answered: {np.mean([r['reply'] in r['acceptable'] for r in soc]) if soc else None} (n={len(soc)}); "
          f"greeted first (initiative): {sum(1 for r in ini if r['category'] == 'greeting')}")
