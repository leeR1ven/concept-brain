# -*- coding: utf-8 -*-
"""Life-mode comparison (tune's summary format): python3 life_table.py name=path.json ...  (tune: ../tune/out/life/...)"""
import sys, json
import numpy as np
rows = []
for a in sys.argv[1:]:
    name, path = a.split('=', 1)
    r = json.load(open(path)); s = r['summary']
    g = s['groups']; t = s['tasks']; ini = s['initiative'] or {}
    qa = {d['encounters']: (d['n'], d['acc']) for d in (s['retention'] or {}).get('qa_by_encounters', [])}
    st = s['stability'] or []
    acc_bins = [b.get('acc') for b in st]
    first = next((x for x in acc_bins if x is not None), None); last = next((x for x in reversed(acc_bins) if x is not None), None)
    spoke = [b.get('spoke') for b in st if b.get('spoke') is not None]
    f = lambda x: '-' if x is None else f"{x:.2f}"
    print(f"== {name}: {s['hours']:.1f} h, {s['ticks']} ticks, {r.get('secs')} s CPU")
    print(f"   free acc {f(s['acc'])} (n {s['n_free']}) | scene(new) {f(g['scene']['acc'])} (n {g['scene']['n']}) | scene repeat {f(g['scene_repeat']['acc'])} | "
          f"held-out {f(g['held_out']['acc'])} (n {g['held_out']['n']}) | memory {f(g['memory']['acc'])} (n {g['memory']['n']}) | social {f(g['social']['acc'])} (n {g['social']['n']})")
    print(f"   items heard 1x / 2x / 3-4x / 5-8x / 9+x: " + " / ".join(f"{f(qa.get(k, (0, None))[1])}" for k in ('1', '2', '3-4', '5-8', '9+')) +
          f" | acc first->last tenth {f(first)} -> {f(last)} | answer rate {f(min(spoke) if spoke else None)}-{f(max(spoke) if spoke else None)}")
    print(f"   initiative n {ini.get('n')} ({ini.get('per_1000_quiet_ticks', 0):.2f}/1000 quiet), appropriate {f(ini.get('appropriate_share'))} "
          f"(partner present {f(ini.get('appropriate_share_partner_present'))}), reward {ini.get('reward')}, cats {ini.get('categories')}")
    print(f"   overlap {s['overlap']} | monologue {s['monologue']} | flags {s['flags']}")
    print("   per tenth: acc " + " ".join(f(b.get('acc')) for b in st) + " | answer rate " + " ".join(f(b.get('spoke')) for b in st) +
          " | initiatives/1000q " + " ".join(f(b.get('init_per_1000_quiet')) for b in st))
    print("   tasks: " + " ".join(f"{k}={v['acc']:.2f}/{v['n']}" for k, v in t.items() if isinstance(v, dict) and v.get('acc') is not None and k not in ('ALL', 'SCENE', 'NOSCENE', 'one_shot')))
