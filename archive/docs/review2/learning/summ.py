# -*- coding: utf-8 -*-
"""Summarise gen_probe_out.jsonl and gen_probe2_out.jsonl (mean over seeds per condition).  usage: python3 summ.py"""
import json, os, collections
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
for fn in ('gen_probe_out.jsonl', 'gen_probe2_out.jsonl'):
    p = os.path.join(HERE, fn)
    if not os.path.exists(p):
        continue
    G = collections.defaultdict(list)
    for l in open(p):
        r = json.loads(l)
        G[(r['variant'], r.get('heldstream', 1), r.get('tutor', 'recast'))].append(r)
    print('==', fn)
    for k, R in G.items():
        fs = [r['frozen']['seen'] for r in R]; fh = [r['frozen']['held'] for r in R]
        lh = [r['ALL']['held'] for r in R if r['ALL']['held'] is not None]
        pl = [r['frozen']['held_cls'].get('partner_line', 0) for r in R]
        print(f'{k[0]:8s} heldstream={k[1]} tutor={k[2]:6s} seeds={[r["seed"] for r in R]} frozen seen {np.mean(fs):.2f} '
              f'held {np.mean(fh):.2f} {[round(x, 2) for x in fh]} partner-line share {np.mean(pl):.2f} | live held '
              f'{np.mean(lh) if lh else float("nan"):.2f}')
