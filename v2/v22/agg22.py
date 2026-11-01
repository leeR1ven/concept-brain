# -*- coding: utf-8 -*-
"""Aggregate run_v22 tags (5-seed means ± SD): J, J2, F, RET, H, DR, drift losses, role confusion, held-out vs best
constant, per-task accuracy and held-out, secs, peak RSS, dips, R1 blocks.  usage: python3 agg22.py tag [tag ...]"""
import json, glob, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
TASKS = ['what', 'slot', 'where', 'offer', 'taste', 'colour', 'count', 'ritual', 'check']
for tag in sys.argv[1:]:
    R = []
    for f in sorted(glob.glob(os.path.join(HERE, 'out', 'runs', f'{tag}_s*.json'))):
        for l in open(f):
            r = json.loads(l)
            if r['tag'] == tag:
                R.append(r)
    if not R:
        continue
    ms = lambda xs: f'{np.mean(xs):.3f} ± {np.std(xs):.3f}'
    d0 = [r['drift']['before']['ALL'][0] for r in R]
    l3 = [d0[i] - r['drift']['after']['ALL'][0] for i, r in enumerate(R)]
    l9 = [d0[i] - r['drift']['after2']['ALL'][0] for i, r in enumerate(R)]
    print(f"== {tag} seeds {[r['seed'] for r in R]} viol {sum(r['viol'] for r in R)}")
    print('  ' + ' | '.join(f"{k} {ms([r['score'][k] for r in R])}" for k in ('J', 'J2', 'F', 'RET', 'H', 'DR')))
    print(f"  loss300 {ms(l3)} | loss900 {ms(l9)} | role {ms([r['role']['role_conf'] for r in R])} | role held {ms([r['role_held']['role_conf'] for r in R])}"
          f" | held agent {ms([r['refs']['held']['agent'] for r in R])} vs const {np.mean([r['refs']['held']['const_best'] for r in R]):.3f}"
          f" | all agent {np.mean([r['refs']['all']['agent'] for r in R]):.3f} echo {np.mean([r['refs']['all']['echo'] for r in R]):.3f} const {np.mean([r['refs']['all']['const_best'] for r in R]):.3f}")
    a3 = {}
    for i, r in enumerate(R):
        for k, v in (r['drift'].get('after3') or {}).items():
            a3.setdefault(k, []).append(d0[i] - v['acc'])
    if a3:
        print('  long untutored loss: ' + ' | '.join(f"{k}: {ms(v)} (acc {np.mean([d0[i] - x for i, x in enumerate(v)]):.3f})" for k, v in sorted(a3.items(), key=lambda x: int(x[0]))))
    pt = lambda t, j: np.mean([r['per_task'][t][j] for r in R if r['per_task'].get(t, {}).get(j) is not None])
    print('  task acc  ' + ' '.join(f"{t} {pt(t, 'acc'):.2f}" for t in TASKS))
    print('  held-out  ' + ' '.join(f"{t} {pt(t, 'held'):.2f}" for t in TASKS if t != 'ritual'))
    les = {}
    for r in R:
        for k, v in (r.get('lesions') or {}).items():
            les.setdefault(k, []).append(((v['B'][0] + v['C'][0] + v['D'][0]) / 3, r['score']['F']))
    if les:
        print('  lesions (mean B/C/D final probe, intact F): ' + ' | '.join(f"{k} {np.mean([x[0] for x in v]):.3f} vs {np.mean([x[1] for x in v]):.3f}" for k, v in les.items()))
    print(f"  secs {np.mean([r['secs'] for r in R]):.0f} | peak RSS {np.mean([r.get('peak_rss_mb', 0) for r in R]):.0f} MB | dips {[r['val']['dips'] for r in R]} | R1 blocks {[r['tstats'].get('blocked_R1') for r in R]} | rbar {[r.get('rbar') for r in R]}")
