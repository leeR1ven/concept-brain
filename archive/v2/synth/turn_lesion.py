# -*- coding: utf-8 -*-
"""Turn-taking lesions in a world WITH feedback (recast tutor), frozen copies of the final agent: does the learned
brake / eagerness reach behaviour?  Measures speech after the partner's recast, accuracy, latency, overlaps."""
import os, sys, copy, pickle
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np
from agent_synth import SynthAgent  # noqa
from agent import BENCH
sys.path.insert(0, BENCH)
from world import World
from metrics import evaluate
ag0 = pickle.load(open(sys.argv[1], 'rb'))
n = int(sys.argv[2]) if len(sys.argv) > 2 else 150
for name in ('full', 'P_oth', 'V_go'):
    ag = copy.deepcopy(ag0); ag.learning = False
    if name == 'P_oth':
        ag.P_oth.w = {}
    elif name == 'V_go':
        ag.V_go.w = {}; ag.rbar = 0.5
    w = World('BC', seed=0, tutor='recast', eye_layout='retino')
    w.run(ag, n)
    rc = [r for r in w.log if r['kind'] == 'recast']
    st = [r for r in w.log if r['kind'] == 'statement']
    fr = [r for r in w.log if r['kind'] in ('probe', 'recast')]
    lat = [r['latency'] for r in fr if r.get('latency') is not None]
    acc = np.mean([r['reply'] in r['acceptable'] for r in fr])
    after = np.mean([bool(r.get('after_recast_reply')) for r in rc]) if rc else None
    ov = sum(r.get('mouth_while_partner', 0) for r in w.log)
    print(f"{name:6s} replies {len(fr)} acc {acc:.3f} spoke {len(lat)/max(len(fr),1):.3f} latency {np.mean(lat):.2f} | "
          f"speaks after the partner's recast {after:.3f} (n {len(rc)}) | overlap ticks {ov}", flush=True)
