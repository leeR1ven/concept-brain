# -*- coding: utf-8 -*-
"""Accuracy against allowed thinking time, on a FROZEN trained agent (develop's ablation protocol: probe-only worlds,
B/C learning off; D: the hippocampal index stores, readouts frozen).  Policies: budget:k (release as soon as k thinking
steps are done; the thought is frozen at k), free (the learned intention), v1 (the fixed GO timer, completed thought).
usage: python3 eval_frozen.py agent.pkl stages=BCD n=150 nD=200 seed=0 pols=budget:0,budget:1,...,free,v1"""
import os, sys, copy, json, pickle, time, collections
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1"); sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from agent_think import ThinkAgent, BENCH
sys.path.insert(0, BENCH)
from world import World
from metrics import evaluate
import numpy as np

A = dict(a.split("=", 1) for a in sys.argv[2:] if "=" in a)
agent0 = pickle.load(open(sys.argv[1], "rb"))
for k, v in dict(scene_react=None, nogo_late=0.05, R_line=False, life_eye=False, eye_feats=False, no_react=False,
                 oth_kind=False, _E_now=0.0).items():
    if not hasattr(agent0, k):
        setattr(agent0, k, v)
for k, v in dict(brake_now=False, now_blocked=0).items():
    if not hasattr(agent0.body, k):
        setattr(agent0.body, k, v)
stages, n, nD, seed = A.get("stages", "BCD"), int(A.get("n", 150)), int(A.get("nD", 200)), int(A.get("seed", 0))
pols = A.get("pols", "budget:0,budget:1,budget:2,budget:3,budget:4,budget:6,budget:8,free,v1").split(",")
out = {}
t0 = time.time()
for st in stages:
    for pol in pols:
        ag = copy.deepcopy(agent0)
        ag.policy = pol
        ag.body.go_mode = 'v1' if pol == 'v1' else 'gate'
        ag.learn_turn = False
        ag.th = None; ag.pend = []; ag.th_log = []
        if st in "BC":
            ag.learning = False
        else:
            ag.learning = True; ag.core.c.learn_listen = False
        w = World(st, seed=seed, tutor="recast", probe_frac=1.0, held_frac=0.3)
        w.run(ag, nD if st == 'D' else n)
        res = evaluate(w.log)[st]
        fr = [r for r in w.log if r['stage'] == st and r['kind'] in ('probe', 'recast')]
        lat = collections.defaultdict(list)
        for r in fr:
            if r.get('latency') is not None:
                lat[r['task']].append(r['latency']); lat['ALL'].append(r['latency'])
        row = {k: dict(acc=round(v['acc'], 3) if v.get('acc') is not None else None, n=v.get('n'),
                       held=round((v.get('held') or {}).get('acc') or 0, 3),
                       spoke=round(v.get('spoke') or 0, 3),
                       lat=round(float(np.mean(lat[k])), 2) if lat.get(k) else None)
               for k, v in res.items() if isinstance(v, dict) and 'acc' in v}
        ksteps = [t['k_rel'] for t in ag.th_log if t['kind'] == 'reply' and t['k_rel'] is not None]
        row['_k_at_release'] = round(float(np.mean(ksteps)), 2) if ksteps else None
        row['_overlap'] = int(sum(r.get('mouth_while_partner', 0) for r in w.log))
        out[f"{st}|{pol}"] = row
        print(f"[{st}] {pol:9s} ALL {row['ALL']['acc']} (held {row['ALL']['held']}, spoke {row['ALL']['spoke']}, lat {row['ALL']['lat']}, "
              f"k@rel {row['_k_at_release']}, overlap {row['_overlap']})  " +
              " ".join(f"{k}={v['acc']}/{v['lat']}" for k, v in row.items() if not k.startswith('_') and k not in ('ALL', 'SCENE', 'NOSCENE', 'one_shot')),
              f"({time.time()-t0:.0f}s)", flush=True)
json.dump(out, open(sys.argv[1].replace("_agent.pkl", "_frozen.json"), "w"), ensure_ascii=False)
