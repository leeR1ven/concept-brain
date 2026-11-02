# -*- coding: utf-8 -*-
"""ARCH_v2-develop on the shared bench (copy in bench_copy/): pre-verbal listening -> babbling play -> stage B -> C.
usage: python3 run_bench.py seed=0 nA=200 nB=2000 nC=2000 babbles=150000 tag=x [core cfg keys ...]"""
import os, sys, time, json, pickle
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from agent import DevAgent, BENCH          # noqa: E402
sys.path.insert(0, BENCH)
from world import World                    # noqa: E402
from metrics import evaluate, headline     # noqa: E402
import numpy as np                         # noqa: E402

A = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
def arg(k, d):
    v = A.pop(k, None)
    return d if v is None else type(d)(v)
seed, nA, nB, nC = arg("seed", 0), arg("nA", 200), arg("nB", 2000), arg("nC", 2000)
nD, nE = arg("nD", 2000), arg("nE", 1000)
nbab, tag, tutor = arg("babbles", 35960), arg("tag", "run"), arg("tutor", "recast")
stages = arg("stages", "ABC")
save = arg("save", 0)
cfgkw = {}
from core import Cfg                       # noqa: E402
for k, v in A.items():
    d = getattr(Cfg(), k)
    cfgkw[k] = (v in ("1", "True", "true")) if isinstance(d, bool) else type(d)(v)
OUT = os.path.join(HERE, "out", f"bench_{tag}")
os.makedirs(os.path.join(HERE, "out"), exist_ok=True)
logf = open(OUT + ".log", "w")
def log(s):
    print(s, flush=True); logf.write(s + "\n"); logf.flush()

t0 = time.time()
res_all = {}
def per_task(res, st):
    return json.dumps({k: (v.get('n'), v.get('acc'), (v.get('held') or {}).get('acc'), (v.get('seen') or {}).get('acc'))
                       for k, v in res[st].items() if isinstance(v, dict) and 'acc' in v}, ensure_ascii=False)
def examples(w, st, k=20):
    ex = [(r['task'], r['prompt'], r['reply'], r['acceptable'][0], r['held_out']) for r in w.log
          if r['stage'] == st and r['kind'] in ('probe', 'recast')][-k:]
    return " | ".join(f"{t}:{p}->{rp or '-'}({'H' if h else 'S'}, want {a})" for t, p, rp, a, h in ex)
prevw = None
all_logs = []
for st in stages:
    n = {'A': nA, 'B': nB, 'C': nC, 'D': nD, 'E': nE}[st]
    w = World(st, seed=seed, tutor=tutor, prev=prevw)
    if prevw is None:
        agent = DevAgent(w, seed=seed, **cfgkw)
        log(f"seed {seed} tutor {tutor} stages {stages} cfg {cfgkw}  FEAT0 {agent.NF}")
    w.run(agent, n)
    if st == 'A':
        log(f"[A] listening {n} A-dialogues: categories {agent.Z.n}, items {agent.stats['items']}, lines {agent.stats['lines']} ({time.time()-t0:.0f}s)")
        hist = agent.babble(w.voice, nbab, log_every=max(1, nbab // 8))
        log(f"[A] babbling {nbab} muscle sets: programs bound {len(agent.cat2prog)} of {agent.Z.n} categories; curve {hist}")
    else:
        res = evaluate(w.log); res_all[st] = res
        log(f"[{st}] " + headline(res, st))
        log(f"[{st}] per task (n, acc, held, seen): " + per_task(res, st))
        log(f"[{st}] examples: " + examples(w, st))
        lr = res[st].get('learning') or {}
        log(f"[{st}] learning in use (first vs last third): {json.dumps(lr, ensure_ascii=False)[:300]}  ({time.time()-t0:.0f}s)")
        if st == 'D':
            log(f"[D] one-shot: {json.dumps(res[st].get('one_shot'), ensure_ascii=False)[:400]}")
    all_logs += w.log
    prevw = w
log(f"done {time.time()-t0:.0f}s; stats {agent.stats}; mix {agent.core.mix.n}; hip {agent.core.hip.n}")
json.dump(res_all, open(OUT + ".json", "w"), ensure_ascii=False, default=str)
pickle.dump(all_logs, open(OUT + "_log.pkl", "wb"))
if save:
    pickle.dump(agent, open(OUT + "_agent.pkl", "wb"))
