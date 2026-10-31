# -*- coding: utf-8 -*-
"""Size / tuning vs structure: one variant per run.
A (listen) -> babble -> 3 stages (seq: B, C, D | inter: B, BC, BCD); after each stage the FROZEN agent answers the B, C
and D probe schedules (ablate.run_one 'full': B/C learning off; D storage on, readouts frozen).  At the end: one-shot
(D oneshot task), and drift in untutored free use (C: learning on, nobody teaches, 300 dialogues; C probes before/after).
usage: python3 sizetune.py tag=x seed=0 cur=seq nS=1000 evaln=80 [core cfg keys ...]"""
import os, sys, json, time, copy, pickle
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from agent import DevAgent, BENCH          # noqa: E402
sys.path.insert(0, BENCH)
from world import World                    # noqa: E402
from metrics import evaluate               # noqa: E402
from ablate import run_one                 # noqa: E402
from core import Cfg                       # noqa: E402

A = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
def arg(k, d):
    v = A.pop(k, None)
    return d if v is None else type(d)(v)
tag, seed, cur = arg("tag", "x"), arg("seed", 0), arg("cur", "seq")
nA, nS, evaln, drift_n, save = arg("nA", 300), arg("nS", 1000), arg("evaln", 80), arg("drift", 300), arg("save", 0)
cfgkw = {}
for k, v in A.items():
    d = getattr(Cfg(), k)
    cfgkw[k] = (v in ("1", "True", "true")) if isinstance(d, bool) else type(d)(v)
cfgkw.setdefault("floor", 0.3)
OUT = os.path.join(HERE, "out", f"st_{tag}_s{seed}")
logf = open(OUT + ".log", "w")
def log(s):
    print(s, flush=True); logf.write(s + "\n"); logf.flush()

t0 = time.time()
wA = World("A", seed=seed, tutor="recast")
agent = DevAgent(wA, seed=seed, **cfgkw)
log(f"variant {tag} seed {seed} curriculum {cur} nS {nS} cfg {cfgkw}")
wA.run(agent, nA); agent.babble(wA.voice, 35960)
stages = ["B", "C", "D"] if cur == "seq" else ["B", "BC", "BCD"]
prev, table = wA, {}
def acc_of(r):
    a = r.get("ALL")
    return None if a is None else a[0]
for i, st in enumerate(stages):
    w = World(st, seed=seed, tutor="recast", prev=prev)
    w.run(agent, nS); prev = w
    ev = {}
    for es in "BCD":
        r = run_one(agent, "full", {}, es, evaln, seed)
        ev[es] = r
    table[f"after{i + 1}:{st}"] = ev
    log(f"[{tag}] after stage {i + 1} ({st}): frozen probes B {acc_of(ev['B'])}  C {acc_of(ev['C'])}  D {acc_of(ev['D'])}  | "
        + " ".join(f"{es}.{t}={v[0]}" for es in "BCD" for t, v in ev[es].items() if t not in ("ALL", "SCENE", "NOSCENE", "after_model", "one_shot"))
        + f"  mix={agent.core.mix.n} hip={agent.core.hip.n} ({time.time() - t0:.0f}s)")
# one-shot details from the last D evaluation
one = table[list(table)[-1]]["D"].get("oneshot")
# drift: untutored free use with learning on
if drift_n:
    base = run_one(agent, "full", {}, "C", evaln, seed)
    ag2 = copy.deepcopy(agent); ag2.learning = True
    wf = World("C", seed=seed, tutor="recast", probe_frac=1.0, held_frac=0.0)
    wf.run(ag2, drift_n)
    after = run_one(ag2, "full", {}, "C", evaln, seed)
    log(f"[{tag}] drift: C probes {acc_of(base)} -> {acc_of(after)} after {drift_n} untutored dialogues with learning on")
    table["drift"] = (acc_of(base), acc_of(after))
log(f"[{tag}] one-shot (D oneshot probes after the last stage): {one}")
json.dump(table, open(OUT + ".json", "w"), ensure_ascii=False, default=str)
if save:
    pickle.dump(agent, open(OUT + "_agent.pkl", "wb"))
log(f"[{tag}] done {time.time() - t0:.0f}s")
