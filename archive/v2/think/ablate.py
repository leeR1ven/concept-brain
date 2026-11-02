# -*- coding: utf-8 -*-
"""Causal ablations: every learned part must reach behaviour.  From a trained agent (pickle or in memory), each part is
switched off and the agent answers the SAME probe-only schedule (World(stage, seed, probe_frac=1, held_frac=.3)):
B and C frozen (learning off); D with storage on (one-shot memory is what D tests), learning of readouts off.
usage: python3 ablate.py agent.pkl stages=BCD n=150 seed=0 [procs=1]"""
import os, sys, copy, json, pickle, time
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from agent import DevAgent, BENCH          # noqa: E402
sys.path.insert(0, BENCH)
from world import World                    # noqa: E402
from metrics import evaluate               # noqa: E402

ABL = {
    "full": {},
    "no GEN (cortex readout)": dict(use_gen=False),
    "no COPY_prompt gate": dict(copy_prompt=False),
    "no hippocampal recall routes": dict(copy_next=False, copy_comp=False),
    "no hippocampus at all": dict(hippo=False),
    "no learned attention": dict(att_learn=False),
    "no comparator / primes": dict(match=False, vis_rec=False),
    "no visual recognition VR": dict(vis_rec=False),
    "no scene focus F": dict(focus=False),
    "no slow context CTX": dict(ctx=False),
}


def run_one(agent, name, flags, stage, n, seed):
    ag = copy.deepcopy(agent)
    for k, v in flags.items():
        setattr(ag.core.c, k, v)
    if stage in "BC":
        ag.learning = False
    else:                                   # D: the index must store the one-shot facts; readouts frozen
        ag.learning = True
        ag.core.c.learn_listen = False
    w = World(stage, seed=seed, tutor="recast", probe_frac=1.0, held_frac=0.3)   # same seed = same voice and eye codes
    w.run(ag, n)
    res = evaluate(w.log)[stage]
    return {k: (round(v["acc"], 3) if v.get("acc") is not None else None, v.get("n"),
                round((v.get("held") or {}).get("acc") or 0, 3)) for k, v in res.items() if isinstance(v, dict) and "acc" in v}


_AG = None


def _work(job):
    st, nm = job
    agent, n, seed = _AG
    return run_one(agent, nm, ABL[nm], st, n, seed)


def main():
    A = dict(a.split("=", 1) for a in sys.argv[2:] if "=" in a)
    agent = pickle.load(open(sys.argv[1], "rb"))
    stages, n, seed = A.get("stages", "BCD"), int(A.get("n", 150)), int(A.get("seed", 0))
    names = A.get("only", "").split(",") if A.get("only") else list(ABL)
    out = {}
    procs = int(A.get("procs", 1))
    jobs = [(st, nm) for st in stages for nm in names]
    def show(st, nm, r):
        print(f"[{st}] {nm:30s} ALL {r.get('ALL')}  " + " ".join(f"{k}={v[0]}/held {v[2]}" for k, v in r.items()
              if k not in ('ALL', 'SCENE', 'NOSCENE', 'after_model', 'one_shot')), flush=True)
    if procs > 1:
        import multiprocessing as mp
        global _AG
        _AG = (agent, n, seed)
        with mp.get_context("fork").Pool(procs) as pool:
            for (st, nm), r in zip(jobs, pool.map(_work, jobs, chunksize=1)):
                out[(st, nm)] = r; show(st, nm, r)
    else:
        for st, nm in jobs:
            r = run_one(agent, nm, ABL[nm], st, n, seed); out[(st, nm)] = r; show(st, nm, r)
    json.dump({f"{k[0]}|{k[1]}": v for k, v in out.items()}, open(sys.argv[1].replace(".pkl", "_ablate.json"), "w"), ensure_ascii=False)


if __name__ == "__main__":
    main()
