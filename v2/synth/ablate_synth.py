# -*- coding: utf-8 -*-
"""Causal lesions of the trained synthesized agent (principle 7: every learned part must reach behaviour).
From the saved final agent (after B -> BC -> BCDD), each part is switched off and a frozen copy answers the SAME
probe-only schedule World(stage, seed, probe_frac=1, held_frac=.3, tutor='recast'): B and C frozen (learning off);
D with episode storage on, readouts frozen (as tune/develop).  Scored per task AND probe-pooled over B+C+D (brainlike's
lesson: per-task means of small tasks swing).  Also logs timing (latency, speaks-after-statement) and mechanics.
usage: python3 ablate_synth.py out/agent_<tag>_s<seed>.pkl seed=0 n=150 [procs=3] [only=a,b]"""
import os, sys, copy, json, pickle, time
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np                          # noqa: E402
from agent_synth import SynthAgent          # noqa: E402,F401
from agent import BENCH                     # noqa: E402
sys.path.insert(0, BENCH)
from world import World                     # noqa: E402
from metrics import evaluate                # noqa: E402

ABL = {
    "full": {},
    "GEN (cortex readout)": dict(cfg=dict(use_gen=False)),
    "BG copy gate (prompt)": dict(cfg=dict(copy_prompt=False)),
    "hippocampal recall routes": dict(cfg=dict(copy_next=False, copy_comp=False)),
    "whole hippocampus": dict(cfg=dict(hippo=False)),
    "learned attention (word->slot)": dict(cfg=dict(att_learn=False)),
    "comparator / primes": dict(cfg=dict(match=False, vis_rec=False)),
    "visual recognition VR": dict(cfg=dict(vis_rec=False)),
    "scene focus F": dict(cfg=dict(focus=False)),
    "slow context CTX": dict(cfg=dict(ctx=False)),
    "learned brake P_oth": dict(agent='P_oth'),
    "learned eagerness V_go": dict(agent='V_go'),
}


def lesion(agent, spec):
    ag = copy.deepcopy(agent)
    for k, v in spec.get('cfg', {}).items():
        setattr(ag.core.c, k, v)
    a = spec.get('agent')
    if a == 'P_oth':
        ag.P_oth.w = {}
    elif a == 'V_go':
        ag.V_go.w = {}; ag.rbar = 0.5
    return ag


def run_one(agent, name, stage, n, seed, wkw):
    ag = lesion(agent, ABL[name])
    if stage in "BC":
        ag.learning = False
    else:
        ag.learning = True
        ag.core.c.learn_listen = False
    w = World(stage, seed=seed, tutor="recast", probe_frac=1.0, held_frac=0.3, **wkw)
    w.run(ag, n)
    res = evaluate(w.log)
    S = res[stage]
    fr = [r for r in w.log if r['kind'] in ('probe', 'recast')]
    lat = [r['latency'] for r in fr if r.get('latency') is not None]
    m = S.get('mech', {})
    out = {k: (round(v["acc"], 3) if v.get("acc") is not None else None, v.get("n"),
               round((v.get("held") or {}).get("acc") or 0, 3)) for k, v in S.items() if isinstance(v, dict) and "acc" in v}
    out['_pool'] = (int(sum(r['reply'] in r['acceptable'] for r in fr)), len(fr))
    lo = [r['latency'] for r in fr if r.get('latency') is not None and r['reply'] in r['acceptable']]
    lb = [r['latency'] for r in fr if r.get('latency') is not None and r['reply'] not in r['acceptable']]
    out['_timing'] = dict(spoke=round(len(lat) / max(len(fr), 1), 3), lat_mean=round(float(np.mean(lat)), 2) if lat else None,
                          lat_ok=round(float(np.mean(lo)), 2) if lo else None, lat_bad=round(float(np.mean(lb)), 2) if lb else None,
                          overlap=int(m.get('mouth_while_partner') or 0), garble=int(m.get('garble') or 0))
    return out


_AG = None


def _work(job):
    st, nm = job
    agent, n, seed, wkw = _AG
    return run_one(agent, nm, st, n, seed, wkw)


def main():
    A = dict(a.split("=", 1) for a in sys.argv[2:] if "=" in a)
    agent = pickle.load(open(sys.argv[1], "rb"))
    stages, n, seed = A.get("stages", "BCD"), int(A.get("n", 150)), int(A.get("seed", 0))
    wkw = {}
    if getattr(agent.core.c, 'rel', False):
        wkw['eye_layout'] = 'retino'
    names = A.get("only", "").split(",") if A.get("only") else list(ABL)
    procs = int(A.get("procs", 1))
    jobs = [(st, nm) for nm in names for st in stages]
    out = {}
    t0 = time.time()
    def show(st, nm, r):
        print(f"[{st}] {nm:32s} ALL {r.get('ALL')} pool {r['_pool']} {r['_timing']} " +
              " ".join(f"{k}={v[0]}" for k, v in r.items() if not k.startswith('_') and k not in
                       ('ALL', 'SCENE', 'NOSCENE', 'after_model', 'one_shot', 'guided', 'learning', 'mech')),
              flush=True)
    if procs > 1:
        import multiprocessing as mp
        global _AG
        _AG = (agent, n, seed, wkw)
        with mp.get_context("fork").Pool(procs) as pool:
            for (st, nm), r in zip(jobs, pool.map(_work, jobs, chunksize=1)):
                out[(st, nm)] = r; show(st, nm, r)
    else:
        for st, nm in jobs:
            r = run_one(agent, nm, st, n, seed, wkw); out[(st, nm)] = r; show(st, nm, r)
    pooled = {}
    for nm in names:
        c = sum(out[(st, nm)]['_pool'][0] for st in stages); t = sum(out[(st, nm)]['_pool'][1] for st in stages)
        pooled[nm] = (c, t, round(c / max(t, 1), 4))
    full = pooled.get('full', (0, 1, 0))[2]
    print("\npooled over " + stages + ":")
    for nm, (c, t, a) in pooled.items():
        print(f"  {nm:32s} {a:.3f} ({c}/{t})  drop {full - a:+.3f}")
    json.dump(dict(per={f"{k[0]}|{k[1]}": v for k, v in out.items()}, pooled=pooled, secs=round(time.time() - t0)),
              open(sys.argv[1].replace(".pkl", "_ablate.json"), "w"), ensure_ascii=False)


if __name__ == "__main__":
    main()
