# -*- coding: utf-8 -*-
"""ARCH_v2-tune driver: one configuration, one world seed, one rung of the successive-halving search.
Protocol (develop/sizetune.py + mechanics + held-out + a configurable curriculum):
  A (listen, nA dialogues) -> babbling pass (35,960 sets) -> stages 1..R (nS dialogues each):
     cur=seq:  B, C, D        cur=inter: B, BC, BCD      cur=mix: stage 2 'B'*b2+'C'*c2, stage 3 'B'*b3+'C'*c3+'D'*d3
  after each stage: the FROZEN agent answers the B/C/D probe schedules World(stage, seed, probe_frac=1, held_frac=0.3)
     (evaln dialogues; B, C learning off; D: storage on, readouts frozen) -> acc ALL, held-out acc, per task.
  drift: probes of stage `dstage` before -> after drift_n untutored dialogues (probe_frac=1, held_frac=0), learning on.
  mechanics: overlap / not-3-tick syllables / garble / extra turns summed over every training and probe world.
usage: python3 tune.py tag=x seed=0 R=3 cur=seq nA=300 nS=1000 evaln=80 drift=300 out=out/res.jsonl [cfg keys ...]"""
import os, sys, json, time, copy
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from agent import DevAgent, BENCH          # noqa: E402
sys.path.insert(0, BENCH)
from world import World                    # noqa: E402
from metrics import evaluate               # noqa: E402
from core import Cfg                       # noqa: E402

MECH_KEYS = ('mouth_while_partner', 'not_3_ticks', 'garble', 'extra_turns')
AGENT_KW = {'theta_z': float, 'z_rate': float, 'quiet_ticks': int}


def mech_add(tot, res):
    for st, S in res.items():
        if isinstance(S, dict) and 'mech' in S:
            m = S['mech']
            for k in MECH_KEYS:
                tot[k] = tot.get(k, 0) + int(m.get(k) or 0)
            tot['replies'] = tot.get('replies', 0) + int(m.get('exchanges') or 0)
            lat = m.get('latency_median')
            if lat is not None:
                tot.setdefault('lat', []).append(lat)


def probe(agent, stage, n, seed, mech):
    ag = copy.deepcopy(agent)
    if stage in "BC":
        ag.learning = False
    else:                                   # D: the index must store the one-shot facts; readouts frozen
        ag.learning = True
        ag.core.c.learn_listen = False
    w = World(stage, seed=seed, tutor="recast", probe_frac=1.0, held_frac=0.3)
    w.run(ag, n)
    res = evaluate(w.log)
    mech_add(mech, res)
    S = res[stage]
    out = {k: [round(v["acc"], 4) if v.get("acc") is not None else None, v.get("n"),
               (round(v["held"]["acc"], 4) if (v.get("held") or {}).get("acc") is not None else None),
               (v.get("held") or {}).get("n")]
           for k, v in S.items() if isinstance(v, dict) and "acc" in v}
    return out


def main():
    A = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
    def arg(k, d):
        v = A.pop(k, None)
        return d if v is None else type(d)(v)
    tag, seed, R, cur = arg("tag", "x"), arg("seed", 0), arg("R", 3), arg("cur", "seq")
    nA, nS, evaln, drift_n = arg("nA", 300), arg("nS", 1000), arg("evaln", 80), arg("drift", 300)
    dstage = arg("dstage", "C" if R >= 2 else "B")
    outp = arg("out", "")
    b2, c2, b3, c3, d3 = arg("b2", 1), arg("c2", 1), arg("b3", 1), arg("c3", 1), arg("d3", 1)
    nS1 = arg("nS1", nS)
    akw = {k: AGENT_KW[k](A.pop(k)) for k in list(A) if k in AGENT_KW}
    cfgkw = {}
    for k, v in A.items():
        d = getattr(Cfg(), k)
        cfgkw[k] = (v in ("1", "True", "true")) if isinstance(d, bool) else type(d)(v)
    cfgkw.setdefault("floor", 0.3)
    if cur == "seq":
        stages = ["B", "C", "D"]
    elif cur == "inter":
        stages = ["B", "BC", "BCD"]
    else:
        stages = ["B", "B" * b2 + "C" * c2, "B" * b3 + "C" * c3 + "D" * d3]
    stages = stages[:R]
    t0 = time.time()
    def log(s):
        print(f"[{tag} s{seed} {time.time() - t0:.0f}s] {s}", flush=True)
    wA = World("A", seed=seed, tutor="recast")
    agent = DevAgent(wA, seed=seed, **akw, **cfgkw)
    log(f"start R={R} cur={cur} stages={stages} nS={nS} agent={akw} cfg={cfgkw}")
    wA.run(agent, nA); agent.babble(wA.voice, 35960)
    mech = {}
    mech_train = {}
    prev, table = wA, {}
    for i, st in enumerate(stages):
        w = World(st, seed=seed, tutor="recast", prev=prev)
        w.run(agent, nS1 if i == 0 else nS); prev = w
        mech_add(mech_train, evaluate(w.log))
        ev = {es: probe(agent, es, evaln, seed, mech) for es in ("BCD"[:max(R, 1)] if R < 3 else "BCD")}
        table[f"s{i + 1}"] = ev
        log(f"after stage {i + 1} ({st}): " + "  ".join(f"{es} {ev[es]['ALL'][0]} (held {ev[es]['ALL'][2]})" for es in ev)
            + f" | mix {agent.core.mix.n} hip {agent.core.hip.n} splits {agent.core.stats['splits']} "
              f"recalls {agent.core.stats['awake_recalls']}")
    drift = None
    if drift_n:
        base = probe(agent, dstage, evaln, seed, mech)
        ag2 = copy.deepcopy(agent); ag2.learning = True
        wf = World(dstage, seed=seed, tutor="recast", probe_frac=1.0, held_frac=0.0)
        wf.run(ag2, drift_n)
        mech_add(mech, evaluate(wf.log))
        after = probe(ag2, dstage, evaln, seed, mech)
        drift = dict(stage=dstage, before=base, after=after)
        log(f"drift {dstage}: {base['ALL'][0]} -> {after['ALL'][0]} after {drift_n} untutored dialogues")
    for m in (mech, mech_train):
        if 'lat' in m:
            m['lat'] = sorted(set(m['lat']))
    viol = sum(mech.get(k, 0) + mech_train.get(k, 0) for k in MECH_KEYS)
    rec = dict(tag=tag, seed=seed, R=R, cur=cur, stages=stages, nA=nA, nS=nS, evaln=evaln, drift_n=drift_n,
               agent=akw, cfg=cfgkw, table=table, drift=drift, mech_probe=mech, mech_train=mech_train, viol=viol,
               stats={k: (int(v) if isinstance(v, (int, float)) else v) for k, v in agent.core.stats.items()},
               mix=int(agent.core.mix.n), hip=int(agent.core.hip.n), secs=round(time.time() - t0, 1))
    if not outp:
        os.makedirs(os.path.join(HERE, "out", "runs"), exist_ok=True)
        outp = os.path.join(HERE, "out", "runs", f"{tag}_s{seed}.json")
    with open(outp, "a") as f:
        f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
    log(f"done viol={viol} secs={rec['secs']}")


if __name__ == "__main__":
    main()
