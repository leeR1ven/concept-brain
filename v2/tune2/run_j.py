# -*- coding: utf-8 -*-
"""Staged-mode objective J for the synthesized agent (tune.py's protocol, unchanged metrics; analyze.py scores it).
  A (listen, nA dialogues) -> babbling pass -> stages B, 'BC', 'BCDD' (nS dialogues each; tune's mixB curriculum)
  after each stage: a FROZEN COPY answers the B/C/D probe schedules (80 dialogues each, held_frac 0.3)
  drift: C probes before -> after 300 untutored C dialogues (learning on)
  mechanics: overlap / non-3-tick syllables / garble / extra turns summed over every world (must be 0)
usage: python3 run_j.py tag=x seed=0 [nA=300 nS=1000 evaln=80 drift=300] [save=1] [w_<world key>=v] [agent/cfg keys]
Agent keys are ThinkAgent's (policy, u_reply, brake_now, own_tail, ...) and SynthAgent's; all other keys are core cfg
keys (Cfg / TCfg / SCfg).  Defaults = tune's recommended configuration (recall 8 / 0.2 / uniform, 10k hippocampus)."""
import os, sys, json, time, copy, pickle, inspect
import numpy as np
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from agent_synth import SynthAgent, THINK_KEYS, SYNTH_AGENT_KEYS     # noqa: E402
from agent_think import ThinkAgent                                  # noqa: E402
from agent import BENCH                                             # noqa: E402
sys.path.insert(0, BENCH)
from world import World                                             # noqa: E402
from metrics import evaluate                                        # noqa: E402
from core_synth import SCfg                                         # noqa: E402

MECH_KEYS = ('mouth_while_partner', 'not_3_ticks', 'garble', 'extra_turns')
TUNE_BEST = dict(recall_n=8, eta_recall=0.2, rec_pow=0.0, hip_cap=10000, hip_ndg=6000, hip_kdg=120, floor=0.3)
THINK_FINAL = dict(policy='free', ior=0, think_k=3, urg_gate=False, brake_now=True, late_release=False, u_reply=0.15,
                   u_R1=1.0, own_tail=True)
_SIG = inspect.signature(ThinkAgent.__init__).parameters


def typed(k, v):
    if k in SYNTH_AGENT_KEYS:
        return SYNTH_AGENT_KEYS[k](v)
    if k in _SIG and _SIG[k].default is not inspect.Parameter.empty:
        d = _SIG[k].default
    else:
        d = getattr(SCfg(), k)
    if isinstance(d, bool):
        return v in ("1", "True", "true")
    if isinstance(d, str):
        return v
    return type(d)(v)


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


def probe(agent, stage, n, seed, mech, wkw):
    ag = copy.deepcopy(agent)
    if stage in "BC":
        ag.learning = False
    else:                                   # D: the index must store the one-shot facts; readouts frozen
        ag.learning = True
        ag.core.c.learn_listen = False
    w = World(stage, seed=seed, tutor="recast", probe_frac=1.0, held_frac=0.3, **wkw)
    w.run(ag, n)
    res = evaluate(w.log)
    mech_add(mech, res)
    S = res[stage]
    out = {k: [round(v["acc"], 4) if v.get("acc") is not None else None, v.get("n"),
               (round(v["held"]["acc"], 4) if (v.get("held") or {}).get("acc") is not None else None),
               (v.get("held") or {}).get("n"), round(v["spoke"], 3) if v.get("spoke") is not None else None]
           for k, v in S.items() if isinstance(v, dict) and "acc" in v}
    ov = [dict(task=r.get('task'), kind=r.get('kind'), prompt=r.get('prompt'), reply=r.get('reply'), lat=r.get('latency'),
               mwp=r.get('mouth_while_partner'), stop=r.get('stop')) for r in w.log if (r.get('mouth_while_partner') or 0) > 0]
    if ov:
        mech.setdefault('overlap_records', []).extend(ov[:5])
    fr = [r for r in w.log if r['kind'] in ('probe', 'recast')]
    first, later, last_dlg = [], [], None
    for r in fr:
        (first if r['dlg'] != last_dlg else later).append(bool(r['reply']))
        last_dlg = r['dlg']
    out['_spoke_pos'] = [round(float(np.mean(first)), 3) if first else None, round(float(np.mean(later)), 3) if later else None]
    return out


def main():
    A = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
    def arg(k, d):
        v = A.pop(k, None)
        return d if v is None else type(d)(v)
    tag, seed = arg("tag", "x"), arg("seed", 0)
    nA, nS, evaln, drift_n = arg("nA", 300), arg("nS", 1000), arg("evaln", 80), arg("drift", 300)
    save, base = arg("save", 0), arg("base", "think")
    outp = arg("out", "")
    stages = ["B", "BC", "BCDD"]
    wkw = {}
    for k in list(A):
        if k.startswith("w_"):
            v = A.pop(k)
            wkw[k[2:]] = v if not v.replace('.', '').replace('-', '').isdigit() else (float(v) if '.' in v else int(v))
    kw = dict(TUNE_BEST)
    if base == "think":
        kw.update(THINK_FINAL)
    elif base == "v1":
        kw.update(policy='v1', think_k=3, ior=0)
    for k, v in A.items():
        kw[k] = typed(k, v)
    t0 = time.time()
    def log(s):
        print(f"[{tag} s{seed} {time.time() - t0:.0f}s] {s}", flush=True)
    wA = World("A", seed=seed, tutor="recast", **wkw)
    agent = SynthAgent(wA, seed=seed, **kw)
    log(f"start base={base} stages={stages} nS={nS} world={wkw} kw={kw}")
    wA.run(agent, nA); agent.babble(wA.voice, 35960)
    log(f"A: categories {agent.Z.n} programs {len(agent.cat2prog)}")
    mech, mech_train = {}, {}
    prev, table, train_res = wA, {}, {}
    for i, st in enumerate(stages):
        w = World(st, seed=seed, tutor="recast", prev=prev, **wkw)
        w.run(agent, nS); prev = w
        rs = evaluate(w.log)
        mech_add(mech_train, rs)
        train_res[f"s{i + 1}"] = {s: (round(rs[s]['ALL']['acc'], 4) if rs.get(s, {}).get('ALL', {}).get('acc') is not None
                                     else None) for s in "BCD" if s in rs}
        ev = {es: probe(agent, es, evaln, seed, mech, wkw) for es in "BCD"}
        table[f"s{i + 1}"] = ev
        log(f"after stage {i + 1} ({st}): " + "  ".join(f"{es} {ev[es]['ALL'][0]} (held {ev[es]['ALL'][2]})" for es in ev)
            + f" | online {train_res[f's{i + 1}']} | mix {agent.core.mix.n} hip {agent.core.hip.n} "
              f"recalls {agent.core.stats['awake_recalls']} mech_train {mech_train} | t {agent.tstats}")
    drift = None
    if drift_n:
        base_p = probe(agent, "C", evaln, seed, mech, wkw)
        ag2 = copy.deepcopy(agent); ag2.learning = True
        wf = World("C", seed=seed, tutor="recast", probe_frac=1.0, held_frac=0.0, **wkw)
        wf.run(ag2, drift_n)
        mech_add(mech, evaluate(wf.log))
        ovd = [dict(task=r.get('task'), kind=r.get('kind'), prompt=r.get('prompt'), reply=r.get('reply'),
                    lat=r.get('latency'), mwp=r.get('mouth_while_partner')) for r in wf.log if (r.get('mouth_while_partner') or 0) > 0]
        if ovd:
            mech.setdefault('overlap_records', []).extend(ovd[:5])
        after = probe(ag2, "C", evaln, seed, mech, wkw)
        drift = dict(stage="C", before=base_p, after=after)
        log(f"drift C: {base_p['ALL'][0]} -> {after['ALL'][0]} after {drift_n} untutored dialogues")
    for m in (mech, mech_train):
        if 'lat' in m:
            m['lat'] = sorted(set(m['lat']))
    viol = sum(mech.get(k, 0) + mech_train.get(k, 0) for k in MECH_KEYS)
    if mech.get('overlap_records'):
        log(f"overlap records: {mech['overlap_records']}")
    rec = dict(tag=tag, seed=seed, R=3, cur="mix", stages=stages, nA=nA, nS=nS, evaln=evaln, drift_n=drift_n,
               agent={}, cfg={k: (v if isinstance(v, (int, float, str, bool)) else str(v)) for k, v in kw.items()},
               world=wkw, table=table, train=train_res, drift=drift, mech_probe=mech, mech_train=mech_train, viol=viol,
               stats={k: (int(v) if isinstance(v, (int, float)) else v) for k, v in agent.core.stats.items()},
               tstats=agent.tstats, ct=agent.core.ct_stats, name=agent.core.name_stats,
               mix=int(agent.core.mix.n), hip=int(agent.core.hip.n), secs=round(time.time() - t0, 1))
    if not outp:
        outp = os.path.join(HERE, "out", "runs", f"{tag}_s{seed}.json")
    with open(outp, "a") as f:
        f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
    if save:
        agent.th_log = []
        with open(os.path.join(HERE, "out", f"agent_{tag}_s{seed}.pkl"), "wb") as f:
            pickle.dump(agent, f)
    log(f"done viol={viol} secs={rec['secs']}")


if __name__ == "__main__":
    main()
