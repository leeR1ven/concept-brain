# -*- coding: utf-8 -*-
"""P5.2: the staged J protocol of ref/run_j.py (unchanged: A 300 -> babbling -> B, BC, BCDD x 1000; frozen B/C/D probes
after each stage; drift = C after 300 untutored C dialogues) with the torch brain (TorchAgent, B = 1) instead of the
numpy SynthAgent.  Writes the same record format, so ref/analyze.py scores it unchanged.
usage: python3 run_j_torch.py seed=0 [tag=torch] [nA=300 nS=1000 evaln=80 drift=300] [out=...]"""
import os, sys, json, time, copy
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import torch                                          # noqa: E402
from agents_torch import TorchAgent                   # noqa: E402
from brain_torch import Cfg                           # noqa: E402
from world import World                               # noqa: E402
from metrics import evaluate                          # noqa: E402

MECH_KEYS = ('mouth_while_partner', 'not_3_ticks', 'garble', 'extra_turns')


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
    else:
        ag.learning = True
        ag.set_learn_listen(False)
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
    del ag
    return out


def main():
    A = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
    arg = lambda k, d: type(d)(A.get(k, d))
    tag, seed = arg("tag", "torch"), arg("seed", 0)
    nA, nS, evaln, drift_n = arg("nA", 300), arg("nS", 1000), arg("evaln", 80), arg("drift", 300)
    outp = arg("out", os.path.join(HERE, "out", "jruns", f"{tag}_s{seed}.json"))
    dev = arg("device", "cuda")
    if dev == 'cuda':
        torch.cuda.set_per_process_memory_fraction(float(A.get('memfrac', 0.3)))
    stages = ["B", "BC", "BCDD"][:arg("nstages", 3)]
    wkw = dict(eye_layout='retino')
    t0 = time.time()
    log = lambda s: print(f"[{tag} s{seed} {time.time() - t0:.0f}s] {s}", flush=True)
    wA = World("A", seed=seed, tutor="recast", **wkw)
    agent = TorchAgent(wA, seed=seed, cfg=Cfg(), device=dev)
    log(f"start torch port, exact={agent.cfg.exact}")
    wA.run(agent, nA); agent.babble(wA.voice, 35960)
    log(f"A: {agent.summary()}")
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
            + f" | online {train_res[f's{i + 1}']} | {agent.summary()} mech_train {mech_train}")
    drift = None
    if drift_n:
        base_p = probe(agent, "C", evaln, seed, mech, wkw)
        ag2 = copy.deepcopy(agent); ag2.learning = True
        wf = World("C", seed=seed, tutor="recast", probe_frac=1.0, held_frac=0.0, **wkw)
        wf.run(ag2, drift_n)
        mech_add(mech, evaluate(wf.log))
        after = probe(ag2, "C", evaln, seed, mech, wkw)
        drift = dict(stage="C", before=base_p, after=after)
        log(f"drift C: {base_p['ALL'][0]} -> {after['ALL'][0]} after {drift_n} untutored dialogues")
    for m in (mech, mech_train):
        if 'lat' in m:
            m['lat'] = sorted(set(m['lat']))
    viol = sum(mech.get(k, 0) + mech_train.get(k, 0) for k in MECH_KEYS)
    b = agent.brain
    rec = dict(tag=tag, seed=seed, R=3, cur="mix", stages=stages, nA=nA, nS=nS, evaln=evaln, drift_n=drift_n,
               agent={}, cfg=dict(port='torch', exact=agent.cfg.exact), world=wkw, table=table, train=train_res,
               drift=drift, mech_probe=mech, mech_train=mech_train, viol=viol, stats=b.stats,
               mix=int(b.conj.n[0].item()), hip=int(b.hn[0]), secs=round(time.time() - t0, 1),
               gpu_mem_peak_mb=round(torch.cuda.max_memory_allocated() / 2 ** 20, 1) if dev == 'cuda' else None)
    os.makedirs(os.path.dirname(outp), exist_ok=True)
    with open(outp, "a") as f:
        f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
    log(f"done viol={viol} secs={rec['secs']}")


if __name__ == "__main__":
    main()
