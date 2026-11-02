# -*- coding: utf-8 -*-
"""tune2 staged-J driver = synth's run_j.py protocol (unchanged metrics, scored by analyze.py) with the T2Agent, plus
  * probes=min : only the frozen probe sets J needs (s1: B; s2: B, C; s3: B, C, D) - training is untouched (every probe
                 runs on a deep copy), so J is identical to probes=all; the drift 'before' = the s3 C probe (identical:
                 same frozen agent, same probe world)
  * ctxp=1     : CONTEXT PROBE on a copy of the final agent (before drift): can the answer depend on an earlier line?
                 dialogue = partner says X (one of 6 objects) -> `gap` filler lines ('好的') -> partner asks 刚才说什么
                 -> acceptable X / 是X.  ctx_train recast dialogues (learning on), then ctx_test frozen probes.
                 gap 0: X is the line before last; gap 1: two lines back.  The hippocampus cannot help (the cue X
                 shares no item with the question); only the thought can carry X.  chance ~ 1/6.
  * records: GEN non-zero weights, MIX units, the thought's context statistics, wiring statistics, secs per phase.
usage: python3 run_t2.py tag=x seed=0 [nA=300 nS=1000 evaln=80 drift=300 probes=min ctxp=1] [w_<k>=v] [agent/cfg keys]"""
import os, sys, json, time, copy, inspect
import numpy as np
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from agent_t2 import T2Agent, T2_KEYS                                 # noqa: E402
from agent_synth import SYNTH_AGENT_KEYS                              # noqa: E402
from agent_think import ThinkAgent                                    # noqa: E402
from agent import BENCH                                               # noqa: E402
sys.path.insert(0, BENCH)
from world import World, OBJECTS                                      # noqa: E402
from metrics import evaluate                                          # noqa: E402
from core_t2 import T2Cfg                                             # noqa: E402
from run_j import TUNE_BEST, THINK_FINAL, mech_add, MECH_KEYS        # noqa: E402
import run_j                                                          # noqa: E402

ARCH_V2 = dict(rebind=1, ct=True, init_min_tau=10, hip_rho=2.0, ctx=False)     # synth's final_noctx keys
_SIG = inspect.signature(ThinkAgent.__init__).parameters


def typed(k, v):
    if k in SYNTH_AGENT_KEYS:
        return SYNTH_AGENT_KEYS[k](v)
    if k in _SIG and _SIG[k].default is not inspect.Parameter.empty:
        d = _SIG[k].default
    else:
        d = getattr(T2Cfg(), k)
    if isinstance(d, bool):
        return v in ("1", "True", "true")
    if isinstance(d, str):
        return v
    return type(d)(v)


def refs(log, tasks=None):
    """trivial references on the SAME scored replies: echo (reply = the prompt) and the best constant answer per task
    type (the single string that is acceptable most often among that task's records: an upper bound on any constant
    answer); also per-task seen / held-out accuracy of the agent."""
    out = {}
    recs = [r for r in log if r['kind'] in ('probe', 'recast')]
    by = {}
    for r in recs:
        by.setdefault(r['task'], []).append(r)
    allr = dict(n=0, agent=0, echo=0, const=0)
    for t, L in by.items():
        cnt = {}
        for r in L:
            for a in set(r['acceptable']):
                cnt[a] = cnt.get(a, 0) + 1
        best = max(cnt, key=cnt.get) if cnt else ''
        n = len(L)
        ag = sum(r['reply'] in r['acceptable'] for r in L)
        ec = sum(r['prompt'] in r['acceptable'] for r in L)
        held = [r for r in L if r.get('held_out')]
        out[t] = dict(n=n, agent=round(ag / n, 4), echo=round(ec / n, 4), const=round(cnt.get(best, 0) / n, 4),
                      const_ans=best, held_n=len(held),
                      held_agent=round(sum(r['reply'] in r['acceptable'] for r in held) / len(held), 4) if held else None,
                      held_const=round(sum(best in r['acceptable'] for r in held) / len(held), 4) if held else None)
        allr['n'] += n; allr['agent'] += ag; allr['echo'] += ec; allr['const'] += cnt.get(best, 0)
    if allr['n']:
        out['ALL'] = dict(n=allr['n'], agent=round(allr['agent'] / allr['n'], 4), echo=round(allr['echo'] / allr['n'], 4),
                          const=round(allr['const'] / allr['n'], 4))
    return out


def probe(agent, stage, n, seed, mech, wkw):
    """run_j.probe + the trivial references of the same probe set (key '_refs')."""
    box = {}
    orig = run_j.evaluate
    def ev(log):
        box['log'] = log
        return orig(log)
    run_j.evaluate = ev
    try:
        out = run_j.probe(agent, stage, n, seed, mech, wkw)
    finally:
        run_j.evaluate = orig
    out['_refs'] = refs(box['log'])
    return out


def ctx_probe(agent, seed, gap, n_train, n_test, wkw, filler='好的', q='刚才说什么'):
    ag = copy.deepcopy(agent)
    ag.learning = True
    w = World('B', seed=seed, tutor='recast', **wkw)            # same seed = same voices (content is scripted here)
    w.agent = ag
    w._hook('bench_attach', w)
    w._silence(w.p['lead_in'])
    rng = np.random.default_rng(7919 * seed + 31 * gap + 5)
    X = OBJECTS[:6]
    for i in range(n_train + n_test):
        if i == n_train:
            ag.learning = False
        x = X[int(rng.integers(len(X)))]
        ev = [('scene', None), ('stmt', dict(task='ctx_cue', text=x))]
        ev += [('stmt', dict(task='ctx_fill', text=filler))] * gap
        ev.append(('ex', dict(stage='S', task='ctx', prompt=q, acceptable=[x, '是' + x], model=x, item=('ctx', x),
                              kind='recast' if i < n_train else 'probe', held_out=False, needs_scene=False)))
        w._run_dialogue(dict(stage='S', events=ev))
    ex = [r for r in w.log if r.get('task') == 'ctx' and r['kind'] in ('probe', 'recast')]
    tr, te = ex[:n_train], ex[n_train:]
    acc = lambda L: round(float(np.mean([r['reply'] in r['acceptable'] for r in L])), 4) if L else None
    first = lambda L: round(float(np.mean([bool(r['reply']) and r['reply'][0] == r['acceptable'][0][0] for r in L])), 4) if L else None
    obj = lambda L: round(float(np.mean([any(o in r['reply'] for o in X) for r in L])), 4) if L else None
    mech = {k: int(sum(int(r.get(k) or 0) for r in w.log)) for k in ('mouth_while_partner', 'garble', 'extra_turns')}
    tg = [r['acceptable'][0] for r in te]
    const = round(max(tg.count(x) for x in set(tg)) / len(tg), 4) if tg else None
    return dict(gap=gap, const=const, echo=0.0, train_acc=acc(tr), train_last=acc(tr[-50:]), test_acc=acc(te), test_first=first(te),
                test_says_object=obj(te), spoke=round(float(np.mean([bool(r['reply']) for r in te])), 3) if te else None,
                n_test=len(te), mech=mech, replies=[r['reply'] for r in te[:12]], targets=[r['acceptable'][0] for r in te[:12]])


def main():
    A = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
    def arg(k, d):
        v = A.pop(k, None)
        return d if v is None else type(d)(v)
    tag, seed = arg("tag", "x"), arg("seed", 0)
    nA, nS, evaln, drift_n = arg("nA", 300), arg("nS", 1000), arg("evaln", 80), arg("drift", 300)
    probes, ctxp = arg("probes", "min"), arg("ctxp", 1)
    ctx_train, ctx_test = arg("ctx_train", 150), arg("ctx_test", 60)
    train_held = arg("train_held", -1.0)          # >= 0: held_frac of the TRAINING worlds (control; bench default 0.15)
    outp = arg("out", "")
    stages = ["B", "BC", "BCDD"]
    need = {0: "B", 1: "BC", 2: "BCD"} if probes == "min" else {0: "BCD", 1: "BCD", 2: "BCD"}
    wkw = {'eye_layout': 'retino'}
    for k in list(A):
        if k.startswith("w_"):
            v = A.pop(k)
            wkw[k[2:]] = v if not v.replace('.', '').replace('-', '').isdigit() else (float(v) if '.' in v else int(v))
    kw = dict(TUNE_BEST); kw.update(THINK_FINAL); kw.update(ARCH_V2)
    for k, v in A.items():
        kw[k] = typed(k, v)
    t0 = time.time()
    tph = {}
    def log(s):
        print(f"[{tag} s{seed} {time.time() - t0:.0f}s] {s}", flush=True)
    wA = World("A", seed=seed, tutor="recast", **wkw)
    agent = T2Agent(wA, seed=seed, **kw)
    log(f"start stages={stages} nS={nS} world={wkw} kw={kw}")
    wA.run(agent, nA); agent.babble(wA.voice, 35960)
    tph['A'] = round(time.time() - t0, 1)
    mech, mech_train = {}, {}
    prev, table, train_res = wA, {}, {}
    for i, st in enumerate(stages):
        ts = time.time()
        tkw = dict(wkw)
        if train_held >= 0:
            tkw['held_frac'] = train_held
        w = World(st, seed=seed, tutor="recast", prev=prev, **tkw)
        w.run(agent, nS); prev = w
        tph[f's{i + 1}_train'] = round(time.time() - ts, 1)
        rs = evaluate(w.log)
        mech_add(mech_train, rs)
        train_res[f"s{i + 1}"] = {s: (round(rs[s]['ALL']['acc'], 4) if rs.get(s, {}).get('ALL', {}).get('acc') is not None
                                     else None) for s in "BCD" if s in rs}
        ts = time.time()
        ev = {es: probe(agent, es, evaln, seed, mech, wkw) for es in need[i]}
        tph[f's{i + 1}_probe'] = round(time.time() - ts, 1)
        table[f"s{i + 1}"] = ev
        log(f"after stage {i + 1} ({st}): " + "  ".join(f"{es} {ev[es]['ALL'][0]} (held {ev[es]['ALL'][2]})" for es in ev)
            + f" | online {train_res[f's{i + 1}']} | mix {agent.core.mix.n} hip {agent.core.hip.n} "
              f"recalls {agent.core.stats['awake_recalls']} mech_train {mech_train}")
    ctx = []
    if ctxp:
        ts = time.time()
        for gap in (0, 1):
            ctx.append(ctx_probe(agent, seed, gap, ctx_train, ctx_test, wkw))
            log(f"context probe gap {gap}: {ctx[-1]}")
        tph['ctx_probe'] = round(time.time() - ts, 1)
    drift = None
    if drift_n:
        ts = time.time()
        base_p = table["s3"]["C"] if probes == "min" else probe(agent, "C", evaln, seed, mech, wkw)
        ag2 = copy.deepcopy(agent); ag2.learning = True
        wf = World("C", seed=seed, tutor="recast", probe_frac=1.0, held_frac=0.0, **wkw)
        wf.run(ag2, drift_n)
        mech_add(mech, evaluate(wf.log))
        after = probe(ag2, "C", evaln, seed, mech, wkw)
        drift = dict(stage="C", before=base_p, after=after)
        tph['drift'] = round(time.time() - ts, 1)
        log(f"drift C: {base_p['ALL'][0]} -> {after['ALL'][0]} after {drift_n} untutored dialogues")
    for m in (mech, mech_train):
        if 'lat' in m:
            m['lat'] = sorted(set(m['lat']))
    viol = sum(mech.get(k, 0) + mech_train.get(k, 0) for k in MECH_KEYS)
    core = agent.core
    nnz, nmix = core.gen_nnz()
    cs = core.ctx_stats
    ctxs = dict(reply_views=cs['reply_views'], mean_held=round(cs['held_sum'] / max(cs['reply_views'], 1), 3),
                back=[round(b / max(cs['reply_views'], 1), 4) for b in cs['back']],
                back_any=[round(b / max(cs['reply_views'], 1), 4) for b in cs['back_any']])
    ws = dict(core.wstats)
    ws.update(core.wiring_summary())
    if ws['masked_events']:
        ws['allowed_frac'] = round(ws['allowed_frac_sum'] / ws['masked_events'], 4)
    rec = dict(tag=tag, seed=seed, R=3, cur="mix", stages=stages, nA=nA, nS=nS, evaln=evaln, drift_n=drift_n,
               probes=probes, train_held=train_held, agent={}, cfg={k: (v if isinstance(v, (int, float, str, bool)) else str(v)) for k, v in kw.items()},
               world=wkw, table=table, train=train_res, drift=drift, ctx_probe=ctx, mech_probe=mech,
               mech_train=mech_train, viol=viol,
               stats={k: (int(v) if isinstance(v, (int, float)) else v) for k, v in core.stats.items()},
               tstats=agent.tstats, ct=core.ct_stats, ctx_stats=ctxs, wiring=ws, gen_nnz=nnz,
               gen_nnz_per_unit=round(nnz / max(nmix, 1), 2), mix=int(nmix), hip=int(core.hip.n),
               cats=int(agent.Z.n), phases=tph, secs=round(time.time() - t0, 1), numpy=np.__version__)
    if not outp:
        os.makedirs(os.path.join(HERE, "out", "runs"), exist_ok=True)
        outp = os.path.join(HERE, "out", "runs", f"{tag}_s{seed}.json")
    with open(outp, "a") as f:
        f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
    log(f"done viol={viol} secs={rec['secs']} J-inputs ok; gen nnz/unit {rec['gen_nnz_per_unit']} mix {nmix}")


if __name__ == "__main__":
    main()
