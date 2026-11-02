# -*- coding: utf-8 -*-
"""ARCH_v2.1 staged driver = synth/run_j.py's protocol (metric code of the bench untouched) + honest measurement.

  proto=frozen (NEW, default): the training worlds use held_frac 0, so a held-out (task, filler) pair is never heard
                 in training at all; held-out pairs are asked only of FROZEN copies (probe worlds, held_frac 0.3).
  proto=old:     ARCH_v2's protocol (training worlds keep held_frac 0.15: held-out pairs are asked live, without
                 feedback, with learning on), for comparison.
Added measurements (all in this driver; bench metrics unchanged):
  * per-task accuracy and held-out accuracy of the final (s3) frozen probes;
  * role confusion: share of free replies that are partner lines (an exact partner line, a prefix of one, or a reply
    with a question syllable 什么吗哪几谁) or echoes (the prompt or a >= 2-char piece of it), when not acceptable;
  * same-run trivial references on the same probe items: echo (say the prompt), best constant per task type (the one
    reply per task that scores best on these very items: an upper bound for any constant responder) and the prior
    constant (the reply most often taught for that task in this run's training);
  * J  = F + 0.5 RET + 0.25 H + 0.25 DR   (ARCH_v2's objective, analyze.py)
    J2 = F + 0.5 RET + 1.00 H + 0.25 DR   (proposed: generalisation weighs as much as practised accuracy)
  * drift after 300 AND 900 untutored C dialogues (learning on), from the same copy;
  * comprehension vs production on the s3 frozen probes (develop's D2/D3): the slot the reply thought attends
    (where/colour/count) vs X's slot, and whether visual recognition at the pointed slot holds X's name ('what').
usage: python3 run_v21.py tag=x seed=0 [proto=frozen|old] [nA=300 nS=1000 evaln=80 drift=300 drift2=900] [save=1]
       [w_<world key>=v] [agent / cfg keys incl. ral=1 ...]"""
import os, sys, json, time, copy, pickle, inspect, collections
import numpy as np
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'synth'))  # repo: unchanged ARCH_v2 core + bench_copy
from agent_v21 import AgentV21, V21_AGENT_KEYS                       # noqa: E402
from agent_synth import SYNTH_AGENT_KEYS                             # noqa: E402
from agent_think import ThinkAgent                                   # noqa: E402
from agent import BENCH                                              # noqa: E402
sys.path.insert(0, BENCH)
from world import World                                              # noqa: E402
from metrics import evaluate                                         # noqa: E402
from core_v21 import RCfg                                            # noqa: E402
from physics import Stream                                           # noqa: E402

MECH_KEYS = ('mouth_while_partner', 'not_3_ticks', 'garble', 'extra_turns')
TUNE_BEST = dict(recall_n=8, eta_recall=0.2, rec_pow=0.0, hip_cap=10000, hip_ndg=6000, hip_kdg=120, floor=0.3)
THINK_FINAL = dict(policy='free', ior=0, think_k=3, urg_gate=False, brake_now=True, late_release=False, u_reply=0.15,
                   u_R1=1.0, own_tail=True)
ARCH_V2 = dict(rebind=1, ct=True, init_min_tau=10, hip_rho=2.0, ctx=False)          # final_noctx
_SIG = inspect.signature(ThinkAgent.__init__).parameters
QW = set('什么吗哪几谁')
OBJECTS = ['苹果', '香蕉', '西瓜', '葡萄', '饼干', '牛奶', '皮球', '汽车', '小狗', '小猫', '小鸟', '鸭子']


def typed(k, v):
    if k in SYNTH_AGENT_KEYS:
        return SYNTH_AGENT_KEYS[k](v)
    if k in V21_AGENT_KEYS:
        return V21_AGENT_KEYS[k](v)
    if k in _SIG and _SIG[k].default is not inspect.Parameter.empty:
        d = _SIG[k].default
    else:
        d = getattr(RCfg(), k)
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


def classify(r, prompts):
    rep, pr, acc = r['reply'], r['prompt'], r.get('acceptable') or []
    if rep in acc:
        return 'ok'
    if not rep:
        return 'silent'
    if rep == pr or (len(rep) >= 2 and rep in pr):
        return 'echo'
    if rep in prompts or (len(rep) >= 2 and any(p.startswith(rep) for p in prompts)) or (set(rep) & QW):
        return 'partner'
    return 'other'


def role_summary(recs, prompts):
    cls = collections.Counter(classify(r, prompts) for r in recs)
    n = max(1, len(recs))
    return dict(n=len(recs), **{k: round(v / n, 4) for k, v in cls.items()},
                role_conf=round((cls['echo'] + cls['partner']) / n, 4))


def trivial_refs(recs, prior):
    """echo, best constant per task (oracle-chosen on these items), prior constant per task (most taught reply)."""
    out = {}
    for sel, name in ((lambda r: True, 'all'), (lambda r: r['held'], 'held'), (lambda r: not r['held'], 'seen')):
        R = [r for r in recs if sel(r)]
        if not R:
            continue
        echo = np.mean([r['prompt'] in (r.get('acceptable') or []) for r in R])
        best = 0
        by = collections.defaultdict(list)
        for r in R:
            by[r['task']].append(r)
        per_task = {}
        for t, rs in by.items():
            cnt = collections.Counter(a for r in rs for a in set(r.get('acceptable') or []))
            b = cnt.most_common(1)[0][1] if cnt else 0
            best += b
            per_task[t] = dict(n=len(rs), const_best=round(b / len(rs), 3),
                               const_word=cnt.most_common(1)[0][0] if cnt else None,
                               prior=round(np.mean([prior.get(t) in (r.get('acceptable') or []) for r in rs]), 3),
                               echo=round(np.mean([r['prompt'] in (r.get('acceptable') or []) for r in rs]), 3),
                               agent=round(np.mean([r['ok'] for r in rs]), 3))
        out[name] = dict(n=len(R), agent=round(np.mean([r['ok'] for r in R]), 4), echo=round(float(echo), 4),
                         const_best=round(best / len(R), 4),
                         prior=round(np.mean([prior.get(r['task']) in (r.get('acceptable') or []) for r in R]), 4),
                         per_task=per_task)
    return out


def name_cats(agent, voice):
    """the agent's own categories for each object name (its ear hearing the partner say it; frozen)."""
    names = {}
    for X in OBJECTS:
        cats = []
        for ch in X:
            st = Stream(voice.F, voice.T, voice.T // 3, 3, voice.levels)
            e = agent._ear_copy(); none = (None, None, False); g = voice.hear(ch, 'other'); out = None
            for k in range(3):
                out = e.step(st.tick(((ch, g, k == 0), none, none)))
            j, m = agent.Z.best(out[agent.F0])
            cats.append(int(j) if m >= agent.theta_z else None)
        names[X] = tuple(cats)
    return names


def probe(agent, stage, n, seed, mech, wkw, keep=None, comp=None):
    ag = copy.deepcopy(agent)
    if stage in "BC":
        ag.learning = False
    else:                                   # D: the index must store the one-shot facts; readouts frozen
        ag.learning = True
        ag.core.c.learn_listen = False
    spy_rec = []
    if comp is not None:
        core = ag.core
        orig = core.think_step
        def spy(th):
            e = orig(th)
            if th['kind'] == 'reply' and th['k'] in (2, 3):
                VR = th['st'].get('VR')
                spy_rec.append((ag.t, th['k'], th['st'].get('att'),
                                tuple(sorted(int(x) for x in VR)) if VR is not None else ()))
            return e
        core.think_step = spy
    t_off = ag.t
    w = World(stage, seed=seed, tutor="recast", probe_frac=1.0, held_frac=0.3, **wkw)
    w.run(ag, n)
    res = evaluate(w.log)
    mech_add(mech, res)
    S = res[stage]
    out = {k: [round(v["acc"], 4) if v.get("acc") is not None else None, v.get("n"),
               (round(v["held"]["acc"], 4) if (v.get("held") or {}).get("acc") is not None else None),
               (v.get("held") or {}).get("n"), round(v["spoke"], 3) if v.get("spoke") is not None else None]
           for k, v in S.items() if isinstance(v, dict) and "acc" in v}
    if keep is not None:
        for r in w.log:
            if r.get('kind') == 'probe' and r.get('stage') == stage:
                keep.append(dict(stage=stage, task=r['task'], held=bool(r.get('held_out')), prompt=r['prompt'],
                                 reply=r['reply'], ok=r['reply'] in (r.get('acceptable') or []),
                                 acceptable=list(r.get('acceptable') or [])))
    if comp is not None:
        names = comp.get('names')
        for r in w.log:
            if r.get('kind') != 'probe' or not r.get('item'):
                continue
            tp = r['t_prompt_end'] + t_off
            k = 'held' if r.get('held_out') else 'seen'
            ok = r['reply'] in (r.get('acceptable') or [])
            if r['task'] in ('where', 'colour', 'count'):
                X = r['item'][1]
                slots = [i for i, it in enumerate(r['scene'] or ()) if it is not None and it[0] == X]
                th = [x for x in spy_rec if tp - 1 <= x[0] <= tp + 8 and x[1] == 2 and x[2] is not None]
                if not slots:
                    continue
                a = comp.setdefault(f"attend|{r['task']}|{k}", [0, 0, 0])
                a[0] += 1; a[1] += int(bool(th) and th[0][2] == slots[0]); a[2] += int(ok)
            if r['task'] == 'what' and names is not None:
                X = r['item'][1]
                vr = [x[3] for x in spy_rec if tp - 1 <= x[0] <= tp + 8 and x[1] == 3]
                vr = vr[0] if vr else ()
                has = len(vr) > 0 and all(c in vr for c in names[X] if c is not None)
                a = comp.setdefault(f"vr_name|what|{k}", [0, 0, 0])
                a[0] += 1; a[1] += int(has); a[2] += int(ok)
    return out, w.log


def score(table, drift):
    a = lambda s, st, h=False: (table[s][st]['ALL'][2 if h else 0] or 0.0)
    F = (a('s3', 'B') + a('s3', 'C') + a('s3', 'D')) / 3
    RET = (min(0, a('s3', 'B') - a('s1', 'B')) + min(0, a('s3', 'C') - a('s2', 'C'))) / 2
    H = (a('s3', 'B', 1) + a('s3', 'C', 1) + a('s3', 'D', 1)) / 3
    DR = drift['after']['ALL'][0] if drift else 0.0
    return dict(F=round(F, 4), RET=round(RET, 4), H=round(H, 4), DR=round(DR, 4),
                J=round(F + 0.5 * RET + 0.25 * H + 0.25 * DR, 4), J2=round(F + 0.5 * RET + 1.0 * H + 0.25 * DR, 4))


def main():
    A = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
    def arg(k, d):
        v = A.pop(k, None)
        return d if v is None else type(d)(v)
    tag, seed = arg("tag", "x"), arg("seed", 0)
    proto = arg("proto", "frozen")
    nA, nS, evaln, drift_n, drift2 = arg("nA", 300), arg("nS", 1000), arg("evaln", 80), arg("drift", 300), arg("drift2", 900)
    save, outp, comp_on = arg("save", 0), arg("out", ""), arg("comp", 1)
    lesions = [x for x in arg("lesion", "").split(',') if x]     # e.g. lesion=say_gain:0,act:0 (frozen re-probes)
    stages = ["B", "BC", "BCDD"]
    wkw = {'eye_layout': 'retino'}
    for k in list(A):
        if k.startswith("w_"):
            v = A.pop(k)
            wkw[k[2:]] = v if not v.replace('.', '').replace('-', '').isdigit() else (float(v) if '.' in v else int(v))
    wtrain = dict(wkw, held_frac=0.0) if proto == 'frozen' else dict(wkw)
    kw = dict(TUNE_BEST); kw.update(THINK_FINAL); kw.update(ARCH_V2)
    for k, v in A.items():
        kw[k] = typed(k, v)
    t0 = time.time()
    def log(s):
        print(f"[{tag} s{seed} {time.time() - t0:.0f}s] {s}", flush=True)
    wA = World("A", seed=seed, tutor="recast", **wtrain)
    agent = AgentV21(wA, seed=seed, **kw)
    log(f"start proto={proto} nS={nS} world={wkw} train={wtrain} kw={kw}")
    wA.run(agent, nA); agent.babble(wA.voice, 35960)
    log(f"A: categories {agent.Z.n} programs {len(agent.cat2prog)}")
    prompts = set()
    taught = collections.defaultdict(collections.Counter)
    def harvest(lg):
        for r in lg:
            if isinstance(r.get('prompt'), str) and r.get('kind') in ('probe', 'recast', 'model', 'statement', 'overheard'):
                prompts.add(r['prompt'])
            if r.get('kind') == 'recast' and r.get('recast'):
                taught[r['task']][r['recast']] += 1
            if r.get('kind') in ('model', 'overheard') and r.get('model_reply'):
                taught[r['task']][r['model_reply']] += 1
    harvest(wA.log)
    mech, mech_train = {}, {}
    prev, table, train_res, replies, comp = wA, {}, {}, [], {}
    for i, st in enumerate(stages):
        w = World(st, seed=seed, tutor="recast", prev=prev, **wtrain)
        w.run(agent, nS); prev = w
        harvest(w.log)
        rs = evaluate(w.log)
        mech_add(mech_train, rs)
        train_res[f"s{i + 1}"] = {s: (round(rs[s]['ALL']['acc'], 4) if rs.get(s, {}).get('ALL', {}).get('acc') is not None
                                     else None) for s in "BCD" if s in rs}
        last = i == len(stages) - 1
        if last and comp_on:
            try:
                comp['names'] = name_cats(copy.deepcopy(agent), w.voice)
            except Exception as e:                        # noqa: BLE001
                log(f"name cats failed {e}"); comp['names'] = None
        ev = {}
        for es in "BCD":
            ev[es], plog = probe(agent, es, evaln, seed, mech, wkw, keep=replies if last else None,
                                 comp=comp if (last and comp_on and es in 'BC') else None)
            harvest(plog)
        table[f"s{i + 1}"] = ev
        log(f"after stage {i + 1} ({st}): " + "  ".join(f"{es} {ev[es]['ALL'][0]} (held {ev[es]['ALL'][2]})" for es in ev)
            + f" | online {train_res[f's{i + 1}']} | mix {agent.core.mix.n} hip {agent.core.hip.n} "
              f"| ral {agent.core.ral_stats}")
    drift = None
    if drift_n:
        base_p, _ = probe(agent, "C", evaln, seed, mech, wkw)
        ag2 = copy.deepcopy(agent); ag2.learning = True
        wf = World("C", seed=seed, tutor="recast", probe_frac=1.0, held_frac=0.0, **wkw)
        wf.run(ag2, drift_n)
        mech_add(mech, evaluate(wf.log))
        after_rep = []
        after, _ = probe(ag2, "C", evaln, seed, mech, wkw, keep=after_rep)
        drift = dict(stage="C", before=base_p, after=after, n=drift_n)
        log(f"drift C: {base_p['ALL'][0]} -> {after['ALL'][0]} after {drift_n} untutored dialogues")
        if drift2 and drift2 > drift_n:
            wf.run(ag2, drift2 - drift_n)
            mech_add(mech, evaluate(wf.log))
            after2_rep = []
            after2, _ = probe(ag2, "C", evaln, seed, mech, wkw, keep=after2_rep)
            drift['after2'] = after2; drift['n2'] = drift2
            drift['role_after2'] = role_summary(after2_rep, prompts)
            log(f"drift C: -> {after2['ALL'][0]} after {drift2} untutored dialogues")
        drift['role_before'] = role_summary([r for r in replies if r['stage'] == 'C'], prompts)
        drift['role_after'] = role_summary(after_rep, prompts)
    les = {}
    for spec in lesions:                       # principle 7: the learned part must change behaviour when removed
        k, v = spec.split(':')
        ag3 = copy.deepcopy(agent)
        if k == 'act' and getattr(ag3.core, 'act', None) is not None:
            ag3.core.act.W[:] = 0
        elif k == 'gateR' and getattr(ag3.core, 'gateR', None) is not None:
            ag3.core.gateR.W[:] = 0
        elif hasattr(ag3.core.c, k):
            setattr(ag3.core.c, k, type(getattr(ag3.core.c, k))(float(v)))
        else:
            setattr(ag3, k, type(getattr(ag3, k))(float(v)))
        les[spec] = {}
        for es in "BCD":
            kept = []
            o, _ = probe(ag3, es, evaln, seed, {}, wkw, keep=kept)
            les[spec][es] = [o['ALL'][0], o['ALL'][2]]
            for r_ in kept:
                les[spec].setdefault('tasks', {}).setdefault(r_['task'] + ('|held' if r_['held'] else ''), []).append(r_['ok'])
        les[spec]['tasks'] = {t: round(float(np.mean(v_)), 3) for t, v_ in les[spec].get('tasks', {}).items()}
        log(f"lesion {spec}: " + str({es: les[spec][es] for es in 'BCD'}))
    viol = sum(mech.get(k, 0) + mech_train.get(k, 0) for k in MECH_KEYS)
    sc = score(table, drift)
    prior = {t: c.most_common(1)[0][0] for t, c in taught.items() if c}
    refs = trivial_refs(replies, prior)
    role = role_summary(replies, prompts)
    role_held = role_summary([r for r in replies if r['held']], prompts)
    per_task = {}
    for st_ in 'BCD':
        for task, v in table['s3'][st_].items():
            if task in ('ALL', 'SCENE', 'NOSCENE') or task.startswith('_') or not isinstance(v, list):
                continue
            per_task[task] = dict(acc=v[0], n=v[1], held=v[2], n_held=v[3])
    compo = {k: v for k, v in comp.items() if k != 'names'}
    rec = dict(tag=tag, seed=seed, proto=proto, R=3, stages=stages, nA=nA, nS=nS, evaln=evaln, drift_n=drift_n,
               cfg={k: (v if isinstance(v, (int, float, str, bool)) else str(v)) for k, v in kw.items()},
               world=wkw, table=table, train=train_res, drift=drift, viol=viol, mech_probe=mech, mech_train=mech_train,
               score=sc, per_task=per_task, lesions=les, refs=refs, role=role, role_held=role_held, comp=compo,
               ral=agent.core.ral_stats, say=agent.core.say_stats, val=agent.core.val_stats, stats={k: (int(v) if isinstance(v, (int, float)) else v)
                                                for k, v in agent.core.stats.items()},
               tstats=agent.tstats, mix=int(agent.core.mix.n), hip=int(agent.core.hip.n), replies=replies,
               secs=round(time.time() - t0, 1))
    if not outp:
        os.makedirs(os.path.join(HERE, "out", "runs"), exist_ok=True)
        outp = os.path.join(HERE, "out", "runs", f"{tag}_s{seed}.json")
    with open(outp, "a") as f:
        f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
    if save:
        agent.th_log = []
        with open(os.path.join(HERE, "out", f"agent_{tag}_s{seed}.pkl"), "wb") as f:
            pickle.dump(agent, f)
    log(f"score {sc} | role {role} | refs all {({k: refs.get('all', {}).get(k) for k in ('agent', 'echo', 'const_best', 'prior')})} "
        f"held {({k: refs.get('held', {}).get(k) for k in ('agent', 'echo', 'const_best', 'prior')})} | comp {compo}")
    log(f"done viol={viol} secs={rec['secs']}")


if __name__ == "__main__":
    main()
