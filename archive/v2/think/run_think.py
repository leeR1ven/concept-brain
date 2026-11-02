# -*- coding: utf-8 -*-
"""ARCH_v2-think on the shared bench (copy in bench_copy/): listening -> babbling -> stages B, C, D (recast tutor).
usage: python3 run_think.py seed=0 nA=300 nB=2000 nC=2000 nD=2000 stages=ABCD tag=x agent=think policy=free
        [intention keys: theta_I g0_reply g0_scene u_reply beta nogo L ...] [core cfg keys: floor think_k ior ...]
        [world keys: w_<param>=v ;  probe world: ss_p=0.3 ss_mode=plain|point ss_wait=30]"""
import os, sys, time, json, pickle, collections
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from agent import DevAgent, BENCH           # noqa: E402
from agent_think import ThinkAgent          # noqa: E402
sys.path.insert(0, BENCH)
from world import World                     # noqa: E402
from metrics import evaluate, headline      # noqa: E402
from core_think import TCfg                 # noqa: E402
import numpy as np                          # noqa: E402

A = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
def arg(k, d):
    v = A.pop(k, None)
    if v is None:
        return d
    if isinstance(d, bool):
        return v in ("1", "True", "true")
    return type(d)(v)
seed, nA, nB, nC, nD = arg("seed", 0), arg("nA", 300), arg("nB", 2000), arg("nC", 2000), arg("nD", 2000)
nbab, tag, tutor = arg("babbles", 35960), arg("tag", "run"), arg("tutor", "recast")
stages, save, kind = arg("stages", "ABCD"), arg("save", 0), arg("agent", "think")
ss_p, ss_mode, ss_wait = arg("ss_p", 0.0), arg("ss_mode", "plain"), arg("ss_wait", 30)
ss_stages = arg("ss_stages", "BCD")
akw, cfgkw, wkw = {}, {}, {}
AG_KEYS = dict(policy='free', theta_I=1.0, g0_reply=1.0, g0_scene=0.3, u_reply=0.05, u_scene=0.0, beta=2.0,
               g_max=3.0, nogo=0.25, nogo_late=0.05, L=10, eta_oth=0.2, eta_go=0.2, learn_turn=True, tau_w=7, urg_gate=True,
               close_on_scene=True, oth_kind=False, brake_now=False, life_eye=False, eye_feats=False, no_react=False, late_release=True, u_R1=1.0, own_tail=False)
for k, v in A.items():
    if k.startswith("w_"):
        wkw[k[2:]] = float(v) if "." in v else int(v)
    elif k in AG_KEYS:
        d = AG_KEYS[k]
        akw[k] = (v in ("1", "True", "true")) if isinstance(d, bool) else type(d)(v)
    else:
        d = getattr(TCfg(), k)
        cfgkw[k] = (v in ("1", "True", "true")) if isinstance(d, bool) else type(d)(v)
OUT = os.path.join(HERE, "out", f"think_{tag}")
os.makedirs(os.path.join(HERE, "out"), exist_ok=True)
logf = open(OUT + ".log", "w")
def log(s):
    print(s, flush=True); logf.write(s + "\n"); logf.flush()

if ss_p > 0:
    from world_probe import ProbeWorld as W
    def mk(st, prev):
        return W(st, seed=seed, tutor=tutor, prev=prev, ss_p=ss_p if st in ss_stages else 0.0, ss_mode=ss_mode,
                 ss_wait=ss_wait, **wkw)
else:
    def mk(st, prev):
        return World(st, seed=seed, tutor=tutor, prev=prev, **wkw)


def per_task(res, st):
    return json.dumps({k: (v.get('n'), round(v['acc'], 3) if v.get('acc') is not None else None,
                           round((v.get('held') or {}).get('acc') or 0, 3), round((v.get('seen') or {}).get('acc') or 0, 3))
                       for k, v in res[st].items() if isinstance(v, dict) and 'acc' in v}, ensure_ascii=False)


def timing(L, st):
    """latency per task (free replies that spoke), turn-taking outside free replies."""
    fr = [r for r in L if r['stage'] == st and r['kind'] in ('probe', 'recast')]
    out = {}
    for task in sorted({r['task'] for r in fr}) + ['ALL']:
        rs = [r for r in fr if (task == 'ALL' or r['task'] == task)]
        lat = [r['latency'] for r in rs if r.get('latency') is not None]
        lat_ok = [r['latency'] for r in rs if r.get('latency') is not None and r['reply'] in r['acceptable']]
        lat_bad = [r['latency'] for r in rs if r.get('latency') is not None and r['reply'] not in r['acceptable']]
        held = [r['latency'] for r in rs if r.get('latency') is not None and r['held_out']]
        out[task] = dict(n=len(rs), spoke=round(len(lat) / max(len(rs), 1), 3),
                         lat_med=float(np.median(lat)) if lat else None, lat_mean=round(float(np.mean(lat)), 2) if lat else None,
                         lat_ok=round(float(np.mean(lat_ok)), 2) if lat_ok else None,
                         lat_bad=round(float(np.mean(lat_bad)), 2) if lat_bad else None,
                         lat_held=round(float(np.mean(held)), 2) if held else None)
    rc = [r for r in L if r['stage'] == st and r['kind'] == 'recast']
    stm = [r for r in L if r['stage'] == st and r['kind'] == 'statement']
    ex = [r for r in L if r['stage'] == st]
    out['_turns'] = dict(after_recast_spoke=round(np.mean([bool(r.get('after_recast_reply')) for r in rc]), 3) if rc else None,
                         after_statement_spoke=round(np.mean([bool(r['reply']) for r in stm]), 3) if stm else None,
                         overlap_ticks=int(sum(r.get('mouth_while_partner', 0) for r in ex)),
                         exchanges_with_overlap=int(sum(r.get('mouth_while_partner', 0) > 0 for r in ex)),
                         extra_turns=int(sum(r.get('extra_turns', 0) for r in fr)),
                         max_reply_len=max([len(r['reply']) for r in fr] + [0]),
                         cut=int(sum(r.get('stop') == 'cut' for r in fr)))
    return out


def curve(L, st, nb=8):
    fr = [r for r in L if r['stage'] == st and r['kind'] in ('probe', 'recast') and not r['held_out']]
    if not fr:
        return []
    d0, d1 = fr[0]['dlg'], fr[-1]['dlg'] + 1
    edges = np.linspace(d0, d1, nb + 1)
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        rs = [r for r in fr if a <= r['dlg'] < b]
        if rs:
            lat = [r['latency'] for r in rs if r.get('latency') is not None]
            out.append((round(float(np.mean([r['reply'] in r['acceptable'] for r in rs])), 3),
                        round(float(np.mean(lat)), 1) if lat else None))
    return out


t0 = time.time()
res_all, tim_all, cur_all = {}, {}, {}
prevw = None
all_logs = []
agent = None
for st in stages:
    n = {'A': nA, 'B': nB, 'C': nC, 'D': nD}[st]
    w = mk(st, prevw)
    if agent is None:
        if kind == 'dev':
            agent = DevAgent(w, seed=seed, **cfgkw)
        else:
            agent = ThinkAgent(w, seed=seed, **akw, **cfgkw)
        log(f"seed {seed} tutor {tutor} stages {stages} agent {kind} {akw} cfg {cfgkw} world {wkw} "
            f"ss {ss_p},{ss_mode},{ss_wait}  FEAT0 {agent.NF}")
    w.run(agent, n)
    if st == 'A':
        log(f"[A] listening {n} A-dialogues: categories {agent.Z.n}, items {agent.stats['items']}, lines {agent.stats['lines']} ({time.time()-t0:.0f}s)")
        hist = agent.babble(w.voice, nbab, log_every=max(1, nbab // 8))
        log(f"[A] babbling {nbab} muscle sets: programs bound {len(agent.cat2prog)} of {agent.Z.n} categories")
    else:
        res = evaluate(w.log); res_all[st] = res
        log(f"[{st}] " + headline(res, st))
        log(f"[{st}] per task (n, acc, held, seen): " + per_task(res, st))
        tim = timing(w.log, st); tim_all[st] = tim
        log(f"[{st}] timing: " + json.dumps(tim, ensure_ascii=False))
        cu = curve(w.log, st); cur_all[st] = cu
        log(f"[{st}] learning curve (acc, latency) per eighth: {cu}")
        lr = res[st].get('learning') or {}
        log(f"[{st}] learning in use: early {lr.get('acc_early')} late {lr.get('acc_late')}  ({time.time()-t0:.0f}s)")
        if st == 'D':
            log(f"[D] one-shot: {json.dumps(res[st].get('one_shot'), ensure_ascii=False)[:300]}")
        ss = [r for r in w.log if r.get('kind') == 'self_init']
        if ss:
            from world_probe import ss_summary
            log(f"[{st}] self-initiated probes: {json.dumps(ss_summary(ss), ensure_ascii=False)}")
    if hasattr(agent, 'tstats'):
        log(f"[{st}] thought stats: {agent.tstats} core {agent.core.think_stats} body blocked {agent.body.go_blocked} "
            f"first_blocked {agent.body.first_blocked} | P_oth {len(agent.P_oth.w)} keys, V_go {len(agent.V_go.w)} keys, rbar {agent.rbar:.2f}")
    all_logs += w.log
    prevw = w
log(f"done {time.time()-t0:.0f}s; stats {agent.stats}; mix {agent.core.mix.n}; hip {agent.core.hip.n}")
json.dump(dict(res=res_all, timing=tim_all, curve=cur_all), open(OUT + ".json", "w"), ensure_ascii=False, default=str)
pickle.dump(all_logs, open(OUT + "_log.pkl", "wb"))
if hasattr(agent, 'th_log'):
    pickle.dump(agent.th_log, open(OUT + "_thoughts.pkl", "wb"))
if save:
    agent.th_log = []
    pickle.dump(agent, open(OUT + "_agent.pkl", "wb"))
