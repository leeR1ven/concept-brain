# -*- coding: utf-8 -*-
"""Train one agent through a curriculum, then a causal audit: fork one process per lesion; each runs the SAME next
content (new World with the same stage/seed/tutor and prev) with one learned part silenced (learning stays on).
Usage: python3 train_audit.py --seed 0 --A 600 --plan B:2500,C:2500,D:2500 --audit B:300,C:300,D:300 --out s0"""
import argparse
import copy
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.environ.get('PRED_HERE', os.path.dirname(os.path.abspath(__file__))))
from run_lib import DiagWorld, new_agent, z_report, lexicon_chars, Timer  # noqa: E402
from metrics import evaluate, headline  # noqa: E402
from agent import Cfg  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--seed', type=int, default=0); ap.add_argument('--A', type=int, default=600)
ap.add_argument('--babble', type=int, default=170000)
ap.add_argument('--plan', default='B:2500,C:2500,D:2500'); ap.add_argument('--tutor', default='model')
ap.add_argument('--audit', default='B:300,C:300,D:300')
ap.add_argument('--lesions', default='none,x1+x2,start,gline,ghold,gtopic,B,V,V2,C2,H,Brec,gate,S,go,fwd')
ap.add_argument('--sleep_every', type=int, default=0); ap.add_argument('--sleep_n', type=int, default=2000)
ap.add_argument('--set', default=''); ap.add_argument('--out', default='s0'); ap.add_argument('--procs', type=int, default=12)
args = ap.parse_args()
cfg = Cfg(seed=args.seed)
for kv in [x for x in args.set.split(',') if x]:
    k, v = kv.split('=')
    if k.startswith('use.'):
        cfg.use[k[4:]] = bool(int(v))
    else:
        cur = getattr(cfg, k); setattr(cfg, k, type(cur)(v) if not isinstance(cur, bool) else bool(int(v)))
T = Timer()
summary = {'args': vars(args)}


def run_stage(a, w_prev, st, n, sleep_every=0):
    w = DiagWorld(st, seed=args.seed, tutor=args.tutor, prev=w_prev)
    if sleep_every > 0:
        done = 0
        while done < n:
            m = min(sleep_every, n - done); w.run(a, n_dialogues=m); done += m; a.sleep(args.sleep_n)
        return w, w.log
    return w, w.run(a, n_dialogues=n)


def brief(res, st):
    out = {}
    for task, r in res[st].items():
        if isinstance(r, dict) and r.get('n') and 'held' in r:
            out[task] = dict(n=r['n'], acc=r['acc'], first=r['first'], held=r['held']['acc'], seen=r['seen']['acc'],
                             echo_free=r['echo_free'], parrot=r['parrot'])
    out['mech'] = {k: res[st]['mech'][k] for k in ('mouth_while_partner', 'latency_median', 'not_3_ticks', 'syllables',
                                                    'garble', 'extra_turns', 'stop')}
    out['after_model'] = res[st]['after_model']; out['learning'] = res[st]['learning']; out['one_shot'] = res[st]['one_shot']
    return out


w = DiagWorld('A', seed=args.seed, tutor='model')
a = new_agent(w, cfg); a.log_events = []
w.run(a, n_dialogues=args.A)
zr = z_report(a, w, lexicon_chars()); print('A', zr, T(), flush=True); summary['A'] = zr
w.babble(a, args.babble, bout=(4, 8))
zr = z_report(a, w, lexicon_chars()); print('babble', zr, T(), flush=True); summary['babble'] = zr
a.log_events = None
for item in args.plan.split(','):
    st, n = item.split(':'); n = int(n)
    pl0 = len(a.pred_log)
    w, log = run_stage(a, w, st, n, args.sleep_every)
    res = evaluate(log)
    print(headline(res, stage=st), T(), flush=True)
    summary[st] = brief(res, st)
    summary[st]['pred_acc'] = float(np.mean(a.pred_log[pl0:])) if len(a.pred_log) > pl0 else None
    for task, r in summary[st].items():
        if isinstance(r, dict) and 'acc' in r:
            print(f"   {task:8s} n {r['n']:4d} acc {r['acc']:.3f} held {r['held']} seen {r['seen']}", flush=True)
    print('   one_shot', summary[st]['one_shot'], ' pred_acc', summary[st]['pred_acc'], flush=True)
    json.dump(summary, open(f'{args.out}.json', 'w'), default=str, indent=1)
w.truth = {}


def audit_one(job):
    st, n, L = job
    b = copy.deepcopy(a)
    b.lesion = set() if L == 'none' else set(L.split('+'))
    b.predict()
    w2 = DiagWorld(st, seed=args.seed, tutor=args.tutor, prev=w)
    log = w2.run(b, n_dialogues=n)
    r = evaluate(log)[st]
    row = {t: (v['n'], v['acc'], v['held']['acc'], v['seen']['acc']) for t, v in r.items()
           if isinstance(v, dict) and v.get('n') and 'held' in v}
    return st, L, row


if args.audit:
    jobs = [(it.split(':')[0], int(it.split(':')[1]), L) for it in args.audit.split(',') for L in args.lesions.split(',')]
    ctx = mp.get_context('fork')
    with ctx.Pool(min(args.procs, len(jobs))) as pool:
        out = {}
        for st, L, row in pool.imap_unordered(audit_one, jobs):
            out.setdefault(st, {})[L] = row
            f = lambda x: '-' if x is None else f'{x:.2f}'  # noqa: E731
            print(f"AUDIT {st} lesion {L:10s} ALL {f(row['ALL'][1])} (held {f(row['ALL'][2])} seen {f(row['ALL'][3])}) | " +
                  ' '.join(f"{t} {f(v[1])}" for t, v in sorted(row.items()) if t not in ('ALL', 'SCENE', 'NOSCENE')), flush=True)
    summary['audit'] = out
json.dump(summary, open(f'{args.out}.json', 'w'), default=str, indent=1)
print('DONE', T(), flush=True)
