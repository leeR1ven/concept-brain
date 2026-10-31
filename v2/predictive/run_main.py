# Main pipeline on the bench: A (hear; no programs yet) -> private babbling (no tutor) -> B -> C [-> D -> recast/E].
import sys, json, time, pickle, argparse
import numpy as np
import os; sys.path.insert(0, os.environ.get('PRED_HERE', __import__('os').path.dirname(__import__('os').path.abspath(__file__))))
from run_lib import DiagWorld, new_agent, z_report, lexicon_chars, Timer
from metrics import evaluate, headline
from agent import Cfg
ap = argparse.ArgumentParser()
ap.add_argument('--seed', type=int, default=0); ap.add_argument('--A', type=int, default=300)
ap.add_argument('--babble', type=int, default=170000)
ap.add_argument('--plan', default='B:1500,C:1500'); ap.add_argument('--tutor', default='model')
ap.add_argument('--out', default='main_s0'); ap.add_argument('--save', type=int, default=0)
ap.add_argument('--sleep_every', type=int, default=0); ap.add_argument('--sleep_n', type=int, default=2000)
ap.add_argument('--set', default='')          # cfg overrides  key=val,key=val ; use.X=0 for ablations
args = ap.parse_args()
cfg = Cfg(seed=args.seed)
for kv in [x for x in args.set.split(',') if x]:
    k, v = kv.split('=')
    if k.startswith('use.'):
        cfg.use[k[4:]] = bool(int(v))
    else:
        cur = getattr(cfg, k); setattr(cfg, k, type(cur)(v) if not isinstance(cur, bool) else bool(int(v)))
T = Timer()
w = DiagWorld('A', seed=args.seed, tutor='model')
a = new_agent(w, cfg); a.log_events = []
w.run(a, n_dialogues=args.A)
print('A listen', z_report(a, w, lexicon_chars()), T(), flush=True)
w.babble(a, args.babble, bout=(4, 8))
print('babble', z_report(a, w, lexicon_chars()), {k: v for k, v in a.stats.items()}, T(), flush=True)
a.log_events = None
out = {}
for item in args.plan.split(','):
    st, n = item.split(':'); n = int(n)
    w = DiagWorld(st, seed=args.seed, tutor=args.tutor, prev=w)
    if args.sleep_every > 0:
        done = 0
        while done < n:
            m = min(args.sleep_every, n - done)
            w.run(a, n_dialogues=m); done += m
            a.sleep(args.sleep_n)
        log = w.log
    else:
        log = w.run(a, n_dialogues=n)
    res = evaluate(log); out[st] = res
    print(headline(res, stage=st), T(), flush=True)
    for task, r in res[st].items():
        if isinstance(r, dict) and r.get('n') and 'held' in r:
            print(f'   {task:8s} n {r["n"]:4d} acc {r["acc"]:.2f} first {r["first"]:.2f} held {r["held"]["acc"]} seen {r["seen"]["acc"]}', flush=True)
    print('   after_model', res[st]['after_model'], '\n   learning', res[st]['learning'], '\n   one_shot', res[st]['one_shot'], flush=True)
    pl = a.pred_log[-3000:]
    print('   units', a.nz, 'programs', sum(p is not None for p in a.z_prog), a.stats, 'next-char prediction acc (last 3000 heard chars)', round(float(np.mean(pl)), 3) if pl else None, 'episodes stored', len(a.episodes), flush=True)
    json.dump(out, open(f'{args.out}.json', 'w'), default=str)
if args.save:
    w.truth = {}                                    # diagnostics only; keep the pickle small
    pickle.dump((a, w), open(f'{args.out}_agent.pkl', 'wb'))
