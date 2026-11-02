# The v1 reference agent ("the bar", bench ref_v1.RefV1 cfg='mem', teacher forcing) through a curriculum B -> C -> D.
import sys, os, json, time, argparse
sys.path.insert(0, os.environ.get('PRED_BENCH', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'bench_snapshot')))
from world import World
from metrics import evaluate, headline
from ref_v1 import RefV1
ap = argparse.ArgumentParser(); ap.add_argument('--seed', type=int, default=0); ap.add_argument('--plan', default='B:2500,C:2500,D:2500')
ap.add_argument('--tutor', default='force'); ap.add_argument('--out', default='ref_s0'); args = ap.parse_args()
t0 = time.time(); w = None; agent = None; out = {}
for item in args.plan.split(','):
    st, n = item.split(':'); n = int(n)
    w = World(st, seed=args.seed, tutor=args.tutor, prev=w)
    if agent is None:
        agent = RefV1(w, cfg='mem', seed=args.seed)
    log = w.run(agent, n)
    res = evaluate(log); out[st] = {t: dict(n=r['n'], acc=r['acc'], held=r['held']['acc'], seen=r['seen']['acc'])
                                    for t, r in res[st].items() if isinstance(r, dict) and r.get('n') and 'held' in r}
    out[st]['one_shot'] = res[st]['one_shot']; out[st]['guided'] = res[st]['guided']
    print(headline(res, stage=st), f'{time.time()-t0:.0f}s', flush=True)
    for t, r in out[st].items():
        if isinstance(r, dict) and 'held' in r: print(f"   {t:8s} n {r['n']:4d} acc {r['acc']} held {r['held']} seen {r['seen']}", flush=True)
    json.dump(out, open(args.out + '.json', 'w'), default=str, indent=1)
