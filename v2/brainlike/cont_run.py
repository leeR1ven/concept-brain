# continue from a saved stage-A agent: python3 -B cont_run.py AGENT.pkl WORLD.pkl STAGES N_PER_STAGE [key=value]
import sys, time, pickle, numpy as np
sys.argv, argv = ['x'], sys.argv
import bench_run as br
from metrics import evaluate, headline
a = pickle.load(open(argv[1], 'rb')); w = pickle.load(open(argv[2], 'rb'))
for k, v in dict(A_hold=-1, rel_hold=0.0, rel=0.0, last_feats=None, t_mouth=-100, line_seq=[], rec_seq=[], rec_feats=None).items():
    if not hasattr(a, k): setattr(a, k, v)
from agent import DEFAULT
for k, v in DEFAULT.items(): a.c.setdefault(k, v)
stages, N = argv[3], int(argv[4])
for kv in argv[5:]:
    k, v = kv.split('='); a.c[k] = type(a.c[k])(v) if not isinstance(a.c[k], bool) else v in ('1', 'True')
a.c['babble'] = False
t0 = time.time(); prev = w
for st in stages:
    w = br.mk(st, 0, prev=prev)
    for k in range(4):
        w.run(a, N // 4)
        res = evaluate(w.log)
        print(f"{st} {(k+1)*N//4} {time.time()-t0:.0f}s :: {br.brief(res, st)}", flush=True)
    ex = [f"{r['prompt']}->{r['reply']}" for r in w.log if r['kind'] == 'probe'][-16:]
    print('   ', ' '.join(ex), flush=True)
    m = res[st].get('mech', {}); print('    mech', {k: m.get(k) for k in ('mouth_while_partner', 'latency_median', 'frac_not_3', 'garble', 'spoke', 'stop', 'agent_stop')}, flush=True)
    prev = w
print(a.stats)
if len(argv) > 5 and False:
    pass
pickle.dump(a, open(argv[1].replace('.pkl', f'_{stages}.pkl'), 'wb')); pickle.dump(prev, open(argv[2].replace('.pkl', f'_{stages}.pkl'), 'wb'))
