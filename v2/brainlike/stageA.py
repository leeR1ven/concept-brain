# python3 -B stageA.py SEED NA OUT_PREFIX [key=value]   -> OUT_PREFIX_agent.pkl, OUT_PREFIX_world.pkl
import sys, time, pickle
argv = sys.argv; sys.argv = ['x']
import bench_run as br
from agent import Agent, DEFAULT
seed, NA, out = int(argv[1]), int(argv[2]), argv[3]
cfg = {'seed': seed, 'p_go': 0.5, 'p_new': 0.8}
for kv in argv[4:]:
    k, v = kv.split('='); cfg[k] = (v in ('1', 'True')) if isinstance(DEFAULT[k], bool) else type(DEFAULT[k])(v)
eye = cfg.pop('eye', None) if 'eye' in cfg else None
a = Agent(cfg); w = br.mk('A', seed)
for win in w.warmup_windows(): a.warm(win)
a.end_warmup(); t0 = time.time()
w.run(a, NA)
print('A done', round(time.time() - t0), a.stats, flush=True)
pickle.dump(a, open(out + '_agent.pkl', 'wb')); pickle.dump(w, open(out + '_world.pkl', 'wb'))
