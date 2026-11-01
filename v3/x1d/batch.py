"""run phase-2 lives one after another in ONE process:  python3 -I batch.py arm1,arm2 seeds(e.g. 0-4) [key=value ...]"""
import sys, os, time, traceback
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'x1c'))   # repo: core3.py, world.py, run.py from ../x1c
import run2
from run import parse
arms = sys.argv[1].split(',')
a, b = (sys.argv[2].split('-') + [None])[:2]
seeds = range(int(a), int(b or a) + 1)
kv = {k: parse(v) for k, v in (x.split('=', 1) for x in sys.argv[3:])}
nb, nt, nc, nf = kv.pop('n_birth', 200), kv.pop('n_teach', 600), kv.pop('n_corr', 300), kv.pop('n_free', 1000)
t0 = time.time()
for arm in arms:
    for s in seeds:
        if os.path.exists(os.path.join(run2.HERE, 'out2', f"{arm}{kv.get('tag', '')}_s{s}.json")):
            print('## skip', arm, s, flush=True); continue
        try:
            run2.main(arm, s, nb, nt, nc, nf, dict(kv))
        except Exception:
            traceback.print_exc()
        print(f'## done {arm} s{s} {time.time() - t0:.0f}s', flush=True)
print('## batch finished', flush=True)
