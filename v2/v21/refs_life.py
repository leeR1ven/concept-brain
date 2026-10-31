# -*- coding: utf-8 -*-
"""Same-life trivial references (bench life_refs.py agents) on the life that ARCH_v2/v2.1 live (seed 5, tutor recast,
eye_layout retino): silent, random namer (privileged: told the scene), echo, oracle; + v2.1 life metrics for each.
usage: python3 refs_life.py [seed=5] [ticks=1000000] -> out/life/refs_s<seed>.json"""
import json, os, sys, time
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_v] = '1'
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'bench_copy')); sys.path.insert(0, HERE)
sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'synth'))  # repo: unchanged ARCH_v2 core + bench_copy
from life import Life, evaluate_life, headline_life, RandomNamer, life_baseline   # noqa: E402
from baselines import Silent                                                     # noqa: E402
from life_metrics import metrics                                                 # noqa: E402
A = dict(a.split('=', 1) for a in sys.argv[1:])
seed, ticks = int(A.get('seed', 5)), int(float(A.get('ticks', 1e6)))
out = {}
for nm in ('silent', 'namer', 'echo', 'oracle'):
    L = Life(seed=seed, tutor='recast', eye_layout='retino')
    agent = {'silent': Silent, 'namer': lambda: RandomNamer(200)}.get(nm, lambda: life_baseline(nm))()
    t0 = time.time()
    log = L.run(agent, ticks=ticks)
    res = evaluate_life(log)
    out[nm] = dict(headline=headline_life(res), v21=metrics(log), initiative=res['initiative'],
                   groups={k: dict(acc=v.get('acc'), n=v.get('n')) for k, v in res['groups'].items()})
    print(nm, f'{time.time() - t0:.0f}s', headline_life(res), flush=True)
os.makedirs(os.path.join(HERE, 'out', 'life'), exist_ok=True)
json.dump(out, open(os.path.join(HERE, 'out', 'life', f'refs_s{seed}.json'), 'w'), ensure_ascii=False, default=str)
