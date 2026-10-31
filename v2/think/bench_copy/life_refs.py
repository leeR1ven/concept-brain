# reference runs in life: RefV1 (force: the only mode in which v1 learns; model: silent), RandomNamer, oracle, echo,
# lookup, silent.  usage: python3 life_refs.py ticks [names...]   -> results_life/<name>_s0.json
import json, os, sys, time
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_v] = '1'
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from life import Life, evaluate_life, headline_life, RandomNamer, life_baseline
from baselines import Silent
from ref_v1 import RefV1
TICKS = int(sys.argv[1]) if len(sys.argv) > 1 else 1000000
NAMES = sys.argv[2:] or ['silent', 'namer', 'oracle', 'echo', 'lookup', 'refv1_force', 'refv1_model']
os.makedirs('results_life', exist_ok=True)
for nm in NAMES:
    tutor = 'force' if nm == 'refv1_force' else 'model'
    L = Life(seed=0, tutor=tutor)
    agent = {'silent': Silent, 'namer': lambda: RandomNamer(200), 'refv1_force': lambda: RefV1(L, cfg='mem', seed=0),
             'refv1_model': lambda: RefV1(L, cfg='mem', seed=0)}.get(nm, lambda: life_baseline(nm))()
    t0 = time.time()
    ticks = TICKS if not nm.startswith('refv1_model') else min(TICKS, 200000)
    log = L.run(agent, ticks=ticks)
    res = evaluate_life(log)
    res['secs'] = round(time.time() - t0, 1); res['us_per_tick'] = round((time.time() - t0) / L.t * 1e6, 2)
    if nm.startswith('refv1'):
        res['programs'] = int(agent.body.n_prog)
        res['examples'] = [(r['task'], r['prompt'], r['reply'], r['acceptable'][0]) for r in log
                           if r.get('type') == 'ex' and r['kind'] in ('probe', 'social')][-25:]
    json.dump(res, open(f'results_life/{nm}_s0.json', 'w', encoding='utf-8'), ensure_ascii=False, default=str)
    print(f'{nm} ({tutor}): {L.t} ticks {res["secs"]} s ({res["us_per_tick"]} us/tick)\n    ' + headline_life(res), flush=True)
print('LIFE REFS DONE', flush=True)
