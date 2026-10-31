# local: the bar for seeds 1 and 2 (RefV1 mem, force, B/C/D); results into results_all/
import os, sys, time
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_v] = '1'
sys.argv = ['bar.py', '2000', '1,2', '1', 'results_all']
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bar
specs = [(f'refv1_mem_{st}_force_s{sd}', 'refv1_mem', st, sd, 'force', 2000) for sd in (1, 2) for st in 'BCD']
os.makedirs('results_all', exist_ok=True)
done = {f[:-5] for f in os.listdir('results_all') if f.endswith('.json')}
t0 = time.time()
for s in specs:
    if s[0] in done:
        continue
    name, secs, hl = bar.job(s)
    print(f'[{time.time() - t0:6.0f}s] {name} ({secs}s)\n    ' + hl.replace('\n', '\n    '), flush=True)
print('LOCAL3 DONE', flush=True)
