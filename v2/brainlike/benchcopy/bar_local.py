# local subset of bar.py: the headline bar (RefV1 mem, force, B/C/D, seed 0) + baselines for seed 0; same job() as bar.py
import os, sys, time
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_v] = '1'
sys.argv = ['bar.py', '2000', '0', '1', 'results_local']
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bar
specs = []
for stage in ('B', 'C', 'D'):
    specs.append((f'refv1_mem_{stage}_force_s0', 'refv1_mem', stage, 0, 'force', 2000))
for stage in ('B', 'C', 'D'):
    for tutor in ('force', 'model'):
        for b in ('echo', 'last', 'prior', 'lookup', 'ngram', 'oracle'):
            specs.append((f'{b}_{stage}_{tutor}_s0', b, stage, 0, tutor, 2000))
done = {f[:-5] for f in os.listdir('results_local') if f.endswith('.json')}
t0 = time.time()
for s in specs:
    if s[0] in done:
        continue
    name, secs, hl = bar.job(s)
    print(f'[{time.time() - t0:6.0f}s] {name} ({secs}s)\n    ' + hl.replace('\n', '\n    '), flush=True)
print('LOCAL DONE', flush=True)
