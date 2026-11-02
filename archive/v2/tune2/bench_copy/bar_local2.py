# local follow-up of bar.py: v1 without a teacher (model, recast), stage E, base config on the new D; seed 0
import os, sys, time
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_v] = '1'
sys.argv = ['bar.py', '2000', '0', '1', 'results_all']
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bar
specs = [('refv1_mem_B_model_s0', 'refv1_mem', 'B', 0, 'model', 2000),
         ('refv1_mem_C_recast_s0', 'refv1_mem', 'C', 0, 'recast', 2000),
         ('refv1_mem_E_force_s0', 'refv1_mem', 'E', 0, 'force', 2000)] + \
        [(f'{b}_E_force_s0', b, 'E', 0, 'force', 2000) for b in ('echo', 'last', 'prior', 'lookup', 'ngram', 'oracle')] + \
        [('refv1_base_D_force_s0', 'refv1_base', 'D', 0, 'force', 2000)]
os.makedirs('results_all', exist_ok=True)
done = {f[:-5] for f in os.listdir('results_all') if f.endswith('.json')}
t0 = time.time()
for s in specs:
    if s[0] in done:
        continue
    name, secs, hl = bar.job(s)
    print(f'[{time.time() - t0:6.0f}s] {name} ({secs}s)\n    ' + hl.replace('\n', '\n    '), flush=True)
print('LOCAL2 DONE', flush=True)
