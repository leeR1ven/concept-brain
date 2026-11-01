"""one process: all arms x 5 seeds, sequentially (box rule: one local process under 600 MB)."""
import sys, os, time, traceback
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run
order = ['B', 'C', 'C2', 'A', 'B_nlms', 'B_nopart', 'B_linear', 'B_fast', 'B_hip', 'B_dist', 'B_real']
full = {'B', 'B_real'}
t0 = time.time()
for arm in order:
    for seed in range(5):
        try:
            run.main(arm, seed, n_teach=600, n_corr=300, n_free=3000 if arm in full else 1000, extra={})
        except Exception:
            traceback.print_exc()
        print(f'## done {arm} s{seed} {time.time() - t0:.0f}s', flush=True)
