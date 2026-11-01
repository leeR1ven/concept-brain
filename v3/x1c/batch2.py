import sys, os, time, traceback
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run
t0 = time.time()
for arm, nf in (('B_loop', 3000), ('B_linear', 1000)):
    for seed in range(5):
        try:
            run.main(arm, seed, n_teach=600, n_corr=300, n_free=nf, extra={})
        except Exception:
            traceback.print_exc()
        print(f'## done {arm} s{seed} {time.time() - t0:.0f}s', flush=True)
print('## batch2 finished', flush=True)
