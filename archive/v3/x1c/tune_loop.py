# seed-0 tuning of the single loop gain for arm B_loop: teach N exchanges, report thought motion in the empty room
import sys, os, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from world import Physics, World
from run import make_brain, run_until, thought_motion, probe, summarize_probe, parse
kv = {k: parse(v) for k, v in (a.split('=', 1) for a in sys.argv[1:])}
n = kv.pop('n', 300)
ph = Physics(); w = World(ph, 0, empty_every=30)
b = make_brain(ph, w, 0, dict(act='global', kappa=0.0, w33=0.4, pred_rel=False, **kv))
rec = dict(empty_ticks=0, pas=[]); t0 = time.time()
run_until(b, w, n, rec)
mo = thought_motion(rec['pas'][-1500:])
c3 = np.array(b.cnt['n3']); c1 = np.array(b.cnt['n1'])
print(kv, 'motion(empty room)', mo, 'n3 mean %.1f min %d max %d p0 %.2f' % (c3.mean(), c3.min(), c3.max(), (c3 == 0).mean()),
      'n1 mean %.1f' % c1.mean(), 'acc', summarize_probe(probe(b, ph, w, [(f, x) for f in range(4) for x in range(8)]))['seen'], '%.0fs' % (time.time() - t0), flush=True)
