import sys, time
X = __import__('os').path.join(__import__('os').path.dirname(__import__('os').path.abspath(__file__)), '..')
sys.path.insert(0, X)
import numpy as np
import core8, common6
common6.Brain, common6.DEF = core8.Brain, core8.DEF
from common6 import make_brain, question, SID, render, C1
from run8 import build_cfg
from wire8 import wire_speech
N = int(sys.argv[1]); nb = int(sys.argv[2]); extra = dict(a.split('=') for a in sys.argv[3:])
kv = {k: (float(v) if '.' in v else (int(v) if v.lstrip('-').isdigit() else v)) for k, v in extra.items()}
bk = build_cfg(dict(N=N, **kv))
t0 = time.time()
ph, b = make_brain(0, bk, n_birth=nb)
wire_speech(b, ph)
print('build', round(time.time() - t0, 1), 's  n1', b.n1, 'na1', b.na1, 'content lines', int(b.l1_content.sum()),
      'muscle', b.nm, 'stop', b.nbk, 'n2', b.c['n2'], 'n3', b.c['n3'], 'A0', 2 * ph.n_a, 'patches3', b.PM3.shape[1], 'patches1', b.PM1.shape[1])
print('M31 frac', round(float(b.M31.mean()), 3), 'B31 sum', float(b.B31.sum()), 'in3 median', round(float(np.median(b.in3)), 3),
      'thr3 median', round(float(np.median(b.c['f3'] * b.in3)), 3), 'h3 range', round(float(b.h3.min()), 3), round(float(b.h3.max()), 3),
      'thr2', round(float(np.median(b.c['f2'] * b.in2)), 3), 'h2 range', round(float(b.h2.min()), 3), round(float(b.h2.max()), 3))
rng = np.random.default_rng(5)
f, x = 0, 1
sched = [None] * 6 + [SID[s] for s in question(f, x) for _ in range(3)] + [None] * 8
snap = b.snapshot()
for t, cg in enumerate(sched):
    ext_prev = b.ext_prev.copy() if hasattr(b, 'ext_prev') else None
    b.step(render(ph, [('s', cg, C1, 1.0)] if cg is not None else [], rng), 0.0, learn=False)
    b.mus[:] = False
    ext = b.l1 & ~b.pred
    x3 = b._x3
    thr = b.c['f3'] * b.in3 + b.h3
    print(t - (len(sched) - 9), 'in' if cg is not None else '--', 'L1ext', int(ext.sum()), 'pred', int(b.pred.sum()), 'L2', int(b.l2.sum()),
          'L3', int(b.l3.sum()), 'x3max', round(float(x3.max()), 3), 'thr3min', round(float(thr.min()), 3), 'max(x3-thr)', round(float((x3 - thr).max()), 3))
b.restore(snap)
