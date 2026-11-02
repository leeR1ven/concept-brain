"""where does the birth return loop (birth31=recip) fail to hold the thought in the pause? newborn, learning off."""
import sys, numpy as np
sys.path.insert(0, '.')
for _x in ('x1f', 'x1d', 'x1c'):   # repo: core5/core6/common6/world6 from ../x1f, core4/phase1 from ../x1d, core3/world/run from ../x1c
    _p = __import__('os').path.join(__import__('os').path.dirname(__import__('os').path.abspath(__file__)), '..', _x)
    if _p not in __import__('sys').path:
        __import__('sys').path.append(_p)
import core8, common6
common6.Brain, common6.DEF = core8.Brain, core8.DEF
from common6 import make_brain, GEN0, question, SID, render, C1
from run8 import build_cfg
from run import parse
kv = {k: parse(v) for k, v in (a.split('=', 1) for a in sys.argv[1:])}
bk = build_cfg(dict(kv))
ph, b = make_brain(0, bk)
from wire8 import wire_speech
wire_speech(b, ph)
rng = np.random.default_rng(1)
sched = [None] * 4 + [SID[s] for s in question(0, 1) for _ in range(3)] + [None] * 8
print('B31 total', round(float(b.B31.sum()), 1), 'B31 rows>0', int((b.B31.sum(1) > 0).sum()), 'of', b.c['n3'])
for t, cg in enumerate(sched):
    b.step(render(ph, [('s', cg, C1, 1.0)] if cg is not None else [], rng), 0.0, learn=False)
    b.mus[:] = False
    x1 = b._x1
    xn = x1 / (b.c['eps1'] + b._tot1)
    print(f't{t:2d} in={cg if cg is None else ph.content.shape and cg!s:>4} L1ext={int(b.ext_prev.sum()):3d} pred={int(b.pred.sum()):3d} L2={int(b.l2.sum()):3d} L3={int(b.l3.sum()):3d} '
          f'tot1={b._tot1:6.2f} max x1={x1.max():.3f} max share={xn.max():.4f} n(share>=th1)={int((xn >= b.c["th1"]).sum())} n(x>=rho*max)={int((x1 >= b.c["rho1"] * x1.max()).sum())}')
