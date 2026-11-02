"""core8 equivalence: (1) defaults == core6 bit-exactly; (2) lat_fly (and its lazy decay / label patch mean) vs core6's
materialised P33 on the same inputs: identical while L33 does not learn (up to float rounding); with L33 learning the
refresh timing differs, so report first divergence and activity statistics."""
import sys, numpy as np, time
sys.path.insert(0, '.')
for _x in ('x1f', 'x1d', 'x1c'):   # repo: core5/core6/common6/world6 from ../x1f, core4/phase1 from ../x1d, core3/world/run from ../x1c
    _p = __import__('os').path.join(__import__('os').path.dirname(__import__('os').path.abspath(__file__)), '..', _x)
    if _p not in __import__('sys').path:
        __import__('sys').path.append(_p)
import common6
from common6 import GEN0, run_until
from world import Physics, World
from world6 import OverhearWorld
from run import birth_rates
import phase1, core6, core8

def born(cls, seed, bk, n_birth=60):
    ph = Physics(); w0 = World(ph, seed)
    nol = dict(bk); keep = {k: bk.get(k) for k in phase1.NOLEARN}; nol.update(phase1.NOLEARN)
    b = cls(ph, seed, rates=birth_rates(ph, w0, n=1000), **nol)
    run_until(b, World(ph, seed + 50), n_birth)
    for k in phase1.NOLEARN:
        b.c[k] = keep[k] if keep[k] is not None else core6.DEF[k]
    return ph, b

def trace(ph, b, n_dlg, seed):
    ow = OverhearWorld(ph, seed)
    L3, P = [], []
    while ow.n_ex < n_dlg:
        mus = b.step(ow.sound, 0.0); ow.step(mus)
        L3.append(b.l3.copy()); P.append(b.pred.copy())
    return np.array(L3), np.array(P)

for name, extra in (('defaults', {}), ('fly_noL33', dict(lat_fly=True, eta33=0.0)), ('fly_L33', dict(lat_fly=True))):
    bk = dict(GEN0, **{k: v for k, v in extra.items() if k != 'lat_fly'})
    t0 = time.time()
    ph, b6 = born(core6.Brain, 0, bk)
    A3, AP = trace(ph, b6, 150, 0)
    t1 = time.time()
    ph, b8 = born(core8.Brain, 0, dict(bk, **extra))
    B3, BP = trace(ph, b8, 150, 0)
    t2 = time.time()
    n = min(len(A3), len(B3))
    diff = (A3[:n] != B3[:n]).any(1)
    first = int(np.argmax(diff)) if diff.any() else None
    print(f'{name}: ticks {n} | first differing tick {first} | ticks differing {diff.mean():.3f} | '
          f'L3 mean core6 {A3.sum(1).mean():.1f} core8 {B3.sum(1).mean():.1f} | pred {AP.sum(1).mean():.1f} {BP.sum(1).mean():.1f} | '
          f's/tick core6 {1000*(t1-t0)/len(A3):.2f}ms core8 {1000*(t2-t1)/len(B3):.2f}ms (incl. birth)', flush=True)
    if 'L33' in name and name != 'fly_noL33':
        L6 = b6.L33; L8 = b8.L33_true()
        print('   L33 pos frac', (L6 > 1e-3).mean().round(4), (L8 > 1e-3).mean().round(4), ' max', L6.max().round(3), L8.max().round(3))
