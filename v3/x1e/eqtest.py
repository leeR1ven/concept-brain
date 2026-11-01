"""defaults of core5 reproduce core4 (eta33 = 0): L3 states and L31 identical."""
import sys, os, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
for _x in ('x1d', 'x1c'):   # repo: core4/phase1/run2/regimes from ../x1d, core3/world/run from ../x1c
    _p = __import__('os').path.join(__import__('os').path.dirname(__import__('os').path.abspath(__file__)), '..', _x)
    if _p not in __import__('sys').path:
        __import__('sys').path.append(_p)
from world import Physics, World
from run import birth_rates
import core4, core5
from regimes import RNpm, R1e
for name, b in (('RNpm', RNpm), ('R1e', R1e)):
    ph = Physics(); w = World(ph, 0)
    rates = birth_rates(ph, w)
    B4 = core4.Brain(ph, 0, rates=rates, **b)
    B5 = core5.Brain(ph, 0, rates=rates, **b)
    w4, w5 = World(ph, 0), World(ph, 0)
    t0 = time.time(); same = True
    for t in range(3000):
        m4 = B4.step(w4.sound, getattr(w4, 'r_last', 0.0)); s4, r4 = w4.step(m4); w4.r_last = r4
        m5 = B5.step(w5.sound, getattr(w5, 'r_last', 0.0)); s5, r5 = w5.step(m5); w5.r_last = r5
        if not (np.array_equal(B4.l3, B5.l3) and np.array_equal(m4, m5)):
            same = False; print(name, 'DIFF at', t); break
    print(name, 'identical 3000 ticks:', same, 'L31 equal:', np.allclose(B4.L31, B5.L31), 'P33 equal:', np.allclose(B4.P33, B5.P33), f'{time.time()-t0:.1f}s')
