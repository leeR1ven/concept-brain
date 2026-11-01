"""core6 with no hand-added links must reproduce core5 exactly (L3 states over 3,000 world ticks, learning on)."""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from world import Physics, World
from run import birth_rates
import core5, core6
from common6 import BASE

ph = Physics()
w = World(ph, 0)
rates = birth_rates(ph, w)
bs = []
for C in (core5.Brain, core6.Brain):
    b = C(ph, 0, rates=rates, **BASE)
    ww = World(ph, 0)
    L = []
    for t in range(3000):
        mus = b.step(ww.sound, getattr(ww, 'r_last', 0.0))
        snd, r = ww.step(mus)
        ww.r_last = r
        L.append(b.l3.copy())
    bs.append((np.array(L), b.L31.copy(), b.L33.copy()))
print('L3 identical', bool((bs[0][0] == bs[1][0]).all()), 'L31', bool(np.allclose(bs[0][1], bs[1][1])),
      'L33', bool(np.allclose(bs[0][2], bs[1][2])), 'mean n3', bs[0][0].sum(1).mean())
