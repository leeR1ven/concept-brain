# unit test + timing of RandMix on synthetic contexts (MIX-style keys)
import sys, os, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
sys.path.insert(1, os.path.join(HERE, '..', 'synth'))   # repo: the unchanged ARCH_v2 core and its bench_copy
import numpy as np
from randlayer import RandMix
from rules import pair_keys
from core_think import ThinkCore
PAIRS = [('P', 'M'), ('L', 'M'), ('P', 'L'), ('P', 'N'), ('L', 'N'), ('P1', 'N'), ('PL', 'N'),
         ('P', 'CP'), ('L', 'CP'), ('N', 'CP'), ('P', 'CN'), ('L', 'CN'), ('FAM', 'CN'),
         ('P', 'CC'), ('L', 'CC'), ('FAM', 'CC'), ('P', 'FAM'), ('N', 'FAM'),
         ('P', 'A'), ('L', 'A'), ('N', 'A'), ('P', 'F'), ('L', 'F'), ('P', 'VR'), ('L', 'VR'), ('N', 'VR')]
rng = np.random.default_rng(0)
def ctx(P, L, n, F):
    g = {'P': P, 'L': [L], 'N': [n], 'FAM': [1], 'P1': [P[0]], 'PL': [P[-1]], 'CP': [P[0]], 'CN': [-2], 'CC': [-2],
         'F': [('r', b) for b in F], 'A': [1], 'VR': [P[0]], 'M': [1]}
    return pair_keys(g, PAIRS)
F1 = sorted(rng.choice(240, 32, replace=False)); F2 = sorted(rng.choice(240, 32, replace=False))
for scale in (1.0, 4.0):
    for fan, alpha in ((3, 0.01), (6, 0.01), (3, 0.03)):
        t0 = time.time(); m = RandMix(scale=scale, fan=fan, alpha=alpha, code='rand'); tb = time.time() - t0
        k1 = ctx([10, 11, 20, 21, 22], -1, 0, F1)      # 苹果好不好吃 at line start
        k2 = ctx([12, 13, 20, 21, 22], -1, 0, F2)      # 香蕉好不好吃 (same frame, other filler, other object)
        k3 = ctx([10, 11, 30, 31, 32], -1, 0, F1)      # 苹果 in another frame
        k1b = ctx([10, 99, 20, 21, 22], -1, 0, F1)     # one syllable misheard (果 -> 99)
        t0 = time.time()
        for i in range(200):
            m.cache = {}; A1 = m.units(k1)
        ta = (time.time() - t0) / 200
        A2, A3, A1b = m.units(k2), m.units(k3), m.units(k1b)
        j = lambda a, b: len(np.intersect1d(a, b)) / max(len(np.union1d(a, b)), 1)
        print(f'scale {scale} fan {fan} alpha {alpha}: N {m.N} active {len(A1)} build {tb:.1f}s act {ta*1000:.2f} ms | '
              f'J(same frame) {j(A1, A2):.2f} J(same filler) {j(A1, A3):.2f} J(misheard) {j(A1, A1b):.2f}')
# the same three comparisons for the dict MIX: overlap of the key sets (= MIX units) of the contexts
jk = lambda a, b: len(set(a) & set(b)) / max(len(set(a) | set(b)), 1)
k1 = ctx([10, 11, 20, 21, 22], -1, 0, F1); k2 = ctx([12, 13, 20, 21, 22], -1, 0, F2)
k3 = ctx([10, 11, 30, 31, 32], -1, 0, F1); k1b = ctx([10, 99, 20, 21, 22], -1, 0, F1)
print(f'dict MIX: keys {len(set(k1))} | J(same frame) {jk(k1, k2):.2f} J(same filler) {jk(k1, k3):.2f} J(misheard) {jk(k1, k1b):.2f}')
