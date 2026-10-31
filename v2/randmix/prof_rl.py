import sys, os, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
sys.path.insert(1, os.path.join(HERE, '..', 'synth'))   # repo: the unchanged ARCH_v2 core and its bench_copy
import numpy as np
from randlayer import RandMix
exec(open(os.path.join(HERE, 'test_rl.py')).read().split('for scale in')[0])
m = RandMix(scale=1.0, fan=3, alpha=0.01, code='rand')
k1 = ctx([10, 11, 20, 21, 22], -1, 0, F1)
dk = tuple(k for k in k1 if len(k) == 2)
T = {}
def tm(name, f, n=200):
    t0 = time.time()
    for _ in range(n): r = f()
    T[name] = (time.time() - t0) / n * 1000; return r
X = tm('lines_of', lambda: m.lines_of(dk))
def part():
    st, en = m.csr_ptr[X], m.csr_ptr[X + 1]; cnt = en - st; tot = int(cnt.sum()); cs = np.cumsum(cnt)
    idx = np.arange(tot) - np.repeat(cs - cnt, cnt) + np.repeat(st, cnt); return m.csr_syn[idx]
syn = tm('gather', part)
s = tm('bincount', lambda: np.bincount(syn // 3, weights=m.w.ravel()[syn], minlength=2 * m.N).reshape(m.N, 2))
el = tm('elig', lambda: (s[:, 0] >= 1) & ((s[:, 1] >= 1) | m.direct))
E = tm('nonzero', lambda: np.flatnonzero(el))
def sel():
    sc = (s[E, 0] + s[E, 1]).astype(np.float32) + m.tie[E]; tE = m.type_of[E]; o = np.lexsort((-sc, tE)); ts = tE[o]
    first = np.searchsorted(ts, ts, side='left'); rank = np.arange(len(ts)) - first; keep = rank < m.kt[ts]
    return np.sort(E[o[keep]])
tm('select', sel)
print({k: round(v, 3) for k, v in T.items()}, 'X', len(X), 'syn', len(syn), 'E', len(E))
