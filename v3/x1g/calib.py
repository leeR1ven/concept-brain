"""threshold calibration for the sparse arms: L2 / L3 active counts while overhearing and per-syllable L3 code sizes.
python3 -I calib.py N f2,f3[;f2,f3...] [n_dlg] [eta33] [seed]"""
import sys, time, numpy as np
sys.path.insert(0, '.')
for _x in ('x1f', 'x1d', 'x1c'):   # repo: core5/core6/common6/world6 from ../x1f, core4/phase1 from ../x1d, core3/world/run from ../x1c
    _p = __import__('os').path.join(__import__('os').path.dirname(__import__('os').path.abspath(__file__)), '..', _x)
    if _p not in __import__('sys').path:
        __import__('sys').path.append(_p)
from world import Physics, World, render, SYL, C1, C2
from world6 import OverhearWorld
from run import birth_rates
from common6 import GEN0, run_until
import phase1, core8
from wire8 import wire_speech

def born(seed, bk, n_birth=100):
    ph = Physics(); w0 = World(ph, seed)
    nol = dict(bk); keep = {k: bk.get(k) for k in phase1.NOLEARN}; nol.update(phase1.NOLEARN)
    b = core8.Brain(ph, seed, rates=birth_rates(ph, w0), **nol)
    run_until(b, World(ph, seed + 50), n_birth)
    for k in phase1.NOLEARN:
        b.c[k] = keep[k] if keep[k] is not None else core8.DEF[k]
    return ph, b

def codes(b, ph, seed, reps=3):
    snap = b.snapshot(); rng = np.random.default_rng(seed + 5)
    out = {}
    for v in (C1, C2):
        for s in range(len(SYL)):
            acc = []
            for r in range(reps):
                b.restore(snap)
                for t in range(4):
                    b.step(render(ph, [], rng), 0.0, learn=False)
                for t in range(5):                       # L3 reflects the input 2 ticks later: record ticks 2-4
                    b.step(render(ph, [('s', s, v, 1.0)] if t < 3 else [], rng), 0.0, learn=False)
                    if t >= 2:
                        acc.append(b.l3.copy())
            out[(s, v)] = np.mean(acc, 0)
    b.restore(snap)
    rel = {k: np.nonzero(p >= 0.6)[0] for k, p in out.items()}
    n = [len(rel[(s, C1)]) for s in range(len(SYL))]
    both = [len(np.intersect1d(rel[(s, C1)], rel[(s, C2)])) for s in range(len(SYL))]
    # overlap between syllable codes (same voice)
    J = []
    for s in range(len(SYL)):
        for u in range(s + 1, len(SYL)):
            a, c = rel[(s, C1)], rel[(u, C1)]
            if len(a) and len(c):
                J.append(len(np.intersect1d(a, c)) / len(np.union1d(a, c)))
    return n, both, J

N = int(sys.argv[1])
grid = [tuple(float(x) for x in g.split(',')) for g in sys.argv[2].split(';')]
n_dlg = int(sys.argv[3]) if len(sys.argv) > 3 else 100
eta33 = float(sys.argv[4]) if len(sys.argv) > 4 else 0.02
seed = int(sys.argv[5]) if len(sys.argv) > 5 else 0
extra = dict(kv.split('=') for kv in sys.argv[6:])
extra = {k: float(v) for k, v in extra.items()}
for f2, f3 in grid:
    t0 = time.time()
    bk = dict(GEN0, n2=N, n3=N, f2=f2, f3=f3, lat_fly=True, eta33=eta33, **extra)
    if N > 400:
        bk['n_pm'] = N // 33
    ph, b = born(seed, bk)
    wire_speech(b, ph)
    n, both, J = codes(b, ph, seed)
    ow = OverhearWorld(ph, seed)
    n2, n3, npr, utt = [], [], [], 0
    t1 = time.time()
    while ow.n_ex < n_dlg:
        mus = b.step(ow.sound, 0.0); ow.step(mus)
        n2.append(b.l2.sum()); n3.append(b.l3.sum()); npr.append(b.pred.sum())
    dt = (time.time() - t1) / len(n2)
    n_, both_, J_ = codes(b, ph, seed)
    n2, n3 = np.array(n2), np.array(n3)
    q = lambda v: '%.0f/%.0f/%.0f/%d' % (np.percentile(v, 10), v.mean(), np.percentile(v, 90), v.max())
    print(f'N={N} f2={f2} f3={f3}: overhear {n_dlg}: L2 p10/mean/p90/max {q(n2)} | L3 {q(n3)} P(L3=0) {(n3==0).mean():.3f} '
          f'| pred mean {np.mean(npr):.1f} | birth L3 code C1 min/med/max {min(n)}/{np.median(n):.0f}/{max(n)} x-voice med {np.median(both):.0f} J {np.mean(J):.3f} '
          f'| after: {min(n_)}/{np.median(n_):.0f}/{max(n_)} xv {np.median(both_):.0f} J {np.mean(J_):.3f} | child utts/dlg {len(ow.utts)/n_dlg:.2f} | {1000*dt:.2f} ms/tick, {time.time()-t0:.0f}s', flush=True)
