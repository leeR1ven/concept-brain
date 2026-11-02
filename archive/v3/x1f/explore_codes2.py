import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common6 import make_brain, BASE
from world import SYL, SID, render, OWN, C1, C2, D1, D2, question, answer, World
from run import parse
kv = {k: parse(v) for k, v in (a.split('=', 1) for a in sys.argv[1:])}
seed = kv.pop('seed', 0)
ph, b = make_brain(seed, dict(BASE, **kv), n_birth=100)
rng = np.random.default_rng(1)
w = World(ph, seed)
snap = b.snapshot()
V = [C1, C2, D1, D2, OWN]
rec = {(s, v): [] for s in range(len(SYL)) for v in range(5)}
for rep in range(2):
    for f in range(4):
        for x in range(8):
            for line in (question(f, x), answer(f, x, w.props)):
                for vi, v in enumerate(V):
                    b.restore(snap)
                    seq = [None] * 4 + [SID[s] for s in line for _ in range(3)] + [None] * 4
                    L2, L3 = [], []
                    for cg in seq:
                        srcs = [('s', cg, v, 1.0)] if cg is not None else []
                        b.step(render(ph, srcs, rng), 0.0, learn=False)
                        L2.append(b.l2.copy()); L3.append(b.l3.copy())
                    for i, s in enumerate(line):
                        on = 4 + 3 * i
                        for k in (2, 3, 4):
                            rec[(SID[s], vi)].append((L2[on + k - 1], L3[on + k]))
out = []
for lay in (0, 1):
    within, across, nact = [], [], []
    for s in range(len(SYL)):
        if not rec[(s, 0)]: continue
        Pv = [np.mean([r[lay] for r in rec[(s, vi)]], 0) for vi in range(5)]
        within.append(np.mean([(P >= 0.6).sum() for P in Pv]))
        Pall = np.mean(Pv, 0)
        across.append((Pall >= 0.6).sum())
        nact.append(np.mean([r[lay].sum() for vi in range(5) for r in rec[(s, vi)]]))
    within, across, nact = map(np.array, (within, across, nact))
    out.append(f"L{lay+2}: active med {np.median(nact):.0f} min {nact.min():.0f} | reliable within-voice med {np.median(within):.0f} min {within.min():.0f} | across voices med {np.median(across):.0f} min {across.min():.0f}, #syl<8 {int((across<8).sum())}")
print(kv, ' || '.join(out), flush=True)
