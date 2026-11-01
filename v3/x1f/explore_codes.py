import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common6 import make_brain, BASE
from world import SYL, SID, render, OWN, C1, C2, D1, D2, question, answer, World
from wire import wire_speech
from run import parse
kv = {k: parse(v) for k, v in (a.split('=', 1) for a in sys.argv[1:])}
seed = kv.pop('seed', 0)
ph, b = make_brain(seed, dict(BASE, **kv), n_birth=100)
wire_speech(b, ph)
lab = b.PM3.argmax(1)

rng = np.random.default_rng(1)
w = World(ph, 0)
snap = b.snapshot()
# record L3 at onset+2..+4 for each syllable heard in questions and answers, various voices
rec = {s: [] for s in range(len(SYL))}
sil4 = []
for rep in range(4):
    for f in range(4):
        for x in range(8):
            for line in (question(f, x), answer(f, x, w.props)):
                v = [C1, C2, D1, D2, OWN][int(rng.integers(5))]
                b.restore(snap)
                seq = [None] * 4 + [SID[s] for s in line for _ in range(3)] + [None] * 4
                L = []
                for cg in seq:
                    srcs = [('s', cg, v, 1.0)] if cg is not None else []
                    b.step(render(ph, srcs, rng), 0.0, learn=False)
                    L.append(b.l3.copy())
                sil4.append(b.l3.sum())
                for i, s in enumerate(line):
                    on = 4 + 3 * i
                    for k in (2, 3, 4):
                        if on + k < len(L):
                            rec[SID[s]].append(L[on + k])
codes = {}
for s, Ls in rec.items():
    if not Ls: continue
    P = np.mean(Ls, 0)
    codes[s] = P
    codes[s] = P
ks = sorted(codes)
C6 = np.array([codes[s] >= 0.5 for s in ks])
inter = C6.astype(int) @ C6.T.astype(int); n = C6.sum(1)
J = inter / np.maximum(n[:, None] + n[None, :] - inter, 1)
np.fill_diagonal(J, 0)
rel = np.array([(codes[s] >= 0.6).sum() for s in ks]); act = np.array([np.mean([l.sum() for l in rec[s]]) for s in ks])
print(kv, 'reliable units (p>=0.6) min/median/max', rel.min(), int(np.median(rel)), rel.max(), '| #syl with <8:', int((rel < 8).sum()), 'of', len(ks),
      '| mean active min/median/max', round(act.min(), 1), round(float(np.median(act)), 1), round(act.max(), 1), '| code J mean/max', J.mean().round(3), J.max().round(3),
      '| n3 after 4 silent', round(float(np.mean(sil4)), 1), flush=True)
