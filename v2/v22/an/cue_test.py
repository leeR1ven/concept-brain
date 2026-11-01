# -*- coding: utf-8 -*-
"""Step 3, candidate cues tested on v2.1 life logs (ground truth from the world log; the agent never reads it).
For partner lines heard after my own utterance (R = 1) that I speak right after (<= 12 ticks): could a cue separate
corrections (should commit) from questions (should cancel)?
  timing: gap from my utterance's end to the line's onset <= G  -> 'contingent'
  take-up: my next utterance repeats the line (exact, or a >= 2-char piece of it)
usage: python3 an/cue_test.py <log.pkl> ..."""
import sys, pickle, collections
import numpy as np
sys.path.insert(0, '.')
from life_diag import lines


def rows(L, W=12):
    P, U = lines(L)
    Ut0 = np.array([u[0] for u in U]); Ut1 = np.array([u[1] for u in U])
    out = []
    for i, p in enumerate(P):
        t0, t1, txt, ty = p[0], p[1], p[2], p[3]
        prev_end = P[i - 1][1] if i else -10 ** 9
        m = (Ut1 <= t0) & (Ut0 >= prev_end)
        gap = int(t0 - Ut1[m].max()) if m.any() else None
        nxt = P[i + 1][0] if i + 1 < len(P) else 10 ** 12
        j = int(np.searchsorted(Ut0, t1))
        sp = j < len(U) and U[j][0] - t1 <= W and U[j][0] < nxt
        rep = U[j][2] if sp else None
        exact = sp and rep == txt
        piece = sp and (exact or (len(rep) >= 2 and rep in txt) or (len(txt) >= 2 and txt in rep))
        out.append(dict(ty=ty, gap=gap, spoke=sp, exact=exact, piece=piece))
    return out


for path in sys.argv[1:]:
    R = rows(pickle.load(open(path, 'rb')))
    cor = [r for r in R if r['ty'] in ('correct',) and r['gap'] is not None and r['spoke']]
    q = [r for r in R if r['ty'] == 'question' and r['gap'] is not None and r['spoke']]
    print(f"== {path}: R=1 lines I spoke after: corrections {len(cor)}, questions {len(q)}")
    for G in (10, 20, 30, 40, 45, 60):
        a = np.mean([r['gap'] <= G for r in cor]); b = np.mean([r['gap'] <= G for r in q])
        print(f"   timing gap<={G:3d}: corrections captured {a:.2f} | questions wrongly captured {b:.2f}")
    print(f"   take-up exact repeat: corrections {np.mean([r['exact'] for r in cor]):.2f} | questions {np.mean([r['exact'] for r in q]):.2f}")
    print(f"   take-up repeat/piece: corrections {np.mean([r['piece'] for r in cor]):.2f} | questions {np.mean([r['piece'] for r in q]):.2f}")
    allq = [r for r in R if r['ty'] == 'question']
    print(f"   questions heard after my own utterance (R=1): {np.mean([r['gap'] is not None for r in allq]):.2f} of {len(allq)}")
