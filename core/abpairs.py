"""Hearing A, does it think of B? (the designer's test: after A, B's features light up.) The 20 commonest steps A -> B
inside the training dialogues (A != B). For each, A alone after quiet, nothing learned during the probe; over the 9
ticks after A starts: the share of B's own copies (lit by B, not by C or A) that are thought of, and, for comparison,
the same share for C, the B of another pair (one that rarely follows A). A syllable's copies = what it lights from
outside when heard alone after quiet.
usage: python -u -m core.abpairs <ckpt or 'newborn'> [key=value]"""
import json
import sys
import time

import numpy as np
import torch

from .config import config
from .corpus_life import CorpusLife
from .world import phon

QUIET = 30


def main():
    src = sys.argv[1]
    over = {k: json.loads(v) for k, v in (a.split('=', 1) for a in sys.argv[2:] if '=' in a)}
    t0 = time.time()
    if src == 'newborn':
        life = CorpusLife(config(**{'levels_per_s': 4, 'n_brake': 20, 'birth_dialogues': 80, **over}), 0, 'data')
    else:
        life = torch.load(src, weights_only=False)['life']
    b, p, D = life.brain, life.brain.pfc, life.train
    s = D.syl.astype(np.int64)
    keep = np.ones(len(s) - 1, bool)
    st = D.turn_off[(D.turn_off > 0) & (D.turn_off < len(s))]
    keep[st - 1] = False
    keep &= s[:-1] != s[1:]
    code, cnt = np.unique(s[:-1][keep] * 4096 + s[1:][keep], return_counts=True)
    pairs, used_b = [], set()
    for c in code[np.argsort(-cnt)]:
        a_, b_ = int(c // 4096), int(c % 4096)
        if b_ not in used_b:
            pairs.append((a_, b_))
            used_b.add(b_)
        if len(pairs) == 20:
            break
    follows = dict(zip(code.tolist(), cnt.tolist()))

    def srcs(x):
        return [None] * QUIET + [(x, k, 'C1', 0.0, 0.0) for k in range(3)] + [None] * 9

    def copies(x):
        E = []
        life.play(srcs(x), rec=lambda t: E.append(p.L1ext.on.clone()))
        E = torch.stack(E)
        return E[QUIET:QUIET + 6].any(0) & ~E[QUIET - 1]

    cp = {x: copies(x) for x in {y for pr in pairs for y in pr}}

    def thoughts(x):
        T = []
        life.play(srcs(x), rec=lambda t: T.append(p.x_int.clone()))
        return torch.stack(T)[QUIET + 1:QUIET + 10]

    bases = phon.load_inventory()
    name = lambda x: bases[x // 5] + str(x % 5 + 1)
    rows = []
    for i, (a_, b_) in enumerate(pairs):
        c_ = next(pairs[(i + j) % len(pairs)][1] for j in range(1, len(pairs))
                  if follows.get(a_ * 4096 + pairs[(i + j) % len(pairs)][1], 0) * 20 < follows[a_ * 4096 + b_])
        T = thoughts(a_).any(0)                             # copies thought of in the 9 ticks
        bx, cx = cp[b_] & ~cp[c_] & ~cp[a_], cp[c_] & ~cp[b_] & ~cp[a_]   # what only B / only C lights
        rows.append(dict(A=name(a_), B=name(b_), C=name(c_), thoughts=int(T.sum()),
                         B_share=round(float((T & bx).sum()) / max(1.0, float(bx.sum())), 3),
                         C_share=round(float((T & cx).sum()) / max(1.0, float(cx.sum())), 3)))
    bs, cs = np.array([r['B_share'] for r in rows]), np.array([r['C_share'] for r in rows])
    print('ABPAIRS', src, json.dumps(dict(B_share=round(float(bs.mean()), 3), C_share=round(float(cs.mean()), 3),
                                          B_wins=int((bs > cs).sum()), pairs=len(rows), rows=rows,
                                          secs=round(time.time() - t0))), flush=True)


if __name__ == '__main__':
    main()
