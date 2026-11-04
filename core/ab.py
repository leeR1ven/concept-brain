"""The simplest check of the PFC's one job (ARCH 三 前额叶: the last layer now -> the first layer next, Hebbian):
hear two syllables A B many times; then hear A alone. Are B's copies lit from inside afterwards?
B's copies = the first-layer copies B lights from outside (heard after A, before any learning).
Per tick after A starts (A alone, from quiet, nothing learned during the probe):
  lit_B     B's copies lit as thought            thoughts   all copies lit as thought (the budget used)
  over_B    B's copies whose push is above theta1 (could be lit)
  push_B    mean push (e1) on B's copies          cut        the weakest push that still got a slot
  of the thoughts: B's / A's own copies / others
usage: python -u -m core.ab <ckpt or 'newborn'> [reps=20] [pair=new | A=sid B=sid] [key=value]"""
import json
import sys
import time

import numpy as np
import torch

from .config import config
from .corpus_life import CorpusLife
from .net import DEV


def main():
    src = sys.argv[1]
    over = {k: json.loads(v) for k, v in (a.split('=', 1) for a in sys.argv[2:] if '=' in a)}
    reps = over.pop('reps', 20)
    t0 = time.time()
    if src == 'newborn':
        life = CorpusLife(config(**{'levels_per_s': 4, 'n_brake': 20, 'birth_dialogues': 80,
                                    **{k: v for k, v in over.items() if k not in ('A', 'B', 'pair')}}), 0, 'data')
    else:
        life = torch.load(src, weights_only=False)['life']
        life.c.update({k: v for k, v in over.items() if k in life.c})
    b, p, c = life.brain, life.brain.pfc, life.c
    D = life.train
    if over.get('pair') == 'new':                       # two syllables never heard one after the other in training
        s = D.syl.astype(np.int64)
        keep = np.ones(len(s) - 1, bool)
        st = D.turn_off[(D.turn_off > 0) & (D.turn_off < len(s))]
        keep[st - 1] = False
        seen = np.unique(s[:-1][keep] * 4096 + s[1:][keep])
        t = next(tt[i:i + 2] for d in range(life.heldout.n) for tt in life.heldout.turns(d) for i in range(len(tt) - 1)
                 if not np.isin(int(tt[i]) * 4096 + int(tt[i + 1]), seen))
    else:                                               # the first two syllables of the first training line
        t = next(tt for d in range(D.n) for tt in D.turns(d) if len(tt) >= 2)
    A, B = int(over.get('A', t[0])), int(over.get('B', t[1]))
    quiet = 30

    def srcs(seq):
        return [None] * quiet + [(x, k, 'C1', 0.0, 0.0) for x in seq for k in range(3)] + [None] * 8

    def play(seq, rec):
        life.play(srcs(seq), rec=rec)

    ext = []
    play([A, B], lambda tt: ext.append(p.L1ext.on.clone()))
    E = torch.stack(ext)
    a_cp = E[quiet:quiet + 3].any(0) & ~E[quiet - 1]
    b_cp = E[quiet + 3:quiet + 6].any(0) & ~E[quiet:quiet + 3].any(0)     # what B adds after A

    def probe():
        rows = []

        def r(tt):
            if tt < quiet:
                return
            x, e1 = p.x_int, p.e1
            k = int(c['context'] * p.L1ext.on.sum())
            cut = float(torch.topk(e1, k).values[-1]) if k > 0 else float('inf')
            rows.append(dict(tick=tt - quiet, lit_B=int((x & b_cp).sum()), thoughts=int(x.sum()),
                             over_B=int(((e1 > c['theta1']) & b_cp).sum()), push_B=round(float(e1[b_cp].mean()), 2),
                             cut=round(cut, 2), of_thoughts_A=int((x & a_cp).sum()),
                             of_thoughts_other=int((x & ~a_cp & ~b_cp).sum())))
        play([A], r)
        return rows

    before = probe()
    was, b.learning = b.learning, True
    for _ in range(reps):
        for x in srcs([A, B]):
            b.step(torch.as_tensor(life.coch.a0([] if x is None else [x]), device=DEV), 0.0)
    b.learning = was
    after = probe()
    print('AB', src, json.dumps(dict(A=A, B=B, n_A_copies=int(a_cp.sum()), n_B_copies=int(b_cp.sum()), reps=reps,
                                     theta1=c['theta1'], before=before[:12], after=after[:12],
                                     secs=round(time.time() - t0))), flush=True)


if __name__ == '__main__':
    main()
