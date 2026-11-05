"""A short local learning probe (s=1): do the last layer's own links, with the cap on, keep the code set by the input?
Before learning and after `dialogues` overheard dialogues (learning on, the real stream), the same 20 syllables are
heard alone after quiet twice (ear noise differs) and in another voice; the last-layer code (ticks 4..6) overlap:
same syllable / other voice / different syllables, with chance = density. Also the own links' total push per unit
(mean over lit units) against the input push's spread. cap=off turns the cap off; cap=<number> sets it.
usage (from a directory holding core/ and data/):  python3 cap_probe.py [dialogues=150] [cap=on|off|<number>] [key=value]"""
import json
import sys
import time

import numpy as np
import torch

from core.config import config
from core.corpus_life import CorpusLife

PRE = 12


def main():
    over = {k: json.loads(v) if v not in ('on', 'off') else v for k, v in (a.split('=', 1) for a in sys.argv[1:])}
    n_dlg, cap = over.pop('dialogues', 150), over.pop('cap', 'on')
    t0 = time.time()
    life = CorpusLife(config(**{'levels_per_s': 4, 'n_brake': 20, 'birth_dialogues': 80, **over}), 0, 'data')
    b, p = life.brain, life.brain.pfc
    if cap == 'off':
        p.L44.cap = None
    elif cap != 'on':
        p.L44.cap = float(cap)                      # a value: the total own-link push per unit
    f = np.bincount(life.train.syl.astype(np.int64))
    syl = [int(x) for x in np.argsort(-f)[:20]]

    def code(x, v='C1'):
        rec = []
        life.play([None] * PRE + [(x, k, v, 0.0, 0.0) for k in range(3)] + [None] * 6,
                  rec=lambda t: rec.append(p.L4.on.clone()) if PRE + 4 <= t <= PRE + 6 else None)
        return torch.stack(rec).any(0)

    def ov(a, c):
        return float((a & c).sum() / max(1.0, float((a.sum() * c.sum()).float().sqrt())))

    def measure():
        A = [code(x) for x in syl]
        B = [code(x) for x in syl]
        D = [code(x, 'D1') for x in syl]
        own = p.L44.drive(p.L4.on)
        return dict(same=round(float(np.mean([ov(a, c) for a, c in zip(A, B)])), 3),
                    voice=round(float(np.mean([ov(a, c) for a, c in zip(A, D)])), 3),
                    other=round(float(np.mean([ov(A[i], B[(i + 7) % 20]) for i in range(20)])), 3),
                    density=round(float(torch.stack(A).float().mean()), 4),
                    own_push_on_lit=round(float(own[p.L4.on].mean()) if p.L4.on.any() else 0.0, 3),
                    own_w_max=round(float(p.L44.w.max()), 3), cap=p.L44.cap)
    before = measure()
    print('CAP before', json.dumps(before), flush=True)
    b.learning = True
    n0 = life.env.n_ex
    while life.env.n_ex - n0 < n_dlg:
        life.tick()
    b.learning = False
    after = measure()
    print('CAP after', json.dumps(dict(dialogues=n_dlg, ticks=b.t, secs=round(time.time() - t0), **after)), flush=True)


if __name__ == '__main__':
    main()
