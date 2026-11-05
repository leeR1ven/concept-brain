"""Record what the fixed newborn network does while it hears the stream, once, so every readout can be computed from
the recording in seconds instead of re-running the life. Nothing in the network learns. Saved per tick: which last-layer
units and which outside copies are lit, and the true syllable (kept for measuring only, the network never sees it).
At the end: heard lines and never-heard lines, each in its own order and shuffled.
usage: python -u stream_record.py <dialogues> <out.npz> [key=value]   (from a folder with core/ and data/)"""
import json
import sys

import numpy as np
import torch

from core.config import config
from core.corpus_life import CorpusLife
from core.exam import first_lines

torch.set_num_threads(4)


def main():
    total, out = int(sys.argv[1]), sys.argv[2]
    over = {k: json.loads(v) for k, v in (a.split('=', 1) for a in sys.argv[3:] if '=' in a)}
    life = CorpusLife(config(**{'levels_per_s': 4, 'n_brake': 20, 'birth_dialogues': 80, 's': 1, **over}), 0, 'data')
    b, p = life.brain, life.brain.pfc
    b.learning = False
    n = p.L4.on.shape[0]
    A, X, lab = [], [], []

    def keep(label):
        A.append(torch.nonzero(p.L4.on).squeeze(1).numpy().astype(np.uint16))
        X.append(torch.nonzero(p.L1ext.on).squeeze(1).numpy().astype(np.uint16))
        lab.append(label)

    n0 = life.env.n_ex
    while life.env.n_ex - n0 < total:
        life.tick()
        c = life.env.cur
        keep((-1, -1, -1, -1, -1) if c is None else (c[0], c[1], c[5], c[6], c[7]))
    train_ticks = len(lab)
    print('heard', train_ticks, 'ticks', flush=True)

    rng = np.random.default_rng(3)

    def shuf(ln):
        for _ in range(20):
            s = [ln[i] for i in rng.permutation(len(ln))]
            if s != ln:
                return s
        return ln

    heard = [list(map(int, t)) for d in range(total) for t in life.train.turns(d % life.train.n) if 4 <= len(t) <= 10]
    heard = [heard[i] for i in rng.permutation(len(heard))[:120]]
    new = first_lines(life.heldout, range(life.heldout.n), 120, lo=4, hi=10)
    tests = []
    for kind, lines in ((0, heard), (1, new)):
        for li, ln in enumerate(lines):
            tests += [(kind, li, 0, ln), (kind, li, 1, shuf(ln))]
    for kind, li, order, ln in tests:
        seq = [None] * 15 + [(x, k, 'C1', 0.0, 0.0) for x in ln for k in range(3)] + [None] * 3
        pos = [-1] * 15 + [j for j in range(len(ln)) for _ in range(3)] + [-1] * 3
        tag = 1000 + 4 * li + 2 * kind * 1000 + order

        def rec(t, seq=seq, pos=pos, tag=tag):
            s = seq[t]
            keep((-1, -1, tag, -1, -1) if s is None else (s[0], s[1], tag, -1, pos[t]))
        life.play(seq, rec=rec)
    off = lambda L: np.cumsum([0] + [len(x) for x in L])
    np.savez(out, n=n, train_ticks=train_ticks, a=np.concatenate(A), a_off=off(A), x=np.concatenate(X), x_off=off(X),
             lab=np.array(lab, np.int32))
    print('RECORDED', out, len(lab), 'ticks; lit per tick: last layer', len(np.concatenate(A)) / len(lab),
          'copies', len(np.concatenate(X)) / len(lab), 'n', n, flush=True)


if __name__ == '__main__':
    main()
