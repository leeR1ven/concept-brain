"""Does the PFC's thinking follow regularities? For the most often heard first lines, each line is said in two
different voices (frozen, state restored each time). What lights inside after it (the thinking range: the PFC's
internally lit copies over the 12 quiet ticks after the line; and the hippocampus's recall) is compared:
  same line, two voices  vs  two different lines  (Jaccard overlap)
Thinking that follows the line, not the voice, gives same > different.
Usage: python -u -m core.regular <ckpt or 'newborn'> [n_lines] [key=value for a newborn]"""
import json
import sys

import numpy as np
import torch

from .config import config
from .corpus_life import CorpusLife
from .net import DEV
from .world.corpus import common_order


def jac(a, b):
    return float((a & b).sum()) / max(1.0, float((a | b).sum()))


def main():
    src, n = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 20
    if src == 'newborn':
        over = {k: json.loads(v) for k, v in (a.split('=', 1) for a in sys.argv[3:] if '=' in a)}
        life = CorpusLife(config(**{'levels_per_s': 4, 'n_brake': 20, 'birth_dialogues': 80, **over}), 0, 'data')
    else:
        life = torch.load(src, weights_only=False)['life']
    b, D = life.brain, life.train
    lines = []
    for d in common_order(D):
        t = D.turns(int(d))[0]
        if 2 <= len(t) <= 10:
            lines.append([int(x) for x in t])
        if len(lines) >= n:
            break
    res = {'pfc': {}, 'hip': {}}
    for i, ln in enumerate(lines):
        for v in ('C1', 'D2'):
            srcs = [None] * 4 + [(x, k, v, 0.0, 0.0) for x in ln for k in range(3)] + [None] * 12
            rec = {'pfc': [], 'hip': []}
            life.play(srcs, rec=lambda t: (rec['pfc'].append(b.pfc.x_int.clone()), rec['hip'].append(b.hip.h_int.clone())))
            for k in rec:
                res[k][(i, v)] = torch.stack(rec[k])[-12:].any(0)
    out = {}
    for k, R in res.items():
        same = [jac(R[(i, 'C1')], R[(i, 'D2')]) for i in range(len(lines))]
        diff = [jac(R[(i, 'C1')], R[(j, 'D2')]) for i in range(len(lines)) for j in range(len(lines)) if i != j]
        size = np.mean([float(x.sum()) for x in R.values()])
        out[k] = dict(same_line=round(float(np.mean(same)), 3), different_lines=round(float(np.mean(diff)), 3),
                      units_lit=round(float(size)))
    print('REGULAR', src, json.dumps(out), flush=True)


if __name__ == '__main__':
    main()
