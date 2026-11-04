"""Theory part 1, the fixed path (ear -> sound areas -> PFC first layer -> layers 2, 3 -> last layer): how alike are the
codes of the same thing, and how apart are those of different things, layer by layer? Forward computation only.
The 30 commonest syllables, each heard after quiet (voice C1), the code = the units lit while it is heard and the few
ticks it takes to reach the last layer. Overlap of two codes = shared / sqrt(size x size) (chance = the density):
  noise    the same syllable again (the ear differs)        voice    the same syllable in another voice (D1)
  context  the same syllable after two different syllables  other    different syllables
usage: python -u -m core.theory1 <ckpt or 'newborn'> [tick=1: one tick per layer instead of 7 merged] [key=value]"""
import json
import sys

import numpy as np
import torch

from .config import config
from .corpus_life import CorpusLife

PRE = 15
TICK = 0
LAYERS = ('input', 'L2', 'L3', 'L4')


def main():
    src = sys.argv[1]
    over = {k: json.loads(v) for k, v in (a.split('=', 1) for a in sys.argv[2:] if '=' in a)}
    global TICK
    TICK = over.pop('tick', 0)
    if src == 'newborn':
        life = CorpusLife(config(**{'levels_per_s': 4, 'n_brake': 20, 'birth_dialogues': 80, **over}), 0, 'data')
    else:
        life = torch.load(src, weights_only=False)['life']
    b, p = life.brain, life.brain.pfc
    f = np.bincount(life.train.syl.astype(np.int64))
    syl = [int(x) for x in np.argsort(-f)[:30]]

    def code(seq, v='C1'):
        """codes of the last syllable of seq, per layer: lit in its 3 ticks and the 4 after (the path's delay)"""
        rec = []
        life.play([None] * PRE + [(x, k, v, 0.0, 0.0) for x in seq for k in range(3)] + [None] * 4,
                  rec=lambda t: rec.append(torch.stack([p.L1ext.on, p.L2.on, p.L3.on, p.L4.on])))
        X = torch.stack(rec)
        s = PRE + 3 * (len(seq) - 1)
        if TICK:                                                     # one tick per layer: the path's delay
            return torch.stack([X[s + 2 + q, q] for q in range(4)])
        return X[s:s + 7].any(0)                                     # [layers, n]

    def ov(a, c):
        return (a & c).sum(-1).float() / ((a.sum(-1) * c.sum(-1)).float().sqrt().clamp_min(1))

    C1 = [code([x]) for x in syl]
    C1b = [code([x]) for x in syl]
    D1 = [code([x], 'D1') for x in syl]
    X1 = [code([syl[(i + 7) % 30], x]) for i, x in enumerate(syl)]
    X2 = [code([syl[(i + 13) % 30], x]) for i, x in enumerate(syl)]
    m = {'noise': [ov(a, c) for a, c in zip(C1, C1b)], 'voice': [ov(a, c) for a, c in zip(C1, D1)],
         'context': [ov(a, c) for a, c in zip(X1, X2)],
         'other': [ov(C1[i], C1[j]) for i in range(30) for j in range(i + 1, 30)]}
    dens = torch.stack(C1).float().mean((0, 2))
    res = {}
    for q, nm in enumerate(LAYERS):
        d = float(dens[q])
        r = {k: round(float(torch.stack(v)[:, q].mean()), 3) for k, v in m.items()}
        r['density'] = round(d, 4)
        res[nm] = r
    # common vs own part of the last layer's codes: a unit lit for at least half of the syllables is common; it
    # pushes every copy alike, so each copy's usual level takes it out and it does not interfere
    L4 = torch.stack([x[3] for x in C1]).float()                 # [30, n]
    freq = L4.mean(0)
    hub = freq >= 0.5
    own = lambda X: [x[3] & ~hub for x in X]
    m2 = {'noise': [ov(a, c) for a, c in zip(own(C1), own(C1b))], 'voice': [ov(a, c) for a, c in zip(own(C1), own(D1))],
          'context': [ov(a, c) for a, c in zip(own(X1), own(X2))],
          'other': [ov(a, c) for i, a in enumerate(own(C1)) for c in own(C1)[i + 1:]]}
    split = dict(common_share=round(float((L4[:, hub].sum(1) / L4.sum(1).clamp_min(1)).mean()), 3),
                 own_lit=round(float(L4[:, ~hub].sum(1).mean()), 1), own_density=round(float(L4[:, ~hub].mean()), 4),
                 **{k: round(float(torch.stack(v).mean()), 3) for k, v in m2.items()})
    print('THEORY1', src, json.dumps(dict(over=over, layers=res, last_layer_own_part=split)), flush=True)


if __name__ == '__main__':
    main()
