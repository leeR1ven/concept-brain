"""Measure the OLD line's concept area (binary units, rule b + sight; build 9) with the designer's definition
(concept_eval.py), on the same probe dialogues as the spiking generator (train 15000..15767), sound only, frozen.
A measuring tool only: no learning (life.play restores every state).

Usage (from the repo root):  python -m core.gen.old_concept_eval <ckpt> [nd=768] [d0=15000] [ntr=6]

Lags (structure of Brain.step, one tick per syllable, hold 1): sound layer l at tick t reads the input of tick
t - (l - 1); the concept area at t reads the sound copies of t - 1. So a syllable heard at tick t is read at
ear t, A1 t, A2 t+1, A3 t+2, A4 t+3, concept t+1 (reader.py's timing check on this life's concept copies agrees:
cosine with the class mean 0.30 / 0.68 / 0.30 at lags 0 / 1 / 2). The ear row is a fresh draw of the ear's noise."""
import json
import sys
import time

import numpy as np
import torch

from ..world.corpus import script_dialogue
from ..world.voice import ADULTS
from .concept_eval import evaluate, random_projection

VIDX = {a: i for i, a in enumerate(ADULTS)}
T0 = time.time()


def say(*a):
    print(f'[{time.time() - T0:6.0f} s]', *a, flush=True)


def main(ckpt, nd=768, d0=15000, ntr=6):
    import core.corpus_life  # noqa: F401  (unpickling)
    st = torch.load(ckpt, weights_only=False)
    life = st['life']
    b, c = life.brain, life.c
    cc = getattr(b, 'concept', None)
    assert cc is not None, 'no concept area'
    assert c.get('syl_ticks', 3) == 1 and c.get('syl_hold', 1) == 1, 'one tick per syllable only'
    say(f'{ckpt}: heard {life.env.n_ex} dialogues; concept {cc.n} units, {cc.k} lit; rule '
        f'{"b" if getattr(cc, "E", None) is not None else "old"}; sight_bits {c.get("sight_bits", 0)} '
        f'p_see {c.get("p_see")}; concept_sync {c.get("concept_sync")}')
    gt = tuple(c.get('gap_turn', (4, 10)))
    lag = dict(ear=0, A1=0, A2=1, A3=2, A4=3, concept=1)
    X = {k: [] for k in lag}
    y, v, d = [], [], []
    for di in range(d0, d0 + nd):
        src = script_dialogue(life.train, di, np.random.default_rng([life.seed, 9300, di]), gt, 8, 1, 1)
        src = src + [None] * 4                              # room for the deepest lag
        rec = {k: [] for k in lag}

        def r_(t):
            for l in range(4):
                rec[f'A{l + 1}'].append(b.sound.state[l].clone())
            rec['concept'].append(cc.pop.on.clone())
        for x in src:
            rec['ear'].append(torch.as_tensor(life.coch.a0([] if x is None else [x[:5]]), device=cc.pop.on.device))
        life.play([None if x is None else x[:5] for x in src], rec=r_)
        R = {k: torch.stack(rec[k]) for k in lag}
        for t, x in enumerate(src):
            if x is None or t + 3 >= len(src):
                continue
            for k, l_ in lag.items():
                X[k].append(R[k][t + l_].cpu())
            y.append(int(x[0]))
            v.append(VIDX[x[2]])
            d.append(di)
        if (di - d0) % 128 == 127:
            say(f'heard {di - d0 + 1} dialogues, {len(y)} hearings')
    X = {k: torch.stack(val).float().cuda() for k, val in X.items()}
    y, v, d = np.array(y), np.array(v), np.array(d)
    say('lit per hearing', {k: round(float(val.sum(1).mean()), 1) for k, val in X.items()},
        'of', {k: val.shape[1] for k, val in X.items()})
    X['RP'] = random_projection(X['A1'], cc.n, float(X['concept'].sum(1).mean()), seed=1)
    for h in (0, 1):
        ev = evaluate(X, y, v, d, n_tr=ntr, seed=1, half=h)
        print('CONCEPT EVAL OLD' + (' (halves swapped)' if h else ''), json.dumps(ev), flush=True)


if __name__ == '__main__':
    kw = dict(a.split('=', 1) for a in sys.argv[2:])
    main(sys.argv[1], **{k: int(val) for k, val in kw.items()})
