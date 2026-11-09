"""Listening to others (designer 11-09; ARCH 五: the brain hears many dialogues and learns what comes next, no reward):
dialogues of the logic language ("T ? YES", "NOT T ? NO", ...) are shown and the PFC learns as it always does (each
first-layer copy checks what the last layer lit in it, the timing rule). The muscles stay still; nothing is rewarded.
After every block a test without learning: at the frame before the answer word appears, which word does the PFC
expect? Its push into the copies the YES word lights, minus the copies the NO word lights (the copies' drive above
their usual level: what it would think of next).
  python listen.py <brain.pt> <out.pt> <dialogues> [seed]   (LOGIC_STAGE: dialogues of stages up to it, default 5;
                                                             LOGIC_LOOP_X, LOGIC_L4_USUAL, LOGIC_SCALE as in life.py)"""
import collections
import copy
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'game'))
sys.path.insert(0, os.path.join(HERE, '..', '..'))
import game_brain                                  # noqa: E402
from game_brain import DEV                         # noqa: E402
import world as W                                  # noqa: E402

LOOP_X = float(os.environ.get('LOGIC_LOOP_X', 1.0))
BLOCK, TESTS = 500, 200


def word_copies(b, G):
    """the vision copies the YES word lights and the NO word does not, and the other way round"""
    nv = b.ch.n_vision
    rate = {}
    for w in ('YES', 'NO'):
        bb = copy.deepcopy(b)
        bb.pfc.adapt = False
        on = torch.zeros(b.ch.n, device=DEV)
        blank = np.zeros((W.ROWS, W.COLS), bool)
        for _ in range(20):
            for cells in [blank] * 4 + [G[w]] * 3:
                bb.step(cells, np.zeros(2, bool))
            on += bb.pfc.L1ext.on.float()
        rate[w] = on[:nv] / 20
    yes = torch.nonzero((rate['YES'] > 0.5) & (rate['NO'] < 0.1)).squeeze(1)
    no = torch.nonzero((rate['NO'] > 0.5) & (rate['YES'] < 0.1)).squeeze(1)
    return yes, no


def expect(b):
    """the push the last layer gives every first-layer copy at the next tick, above its usual level"""
    p = b.pfc
    d = p.loop.drive(p.L4.on) + p.L41.drive(p.L4.on)
    return d - p.usual


def test(b, G, stage, rng, yes, no):
    """no learning: per dialogue, the expectation just before the answer word (YES-word copies minus NO-word ones)"""
    bb = copy.deepcopy(b)
    out = []
    for _ in range(TESTS):
        toks, truth, s = W.sentence(stage, rng)
        fr = W.dialogue(toks, truth, G)
        for j, (cells, phase) in enumerate(fr):
            bb.step(cells, np.zeros(2, bool))
            if fr[j + 1][1] == 'answer':
                e = expect(bb)
                out.append((s, truth, float(e[yes].mean() - e[no].mean())))
                break
    return out


def main():
    path, outp, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
    seed = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    rng = np.random.default_rng(seed)
    b = torch.load(path, weights_only=False, map_location=DEV)
    game_brain.keys_hear_all(b)
    b.pfc.adapt = True
    if os.environ.get('LOGIC_L4_USUAL') == '1':
        b.pfc.c['l4_usual'] = True
    if LOOP_X != 1.0 and not getattr(b, 'loop_scaled', False):
        b.pfc.loop.w *= LOOP_X
        b.loop_scaled = True
    stage = int(os.environ.get('LOGIC_STAGE', 5))
    G = W.glyphs()
    yes, no = word_copies(b, G)
    print(f'listening: stage <= {stage}, loop x{LOOP_X}, l4 usual {b.pfc.c.get("l4_usual", False)}; copies the YES word '
          f'lights {len(yes)}, the NO word {len(no)}', flush=True)
    t, t0 = 0, time.time()
    for i in range(n + 1):
        if i % BLOCK == 0:
            res = test(b, G, stage, np.random.default_rng(1000 + i), yes, no)
            by = collections.defaultdict(list)
            for s, truth, v in res:
                by[s].append((v > 0) == truth)
            print(f'LISTEN {i} dialogues heard | expects the right word: all {np.mean([r for v in by.values() for r in v]):.2f} | '
                  + ' '.join(f'stage {k}: {np.mean(v):.2f}' for k, v in sorted(by.items())) + f' | {time.time() - t0:.0f} s', flush=True)
        if i == n:
            break
        toks, truth, s = W.sentence(stage, rng)
        for cells, phase in W.dialogue(toks, truth, G):
            b.step(cells, np.zeros(2, bool), b.pfc.learn)
            t += 1
            if t % b.c['decay_every'] == 0:
                for tb in b.pfc.plastic():
                    tb.decay()
    torch.save(b, outp)


if __name__ == '__main__':
    main()
