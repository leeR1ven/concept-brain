"""What does the brain think of by itself? (designer 11-09: can it think up new content in the logic language?) After
listening, the sheet goes quiet, or shows only a start (some words); the PFC runs on. Each tick the copies lit from
inside (thoughts) are read: the word whose own copies take the biggest share of the thoughts, if clearly above the
others. Nothing learns while it thinks.
  python think.py <brain.pt> <start words, e.g. "T ?" or "-"> [ticks] [runs]   (LOGIC_SCALE as when it was built)"""
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'game'))
import game_brain                                  # noqa: E402
from game_brain import DEV                         # noqa: E402
import world as W                                  # noqa: E402


def own_copies(b, G):
    """each word's own copies: lit by it (most of the time it is shown) and by at most two other words"""
    nv = b.ch.n_vision
    blank = np.zeros((W.ROWS, W.COLS), bool)
    rate = {}
    for w in W.TOKENS:
        b.reset()
        on = torch.zeros(b.ch.n, device=DEV)
        for _ in range(10):
            for cells in [blank] * 4 + [G[w]] * 3:
                b.step(cells, b.still())
            on += b.pfc.L1ext.on.float()
        rate[w] = (on[:nv] / 10) > 0.5
    lit_by = torch.stack(list(rate.values())).sum(0)
    return {w: torch.nonzero(r & (lit_by <= 3)).squeeze(1) for w, r in rate.items()}


def main():
    path, start = sys.argv[1], sys.argv[2]
    ticks = int(sys.argv[3]) if len(sys.argv) > 3 else 40
    runs = int(sys.argv[4]) if len(sys.argv) > 4 else 3
    b0 = torch.load(path, weights_only=False, map_location=DEV)
    b0.pfc.adapt = False                           # nothing changes while we look
    if os.environ.get('LOGIC_L4_USUAL') == '1':
        b0.pfc.c['l4_usual'] = True
    G = W.glyphs()
    import copy
    words = own_copies(copy.deepcopy(b0), G)
    print('own copies per word: ' + ' '.join(f'{w}:{len(ix)}' for w, ix in words.items()), flush=True)
    blank = np.zeros((W.ROWS, W.COLS), bool)
    toks = [] if start == '-' else start.split()
    for r in range(runs):
        b = copy.deepcopy(b0)
        rng = np.random.default_rng(r)
        for _ in range(int(rng.integers(5, 15))):  # some quiet before
            b.step(blank, b.still())
        for t in toks:
            for cells in [G[t]] * 3 + [blank]:
                b.step(cells, b.still())
        seq = []
        for k in range(ticks):
            b.step(blank, b.still())
            th = b.pfc.x_int
            n = max(int(th.sum()), 1)
            share = {w: int(th[ix].sum()) / n for w, ix in words.items()}   # its share of this tick's thoughts
            best = sorted(share.items(), key=lambda kv: -kv[1])
            w, s = best[0]
            seq.append(f'{w}({s:.0%})' if s >= 0.2 and s >= 1.5 * best[1][1] else '_')   # _: no clear word
        print(f'start "{start}", run {r}: thinks ' + ' '.join(seq), flush=True)


if __name__ == '__main__':
    main()
