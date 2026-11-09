"""Listening to others (designer 11-09; ARCH 五: the brain hears many dialogues and learns what comes next, no reward):
dialogues of the logic language ("T ? YES", "NOT T ? NO", ...) are shown and the PFC learns as it always does (each
first-layer copy checks what the last layer lit in it, the timing rule). The muscles stay still; nothing is rewarded.
After every block a test without learning: at the frame before the answer word appears, which word does the PFC
expect? Its push into the copies the YES word lights, minus the copies the NO word lights (the copies' drive above
their usual level: what it would think of next).
  python listen.py <brain.pt> <out.pt> <dialogues> [seed]   (LOGIC_STAGE: dialogues of stages up to it, default 5;
                                                             LOGIC_LOOP_X, LOGIC_L4_USUAL, LOGIC_SCALE as in life.py;
                                                             LOGIC_RANGE...: the PFC's range layer, add_range)"""
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
BLOCK = int(os.environ.get('LOGIC_TEST_EVERY', 500))   # dialogues between tests
TESTS = int(os.environ.get('LOGIC_TESTS', 200))        # dialogues per test


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


def usual_input(b, G, n=200):
    """what the copies usually receive: n logic dialogues heard without learning (for calibrating new parts)"""
    bb = copy.deepcopy(b)
    bb.pfc.adapt = False
    X, r = [], np.random.default_rng(11)
    for _ in range(n):
        toks, truth, _ = W.sentence(5, r)
        for cells, _ in W.dialogue(toks, truth, G):
            X.append(bb.step(cells, np.zeros(2, bool)))
    return torch.stack(X)


def add_range(b, G):
    """the PFC's range layer (core/brain.py PFC.setup_range; designer ideas 15, 23, 11-10): LOGIC_RANGE (how many times
    a word's share it holds), LOGIC_RANGE_RATE (a word's share of the layer), LOGIC_RANGE_FADE (its links x this per
    tick), LOGIC_RANGE_RR (their strength vs the input's spread). Calibrated on what the copies usually receive."""
    b.pfc.c.update(pfc_range=float(os.environ['LOGIC_RANGE']),
                   range_rate=float(os.environ.get('LOGIC_RANGE_RATE', 0.005)),
                   range_fade=float(os.environ.get('LOGIC_RANGE_FADE', 0.8)),
                   range_rr=float(os.environ.get('LOGIC_RANGE_RR', 2.0)))
    b.pfc.setup_range(usual_input(b, G), np.random.default_rng(0))
    print(f'PFC range: x{b.pfc.c["pfc_range"]} (lights {b.pfc.k_range:.0f}), a word {b.pfc.c["range_rate"]}, '
          f'fade {b.pfc.c["range_fade"]}, strength {b.pfc.c["range_rr"]}', flush=True)


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
                p = bb.pfc                         # what it would think of next: copies not lit from outside, over
                cand = e.masked_fill(p.L1ext.on.bool(), -1e9)   # theta1, the most driven within the thought budget
                k = int(max(p.budget, 1))
                v, ix = torch.topk(cand, k)
                th = ix[v > p.c['theta1']]
                ny, nn = int(torch.isin(th, yes).sum()), int(torch.isin(th, no).sum())
                out.append((s, truth, float(e[yes].mean() - e[no].mean()), ny - nn))
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
    if os.environ.get('LOGIC_CHECK_STEP'):         # the copies' own check: how much one hearing moves the links (ARCH 六)
        b.c['check_step'] = float(os.environ['LOGIC_CHECK_STEP'])
    if os.environ.get('LOGIC_SIGNED') == '1':      # the learned last -> first layer links may turn inhibitory (ARCH 二.1:
        b.pfc.L41.near.signed = True               # an inhibitory neuron is one whose output weights are negative)
    stage = int(os.environ.get('LOGIC_STAGE', 5))
    G = W.glyphs()
    if os.environ.get('LOGIC_RANGE') and getattr(b.pfc, 'RR', None) is None:
        add_range(b, G)
    yes, no = word_copies(b, G)
    print(f'listening: signed {getattr(b.pfc.L41.near, "signed", False)}, check step {b.c["check_step"]}, stage <= {stage}, loop x{LOOP_X}, l4 usual {b.pfc.c.get("l4_usual", False)}; copies the YES word '
          f'lights {len(yes)}, the NO word {len(no)}', flush=True)
    t, t0 = 0, time.time()
    for i in range(n + 1):
        if i % BLOCK == 0:
            res = test(b, G, stage, np.random.default_rng(1000 + i), yes, no)
            by = collections.defaultdict(list)
            byt = collections.defaultdict(list)
            for s, truth, v, d in res:
                by[s].append((v > 0) == truth)
                byt[s].append(d != 0 and ((d > 0) == truth))   # thinks of the right word (more of its copies)
            print(f'LISTEN {i} dialogues heard | expects the right word: all {np.mean([r for v in by.values() for r in v]):.2f} | '
                  + ' '.join(f'stage {k}: {np.mean(v):.2f}' for k, v in sorted(by.items())) +
                  ' | THINKS the right word: ' + ' '.join(f'stage {k}: {np.mean(v):.2f}' for k, v in sorted(byt.items())) +
                  f' | {time.time() - t0:.0f} s', flush=True)
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
