"""A small invented language of logic, for a brain that answers with its muscles (designer 11-09: not a human language,
simpler; the brain moves its muscles at random, an act that fits the sentence's logic is rewarded, one against it is
punished; two-way choices first, harder ones later, until it can say things with logic and talk with us).

Each token is a fixed random pattern of lit cells on a small sheet. A sentence is shown token by token (TOK frames
each, a blank frame between), it ends with "?", and "?" stays on the sheet during the answer window: the brain answers
by pressing one of two muscles, YES or NO. The world knows the answer (the logic); the brain does not. The first frame
in the window with a press decides (both at once is wrong); no press in the window counts as no answer.

Stages (a stage adds sentences, the earlier ones stay, half of the sentences are of the newest stage):
  1  T ?   F ?                              truth words
  2  NOT T ?                                negation
  3  T AND F ?   F OR T ?                   two-place
  4  A SAME A ?   A SAME B ?                are two names the same (holds the first)
  5  A IS T . A ?                           a name bound to a value, asked later"""
import os

import numpy as np

TOKENS = ('T', 'F', 'NOT', 'AND', 'OR', 'SAME', 'IS', '.', '?', 'A', 'B', 'C', 'D', 'YES', 'NO')   # YES, NO: the answer words
NAMES = ('A', 'B', 'C', 'D')
SCALE = int(os.environ.get('LOGIC_SCALE', 1))       # the sheet (and so the whole brain) grows with it
ROWS, COLS = 12 * SCALE, 16 * SCALE
YES, NO = 0, 1                                 # the two muscles (button copies 0 and 1)
STAGES = 5


def glyphs(seed=0, density=0.3):
    """each token's own fixed pattern"""
    rng = np.random.default_rng(seed)
    return {t: rng.random((ROWS, COLS)) < density for t in TOKENS}


def sentence(stage, rng):
    """a random sentence of this stage or an earlier one: (tokens, the true answer, which stage). The answer is YES
    for half of each kind (otherwise "AND -> always NO" is already right 3 times in 4 and the brain stops there)"""
    s = stage if rng.random() < 0.5 else int(rng.integers(1, stage + 1))
    want = rng.random() < 0.5                                     # the answer this sentence will have
    tv = lambda: 'T' if rng.random() < 0.5 else 'F'            # noqa: E731
    if s == 1:
        x = tv()
        return [x, '?'], x == 'T', s
    if s == 2:
        x = tv()
        return ['NOT', x, '?'], x == 'F', s
    if s == 3:
        op = ('AND', 'OR')[int(rng.integers(2))]
        while True:
            x, y = tv(), tv()
            v = (x == 'T' and y == 'T') if op == 'AND' else (x == 'T' or y == 'T')
            if v == want:
                return [x, op, y, '?'], v, s
    if s == 4:
        a = NAMES[int(rng.integers(len(NAMES)))]
        b = a if want else NAMES[(NAMES.index(a) + 1 + int(rng.integers(len(NAMES) - 1))) % len(NAMES)]
        return [a, 'SAME', b, '?'], a == b, s
    a, x = NAMES[int(rng.integers(len(NAMES)))], tv()
    return [a, 'IS', x, '.', a, '?'], x == 'T', s


def frames(tokens, G, tok=3, window=8, rest=4):
    """the sheet frame by frame: (cells, phase) with phase 'say' (tokens), 'ask' (the answer window), 'rest'"""
    blank = np.zeros((ROWS, COLS), bool)
    out = []
    for t in tokens[:-1]:
        out += [(G[t], 'say')] * tok + [(blank, 'say')]
    out += [(G[tokens[-1]], 'say')] * tok
    out += [(G[tokens[-1]], 'ask')] * window
    out += [(blank, 'rest')] * rest
    return out


def dialogue(tokens, truth, G, tok=3, rest=4):
    """someone asks, someone else answers (designer 11-09 / ARCH 五: the brain listens to others; it learns what comes
    next): the question, then the answer word on the sheet. Phases: 'say', 'answer' (the answer word), 'rest'"""
    blank = np.zeros((ROWS, COLS), bool)
    out = []
    for t in tokens:
        out += [(G[t], 'say')] * tok + [(blank, 'say')]
    out += [(G['YES' if truth else 'NO'], 'answer')] * tok
    out += [(blank, 'rest')] * rest
    return out


def turn(tokens, G, tok=3, window=10, rest=4):
    """a question to the brain, then silence for its answer ('ask'), then rest"""
    blank = np.zeros((ROWS, COLS), bool)
    out = []
    for t in tokens:
        out += [(G[t], 'say')] * tok + [(blank, 'say')]
    return out + [(blank, 'ask')] * window + [(blank, 'rest')] * rest


def own_voice(G):
    """the brain's own voice: each word softer (half of its cells, always the same half)"""
    rng = np.random.default_rng(77)
    return {w: g & (rng.random(g.shape) < 0.5) for w, g in G.items()}


def birth_stream(n_frames, rng, G):
    """frames for the brain's birth (its vision calibrated on what this world shows): sentences of every stage, and
    random presses (the muscles move at random from the start)"""
    C = []
    while len(C) < n_frames:
        toks, _, _ = sentence(STAGES, rng)
        C += [c for c, _ in frames(toks, G)]
    C = np.array(C[:n_frames])
    K = rng.random((n_frames, 2)) < 0.5
    return C, K
