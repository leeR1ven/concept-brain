"""Closed loop: the readout from game_fit.py plays the real game, one tick per game frame. The keys it presses go back in as copies.
usage: python game_play.py <net.pt> <weights.npz> <kind> <games> <first seed>     kind: truth, pixels, copies, last_layer, last_layer_seen, learned"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from game_brain import COLS, ROWS, key_bits     # noqa: E402
from game_brain import DEV                        # noqa: E402
from game_data import locate, to_cells          # noqa: E402
from game_env import LEFT, NOOP, RIGHT, Breakout  # noqa: E402
from game_fit import BINS                       # noqa: E402

torch.set_num_threads(1)
ARGMAX = os.environ.get('GAME_POLICY') == 'argmax'
SPEED = float(os.environ.get('GAME_SPEED', 0.0))     # cells the paddle moves per tick a key is held (for GAME_OWN)
SIDE = os.environ.get('GAME_POLICY') == 'side'      # right minus left; move when it leans clearly, the bar set by the key
MOVE = float(os.environ.get('GAME_MOVE', 0.6))
CENTER = os.environ.get('GAME_CENTER') == '1'      # lean measured from its own recent mean      # neurons' own rate: they fire on about this share of ticks
LAG = int(os.environ.get('GAME_LAG', 0))
OWN = os.environ.get('GAME_OWN') == '1'             # check only: add the paddle's own moves since then (efference copy)           # check readouts only: see the ball and paddle as they were LAG ticks ago
FEAT = os.environ.get('GAME_FEAT')                  # learned links from a check feature (columns) instead of the last layer
TRACE = os.environ.get('GAME_TRACE')             # file: per tick ball x, paddle x, P right, P left, action


def main():
    net, wf, kind, games, seed = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
    z = np.load(wf)
    if kind == 'learned':                              # links the model grew itself (game_learn.py): last layer -> key copies
        U = z['w'].shape[1] // 4
        P_of = np.stack([np.r_[np.ones(U), np.zeros(U), -np.ones(U), np.zeros(U)],      # press right minus stop right
                         np.r_[np.zeros(U), np.ones(U), np.zeros(U), -np.ones(U)]], 1)  # press left minus stop left
        w, use, theta = np.vstack([z['w'] @ P_of, np.zeros((1, 2))]), np.arange(z['w'].shape[0]), float(z['theta'])
        if os.environ.get('GAME_THETA') is not None:     # e.g. 0: press when the press neurons are driven more than the stop neurons
            theta = float(os.environ['GAME_THETA'])
    else:
        w, use, theta = z[f'{kind}_w'], z[f'{kind}_use'], float(z[f'{kind}_theta'])
    if kind == 'learned' and FEAT:
        kind = FEAT
    N = {'truth': BINS, 'pixels': ROWS * COLS, 'columns': 2 * COLS}.get(kind)
    b0 = torch.load(net, weights_only=False, map_location=DEV) if kind in ('copies', 'last_layer', 'last_layer_seen', 'learned') else None
    if b0 is not None:
        N = b0.pfc.n
    row = np.full(N, -1)
    row[use] = np.arange(len(use))
    bias = w[-1]
    for g in range(games):
        e = Breakout(seed + g)
        f = e.reset()
        b = torch.load(net, weights_only=False, map_location=DEV) if b0 is not None else None
        keys = np.zeros(2, bool)
        ticks, moved = 0, 0
        trace = []
        lean = []
        raw = []
        seen = []
        did = []
        for t in range(20000):
            cells = to_cells(f)
            seen.append(locate(f))
            if kind == 'truth':
                bx, _, px = seen[max(0, len(seen) - 1 - LAG)]
                if OWN and px >= 0 and LAG:
                    px = px + SPEED * sum(did[-LAG:])
                on = np.zeros(0, int) if bx < 0 or px < 0 else np.array([int(np.clip(np.round((bx - px) / 2.0), -6, 6)) + 6])
            elif kind == 'columns':
                bx, _, px = seen[max(0, len(seen) - 1 - LAG)]
                on = np.array(([int(np.clip(bx, 0, COLS - 1))] if bx >= 0 else []) + ([COLS + int(np.clip(px, 0, COLS - 1))] if px >= 0 else []), int)
            elif kind == 'pixels':
                on = np.nonzero(cells.ravel())[0]
            else:
                x = b.step(cells, keys)
                on = torch.nonzero(b.pfc.L4.on if kind != 'copies' else x).squeeze(1).numpy()
            r = row[on]
            P = w[r[r >= 0]].sum(0) + bias
            if SIDE:
                s_ = float(P[0] - P[1])
                raw.append(s_)
                if CENTER:                                 # each key neuron's own usual drive taken out (ARCH 二.4)
                    s_ -= float(np.mean(raw[-300:]))
                lean.append(abs(s_))
                bar = float(np.quantile(lean[-300:], 1.0 - MOVE)) if len(lean) > 30 else 0.0
                a = NOOP if abs(s_) <= bar else (RIGHT if s_ > 0 else LEFT)
            elif ARGMAX:                                 # right, left, or neither: whichever the readout rates highest
                k = int(np.argmax([P[0], P[1], 1.0 - P[0] - P[1]]))
                a = (RIGHT, LEFT, NOOP)[k]
            else:
                k = int(np.argmax(P))
                a = NOOP if P[k] < theta else (RIGHT if k == 0 else LEFT)
            moved += a != NOOP
            did.append({NOOP: 0, RIGHT: 1, LEFT: -1}[a])
            if TRACE:
                trace.append((*locate(f)[::2], P[0], P[1], a))
            ticks += 1
            keys = key_bits({NOOP: 0, RIGHT: 3, LEFT: 4}[a])
            f, over = e.step(a)
            if over:
                break
        if TRACE:
            np.save(f'{TRACE}_{seed + g}.npy', np.array(trace, np.float32))
        print('GAME', kind, seed + g, 'score', e.score, 'ticks', ticks, 'moved', round(moved / ticks, 2), flush=True)


if __name__ == '__main__':
    main()
