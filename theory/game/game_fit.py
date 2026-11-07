"""Can a readout from each kind of activity tell what the human's next key is, and which side of the paddle the ball is on?
Regression, one row per tick, trained on most trials, the strength chosen on a validation trial, scored on trials never used.
Activity compared: truth (ball minus paddle, a perfect eye: the check that this whole test can work at all), pixels, copies, last layer.
Saves the weights for game_play.py.   usage: python game_fit.py <cells dir> <rec dir> <out.npz> [delays]"""
import glob
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from stream_fit import Rows, chunks             # noqa: E402  (stream_fit sets torch to 4 threads; reset below)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from game_brain import COLS, ROWS, key_bits     # noqa: E402

torch.set_num_threads(16)
VAL = tuple(os.environ.get('GAME_VAL', '134').split(','))
TEST = tuple(os.environ.get('GAME_TEST', '153,205,490').split(','))
TRAIN_LIMIT = int(os.environ.get('GAME_TRAIN', '0'))          # use only this many training trials (0: all), for learning curves
LAM = (30.0, 300.0, 3000.0, 30000.0)
BINS = 13


class Lean(Rows):
    """Rows kept as int32, built straight from the per-trial lists"""

    def __init__(self, parts, n):
        self.ix = np.concatenate([p[0] for p in parts]).astype(np.int32)
        offs, base = [np.zeros(1, np.int64)], 0
        for ix, off in parts:
            offs.append(off[1:].astype(np.int64) + base)
            base += len(ix)
        self.off, self.n, self.t = np.concatenate(offs), n, len(np.concatenate(offs)) - 1


def from_list(L, n):
    return (np.concatenate(L).astype(np.int32), np.cumsum([0] + [len(v) for v in L]))


def truth_rows(bx, px):
    """one active unit: how far the ball is from the paddle, in 13 bins (no unit when the ball is out of sight)"""
    L = []
    for b, p in zip(bx, px):
        if b < 0 or p < 0:
            L.append(np.zeros(0, np.int64))
        else:
            L.append(np.array([int(np.clip(np.round((b - p) / 2.0), -6, 6)) + 6]))
    return from_list(L, BINS)


def column_rows(bx, px):
    """two active units: the ball's column and the paddle's column, each on its own, nothing about the two together"""
    L = []
    for b, p in zip(bx, px):
        L.append(np.array(([int(np.clip(b, 0, COLS - 1))] if b >= 0 else []) + ([COLS + int(np.clip(p, 0, COLS - 1))] if p >= 0 else []), np.int64))
    return from_list(L, 2 * COLS)


def main():
    src, rec, out = sys.argv[1:4]
    delays = tuple(int(d) for d in sys.argv[4].split(',')) if len(sys.argv) > 4 else (1, 2, 4, 8)
    trials = sorted(os.path.basename(p)[:-4] for p in glob.glob(f'{rec}/*.npz'))
    if TRAIN_LIMIT:
        train = [tr for tr in trials if tr not in VAL + TEST][:TRAIN_LIMIT]
        trials = [tr for tr in trials if tr in train or tr in VAL + TEST]
    Y, S, tag, parts = [], [], [], {'truth': [], 'pixels': [], 'columns': []}
    for tr in trials:
        z = np.load(f'{src}/{tr}.npz')
        T = len(z['action'])
        parts['pixels'].append(from_list([np.nonzero(c.ravel())[0] for c in z['cells']], ROWS * COLS))
        parts['truth'].append(truth_rows(z['ball_x'], z['paddle_x']))
        parts['columns'].append(column_rows(z['ball_x'], z['paddle_x']))
        Y.append(np.array([key_bits(a) for a in z['action']]))
        d = z['ball_x'] - z['paddle_x']
        seen = (z['ball_x'] >= 0) & (z['paddle_x'] >= 0) & (np.abs(d) > 1)
        S.append(np.where(seen, (d > 0) * 1 - (d < 0) * 1, 0))
        tag.append(np.full(T, 0 if tr not in VAL + TEST else 1 if tr in VAL else 2))
    n = int(np.load(f'{rec}/{trials[0]}.npz')['n'])
    Y, S, tag = np.concatenate(Y).astype(np.float32), np.concatenate(S), np.concatenate(tag)
    N = len(Y)

    def feature(name):
        if name in ('truth', 'pixels', 'columns'):
            return Lean(parts[name], {'truth': BINS, 'pixels': ROWS * COLS, 'columns': 2 * COLS}[name])
        key = 'a' if name == 'last layer' else 'x'
        out = []
        for tr in trials:
            r = np.load(f'{rec}/{tr}.npz')
            out.append((r[key].astype(np.int32), r[key + '_off']))
        return Lean(out, n)
    print('ticks', N, 'train', int((tag == 0).sum()), 'val', int((tag == 1).sum()), 'test', int((tag == 2).sum()),
          'key held', float(Y.any(1).mean()), flush=True)
    target = np.stack([Y[:, 0] - Y[:, 1], (S > 0) * 1.0 - (S < 0) * 1.0, Y[:, 0], Y[:, 1]], 1).astype(np.float32)  # key side, ball side, right, left
    tt = torch.from_numpy(target)
    best = {}
    n_vision = torch.load(os.environ['GAME_NET'], weights_only=False).ch.n_vision if 'GAME_NET' in os.environ else n
    for name in os.environ.get('GAME_KINDS', 'truth,pixels,copies,last layer,last layer seen').split(','):
        F = feature(name.replace(' seen', ''))
        cnt = np.bincount(F.ix, minlength=F.n)
        use = np.nonzero(cnt > 0)[0]
        if name.endswith('seen'):                  # only the part made from what is seen, not from the keys pressed before
            use = use[use < n_vision]
        m = len(use)
        G = torch.zeros(m + 1, m + 1)
        tr_rows = np.nonzero(tag == 0)[0]
        tr_rows = tr_rows[tr_rows >= 16]

        def dense(rows):
            D = torch.from_numpy(F.dense(rows, use))
            return torch.cat([D, torch.ones(len(rows), 1)], 1)
        for c in chunks(tr_rows):
            D = dense(c)
            G += D.T @ D
        print(name, 'active units', m, flush=True)
        B = {}
        for dl in delays:
            B[dl] = torch.zeros(m + 1, target.shape[1])
            for c in chunks(tr_rows):
                B[dl] += dense(c - dl).T @ tt[c]
        for lam in LAM:
            G.diagonal().add_(lam)
            L = torch.linalg.cholesky(G)
            G.diagonal().sub_(lam)
            for dl in delays:
                W = torch.cholesky_solve(B[dl], L)
                res = {}
                for tg, label in ((1, 'val'), (2, 'test')):
                    rows = np.nonzero(tag == tg)[0]
                    rows = rows[rows >= 16]
                    P = torch.cat([dense(c - dl) @ W for c in chunks(rows)]).numpy()
                    moved = target[rows, 0] != 0
                    side = target[rows, 1] != 0
                    res[label] = (float((np.sign(P[moved, 0]) == target[rows, 0][moved]).mean()),
                                  float((np.sign(P[side, 1]) == target[rows, 1][side]).mean()))
                print(f'  {name:11s} delay {dl} lambda {lam:g}: key side val {100 * res["val"][0]:.1f}% test {100 * res["test"][0]:.1f}%;'
                      f' ball side val {100 * res["val"][1]:.1f}% test {100 * res["test"][1]:.1f}%', flush=True)
                if dl == delays[0] and (name not in best or res['val'][0] > best[name][0]):
                    rows = np.nonzero(tag == 1)[0]
                    rows = rows[rows >= 16]
                    P = torch.cat([dense(c - dl) @ W[:, 2:] for c in chunks(rows)]).numpy()
                    theta = float(np.quantile(P.max(1), 1 - float(target[rows, 2:].max(1).mean())))     # moves as often as the human
                    best[name] = (res['val'][0], lam, W[:, 2:].numpy(), use, theta)
            del L
        del G, F
    np.savez(out, **{f'{k.replace(" ", "_")}_w': v[2] for k, v in best.items()}, **{f'{k.replace(" ", "_")}_use': v[3] for k, v in best.items()},
             **{f'{k.replace(" ", "_")}_lam': v[1] for k, v in best.items()}, **{f'{k.replace(" ", "_")}_theta': v[4] for k, v in best.items()})


if __name__ == '__main__':
    main()
