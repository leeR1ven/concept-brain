"""The real thing instead of the shortcut: last-layer -> key-copy links learned tick by tick on the recording, by the designer's
rule (11-04), then used to play.
  hebb  A -> B grows when B comes on while A was on one or two ticks before (A just lit, held, or just off); A and B coming on
        together: nothing; B before A weakens: A comes on while B was on, or B goes off while A is still on (so an A that stays
        on through B's whole visit nets to nothing). Growth slows as a link gets bigger (concave, ARCH 四); one global decay.
  match the input neuron's own check (perceptron): the links from what was lit grow when B came on from outside but the
        last layer did not light it, and weaken when the last layer lit B but it did not come on; nothing when it was right.
  delta prediction error: A -> B moves toward what B actually did (the theory's version; its limit is the ridge readout).
The key copies are the targets: press and stop for right and left, 24 copies each.
usage: python game_learn.py <net.pt> <rec dir> <cells dir> <rule> <out.npz> <train trials (comma)> <test trials (comma)>"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from core.config import config                  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from game_brain import UNITS_PER_KEY            # noqa: E402

C = config()
ETA, W_REF, FLOOR, GAMMA, EVERY = C['eta'], C['w_ref'], C['floor'], C['gamma_pfc'], C['decay_every']
MARGIN = float(os.environ.get('LEARN_MARGIN', 2.0))
ETA_D = float(os.environ.get('LEARN_ETA', 0.03))     # step of the own-check and delta rules (per tick, shared by the lit units)


def plast(w):
    return np.maximum((1.0 + w / W_REF) ** -1.0, FLOOR)


def ticks(rec, tr, n_vision):
    r = np.load(f'{rec}/{tr}.npz')
    a, ao, x, xo = r['a'], r['a_off'], r['x'], r['x_off']
    A = [a[ao[t]:ao[t + 1]].astype(np.int64) for t in range(len(ao) - 1)]
    K = np.zeros((len(xo) - 1, int(r['n']) - n_vision), bool)
    for t in range(len(xo) - 1):
        k = x[xo[t]:xo[t + 1]].astype(np.int64)
        k = k[k >= n_vision] - n_vision
        K[t, k] = True
    return A, K


def learn(W, A, K, rule, eta_d):
    for t in range(2, len(A)):
        rose_b = K[t] & ~K[t - 1]
        if rule == 'hebb':
            before = np.union1d(A[t - 1], A[t - 2])                 # A lit one or two ticks before B
            if rose_b.any():
                cols = np.nonzero(rose_b)[0]
                ix = np.ix_(before, cols)
                w = W[ix]
                W[ix] = w + ETA * plast(w)
            a_rose = np.setdiff1d(A[t], A[t - 1])                    # A comes on while B was already on: B before A
            cols = np.nonzero(K[t - 1])[0]
            if len(a_rose) and len(cols):
                ix = np.ix_(a_rose, cols)
                w = W[ix]
                W[ix] = np.maximum(w - 0.5 * ETA * plast(w), 0.0)
            fell = np.nonzero(K[t - 1] & ~K[t])[0]                   # B goes off while A is still on
            if len(fell):
                still = np.intersect1d(A[t], A[t - 1])
                ix = np.ix_(still, fell)
                w = W[ix]
                W[ix] = np.maximum(w - ETA * plast(w), 0.0)
        elif rule == 'match':                     # the input neuron's own check: lit from outside but not by the last
            rows = A[t - 1]                         # layer -> grow; lit by the last layer but not from outside -> weaken
            if len(rows):
                d = W[rows].sum(0)
                miss = K[t] & (d < 1.0 + MARGIN)
                false = ~K[t] & (d > 1.0 - MARGIN)
                fix = miss.astype(np.float32) - false.astype(np.float32)
                if fix.any():
                    W[rows] += (eta_d / len(rows)) * fix
        else:
            rows = A[t - 1]
            if len(rows):
                y = W[rows].sum(0)
                W[rows] += eta_d / len(rows) * (K[t].astype(np.float32) - y)
        if t % EVERY == 0:
            W -= (1.0 - GAMMA) * W * (plast(np.maximum(W, 0)) if rule == 'hebb' else 1.0)
    return W


def decide(W, A, theta):
    """per tick: right, left or nothing, from the drive into press minus stop copies of each key"""
    U = UNITS_PER_KEY
    out = []
    for rows in A:
        d = W[rows].sum(0)
        s = np.array([d[0:U].sum() - d[2 * U:3 * U].sum(), d[U:2 * U].sum() - d[3 * U:4 * U].sum()])
        k = int(np.argmax(s))
        out.append(k if s[k] > theta else 2)
    return np.array(out)


def truth(K):
    U = UNITS_PER_KEY
    r, l = K[:, 0:U].any(1), K[:, U:2 * U].any(1)
    return np.where(r, 0, np.where(l, 1, 2))


def main():
    net, rec, src, rule, out, train, test = sys.argv[1:8]
    b = torch.load(net, weights_only=False)
    nv, n = b.ch.n_vision, b.pfc.n
    W = np.zeros((n, 4 * UNITS_PER_KEY), np.float32)
    for i, tr in enumerate(train.split(',')):
        A, K = ticks(rec, tr, nv)
        W = learn(W, A, K, rule, ETA_D)
        if i % 10 == 9 or i == len(train.split(',')) - 1:
            print('trained on', i + 1, 'games; weight sum', float(W.sum()), 'max', float(W.max()), flush=True)
    A, K = [], []
    for tr in test.split(','):
        a, k = ticks(rec, tr, nv)
        A += a[2:-1]
        K.append(k[3:])
    K = np.concatenate(K)
    want = truth(K)                                                   # the keys one tick after the last layer read
    U = UNITS_PER_KEY
    sr = np.array([(lambda d: (d[0:U].sum() - d[2 * U:3 * U].sum()) - (d[U:2 * U].sum() - d[3 * U:4 * U].sum()))(W[r].sum(0)) for r in A])
    mv = want != 2
    print(f'when the player moves, right or left told right: {100 * float(((sr[mv] > 0) == (want[mv] == 0)).mean()):.1f}%', flush=True)
    best = None
    for q in (0.5, 0.7, 0.8, 0.9):
        U = UNITS_PER_KEY
        S = np.array([[W[r].sum(0)[0:U].sum() - W[r].sum(0)[2 * U:3 * U].sum(), W[r].sum(0)[U:2 * U].sum() - W[r].sum(0)[3 * U:4 * U].sum()]
                      for r in A[::7]]).max(1)
        theta = float(np.quantile(S, q))
        got = decide(W, A, theta)
        moved = want != 2
        acc = float((got == want).mean())
        side = float((got[moved] == want[moved]).mean())
        print(f'theta at quantile {q}: all ticks right {100 * acc:.1f}% (always nothing {100 * float((want == 2).mean()):.1f}%),'
              f' when the player moves: right side {100 * side:.1f}%, moves {100 * float((got != 2).mean()):.1f}% of ticks', flush=True)
        if best is None or acc > best[0]:
            best = (acc, theta)
    U = UNITS_PER_KEY                                                 # play: move as often as the player did
    S = np.array([(lambda d: max(d[0:U].sum() - d[2 * U:3 * U].sum(), d[U:2 * U].sum() - d[3 * U:4 * U].sum()))(W[r].sum(0)) for r in A[::3]])
    theta = float(np.quantile(S, 1.0 - float((want != 2).mean())))
    got = decide(W, A, theta)
    print(f'theta moving as often as the player ({100 * float((want != 2).mean()):.1f}%): when the player moves, right side'
          f' {100 * float((got[want != 2] == want[want != 2]).mean()):.1f}%', flush=True)
    np.savez(out, w=W, theta=theta, n_vision=nv)


if __name__ == '__main__':
    main()
