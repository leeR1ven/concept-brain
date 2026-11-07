"""Mario: last layer -> button copies learned tick by tick on the recordings by the input neurons' own check (game_learn.py,
rule 'match'), then measured on levels never trained (NEW) and on held-out runs of trained levels.
What counts is timing, not holding (a held button is easy: it was held a moment ago). Per button, over test frames:
  press AUC    frames where the button is up: does the drive rank the frames where it goes down next above the others
  ahead AUC    the same with "goes down within the next 10 frames"
  release AUC  frames where the button is down: the frames where it comes up next
AUC 0.5 = no idea, 1.0 = always ranks right. The drive of a button = its press copies minus its stop copies.
usage: python mario_learn.py <net.pt> <rec dir> <cells dir> <out.npz> [train episodes limit]"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import game_learn as G                            # noqa: E402
from game_brain import UNITS_PER_KEY as U         # noqa: E402
from mario_build import NEW, SEEN_TRY, episodes   # noqa: E402

NAMES = ('A jump', 'B run', 'left', 'right', 'down')


def auc(score, pos):
    pos = np.asarray(pos, bool)
    if pos.all() or not pos.any():
        return float('nan')
    r = np.argsort(np.argsort(score, kind='stable'), kind='stable') + 1.0
    n1 = pos.sum()
    return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(pos) - n1)))


def drives(W, A, nk):
    D = np.array([W[r].sum(0) if len(r) else np.zeros(W.shape[1], np.float32) for r in A])
    return np.stack([D[:, k * U:(k + 1) * U].mean(1) - D[:, (nk + k) * U:(nk + k + 1) * U].mean(1) for k in range(nk)], 1)


def measure(W, eps, rec, nv, label):
    nk = W.shape[1] // (2 * U)
    S, H, H1, AH = [], [], [], []
    for name in eps:
        A, K = G.ticks(rec, name, nv)
        held = K[:, :nk * U].reshape(len(K), nk, U).any(2)
        T = len(A)
        s = drives(W, A[:T - 1], nk)
        nxt = held[1:]
        ahead = np.zeros_like(held[:T - 1])
        for d in range(1, 11):                       # goes down within the next 10 frames
            ahead[:T - d] |= held[d:] & ~held[:T - d]
        S.append(s[10:])
        H.append(held[:T - 1][10:])
        H1.append(nxt[10:])
        AH.append(ahead[10:])
    S, H, H1, AH = (np.concatenate(v) for v in (S, H, H1, AH))
    print(f'{label}: {len(S)} frames', flush=True)
    for k, nm in enumerate(NAMES[:nk]):
        up, down = ~H[:, k], H[:, k]
        print(f'   {nm:7s} held {100 * H1[:, k].mean():4.1f}% | press AUC {auc(S[up, k], H1[up, k]):.3f}  ahead AUC'
              f' {auc(S[up, k], AH[up, k]):.3f}  release AUC {auc(-S[down, k], ~H1[down, k]):.3f}', flush=True)


def main():
    net, rec, src, out = sys.argv[1:5]
    limit = int(sys.argv[5]) if len(sys.argv) > 5 else None
    nv = torch.load(net, weights_only=False).ch.n_vision
    eps = episodes(src)
    held_out = {}
    for name, lv in eps:                             # one run of each tried training level is kept out of training
        if lv in SEEN_TRY and lv not in held_out:
            held_out[lv] = name
    train = [n for n, lv in eps if lv not in NEW and n not in held_out.values()]
    rng = np.random.default_rng(1)
    train = [train[i] for i in rng.permutation(len(train))][:limit]
    test_new = [n for n, lv in eps if lv in NEW]
    test_seen = list(held_out.values())
    W, held, frames = None, 0, 0
    for i, name in enumerate(train):
        A, K = G.ticks(rec, name, nv)
        nk = K.shape[1] // (2 * U)
        held = held + K[:, :nk * U].reshape(len(K), nk, U).any(2).sum(0)
        frames += len(K)
        if W is None:
            W = np.zeros((int(np.load(f'{rec}/{name}.npz')['n']), K.shape[1]), np.float32)
        W = G.learn(W, A, K, 'match', G.ETA_D)
        if i % 20 == 19 or i == len(train) - 1:
            print('trained on', i + 1, 'runs', flush=True)
            if i % 100 == 99 or i == len(train) - 1:
                measure(W, test_new, rec, nv, f'  after {i + 1} runs, NEW levels {NEW}')
    measure(W, test_seen, rec, nv, f'held-out runs of trained levels {SEEN_TRY}')
    rate = held / frames                             # how often each button is held: its own firing share in play
    print('buttons held share', dict(zip(NAMES, np.round(rate, 3))), flush=True)
    np.savez(out, w=W, n_vision=nv, train=np.array(train), rate=rate)


if __name__ == '__main__':
    main()
