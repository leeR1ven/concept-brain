"""The Mario brain: the same GameBrain (vision area by the one generator, chain copies, keys as copies with press and stop
neurons, the speech model's PFC), here with 52 x 64 cells and five buttons (A, B, left, right, down). Built once from a
birth period of frames spread over the training levels; every later run loads it.
Levels never seen in training (the test of generalization): NEW. Training levels also tried in play: SEEN_TRY.
usage: python mario_build.py <cells dir> <net.pt> [birth frames]"""
import glob
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import game_brain                                  # noqa: E402  (so the saved brain loads from any script)
from game_brain import GameBrain, mario_bits       # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from core.config import config                     # noqa: E402

NEW = ('1-1', '3-1', '6-1')
SEEN_TRY = ('2-1', '4-1', '5-1')


def episodes(src):
    """(name, level) of every episode, sorted"""
    out = []
    for p in sorted(glob.glob(f'{src}/*.npz')):
        z = np.load(p)
        out.append((os.path.basename(p)[:-4], str(z['level'])))
    return out


def load(src, name):
    z = np.load(f'{src}/{name}.npz')
    cells = np.unpackbits(z['cells'], axis=-1)[..., :int(z['shape'][-1])].astype(bool)
    return cells, z['action']


def main():
    src, out = sys.argv[1], sys.argv[2]
    n_birth = int(sys.argv[3]) if len(sys.argv) > 3 else 3000
    eps = [e for e in episodes(src) if e[1] not in NEW]
    rng = np.random.default_rng(0)
    pick = rng.choice(len(eps), 30, replace=False)
    C, K = [], []
    for i in pick:                                 # a stretch from each of 30 training episodes
        cells, act = load(src, eps[i][0])
        s = int(rng.integers(0, max(1, len(act) - 3 * n_birth // 30)))
        sl = slice(s, s + 3 * n_birth // 30, 3)
        keys = np.array([mario_bits(a) for a in act])
        C.append(cells[sl])
        K.append(np.concatenate([np.zeros((1, keys.shape[1]), bool), keys[:-1]])[sl])
    C, K = np.concatenate(C), np.concatenate(K)
    c = config(s=1, dz=(2, 4, 8, 16), levels_per_s=4, r_learn=0.01)
    torch.set_num_threads(8)
    b = GameBrain(c, 0, C, K, game='mario')
    print('birth frames', len(C), 'vision units', b.area.n, 'feature units', [len(i) for i in b.ch.idx], 'copies', b.ch.n,
          flush=True)
    torch.save(b, out)
    print('saved', os.path.getsize(out) // 1000000, 'MB')


if __name__ == '__main__':
    game_brain.main = main                         # keep pickles pointing at the game_brain module
    main()
