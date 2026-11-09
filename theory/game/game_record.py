"""The newborn brain watches one recorded trial, one tick per game frame; nothing learns. Saved per tick: which last-layer units
and which copies are lit. Trials can run in parallel: the brain is fixed.
usage: python game_record.py <net.pt> <cells dir> <trial> <out dir>"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from game_brain import key_bits                 # noqa: E402
from game_brain import DEV                        # noqa: E402

torch.set_num_threads(1)


def main():
    net, src, trial, out = sys.argv[1:5]
    b = torch.load(net, weights_only=False, map_location=DEV)
    z = np.load(f'{src}/{trial}.npz')
    cells, action = z['cells'], z['action']
    if 'shape' in z:                                # packed bits (mario_data.py)
        cells = np.unpackbits(cells, axis=-1)[..., :int(z['shape'][-1])].astype(bool)
    p = b.pfc
    A, X = [], []
    keys = b.keys_of(0) if hasattr(b, 'keys_of') else np.zeros(2, bool)
    keys[:] = False
    for t in range(len(action)):
        x = b.step(cells[t], keys)
        keys = b.keys_of(action[t]) if hasattr(b, 'keys_of') else key_bits(action[t])
        A.append(torch.nonzero(p.L4.on).squeeze(1).numpy().astype(np.uint16))
        X.append(torch.nonzero(x).squeeze(1).numpy().astype(np.uint16))
        if t % 3000 == 0:
            print(trial, t, 'last layer', len(A[-1]), 'copies', len(X[-1]), 'inside', int(p.x_int.sum()), flush=True)
    off = lambda L: np.cumsum([0] + [len(v) for v in L])
    np.savez(f'{out}/{trial}.npz', a=np.concatenate(A), a_off=off(A), x=np.concatenate(X), x_off=off(X), n=p.n)
    print('done', trial, flush=True)


if __name__ == '__main__':
    main()
