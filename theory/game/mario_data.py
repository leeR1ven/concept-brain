"""Super Mario Bros human play (rafaelcp/smbdataset: one player, 60 frames a second, every frame a PNG named with the buttons
held) -> per episode: cells [T, 52, 64] (4x4-pixel cells below the score bar: is anything there that is not the background
colour), buttons [T] (the dataset's 8-bit code), level, win.
usage: python -I mario_data.py <raw dir> <out dir> <worker> <workers>"""
import os
import re
import sys

import numpy as np
from PIL import Image

TOP, CELL = 32, 4                      # the score bar above row 32 is left out
ROWS, COLS = (240 - TOP) // CELL, 256 // CELL
BUTTON = {'A': 128, 'up': 64, 'left': 32, 'B': 16, 'start': 8, 'right': 4, 'down': 2, 'select': 1}


def to_cells(a):
    """a: palette-index frame [240, 256]; the commonest colour is the background"""
    a = a[TOP:]
    vals, cnt = np.unique(a, return_counts=True)
    on = a != vals[np.argmax(cnt)]
    return on.reshape(ROWS, CELL, COLS, CELL).any(axis=(1, 3))


def main():
    raw, out, k, m = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    os.makedirs(out, exist_ok=True)
    for i, d in enumerate(sorted(os.listdir(raw))):
        if i % m != k:
            continue
        name = re.match(r'Rafael_([a-z0-9]+)_e(\d+)_(\d-\d)_(win|fail)', d)
        fs = []
        for f in os.listdir(os.path.join(raw, d)):
            g = re.search(r'_f(\d+)_a(\d+)_', f)
            fs.append((int(g.group(1)), int(g.group(2)), f))
        fs.sort()
        C = np.array([to_cells(np.array(Image.open(os.path.join(raw, d, f)))) for _, _, f in fs])
        A = np.array([a for _, a, _ in fs], np.uint8)
        tag = f'{name.group(1)}_e{int(name.group(2)):03d}'
        np.savez_compressed(f'{out}/{tag}.npz', cells=np.packbits(C, axis=-1), action=A, level=name.group(3),
                            win=name.group(4) == 'win', shape=np.array(C.shape))
        print(tag, name.group(3), name.group(4), len(A), 'cells lit %.0f' % C.reshape(len(C), -1).sum(1).mean(), flush=True)


if __name__ == '__main__':
    main()
