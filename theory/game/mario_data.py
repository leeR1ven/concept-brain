"""Super Mario Bros human play (rafaelcp/smbdataset: one player, 60 frames a second, every frame a PNG named with the buttons
held) -> per episode: cells [T, 52, 64] (4x4-pixel cells below the score bar: is anything there that is not the background
colour), buttons [T] (the dataset's 8-bit code), level, win.
MARIO_GAZE=1: the eye looks at Mario (a stand-in for an eye that fixates the self, found here by his red; the PFC is to
drive the eye later): the same 52 x 64 window, centred on him, empty outside the screen; MARIO_VIEW=2: a window twice
as wide and high (the whole screen always in view).
usage: python -I mario_data.py <raw dir> <out dir> <worker> <workers>"""
import os
import re
import sys

import numpy as np
from PIL import Image

TOP, CELL = 32, 4                      # the score bar above row 32 is left out
ROWS, COLS = (240 - TOP) // CELL, 256 // CELL
GAZE = os.environ.get('MARIO_GAZE') == '1'
VIEW = int(os.environ.get('MARIO_VIEW', '1'))  # the eye's window, x the screen each way (2: the whole screen wherever he is)
RED = (216, 40, 0)                     # Mario's cap and shirt in the dataset's palette (the emulator's is (248, 56, 0))
BUTTON = {'A': 128, 'up': 64, 'left': 32, 'B': 16, 'start': 8, 'right': 4, 'down': 2, 'select': 1}


def to_cells(a):
    """a: palette-index frame [240, 256]; the commonest colour is the background"""
    a = a[TOP:]
    vals, cnt = np.unique(a, return_counts=True)
    on = a != vals[np.argmax(cnt)]
    return on.reshape(ROWS, CELL, COLS, CELL).any(axis=(1, 3))


class Eye:
    """where Mario is (pixels below the score bar): the centre of his red near where he was, else anywhere, else as before"""

    def __init__(self):
        self.x, self.y = 80.0, 160.0
        self.held = False                  # placed by from_memory: look() keeps that place

    def from_memory(self, ram):
        """where the game's memory says Mario is, shifted as the centre of his red sits (17 px lower): the same place as
        look() finds, but lava (1-4) or a red shell, which have his red, cannot pull it away"""
        self.x = float((int(ram[0x86]) - int(ram[0x71c])) % 256 + 8)
        self.y = float(int(ram[0x3b8]) + 25 - TOP)
        self.held = True

    def look(self, red):
        if self.held:
            return self.x, self.y
        ys, xs = np.nonzero(red)
        if len(xs):
            near = (np.abs(xs - self.x) < 32) & (np.abs(ys - self.y) < 32)
            if near.any():
                xs, ys = xs[near], ys[near]
            self.x, self.y = float(xs.mean()), float(ys.mean())
        return self.x, self.y


def centred(cells, x, y):
    """a window VIEW x the screen each way, centred on (x, y) in pixels; outside the screen is empty"""
    R, C = VIEW * ROWS, VIEW * COLS
    cx, cy = int(x // CELL), int(y // CELL)
    big = np.zeros((2 * R + ROWS, 2 * C + COLS), bool)
    big[R:R + ROWS, C:C + COLS] = cells
    r0, c0 = R + cy - R // 2, C + cx - C // 2
    return big[r0:r0 + R, c0:c0 + C]


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
        C, eye, red = [], Eye(), None
        for _, _, f in fs:
            im = Image.open(os.path.join(raw, d, f))
            a = np.array(im)
            c = to_cells(a)
            if GAZE:
                if red is None:
                    P = np.array(im.getpalette()).reshape(-1, 3)
                    red = np.nonzero((P == RED).all(1))[0]
                c = centred(c, *eye.look(np.isin(a[TOP:], red)))
            C.append(c)
        C = np.array(C)
        A = np.array([a for _, a, _ in fs], np.uint8)
        tag = f'{name.group(1)}_e{int(name.group(2)):03d}'
        np.savez_compressed(f'{out}/{tag}.npz', cells=np.packbits(C, axis=-1), action=A, level=name.group(3),
                            win=name.group(4) == 'win', shape=np.array(C.shape))
        print(tag, name.group(3), name.group(4), len(A), 'cells lit %.0f' % C.reshape(len(C), -1).sum(1).mean(), flush=True)


if __name__ == '__main__':
    main()
