"""Atari-HEAD Breakout (human play, one data frame = one game frame) to compact arrays, one file per trial.
Saved: cells [T, 41, 36] bool, what the eye gets: the playfield cut to 41 x 36 cells of 4 x 4 pixels, on if anything is there;
action [T] the key at that frame; ball_x, ball_y, paddle_x [T] in cells, -1 if not seen, used only for measuring.
usage: python game_data.py <breakout.zip> <outdir>"""
import io
import os
import sys
import tarfile
import zipfile

import numpy as np
from PIL import Image

TOP, BOT, LEFT, RIGHT, CELL = 32, 196, 8, 152, 4
ROW_PADDLE = (189, 193)
ROW_BALL = (94, 188)


def to_cells(g):
    """g: gray frame [210, 160]"""
    a = g[TOP:BOT, LEFT:RIGHT] > 30
    return a.reshape(41, CELL, 36, CELL).any(axis=(1, 3))


def locate(g):
    """ball and paddle position in cells, for measuring only"""
    on = g > 30
    p = np.nonzero(on[ROW_PADDLE[0]:ROW_PADDLE[1], LEFT:RIGHT].any(0))[0]
    px = (p.mean() / CELL) if len(p) else -1.0
    ys, xs = np.nonzero(on[ROW_BALL[0]:ROW_BALL[1], LEFT:RIGHT])
    if len(ys) == 0:
        return -1.0, -1.0, px
    return xs.mean() / CELL, (ys.mean() + ROW_BALL[0] - TOP) / CELL, px


def main():
    src, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    z = zipfile.ZipFile(src)
    names = sorted(i.filename[:-4] for i in z.infolist() if i.filename.endswith('.txt'))
    for nm in names:
        base = nm.split('/')[-1]
        trial = base.split('_')[0]
        path = f'{out}/{trial}.npz'
        if os.path.exists(path):
            continue
        rows = z.read(nm + '.txt').decode().split('\n')[1:]
        ids, act = [], []
        for r in rows:
            f = r.split(',')
            if len(f) > 5:
                ids.append(f[0])
                act.append(0 if f[5] == 'null' else int(f[5]))
        T = len(ids)
        cells = np.zeros((T, 41, 36), bool)
        pos = np.full((T, 3), -1.0, np.float32)
        order = {f'{base}/{k}.png': i for i, k in enumerate(ids)}
        with tarfile.open(fileobj=io.BytesIO(z.read(nm + '.tar.bz2')), mode='r:bz2') as t:
            for m in t:
                i = order.get(m.name)
                if i is None:
                    continue
                g = np.asarray(Image.open(t.extractfile(m)).convert('L'))
                cells[i] = to_cells(g)
                pos[i] = locate(g)
        np.savez_compressed(path, cells=cells, action=np.array(act, np.uint8), ball_x=pos[:, 0], ball_y=pos[:, 1], paddle_x=pos[:, 2])
        print(trial, T, 'frames', 'ball seen', float((pos[:, 0] >= 0).mean()), flush=True)


if __name__ == '__main__':
    main()
