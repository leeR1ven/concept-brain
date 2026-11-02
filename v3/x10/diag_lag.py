"""When does the PFC 'think' x after hearing it? (for a learned-speech scaffold timed to the inner state)
On P1, frozen: an adult says x (3 ticks), then silence; per tick after the onset, Jaccard of L4 with x's inner code
vs other syllables' inner codes (specificity = difference), and with x's heard code; also the L1 internal copies of
x's sound code (share lit internally = inner repetition).
usage: python3 diag_lag.py --out diag_lag.json --set key=value ..."""
import argparse

import numpy as np
import torch

from common import DEV, config, log, save_json
from g1p import measure_l4_inner
from life import Life
from probes import Frozen, code_mask, play
from wiring import measure_codes, select_codes
from world_adapter import C1, C2, NSYL

ap = argparse.ArgumentParser()
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--out', default='diag_lag.json')
ap.add_argument('--set', nargs='*', default=[])
a = ap.parse_args()


def val(v):
    if v in ('True', 'False'):
        return v == 'True'
    try:
        return int(v)
    except ValueError:
        try:
            return float(v)
        except ValueError:
            return v


over = {k: val(v) for k, v in (x.split('=') for x in a.set)}
life = Life(config(1, **over), a.seed, wire=False)
life.codes = select_codes(measure_codes(life.brain, life.coch))
inner = measure_l4_inner(life) >= 0.5
jac = lambda p, q: float((p & q).sum()) / max(1.0, float((p | q).sum()))
T = 24
Jx, Jo, Lin, Fx = (np.zeros((NSYL, 2, T)) for _ in range(4))
b = life.brain
with Frozen(life) as snap:
    for x in range(NSYL):
        cm = code_mask(life, [x])
        for vi, v in enumerate((C1, C2)):
            b.restore(snap)
            rec = []
            play(life, [None] * 4 + [(x, v, 1.0)] * 3 + [None] * (T - 3), mute=True,
                 rec=lambda lf, t: rec.append((lf.brain.pfc.L4.on.clone(), lf.brain.pfc.x_int.clone())))
            for k in range(T):
                l4, xi = rec[4 + k]
                js = [jac(l4, inner[y]) for y in range(NSYL)]
                Jx[x, vi, k] = js[x]
                Jo[x, vi, k] = np.mean([j for y, j in enumerate(js) if y != x])
                Fx[x, vi, k] = float(int(np.argmax(js)) == x)
                Lin[x, vi, k] = float((xi & cm).sum()) / max(1, int(cm.sum()))
res = dict(config=over, ticks_after_onset=list(range(T)), J_inner_x=Jx.mean((0, 1)).round(3).tolist(),
           J_inner_other=Jo.mean((0, 1)).round(3).tolist(), specificity=(Jx - Jo).mean((0, 1)).round(3).tolist(),
           x_first=Fx.mean((0, 1)).round(2).tolist(), L1_internal_share_of_x_code=Lin.mean((0, 1)).round(3).tolist())
for k, v in res.items():
    if k != 'config':
        log(k, v)
save_json(a.out, res)
