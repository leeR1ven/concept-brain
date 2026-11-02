"""Pilots for the 11-02 fixes (after audit). Whole brain on the c2 base, learning on, r = 0.
  --mode A: PFC count preservation (pfc_count='k') with g44 0.1 / 1 (control: thresholds). Newborn a1 and a2 / lag-0
            decoding; L4 tracking (current-syllable decoding, J(L4(t), L4(t-3)), L4 counts) at birth and after
            --dialogues overheard dialogues; activity windows every 100 dialogues.
  --mode B: hippocampal recall (hs_count='k', g_hh 1, g_h1 1 or --g_h1): activity windows incl. Hs, H1int per tick,
            A1-A4 over --dialogues overheard dialogues; then a held-out Q -> A pair heard once, D ticks of other
            dialogue, Q again: h_int on A's code (rank), A's sound units re-lit (probes2.hip_recall).
usage: python3 run_fix.py --mode A --arm k_g44_0.1 --out out_fix --set <base>"""
import argparse
import os
import time

import numpy as np
import torch

from common import config, log, save_json
from life import Life
import probes
import probes2

ap = argparse.ArgumentParser()
ap.add_argument('--mode', default='A')
ap.add_argument('--arm', default='')
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--dialogues', type=int, default=1000)
ap.add_argument('--out', default='out_fix')
ap.add_argument('--quick', action='store_true')
ap.add_argument('--set', nargs='*', default=[])
a = ap.parse_args()
t0 = time.time()
os.makedirs(a.out, exist_ok=True)


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
c = config(1, birth_dialogues=30 if a.quick else 200, **over)
life = Life(c, a.seed)
b = life.brain
names = b.count_names()
ix = {n: i for i, n in enumerate(names)}
nd = 20 if a.quick else a.dialogues
res = dict(mode=a.mode, arm=a.arm, config=over, calib=b.calib, windows=[])
act = probes.activity(life, 5 if a.quick else 50)
res['a1'], res['a1_pass'] = act, probes.a1_pass(act)
res['codes_birth'] = probes.newborn_codes(life, n_ctx=2, n_noise=1, lags=12)
res['l4_birth'] = probes2.l4_tracking(life)
log(a.arm, 'a1', res['a1_pass'], 'lag0', res['codes_birth']['L4_decode'].get(0), 'tracking at birth', res['l4_birth'],
    'Hs J margin', res['codes_birth'].get('Hs_J_margin'))
rows = []


def rec(lf):
    rows.append(lf.brain.counts())


b.learning = True
done = 0
while done < nd:
    life.run_dialogues(10 if a.quick else 100, record=rec)
    done += 10 if a.quick else 100
    C = torch.stack(rows).float()
    m, mn, mx = C.mean(0).cpu().numpy(), C.min(0).values.cpu().numpy(), C.max(0).values.cpu().numpy()
    wdw = dict(dialogue=done, **{n: float(m[i]) for n, i in ix.items()},
               Hs_min=float(mn[ix['Hs']]) if 'Hs' in ix else None, H1int_max=float(mx[ix['H1int']]) if 'H1int' in ix else None,
               L4_L3=float(m[ix['L4']] / max(m[ix['L3']], 1)))
    res['windows'].append(wdw)
    rows = []
    log(a.arm, 'dialogue', done, {k: round(v, 1) for k, v in wdw.items() if k in ('L1', 'L2', 'L3', 'L4', 'Hs', 'Hs_min', 'H1int',
                                                                                 'H1int_max', 'A1', 'A2', 'A3', 'A4', 'L4_L3')})
    save_json(os.path.join(a.out, f'fix{a.mode}_{a.arm}_seed{a.seed}.json'), res)
b.learning = False
res['l4_after'] = probes2.l4_tracking(life)
res['codes_after'] = probes.newborn_codes(life, n_ctx=2, n_noise=1, lags=12)
log(a.arm, 'tracking after', nd, res['l4_after'], 'lag0 after', res['codes_after']['L4_decode'].get(0),
    'Hs J margin birth / after', res['codes_birth'].get('Hs_J_margin'), res['codes_after'].get('Hs_J_margin'))
if a.mode == 'B':
    res['hip_trace'] = probes2.hip_trace(life, Ds=(50,) if a.quick else (300, 3000))
    log(a.arm, 'hip trace', res['hip_trace'])
res['minutes'] = (time.time() - t0) / 60
save_json(os.path.join(a.out, f'fix{a.mode}_{a.arm}_seed{a.seed}.json'), res)
