"""Fast gain sweep (designer 11-02: "gain searches should take minutes"). One birth per process; every arm starts
from the same post-birth state (snapshot_full) with its own runtime gains / rates on the learned hippocampal tables:
--dialogues overheard dialogues (learning on), then the frozen checks and the recall trace (300-tick gap).
Per arm: J(Hs(t), Hs(t-3)) during adult speech in bins of 20 dialogues (lock-in trend); the Hs J margin (a2 form);
the recall trace (probes2.hip_trace); H1int per tick, Hs, A1-A4 means over the run.
usage: python3 run_sweep.py --arms '[{"g_hh":0.1,"g_h1":1,"eta_x":1}, ...]' --out out_sweep --tag p0 --set <base>"""
import argparse
import json
import os
import time

import numpy as np
import torch

from common import config, log, save_json
from life import Life
import probes
import probes2
from probes import jac

ap = argparse.ArgumentParser()
ap.add_argument('--arms', required=True)
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--dialogues', type=int, default=100)
ap.add_argument('--D', default='300')
ap.add_argument('--out', default='out_sweep')
ap.add_argument('--tag', default='p0')
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
life = Life(config(1, **over), a.seed)
b = life.brain
hip = b.hip
names = b.count_names()
ix = {n: i for i, n in enumerate(names)}
post_birth = b.snapshot_full()
eta0 = (hip.HH.eta, hip.HH.eta_m, hip.H1t.eta, hip.H1t.eta_m)
log('birth done', round((time.time() - t0) / 60, 1), 'min')
out = []
for arm in json.loads(a.arms):
    ta = time.time()
    b.restore_full(post_birth)
    hip.HH.gain, hip.H1t.gain = float(arm['g_hh']), float(arm['g_h1'])
    ex = float(arm.get('eta_x', 1))
    hip.HH.eta, hip.HH.eta_m, hip.H1t.eta, hip.H1t.eta_m = (e * ex for e in eta0)
    b.learning = True
    ring, js, rows = [], [], []
    bin_j, bins = [], []
    n_ex0 = life.env.w.n_ex

    def rec(lf):
        hs = hip.Hs.on.clone()
        ring.append(hs)
        if len(ring) > 4:
            ring.pop(0)
        if lf.env.adult_speaking() and len(ring) == 4:
            bin_j.append(jac(hs, ring[0]).item())
        rows.append(b.counts())
        d = lf.env.w.n_ex - n_ex0
        if d > 0 and d % 20 == 0 and bin_j and (not bins or bins[-1][0] != d):
            bins.append((d, float(np.mean(bin_j))))
            bin_j.clear()
    life.run_dialogues(a.dialogues, record=rec)
    b.learning = False
    C = torch.stack(rows).float()
    m = C.mean(0).cpu().numpy()
    cod = probes.newborn_codes(life, n_ctx=2, n_noise=1, lags=0)
    tr = probes2.hip_trace(life, Ds=tuple(int(x) for x in a.D.split(',')))
    r = dict(arm=arm, J_Hs_speech_bins=bins, J_Hs_speech_last=bins[-1][1] if bins else None,
             Hs_J_margin=cod.get('Hs_J_margin'), L4_lag0=cod['L4_decode'].get(0),
             H1int_per_tick=float(m[ix['H1int']]), H1int_max=float(C[:, ix['H1int']].max()), Hs=float(m[ix['Hs']]),
             A=[float(m[ix[f'A{l}']]) for l in range(1, 5)], trace=tr, minutes=(time.time() - ta) / 60)
    out.append(r)
    ho = tr[f'D={a.D.split(",")[0]}']['heard_once']
    ct = tr[f'D={a.D.split(",")[0]}']['control']
    log(a.tag, arm, 'J Hs speech', [round(x[1], 2) for x in bins], 'margin', round(r['Hs_J_margin'] or 0, 3),
        'Q refound own/other', round(ho['Hs_rehear_J_own_Q'], 3), round(ho['Hs_rehear_J_other_Q'], 3),
        'A drive sub/theta A, other', ho['H1_drive_over_theta']['sub'], 'h_int A share vs ctl', round(ho['A_hint_share'], 3),
        round(ct['A_hint_share'], 3), 'other max', round(ho['other_hint_share_max'], 3), 'H1int/tick', round(r['H1int_per_tick'], 2), 'min', round(r['minutes'], 1))
    save_json(os.path.join(a.out, f'sweep_{a.tag}_seed{a.seed}.json'), out)
