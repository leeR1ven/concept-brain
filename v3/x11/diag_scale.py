"""Diagnostic (frozen, learning OFF; reads a copy of a curve checkpoint, never writes it): why is the PFC's
next-syllable prediction at chance?
 (a) echo test: the same x_int window (t0+1..t0+3) ranked against the target, the previous syllable and the one
     before it (codes in the same window);
 (b) source split: the internal L1 drive from the learned L4->L1 links alone and from the birth loop alone (raw drive
     into sound copies on t0+1..t0+3), each ranked against the target (and the previous syllable);
 (c) hippocampus: H1's internal drive (learned Hs->H1, after its patch inhibition) against theta_h1;
 (d) L4 sharing: mean Jaccard of L4 on t0+3 between occurrences of different / same syllables.
usage: python3 diag_scale.py --run x11_s1 --seed 0 --out diag.json [--n_heldout 100]"""
import argparse
import os

import numpy as np
import torch

from common import DEV, log, save_json
from life11 import Life11, load_bases, x11_config
from probes11 import Frozen, heldout_plan, n_sound_copies, rank_scores
from world11 import Dialogues

HERE = os.path.dirname(os.path.abspath(__file__))

ap = argparse.ArgumentParser()
ap.add_argument('--run', default='x11_s1')
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--out', default='diag.json')
ap.add_argument('--n_heldout', type=int, default=100)
a = ap.parse_args()
ck = os.path.expanduser(f'~/ckpt/{a.run}_seed{a.seed}')
c = x11_config(1)
bases = load_bases(HERE)
train = Dialogues.load(os.path.join(HERE, 'data', 'train.npz'))
test = Dialogues.load(os.path.join(HERE, 'data', 'heldout.npz'))
life, _ = Life11.load(os.path.join(ck, 'life.pt'), c, a.seed, train, bases)
cd = torch.load(os.path.join(ck, 'codes.pt'), map_location=DEV, weights_only=False)
b = life.brain
b.learning = False
nsnd = n_sound_copies(b)
log('loaded at', life.world.n_ex, 'dialogues')
stream, occ = heldout_plan(test, a.n_heldout, 4242)
pos = {s: i for i, s in enumerate(cd['sids'])}
occ = [o for o in occ if o[1] in pos and o[2] in pos]
n, T = len(occ), len(stream)
o3 = np.full(T + 4, -1, np.int64)
for j, o in enumerate(occ):
    o3[o[0] + 1:o[0] + 4] = j
P = {k: torch.zeros((n, nsnd), device=DEV) for k in ('x_int', 'learned', 'loop')}
h_drive, h_max = [], []
L4at = torch.zeros((n, b.pfc.n), dtype=torch.bool, device=DEV)
with Frozen(life):
    life.coch.rng = np.random.default_rng(4242)
    for t, x in enumerate(stream):
        a0 = life.coch.a0([] if x is None else [x[:5]])
        b.step(torch.as_tensor(a0, device=DEV), 0.0)
        j = o3[t]
        if j >= 0:
            a4 = b.pfc.L4.prev
            P['x_int'][j] += b.pfc.x_int[:nsnd].float()
            P['learned'][j] += b.pfc.L41.drive(a4)[:nsnd]
            P['loop'][j] += b.pfc.loop.drive(a4)[:nsnd]
            if t == occ[j][0] + 3:
                L4at[j] = b.pfc.L4.on
        if x is not None and t % 5 == 0:
            d = b.hip.inhib1(b.hip.H1t.drive(b.hip.Hs.prev))
            h_drive.append(d.max().item())
            h_max.append(float((d > 0.5 * c['theta_h1']).float().sum()))
    life.coch.rng = np.random.default_rng(0)
true = np.array([pos[o[1]] for o in occ])
prev1 = np.array([pos[o[2]] for o in occ])
keep2 = np.array([o[3] in pos for o in occ])
res = dict(at_dialogues=life.world.n_ex, n_occ=n)
for k, M in P.items():
    res[k] = dict(target=rank_scores(M, cd['code3'], cd['sids'], true)[0],
                  previous=rank_scores(M, cd['code3'], cd['sids'], prev1)[0],
                  before_previous=rank_scores(M[keep2], cd['code3'], cd['sids'],
                                              np.array([pos[o[3]] for o, kk in zip(occ, keep2) if kk]))[0],
                  mean_total=float(M.sum(1).mean()))
    log(k, res[k])
res['hip'] = dict(h1_internal_drive_max_median=float(np.median(h_drive)), h1_internal_drive_max_max=float(np.max(h_drive)),
                  units_above_half_theta_mean=float(np.mean(h_max)), theta_h1=c['theta_h1'])
log('hip', res['hip'])
# (d) L4 sharing at t0+3 (the L4 state that drives x_int on t0+4)
L = L4at.float()
inter = L @ L.T
cnt = L.sum(1)
J = inter / (cnt[:, None] + cnt[None] - inter).clamp_min(1)
same = torch.as_tensor(true[:, None] == true[None], device=DEV)
off = ~torch.eye(n, dtype=torch.bool, device=DEV)
res['L4'] = dict(frac_on=float(cnt.mean() / b.pfc.n), J_same_syllable=float(J[same & off].mean()),
                 J_different=float(J[~same].mean()))
log('L4', res['L4'])
save_json(a.out, res)
