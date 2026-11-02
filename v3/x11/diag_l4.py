"""Frozen diagnostic (learning OFF, checkpoint copy; corrects diag_scale's (d), which grouped L4 at t0+3 by the target,
whose sound has not reached L4 by then): how syllable-specific is each layer at each latency? For every held-out
occurrence (onset t0) and lag k = 1..7 the state of L1ext (sound copies), L2, L3, L4 at t0+k; mean Jaccard between
occurrences of the same syllable vs of different syllables, grouped by the target and by the previous syllable.
usage: python3 diag_l4.py --run x11_diag --seed 0 --out diag_l4.json [--n_heldout 120]"""
import argparse
import os

import numpy as np
import torch

from common import DEV, log, save_json
from life11 import Life11, load_bases, x11_config
from probes11 import Frozen, heldout_plan, n_sound_copies
from world11 import Dialogues

HERE = os.path.dirname(os.path.abspath(__file__))
LAGS = range(1, 8)


def specificity(life, test, n_heldout):
    b = life.brain
    b.learning = False
    nsnd = n_sound_copies(b)
    stream, occ = heldout_plan(test, n_heldout, 4242)
    n = len(occ)
    want = {}
    for j, o in enumerate(occ):
        for k in LAGS:
            want.setdefault(o[0] + k, []).append((j, k))
    layers = dict(L1ext=nsnd, L2=b.pfc.n, L3=b.pfc.n, L4=b.pfc.n)
    S = {(nm, k): torch.zeros((n, sz), dtype=torch.bool, device=DEV) for nm, sz in layers.items() for k in LAGS}
    p = b.pfc
    with Frozen(life):
        life.coch.rng = np.random.default_rng(4242)
        for t, x in enumerate(stream):
            a0 = life.coch.a0([] if x is None else [x[:5]])
            b.step(torch.as_tensor(a0, device=DEV), 0.0)
            for j, k in want.get(t, []):
                S[('L1ext', k)][j] = p.L1ext.on[:nsnd]
                S[('L2', k)][j] = p.L2.on
                S[('L3', k)][j] = p.L3.on
                S[('L4', k)][j] = p.L4.on
        life.coch.rng = np.random.default_rng(0)
    tgt = torch.as_tensor([o[1] for o in occ], device=DEV)
    prv = torch.as_tensor([o[2] for o in occ], device=DEV)
    off = ~torch.eye(n, dtype=torch.bool, device=DEV)
    out = {}
    for (nm, k), M in S.items():
        L = M.float()
        inter = L @ L.T
        cnt = L.sum(1)
        J = inter / (cnt[:, None] + cnt[None] - inter).clamp_min(1)
        row = dict(frac_on=float(cnt.mean() / M.shape[1]))
        for g, lab in (('target', tgt), ('previous', prv)):
            same = (lab[:, None] == lab[None]) & off
            row[f'J_same_{g}'] = float(J[same].mean())
            row[f'J_diff_{g}'] = float(J[~same & off].mean())
            row[f'margin_{g}'] = row[f'J_same_{g}'] - row[f'J_diff_{g}']
        out[f'{nm}@t0+{k}'] = row
        del J, inter
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', default='x11_diag')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default='diag_l4.json')
    ap.add_argument('--n_heldout', type=int, default=120)
    a = ap.parse_args()
    ck = os.path.expanduser(f'~/ckpt/{a.run}_seed{a.seed}')
    c = x11_config(1)
    bases = load_bases(HERE)
    train = Dialogues.load(os.path.join(HERE, 'data', 'train.npz'))
    birth = Dialogues.load(os.path.join(HERE, 'data', 'birth.npz'))
    test = Dialogues.load(os.path.join(HERE, 'data', 'heldout.npz'))
    res = {}
    life, _ = Life11.load(os.path.join(ck, 'life.pt'), c, a.seed, train, bases)
    res['checkpoint_dialogues'] = life.world.n_ex
    res['checkpoint'] = specificity(life, test, a.n_heldout)
    save_json(a.out, res)
    del life
    torch.cuda.empty_cache()
    nb = Life11(c, a.seed, train, birth, bases)
    res['newborn'] = specificity(nb, test, a.n_heldout)
    save_json(a.out, res)
    for st in ('newborn', 'checkpoint'):
        for key, r in res[st].items():
            log(st, key, {k: round(v, 3) for k, v in r.items()})
