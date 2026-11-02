"""Why is 'thinking x' said as garble? Per tick of forced inner hearing (x's code lit in PFC L1 on ticks 6-8, as in
b1), muted and unmuted: the decoded mouth (clear x / other syllable / garble / silent), lit-return share of x's
muscles vs the best rival, muscles on, and the MC excitatory pool activity of x vs the best other syllable.
usage: python3 diag_say.py --out diag_say.json --set key=value ..."""
import argparse

import numpy as np
import torch

from common import DEV, config, log, save_json
from life import Life
from probes import Frozen, code_mask, play
from world_adapter import NSYL

ap = argparse.ArgumentParser()
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--out', default='diag_say.json')
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
life = Life(config(1, **over), a.seed)
b, ph, mot = life.brain, life.ph, life.brain.motor
need = torch.as_tensor(ph.need_mat, device=DEV).float()
T = 22
res = dict(config=over, registry=life.registry)
for mute in (True, False):
    K = {k: np.zeros((NSYL, T)) for k in ('clear_x', 'other', 'garble', 'silent', 'x_share', 'best_rival', 'muscles_on',
                                          'pool_x', 'pool_best_other', 'stop_on')}
    with Frozen(life) as snap:
        for x in range(NSYL):
            b.restore(snap)
            life.probe_env.reset_track()
            rows = []

            def rec(lf, t):
                R = lf.brain.ret_force.float()
                per_mus = torch.zeros(24, device=DEV).index_add_(0, mot.muscle_of, R) / mot.m
                sh = need @ per_mus / 4.0
                dm = ph.decode(lf.last_mouth)
                pe = mot.pool_e.on.float().view(NSYL, -1).mean(1) if mot.MC is not None else torch.zeros(NSYL, device=DEV)
                rows.append((sh.cpu().numpy(), pe.cpu().numpy(), int(mot.stop.on.sum()), dm, int(lf.last_mouth.sum())))
            play(life, [None] * T, mute=mute, force={6 + k: code_mask(life, [x]) for k in range(3)}, rec=rec)
            for t, (sh, pe, st, dm, nm) in enumerate(rows):
                K['clear_x'][x, t] = dm is not None and dm[0] == 's' and dm[1] == x
                K['other'][x, t] = dm is not None and dm[0] in ('s', 'm') and dm[1] != x
                K['garble'][x, t] = dm is not None and dm[0] == 'g'
                K['silent'][x, t] = dm is None
                K['muscles_on'][x, t] = nm
                K['x_share'][x, t] = sh[x]
                K['best_rival'][x, t] = np.delete(sh, x).max()
                K['pool_x'][x, t] = pe[x]
                K['pool_best_other'][x, t] = np.delete(pe, x).max()
                K['stop_on'][x, t] = st
    res['muted' if mute else 'unmuted'] = {k: v.mean(0).round(3).tolist() for k, v in K.items()}
    log('muted' if mute else 'unmuted', {k: v.mean(0).round(2).tolist() for k, v in K.items()})
save_json(a.out, res)
