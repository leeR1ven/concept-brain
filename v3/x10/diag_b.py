"""Stage (b) diagnosis: where does a forced inner hearing turn into garble? Per tick of a forced syllable: A1 units
of the code lit (by return / by the input), A1 background units, G1 drive on the syllable's own muscles vs the best
other muscle, pooled drive, ladder blocking, stop neurons, force neurons fired, and the decoded mouth.
usage: python3 diag_b.py --s 1 [--set key=value ...]"""
import argparse
import json

import numpy as np
import torch

from common import DEV, config, log, save_json
from life import Life
import probes
from world_adapter import SYL

ap = argparse.ArgumentParser()
ap.add_argument('--s', type=int, default=1)
ap.add_argument('--set', nargs='*', default=[])
ap.add_argument('--out', default='diag_b.json')
ap.add_argument('--n', type=int, default=12)
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
c = config(a.s, birth_dialogues=60, **over)
life = Life(c, 0)
b, mot, ph = life.brain, life.brain.motor, life.ph
n0 = len(b.chains.sound_idx[0])
out = []
with probes.Frozen(life) as snap:
    for s in range(a.n):
        b.restore(snap)
        code = life.codes[s]
        a1code = b.chains.sound_idx[0][code[code < n0]]
        own = torch.as_tensor(ph.need_mat[s], device=DEV)[mot.muscle_of]
        rows = []

        def rec(lf, t):
            st = lf.brain.sound.state[0]
            d = mot.last['d']
            fired = mot.force.on
            mus = (torch.zeros(24, device=DEV).index_add_(0, mot.muscle_of, fired.float()) > 0).cpu().numpy()
            dec = ph.decode(mus)
            rows.append(dict(t=t, a1_code_on=int(st[a1code].sum()), a1_code_size=int(len(a1code)),
                             a1_other_on=int(st.sum() - st[a1code].sum()),
                             ret_a1=int((lf.brain.ret_sound[0] > 0).sum()),
                             d_own_min=round(float(d[own].min()), 3), d_own_med=round(float(d[own].median()), 3),
                             d_other_max=round(float(d[~own].max()), 3), P=round(float(mot.last['P']), 2),
                             ladder=int((mot.last['P'] >= mot.ladder).sum()), stops=int(mot.stop.on.sum()),
                             thr_own=round(float((mot.thr_f + c['b_block'] * (mot.n_tonic + (mot.last['P'] >= mot.ladder).sum()) / (8 * mot.s) + c['c_stop'] * mot.stop.on.sum() / (8 * mot.s))[own].min()), 3),
                             force_own=int(fired[own].sum()), force_other=int(fired[~own].sum()),
                             mouth=None if dec is None else (dec[0], SYL[dec[1]] if dec[0] != 'g' else len(dec[1]))))
        life.probe_env.reset_track()
        probes.play(life, [None] * 16, rec=rec, force={6 + k: probes.code_mask(life, [s]) for k in range(3)})
        out.append(dict(syl=SYL[s], said=list(life.probe_env.syl), ticks=rows[4:14]))
        log(SYL[s], 'said', life.probe_env.syl)
        for r in rows[4:14]:
            log('   ', r)
save_json(a.out, dict(config=over, registry=life.registry, trials=out))
