"""Stage (b): speech basics [PLAN 6b, designer 11-02].
  --arm g1p  : hand-wired G1P inside the PFC (L4 syllable units -> L1 force copies); --set g1p_own_excl=True for
               arm (ii). Then b1-b7, the speech-link probes and the link diagnostics.
  --arm F1|F2|F3 : the learned arm: no speech wiring; the gen-0 scaffold 'forced regular speaking' (scaffold.py) with
               learning on, then the same probes with the scaffold off, plus the stage-(a) properties afterwards.
usage: python3 run_b.py --arm g1p --s 1 --seed 0 --out out_b [--overhear 1000] [--set key=value ...]"""
import argparse
import os

from common import config, log, save_json
from life import Life
import probes
import scaffold
from g1p import select_l4
from wiring import measure_codes, select_codes

ap = argparse.ArgumentParser()
ap.add_argument('--arm', default='g1p')
ap.add_argument('--s', type=int, default=1)
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--out', default='out_b')
ap.add_argument('--overhear', type=int, default=1000)
ap.add_argument('--per_syl', type=int, default=30)
ap.add_argument('--quick', action='store_true')
ap.add_argument('--set', nargs='*', default=[], help='config overrides key=value')
a = ap.parse_args()
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
c = config(a.s, birth_dialogues=30 if a.quick else 200, **over)
name = f'{a.arm}_s{a.s}_seed{a.seed}' + ''.join(f'_{k}{v}' for k, v in sorted(over.items()))
res = dict(arm=a.arm, config=over)
if a.arm == 'g1p':
    life = Life(c, a.seed)
    sel = life.l4_sel
else:
    life = Life(c, a.seed, wire=False)
    life.wire_mc()                                                       # gen-0 motor collapse wiring (if mc)
    life.codes = select_codes(measure_codes(life.brain, life.coch))      # sound codes for the forced probes
    sel = select_l4(life)[0]
    res['links_before'] = probes.force_link_stats(life, sel)
    res['scaffold'] = scaffold.run(life, a.arm, per_syl=6 if a.quick else a.per_syl)
    log('scaffold', {k: v for k, v in res['scaffold'].items() if k != 'windows'})
res['registry'] = life.registry
res['calib'] = life.brain.calib
res['l4_units_per_syllable'] = sel.sum(1).tolist()
res['links'] = probes.force_link_stats(life, sel)
res['speech_links'] = probes.speech_links(life)
log('links', res['links'], 'speech links', res['speech_links'])
res.update(probes.speech_basics(life, n_overhear=5 if a.quick else a.overhear))
if a.arm != 'g1p':                                                       # does the scaffold damage stage-(a) properties?
    res['codes_after'] = probes.newborn_codes(life, n_ctx=2 if a.quick else 4, n_noise=1, lags=12)
save_json(os.path.join(a.out, name + '.json'), res)
log(name, 'b pass', res['pass'], {k: round(v, 3) for k, v in res.items() if k.startswith('b') and isinstance(v, float)})
