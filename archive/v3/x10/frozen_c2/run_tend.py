"""What tendency reaches the motor area when the PFC 'thinks' a syllable? (input model for the MC collapse test)
usage: python3 run_tend.py --arm g1p|F3 --seed 0 --out out_tend [--set key=value ...]"""
import argparse
import os

from common import config, log, save_json
from life import Life
import probes
import probes2
import scaffold
from wiring import measure_codes, select_codes

ap = argparse.ArgumentParser()
ap.add_argument('--arm', default='g1p')
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--out', default='out_tend')
ap.add_argument('--set', nargs='*', default=[])
ap.add_argument('--a1', action='store_true')
ap.add_argument('--a1L', type=int, default=0)
ap.add_argument('--tag', default='')
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
os.makedirs(a.out, exist_ok=True)
c = config(1, **over)
if a.arm == 'g1p':
    life = Life(c, a.seed)
else:
    life = Life(c, a.seed, wire=False)
    life.codes = select_codes(measure_codes(life.brain, life.coch))
    scaffold.run(life, a.arm, per_syl=30)
res = dict(arm=a.arm, tag=a.tag, config=over, registry=life.registry, calib=life.brain.calib,
           chain_report=life.brain.chain_report, profile=probes2.tendency_profile(life))
log(a.arm, a.tag, res['profile'])
fn = os.path.join(a.out, f'tend_{a.arm}{a.tag}_seed{a.seed}.json')
save_json(fn, res)
if a.a1:
    act = probes.activity(life, 50)
    res['a1'], res['a1_pass'] = act, probes.a1_pass(act)
    log('a1', res['a1_pass'], {k: (round(act[k]['min']), round(act[k]['silence']), round(act[k]['speech']))
                               for k in ('L1', 'L2', 'L3', 'L4', 'Hs')})
    save_json(fn, res)
if a.a1L:
    res['a1L'] = probes.stability(life, a.a1L)
    log('a1L', {k: v for k, v in res['a1L'].items() if k != 'windows'})
    save_json(fn, res)
