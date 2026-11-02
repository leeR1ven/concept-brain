"""Stage (b): generation-0 speech basics [PLAN 6b]. usage: python3 run_b.py --s 1 --seed 0 --out out_b"""
import argparse
import os

from common import config, log, save_json
from life import Life
import probes

ap = argparse.ArgumentParser()
ap.add_argument('--s', type=int, default=1)
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--out', default='out_b')
ap.add_argument('--overhear', type=int, default=1000)
ap.add_argument('--quick', action='store_true')
ap.add_argument('--set', nargs='*', default=[], help='config overrides key=value (registered (a)/(b) choices)')
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
life = Life(c, a.seed)
res = dict(config=over, registry=life.registry, calib=life.brain.calib)
res.update(probes.speech_basics(life, n_overhear=5 if a.quick else a.overhear))
save_json(os.path.join(a.out, f's{a.s}_seed{a.seed}.json'), res)
log('b', res['pass'])
