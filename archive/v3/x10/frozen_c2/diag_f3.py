"""Why does the learned speech arm (F3) carry nothing? (coordinator 11-02) On P1: at the forced muscle onsets during
the scaffold, Jaccard of the L4 state with x's inner code (PFC thinking x), x's heard code, other syllables' inner
codes and the ordinary held state; and the learned L4 -> force-copy links read from x's inner code and heard code
(true/other ratio, margin, top-4 exact).
usage: python3 diag_f3.py --arm F3 --out diag_f3.json --set key=value ..."""
import argparse

import numpy as np
import torch

from common import DEV, config, log, save_json
from g1p import background_l4, measure_l4, measure_l4_inner
from life import Life
import probes
import scaffold
from wiring import measure_codes, select_codes
from world_adapter import ADULTS, NSYL

ap = argparse.ArgumentParser()
ap.add_argument('--arm', default='F3')
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--out', default='diag_f3.json')
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
inner = measure_l4_inner(life) >= 0.5                                          # [34, n4]
heard = measure_l4(life, ADULTS).mean(1) >= 0.5
held = background_l4(life) >= 0.5
jac = lambda p, q: float((p & q).sum()) / max(1.0, float((p | q).sum()))
log('codes: inner size', float(inner.float().sum(1).mean()), 'heard size', float(heard.float().sum(1).mean()),
    'held size', float(held.float().sum()))
rows = {'t0-1 (before forcing)': [], 't0 (forced tick)': [], 't0+1 (force copies rise)': []}
state = dict(after_left=0, x=None)


def on_tick(b, x, flag):
    """pre-call (flag bool) sees L4(t-1); 'after' sees L4(t). L41 learns L4(t0) -> force copies rising at t0+1."""
    if flag == 'after':
        if state['after_left'] == 2:
            rows['t0 (forced tick)'].append((state['x'], b.pfc.L4.on.clone()))
        elif state['after_left'] == 1:
            rows['t0+1 (force copies rise)'].append((state['x'], b.pfc.L4.on.clone()))
        state['after_left'] = max(0, state['after_left'] - 1)
        return
    if flag:
        rows['t0-1 (before forcing)'].append((x, b.pfc.L4.on.clone()))
        state.update(after_left=2, x=x)


sc = scaffold.run(life, a.arm, per_syl=30, on_tick=on_tick)
res = dict(arm=a.arm, config=over, scaffold={k: v for k, v in sc.items() if k not in ('windows', 'budget')})
for k, rr in rows.items():
    ji = [jac(s, inner[x]) for x, s in rr]
    jh = [jac(s, heard[x]) for x, s in rr]
    jo = [np.mean([jac(s, inner[y]) for y in range(NSYL) if y != x]) for x, s in rr]
    jb = [jac(s, held) for x, s in rr]
    res[k] = dict(n=len(rr), J_inner_x=float(np.mean(ji)), J_heard_x=float(np.mean(jh)),
                  J_inner_other=float(np.mean(jo)), J_held=float(np.mean(jb)),
                  size=float(np.mean([s.float().sum().item() for _, s in rr])))
    log(k, res[k])
res['links_from_inner_code'] = probes.force_link_stats(life, inner)
res['links_from_heard_code'] = probes.force_link_stats(life, heard)
log('links from inner code', res['links_from_inner_code'])
log('links from heard code', res['links_from_heard_code'])
save_json(a.out, res)
