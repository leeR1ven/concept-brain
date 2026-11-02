"""Why does speech collapse within a few dozen practice trials? Same life, three arms from the same start (weights
restored between arms): (A) practice with the caregiver's reaction (reward on), (B) practice with no reaction (plain
learning), (C) the same as A without the 3 ordinary dialogues per 26 trials. After n trials: the say test, the outcome
mix of 26 practice trials run frozen (learning off, no reaction), and per acquired table the weight sum and the summed
absolute change from the start.
usage: python3 diag_b2.py --n 52 --out diag_b2.json --set key=value ..."""
import argparse

import numpy as np
import torch

from b2lib import practice_trial, say_test
import probes2
from common import config, log, save_json
from life import Life
from world_adapter import C1, C2, NSYL

ap = argparse.ArgumentParser()
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--n', type=int, default=52)
ap.add_argument('--out', default='diag_b2.json')
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
life = Life(config(1, reward=True, **over), a.seed)
b = life.brain
rng = np.random.default_rng(a.seed + 7100)
tabs = b.plastic()
W0 = [t.w.clone() for t in tabs]


def measure(label):
    was = b.learning
    b.learning = False
    snap = b.snapshot()
    seg = [sg for sg in b.chains.segs if sg[0] == 'stop'][0]
    stop_int = []
    oc = []
    for i, x in enumerate(rng.permutation(NSYL)[:26]):
        oc.append(practice_trial(life, rng, int(x), C1 if i % 2 == 0 else C2, react=False)[0])
        stop_int.append(float(b.pfc.x_int[seg[2]:seg[3]].float().mean()))
    b.restore(snap)
    b.learning = was
    tp = probes2.tendency_profile(life)
    ix = [i for i, t in enumerate(tp['tick']) if 10 <= t <= 13]
    tend = {k: float(np.mean([tp[k][i] for i in ix])) for k in ('x', 'best', 'first', 'total')}
    tend['nothing_thought_total'] = float(np.mean([tp['baseline_total'][i] for i in ix]))
    r = dict(say=say_test(life, list(range(NSYL))), tendency_ticks10_13=tend,
             stop_copies_internal_end_of_trial=float(np.mean(stop_int)),
             frozen_practice={k: oc.count(k) / len(oc) for k in sorted(set(oc))},
             tables={t.name: dict(sum=float(t.w.sum()), abs_change=float((t.w - w0).abs().sum()),
                                  rel_change=float((t.w - w0).abs().sum() / w0.abs().sum().clamp_min(1e-9)))
                     for t, w0 in zip(tabs, W0)})
    log(label, 'say', r['say'], 'frozen practice', r['frozen_practice'], 'tendency', tend,
        'stop copies internal', r['stop_copies_internal_end_of_trial'])
    log(label, 'tables', {k: round(v['rel_change'], 4) for k, v in r['tables'].items()})
    return r


res = dict(config=over, start=measure('start'))
start = b.snapshot_full()
ARMS = [('A reaction + dialogues', True, True, 'all'), ('B no reaction + dialogues', False, True, 'all'),
        ('C reaction, no dialogues', True, False, 'all'), ('D reaction, reverse strengthenings only', True, True, 'pos')]
for arm, react, dialogues, rev in ARMS:
    b.restore_full(start)
    b.c['r_reverse'] = rev
    b.rstats = {}
    b.learning = True
    outs = []
    for k in range(a.n):
        outs.append(practice_trial(life, rng, int(rng.integers(NSYL)), C1 if k % 2 == 0 else C2, react=react))
        if dialogues and (k + 1) % 26 == 0:
            life.run_dialogues(3)
    b.learning = False
    r = measure(arm)
    r['practice_outcomes'] = {kd: sum(o == kd for o, _ in outs) / len(outs) for kd in sorted(set(o for o, _ in outs))}
    r['reactions'] = {str(rc): sum(x == rc for _, x in outs) / len(outs) for rc in ('happy', 'sad', None)}
    r['reward_logged_changes'] = b.rstats                 # per table: summed strengthenings / weakenings re-applied
    log(arm, 'reward stats', b.rstats)
    res[arm] = r
    save_json(a.out, res)
