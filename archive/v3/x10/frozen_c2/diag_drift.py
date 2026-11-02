"""Does the gen-0 speech link survive ordinary overhearing? (b7 showed b1 -> 0 after 100 overheard dialogues.)
Say test (b1 form) and the inner-hearing tendency at the start and after 200 and 700 overheard dialogues (learning on,
r = 0), with the relative change of each gen-0 / acquired table and the child's utterances per dialogue.
usage: python3 diag_drift.py --out drift.json --set key=value ..."""
import argparse

import numpy as np

from b2lib import say_test
from common import config, log, save_json
from life import Life
import probes2

ap = argparse.ArgumentParser()
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--out', default='drift.json')
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
b = life.brain
tabs = b.plastic()
W0 = [t.w.clone() for t in tabs]
w = life.env.w
res = dict(config=over, steps=[])


def measure(n):
    tp = probes2.tendency_profile(life)
    ix = [i for i, t in enumerate(tp['tick']) if 10 <= t <= 13]
    r = dict(dialogues=n, say=say_test(life, list(range(34))),
             tendency={k: float(np.mean([tp[k][i] for i in ix])) for k in ('x', 'best', 'first', 'total')},
             nothing_thought=float(np.mean([tp['baseline_total'][i] for i in ix])),
             rel_change={t.name.split(' (')[0]: float((t.w - w0).abs().sum() / w0.abs().sum().clamp_min(1e-9))
                         for t, w0 in zip(tabs, W0) if 'G1P' in t.name or 'MC' in t.name})
    log(over.get('gen0_strength'), n, 'say', r['say'], 'tendency', r['tendency'], 'rel change', r['rel_change'])
    return r


res['steps'].append(measure(0))
b.learning = True
done = 0
for n in (200, 700):
    u0, e0 = len(w.utts), w.n_ex
    life.run_dialogues(n - done)
    done = n
    b.learning = False
    r = measure(n)
    r['utts_per_dialogue'] = (len(w.utts) - u0) / max(1, w.n_ex - e0)
    r['garbled_share'] = float(np.mean([u['garble'] > 0 for u in w.utts[u0:]])) if len(w.utts) > u0 else None
    res['steps'].append(r)
    b.learning = True
    save_json(a.out, res)
