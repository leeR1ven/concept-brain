"""paired split/delete tolerance: identical stream, probes and seeds; with and without a few % of the copied feature
units deleted (IDs kept, dead) and the same number of children added (new IDs) at t = 100k. Only probes written
before 100k are compared, tested at the same ages after it. Run: python3 -I x3_pert.py"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import x3_hippo as X
TP = 100000
test_at = [TP + 300, TP + 1500, TP + 4000, TP + 8000, TP + 15000]
out = {}
for name, ov in (('none', {}), ('3pct', dict(pert_frac=0.03, pert_t=TP)), ('5pct', dict(pert_frac=0.05, pert_t=TP))):
    cfg = dict(X.BASE, probe_every=800, **ov)
    r = X.run(cfg, TP + 15001, test_at, verbose=False)
    T = [t for t in r['tests'] if t['t_test'] - t['age'] < TP]
    out[name] = {(t['i'], t['t_test']): t for t in T}
    print(name, 'rows', len(T), 'rss', r['rss_mb'], flush=True)
keys = sorted(set(out['none']) & set(out['3pct']) & set(out['5pct']))
res = {}
for m in ('r50', 'r20', 'i100', 'fid50'):
    for lo, hi in ((0, 5000), (5000, 10000), (10000, 20000)):
        ks = [k for k in keys if lo <= out['none'][k]['age'] < hi]
        if ks:
            res[f'{m} age {lo}-{hi} (n={len(ks)})'] = {n: round(float(np.mean([out[n][k][m] for k in ks])), 3) for n in out}
for k, v in res.items():
    print(k, v)
json.dump(res, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'x3_pert.json'), 'w'), indent=1)
