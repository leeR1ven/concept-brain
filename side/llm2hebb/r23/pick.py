# -*- coding: utf-8 -*-
"""pick R3 configs from the dev grid: best (rewrite + para) with neighbour specificity >= 0.9, per key-covariance."""
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1]) if '"mode": "dev"' in l]
which = sys.argv[2]           # I | C | subj


def score(r):
    return r['rewrite'] + r['para'] + 0.5 * r['qa']


cand = [r for r in rows if (r['cov'] == which if which in 'IC' else (r['site'] == 'subj' and r['cov'] == 'C'))]
good = [r for r in cand if r['neigh_same'] >= 0.9] or cand
b = max(good, key=score)
print(f"layer={b['layer']} site={b['site']} value={b['value']} g={b['g']} s={b['s']} cov={b['cov']}")
