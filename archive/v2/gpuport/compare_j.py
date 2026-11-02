# -*- coding: utf-8 -*-
"""P5.2 / P5.5: compare staged-J records per seed.
  exact: probe tables, online accuracies, drift and MIX / hippocampus sizes identical to the canonical numpy runs?
  J (ref/analyze.py's objective) per seed: torch vs numpy canonical vs the original final_noctx numbers.
usage: python3 compare_j.py torch_glob [canon_glob] [orig_glob]"""
import glob, json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'synth'))
from analyze import score          # noqa: E402


def load(pat):
    out = {}
    for f in sorted(glob.glob(pat)):
        for line in open(f):
            if line.strip():
                r = json.loads(line)
                out[r['seed']] = r
    return out


def main():
    a = sys.argv[1:]
    T = load(a[0])
    C = load(a[1] if len(a) > 1 else os.path.join(HERE, 'out', 'jruns', 'np_canon_s*.json'))
    O = load(a[2] if len(a) > 2 else os.path.join(HERE, '..', 'synth', 'results', 'final_noctx_s*.json'))
    rows = []
    for s in sorted(set(T) | set(C) | set(O)):
        t, c, o = T.get(s), C.get(s), O.get(s)
        same = None
        if t and c:
            same = all(t[k] == c[k] for k in ('table', 'train', 'drift', 'viol')) and t['mix'] == c['mix'] and t['hip'] == c['hip']
        J = lambda r: round(score(r)['J'], 4) if r else None
        rows.append(dict(seed=s, J_torch=J(t), J_canon=J(c), J_orig=J(o), identical_to_canon=same,
                         mix=(t or {}).get('mix'), viol=(t or {}).get('viol'), secs=(t or {}).get('secs')))
        print(json.dumps(rows[-1]))
    import numpy as np
    for k in ('J_torch', 'J_canon', 'J_orig'):
        v = [r[k] for r in rows if r[k] is not None]
        if v:
            print(k, 'mean %.4f over %d seeds' % (np.mean(v), len(v)))


if __name__ == '__main__':
    main()
