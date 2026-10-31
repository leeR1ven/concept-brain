# -*- coding: utf-8 -*-
"""Score tune2 runs (J exactly as tune/synth analyze.py) and pair them with the base of the same rung by seed.
usage: python3 analyze_t2.py <prefix e.g. r1> [--base base] [--md]"""
import glob, json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from analyze import score   # noqa: E402


def load(prefix):
    recs = {}
    for f in sorted(glob.glob(os.path.join(HERE, 'out', 'runs', f'{prefix}_*.json'))):
        for line in open(f):
            if line.strip():
                r = json.loads(line)
                recs[(r['tag'], r['seed'])] = r
    return recs


def row(r):
    s = score(r)
    ctx = {c['gap']: c for c in r.get('ctx_probe') or []}
    s['ctx0'] = ctx.get(0, {}).get('test_acc'); s['ctx1'] = ctx.get(1, {}).get('test_acc')
    s['mix'] = r['mix']; s['secs'] = r['secs']; s['nnz'] = r.get('gen_nnz_per_unit')
    s['held'] = (r.get('ctx_stats') or {}).get('mean_held')
    last = r['table'][f"s{r['R']}"]
    for st in 'BCD':
        rf = (last.get(st) or {}).get('_refs', {}).get('ALL', {})
        s[f'echo{st}'] = rf.get('echo'); s[f'const{st}'] = rf.get('const')
    return s


def main():
    a = sys.argv[1:]
    prefix = a[0]
    base = a[a.index('--base') + 1] if '--base' in a else 'base'
    recs = load(prefix)
    S = {k: row(r) for k, r in recs.items()}
    tags = sorted(set(t for t, _ in S))
    btag = f'{prefix}_{base}'
    out = []
    for t in tags:
        seeds = sorted(s for tt, s in S if tt == t)
        Js = [S[(t, s)]['J'] for s in seeds]
        d = [S[(t, s)]['J'] - S[(btag, s)]['J'] for s in seeds if (btag, s) in S]
        m = lambda k: np.mean([S[(t, s)][k] for s in seeds if S[(t, s)].get(k) is not None]) if any(S[(t, s)].get(k) is not None for s in seeds) else float('nan')
        out.append((np.mean(d) if d else float('nan'), t, seeds, Js, m))
    out.sort(key=lambda x: -x[0] if not np.isnan(x[0]) else 9)
    keys = ['F', 'RET', 'H', 'DR', 'B3', 'C3', 'D3', 'colour', 'count', 'name', 'check', 'ctx0', 'ctx1', 'held', 'mix', 'secs']
    print(f"{'tag':22s} seeds  J(per seed)          dJ    " + ' '.join(f'{k:>6s}' for k in keys) + '  viol')
    for dJ, t, seeds, Js, m in out:
        viol = sum(S[(t, s)]['viol'] for s in seeds)
        print(f"{t:22s} {','.join(map(str, seeds)):5s}  {'/'.join(f'{j:.3f}' for j in Js):20s} {dJ:+.3f} " +
              ' '.join(f'{m(k):6.3f}' if k not in ('mix', 'secs') else f'{m(k):6.0f}' for k in keys) + f'  {viol}')


if __name__ == '__main__':
    main()
