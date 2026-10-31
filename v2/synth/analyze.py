# -*- coding: utf-8 -*-
"""Collect tune.py results and score them.
Objective (stated exactly; mechanics must be 0 violations, else the run is rejected):
  full curriculum (R=3, stages s1, s2, s3; frozen probes B/C/D after each stage):
    F   = (B3 + C3 + D3) / 3                         final probe accuracy (ALL, acceptable-reply rate)
    RET = (min(0, B3 - B1) + min(0, C3 - C2)) / 2    retention loss of B and C (vs just after they were learned)
    H   = (hB3 + hC3 + hD3) / 3                      held-out accuracy at the end (never-taught filler x frame pairs)
    DR  = C probes after 300 untutored C dialogues with learning on (free-use drift; level after)
    J   = F + 0.5 RET + 0.25 H + 0.25 DR
  short rung (R=1: A + B only):  J1 = B1 + 0.25 hB1 + 0.25 DR_B   (DR_B = B probes after 300 untutored B dialogues)
  one-shot (D oneshot acc at the end) is reported, not optimised.
usage: python3 analyze.py [glob] [--R 1|2|3] [--top 30] [--by tag]"""
import glob, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def load(pattern):
    recs = []
    for f in sorted(glob.glob(pattern)):
        for line in open(f):
            line = line.strip()
            if line:
                try:
                    recs.append(json.loads(line))
                except Exception:
                    pass
    return recs


def acc(t, stage, key='ALL', held=False):
    try:
        v = t[stage][key]
        return v[2] if held else v[0]
    except Exception:
        return None


def score(r):
    T, R = r['table'], r['R']
    out = dict(tag=r['tag'], seed=r['seed'], R=R, viol=r['viol'], secs=r['secs'])
    z = lambda x: 0.0 if x is None else x
    if R == 1:
        B1 = z(acc(T['s1'], 'B')); hB1 = z(acc(T['s1'], 'B', held=True))
        dr = r.get('drift') or {}
        drb = z(acc(dr, 'before')) if dr else None
        dra = z(acc(dr, 'after')) if dr else 0.0
        out.update(B1=B1, hB1=hB1, DRb=drb, DR=dra, J=B1 + 0.25 * hB1 + 0.25 * dra)
        for k in ('what', 'slot', 'where'):
            out[k] = acc(T['s1'], 'B', k)
    else:
        last = T[f's{R}']
        B1 = z(acc(T['s1'], 'B')); C2 = z(acc(T['s2'], 'C')) if R >= 2 else 0.0
        B3, C3 = z(acc(last, 'B')), z(acc(last, 'C'))
        D3 = z(acc(last, 'D')) if R >= 3 else 0.0
        hB, hC = z(acc(last, 'B', held=True)), z(acc(last, 'C', held=True))
        hD = z(acc(last, 'D', held=True)) if R >= 3 else 0.0
        n_st = 3 if R >= 3 else 2
        F = (B3 + C3 + D3) / n_st
        RET = (min(0, B3 - B1) + (min(0, C3 - C2) if R >= 3 else 0.0)) / (2 if R >= 3 else 1)
        H = (hB + hC + hD) / n_st
        dr = r.get('drift') or {}
        drb = acc(dr, 'before') if dr else None
        dra = z(acc(dr, 'after')) if dr else 0.0
        J = F + 0.5 * RET + 0.25 * H + 0.25 * dra
        out.update(B1=B1, C2=C2, B3=B3, C3=C3, D3=D3, F=F, RET=RET, H=H, hB=hB, hC=hC, hD=hD, DRb=drb, DR=dra, J=J)
        if R >= 3:
            for k in ('check', 'name', 'before', 'oneshot'):
                out[k] = acc(last, 'D', k)
            for k in ('colour', 'count', 'offer', 'taste', 'ritual'):
                out[k] = acc(last, 'C', k)
            for k in ('what', 'slot', 'where'):
                out[k] = acc(last, 'B', k)
            out['B2'] = acc(T['s2'], 'B')
    if r['viol']:
        out['J'] = -1.0
    return out


def main():
    a = sys.argv[1:]
    pat = a[0] if a and not a[0].startswith('--') else os.path.join(HERE, 'out', 'runs', '*.json')
    Rf = int(a[a.index('--R') + 1]) if '--R' in a else None
    top = int(a[a.index('--top') + 1]) if '--top' in a else 40
    recs = [r for r in load(pat) if Rf is None or r['R'] == Rf]
    S = [score(r) for r in recs]
    by = {}
    for s, r in zip(S, recs):
        by.setdefault(s['tag'], []).append((s, r))
    rows = []
    for tag, lst in by.items():
        Js = [s['J'] for s, _ in lst]
        rows.append((float(np.mean(Js)), tag, lst))
    rows.sort(key=lambda x: -x[0])
    for J, tag, lst in rows[:top]:
        s0 = lst[0][0]
        keys = [k for k in ('B1', 'hB1', 'DRb', 'DR', 'C2', 'B3', 'C3', 'D3', 'F', 'RET', 'H', 'check', 'name', 'oneshot',
                            'colour', 'count') if k in s0]
        vals = "  ".join(f"{k} " + "/".join(f"{s[k]:.2f}" if isinstance(s.get(k), (int, float)) else "-" for s, _ in lst)
                         for k in keys)
        print(f"J {J:.3f} [{len(lst)} seeds {','.join(str(s['seed']) for s, _ in lst)}] {tag:28s} viol {sum(s['viol'] for s, _ in lst)}  {vals}")


if __name__ == '__main__':
    main()
