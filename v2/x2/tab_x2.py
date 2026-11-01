# -*- coding: utf-8 -*-
"""x2 summary table: per arm, C accuracy per mark (per seed and mean), failure shape at the last mark, online silence
and RAL commits (unanswered lines) per dialogue in each segment, tutored-era replay pool (pool_old) per mark.
usage: python3 tab_x2.py <run json files ...>"""
import json, sys, collections
import numpy as np


def recs(paths):
    for p in paths:
        for l in open(p):
            l = l.strip()
            if l:
                yield json.loads(l)


def marks_of(r):
    if 'score' in r:
        d = (r.get('drift') or {}).get('x2') or {}
        return d.get('marks', {})
    return (r.get('drift') or {}).get('marks', {})


def main():
    by = collections.defaultdict(dict)
    for r in recs(sys.argv[1:]):
        ms = marks_of(r)
        if ms:
            by[r['tag']][r['seed']] = ms
    allm = sorted({int(k) for a in by.values() for ms in a.values() for k in ms})
    print('arm'.rjust(12), 'seeds', ' '.join(f"{m:>16}" for m in allm), '| last: ok/partner/silent/echo')
    for tag, sd in by.items():
        cells = []
        for m in allm:
            v = [sd[s][str(m)]['acc'] for s in sorted(sd) if str(m) in sd[s]]
            cells.append(f"{np.mean(v):.2f} ({'/'.join(f'{x:.2f}' for x in v)})" if v else '-')
        last = []
        for s in sorted(sd):
            ks = sorted(sd[s], key=int)
            ro = sd[s][ks[-1]]['role']
            last.append('/'.join(f"{ro.get(c, 0):.2f}" for c in ('ok', 'partner', 'silent', 'echo')))
        print(f"{tag:>12} {','.join(map(str, sorted(sd))):>5} " + ' '.join(f"{c:>16}" for c in cells) + ' | ' + ' '.join(last))
    print('\nper segment (mean over seeds): online silent | RAL commits per dialogue | pool_old | pool_new')
    for tag, sd in by.items():
        row = []
        for m in allm:
            S, Cm, Po, Pn = [], [], [], []
            for s in sorted(sd):
                ks = sorted(sd[s], key=int)
                if str(m) not in sd[s] or 'seg' not in sd[s][str(m)]:
                    continue
                i = ks.index(str(m))
                prev = int(ks[i - 1]) if i else 0
                mm = sd[s][str(m)]
                S.append(mm['seg'].get('silent', 0))
                Cm.append(mm['delta'].get('ral_commit', 0) / max(1, m - prev))
                Po.append(mm['hip']['pool_old']); Pn.append(mm['hip']['pool'] - mm['hip']['pool_old'])
            row.append(f"{np.mean(S):.2f}|{np.mean(Cm):.2f}|{np.mean(Po):.0f}|{np.mean(Pn):.0f}" if S else '-')
        print(f"{tag:>12} " + ' '.join(f"{c:>16}" for c in row))
    print('\nlesions at the last lesion mark (acc: base / noN / nohip)')
    for tag, sd in by.items():
        out = []
        for s in sorted(sd):
            for k in sorted(sd[s], key=int, reverse=True):
                les = sd[s][k].get('lesion')
                if les:
                    out.append(f"s{s}@{k} {sd[s][k]['acc']:.2f}/{les['noN']['acc']:.2f}/{les['nohip']['acc']:.2f}")
                    break
        print(f"{tag:>12} " + '  '.join(out))


if __name__ == '__main__':
    main()
