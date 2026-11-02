"""X6b extra tables from the result JSONs (aggregation only, no runs).
python3 -I x6b_extra.py facts <dir> <arm> [<arm> ...]   one-shot facts, own-scene cue (R1s): what came out instead of
                                                          the name, and R1s by delay
python3 -I x6b_extra.py pair <dir> <armA> <armB>          paired per-seed difference A - B on the main metrics"""
import collections
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import x6b_analyze as AN  # noqa: E402


def facts(d, arms):
    print('| arm | life | n | right name | another fact\'s name | other char | nothing | R1s at delay 1k / 2k / 5k / 10k |')
    print('|---|---|---|---|---|---|---|---|')
    for arm in arms:
        for life in ('fresh', 'adult'):
            cnt = collections.Counter(); byd = collections.defaultdict(list)
            for f in sorted(glob.glob(os.path.join(d, f'{life}_{arm}_s*.json'))):
                x = json.load(open(f)); ch = x['chars']
                names = {y['name'][0] for y in x['facts']}
                for r in x['fact']:
                    g = r['R1s'][0]
                    c = ch[g] if g >= 0 else None
                    k = ('right' if c == r['name'][0] else 'other fact' if c in names else 'nothing' if c is None
                         else 'other char')
                    cnt[k] += 1; byd[r['delay']].append(r['R1s_ok'])
            n = sum(cnt.values())
            if not n:
                continue
            print(f'| {arm} | {life} | {n} | ' + ' | '.join(f'{cnt[k] / n:.2f}' for k in ('right', 'other fact', 'other char', 'nothing'))
                  + ' | ' + ' / '.join(f'{np.mean(byd[k]):.2f}' for k in sorted(byd)) + ' |')


KEYS = ['r50', 'r50d', 'i100', 'ov', 'ov6', 'ovx', 'fresh_R1s', 'adult_R1s', 'fresh_I1', 'adult_I1', 'qaP_right',
        'qaP_wrong', 'sep_pS', 'sep_pV', 'sep_pD', 'A50']


def pair(d, a, b):
    R = AN.load(d, '*.json')
    A, B = AN.per_arm(R, a), AN.per_arm(R, b)
    print(f'| metric | {a} | {b} | {a} - {b}, paired | seeds {a} > {b} |')
    print('|---|---|---|---|---|')
    for k in KEYS:
        p = 0 if k == 'A50' else 2
        va = [A[s][k] for s in sorted(A) if k in A[s]]
        vb = [B[s][k] for s in sorted(B) if k in B[s]]
        dd = [A[s][k] - B[s][k] for s in sorted(A) if s in B and k in A[s] and k in B[s]
              and not np.isnan(A[s][k]) and not np.isnan(B[s][k])]
        print(f'| {k} | {AN.fmt(va, p)} | {AN.fmt(vb, p)} | {AN.fmt(dd, p)} | {sum(x > 0 for x in dd)}/{len(dd)} |')


if __name__ == '__main__':
    if sys.argv[1] == 'facts':
        facts(sys.argv[2], sys.argv[3:])
    else:
        pair(sys.argv[2], sys.argv[3], sys.argv[4])
