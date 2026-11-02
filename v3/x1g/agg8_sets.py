"""per-set companion to agg8.py: for every arm and checkpoint, first syllable / silent, fair specificity with its
permutation null, and learned drive, split by pair set (seen / base / new / held), plus L33 pair fraction and the live
worlds split by set.  mean +- SD over seeds.   python3 -I agg8_sets.py <folder> [<folder> ...]"""
import sys, os, json, glob, re
import numpy as np

runs = {}
for fo in sys.argv[1:] or ['out8']:
    for f in sorted(glob.glob(os.path.join(fo, '*.json'))):
        arm = re.sub(r'_s\d+$', '', os.path.basename(f)[:-5])
        runs.setdefault(arm, []).append(json.load(open(f)))


def g(r, *path):
    for p in path:
        if not isinstance(r, dict) or p not in r:
            return None
        r = r[p]
    return r


def ms(v, d=2):
    v = [x for x in v if x is not None]
    if not v:
        return '-'
    return f'{v[0]:.{d}f}' if len(v) == 1 else f'{np.mean(v):.{d}f}±{np.std(v):.{d}f}'


for arm, rs in sorted(runs.items()):
    sets = [s for s in ('seen', 'base', 'new', 'held') if g(rs[0], 'ck', 't1', 'probe', s)]
    print(f'\n### {arm}  (n={len(rs)} seeds)')
    print('| checkpoint | set | first syl | silent | fair spec | null 95 % | top is answer | drive ans / best wrong | answer is top-driven |')
    print('|---|---|---|---|---|---|---|---|---|')
    cks = [c for c in rs[0]['ck'] if c == 't1' or c.startswith('O')]
    for c in cks:
        for s in sets:
            print(f'| {c} | {s} | ' + ' | '.join([
                ms([g(r, 'ck', c, 'probe', s, 'first') for r in rs]), ms([g(r, 'ck', c, 'probe', s, 'silent') for r in rs]),
                ms([g(r, 'ck', c, 'fair', s, 'fair') for r in rs], 3), ms([g(r, 'ck', c, 'fair', s, 'null_95') for r in rs]),
                ms([g(r, 'ck', c, 'fair', s, 'top_is_answer') for r in rs]),
                ms([g(r, 'ck', c, 'drive', s, 'ans') for r in rs]) + ' / ' + ms([g(r, 'ck', c, 'drive', s, 'best_wrong') for r in rs]),
                ms([g(r, 'ck', c, 'drive', s, 'answer_is_top') for r in rs])]) + ' |')
    last = cks[-1]
    print(f'weights at {last}: L33 pairs > 0 ' + ms([g(r, 'ck', last, 'weights', 'L33_pos') for r in rs], 3)
          + ' | top -> L1 links > 0 ' + ms([g(r, 'ck', last, 'weights', 'L31_pos') for r in rs], 3))
    for lv in ('live', 'live_real'):
        L = [r.get(lv) for r in rs if r.get(lv)]
        if L:
            print(f'{lv}: ' + ' | '.join(f'{s} acc ' + ms([g(x, "sets", s, "acc") for x in L]) + ' resp ' + ms([g(x, "sets", s, "resp") for x in L])
                                        for s in ('seen', 'base', 'new') if g(L[0], 'sets', s))
                  + ' | talk-over/ex ' + ms([g(x, 'mech', 'talkover_per_ex') for x in L], 3)
                  + ' yield-fail/ex ' + ms([g(x, 'mech', 'yieldfail_per_ex') for x in L], 3)
                  + ' garble/ex ' + ms([g(x, 'mech', 'garble_per_ex') for x in L], 3))
