"""regression: X3 arm in x6b vs X6's own runs (same seeds). python3 -I regress.py"""
import json, os, numpy as np
X6 = '../x6/out'
pairs = [('uni_X3', 'q18_B0', ('r50', 'i100', 'r50d', 'i100x')), ('qa_X3', 'q17_A1w', ('r50', 'i100')),
         ('sep_X3', 'dsep_r1.0', ('r50', 'i100')), ('fresh_X3', 'q22_fresh', ()), ('adult_X3', 'q22_adult', ())]
for mine, theirs, keys in pairs:
    for s in range(5):
        a, b = f'out/{mine}_s{s}.json', f'{X6}/{theirs}_s{s}.json'
        if not (os.path.exists(a) and os.path.exists(b)):
            continue
        A, B = json.load(open(a)), json.load(open(b))
        same = all(A['all'][i].get(k) == B['all'][i].get(k) for i in range(min(len(A['all']), len(B['all']))) for k in keys)
        extra = ''
        if A['qa']:
            extra = ' qa P same: ' + str([x['P_first'] for x in A['qa']] == [x['P_first'] for x in B['qa']])
        if A['fact']:
            extra = ' fact same: ' + str([x['R1'] for x in A['fact']] == [x['R1'] for x in B['fact']])
        print(mine, s, 'rows', len(A['all']), len(B['all']), 'identical', same, 'live', A['live'], B['live'], extra)
