# -*- coding: utf-8 -*-
"""Life runs side by side (seeds 5 and 6): v2.1 life metrics, same-life references, and the step-3 diagnostics.
usage: python3 life_cmp.py tag1 tag2 ... [seeds=5,6]"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
A = dict(a.split('=') for a in sys.argv[1:] if '=' in a)
tags = [a for a in sys.argv[1:] if '=' not in a]
seeds = [int(s) for s in A.get('seeds', '5,6').split(',')]
f2 = lambda x: '-' if x is None else f'{x:.2f}'
hdr = 'tag            seed  free  last  e|c   scene held social mem  role  init nons  appr apprL  nihaoE(last,mean) greet  corrCommit qCommit  spokeAfterCorr'
print(hdr)
for t in tags:
    for s in seeds:
        p = os.path.join(HERE, 'out', 'life', f'{t}_s{s}.json')
        if not os.path.exists(p):
            print(f'{t:14s} {s}  (missing)'); continue
        d = json.load(open(p))
        m = d.get('v21')
        if m is None:
            print(f'{t:14s} {s}  (unfinished)'); continue
        f = m['free']; g = d.get('diag', {})
        rb = g.get('ral_by_type', {}); at = g.get('after_type', {})
        ne = m['nihao_empty_per_tenth']
        print(f"{t:14s} {s}  {f2(f['all']['agent'])}  {f2(f['all']['agent_last_tenth'])}  {f2(f['all']['echo_or_const'])}  "
              f"{f2(f['scene']['agent'])}  {f2(f['held_out']['agent'])}  {f2(f['social']['agent'])}  {f2(f.get('memory', {}).get('agent'))}  "
              f"{f2(f['all']['role_conf'])}  {m['n_init']:4d} {m['categories'].get('nonsense', 0):4d}  {f2(m['appropriate'])}  "
              f"{f2(m['appropriate_last_tenth'])}  {ne[-1]:3d} {sum(ne) / 10:5.1f}   {m['categories'].get('greeting', 0):4d}   "
              f"{f2(rb.get('correct', {}).get('commit_share'))}/{f2(rb.get('tell_silent', {}).get('commit_share'))}  "
              f"{f2(rb.get('question', {}).get('commit_share'))}   {f2(at.get('correct', {}).get('spoke'))}")
