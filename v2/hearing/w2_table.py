# -*- coding: utf-8 -*-
"""Table of the full-agent world2 runs (out/w2/*.json): per condition, v2.1 front end vs hearing front end; values are
the 2 full agents' sums (lines, episodes ...) or means (brake share)."""
import os, glob, json
HERE = os.path.dirname(os.path.abspath(__file__))
rows = {}
for f in sorted(glob.glob(os.path.join(HERE, 'results', 'w2', 'w2_*.json'))):
    for l in open(f):
        d = json.loads(l)
        rows[d['tag']] = d
conds = sorted({t.split('_', 2)[2] for t in rows}, key=lambda c: (int(c.split('_')[0][1:]), c))
hdr = ('cond', 'front', 'true near cg', 'lines', 'items', 'cats', 'quiet', 'recalls', 'episodes', 'released', 'spoke',
       'refused', 'F on', 'talk-over partner', 'free acc (n)')
print('| ' + ' | '.join(hdr) + ' |'); print('|' + '---|' * len(hdr))
for c in conds:
    for fe in ('v21', 'H'):
        d = rows.get(f'w2_{fe}_{c}')
        if d is None:
            continue
        pa = d.get('per_agent') or []
        acc = [p.get('acc') for p in pa if p.get('acc') is not None]
        mech = d.get('mech') or []
        vals = (c, fe, sum(d['true_cg_near']), sum(d['lines']), sum(d['items']), '/'.join(map(str, d['cats'])),
                sum(d['quiet']), sum(d['awake_recalls']), sum(d['hip']), sum(d['released']), sum(d['spoke']),
                sum(d['go_blocked']), round(sum(d['F_on']) / len(d['F_on']), 3),
                sum(m.get('overlap_partner', 0) for m in mech),
                (f"{sum(acc) / len(acc):.3f} ({sum(p.get('n_free', 0) for p in pa)})" if acc else '-'))
        print('| ' + ' | '.join(map(str, vals)) + ' |')
