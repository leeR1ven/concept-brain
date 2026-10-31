# -*- coding: utf-8 -*-
"""Ear-alone table: v2.1 front end (tag _base, variant v21) vs a hearing variant (tag given, variant gate).
usage: python3 ear_table.py _final"""
import os, sys, glob, json
HERE = os.path.dirname(os.path.abspath(__file__))
tag = sys.argv[1] if len(sys.argv) > 1 else '_final'


def load(t, var):
    out = {}
    for f in glob.glob(os.path.join(HERE, 'results', 'ear', f'*{t}.json')):
        for l in open(f):
            d = json.loads(l)
            if var in d['variants']:
                out[d['rec']] = d['variants'][var]['mean']
    return out


base, new = load('_v21pure', 'v21'), load(tag, 'gate')
order = ['w2_N2_p0.0_s0.0', 'w2_N2_p0.0_s0.5', 'w2_N2_p0.0_s1.0', 'w2_N2_p0.0_s2.0', 'w2_N16_p0.004_s0.0',
         'w2_N16_p0.004_s1.0', 'w2_N16_p0.02_s0.0', 'w2_N16_p0.02_s0.5', 'w2_N16_p0.02_s1.0', 'w2_N64_p0.02_s0.0',
         'w2_N64_p0.02_s0.5', 'w2_N64_p0.02_s1.0']
hdr = ('world (N, chatter p, σ)', 'front end', 'lines found / true', 'precision', 'near recall', 'near coverage (overlapped)',
       'categories (junk / dup)', 'purity', 'quiet moments/day', 'silences ≥200/day', 'F on')
print('| ' + ' | '.join(hdr) + ' |'); print('|' + '---|' * len(hdr))
for o in order:
    k = o + '_seed0.pkl'
    for name, d in (('v2.1', base.get(k)), ('hearing', new.get(k))):
        if d is None:
            continue
        cov = d.get('coverage', {})
        n, p, s = o.split('_')[1:4]
        vals = (f"{n[1:]}, {p[1:]}, {s[1:]}", name, f"{d['detected']:.0f} / {d['true_lines']:.0f}", d['precision'],
                d['recall'].get('near'), f"{cov.get('near', '-')} ({cov.get('near_overlapped', '-')})",
                f"{d['cats']:.0f} ({d['junk_cats']:.0f} / {d.get('dup_cats', '-')})", d['purity'],
                d['quiet_moments_per_day'], d['long_silences_per_day'], d['F_on_frac'])
        print('| ' + ' | '.join(map(str, vals)) + ' |')
