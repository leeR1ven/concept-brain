# -*- coding: utf-8 -*-
"""What does ARCH_v2 say in life when it is wrong?  Existing log synth/out/life/final_noctx_s5_log.pkl (seed 5,
10^6 ticks).  Free replies (probe + recast) split into held-out / new situation (seen item) / repeated situation,
first vs second half of life; replies classified as correct / silent / echo of the prompt / a partner line (a
prompt line of the world or a >=2-char prefix of one, not an acceptable answer) / another answer / other.
usage: python3 life_errors.py"""
import os, pickle, sys, collections
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'synth', 'bench_copy'))
L = pickle.load(open(os.path.join(HERE, '..', '..', 'synth', 'out', 'life', 'final_noctx_s5_log.pkl'), 'rb'))
ex = [r for r in L if r.get('type') == 'ex' and r.get('kind') in ('probe', 'recast')]
prompts = set(r['prompt'] for r in L if r.get('type') == 'ex' and r.get('prompt'))
answers = set(a for r in L if r.get('type') == 'ex' and r.get('acceptable') for a in r['acceptable'])


def cls(r):
    y = r['reply']
    if r['acc']: return 'correct'
    if y == '': return 'silent'
    if y == r['prompt']: return 'echo_prompt'
    if y in prompts or (any(p.startswith(y) and len(y) >= 2 for p in prompts) and y not in answers): return 'partner_line'
    if y in answers: return 'other_answer'
    return 'other'


for name, sel in [('held', lambda r: r['held_out']),
                  ('new_sit_seen', lambda r: (not r['held_out']) and r.get('new_situation')),
                  ('repeated', lambda r: (not r['held_out']) and not r.get('new_situation'))]:
    for half, (a, b) in [('first half', (0, 500000)), ('second half', (500000, 2e9))]:
        R = [r for r in ex if sel(r) and a <= r['life_t'] < b]
        c = collections.Counter(cls(r) for r in R); n = len(R)
        print(f'{name:13s} {half:11s} n={n:4d} ', {k: round(v / n, 2) for k, v in sorted(c.items())})
late = [r for r in ex if r['life_t'] > 500000]
for task in ['what', 'slot', 'where', 'offer', 'taste', 'colour', 'count', 'check']:
    h = [r for r in late if r['task'] == task and r['held_out']]
    if h:
        print(task, 'late held n', len(h), 'acc', round(sum(r['acc'] for r in h) / len(h), 2),
              collections.Counter(r['reply'] for r in h).most_common(5))
