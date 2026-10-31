# -*- coding: utf-8 -*-
"""Tables from bar.py results: python3 summarize_bar.py results_dir > bar.md"""
import collections
import glob
import json
import os
import sys

import numpy as np

D = sys.argv[1] if len(sys.argv) > 1 else 'results'
runs = [json.load(open(f, encoding='utf-8')) for f in sorted(glob.glob(os.path.join(D, '*.json')))]
by = collections.defaultdict(list)
for r in runs:
    by[(r['agent'], r['stage'], r['tutor'])].append(r)


def ms(vals):
    v = [x for x in vals if x is not None]
    if not v:
        return '-'
    return f'{np.mean(v):.2f}' + (f'±{np.std(v):.2f}' if len(v) > 1 else '')


def get(r, st, *path):
    x = r['res'].get(st, {})
    for p in path:
        if not isinstance(x, dict):
            return None
        x = x.get(p)
    return x


AGENTS = ['refv1_mem', 'refv1_base', 'echo', 'last', 'prior', 'lookup', 'ngram', 'oracle', 'silent']
print('## Free-reply acceptable rate (mean ± sd over seeds; n = free replies per run)\n')
print('| agent | stage | tutor | seeds | n | ALL | held-out | seen | scene q. | no-scene q. | first char | echo-free | parrot |')
print('|---|---|---|---|---|---|---|---|---|---|---|---|---|')
for st in ('B', 'C', 'D', 'E'):
    for tutor in ('force', 'model', 'recast'):
        for ag in AGENTS:
            rs = by.get((ag, st, tutor))
            if not rs:
                continue
            print(f"| {ag} | {st} | {tutor} | {len(rs)} | {int(np.mean([get(r, st, 'ALL', 'n') or 0 for r in rs]))} | "
                  f"{ms([get(r, st, 'ALL', 'acc') for r in rs])} | {ms([get(r, st, 'ALL', 'held', 'acc') for r in rs])} | "
                  f"{ms([get(r, st, 'ALL', 'seen', 'acc') for r in rs])} | {ms([get(r, st, 'SCENE', 'acc') for r in rs])} | "
                  f"{ms([get(r, st, 'NOSCENE', 'acc') for r in rs])} | {ms([get(r, st, 'ALL', 'first') for r in rs])} | "
                  f"{ms([get(r, st, 'ALL', 'echo_free') for r in rs])} | {ms([get(r, st, 'ALL', 'parrot') for r in rs])} |")
print('\n## Per task (acceptable rate, all free replies; held-out in brackets)\n')
for st in ('B', 'C', 'D'):
    tasks = sorted({t for r in runs if r['stage'] == st for t in r['res'].get(st, {})
                    if t not in ('ALL', 'SCENE', 'NOSCENE', 'mech', 'guided', 'after_model', 'learning', 'one_shot', 'play')
                    and isinstance(r['res'][st][t], dict) and 'acc' in r['res'][st][t]})
    print(f'### stage {st}\n')
    print('| agent | tutor | ' + ' | '.join(tasks) + ' |')
    print('|---|---|' + '---|' * len(tasks))
    for tutor in ('force', 'model'):
        for ag in AGENTS:
            rs = by.get((ag, st, tutor))
            if not rs:
                continue
            cells = []
            for t in tasks:
                cells.append(f"{ms([get(r, st, t, 'acc') for r in rs])} ({ms([get(r, st, t, 'held', 'acc') for r in rs])})")
            print(f'| {ag} | {tutor} | ' + ' | '.join(cells) + ' |')
    print()
print('## RefV1 mechanics, guided speech, learning in free use, one-shot\n')
print('| agent | stage | tutor | programs | overlap ticks | latency | syllables | not 3 ticks | garble | extra turns | '
      'stop (world) | agent stop | guided exact | backterm hits | acc early→late (seen) | one-shot acc (n) | ms/tick |')
print('|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|')
for (ag, st, tutor), rs in sorted(by.items()):
    if not ag.startswith('refv1'):
        continue
    for r in rs:
        m = r['res'][st]['mech']; g = r['res'][st].get('guided', {}); L = r['res'][st].get('learning', {})
        o = r['res'][st].get('one_shot', {})
        print(f"| {ag} s{r['seed']} | {st} | {tutor} | {r.get('programs')} | {m['mouth_while_partner']} | "
              f"{m['latency']} | {m['syllables']} | {m['not_3_ticks']} | {m['garble']} | {m['extra_turns']} | "
              f"{m['stop']} | {m['agent_stop']} | {g.get('exact')} | {r.get('backterm_hits')} | "
              f"{ms([L.get('acc_early')])}→{ms([L.get('acc_late')])} | {ms([o.get('acc')])} ({o.get('n')}) | "
              f"{r['ms_per_tick']} |")
print('\n## Stage E (corpus) 20-way content metric (ARCH_v1 run.py definition)\n')
print('| agent | seeds | never-heard top-1 | AUC | copy (shuffled) | familiar top-1 | AUC |')
print('|---|---|---|---|---|---|---|')
for ag in AGENTS:
    rs = by.get((ag, 'E', 'force'))
    if not rs:
        continue
    print(f"| {ag} | {len(rs)} | {ms([get(r, 'E', 'corpus_20way', 'top1') for r in rs])} | "
          f"{ms([get(r, 'E', 'corpus_20way', 'auc') for r in rs])} | {ms([get(r, 'E', 'corpus_20way', 'copy') for r in rs])} "
          f"({ms([get(r, 'E', 'corpus_20way', 'copy_shuf') for r in rs])}) | "
          f"{ms([get(r, 'E', 'corpus_fam_20way', 'top1') for r in rs])} | {ms([get(r, 'E', 'corpus_fam_20way', 'auc') for r in rs])} |")
print('\n## Examples (RefV1 mem, force, seed 0, last probes)\n')
for st in ('B', 'C', 'D'):
    rs = by.get(('refv1_mem', st, 'force'))
    if rs:
        r = [x for x in rs if x['seed'] == 0] or rs
        print(f'- {st}: ' + '; '.join(f"{p}→{y or '∅'} (want {a}{', held' if h else ''})" for t, p, y, a, h in r[0]['examples'][-10:]))
print('\n## Runtime\n')
for (ag, st, tutor), rs in sorted(by.items()):
    print(f"- {ag} {st} {tutor}: {ms([r['secs'] for r in rs])} s, {ms([r['ticks'] for r in rs])} ticks, "
          f"{ms([r['ms_per_tick'] for r in rs])} ms/tick")
