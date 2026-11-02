# python3 -B final_tables.py RES.json  -> markdown tables for the report
import sys, json, numpy as np
R = json.load(open(sys.argv[1]))
skip = ('ALL', 'SCENE', 'NOSCENE', 'mech', 'after_model', 'learning', 'one_shot')
f = lambda x: '–' if x is None else f'{x:.2f}'
print('| stage/task | n | acc | seen | held-out | first char | echo-free | parrot |')
print('|---|---|---|---|---|---|---|---|')
for st in ('B', 'C', 'D'):
    for k, v in sorted(R.get(st, {}).items()):
        if isinstance(v, dict) and 'acc' in v and not k.endswith(skip):
            print(f"| {k} | {v['n']} | {f(v['acc'])} | {f(v.get('seen'))} ({v.get('n_seen')}) | {f(v.get('held'))} ({v.get('n_held')}) | {f(v.get('first'))} | {f(v.get('echo_free'))} | {f(v.get('parrot'))} |")
    a = R.get(st, {}).get(f'{st}/ALL')
    if a:
        print(f"| **{st}/ALL** | {a['n']} | **{f(a['acc'])}** | {f(a.get('seen'))} | {f(a.get('held'))} | {f(a.get('first'))} | {f(a.get('echo_free'))} | {f(a.get('parrot'))} |")
for st in ('B', 'C', 'D'):
    m = R.get(st, {}).get(f'{st}/mech'); l = R.get(st, {}).get(f'{st}/learning'); o = R.get(st, {}).get(f'{st}/one_shot')
    print(f"\n{st} mech: {m}\n{st} learning in free use: {l}\n{st} one-shot: {o}")
print('\nexamples:', {st: R.get(st + '_examples', [])[-8:] for st in 'BCD'})
print('\n| lesion (frozen copy, same probe stream) | B | C | D | mean |')
print('|---|---|---|---|---|')
for L, d in R.get('lesion', {}).items():
    row = []
    for st in 'BCD':
        accs = [v['acc'] for k, v in d.items() if k.startswith(st + '/') and isinstance(v, dict) and v.get('acc') is not None and not k.endswith(skip)]
        row.append(np.mean(accs) if accs else None)
    allv = [x for x in row if x is not None]
    print(f"| {L} | " + ' | '.join(f(x) for x in row) + f" | {f(np.mean(allv) if allv else None)} |")
keys = sorted({k for d in R.get('lesion', {}).values() for k, v in d.items() if isinstance(v, dict) and v.get('acc') is not None and not k.endswith(skip)})
print('\n| lesion | ' + ' | '.join(keys) + ' |'); print('|---|' + '---|' * len(keys))
for L, d in R.get('lesion', {}).items():
    print(f'| {L} | ' + ' | '.join(f(d[k]['acc']) if k in d and d[k].get('acc') is not None else '–' for k in keys) + ' |')
print('\ndrift (Jaccard of codes of 2 fixed probe lines, stage to stage):', R.get('drift'), '\nA -> end:', R.get('drift_A_to_end'))
print('stats:', R.get('stats'), 'programs', R.get('n_prog'), 'syl', R.get('n_syl'), 'chunks', R.get('n_chunks'), 'secs', R.get('secs'))
