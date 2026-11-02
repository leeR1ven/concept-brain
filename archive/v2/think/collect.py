# -*- coding: utf-8 -*-
"""Comparison tables from saved logs: per-task accuracy, held-out, latency, turn-taking, learning curves per stage
world (dialogue ranges A 0-299, B 300-2299, C 2300-4299, D 4300-6299).  usage: python3 collect.py name=prefix ..."""
import sys, pickle, json, collections
import numpy as np

FREE = ('probe', 'recast')
RANGES = {'B': (300, 2300), 'C': (2300, 4300), 'D': (4300, 6300)}


def stage_recs(L, st):
    a0, a1 = RANGES[st]
    return [r for r in L if a0 <= r['dlg'] < a1]


def summarize(L):
    out = {}
    for st in 'BCD':
        W = stage_recs(L, st)
        fr = [r for r in W if r['stage'] == st and r['kind'] in FREE]
        acc = lambda rs: float(np.mean([r['reply'] in r['acceptable'] for r in rs])) if rs else None
        lat = [r['latency'] for r in fr if r.get('latency') is not None]
        tasks = {}
        for t in sorted({r['task'] for r in fr}):
            rs = [r for r in fr if r['task'] == t]
            lt = [r['latency'] for r in rs if r.get('latency') is not None]
            tasks[t] = (acc(rs), round(float(np.mean(lt)), 1) if lt else None)
        rc = [r for r in W if r['kind'] == 'recast']
        stm = [r for r in W if r['kind'] == 'statement']
        seen = [r for r in fr if not r['held_out']]
        edges = np.linspace(*RANGES[st], 9)
        curve = [acc([r for r in seen if a <= r['dlg'] < b]) for a, b in zip(edges[:-1], edges[1:])]
        out[st] = dict(acc=acc(fr), held=acc([r for r in fr if r['held_out']]), seen=acc(seen), n=len(fr),
                       spoke=float(np.mean([bool(r['reply']) for r in fr])), lat_mean=float(np.mean(lat)) if lat else None,
                       lat_med=float(np.median(lat)) if lat else None, tasks=tasks,
                       after_recast=float(np.mean([bool(r.get('after_recast_reply')) for r in rc])) if rc else None,
                       after_statement=float(np.mean([bool(r['reply']) for r in stm])) if stm else None,
                       overlap=int(sum(r.get('mouth_while_partner', 0) for r in W)),
                       overlap_ex=int(sum(r.get('mouth_while_partner', 0) > 0 for r in W)),
                       extra=int(sum(r.get('extra_turns', 0) for r in fr)),
                       not3=int(sum(sum(x != 3 for x in r.get('runs', [])) for r in W)),
                       garble=int(sum(r.get('garble', 0) for r in W)),
                       maxlen=max([len(r['reply']) for r in fr] + [0]),
                       curve=curve, early=acc(seen[:len(seen) // 3]), late=acc(seen[-(len(seen) // 3):]))
    return out


def fmt(x, d=2):
    return '-' if x is None else (f'{x:.{d}f}' if isinstance(x, float) else str(x))


if __name__ == '__main__':
    runs = dict(a.split('=', 1) for a in sys.argv[1:])
    S = {k: summarize(pickle.load(open(v + '_log.pkl', 'rb'))) for k, v in runs.items()}
    names = list(S)
    print('| stage | metric | ' + ' | '.join(names) + ' |')
    print('|---|---|' + '---|' * len(names))
    for st in 'BCD':
        for m in ('acc', 'held', 'seen', 'spoke', 'lat_mean', 'lat_med', 'after_recast', 'after_statement', 'overlap',
                  'overlap_ex', 'extra', 'not3', 'garble', 'maxlen', 'early', 'late'):
            print(f'| {st} | {m} | ' + ' | '.join(fmt(S[n][st][m]) for n in names) + ' |')
        for t in sorted(S[names[0]][st]['tasks']):
            print(f'| {st} | {t} (acc / lat) | ' + ' | '.join(
                f"{fmt(S[n][st]['tasks'].get(t, (None, None))[0])} / {fmt(S[n][st]['tasks'].get(t, (None, None))[1], 1)}"
                for n in names) + ' |')
        print(f'| {st} | curve (8 bins, seen) | ' + ' | '.join(' '.join(fmt(c) for c in S[n][st]['curve']) for n in names) + ' |')
    json.dump(S, open('out/collect_' + '_'.join(names) + '.json', 'w'), default=str)
