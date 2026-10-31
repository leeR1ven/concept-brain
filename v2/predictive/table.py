# per-task table: my agent (main run) vs the bench reference (seed 0, 2,000 dialogues; README v1.2)
import json, sys
REF = {'B:slot': (0.00, 0.04, 0.06), 'B:what': (0.06, 0.06, 0.03), 'B:where': (0.33, 0.33, 0.32), 'B:ALL': (0.12, 0.14, 0.13),
       'C:colour': (0.08, 0.14, 0.19), 'C:count': (0.08, 0.32, 0.30), 'C:offer': (0.04, 0.09, 0.49), 'C:ritual': (0.51, 0.21, 0.99),
       'C:taste': (0.46, 1.00, 1.00), 'C:ALL': (0.19, 0.31, 0.54), 'D:before': (0.01, 0.08, 0.09), 'D:check': (0.49, 0.48, 0.30),
       'D:name': (0.00, 0.17, 0.17), 'D:oneshot': (0.00, 0.00, 0.00), 'D:ALL': (0.21, 0.24, 0.16)}
res = json.load(open(sys.argv[1]))
f = lambda x: '  -  ' if x is None else f'{float(x):.2f}'
print('| task | n | mine acc | mine seen | mine held | first char | bar (v1 force) | prior | ngram |')
print('|---|---|---|---|---|---|---|---|---|')
for st in ('B', 'C', 'D'):
    if st not in res: continue
    S = res[st][st]
    for task in sorted(S):
        r = S[task]
        if not isinstance(r, dict) or 'held' not in r or not r.get('n'): continue
        if task in ('SCENE', 'NOSCENE'): continue
        key = f'{st}:{task}'; ref = REF.get(key, (None, None, None))
        print(f"| {key} | {r['n']} | {f(r['acc'])} | {f(r['seen']['acc'])} | {f(r['held']['acc'])} | {f(r['first'])} | {f(ref[0])} | {f(ref[1])} | {f(ref[2])} |")
    m = res[st][st]['mech']
    print(f"| {st} mechanics | overlap {m['mouth_while_partner']}, latency median {m['latency_median']}, not-3-tick {m['not_3_ticks']}/{m['syllables']}, garble {m['garble']}, extra turns {m['extra_turns']}, stops {m['stop']} | | | | | | | |")
