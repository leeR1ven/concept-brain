import json, sys
def last(p, tag=None):
    r = None
    for l in open(p):
        x = json.loads(l)
        if tag is None or x['tag'] == tag:
            r = x
    return r
SYN = '../../../v2/synth/results/runs/'
runs = [('ARCH_v2 nS1000', last(SYN + 'final_noctx_s0.json'))] + [(t, last('veto_j_runs.json', t)) for t in sys.argv[1:]]
def J(r):
    t = r['table']; a = lambda s, st, h=False: t[s][st]['ALL'][2 if h else 0] or 0
    F = (a('s3', 'B') + a('s3', 'C') + a('s3', 'D')) / 3
    RET = (min(0, a('s3', 'B') - a('s1', 'B')) + min(0, a('s3', 'C') - a('s2', 'C'))) / 2
    H = (a('s3', 'B', 1) + a('s3', 'C', 1) + a('s3', 'D', 1)) / 3
    DR = r['drift']['after']['ALL'][0]
    return F + 0.5 * RET + 0.25 * H + 0.25 * DR, F, RET, H, DR
for name, r in runs:
    if r is None: continue
    j = J(r)
    print(f"{name:16s} J {j[0]:.3f} F {j[1]:.3f} RET {j[2]:.3f} H {j[3]:.3f} DR {j[4]:.3f} drift {r['drift']['before']['ALL'][0]:.2f}->{r['drift']['after']['ALL'][0]:.2f} viol {r['viol']}")
tasks = ['what', 'slot', 'where', 'offer', 'taste', 'colour', 'count', 'ritual', 'check', 'name', 'before', 'oneshot']
print('task (s3: all / held)  ' + ' | '.join(n for n, r in runs if r))
for k in tasks:
    row = []
    for n, r in runs:
        if r is None: continue
        v = None
        for st in 'BCD':
            if k in r['table']['s3'][st]:
                v = r['table']['s3'][st][k]
        row.append('-' if v is None else f"{v[0]:.2f} / {('-' if v[2] is None else f'{v[2]:.2f}')} (n {v[1]}/{v[3]})")
    print(f'{k:8s}', ' | '.join(row))
