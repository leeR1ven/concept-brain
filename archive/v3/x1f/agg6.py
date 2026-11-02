"""aggregate out6/<arm>_s*.json: mean +- SD over seeds.  python3 -I agg6.py <arm> [dir]"""
import json, glob, sys, os
import numpy as np

arm = sys.argv[1]
d = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out6')
R = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(d, f'{arm}_s[0-9].json')))]
print(arm, 'seeds', [r['seed'] for r in R])
if not R:
    sys.exit()


def ms(v):
    v = [x for x in v if x is not None]
    return f'{np.mean(v):.2f}±{np.std(v):.2f}' if v else '-'


keys = list(R[0]['probes'].keys())
print('checkpoint | ' + ' | '.join(f'{s}: acc / first / silent / garble' for s in ('base', 'new', 'held')) + ' | path12 base/new/held | spec(new) own-other | n3 mean / p_sat')
for k in keys:
    if k == 't1_forced_answer':
        continue
    row = []
    for s in ('base', 'new', 'held'):
        vals = [r['probes'][k].get(s) for r in R]
        if not any(vals):
            row.append('-')
            continue
        row.append(' / '.join(ms([v[m] for v in vals if v]) for m in ('acc', 'first', 'silent', 'garble')))
    p12 = ' / '.join(ms([r['path'][k].get(s, {}).get('12', r['path'][k].get(s, {}).get(12)) if r['path'].get(k, {}).get(s) else None for r in R]) for s in ('base', 'new', 'held'))
    sp = ms([r['spec'][k]['new']['own_max'] - r['spec'][k]['new']['other_max'] for r in R if 'new' in r['spec'].get(k, {})])
    cn = ms([r['counts'].get(k, {}).get('n3_mean') for r in R]) + ' / ' + ms([r['counts'].get(k, {}).get('p_sat') for r in R])
    print(f'{k} | ' + ' | '.join(row) + f' | {p12} | {sp} | {cn}')
print('forced whole answer (seen):', 'acc', ms([r['probes']['t1_forced_answer']['acc'] for r in R]), 'garble', ms([r['probes']['t1_forced_answer']['garble'] for r in R]))
for k in [k for k in R[0]['mech'] if not k.endswith('_sets')]:
    m = [r['mech'][k] for r in R]
    sets = [r['mech'].get(k + '_sets', {}) for r in R]
    print(f'mech {k}: resp {ms([x["resp_rate"] for x in m])} acc {ms([x["acc"] for x in m])} talkover {ms([x["talkover_per_ex"] for x in m])} '
          f'yieldfail {ms([x["yieldfail_per_ex"] for x in m])} garble {ms([x["garble_per_ex"] for x in m])} extra {ms([x["extra_per_ex"] for x in m])} | '
          + ' '.join(f'{s} acc {ms([x.get(s, {}).get("acc") for x in sets])}' for s in ('base', 'new')))
for k in R[0].get('overhear', {}):
    m = [r['overhear'][k] for r in R]
    print(f'overhear {k}: child utts/dlg {ms([x["child_utts_per_dlg"] for x in m])} garbled/dlg {ms([x["garbled_per_dlg"] for x in m])} '
          f'in Q gap {ms([x["in_q_gap_per_dlg"] for x in m])} during adult speech {ms([x["during_adult_speech_per_dlg"] for x in m])}')
for r in R:
    c = r.get('correction')
    if c:
        print('seed', r['seed'], 'correction', {k: (v['set'], v['was_correct'], v['first_new_after_teachings'], v['seq']) for k, v in c['latency'].items()},
              'probe new/old', c['probe_new'], c['probe_old'], 'unchanged before/after', c['unchanged_before'], c['unchanged_after'])
if 't1_phase1' in R[0]['path']:
    for k in ('t1_phase1', 'O_phase1'):
        if k in R[0]['path']:
            print(k, 'same@0/6/12', ms([r['path'][k]['same'][0] for r in R]), ms([r['path'][k]['same'][6] for r in R]), ms([r['path'][k]['same'][12] for r in R]),
                  'cross@0/12', ms([r['path'][k]['cross'][0] for r in R]), ms([r['path'][k]['cross'][12] for r in R]),
                  'decQ@0/6/12', ms([r['path'][k]['dec_q'][0] for r in R]), ms([r['path'][k]['dec_q'][6] for r in R]), ms([r['path'][k]['dec_q'][12] for r in R]),
                  'decFrame@12', ms([r['path'][k]['dec_frame'][12] for r in R]))
print('time_s', ms([r['time_s'] for r in R]), 'route', R[0].get('speech_route'))
