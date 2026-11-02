"""aggregate out8/*.json (or a given folder) into per-arm tables: mean +- SD over seeds.
python3 -I agg8.py <folder> [<folder> ...] [ck=O10000]"""
import sys, os, json, glob, re
import numpy as np

folders = [a for a in sys.argv[1:] if '=' not in a] or ['out8']
kv = dict(a.split('=', 1) for a in sys.argv[1:] if '=' in a)
runs = {}
for fo in folders:
    for f in sorted(glob.glob(os.path.join(fo, '*.json'))):
        r = json.load(open(f))
        arm = re.sub(r'_s\d+$', '', os.path.basename(f)[:-5])
        runs.setdefault(arm, []).append(r)


def ms(v, d=3):
    v = [x for x in v if x is not None]
    if not v:
        return '-'
    if len(v) == 1:
        return f'{v[0]:.{d}f}'
    return f'{np.mean(v):.{d}f}±{np.std(v):.{d}f}'


def g(r, path, default=None):
    cur = r
    for p in path:
        if isinstance(cur, dict) and p in cur:
            cur = cur[p]
        elif isinstance(cur, list) and isinstance(p, int) and p < len(cur):
            cur = cur[p]
        else:
            return default
    return cur


def table_curve(arm, rs):
    cks = [c for c in rs[0]['ck'] if c.startswith('O') or c == 't1']
    tgt = 'new' if 'new' in rs[0]['ck']['t1']['probe'] else 'seen'
    print(f'\n### {arm}  (n={len(rs)} seeds; target set = {tgt})')
    print('| checkpoint | first syl (target) | acc (target) | silent | garble | held acc | fair spec (seen) | null95 | hub share | dec lag0/6/12 (all 32) | within J | drive ans / best wrong | ans top | L3 mean | P(L3 empty) | child utts/dlg | garbled/dlg |')
    print('|' + '---|' * 17)
    for c in cks:
        row = [c]
        row.append(ms([g(r, ['ck', c, 'probe', tgt, 'first']) for r in rs], 2))
        row.append(ms([g(r, ['ck', c, 'probe', tgt, 'acc']) for r in rs], 2))
        row.append(ms([g(r, ['ck', c, 'probe', tgt, 'silent']) for r in rs], 2))
        row.append(ms([g(r, ['ck', c, 'probe', tgt, 'garble']) for r in rs], 2))
        row.append(ms([g(r, ['ck', c, 'probe', 'held', 'acc']) for r in rs], 2))
        row.append(ms([g(r, ['ck', c, 'fair', 'seen', 'fair']) for r in rs]))
        row.append(ms([g(r, ['ck', c, 'fair', 'seen', 'null_95']) for r in rs], 2))
        row.append(ms([g(r, ['ck', c, 'fair', 'seen', 'hub_share']) for r in rs], 2))
        row.append('/'.join(ms([g(r, ['ck', c, 'path', 'all', str(k)], g(r, ['ck', c, 'path', 'all', k])) for r in rs], 2) for k in (0, 6, 12)))
        row.append(ms([g(r, ['ck', c, 'motion', 'within_J1_mean']) for r in rs], 2))
        dset = tgt if tgt != 'seen' else 'seen'
        row.append(ms([g(r, ['ck', c, 'drive', dset, 'ans']) for r in rs], 2) + ' / ' + ms([g(r, ['ck', c, 'drive', dset, 'best_wrong']) for r in rs], 2))
        row.append(ms([g(r, ['ck', c, 'drive', dset, 'answer_is_top']) for r in rs], 2))
        row.append(ms([g(r, ['ck', c, 'counts', 'n3_mean']) for r in rs], 0))
        row.append(ms([g(r, ['ck', c, 'counts', 'p_empty']) for r in rs], 2))
        row.append(ms([g(r, ['ck', '_overhear', c, 'child_utts_per_dlg']) for r in rs], 2))
        row.append(ms([g(r, ['ck', '_overhear', c, 'garbled_per_dlg']) for r in rs], 2))
        print('| ' + ' | '.join(row) + ' |')
    last = cks[-1]
    sw = g(rs[0], ['ck', last, 'swap'])
    if sw:
        print(f'\nreadout swap at {last} (same learned weights; first syllable / acc on target set; fair spec on seen):')
        print('| readout | first | acc | silent | garble | fair (seen) |')
        print('|---|---|---|---|---|---|')
        for sn in sw:
            print(f'| {sn} | ' + ' | '.join([ms([g(r, ['ck', last, 'swap', sn, tgt, k]) for r in rs], 2) for k in ('first', 'acc', 'silent', 'garble')]
                                           + [ms([g(r, ['ck', last, 'swap', sn, 'seen', 'fair']) for r in rs])]) + ' |')
    p1 = [r.get('path_ctx_end') for r in rs if r.get('path_ctx_end')]
    p0 = [r.get('path_ctx_t1') for r in rs if r.get('path_ctx_t1')]
    for nm, P in (('t1 (newborn)', p0), ('end', p1)):
        if P:
            print(f"context-varied path ({nm}): dec_q lag0/6/12 " + '/'.join(ms([p['dec_q'][k] for p in P], 3) for k in (0, 6, 12))
                  + ' | same/cross@12 ' + ms([p['same'][12] for p in P], 2) + '/' + ms([p['cross'][12] for p in P], 2)
                  + ' | within J1 (lags 0-11) ' + ms([float(np.mean(p['within_J1'][:12])) for p in P], 2)
                  + ' | P(empty)@12 ' + ms([p['p_empty_lag'][12] for p in P], 2)
                  + ' | dec frame@12 ' + ms([p['dec_frame'][12] for p in P], 2))
    lv = [r.get('live') for r in rs if r.get('live')]
    if lv:
        m = [x['mech'] for x in lv]
        print('live (frozen, format caregiver): resp ' + ms([x['resp_rate'] for x in m], 2) + ' acc ' + ms([x['acc'] for x in m], 2)
              + ' talk-over/ex ' + ms([x['talkover_per_ex'] for x in m], 3) + ' yield-fail/ex ' + ms([x['yieldfail_per_ex'] for x in m], 3)
              + ' garble/ex ' + ms([x['garble_per_ex'] for x in m], 3)
              + (' | t1 forced answer said ' + ms([g(r, ['t1', 'forced_answer', 'acc']) for r in rs], 2)))
    print('time s: ' + ms([r.get('time_s') for r in rs], 0) + ' ticks ' + ms([r.get('ticks') for r in rs], 0))


for arm, rs in sorted(runs.items()):
    if 'ck' in rs[0] and 't1' in rs[0]['ck']:
        table_curve(arm, rs)
