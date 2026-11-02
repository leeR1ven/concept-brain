"""aggregate X1e results.
  python3 -I agg3.py grid <dir>                 one line per life (screening), sorted by question decoding at lag 12
  python3 -I agg3.py arms <dir> arm1,arm2,...   5-seed means +- SD of every reported metric, pass criteria
"""
import json, glob, os, sys
import numpy as np


def load(fn):
    return json.load(open(fn))


def teach_tag(r):
    return max((k for k in r['probes'] if k.startswith('teach')), key=lambda k: int(k[5:]))


def line(r, P='teach'):
    p = r['path'][P]
    s, c = np.array(p['same']), np.array(p['cross'])
    tl = teach_tag(r)
    sp = r['spec'][tl]
    h = r['hub'][P]
    return dict(dq0=p['dec_q'][0], dq6=p['dec_q'][6], dq12=p['dec_q'][12], dq10_12=round(float(np.mean(p['dec_q'][10:13])), 3),
                dfr12=p['dec_frame'][12], same12=round(s[12], 2), cross12=round(c[12], 2), gap12=round(float(s[12] - c[12]), 3),
                within=round(float(np.mean(p['within_J1'][1:13])), 2), dmg12=p['damage']['0.1'][12],
                n3_12=round(p['n3_lag'][12]), pempty12=p['p_empty_lag'][12],
                spec=round(sp['own_max'] - sp['other_max'], 3), own=sp['own_max'], other=sp['other_max'],
                hub=sp['hub_share'], hubsyl=sp['hub_syl'], first=r['probes'][tl]['seen']['first'], acc=r['probes'][tl]['seen']['acc'],
                L33pos=h['L33_pos_frac'], stuck=p['motion_long_silence']['stuck20'], turn=p['motion_long_silence']['turnover1'],
                n3mean=r['layer_counts']['n3']['mean'], n3max=r['layer_counts']['n3']['max'], t=r['time_s'])


def grid(d):
    rows = []
    for fn in sorted(glob.glob(os.path.join(d, '*_s0.json'))):
        try:
            r = load(fn)
            rows.append((os.path.basename(fn)[:-8], line(r)))
        except Exception as e:  # noqa
            print('skip', fn, e)
    rows.sort(key=lambda x: -x[1]['dq10_12'])
    keys = ['dq0', 'dq6', 'dq12', 'dq10_12', 'dfr12', 'same12', 'cross12', 'gap12', 'within', 'dmg12', 'n3_12', 'pempty12',
            'spec', 'hub', 'first', 'L33pos', 'stuck', 'turn', 'n3mean', 'n3max']
    print('name'.ljust(46), ' '.join(k[:7].rjust(7) for k in keys))
    for n, l in rows:
        print(n[:46].ljust(46), ' '.join(f'{l[k]:7.3f}' if isinstance(l[k], float) else str(l[k]).rjust(7) for k in keys))
    return rows


def ms(xs, d=3):
    xs = [x for x in xs if x is not None]
    return f'{np.mean(xs):.{d}f}±{np.std(xs):.{d}f}' if xs else 'n/a'


def arms(d, names):
    out = {}
    for a in names:
        R = [load(f) for f in sorted(glob.glob(os.path.join(d, f'{a}_s[0-9].json')))]
        if not R:
            continue
        o = dict(n=len(R))
        L = [line(r) for r in R]
        for k in ('dq0', 'dq6', 'dq12', 'dq10_12', 'dfr12', 'same12', 'cross12', 'gap12', 'within', 'dmg12', 'n3_12', 'pempty12',
                  'own', 'other', 'spec', 'hub', 'first', 'acc', 'L33pos', 'stuck', 'turn', 'n3mean', 'n3max'):
            o[k] = ms([l[k] for l in L])
        o['hubsyl'] = ''.join(l['hubsyl'] for l in L)
        tl = teach_tag(R[0])
        for ph in [tl, 'corr'] + [k for k in R[0]['probes'] if k.startswith('free')][-1:]:
            if ph not in R[0]['probes']:
                continue
            o[f'{ph}_seen_acc/first/held'] = '/'.join(ms([r['probes'][ph][q][m] for r in R], 2) for q, m in
                                                     (('seen', 'acc'), ('seen', 'first'), ('held', 'acc')))
            o[f'{ph}_silent/garble/talk'] = '/'.join(ms([r['probes'][ph]['seen'][q] for r in R], 2) for q in ('silent', 'garble', 'talk'))
            o[f'{ph}_spec own-other'] = ms([r['spec'][ph]['own_max'] - r['spec'][ph]['other_max'] for r in R])
            o[f'{ph}_spec per-lag max(own-other)'] = ms([max(np.array(r['spec'][ph]['own']) - np.array(r['spec'][ph]['other'])) for r in R])
            if 'other1_max' in R[0]['spec'][ph]:
                o[f'{ph}_spec own-other1 (unbiased)'] = ms([r['spec'][ph]['own_max'] - r['spec'][ph]['other1_max'] for r in R])
            o[f'{ph}_top_is_answer max'] = ms([max(r['spec'][ph]['top_is_answer']) for r in R], 2)
            o[f'{ph}_hub'] = ms([r['spec'][ph]['hub_share'] for r in R], 2)
            for q in ('talkover_per_ex', 'yieldfail_per_ex', 'garble_per_ex', 'resp_rate'):
                o[f'{ph}_mech_{q}'] = ms([r['mech'][ph][q] for r in R])
            o[f'{ph}_quality'] = {q: ms([r['quality'][ph][q] for r in R], 3) for q in ('resp', 'garble', 'clear', 'ans_syl_hit', 'first', 'full')}
            o[f'{ph}_pleased'] = ms([r['pleased'][ph] for r in R])
        if 'correction' in R[0]:
            o['correction new/old'] = ms([r['correction']['probe_new'] for r in R], 2) + ' / ' + ms([r['correction']['probe_old'] for r in R], 2)
        for k in ('FORCED_lag6', 'FORCED_lag12'):
            o[k] = {q: ms([r['release'][k][q] for r in R], 2) for q in ('acc', 'first', 'q_with_answer_pred', 'said_when_pred')}
        o['route'] = {q: ms([r['speech_route'][q] for r in R], 2) for q in ('heard', 'inner', 'pfc_motor')}
        o['L3 mean/max/p_zero/p_sat'] = '/'.join(ms([r['layer_counts']['n3'][q] for r in R], 2) for q in ('mean', 'max', 'p_zero', 'p_sat'))
        hs = [r['hub']['teach'] for r in R]
        o['L31_in_w pct50/90/99/max'] = np.round(np.mean([h['L31_in_w_pct'] for h in hs], 0), 3).tolist()
        o['L31_in_p pct50/90/99/max'] = np.round(np.mean([h['L31_in_p_pct'] for h in hs], 0), 3).tolist()
        o['L33_in_w pct50/90/99/max'] = np.round(np.mean([h['L33_in_w_pct'] for h in hs], 0), 3).tolist()
        o['P33_in pct50/90/99/max'] = np.round(np.mean([h['P33_in_pct'] for h in hs], 0), 3).tolist()
        # hub syllable in-strength vs the median syllable (connection strength and propagation)
        hr_w, hr_p = [], []
        for r, h in zip(R, hs):
            from world import SID
            sw, sp_ = np.array(h['syl_in_w']), np.array(h['syl_in_p'])
            hsyl = r['spec'][tl]['hub_syl']
            hr_w.append(float(sw[SID[hsyl]] / max(np.median(sw), 1e-9)))
            hr_p.append(float(sp_[SID[hsyl]] / max(np.median(sp_), 1e-9)))
        o['hub in-strength / median syllable (w; prop)'] = ms(hr_w, 2) + ' ; ' + ms(hr_p, 2)
        o['pass_path(dq12>=.3 & within<.8)'] = f"{np.mean([l['dq12'] for l in L]):.3f} / {np.mean([l['within'] for l in L]):.2f}"
        o['pass_spec(own-other>=.1)'] = f"{np.mean([l['spec'] for l in L]):.3f}"
        o['pass_first(>.2)'] = f"{np.mean([l['first'] for l in L]):.3f}"
        if 'overhear' in R[0]:
            oh = {}
            for mk in R[0]['overhear']:
                rr = [r for r in R if mk in r.get('overhear', {})]
                P = [r['path'][mk] for r in rr]
                oh[mk] = dict(
                    seen_acc=ms([r['overhear'][mk]['probe']['seen']['acc'] for r in rr], 2),
                    first=ms([r['overhear'][mk]['probe']['seen']['first'] for r in rr], 2),
                    held=ms([r['overhear'][mk]['probe']['held']['acc'] for r in rr], 2),
                    silent=ms([r['overhear'][mk]['probe']['seen']['silent'] for r in rr], 2),
                    spec=ms([r['overhear'][mk]['spec']['own_max'] - r['overhear'][mk]['spec']['other_max'] for r in rr]),
                    spec_lag=ms([max(np.array(r['overhear'][mk]['spec']['own']) - np.array(r['overhear'][mk]['spec']['other'])) for r in rr]),
                    spec_fair=ms([r['overhear'][mk]['spec']['own_max'] - r['overhear'][mk]['spec'].get('other1_max', np.nan) for r in rr]),
                    own=ms([r['overhear'][mk]['spec']['own_max'] for r in rr]),
                    hub=ms([r['overhear'][mk]['spec']['hub_share'] for r in rr], 2),
                    top_is_answer=ms([max(r['overhear'][mk]['spec']['top_is_answer']) for r in rr], 2),
                    dq0=ms([p['dec_q'][0] for p in P]), dq6=ms([p['dec_q'][6] for p in P]), dq12=ms([p['dec_q'][12] for p in P]),
                    gap12=ms([p['same'][12] - p['cross'][12] for p in P]), within=ms([float(np.mean(p['within_J1'][1:13])) for p in P], 2),
                    n3_12=ms([p['n3_lag'][12] for p in P], 1), pempty12=ms([p['p_empty_lag'][12] for p in P], 2))
            o['overhear_curve'] = oh
        o['ms_per_tick'] = ms([r['ms_per_tick'] for r in R], 2)
        out[a] = o
        print('=' * 12, a, o['n'], 'seeds')
        for k, v in o.items():
            if k != 'n':
                print(f'  {k}: {v}')
    return out


def table(d, names, labels=None):
    """compact markdown table: rows = metrics, columns = arms (5-seed mean +- SD)."""
    o = {}
    import contextlib, io
    with contextlib.redirect_stdout(io.StringIO()):
        o = arms(d, names)
    names = [n for n in names if n in o]
    labels = labels or names
    tl = [k for k in o[names[0]] if k.endswith('_seen_acc/first/held')][0].split('_')[0]
    fl = [k for k in o[names[0]] if k.startswith('free') and k.endswith('_seen_acc/first/held')]
    rows = [('n lives', 'n'), ('dec Q lag 0', 'dq0'), ('dec Q lag 6', 'dq6'), ('dec Q lag 12', 'dq12'), ('dec frame lag 12', 'dfr12'),
            ('same-cross J lag 12', 'gap12'), ('within J(t,t+1)', 'within'), ('damage 10% J lag 12', 'dmg12'),
            ('L3 count lag 12', 'n3_12'), ('P(L3 empty) lag 12', 'pempty12'), ('long silence turnover/tick', 'turn'),
            ('spec own-other (X1d, biased)', f'{tl}_spec own-other'), ('spec per-lag max', f'{tl}_spec per-lag max(own-other)'),
            ('spec own-other1 (fair)', f'{tl}_spec own-other1 (unbiased)'), ('hub share', f'{tl}_hub'),
            ('seen acc/first/held', f'{tl}_seen_acc/first/held'), ('probe silent/garble/talk', f'{tl}_silent/garble/talk'),
            ('response rate', f'{tl}_mech_resp_rate'), ('garble per exchange', f'{tl}_mech_garble_per_ex'),
            ('talk-over per exchange', f'{tl}_mech_talkover_per_ex'), ('failure to yield per ex.', f'{tl}_mech_yieldfail_per_ex'),
            ('FORCED lag 6 first / said', 'FORCED_lag6'), ('correction new/old', 'correction new/old'),
            ('untutored seen acc/first/held', fl[0] if fl else None), ('L33 learned pair fraction', 'L33pos'),
            ('hub in-strength / median (w ; P31)', 'hub in-strength / median (w ; P31)'.replace('P31', 'prop')),
            ('L3 mean/max/p_zero/p_sat', 'L3 mean/max/p_zero/p_sat')]
    print('| metric | ' + ' | '.join(labels) + ' |')
    print('|---|' + '---|' * len(names))
    for lab, k in rows:
        if k is None:
            continue
        vals = []
        for n in names:
            tln = [kk for kk in o[n] if kk.endswith('_seen_acc/first/held') and kk.startswith('teach')][0].split('_')[0]
            v = o[n].get(k.replace(tl + '_', tln + '_') if k else k, '')
            if k == 'FORCED_lag6' and isinstance(v, dict):
                v = v['first'] + ' / ' + v['said_when_pred']
            if k == 'hub in-strength / median (w ; prop)':
                v = o[n].get('hub in-strength / median syllable (w; prop)', '')
            vals.append(str(v))
        print(f'| {lab} | ' + ' | '.join(vals) + ' |')
    return o


if __name__ == '__main__':
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    mode, d = sys.argv[1], sys.argv[2]
    if mode == 'grid':
        grid(d)
    elif mode == 'table':
        table(d, sys.argv[3].split(','), sys.argv[4].split(',') if len(sys.argv) > 4 else None)
    else:
        o = arms(d, sys.argv[3].split(','))
        json.dump(o, open(os.path.join(d, 'AGG3_' + (sys.argv[4] if len(sys.argv) > 4 else 'main') + '.json'), 'w'), ensure_ascii=False, indent=1)

