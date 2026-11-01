import json, glob, os, sys
import numpy as np
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out2')


def m(xs, d=2):
    xs = [x for x in xs if x is not None]
    return f'{np.mean(xs):.{d}f}±{np.std(xs):.{d}f}' if xs else 'n/a'


def agg(arm):
    R = [json.load(open(f)) for f in sorted(glob.glob(f'{D}/{arm}_s[0-9].json'))]
    if not R:
        return None
    g = lambda f: [f(r) for r in R]
    tl = max((k for k in R[0]['probes'] if k.startswith('teach')), key=lambda k: int(k[5:]))
    fl = max((k for k in R[0]['probes'] if k.startswith('free')), key=lambda k: int(k[4:]))
    o = dict(n=len(R), teach_mark=tl, free_mark=fl)
    for ph in (tl, 'corr', fl):
        o[f'{ph}_seen_acc'] = m(g(lambda r: r['probes'][ph]['seen']['acc']))
        o[f'{ph}_held_acc'] = m(g(lambda r: r['probes'][ph]['held']['acc']))
        o[f'{ph}_first'] = m(g(lambda r: r['probes'][ph]['seen']['first']))
        o[f'{ph}_silent/garble/talk'] = '/'.join(m(g(lambda r: r['probes'][ph]['seen'][q])) for q in ('silent', 'garble', 'talk'))
        o[f'{ph}_spec_own/other_max'] = m(g(lambda r: r['spec'][ph]['own_max'])) + ' / ' + m(g(lambda r: r['spec'][ph]['other_max']))
        o[f'{ph}_top_is_answer@12'] = m(g(lambda r: r['spec'][ph]['top_is_answer'][12]))
        o[f'{ph}_hub_share'] = m(g(lambda r: r['spec'][ph]['hub_share'])) + ' ' + ''.join(g(lambda r: r['spec'][ph]['hub_syl']))
        o[f'{ph}_release'] = m(g(lambda r: r['release'][ph]['q_with_answer_pred'])) + ' pred -> said ' + m(g(lambda r: r['release'][ph]['said_when_pred']))
        for q in ('talkover_per_ex', 'yieldfail_per_ex', 'garble_per_ex', 'extra_per_ex', 'resp_rate', 'echo_per_ex'):
            o[f'{ph}_mech_{q}'] = m(g(lambda r: r['mech'][ph][q]), 3)
    marks = [k for k in R[0]['quality']]
    o['quality_over_time'] = {k: {q: m(g(lambda r: r['quality'][k][q])) for q in ('resp', 'garble', 'clear', 'ans_syl_hit', 'first', 'full')}
                              for k in marks}
    o['pleased_per_ex_over_time'] = {k: m(g(lambda r: r['pleased'][k]), 3) for k in marks}
    for k in ('FORCED_lag6', 'FORCED_lag12'):
        if k in R[0]['release']:
            o[k] = {q: m(g(lambda r: r['release'][k][q])) for q in ('acc', 'first', 'silent', 'garble', 'q_with_answer_pred', 'said_when_pred')}
    P = 'teach'
    o['path_same@0/6/12'] = '/'.join(m(g(lambda r: r['path'][P]['same'][k])) for k in (0, 6, 12))
    o['path_cross@0/6/12'] = '/'.join(m(g(lambda r: r['path'][P]['cross'][k])) for k in (0, 6, 12))
    o['path_decq@0/6/12'] = '/'.join(m(g(lambda r: r['path'][P]['dec_q'][k])) for k in (0, 6, 12))
    o['path_damage0.1@12'] = m(g(lambda r: r['path'][P]['damage']['0.1'][12]))
    o['path_within_J1'] = m(g(lambda r: float(np.mean(r['path'][P]['within_J1'][1:13]))))
    o['layers'] = {L: {q: m(g(lambda r: r['layer_counts'][L][q]), 3) for q in ('mean', 'max', 'p_zero', 'p_sat')} for L in ('n2', 'n3')}
    o['route'] = {q: m(g(lambda r: r['speech_route'][q])) for q in ('heard', 'inner', 'recall', 'pfc_motor', 'onsets')}
    o['correction_new/old'] = m(g(lambda r: r['correction']['probe_new'])) + ' / ' + m(g(lambda r: r['correction']['probe_old']))
    o['motion_free'] = {q: m(g(lambda r: (r['motion']['free'] or {}).get(q))) for q in ('stuck20', 'turnover5', 'distinct')}
    o['empty_per100_teach/free'] = m(g(lambda r: r['empty']['teach']['per100'])) + ' / ' + m(g(lambda r: r['empty']['free']['per100']))
    o['ms_per_tick'] = m(g(lambda r: r['ms_per_tick']))
    return o


if __name__ == '__main__':
    arms = sys.argv[1:] or sorted({os.path.basename(f).rsplit('_s', 1)[0] for f in glob.glob(f'{D}/*_s[0-9].json')})
    out = {}
    for a in arms:
        o = agg(a)
        if o is None:
            continue
        out[a] = o
        print('=' * 10, a, o['n'], 'seeds')
        for k, v in o.items():
            if k != 'n':
                print(f'  {k}: {v}')
    json.dump(out, open(os.path.join(D, 'AGG2.json'), 'w'), ensure_ascii=False, indent=1)
