import json, glob, os, sys
import numpy as np
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out')
order = ['B', 'C', 'C2', 'A', 'B_nlms', 'B_nopart', 'B_linear', 'B_fast', 'B_hip', 'B_dist', 'B_real', 'B_loop']
def m(xs):
    xs = [x for x in xs if x is not None]
    return f'{np.mean(xs):.2f}±{np.std(xs):.2f}' if xs else 'n/a'
rows = {}
for arm in order:
    R = [json.load(open(f)) for f in sorted(glob.glob(f'{D}/{arm}_s[0-9].json'))]
    if not R: continue
    last_free = max((k for k in R[0]['probes'] if k.startswith('free')), key=lambda k: int(k[4:]))
    g = lambda f: [f(r) for r in R]
    out = dict(n=len(R))
    out['teach_seen_acc'] = m(g(lambda r: r['probes']['teach600']['seen']['acc']))
    out['teach_first'] = m(g(lambda r: r['probes']['teach600']['seen']['first']))
    out['teach_echo'] = m(g(lambda r: r['probes']['teach600']['seen']['echo']))
    out['teach_silent'] = m(g(lambda r: r['probes']['teach600']['seen']['silent']))
    out['teach_garble'] = m(g(lambda r: r['probes']['teach600']['seen']['garble']))
    out['teach_talk_during_Q'] = m(g(lambda r: r['probes']['teach600']['seen']['talk']))
    out['held_acc'] = m(g(lambda r: r['probes']['teach600']['held']['acc']))
    out['free_last'] = last_free
    out['free_acc'] = m(g(lambda r: r['probes'][last_free]['seen']['acc']))
    out['free_echo'] = m(g(lambda r: r['probes'][last_free]['seen']['echo']))
    out['free_silent'] = m(g(lambda r: r['probes'][last_free]['seen']['silent']))
    out['corr_new'] = m(g(lambda r: r['correction']['probe_new']))
    out['corr_first_new'] = [v['first_new_after_teachings'] for r in R for v in r['correction']['latency'].values()]
    tk = [k for k in R[0]['mech'] if k.startswith('teach')][-1]
    phs = [('teach', tk), ('free300', 'free300')] + ([('free3000', 'free3000')] if 'free3000' in R[0]['mech'] else [])
    for ph, key in phs:
        for q in ('resp_rate', 'talkover_per_ex', 'yieldfail_per_ex', 'garble_per_ex', 'extra_per_ex', 'online_acc', 'echo_per_ex'):
            out[f'{ph}_{q}'] = m(g(lambda r: r['mech'][key][q]))
    out['empty_teach_per100'] = m(g(lambda r: r['empty']['teach']['per100']))
    out['empty_free_per100'] = m(g(lambda r: r['empty']['free']['per100']))
    for ph in ('teach', 'free'):
        out[f'motion_{ph}'] = {q: m(g(lambda r: (r['motion'][ph] or {}).get(q))) for q in ('n_active', 'turnover1', 'turnover5', 'stuck20', 'distinct')}
    out['layers'] = {L: {q: m(g(lambda r: r['layer_counts'][L][q])) for q in ('mean', 'min', 'max', 'p_zero', 'p_sat')} for L in ('n1', 'n2', 'n3')}
    out['route'] = {q: m(g(lambda r: r['speech_route'][q])) for q in ('heard', 'inner', 'recall', 'pfc_motor', 'onsets')}
    out['ms_per_tick'] = m(g(lambda r: r['ms_per_tick']))
    out['time_s'] = m(g(lambda r: r['time_s']))
    rows[arm] = out
json.dump(rows, open(os.path.join(D, 'AGG.json'), 'w'), ensure_ascii=False, indent=1)
for arm, o in rows.items():
    print('=' * 8, arm, o['n'], 'seeds')
    for k, v in o.items():
        if k != 'n': print(f'  {k}: {v}')
