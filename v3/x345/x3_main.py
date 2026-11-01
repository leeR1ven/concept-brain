"""X3 suite: runs the designer's hippocampus (x3_hippo.py) on the recorded stream, one config after another in one
process (< 600 MB), and writes x3_results.json after every run. Run: python3 -I x3_main.py"""
import json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import x3_hippo as X

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'x3_results.json')
KEYS = ('r100', 'r50', 'r20', 'i100', 'i50', 'fid50', 'fid20')


def brief(r):
    T = r['tests']
    s = {k: X.summarize(T, k) for k in KEYS}
    s['horizon_r50'] = X.horizon(T, 'r50'); s['horizon_r20'] = X.horizon(T, 'r20'); s['horizon_r100'] = X.horizon(T, 'r100')
    s['rB_r50'] = X.summarize(T, 'r50', kinds=('rB',)); s['rS_r50'] = X.summarize(T, 'r50', kinds=('rS',))
    s['u_r50_n1'] = X.summarize(T, 'r50', kinds=('u',))
    for kind in ('pS', 'pV', 'pD'):
        rows = [t for t in T if t['kind'] == kind and 'sep_right50' in t and t['age'] < 20000]
        s[f'sep_{kind}'] = dict(n=len(rows), right=round(float(np.mean([t['sep_right50'] for t in rows])), 3) if rows else None,
                               other=round(float(np.mean([t['sep_other50'] for t in rows])), 3) if rows else None)
    pre = [t for t in T if t.get('pert')]
    if pre:
        s['pert_r50'] = X.summarize(pre, 'r50', kinds=('u', 'rS', 'rB')); s['pert_i100'] = X.summarize(pre, 'i100', kinds=('u', 'rS', 'rB'))
        post = [t for t in T if not t.get('pert')]
        s['nopert_r50'] = X.summarize(post, 'r50', kinds=('u', 'rS', 'rB')); s['nopert_i100'] = X.summarize(post, 'i100', kinds=('u', 'rS', 'rB'))
    s['kicks'] = r['kicks']
    s.update(secs=r['secs'], rss_mb=r['rss_mb'], n_probes=r['n_probes'], live=r['live'], n_change=r['n_change'])
    return s


RUNS = [
    ('base', {}, 400000, True),
    ('arm_recode_no_birth_recurrence', dict(arm='recode'), 200000, True),
    ('arm_copy_no_recode', dict(arm='copy', C=4000), 200000, True),
    ('gamma_0.95', dict(gamma=0.95), 100000, False),
    ('gamma_0.98', dict(gamma=0.98), 150000, False),
    ('gamma_0.995', dict(gamma=0.995), 300000, False),
    ('gamma_0.999', dict(gamma=0.999), 600000, False),
    ('eta_x0.25', dict(eta_seq=0.5, eta_rev=0.5, eta_out=0.5, eta_co=0.05), 200000, False),
    ('eta_x4', dict(eta_seq=8.0, eta_rev=8.0, eta_out=8.0, eta_co=0.8), 200000, False),
    ('k_32', dict(k=32), 200000, False),
    ('k_128', dict(k=128), 200000, False),
    ('C_1000', dict(C=1000), 200000, False),
    ('C_3100', dict(C=3100), 200000, False),
    ('s_0.25', dict(N_H=2500, k=16), 200000, False),
    ('s_0.5', dict(N_H=5000, k=32), 200000, False),
    ('s_2', dict(N_H=20000, k=128, C_out=512), 200000, False),
    ('s_2_C1500', dict(N_H=20000, k=128, C=1500, C_out=512), 200000, False),
    ('s_1_C1500', dict(C=1500), 200000, False),
    ('pert_3pct', dict(pert_frac=0.03, pert_t=100000), 200000, False),
    ('pert_5pct', dict(pert_frac=0.05, pert_t=100000), 200000, False),
]

if __name__ == '__main__':
    only = sys.argv[1:]
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    for name, ov, t_end, kick in RUNS:
        if (only and name not in only) or name in res:
            continue
        cfg = dict(X.BASE, probe_every=1500, **ov)
        test_at = list(range(10000, t_end + 1, 10000))
        r = X.run(cfg, t_end, test_at, do_kicks=kick, verbose=False)
        b = brief(r); b['cfg'] = ov; b['t_end'] = t_end
        res[name] = b
        json.dump(res, open(OUT, 'w'), indent=1)
        print(name, 'horizon r50', b['horizon_r50'], 'r20', b['horizon_r20'], '| r50', [(lo, n, m) for lo, hi, n, m in b['r50']],
              '| i50', [m for *_, m in b['i50']], f"| {b['secs']}s {b['rss_mb']}MB", flush=True)
