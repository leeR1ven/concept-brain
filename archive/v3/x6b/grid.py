import sys, json, numpy as np
sys.path.insert(0, '.')
import x6b_hippo as HB
small = dict(N_H=4000, k=32, C=800, C_out=512, cB=400)
plan = [(3000, p, n) for p, n in zip(HB.H6.PETS, HB.H6.PET_NAMES)]
for line in sys.stdin:
    extra = json.loads(line)
    cfg = dict(HB.BASE6b, **small, **extra)
    r = HB.run6b(cfg, 15000, seed=9, test_every=5000, tests=('all',), fact_plan=plan, test_kw=dict(all=dict(n_old=5)), verbose=False)
    a = [x for x in r['all'] if x['age'] < 10000]
    m = lambda k: round(float(np.mean([x[k] for x in a])), 2)
    ovm = round(float(np.mean([np.mean(x['ov']) for x in a])), 2)
    ox = [np.mean(x['ovx']) for x in a if 'ovx' in x]
    f = r['fact']
    print(json.dumps(extra), '| r50', m('r50'), 'i100', m('i100'), 'ov', ovm, 'ovx', round(float(np.mean(ox)), 2) if ox else None,
          'live', r['live'], 'I1', np.mean([x['I1_ok'] for x in f]), 'R1s', np.mean([x['R1s_ok'] for x in f]),
          'persist', m('persist'), 'rd', m('rd'), 'on sp/si', round(r['ls_stat']['on_speech']), round(r['ls_stat']['on_silence']), flush=True)
