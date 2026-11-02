import json
import sys

import numpy as np

files = sys.argv[1:]
ARMS = ['base', 'same', 'redup', 'imit']


def ms(x):
    x = np.asarray(x, float)
    return f"{x.mean():.2f}±{x.std(ddof=1):.2f}"


for fn in files:
    R = json.load(open(fn))
    st = [r for r in R if r['arm'] == 'standin'][0]
    runs = [r for r in R if r['arm'] != 'standin']
    phys, ratio = runs[0]['phys'], runs[0]['ratio']
    seeds = sorted({r['seed'] for r in runs})
    print(f"\n==== physics={phys}  eta_minus/eta={ratio}  seeds={seeds}")
    si = [st[str(s)]['inner']['correct'] for s in seeds]
    sh = [st[str(s)]['hear']['correct'] for s in seeds]
    sg = [st[str(s)]['inner']['garble'] for s in seeds]
    print(f"  stand-in ceiling: inner correct {ms(si)}  hear correct {ms(sh)}  inner garble {ms(sg)}")
    by = {a: {r['seed']: r for r in runs if r['arm'] == a} for a in ARMS}
    for cp in ('2000', '5000'):
        print(f"  -- after {cp} babbles")
        print("  arm    inner-correct  inner-clear  hear-correct  inner-garble  hear-garble  total-norm-inner  oracle-inner  oracle-hear  top4-exact  w_true/w_other  potent/depress")
        for a in ARMS:
            rr = [by[a][s]['cps'][cp] for s in seeds]
            ic = [r['inner']['correct'] for r in rr]
            icl = [r['inner']['clear'] for r in rr]
            hc = [r['hear']['correct'] for r in rr]
            ig = [r['inner']['garble'] for r in rr]
            hg = [r['hear']['garble'] for r in rr]
            tn = [r['total']['inner']['correct'] for r in rr]
            oi = [r['oracle']['inner'][0] for r in rr]
            oh = [r['oracle']['hear'][0] for r in rr]
            t4 = [r['top4_exact'] for r in rr]
            wr = [r['w_true'] / max(r['w_other'], 1e-9) for r in rr]
            pd = [r['n_plus'] / max(r['n_minus'], 1e-9) for r in rr]
            print(f"  {a:6s} {ms(ic):13s} {ms(icl):12s} {ms(hc):13s} {ms(ig):13s} {ms(hg):12s} {ms(tn):16s} {ms(oi):13s} {ms(oh):12s} {ms(t4):11s} {ms(wr):15s} {ms(pd)}")
        for a in ARMS[1:]:
            d = [by[a][s]['cps'][cp]['inner']['correct'] - by['base'][s]['cps'][cp]['inner']['correct'] for s in seeds]
            d4 = [by[a][s]['cps'][cp]['top4_exact'] - by['base'][s]['cps'][cp]['top4_exact'] for s in seeds]
            print(f"   paired {a} - base: inner-correct {np.mean(d):+.3f}±{np.std(d, ddof=1):.3f} {[round(x, 2) for x in d]}  top4 {np.mean(d4):+.3f}±{np.std(d4, ddof=1):.3f}")
    print("  -- verdict at 2000 (pre-registered: inner correct >= 0.5 and >= base + 0.3 in every seed)")
    for a in ARMS:
        ok = all(by[a][s]['cps']['2000']['inner']['correct'] >= 0.5 and
                 by[a][s]['cps']['2000']['inner']['correct'] - by['base'][s]['cps']['2000']['inner']['correct'] >= 0.3
                 for s in seeds)
        print(f"   {a}: {'PASS' if ok else 'FAIL'}")
    print("  imitations per life:", [by['imit'][s]['stats']['imitations'] for s in seeds],
          " babble outcomes (seed0 base):", by['base'][seeds[0]]['stats'])
