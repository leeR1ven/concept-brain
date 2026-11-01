import json, numpy as np
R = json.load(open('outC/main.json'))
arms = list(R[0]['probes'])
print('28a practice outcomes per seed', [r['practice'] for r in R])
print('learned M->M excitation, same-syllable pairs vs others: %.3f vs %.3f; partners: %.3f vs %.3f' % (
    np.mean([r['E_same'] for r in R]), np.mean([r['E_other'] for r in R]), np.mean([r['V_same'] for r in R]), np.mean([r['V_other'] for r in R])))
for a in arms:
    row = []
    for c in ('clean', 'noise', 'cross'):
        cr = [r['probes'][a][c]['correct'] for r in R]; ga = [r['probes'][a][c]['garble'] for r in R]
        row.append(f"{c}: correct {np.mean(cr):.3f}±{np.std(cr,ddof=1):.3f} garble {np.mean(ga):.3f}±{np.std(ga,ddof=1):.3f}")
    print(f"{a:8s}", ' | '.join(row))
print('paired vs none (garble and correct = mean of noise and cross conditions)')
for a in arms[1:]:
    g = lambda r, k, x: np.mean([r['probes'][k][c][x] for c in ('noise', 'cross')])
    dg = [g(r, a, 'garble') - g(r, 'none', 'garble') for r in R]
    rel = [g(r, a, 'garble') / g(r, 'none', 'garble') - 1 for r in R]
    dc = [g(r, a, 'correct') - g(r, 'none', 'correct') for r in R]
    dcl = [r['probes'][a]['clean']['correct'] - r['probes']['none']['clean']['correct'] for r in R]
    ok = np.mean(rel) <= -0.25 and np.mean(dc) >= 0
    print(f"{a:8s} dgarble {np.mean(dg):+.3f}±{np.std(dg,ddof=1):.3f} (rel {np.mean(rel):+.1%})  dcorrect {np.mean(dc):+.3f}±{np.std(dc,ddof=1):.3f}  dclean-correct {np.mean(dcl):+.3f}  {'PASS' if ok else 'FAIL'}")
P = json.load(open('outC/pace.json'))
print('\n28b pace')
for form in ('held', 'onset'):
    rr = [p for p in P if p['form'] == form]
    print(form, 'grade->next links', {k: round(np.mean([p['grade_link'][k] for p in rr]), 3) for k in ('mu', 'f1', 'f2', 'f3')},
          'input course k=1..5 (/taught onset)', np.round(np.mean([p['course'] for p in rr], 0), 2).tolist())
    for phi in ('0.4', '0.6', '0.8'):
        m = [p['lens'][phi]['mean'] for p in rr]; sd = [p['lens'][phi]['sd'] for p in rr]
        s3 = [p['lens'][phi]['share3'] for p in rr]; s1 = [p['lens'][phi]['share_pm1'] for p in rr]
        ok = abs(np.mean(m) - 3) <= 0.5 and np.mean(s1) >= 0.9
        print(f"   phi {phi}: length {np.mean(m):.2f}±{np.std(m,ddof=1):.2f} (within-run SD {np.mean(sd):.2f}), exactly 3: {np.mean(s3):.2f}, within 3±1: {np.mean(s1):.2f}  {'PASS' if ok else 'FAIL'}")
