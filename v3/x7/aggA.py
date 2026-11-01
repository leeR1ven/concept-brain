import json
import sys

import numpy as np

R = json.load(open(sys.argv[1]))
arms = ['dom1', 'eq4', 'eq7']
by = {a: [r for r in R if r['arm'] == a] for a in arms}
seeds = sorted({r['seed'] for r in R})
CL = ['silence', 'syl1', 'syl2', 'sentence', 'two_spk']


def ms(x):
    x = np.asarray(x, float)
    return f"{x.mean():.3f}±{x.std(ddof=1):.3f}" if len(x) > 1 else f"{x.mean():.3f}"


def msi(x):
    x = np.asarray(x, float)
    return f"{x.mean():.0f}±{x.std(ddof=1):.0f}"


for a in arms:
    print(f"\n== {a}  thetas", [ms([r['thetas'][l] for r in by[a]]) for l in range(4)])
    print(" sizes", by[a][0]['sizes'])
    for key in ('count', 'count_chain'):
        print(f" {key}:")
        for c in CL:
            print(f"   {c:9s}", [msi([r[key][c][l] for r in by[a]]) for l in range(5)])
        rat = [[r[key]['sentence'][l] / max(r[key]['silence'][l], 1) for r in by[a]] for l in range(5)]
        print("   sent/sil ", [ms(x) for x in rat])
        mono = [all(r[key]['silence'][4] < r[key]['syl1'][4] < r[key]['sentence'][4] for r in [rr]) for rr in by[a]]
        print("   L4 monotone sil<syl1<sent per seed:", mono)
    fr = [[np.mean([r['count'][c][l] for c in CL]) / r['sizes'][l] for r in by[a]] for l in range(5)]
    print(" mean fraction over classes", [ms(x) for x in fr])
    for key in ('rho', 'J_diff', 'J_sim_voice', 'J_sim_noise', 'J_sent_diff', 'J_sent_voice',
                'rho_chain', 'Jc_diff', 'Jc_sim_voice', 'Jc_sent_diff', 'Jc_sent_voice'):
        print(f" {key:12s}", [ms([r[key][l] for r in by[a]]) for l in range(5)])
    rs = [[r['J_sim_voice'][l] / r['J_diff'][l] for r in by[a]] for l in range(5)]
    print(" Jsim/Jdiff  ", [ms(x) for x in rs])
    rs = [[r['Jc_sim_voice'][l] / max(r['Jc_diff'][l], 1e-9) for r in by[a]] for l in range(5)]
    print(" Jc sim/diff ", [ms(x) for x in rs])
    for k in by[a][0]['recruit_per_unit']:
        print(f" recruit {k:13s}", [ms([r['recruit_per_unit'][k][l] for r in by[a]]) for l in range(3)])

print("\n== paired vs dom1 at L4 (mean, SD, per-seed)")
d0 = {r['seed']: r for r in by['dom1']}
for a in ('eq4', 'eq7'):
    for key in ('J_diff', 'rho', 'J_sim_voice', 'J_sent_diff', 'Jc_diff', 'rho_chain', 'Jc_sent_diff'):
        d = [r[key][4] - d0[r['seed']][key][4] for r in by[a]]
        print(f"  {a} {key:12s} diff {np.mean(d):+.3f} ± {np.std(d, ddof=1):.3f}", [round(x, 3) for x in d])
    d = [r['J_sim_voice'][4] / r['J_diff'][4] - d0[r['seed']]['J_sim_voice'][4] / d0[r['seed']]['J_diff'][4] for r in by[a]]
    print(f"  {a} Jsim/Jdiff   diff {np.mean(d):+.3f} ± {np.std(d, ddof=1):.3f}")
    for key in ('count', 'count_chain'):
        d = [r[key]['sentence'][4] / max(r[key]['silence'][4], 1) - d0[r['seed']][key]['sentence'][4] / max(d0[r['seed']][key]['silence'][4], 1) for r in by[a]]
        print(f"  {a} {key} sent/sil L4 diff {np.mean(d):+.3f} ± {np.std(d, ddof=1):.3f}")

print("\n== pre-registered verdict")
for a in ('eq4', 'eq7'):
    p1a = all(r['count']['silence'][4] < r['count']['syl1'][4] < r['count']['sentence'][4] and
              r['count']['sentence'][4] >= 2 * r['count']['silence'][4] for r in by[a])
    fr = [np.mean([np.mean([r['count'][c][l] for c in CL]) / r['sizes'][l] for r in by[a]]) for l in range(1, 5)]
    p1b = all(fr[i] > fr[i + 1] for i in range(3))
    dj = [r['J_diff'][4] - d0[r['seed']]['J_diff'][4] for r in by[a]]
    p2a = np.mean(dj) <= -0.05 and all(x < 0 for x in dj)
    rho = np.mean([r['rho'][4] for r in by[a]])
    rho0 = np.mean([r['rho'][4] for r in by['dom1']])
    p2b = rho >= 0.95 and rho >= rho0 - 0.03
    sr = np.mean([r['J_sim_voice'][4] / r['J_diff'][4] for r in by[a]])
    sr0 = np.mean([r['J_sim_voice'][4] / r['J_diff'][4] for r in by['dom1']])
    p2c = sr >= sr0
    P1, P2 = p1a and p1b, p2a and p2b and p2c
    v = 'PASS' if P1 and P2 else ('PARTIAL' if P1 or P2 else 'FAIL')
    print(f"  {a}: P1 counts-follow-richness={p1a} deeper-sparser={p1b} | P2 Jdiff-lower={p2a} rho-ok={p2b} "
          f"(rho {rho:.3f} vs dom1 {rho0:.3f}) sim/diff-ok={p2c} -> {v}")
