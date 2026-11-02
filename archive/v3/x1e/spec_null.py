"""null distribution of X1d's max-over-lags specificity (own_max - other_max) on real probe scores: the answer's
first syllable is replaced by the answer of a random other question (permutation), so true specificity is 0."""
import sys, os
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
for _x in ('x1d', 'x1c'):   # repo: core4/phase1/run2/regimes from ../x1d, core3/world/run from ../x1c
    _p = __import__('os').path.join(__import__('os').path.dirname(__import__('os').path.abspath(__file__)), '..', _x)
    if _p not in __import__('sys').path:
        __import__('sys').path.append(_p)
import run3, phase1
from run3 import Physics, World, Brain, birth_rates, run_until, probe2, pred_specificity, ARMS, parse
from core5 import DEF
arm, seed, nt = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
kv = {k: parse(v) for k, v in (a.split('=', 1) for a in sys.argv[4:])}
bk = dict(ARMS[arm]['b'], **{k: v for k, v in kv.items() if not k.startswith('w_')})
ph = Physics(); world = World(ph, seed)
nol = dict(bk); keep = {k: bk.get(k) for k in phase1.NOLEARN}; nol.update(phase1.NOLEARN)
brain = Brain(ph, seed, rates=birth_rates(ph, world), **nol)
run_until(brain, World(ph, seed + 50), 200)
for k in phase1.NOLEARN:
    brain.c[k] = keep[k] if keep[k] is not None else DEF[k]
n_oh = int(kv.get('w_overhear', 0) or 0)
if n_oh:
    ow = run3.OverhearWorld(ph, seed, gap=kv.get('w_ohgap', 'short'))
    th = brain.c['theta_m']; brain.c['theta_m'] = 99.0
    run_until(brain, ow, n_oh)
    brain.c['theta_m'] = th
if nt:
    run_until(brain, world, nt)
allp = [(f, x) for f in range(4) for x in range(8)]
pr = probe2(brain, ph, world, allp, seed)
real = pred_specificity(pr)
print('real: own_max %.3f other_max %.3f diff %.3f | other1 %.3f fair diff %.3f | lag_diff_max %.3f' % (
    real['own_max'], real['other_max'], real['own_max'] - real['other_max'], real['other1_max'], real['own_max'] - real['other1_max'], real['lag_diff_max']))
rng = np.random.default_rng(1)
nulls, nulls_fair = [], []
for it in range(200):
    perm = rng.permutation(len(pr))
    pp = [dict(r, a1=pr[perm[i]]['a1']) for i, r in enumerate(pr)]
    s = pred_specificity(pp)
    nulls.append(s['own_max'] - s['other_max']); nulls_fair.append(s['own_max'] - s['other1_max'])
print('null (answers permuted, 200x): max-based diff mean %.3f sd %.3f 95%% %.3f | fair diff mean %.3f sd %.3f 95%% %.3f' % (
    np.mean(nulls), np.std(nulls), np.percentile(nulls, 95), np.mean(nulls_fair), np.std(nulls_fair), np.percentile(nulls_fair, 95)))
# positive control: the hand-wired forced inner hearing of the answer at lag 6 must show up as specificity
prf = probe2(brain, ph, world, allp, seed, force=(6, 1.0))
sf = pred_specificity(prf)
print('positive control (forced answer inner hearing at lag 6): max-based diff %.3f | fair diff %.3f | lag_diff_max %.3f' % (
    sf['own_max'] - sf['other_max'], sf['own_max'] - sf['other1_max'], sf['lag_diff_max']))
