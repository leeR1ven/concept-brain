"""why do adults (parents, merged children) learn new lessons slowly? seed 0 fixed_in, 2 rounds of own0+own1"""
import sys, os, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
sys.path.insert(1, os.path.join(HERE, '..', 'x345'))   # repo: the unchanged X4 harness (x4_merge.py)
import x4b as B, x4_merge as X
cfg = dict(X.BASE, trim=0.1)
lin = X.Lineage(seed_lin=1); rng = np.random.default_rng(50)
S, own, C = X.make_lessons(rng, n_sib=2)
sibs = [X.raise_sib(lin, cfg, 11*(k+1), S+own[k]+[(q,a[k]) for q,a in C], 10, 300) for k in range(2)]
best = [ag.scores(q).max() for k, ag in enumerate(sibs) for q, a in S + own[k]]
theta = 0.3*float(np.median(best))
ref = sibs[0]
fr = B.child_agent(lin, cfg, 11, ref, ref.B_pa.copy())
def stats(ag, items, tag):
    f = B.scorer(ag); W = np.maximum(ag.B_pa+ag.D_pa, 0); so = W.sum(1); si = W.sum(0)
    rr = []; ov = []
    for q, a in items:
        sc = f(q); o = np.argsort(-sc); got = X.answer(sc, theta, 1.25)
        rr.append((sc[a]/theta, sc[o[0]]/theta, sc[o[1]]/theta, got == a, got is None, o[0] == a))
        x = ag.code(list(q))[-1]; ov.append(so[x].mean())
    rr = np.array(rr, float)
    print(f'{tag:28s} right/theta {rr[:,0].mean():.2f}  best/theta {rr[:,1].mean():.2f}  2nd/theta {rr[:,2].mean():.2f}  '
          f'right {rr[:,3].mean():.2f} none {rr[:,4].mean():.2f} argmax_right {rr[:,5].mean():.2f}  out-strength(x) {np.mean(ov):.1f}  in-strength(AUD) {si.mean():.1f}')
cur = own[0] + own[1]
for ag, tag in ((fr, 'fresh'), (sibs[0], 'parent0')):
    stats(ag, own[1], tag + ' r0 own1')
    r = np.random.default_rng(7000)
    for rd in range(2):
        items = [list(q)+[a] for q, a in cur] + [list(r.choice(X.V, 3, replace=False)) for _ in range(len(cur)//2)]
        for i in r.permutation(len(items)): ag.hear([int(v) for v in items[i]])
        stats(ag, own[1], tag + f' r{rd+1} own1')
