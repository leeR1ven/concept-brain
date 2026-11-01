import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(1, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '..', 'x345'))   # repo: x4_merge.py
import x8
from x8 import *
arm = sys.argv[1] if len(sys.argv) > 1 else 'A0'
seed = 0
c = make_cfg(arm, {})
lin = Lineage(seed_lin=1 + seed)
rng = np.random.default_rng(50 + seed)
S, own, C = make_lessons(rng, n_sib=2)
lessons = S + own[0] + [(q, a[0]) for q, a in C]
banned = {tuple(q) for q, a in lessons + own[1]}
ag = Agent(lin, c, 100 * seed + 11, None)
r1 = np.random.default_rng(100 * seed + 11 + 1000)
items = [list(q) + [a] for q, a in lessons] * c['n_rep'] + [list(r1.choice(V, 3, replace=False)) for _ in range(c['n_fill'])]
for i in r1.permutation(len(items)): ag.hear([int(v) for v in items[i]])
def sc(a, q): return a.M @ a.ctx.drive(a.pfc.code([a.pattern(ch) for ch in q]))
ag.pfc.theta_ref = ag.pfc.thr.theta
ag.theta = 0.3 * float(np.median([sc(ag, q).max() for q, a in lessons]))
def report(a, tag):
    g = a.test(lessons, 'ctx')
    r = np.mean([x == l[1] for x, l in zip(g, lessons)]); n = np.mean([x is None for x in g])
    rel = []
    for q, aa in lessons:
        s = sc(a, q); o = np.argsort(-s); rel.append((s[aa] / a.theta, s[o[0]] / a.theta, s[o[1]] / a.theta, o[0] == aa))
    rel = np.array(rel, float)
    so = (a.ctx.B + a.ctx.D).sum(1).mean(); si = a.ctx.si[:N_A].mean()
    print(f'{tag:22s} right {r:.2f} none {n:.2f} argmax_right {rel[:,3].mean():.2f} right/th {rel[:,0].mean():.2f} best/th {rel[:,1].mean():.2f} 2nd/th {rel[:,2].mean():.2f}  S_out {so:.1f} S_in {si:.1f} live_thr {a.pfc.thr.theta:.3f}', flush=True)
report(ag, 'after p1')
for kind in ('fill', 'ask', 'time'):
    a2 = clone(ag); r = np.random.default_rng(3000)
    if kind == 'fill':
        for f in filler_lines(r, 600, banned): a2.hear(f)
    elif kind == 'ask':
        for i in r.integers(0, len(lessons), 600): a2.ask(lessons[i][0], 0.0)
    else:
        a2.advance(600 * 24)
    report(a2, kind + ' x600')
    if kind == 'fill':
        for f in filler_lines(r, 2400, banned): a2.hear(f)
        report(a2, kind + ' x3000')
