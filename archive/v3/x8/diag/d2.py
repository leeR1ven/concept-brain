"""H code specificity and recall false alarms, by rho_h"""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(1, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '..', 'x345'))   # repo: x4_merge.py
from x8 import *
seed = 0
lin = Lineage(seed_lin=1 + seed)
rng = np.random.default_rng(50 + seed)
S, own, C = make_lessons(rng, n_sib=2)
lessons = S + own[0] + [(q, a[0]) for q, a in C]
banned = {tuple(q) for q, a in lessons + own[1]}
hb = hippo_birth(5000 + seed, 4096)
for spec in sys.argv[1:]:
    rho_h, ni = float(spec.split(':')[0]), spec.endswith(':s')
    c = make_cfg('A34', dict(rho_h=rho_h, h_need_input=ni))
    ag = Agent(lin, c, 100 * seed + 11, hb)
    r1 = np.random.default_rng(100 * seed + 11 + 1000)
    items = [list(q) + [a] for q, a in lessons] * c['n_rep'] + [list(r1.choice(V, 3, replace=False)) for _ in range(c['n_fill'])]
    for i in r1.permutation(len(items)): ag.hear([int(v) for v in items[i]])
    ag.pfc.theta_ref = ag.pfc.thr.theta; ag.hpop.theta_ref = ag.hpop.thr.theta
    ag.theta = 0.3 * float(np.median([(ag.M @ ag.ctx.drive(ag.pfc.code([ag.pattern(ch) for ch in q]))).max() for q, a in lessons]))
    rr = np.random.default_rng(1)
    J = {'pfc': [], 'hip': []}; Js = {'pfc': [], 'hip': []}
    for q, a in lessons:
        x = rr.integers(V - 1); x += x >= q[0]
        for nm, pop in (('pfc', ag.pfc), ('hip', ag.hpop)):
            A_ = pop.code([ag.pattern(q[0]), ag.pattern(q[1])]); B_ = pop.code([ag.pattern(int(x)), ag.pattern(q[1])]); S_ = pop.code([ag.pattern(q[1])])
            J[nm].append(len(np.intersect1d(A_, B_)) / len(np.union1d(A_, B_))); Js[nm].append(len(np.intersect1d(A_, S_)) / len(np.union1d(A_, S_)))
    # false recall on unheard pairs
    unheard = []
    while len(unheard) < 60:
        q = tuple(int(v) for v in rr.choice(V, 2, replace=False))
        if q not in banned: unheard.append((q, -1))
    fa = np.mean([g is not None for g in ag.test(unheard, 'hip')])
    hacc = acc(ag.test(lessons, 'hip'), lessons)
    # after 600 stretch lines with amnesia: what does H recall on lessons?
    a2 = clone(ag); a2.wipe_hip()
    lg = []
    stretch(a2, lessons, 300, 300, 0.0, np.random.default_rng(3000), banned, {600}, lg)
    gh = a2.test(lessons, 'hip')
    print(f'rho_h {rho_h} silent={ni}: J(q1q2,xq2) pfc {np.mean(J["pfc"]):.2f} hip {np.mean(J["hip"]):.2f} | J(q1q2,q2) pfc {np.mean(Js["pfc"]):.2f} hip {np.mean(Js["hip"]):.2f} | '
          f'hip recall p1 {hacc:.2f} false-alarm unheard {fa:.2f} | amnesia+600: hip released {np.mean([g is not None for g in gh]):.2f} right {acc(gh, lessons):.2f}; ctx {lg[-1]["ctx"]:.2f}', flush=True)
