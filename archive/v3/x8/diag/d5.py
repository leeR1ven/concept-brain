"""why do unanswered asks erode Q->A? after 600 asks, what is the new argmax for lessons that lost their answer?"""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(1, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '..', 'x345'))   # repo: x4_merge.py
from x8 import *
seed = 0
c = make_cfg('A0', {})
lin = Lineage(seed_lin=1 + seed)
rng = np.random.default_rng(50 + seed)
S, own, C = make_lessons(rng, n_sib=2)
lessons = S + own[0] + [(q, a[0]) for q, a in C]
banned = {tuple(q) for q, a in lessons + own[1]}
ag = Agent(lin, c, 100 * seed + 11, None)
r1 = np.random.default_rng(100 * seed + 11 + 1000)
items = [list(q) + [a] for q, a in lessons] * c['n_rep'] + [list(r1.choice(V, 3, replace=False)) for _ in range(c['n_fill'])]
for i in r1.permutation(len(items)): ag.hear([int(v) for v in items[i]])
ag.pfc.theta_ref = ag.pfc.thr.theta
def sc(a, q): return a.M @ a.ctx.drive(a.pfc.code([a.pattern(ch) for ch in q]))
ag.theta = 0.3 * float(np.median([sc(ag, q).max() for q, a in lessons]))
a2 = clone(ag); r = np.random.default_rng(3000)
for i in r.integers(0, len(lessons), 600): a2.ask(lessons[i][0], 0.0)
# "next char after c in some question": for each char c, the set of q2 of questions whose q1 == c
nxt = {}
for q, a in lessons: nxt.setdefault(q[0], set()).add(q[1])
n_lost = n_expl = 0
for q, a in lessons:
    s0, s1 = sc(ag, q), sc(a2, q)
    if np.argmax(s0) == a and np.argmax(s1) != a:
        n_lost += 1; w = int(np.argmax(s1))
        n_expl += w in nxt.get(q[1], set())
print(f'lost argmax {n_lost}; new argmax = a char that follows q2 as the 2nd char of another question: {n_expl}')
# same with codes: overlap of the PFC code after (q1,q2) with the code of q2 heard first
J = [len(np.intersect1d(ag.pfc.code([ag.pattern(q[0]), ag.pattern(q[1])]), ag.pfc.code([ag.pattern(q[1])]))) / max(1, len(ag.pfc.code([ag.pattern(q[1])]))) for q, a in lessons]
print('share of the code of "q2 alone" inside the code of "q1 q2":', round(float(np.mean(J)), 2))
