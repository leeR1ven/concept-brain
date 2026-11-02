"""H drive scale: best H score / cortex theta for stored lessons, unheard pairs, and after amnesia"""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(1, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '..', 'x345'))   # repo: x4_merge.py
from x8 import *
seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
import json as _j
lin = Lineage(seed_lin=1 + seed)
rng = np.random.default_rng(50 + seed)
S, own, C = make_lessons(rng, n_sib=2)
lessons = S + own[0] + [(q, a[0]) for q, a in C]
banned = {tuple(q) for q, a in lessons + own[1]}
_o = {k: _j.loads(v) for k, v in (kv.split('=') for kv in sys.argv[1].split(','))}
hb = hippo_birth(5000 + seed, _o.get('n_h', 4096), _o.get('fan_in', 64))
import json as _j
over = {k: _j.loads(v) for k, v in (kv.split('=') for kv in sys.argv[1].split(','))}
c = make_cfg('A34', over)
print(over)
ag = Agent(lin, c, 100 * seed + 11, hb)
r1 = np.random.default_rng(100 * seed + 11 + 1000)
items = [list(q) + [a] for q, a in lessons] * c['n_rep'] + [list(r1.choice(V, 3, replace=False)) for _ in range(c['n_fill'])]
for i in r1.permutation(len(items)): ag.hear([int(v) for v in items[i]])
ag.pfc.theta_ref = ag.pfc.thr.theta; ag.hpop.theta_ref = ag.hpop.thr.theta
ag.theta = 0.3 * float(np.median([(ag.M @ ag.ctx.drive(ag.pfc.code([ag.pattern(ch) for ch in q]))).max() for q, a in lessons]))
def hs(a, q):
    pats = [a.pattern(ch) for ch in q]
    return a.M @ a.hip.drive(a.hpop.code(pats))
def cs(a, q):
    return a.M @ a.ctx.drive(a.pfc.code([a.pattern(ch) for ch in q]))
def summ(a, items, tag, f=hs):
    b, r2, ok = [], [], []
    for q, aa in items:
        s = f(a, q); o = np.argsort(-s); b.append(s[o[0]] / a.theta); r2.append(s[o[0]] / max(1e-9, s[o[1]])); ok.append(o[0] == aa)
    b, r2 = np.array(b), np.array(r2)
    print(f'{tag:34s} best/theta q10 {np.quantile(b,.1):.2f} med {np.median(b):.2f} q90 {np.quantile(b,.9):.2f} | best/2nd med {np.median(r2):.2f} q90 {np.quantile(r2,.9):.2f} | argmax right {np.mean(ok):.2f}')
rr = np.random.default_rng(1)
unheard = []
while len(unheard) < 60:
    q = tuple(int(v) for v in rr.choice(V, 2, replace=False))
    if q not in banned: unheard.append((q, -1))
print('H code size', np.mean([len(ag.hpop.code([ag.pattern(ch) for ch in q])) for q, a in lessons]))
summ(ag, lessons, 'H lessons after p1')
summ(ag, unheard, 'H unheard after p1')
summ(ag, lessons, 'CTX lessons after p1', cs)
# one-shot new lessons in the adult
a2 = clone(ag)
for q, a in own[1][:20]: a2.hear(list(q) + [a])
for f in filler_lines(rr, 20, banned): a2.hear(f)
summ(a2, own[1][:20], 'H one-shot new (+20 filler)')
summ(a2, own[1][:20], 'CTX one-shot new', cs)

