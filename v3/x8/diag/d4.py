"""trace corrections: per judged ask, cortex and fast-store drive for old / new answers (in theta units)"""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(1, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '..', 'x345'))   # repo: x4_merge.py
import x8
from x8 import *
arm = sys.argv[1]; over = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
n_st = int(sys.argv[3]) if len(sys.argv) > 3 else 300
seed = 0
c = make_cfg(arm, over)
lin = Lineage(seed_lin=1 + seed)
hb = hippo_birth(5000 + seed, c['n_h'], c['fan_in']) if c['hip'] else None
rng = np.random.default_rng(50 + seed)
S, own, C = make_lessons(rng, n_sib=2)
lessons = S + own[0] + [(q, a[0]) for q, a in C]
banned = {tuple(q) for q, a in lessons + own[1]}
ag = Agent(lin, c, 100 * seed + 11, hb)
r1 = np.random.default_rng(100 * seed + 11 + 1000)
items = [list(q) + [a] for q, a in lessons] * c['n_rep'] + [list(r1.choice(V, 3, replace=False)) for _ in range(c['n_fill'])]
for i in r1.permutation(len(items)): ag.hear([int(v) for v in items[i]])
ag.pfc.theta_ref = ag.pfc.thr.theta
if ag.hip is not None: ag.hpop.theta_ref = ag.hpop.thr.theta
ag.theta = 0.3 * float(np.median([(ag.M @ ag.ctx.drive(ag.pfc.code([ag.pattern(ch) for ch in q]))).max() for q, a in lessons]))
w_own = float(over.get('w_own', 0.0))
stretch(ag, lessons, n_st, n_st, w_own, np.random.default_rng(3000 + seed), banned, set(), [])
rc = np.random.default_rng(4000 + seed)
idx = rc.choice(50, 4, replace=False)
changed = []
for i in idx:
    a = int(rc.integers(V - 1)); a += a >= lessons[i][1]
    changed.append((int(i), a))
orig_ask = Agent.ask
log = []
def ask(self, q, w_own=0.0, expect=None, judged=False):
    if judged:
        old = [lessons[i][1] for i, a in changed if lessons[i][0] == q][0]
        pats = [self.pattern(ch) for ch in q]
        cs = self.M @ self.ctx.drive(self.pfc.code(pats)) / self.theta
        hs = self.c['r_hip'] * (self.M @ self.hip.drive(self.hpop.code(pats))) / self.theta if self.hip is not None else np.zeros(V)
        r = orig_ask(self, q, w_own, expect, judged)
        log.append((q, f'ctx old {cs[old]:.2f} new {cs[expect]:.2f} | hip old {hs[old]:.2f} new {hs[expect]:.2f} -> said {r[0]} (old {old}, new {expect}) recast {r[1]}'))
        return r
    return orig_ask(self, q, w_own, expect, judged)
Agent.ask = ask
res = corrections(ag, lessons, changed, w_own, np.random.default_rng(5000 + seed), banned, int(over.get('R', 8)))
for q in sorted({q for q, _ in log}):
    print('Q', q)
    for qq, s in log:
        if qq == q: print('   ', s)
print([p['first'] for p in res['per']], res['rounds'][-1])
