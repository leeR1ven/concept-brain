# debug the context probe: train briefly (A300 + B300) and pickle; then probe with diagnostics
import os, sys, pickle, copy
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from run_t2 import *            # noqa
A = dict(a.split('=', 1) for a in sys.argv[1:])
pk = A.get('pk', 'out/smoke/dbg_agent.pkl')
kw = dict(TUNE_BEST); kw.update(THINK_FINAL); kw.update(ARCH_V2); kw.update(ct_lam=0.75, ctx_keys='age')
wkw = {'eye_layout': 'retino'}
if not os.path.exists(pk):
    wA = World('A', seed=0, tutor='recast', **wkw)
    ag = T2Agent(wA, seed=0, **kw); wA.run(ag, 300); ag.babble(wA.voice, 35960)
    w = World('B', seed=0, tutor='recast', prev=wA, **wkw); w.run(ag, 300)
    pickle.dump(ag, open(pk, 'wb'))
ag = pickle.load(open(pk, 'rb'))
core = ag.core
# instrument: at every reply thought, record held items, old keys, best drive
rec = []
orig = core.think_new
def tn(kind, eye, att, t=0):
    th = orig(kind, eye, att, t=t)
    if kind == 'reply':
        tot, Aa = core._total_ro(th['st'])
        j = int(np.argmax(tot[:core.c.V]))
        rec.append(dict(t=t, held=core.thought_items(), old=th['st'].get('Pold'), P=th['st']['P'], best=j, s=float(tot[j]), nA=len(Aa)))
    return th
core.think_new = tn
import run_t2
run_t2.copy = type('C', (), {'deepcopy': staticmethod(lambda x: x)})
r = run_t2.ctx_probe(ag, 0, int(A.get('gap', 0)), int(A.get('ntr', 40)), int(A.get('nte', 10)), wkw)
print({k: v for k, v in r.items() if k not in ('replies', 'targets')})
print('replies', r['replies'], r['targets'])
for x in rec[-12:]:
    print(x)
print('tstats', ag.tstats if hasattr(ag, 'tstats') else None)
