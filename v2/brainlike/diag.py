# instrumented diagnosis: load a stage-A agent, train N dialogues of a stage (model tutor), then instrumented probes
#   python3 -B diag.py AGENT.pkl WORLD.pkl STAGE N
import sys, pickle, numpy as np
argv = sys.argv; sys.argv = ['x']
import bench_run as br
from agent import NSM, NLM, DEFAULT
from world import OBJECTS
from rules import DecisionPath

a = pickle.load(open(argv[1], 'rb')); w = pickle.load(open(argv[2], 'rb'))
for k, v in dict(A_hold=-1, rel_hold=0.0, rel=0.0, last_feats=None, line_seq=[], rec_seq=[], rec_feats=None, t_mouth=-100).items():
    if not hasattr(a, k):
        setattr(a, k, v)
for k, v in DEFAULT.items():
    a.c.setdefault(k, v)
for s in ["Vr"]:
    if s not in a.G:
        a.G[s] = DecisionPath(a.nx, 1, eta=a.c.eta_g)
a.c['babble'] = False
stage, N = argv[3], int(argv[4])
wB = br.mk(stage, 0, prev=w)
wB.run(a, N // 2)
n0 = len(wB.log)
wB.run(a, N - N // 2)
from metrics import evaluate
res = evaluate(wB.log[n0:])
print('BENCH second half:', br.brief(res, stage[0]) if len(stage) == 1 else '', flush=True)
for st in stage:
    print('  ', st, br.brief(res, st), flush=True)
print('   last probes:', ' '.join(f"{r['prompt']}->{r['reply']}" for r in wB.log if r['kind'] == 'probe')[-400:], flush=True)
import physics
v = wB.voice


def clean(ch):
    st2 = physics.Stream(v.F, v.T, 2, 3, v.levels, 0.0); none = (None, None, False)
    for k in range(3):
        win = st2.tick(((ch, v.hear(ch, 'other'), k == 0), none, none))
    return a.ear.feat0(win)


unit = {}
for ch in v.chars:
    j, cs = a.syl_detect(clean(ch))
    if cs >= a.c.th_det:
        unit[j] = ch
name = lambda j: unit.get(j, '?') if j < NSM else ''.join(unit.get(s, '?') for s in a.chunk_syl.get(j, (0,)))
print('chunks:', ' '.join(name(k) for k in sorted(a.chunk_syl)))
code = wB.cur.code
inv = {ch: j for j, ch in unit.items()}
for o in OBJECTS:
    if all(ch in inv for ch in o) and (inv[o[0]], inv[o[1]]) in a.chunks:
        wid = a.chunks[(inv[o[0]], inv[o[1]])]; t = a.VL['o'].col(wid); bits = code[(0, ('obj', o))] % 240
        print(f"{o} n{a.vl_n[wid]} own-code mass {t[bits].sum():.2f}", end=' | ')
print()
for wd in ['左边', '中间', '右边']:
    if all(ch in inv for ch in wd) and (inv[wd[0]], inv[wd[1]]) in a.chunks:
        wid = a.chunks[(inv[wd[0]], inv[wd[1]])]
        print(wd, 'LA', np.round(a.LA.col(wid), 2), 'n', a.vl_n[wid])
orig = a.boundary


def bnd(body, first):
    sc, se = orig(body, first)
    p, g = a.last_pred
    top = np.argsort(-p[:NSM])[:3]
    cd = a.tag['cands']
    print(f"   {'F' if first else ' '} A{a.A} rel{a.rel:.1f} prev {name(a.prev) if a.prev >= 0 else '^'} gates",
          {k: round(x, 2) for k, x in g.items()},
          'cands', {k: name(int(np.argmax(v))) for k, v in cd.items() if v is not None},
          'top', [(name(j), round(float(p[j]), 2)) for j in top], 'END %.2f' % p[NSM])
    return sc, se


a.boundary = bnd
items = []
rng = np.random.default_rng(5)
for k in range(3):
    objs = rng.choice(OBJECTS, 3, replace=False)
    sc = tuple((str(o), int(rng.integers(6)), int(rng.integers(3))) for o in objs)
    s = k % 3
    items.append(dict(prompt='这是什么', reply=sc[s][0], kind='probe', scene=sc, attend=s, task='what'))
    items.append(dict(prompt=sc[(s + 1) % 3][0] + '在哪里', reply='在' + ['左边', '中间', '右边'][(s + 1) % 3], kind='probe',
                      scene=sc, task='where'))
    items.append(dict(prompt=['左边', '中间', '右边'][s] + '是什么', reply=sc[s][0], kind='probe', scene=sc, task='slot'))
w2 = br.mk(stage, 0, prev=wB)
for it in items:
    print(it['prompt'], it.get('attend'), it['scene'])
    log = w2.run_script(a, [it], stage=stage)
    print('  ->', log[-1]['reply'], log[-1]['acceptable'])
