# One-shot naming on the bench: where does it fail?  (privileged audit on a saved agent; storage on, readouts frozen)
import sys, pickle, collections, numpy as np
sys.dont_write_bytecode = True
sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__)))
from agent import DevAgent, BENCH
sys.path.insert(0, BENCH)
import world as BW
ag = pickle.load(open(sys.argv[1], "rb")); core = ag.core
ag.learning = True; core.c.learn_listen = False
w = BW.World("D", seed=0, tutor="recast", probe_frac=1.0, held_frac=0.0); v = w.voice
def cat_of(ch):
    j, m = ag.Z.best(ag.own_pattern(v, ch)); return j if m >= 0.75 else None
lab = {}
for ch in set("".join(BW.NOVEL + BW.OBJECTS) + "这是什么吗"):
    j = cat_of(ch)
    if j is not None: lab[j] = ch
rec = []
ob, osc = core.own_begin, core.own_scores
def own_begin(eye, att):
    st = ob(eye, att); st['_first'] = True; return st
def own_scores(st):
    out = osc(st)
    if st.get('_first'):
        st['_first'] = False
        A, d, gd, total, cands = core._drives(st)
        top = [lab.get(int(i), '?') for i in np.argsort(-total[:core.c.V])[:3]]
        rec.append(dict(P=''.join(lab.get(i, '?') for i in st['P']), fam=st['fam'],
                        src={k: ''.join(lab.get(i, '?') for i in ch.items) for k, ch in st['src'].items()},
                        cands={k: lab.get(c[0], '-') for k, c in cands.items()}, gates=np.round(np.clip(gd, 0, 1), 2).tolist(), top=top))
    return out
core.own_begin, core.own_scores = own_begin, own_scores
w.run(ag, 120)
n = 0; stats = collections.Counter()
for r in w.log:
    if r['task'] != 'oneshot':
        continue
    X = r['acceptable'][0]
    has_cat = all(cat_of(ch) is not None for ch in X)
    has_prog = all(cat_of(ch) in ag.cat2prog for ch in X if cat_of(ch) is not None)
    stats['n'] += 1; stats['categories'] += has_cat; stats['programs'] += has_prog
    stats['correct'] += r['reply'] in r['acceptable']
    if n < 8:
        print(f"oneshot X={X} reply={r['reply']!r} cats={has_cat} progs={has_prog}")
        n += 1
print(dict(stats))
# first-boundary traces for the 这是什么 questions
qs = [x for x in rec if x['P'] == '这是什么']
for x in qs[:10]:
    print(x)
