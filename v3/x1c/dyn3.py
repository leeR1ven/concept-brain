# train in the world for n exchanges, then test reproducibility and P_in predictions with learning off
import sys, os, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from world import *
from core3 import Brain
from run import parse, run_until, mechanics, probe, summarize_probe, make_brain
kv = {k: parse(v) for k,v in (a.split('=',1) for a in sys.argv[1:])}
n = kv.pop('n', 300); care = kv.pop('care', 'format'); wait = kv.pop('wait', 12)
ph = Physics(); w = World(ph, 0, care=care, wait=wait); b = make_brain(ph, w, 0, kv)
t0 = time.time(); last = 0
for m in range(100, n+1, 100) if n > 0 else []:
    run_until(b, w, m)
    print(m, mechanics(w, last, m), 'L31 max %.2f mean %.4f n1/n2/n3 mean %s max %s min %s'%(b._pred_mat().max(), b._pred_mat().mean(), [round(float(np.mean(b.cnt[k][-200:])),1) for k in ('n1','n2','n3')], [max(b.cnt[k][-200:]) for k in ('n1','n2','n3')], [min(b.cnt[k][-200:]) for k in ('n1','n2','n3')]), '%.0fs'%(time.time()-t0), flush=True); last = m
rng = np.random.default_rng(5)
def run_q(f, x, pre=10, post=14):
    sched = [None]*pre + [SID[s] for s in question(f,x) for _ in range(3)] + [None]*post
    st=[]; pr=[]; global cn
    for cg in sched:
        snd = render(ph, [('s', cg, C1, 1.0)] if cg is not None else [], rng)
        b.step(snd, 0.0, learn=False); st.append(b.l3.copy()); pr.append(b.pred.copy()); cn.append((int(b.l1.sum()),int(b.l2.sum()),int(b.l3.sum())))
    return np.array(st), np.array(pr)
snap = b.snapshot()
cn=[]; S = {}; PR = {}
order = [(2,0),(2,1),(0,0),(3,5),(1,6),(2,0),(0,3),(2,1),(3,5),(0,0),(2,0),(2,1)]
for (f,x) in order:
    st, pr = run_q(f,x); S.setdefault((f,x), []).append(st); PR.setdefault((f,x), []).append(pr)
b.restore(snap)
qe = 19
def jac(a,c): return (a&c).sum()/max(1,(a|c).sum())
for k in (0, 5, 11):
    same = np.mean([jac(S[p][i][qe+k], S[p][j][qe+k]) for p in S for i in range(len(S[p])) for j in range(i+1,len(S[p]))])
    dfill = np.mean([jac(S[(2,0)][i][qe+k], S[(2,1)][j][qe+k]) for i in range(3) for j in range(3)])
    dfr = np.mean([jac(S[(2,0)][i][qe+k], S[(0,0)][j][qe+k]) for i in range(3) for j in range(2)])
    print(f'  Q_end+{k}: same-Q {same:.2f}  same frame/other filler {dfill:.2f}  same filler/other frame {dfr:.2f}')
# what do the predictions (inner hearing) look like after the question? decode by content overlap
C = ph.content
for p in [(2,0),(2,1),(0,0),(3,5)]:
    pr = PR[p][0]
    dec = []
    for k in range(-3, 12):
        v = np.zeros(ph.n_c, bool); idx = np.nonzero(pr[qe+k] & b.l1_content)[0]; v[b.l1_feat[idx]] = True
        if v.sum() == 0: dec.append('.'); continue
        ov = (C & v).sum(1); s = int(np.argmax(ov)); dec.append(SYL[s] + str(int(ov[s])))
    print(p, question(*p), answer(*p, w.props), ' '.join(dec))
print(summarize_probe(probe(b, ph, w, [(f,x) for f in range(4) for x in range(8)])))
print('n1', b.n1, 'L1 content lines', int(b.l1_content.sum()), 'L33 max %.3f'%b.L33.max())
print('counts during probe Qs (l1,l2,l3):', cn[:45:3])
