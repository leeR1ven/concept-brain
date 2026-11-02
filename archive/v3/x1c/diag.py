import sys, os, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from world import *
from core import Brain
from run import run_until, mechanics, probe, summarize_probe
kv = dict(a.split('=',1) for a in sys.argv[1:])
from run import parse
kv = {k: parse(v) for k,v in kv.items()}
n = kv.pop('n', 200); care = kv.pop('care','format')
ph = Physics(); w = World(ph, 0, care=care); b = Brain(ph, 0, **kv)
t0=time.time()
stats=[]
last=0
for m in range(50, n+1, 50):
    nas=[]; npred=[]; thi=[]; tha=[]; mus=0; nbs=[]
    while not (w.n_ex >= m and w.ex['state']=='gap'):
        mm = b.step(w.sound, getattr(w,'r_last',0.0)); s, r = w.step(mm); w.r_last=r
        nas.append(b.p[b.na:].sum()); npred.append(b.pred.sum()); thi.append(b.th_in); tha.append(b.th_as); mus += mm.any(); nbs.append(b.nb)
    me = mechanics(w, last, m); last=m
    print(m, 'ticks', len(nas), 'n_as %.1f [%d-%d]'%(np.mean(nas),min(nas),max(nas)), 'pred %.1f'%np.mean(npred), 'th_in %.2f th_as %.2f'%(np.mean(thi),np.mean(tha)),
          'mouth %d'%mus, 'nb %.2f'%np.mean(nbs), 'Wl %.4f max %.2f'%(b.Wl.mean(), b.Wl.max()), me, '%.0fs'%(time.time()-t0), flush=True)
pr = probe(b, ph, w, [(f,x) for f in range(4) for x in range(8)])
print(summarize_probe(pr))
for r in pr[:32]: print(r['f'], FILL[r['x']], r['A'], r['syl'], r['lat'])
print('pot samples: th_in %.3f th_as %.3f mu_mean %.3f'%(b.th_in, b.th_as, b.mu.mean()))
print('Wl mean %.4f  max %.3f  Wcm max %.3f Wcb max %.3f Wsb max %.3f'%(b.Wl.mean(), b.Wl.max(), b.Wcm.max(), b.Wcb.max(), b.Wsb.max()))
