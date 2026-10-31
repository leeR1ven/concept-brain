# diagnose unmatched detected onsets and junk samples for one recording / one listener (gate variant)
import sys, pickle, collections
import numpy as np
sys.argv += [] 
from ear_eval import Listener, true_lines, label_at, Ear0, feat0
fn, Lid = sys.argv[1], int(sys.argv[2]); cfg = dict(a.split('=',1) for a in sys.argv[3:])
R = pickle.load(open(fn,'rb')); seed = R['seed']
F0 = feat0(Ear0(seed=seed), R['warm']); ear = Ear0(seed=seed); ear.set_feat0(F0)
L = Listener(ear, F0, 'gate', R['warm'], cfg)
fr = R['fresh'][Lid].astype(np.int64); gt = R['gt'][Lid]
win = np.zeros((8,6), np.int64); e2s=[]; thr=[]
for t in range(len(fr)):
    win = np.concatenate([win[:,2:], fr[t]],1); L.tick(t, win, True); e2s.append(L.gate.last['e2']); thr.append(L.gate.last['thr'])
TL = true_lines(gt); starts = {v['t_first']: k for k,v in TL.items()}
bad = [(te,t) for te,t in L.onsets if te not in starts]
print('detected', len(L.onsets), 'unmatched', len(bad), 'mu nu', round(L.gate.mu,2), round(L.gate.nu,2))
for te,t in bad[:12]:
    near = [(k, v['t_first']) for k,v in TL.items() if abs(v['t_first']-te) <= 6]
    print(f' onset t={t} eff={te} e2 {[int(x) for x in e2s[te-3:t+3]]} thr {thr[t]:.1f} gt@t {[ (s[0],s[1],s[3],s[4],s[5]) for s in gt[t]]} true near {near}')
labs = collections.Counter(label_at(gt,t)[0] for t,c,li in L.samples if c is not None)
print('junk sample labels', {k:v for k,v in labs.items() if k in ('mis','none','overlap')})
