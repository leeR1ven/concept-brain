# best-match m of every in-line window by ground-truth label (aligned char / misaligned / overlap / none)
import sys, pickle, collections
import numpy as np
from ear_eval import Listener, label_at, Ear0, feat0
fn = sys.argv[1]; cfg = dict(a.split('=',1) for a in sys.argv[2:])
R = pickle.load(open(fn,'rb')); seed = R['seed']
F0 = feat0(Ear0(seed=seed), R['warm'])
Lid = 0
ear = Ear0(seed=seed); ear.set_feat0(F0)
L = Listener(ear, F0, 'gate', R['warm'], cfg)
fr = R['fresh'][Lid].astype(np.int64); gt = R['gt'][Lid]
win = np.zeros((8,6), np.int64); ms = collections.defaultdict(list)
T = len(fr)
for t in range(T):
    win = np.concatenate([win[:,2:], fr[t]],1); L.tick(t, win, True)
    if t > T // 2 and L.in_line and L.E_prev > 0:
        f = ear.respond(ear.encode(win))[F0]; j, m = L.Z.best(f)
        lab = label_at(gt, t)[0]
        ms['aligned' if lab not in ('mis','none','overlap') else lab].append(m)
for k, v in ms.items():
    v = np.array(v); print(f'{k:8s} n {len(v):6d} m p10/50/90 {np.percentile(v,10):.2f} {np.percentile(v,50):.2f} {np.percentile(v,90):.2f}  P(m>=0.75) {np.mean(v>=0.75):.2f} P(m>=0.85) {np.mean(v>=0.85):.2f}')
