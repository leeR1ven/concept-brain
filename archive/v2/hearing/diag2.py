# unmatched detected onsets: offset to the nearest true onset (any source) and the class of that source
import sys, pickle, collections
import numpy as np
from ear_eval import Listener, true_lines, label_at, Ear0, feat0
fn = sys.argv[1]; cfg = dict(a.split('=',1) for a in sys.argv[2:])
R = pickle.load(open(fn,'rb')); seed = R['seed']
F0 = feat0(Ear0(seed=seed), R['warm'])
tot = collections.Counter(); offs = collections.Counter()
for Lid in list(R['fresh'])[:2]:
    ear = Ear0(seed=seed); ear.set_feat0(F0)
    L = Listener(ear, F0, 'gate', R['warm'], cfg)
    fr = R['fresh'][Lid].astype(np.int64); gt = R['gt'][Lid]
    win = np.zeros((8,6), np.int64)
    for t in range(len(fr)):
        win = np.concatenate([win[:,2:], fr[t]],1); L.tick(t, win, True)
    TL = true_lines(gt); st = sorted((v['t_first'], v['att0']) for v in TL.values())
    ts = np.array([s[0] for s in st])
    for te, t in L.onsets:
        tot['det'] += 1
        i = np.argmin(np.abs(ts - te)); d = te - ts[i]
        if d == 0: tot['exact'] += 1; continue
        if not gt[t]: tot['noise_only'] += 1; continue
        offs[(int(np.clip(d, -4, 4)), min(st[i][1], 3), 'bt' if te != t else '')] += 1
print(dict(tot)); print(sorted(offs.items(), key=lambda x: -x[1])[:20])
