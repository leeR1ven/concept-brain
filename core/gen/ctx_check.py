"""Context check (Fable 11-12 round 10): is the auditory code's spread within a syllable mostly the previous syllable?
Identity (nearest class mean over all training hearings, cosine, lag 0) for test syllables that follow silence vs
those that follow another syllable, on a cached recording of offline_b.  usage: python -m core.gen.ctx_check <cache.pt>"""
import json
import sys

import numpy as np
import torch

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
HOLD = 3


def hearings(rec):
    F, L = rec
    X, Y, first, prev = [], [], [], []
    for b in range(F.shape[0]):
        lab = L[b]
        last = -1
        for t in range(F.shape[1] - HOLD):
            if lab[t] >= 0 and (t == 0 or lab[t - 1] != lab[t]):
                X.append(F[b, t:t + HOLD].sum(0))
                Y.append(lab[t])
                first.append(t == 0 or lab[t - 1] < 0)
                prev.append(last)
            if lab[t] >= 0:
                last = lab[t]
            elif t > 0 and lab[t - 1] >= 0:
                pass
    return torch.stack(X).float().to(DEV), np.array(Y), np.array(first), np.array(prev)


for path in sys.argv[1:]:
    Rtr, Rte, Rl = torch.load(path, weights_only=False)
    Xa, Ya, _, _ = hearings(Rtr)
    Xl, Yl, _, _ = hearings(Rl)
    Xa, Ya = torch.cat([Xa, Xl]), np.concatenate([Ya, Yl])
    Xb, Yb, Fb, Pb = hearings(Rte)
    cnt = np.bincount(Ya)
    top = [int(c) for c in np.argsort(-cnt)[:100]]
    ks = [c for c in top if (Ya == c).sum() >= 2]
    M = torch.stack([Xa[torch.as_tensor(Ya == c, device=DEV)].mean(0) for c in ks])
    Mn = M / M.norm(dim=1, keepdim=True).clamp_min(1e-9)
    pos = {c: i for i, c in enumerate(ks)}
    m = np.array([y in pos for y in Yb])
    Xn = Xb / Xb.norm(dim=1, keepdim=True).clamp_min(1e-9)
    pred = (Xn @ Mn.T).argmax(1).cpu().numpy()
    ti = np.array([pos.get(int(y), -1) for y in Yb])
    ok = (pred == ti) & m
    out = dict(all=round(ok[m].mean(), 3), after_silence=round(ok[m & Fb].mean(), 3),
               mid_turn=round(ok[m & ~Fb].mean(), 3), n_after_silence=int((m & Fb).sum()), n_mid=int((m & ~Fb).sum()))
    # class means per (syllable, previous syllable) where there are enough: does knowing the context help?
    print('CONTEXT', path.split('/')[-1], json.dumps(out), flush=True)
