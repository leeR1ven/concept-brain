"""Count-model reference curve (numpy only, no brain): the same held-out occurrences as probes11.heldout_eval
(200 test dialogues, seed 4242) and the same training stream; direct top-1 / top-5 for every checkpoint amount,
including the response subset. Mirrors probes11.ngram_eval without the code readout.
usage: python3 ngram_curve.py [--amounts 1000,3000,10000,30000,100000]"""
import argparse
import json
import os

import numpy as np

from world11 import Dialogues, script_dialogue

HERE = os.path.dirname(os.path.abspath(__file__))


def occurrences(test, n_dlg, seed=4242):
    occ = []
    for d in range(n_dlg):
        hist = []
        for x in script_dialogue(test, d, seed):
            if x is not None and x[1] == 0:
                if hist:
                    occ.append((x[0], hist[-1], hist[-2] if len(hist) > 1 else -1, hist[-3] if len(hist) > 2 else -1,
                                x[6], x[7]))
                hist.append(x[0])
    return occ


def curve(train, n, occ, K):
    seqs = [np.concatenate(train.turns(d)).astype(np.int64) for d in range(n)]
    uni = np.bincount(np.concatenate(seqs), minlength=K)
    tabs = {}
    for name, keyf, L in (('b1', lambda s: s[:-1] * K + s[1:], 2), ('t1', lambda s: (s[:-2] * K + s[1:-1]) * K + s[2:], 3),
                          ('b2', lambda s: s[:-2] * K + s[2:], 3), ('t2', lambda s: (s[:-3] * K + s[1:-2]) * K + s[3:], 4)):
        tabs[name] = np.unique(np.concatenate([keyf(s) for s in seqs if len(s) >= L]), return_counts=True)
    r0 = np.bincount(np.array([t[0] for d in range(n) for t in train.turns(d)[1:]], np.int64), minlength=K)
    uo, r0o = list(np.argsort(-uni, kind='stable')[:10]), list(np.argsort(-r0, kind='stable')[:5])

    def top(name, ctx, fill):
        u, c = tabs[name]
        lo, hi = np.searchsorted(u, [ctx * K, ctx * K + K])
        L = list((u[lo:hi] - ctx * K)[np.argsort(-c[lo:hi], kind='stable')][:5])
        for x in list(fill) + uo:
            if len(L) >= 5:
                break
            if x not in L:
                L.append(x)
        return L
    kinds = ('unigram', 'reply_unigram', 'bigram', 'trigram', 'bigram_lag2', 'trigram_lag2')
    H = {k: np.zeros((len(occ), 2), bool) for k in kinds}
    for j, (s, p1, p2, p3, turn, pos) in enumerate(occ):
        b1 = top('b1', p1, [])
        b2 = top('b2', p2, []) if p2 >= 0 else uo[:5]
        L = dict(unigram=uo[:5], reply_unigram=r0o, bigram=b1, trigram=top('t1', p2 * K + p1, b1) if p2 >= 0 else b1,
                 bigram_lag2=b2, trigram_lag2=top('t2', p3 * K + p2, b2) if p3 >= 0 else b2)
        for k in kinds:
            H[k][j] = (L[k][0] == s, s in L[k][:5])
    resp = np.array([o[4] >= 1 and o[5] == 0 for o in occ])
    out = {k: dict(top1=round(float(h[:, 0].mean()), 4), top5=round(float(h[:, 1].mean()), 4)) for k, h in H.items()}
    out['response'] = {k: dict(top1=round(float(h[resp, 0].mean()), 4), top5=round(float(h[resp, 1].mean()), 4))
                       for k, h in H.items()}
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--amounts', default='1000,3000,10000,30000,100000')
    a = ap.parse_args()
    train = Dialogues.load(os.path.join(HERE, 'data', 'train.npz'))
    test = Dialogues.load(os.path.join(HERE, 'data', 'heldout.npz'))
    K = 426 * 5
    occ = occurrences(test, 200)
    res = dict(n_occ=len(occ), n_response=int(sum(o[4] >= 1 and o[5] == 0 for o in occ)))
    for n in [int(x) for x in a.amounts.split(',')]:
        res[n] = curve(train, n, occ, K)
        print(n, json.dumps(res[n]), flush=True)
    json.dump(res, open(os.path.join(HERE, 'results', 'ngram_curve.json'), 'w'), indent=1)
