"""Section 4: the learning curve, computed. The readout after the first T dialogues (bigram counts of what was heard so
far, in order) predicts the next syllable that is most common in the whole corpus how often? numpy only.
usage: python3 theory/curve.py <codes.npz> <train.npz> '{"cov": 1, "links": "all", ...}'"""
import json
import sys

import numpy as np

from readout_exact import run


def main():
    z = dict(np.load(sys.argv[1]))
    t = np.load(sys.argv[2])
    kw = json.loads(sys.argv[3]) if len(sys.argv) > 3 else {'cov': 1, 'links': 'all'}
    s, to, do = t['syl'].astype(np.int64), t['turn_off'], t['dlg_off']
    S = len(z['syl'])
    ix = -np.ones(int(s.max()) + 1, np.int64)
    ix[z['syl']] = np.arange(S)
    keep = np.ones(len(s) - 1, bool)
    st = to[(to > 0) & (to < len(s))]
    keep[st - 1] = False
    for T in (300, 1000, 3000, 10000, 30000, 110000):
        end = to[do[min(T, len(do) - 1)]]                     # syllables of the first T dialogues
        a, b = ix[s[:end - 1]], ix[s[1:end]]
        ok = keep[:end - 1] & (a >= 0) & (b >= 0)
        NL = np.zeros((S, S))
        np.add.at(NL, (a[ok], b[ok]), 1.0)
        r = run(z, Nlearn=NL, **kw)
        print(json.dumps(dict(dialogues=T, top1=r['top1'], top5=r['top5'], mean_rank=r['mean_rank'], **kw)), flush=True)


if __name__ == '__main__':
    main()
