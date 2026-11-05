"""Where is the syllable's identity lost: already at the copies (the ear and sound area) or in the fixed path to the
last layer? Nearest-code identification and effective dimension at the first layer (the copies, ticks 1..4) and at
the last layer (ticks 4..6), ear noise and other voice; and whether confusions stay within the same base syllable
(the toned ids are base x 5 + tone - 1). numpy only.
usage: python3 where_lost.py codes.npz"""
import json
import sys

import numpy as np

from kernel import ov
from readout import load


def ident(Ctr, Cq, base):
    K = ov(Cq, Ctr)
    S = len(K)
    nn = K.argmax(1)
    own = np.diag(K)
    Ko = K.copy()
    np.fill_diagonal(Ko, -1)
    return dict(identified=round(float((nn == np.arange(S)).mean()), 3),
                same_base_when_wrong=round(float((base[nn] == base)[nn != np.arange(S)].mean()), 3),
                identified_base=round(float((base[nn] == base).mean()), 3),
                consistency=round(float(own.mean()), 3), best_other=round(float(Ko.max(1).mean()), 3),
                margin_positive=round(float((own > Ko.max(1)).mean()), 3))


def deff(C):
    X = C.astype(np.float32)
    Xc = X - X.mean(0)[None]
    G = Xc @ Xc.T
    return round(float(np.trace(G) ** 2 / (G ** 2).sum()), 1)


def main():
    z = load(sys.argv[1])
    base = z['syl'] // 5
    out = {}
    L1 = [z['ext'][h][:, 2:6].any(1) for h in range(3)]            # copies, ticks 1..4
    L4 = [z['l4'][h][:, 5:8].any(1) for h in range(3)]             # last layer, ticks 4..6
    for name, C in (('first layer (copies)', L1), ('last layer', L4)):
        out[name] = dict(size=round(float(C[0].sum(1).mean()), 1), chance=round(float(C[0].mean()), 4), d_eff=deff(C[0]),
                         ear=ident(C[0], C[1], base), voice=ident(C[0], C[2], base))
    # per tick at the last layer and per sound layer at the first layer are left to the reader; the unions suffice
    out['n_bases_among_600'] = int(len(np.unique(base)))
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
