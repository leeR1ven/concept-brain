"""Identify, then look up (the coordinator's check): nearest class mean on the cue -> the identified syllable's
commonest follower. If this beats the linear readout, the information is in the code and a recognition stage before
the association extracts it. Same hearings and split as multi.py; both weightings (by the cue's corpus frequency, as
the next-syllable figures, and unweighted, as the identification figures). numpy only.
usage: python3 lookup.py multi.npz [layer=1]"""
import json
import sys

import numpy as np


def unpack(a, n):
    return np.unpackbits(a, axis=-1)[..., :n].astype(np.float32)


def cos_sim(A, B):
    An = A / np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-9)
    Bn = B / np.maximum(np.linalg.norm(B, axis=1, keepdims=True), 1e-9)
    return An @ Bn.T


def main():
    z = np.load(sys.argv[1])
    layer = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    n1 = int(z['n1'])
    N = z['N']
    S = N.shape[0]
    piA = N.sum(1) / N.sum()
    best = N.argmax(1)
    tr = unpack(z['train'][:, :, :, layer], n1)
    te = unpack(z['test'][:, :, :, layer], n1).reshape(S, -1, n1)
    vo = unpack(z['voice'][:, :, layer], n1)
    V, R = tr.shape[1], tr.shape[2]
    plans = {'1': [(0, 0)], '3': [(v, 0) for v in range(V)], '6': [(v, k) for v in range(V) for k in range(2)],
             '18': [(v, k) for v in range(V) for k in range(R)]}
    out = {'layer': ['copies', 'last layer'][layer], 'unigram_top1_weighted': round(float((piA * (N.sum(0).argmax() == best)).sum()), 3)}
    for name, sel in plans.items():
        mu = np.stack([tr[:, v, k] for v, k in sel], 1).mean(1)
        rep = {}
        for tname, T in (('fresh', te), ('voice4', vo)):
            Tq = T.reshape(-1, n1)
            tC = np.repeat(np.arange(S), T.shape[1])
            ident = cos_sim(Tq, mu).argmax(1)
            ok_id = ident == tC
            ok_lu = best[ident] == best[tC]
            w = piA[tC] / piA[tC].sum()
            rep[tname] = dict(identified_unweighted=round(float(ok_id.mean()), 3), identified_weighted=round(float((w * ok_id).sum()), 3),
                              lookup_top1_unweighted=round(float(ok_lu.mean()), 3), lookup_top1_weighted=round(float((w * ok_lu).sum()), 3))
        out[name] = rep
    print(json.dumps(out))


if __name__ == '__main__':
    main()
