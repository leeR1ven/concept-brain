"""Theorem 2 (the Hebbian readout picks what follows), evaluated on the training corpus. numpy only.
Model (assumptions A1-A3 in docs/THEORY.md 2): contexts C = the previous syllable; N(C, j) = how often syllable j came
right after C. A cue A lights a code whose units reaching copy j number kp. Each of those units was in A's earlier codes
with probability c (consistency), and is in another context's code with probability o (overlap), independently.
After the usual level is taken out, B beats a competitor X with signal S = kp (c - o) (N(A,B) - N(A,X)) and the
fluctuation is a sum of independent bounded terms with variance
    V = kp [ o (1-o) sum_{C != A} (N(C,B) - N(C,X))^2 + c (1-c) (N(A,B) - N(A,X))^2 ],  each term <= M.
Bernstein: P(X ranks over B) <= exp(-S^2 / (2V + 2/3 M S)); a union over X bounds the error. All of S, V, M scale
with the amount heard in the same way, so the bound does not depend on how long it has listened.
usage: python3 theory/readout.py <train.npz> '{"c": .., "o": .., "kp": [..], "pairs": 20, "delta": 0.05}'"""
import json
import sys

import numpy as np


def bigrams(path):
    z = np.load(path)
    s, to = z['syl'].astype(np.int64), z['turn_off']
    keep = np.ones(len(s) - 1, bool)
    st = to[(to > 0) & (to < len(s))]
    keep[st - 1] = False
    keep &= s[:-1] != s[1:]
    n = int(s.max()) + 1
    M = np.zeros((n, n))
    np.add.at(M, (s[:-1][keep], s[1:][keep]), 1.0)
    return M


def exponents(M, a, b, c, o):
    """per competitor X: S^2 / (2V + 2/3 M S) for kp = 1 (the exponent is kp times this)"""
    d = M[:, b][:, None] - M                                  # N(C,B) - N(C,X), rows C, columns X
    da = d[a]
    rest = np.delete(d, a, axis=0)
    s = (c - o) * da
    v = o * (1 - o) * (rest ** 2).sum(0) + c * (1 - c) * da ** 2
    m = np.abs(d).max(0)
    with np.errstate(divide='ignore', invalid='ignore'):
        e = np.where(s > 0, s ** 2 / (2 * v + 2.0 / 3.0 * m * s), 0.0)
    e[b] = np.inf
    return e


def main():
    M = bigrams(sys.argv[1])
    arg = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
    c, o, delta = arg.get('c', 0.56), arg.get('o', 0.19), arg.get('delta', 0.05)
    kps, npairs = arg.get('kp', [8, 32, 128, 512]), arg.get('pairs', 20)
    off = M.copy()
    pairs, used = [], set()
    for k in np.argsort(-off, axis=None):
        a, b = divmod(int(k), M.shape[0])
        if b not in used:
            pairs.append((a, b))
            used.add(b)
        if len(pairs) == npairs:
            break
    cand = np.nonzero(M.sum(0) > 0)[0]                         # syllables ever heard after something
    res = {}
    for kp in kps:
        bounds, need = [], []
        for a, b in pairs:
            e = exponents(M, a, b, c, o)[cand]
            bounds.append(float(np.minimum(1.0, np.exp(-kp * e[np.isfinite(e)]).sum())))
            fin = e[np.isfinite(e)]
            # smallest kp with union bound <= delta: kp = ln(n / delta) / min exponent (a sufficient choice)
            need.append(float(np.log(len(fin) / delta) / max(fin.min(), 1e-12)) if (fin > 0).all() else float('inf'))
        res[kp] = dict(pairs_error_bound_le_delta=int((np.array(bounds) <= delta).sum()),
                       median_bound=round(float(np.median(bounds)), 3))
    print(json.dumps(dict(c=c, o=o, delta=delta, pairs=npairs, by_kp=res,
                          kp_needed_median=round(float(np.median(need)), 1),
                          kp_needed_per_pair=[round(x, 1) for x in need])))


if __name__ == '__main__':
    main()
