"""Upper bound on what any linear readout can identify from the codes: ridge classification (one-hot targets, kernel
form, the best ridge of a scan) at the first and the last layer, ear noise and other voice. numpy only.
usage: python3 linear_id.py codes.npz"""
import json
import sys

import numpy as np

from kernel import ov
from readout import load


def ridge_id(Ctr, Cq, lams=(1e-3, 1e-2, 1e-1, 1.0, 10.0)):
    S = len(Ctr)
    Ktr, K = ov(Ctr, Ctr), ov(Cq, Ctr)
    best = {}
    for lam in lams:
        A = Ktr + lam * np.trace(Ktr) / S * np.eye(S)
        Y = K @ np.linalg.solve(A, np.eye(S))
        best[lam] = round(float((Y.argmax(1) == np.arange(S)).mean()), 3)
    return best


def main():
    z = load(sys.argv[1])
    L1 = [z['ext'][h][:, 2:6].any(1) for h in range(3)]
    L4 = [z['l4'][h][:, 5:8].any(1) for h in range(3)]
    out = {}
    for name, C in (('first layer', L1), ('last layer', L4)):
        out[name] = dict(ear=ridge_id(C[0], C[1]), voice=ridge_id(C[0], C[2]))
    print(json.dumps(out))


if __name__ == '__main__':
    main()
