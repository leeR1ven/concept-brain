"""Lemma (sensitivity of picking the k most driven): add to every unit's input push f_i an own-link push r_i with
|r_i| <= rho. A unit can change sides only if f_i lies within 2 rho of the k-th push theta (theta moves by at most rho).
So at least  k - min(#members with f_i <= theta + 2 rho, #others with f_i >= theta - 2 rho)  of the k units stay, whatever
the own links do: a guaranteed share of the code set by the input alone. Computed from the last layer's real input push
at birth (core/theory2.py saves it), for rho as a multiple of the push's spread over units. numpy only.
usage: python3 theory/stability.py <codes.npz> [density]"""
import json
import sys

import numpy as np


def main():
    z = np.load(sys.argv[1])
    dens = float(sys.argv[2]) if len(sys.argv) > 2 else 0.015
    P = z['push'].astype(np.float32).reshape(-1, z['push'].shape[-1])        # [S*3, n4] input push per tick
    n = P.shape[1]
    k = max(1, int(round(dens * n)))
    sd = P.std(1, keepdims=True)
    srt = -np.sort(-P, axis=1)
    theta = srt[:, k - 1:k]
    out = {}
    for m in (0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0):
        rho = m * sd
        leave = ((P >= theta) & (P <= theta + 2 * rho)).sum(1)
        enter = ((P < theta) & (P >= theta - 2 * rho)).sum(1)
        keep = 1.0 - np.minimum(np.minimum(leave, enter), k) / k
        out[m] = round(float(keep.mean()), 3)
    print(json.dumps(dict(units=n, k=k, spread=round(float(sd.mean()), 3),
                          guaranteed_share_kept_by_rho_over_spread=out)))


if __name__ == '__main__':
    main()
