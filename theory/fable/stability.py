"""The last layer under its own links: how much of the code the input still sets (docs/THEORY_fable.md section 4).
From the recorded input push (float32, codes.py) at the ticks a syllable's code is formed. The own links add r_i to
unit i. Two cases, both for a total push per unit <= rho:
  worst   the lemma: units that can change sides lie within 2 rho of the k-th push (a guaranteed share);
  typical r_i = rho on a set of k units chosen at random w.r.t. the input (the previous state's partners), 0 elsewhere;
          the share of the k lit units that stay, averaged over cues and ticks.
Also the input push's own spread (sigma) and the gap from the k-th push to the mean, for the parameter table.
usage: python3 stability.py codes.npz"""
import json
import sys

import numpy as np

from readout import load


def main():
    z = load(sys.argv[1])
    P = z['push'][0][:, 1:4]                                # train hearing, ticks 4..6: [S, 3, n]
    L = z['l4'][0][:, 5:8]                                  # the lit sets at those ticks
    S, T, n = P.shape
    P = P.reshape(S * T, n)
    if len(sys.argv) > 2 and sys.argv[2] == 'centred':      # each unit's own mean over cues taken out (the usual
        P = P - P.mean(0, keepdims=True)                    # level u4 does this in the model; cap was computed so)
    k = L.reshape(S * T, n).sum(1)
    ok = k > 0
    P, k = P[ok], k[ok]
    sd = P.std(1)
    srt = -np.sort(-P, axis=1)
    theta = srt[np.arange(len(k)), k - 1]
    rng = np.random.default_rng(0)
    out = {}
    for m in (0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0):
        rho = m * sd
        leave = ((P >= theta[:, None]) & (P <= (theta + 2 * rho)[:, None])).sum(1)
        enter = ((P < theta[:, None]) & (P >= (theta - 2 * rho)[:, None])).sum(1)
        worst = 1.0 - np.minimum(np.minimum(leave, enter), k) / k
        typ = []
        for i in rng.choice(len(k), min(300, len(k)), replace=False):
            r = np.zeros(n, np.float32)
            r[rng.choice(n, int(k[i]), replace=False)] = rho[i]
            new = np.argpartition(-(P[i] + r), int(k[i]) - 1)[:int(k[i])]
            old = np.argpartition(-P[i], int(k[i]) - 1)[:int(k[i])]
            typ.append(len(np.intersect1d(new, old)) / k[i])
        out[m] = dict(worst=round(float(worst.mean()), 3), typical=round(float(np.mean(typ)), 3))
    gap = (theta - P.mean(1)) / sd
    print(json.dumps(dict(units=n, lit_mean=round(float(k.mean()), 1), sigma_input=round(float(sd.mean()), 3),
                          kth_push_above_mean_in_sigma=round(float(gap.mean()), 2),
                          share_kept_by_rho_over_sigma=out)))


if __name__ == '__main__':
    main()
