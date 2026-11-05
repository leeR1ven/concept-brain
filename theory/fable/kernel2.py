"""What the last layer's code must satisfy (docs/THEORY_fable.md section 3), measured at the kernel level. numpy only.
For the recorded codes and for sparser codes re-selected from the same input push (the k' most driven per tick):
  - identification: is the cue's nearest training code its own? (ear noise / other voice), chance 1/S;
  - margin: own overlap minus the best other overlap, in units of chance;
  - effective dimension of the codes, d_eff = (tr C)^2 / tr(C^2), C = the code covariance (how many independent
    directions the 600 contexts really span);
  - kernel ridge (the prediction-error rule converged) top-1 over a range of ridge values.
usage: python3 kernel2.py codes.npz"""
import json
import sys

import numpy as np

from kernel import ov, top1
from readout import load


def resel(push, lit, frac):
    """per tick: the k' = frac x (lit count) most driven units"""
    S, T, n = push.shape
    out = np.zeros((S, T, n), bool)
    for s in range(S):
        for t in range(T):
            k = int(round(frac * lit[s, t]))
            if k > 0:
                out[s, t, np.argpartition(-push[s, t], k - 1)[:k]] = True
    return out


def analyse(Ctr, Cte, Cvo, N, piA, lams=(1e-4, 1e-3, 1e-2, 1e-1, 1.0)):
    S = N.shape[0]
    rep = dict(code_size=round(float(Ctr.sum(1).mean()), 1), density=round(float(Ctr.mean()), 4))
    Ktr = ov(Ctr, Ctr)
    X = Ctr.astype(np.float32)
    Xc = X - X.mean(0)[None]                                    # d_eff from the S x S Gram of the centred codes
    G = Xc @ Xc.T
    rep['d_eff'] = round(float(np.trace(G) ** 2 / (G ** 2).sum()), 1)
    for name, Cq in (('ear', Cte), ('voice', Cvo)):
        K = ov(Cq, Ctr)
        own = np.diag(K).copy()
        Ko = K.copy()
        np.fill_diagonal(Ko, -1)
        best_other = Ko.max(1)
        rep[f'{name}_identified'] = round(float((K.argmax(1) == np.arange(S)).mean()), 3)
        rep[f'{name}_consistency'] = round(float(own.mean()), 3)
        rep[f'{name}_best_other'] = round(float(best_other.mean()), 3)
        rep[f'{name}_mean_other'] = round(float((K.sum(1) - own).mean() / (S - 1)), 3)
        piN = N.sum(1)
        Yb = N / np.maximum(piN, 1)[:, None]
        res = {}
        for lam in lams:
            A = Ktr * piN[None, :] + lam * np.trace(Ktr) / S * np.eye(S)
            res[lam] = top1(K @ (piN[:, None] * np.linalg.solve(A, Yb)), N, piA)
        rep[f'{name}_ls_top1_by_ridge'] = res
        rep[f'{name}_count_top1'] = top1(K @ N, N, piA)
    return rep


def main():
    z = load(sys.argv[1])
    N = z['N']
    piA = N.sum(1) / N.sum()
    out = {}
    ticks = (4, 5, 6)
    lit = np.stack([z['l4'][0][:, t + 1].sum(1) for t in ticks], 1)
    P = [z['push'][h][:, 1:4] for h in (0, 1)]                        # ticks 4..6 of the two C1 hearings
    base = [np.stack([z['l4'][h][:, t + 1] for t in ticks], 1) for h in (0, 1, 2)]
    out['as recorded'] = analyse(base[0].any(1), base[1].any(1), base[2].any(1), N, piA)
    for frac in (0.5, 0.25, 0.1):
        A = resel(P[0], lit, frac).any(1)
        B = resel(P[1], lit, frac).any(1)
        out[f'sparser x{frac}'] = analyse(A, B, B, N, piA)            # no other-voice push saved: voice = ear
        out[f'sparser x{frac}']['voice_note'] = 'voice columns repeat the ear-noise test (no push saved for D1)'
    # per-tick codes instead of the union (tick 4 alone)
    out['tick 4 only'] = analyse(base[0][:, 0], base[1][:, 0], base[2][:, 0], N, piA)
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
