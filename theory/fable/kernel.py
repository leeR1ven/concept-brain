"""The Hebbian readout as a kernel (Theorem R1 in docs/THEORY_fable.md), computed on the recorded codes. numpy only.
With all links, counting, and the score averaged over copies, the push on syllable X after cue A is
    s(X | A) = sum_C K(A, C) N(C, X),     K(A, C) = |code_test(A) & code_train(C)|,
the bigram counts smoothed by the last-layer code overlap. What this script measures:
  - K's diagonal (consistency) and off-diagonal (overlap), with chance;
  - the smearing ratio  sum_{C != A} K(A,C) N(C,.) / (K(A,A) N(A,.))  (how much other contexts weigh in);
  - top-1 of the smoothed bigram (count), of the centred one (cov) and of kernel ridge regression (ls = the
    prediction-error rule converged), for the test hearing and the other voice; and the same for random codes of
    the same size with the same consistency (a clean control).
usage: python3 kernel.py codes.npz"""
import json
import sys

import numpy as np

from readout import load


def ov(A, B):
    """[S, n] x [S, n] -> normalised overlap matrix, shared / sqrt(sizes)"""
    A, B = A.astype(np.float32), B.astype(np.float32)
    return (A @ B.T) / np.sqrt(np.outer(A.sum(1), B.sum(1)).clip(1))


def codes(z, h, ticks=(4, 5, 6)):
    return np.stack([z['l4'][h][:, t + 1] for t in ticks]).any(0)


def top1(score, N, piA):
    best = N.argmax(1)
    pred = score.argmax(1)
    return round(float((piA * (pred == best)).sum()), 3)


def predictors(K_test, K_train, N, piA, lam=1e-2):
    """scores [cue, X] for the three rules, in kernel form (all links, mean over copies)"""
    out = {}
    out['count'] = K_test @ N
    Kc = K_test - (piA @ K_train)[None]                     # the cue's usual level taken out (per unit -> per code)
    out['cov'] = Kc @ (N - (N.sum(1)[:, None] * (N.sum(0) / N.sum())[None]))   # post centred too
    piN = N.sum(1)
    Yb = N / np.maximum(piN, 1)[:, None]
    A = K_train * piN[None, :] + lam * np.trace(K_train) / len(K_train) * np.eye(len(K_train))
    out['ls'] = K_test @ (piN[:, None] * np.linalg.solve(A, Yb))
    return out


def main():
    z = load(sys.argv[1])
    N = z['N']
    S = N.shape[0]
    piA = N.sum(1) / N.sum()
    rep = {}
    Ctr = codes(z, 0)
    for h, name in ((0, 'train (same hearing)'), (1, 'ear noise'), (2, 'voice D1')):
        Cte = codes(z, h)
        K = ov(Cte, Ctr)
        dens = float(Ctr.mean())
        diag = np.diag(K)
        off = (K.sum(1) - diag) / (S - 1)
        Kraw = Cte.astype(np.float32) @ Ctr.astype(np.float32).T
        smear = ((Kraw * N.sum(1)[None]).sum(1) - np.diag(Kraw) * N.sum(1)) / np.maximum(np.diag(Kraw) * N.sum(1), 1e-9)
        Ktr = ov(Ctr, Ctr)
        pr = predictors(K, Ktr, N, piA)
        rep[name] = dict(consistency=round(float(diag.mean()), 3), overlap_other=round(float(off.mean()), 3),
                         chance=round(dens, 4), code_size=round(float(Ctr.sum(1).mean()), 1),
                         smear_ratio_median=round(float(np.median(smear)), 1),
                         **{f'top1_{k}': top1(v, N, piA) for k, v in pr.items()})
    # control: random codes of the same size, test code = train code with the measured consistency
    rng = np.random.default_rng(0)
    n4, k = z['n1'], int(Ctr.sum(1).mean())
    c = rep['ear noise']['consistency']
    Rtr = np.zeros((S, n4), bool)
    Rte = np.zeros((S, n4), bool)
    for i in range(S):
        u = rng.choice(n4, k, replace=False)
        Rtr[i, u] = True
        keep = rng.random(k) < c
        Rte[i, u[keep]] = True
        Rte[i, rng.choice(n4, k - int(keep.sum()), replace=False)] = True
    K, Ktr = ov(Rte, Rtr), ov(Rtr, Rtr)
    pr = predictors(K, Ktr, N, piA)
    rep['random codes, same size and consistency'] = dict(
        consistency=round(float(np.diag(K).mean()), 3), overlap_other=round(float(((K.sum(1) - np.diag(K)) / (S - 1)).mean()), 3),
        chance=round(k / n4, 4), **{f'top1_{k_}': top1(v, N, piA) for k_, v in pr.items()})
    rep['ceiling_bigram_top1_on_itself'] = 1.0
    rep['unigram_top1'] = round(float((piA * (N.sum(0).argmax() == N.argmax(1))).sum()), 3)
    print(json.dumps(rep, indent=1))


if __name__ == '__main__':
    main()
