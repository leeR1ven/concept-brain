"""Evaluate the readout learned by counting (core/theory2.py saves the codes), with controls. numpy only.
Variants: codes = the network's (birth) or random ones of the same size; links = the network's (within a radius) or
all; g = sqrt (the rule's slowing growth) or lin (plain counts); score = mean push over a syllable's copies, or over
only the copies no other syllable lights more often (its own copies).
usage: python3 theory/readout_exact.py <codes.npz> '{"codes": "net|random", "links": "net|all", "g": "sqrt|lin",
        "own": 0|1}'"""
import json
import sys

import numpy as np


def run(z, codes='net', links='net', g='sqrt', own=0, seed=0, cov=0, sparse=0.0, inhib=0.0, Nlearn=None,
        delta=0, lam=1e-3):
    Cp, Cd, Cd2, N = z['cp'].astype(np.float32), z['cd'].astype(np.float32), z['cd2'].astype(np.float32), z['N']
    S, n4 = Cd.shape
    n1 = Cp.shape[1]
    pi = (N.sum(1) / N.sum()).astype(np.float32)
    if sparse or inhib:                                      # codes from the last layer's push: fewer lit per tick,
        P1, P2 = z['push'].astype(np.float32), z['push2'].astype(np.float32)   # [S, 3, n4]
        k = int(round((sparse or float(Cd.sum(1).mean()) / 3 / n4) * n4))

        def topk(P):                                          # exactly k per tick (ties broken arbitrarily)
            idx = np.argpartition(-P, k - 1, axis=-1)[..., :k]
            out = np.zeros(P.shape, bool)
            np.put_along_axis(out, idx, True, axis=-1)
            return out

        if inhib:                                            # and inhibition among units often lit together
            A0 = topk(P1).any(1).astype(np.float32)          # (anti-Hebbian lateral links ~ their correlation)
            m = pi @ A0
            Cv = (A0 - m).T @ ((A0 - m) * pi[:, None])
            sd = np.sqrt(np.maximum(np.diag(Cv), 1e-9))
            Cv = np.maximum(Cv / sd[:, None] / sd[None, :], 0).astype(np.float32)
            np.fill_diagonal(Cv, 0.0)
            for _ in range(2):
                for P in (P1, P2):
                    lat = (topk(P).reshape(-1, n4).astype(np.float32) @ Cv).reshape(P.shape)
                    P -= inhib * lat / float(k)
        Cd, Cd2 = topk(P1).any(1).astype(np.float32), topk(P2).any(1).astype(np.float32)
    if codes == 'random':                                    # same sizes, independent random units; test = train
        rng = np.random.default_rng(seed)
        k = int(Cd.sum(1).mean())
        Cd = np.zeros_like(Cd)
        for i in range(S):
            Cd[i, rng.choice(n4, k, replace=False)] = 1
        Cd2 = Cd.copy()
    if links == 'all':
        R = np.ones((n4, n1), np.float32)
    elif str(links).startswith('frac:'):                     # each link present with probability f
        R = (np.random.default_rng(seed + 1).random((n4, n1)) < float(links[5:])).astype(np.float32)
    else:
        R = np.zeros((n4, int(z['n_post']) + 1), np.float32)
        np.put_along_axis(R, z['idx'], 1.0, axis=1)
        R = R[:, :-1]
    pre = Cd - (pi @ Cd)[None] if cov else Cd                # covariance rule: only above a unit's usual activity
    post = Cp - (pi @ Cp)[None] if cov else Cp
    NL = N if Nlearn is None else Nlearn                     # what was heard so far (the target stays the corpus')
    if delta:                                                # the prediction-error rule, converged: least squares
        # minimise sum_A,B N(A,B) |copies(B) - W code(A)|^2 + lam |W|^2  (all links). With pi_A = sum_B N(A,B) and
        # H = codes [S, n4]: W^T = (H^T Pi H + lam)^-1 H^T Pi Ybar = H^T Pi (H H^T Pi + lam)^-1 Ybar  (push-through),
        # Ybar_A = sum_B N(A,B) copies(B) / pi_A
        piA = NL.sum(1).astype(np.float64)
        Yb = (NL @ Cp.astype(np.float64)) / np.maximum(piA, 1)[:, None]
        H = Cd.astype(np.float64)
        K = H @ H.T
        lam_ = lam * np.trace(K) / len(K)
        A = (K * piA[None, :]) + lam_ * np.eye(len(K))
        Wt = H.T @ (piA[:, None] * np.linalg.solve(A, Yb))      # [n4, n1]
        W = Wt.astype(np.float32)
    else:
        W = pre.T @ (NL.astype(np.float32) @ post)
    if not delta:
        W = (np.sign(W) * (np.sqrt(1 + 2 * float(z['eta']) * np.abs(W)) - 1)) if g == 'sqrt' else W
    W *= R
    Dr = Cd2 @ W
    E = Dr - pi @ Dr
    P = Cp
    if own:                                                  # copies most typical of each syllable
        share = Cp / np.maximum(Cp.sum(0, keepdims=True), 1)
        P = Cp * (share >= 0.5 * share.max(0, keepdims=True))
    score = (E @ P.T) / np.maximum(P.sum(1), 1)
    best = N.argmax(1)
    rank = (score > score[np.arange(S), best][:, None]).sum(1)
    w = N.sum(1) / N.sum()
    G = Cd2 @ Cd.T / max(1.0, float(Cd.sum(1).mean()))
    return dict(codes=codes, links=links, g=g, cov=cov, delta=delta, lam=lam, sparse=sparse, inhib=inhib, size=round(float(Cd.sum(1).mean()), 1),
                same=round(float(np.diag(G).mean()), 3),
                other=round(float((G.sum() - np.trace(G)) / (S * S - S)), 3), top1=round(float((w * (rank == 0)).sum()), 3),
                top5=round(float((w * (rank < 5)).sum()), 3), mean_rank=round(float((w * rank).sum()), 1))


def main():
    z = np.load(sys.argv[1])
    for v in ([json.loads(sys.argv[2])] if len(sys.argv) > 2 else
              [dict(links='all', delta=1, lam=x) for x in (1e-1, 1e-2, 1e-3, 1e-4)] +
              [dict(links='all', delta=1, lam=1e-2, codes='random')]):
        print(json.dumps(run(z, **v)), flush=True)


if __name__ == '__main__':
    main()
