"""Many hearings per syllable (codes_multi.py): does averaging over within-class noise rescue the readout? numpy only.
For n training hearings per syllable (1 = one hearing of one voice, as in kernel.py; 3 = one per voice; 6; 18 = all):
  identification  nearest class mean (cosine), and the same after dividing each unit by its pooled within-class
                  std (an LDA-like diagonal whitening); chance 1/S
  readout         next-syllable top-1 of the counting rule (kernel-smoothed bigram over all hearings), the centred
                  (covariance) rule, and kernel ridge regression (the prediction-error rule converged), fit on all
                  hearings, scored on held-out hearings (fresh ones of the training voices / the fourth voice)
  directions      between-class directions (SVD of the centred class means) whose between-class variance exceeds the
                  within-class variance along them
usage: python3 multi.py multi.npz [layer=1 (0 = copies, 1 = last layer)]"""
import json
import sys
import time

import numpy as np


def unpack(a, n):
    return np.unpackbits(a, axis=-1)[..., :n].astype(np.float32)


def cos_sim(A, B):
    An = A / np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-9)
    Bn = B / np.maximum(np.linalg.norm(B, axis=1, keepdims=True), 1e-9)
    return An @ Bn.T


def top1(score, N, piA):
    best = N.argmax(1)
    return round(float((piA * (score.argmax(1) == best)).sum()), 3)


def main():
    t0 = time.time()
    z = np.load(sys.argv[1])
    layer = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    lams = json.loads(sys.argv[3]) if len(sys.argv) > 3 else [1e-2]       # ridge values to scan for the LS readout
    only = sys.argv[4] if len(sys.argv) > 4 else None                        # run only the plans whose name starts so
    n1 = int(z['n1'])
    N = z['N']
    S = N.shape[0]
    piA = N.sum(1) / N.sum()
    tr = unpack(z['train'][:, :, :, layer], n1)            # [S, V, R, n]
    te = unpack(z['test'][:, :, :, layer], n1)             # [S, V, Rt, n]
    vo = unpack(z['voice'][:, :, layer], n1)               # [S, Rv, n]
    V, R = tr.shape[1], tr.shape[2]
    tests = {'fresh hearings, training voices': te.reshape(S, -1, n1), 'fourth voice': vo}
    out = dict(S=S, layer=['copies', 'last layer'][layer], code_size=round(float(tr[0, 0, 0].sum()), 1),
               corpus_covered=round(float(z['covered']), 3), chance_id=round(1.0 / S, 4),
               unigram_top1=round(float((piA * (N.sum(0).argmax() == N.argmax(1))).sum()), 3))
    plans = {'1 (one voice)': [(0, 0)], '3 (one per voice)': [(v, 0) for v in range(V)]}
    if R >= 2:
        plans['6 (two per voice)'] = [(v, k) for v in range(V) for k in range(2)]
    if R > 2:
        plans[f'{V * R} (all)'] = [(v, k) for v in range(V) for k in range(R)]
    for name, sel in plans.items():
        if only and not name.startswith(only):
            continue
        X = np.stack([tr[:, v, k] for v, k in sel], 1)        # [S, n_ex, n]
        n_ex = X.shape[1]
        mu = X.mean(1)                                        # class means [S, n]
        dev = X - mu[:, None]
        wstd = np.sqrt((dev ** 2).sum((0, 1)) / max(1, S * (n_ex - 1))) if n_ex > 1 else np.ones(n1, np.float32)
        wstd = wstd + 0.1 * wstd.mean() + 1e-6
        rep = dict(n_ex=n_ex)
        # usable directions: between-class scatter directions vs within-class variance along them
        if n_ex > 1:
            Mc = mu - mu.mean(0)
            U, sv, Vt = np.linalg.svd(Mc, full_matrices=False)
            between = sv ** 2 / (S - 1)
            proj = dev.reshape(-1, n1) @ Vt.T                 # within-class deviations along each direction
            within = (proj ** 2).sum(0) / max(1, S * (n_ex - 1))
            rep['directions_between_over_within_gt1'] = int((between > within).sum())
            rep['directions_gt3'] = int((between > 3 * within).sum())
            rep['between_over_within_top10'] = [round(float(x), 1) for x in (between / within)[:10]]
        # readout fits (kernel form over all exemplars)
        H = X.reshape(-1, n1)                                 # rows (C, e)
        rowC = np.repeat(np.arange(S), n_ex)
        Ktr = H @ H.T
        w = np.repeat(piA / n_ex, n_ex)                       # row weights
        Yb = N / np.maximum(N.sum(1), 1)[:, None]            # [S, S] follower distribution per context
        Yrows = Yb[rowC]
        alphas = {}
        for lam in lams:
            A = Ktr * (np.repeat(N.sum(1) / n_ex, n_ex))[None, :] + lam * np.trace(Ktr) / len(Ktr) * np.eye(len(Ktr))
            alphas[lam] = np.repeat(N.sum(1) / n_ex, n_ex)[:, None] * np.linalg.solve(A, Yrows)   # [rows, S]
            del A
        kbar = w @ Ktr                                        # usual level of the kernel per training row
        for tname, T in tests.items():
            Tq = T.reshape(-1, n1)
            tC = np.repeat(np.arange(S), T.shape[1])
            sim = cos_sim(Tq, mu)
            idc = (sim.argmax(1) == tC).mean()
            sim2 = cos_sim(Tq / wstd, mu / wstd)
            idw = (sim2.argmax(1) == tC).mean()
            K = Tq @ H.T                                      # [queries, rows]
            per_cue = lambda M: np.stack([M[tC == c].mean(0) for c in range(S)])
            s_count = per_cue(K @ (np.repeat(N, n_ex, 0) / n_ex))
            Kc = K - kbar[None]
            Nc = N - N.sum(1)[:, None] * (N.sum(0) / N.sum())[None]
            s_cov = per_cue(Kc @ (np.repeat(Nc, n_ex, 0) / n_ex))
            r = dict(identified_mean=round(float(idc), 3), identified_whitened=round(float(idw), 3),
                     top1_count=top1(s_count, N, piA), top1_cov=top1(s_cov, N, piA))
            best = N.argmax(1)
            for lam, alpha in alphas.items():
                P = K @ alpha
                r[f'top1_ls_lam{lam}'] = top1(per_cue(P), N, piA)
                # per hearing (one fresh hearing as the cue), the honest single-cue number
                r[f'top1_ls_per_hearing_lam{lam}'] = round(float((piA[tC] * (P.argmax(1) == best[tC])).sum() / piA[tC].sum()), 3)
            r['top1_count_per_hearing'] = round(float((piA[tC] * ((K @ (np.repeat(N, n_ex, 0) / n_ex)).argmax(1) == best[tC])).sum() / piA[tC].sum()), 3)
            rep[tname] = r
        out[name] = rep
        print(json.dumps({name: rep}), flush=True)
    out['secs'] = round(time.time() - t0)
    print('MULTI', json.dumps(out))


if __name__ == '__main__':
    main()
