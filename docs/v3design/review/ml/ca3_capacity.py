"""Does CA3 capacity grow with s under the v3 scaling rule (fixed fan-in C, fixed active fraction a)?
Willshaw binary auto-associator (one-shot clipped Hebbian), retrieval from a 50 % cue, one step, threshold = the
unit-specific 'all active inputs set' rule (the best one-step rule, Buckingham & Willshaw 1993) and k-WTA.
Capacity = largest stored M with mean F1 >= 0.9 on old (first-stored) patterns.
Run: python3 -I ca3_capacity.py   (one process, < 400 MB)
"""
import numpy as np, sys, json


def run(N, C, a, Ms, seed=0, ntest=40, cue=0.5):
    rng = np.random.default_rng(seed)
    k = max(2, int(round(a * N)))
    dense = C >= N
    if dense:
        W = np.zeros((N, N), dtype=bool)
    else:
        fan = np.empty((N, C), dtype=np.int32)
        for i in range(N):
            fan[i] = rng.choice(N, C, replace=False)
        W = np.zeros((N, C), dtype=bool)
    pats = []
    out = {}
    member = np.zeros(N, dtype=bool)
    m_done = 0
    for M in Ms:
        while m_done < M:
            P = rng.choice(N, k, replace=False); pats.append(P)
            if dense:
                W[np.ix_(P, P)] = True
            else:
                member[:] = False; member[P] = True
                W[P] |= member[fan[P]]
            m_done += 1
        f1s = []
        for t in range(min(ntest, M)):
            P = pats[t * (M // min(ntest, M))]
            c = rng.choice(P, max(1, int(cue * k)), replace=False)
            x = np.zeros(N, dtype=bool); x[c] = True
            if dense:
                h = W[:, c].sum(1); m = np.full(N, len(c))
            else:
                xa = x[fan]; h = (W & xa).sum(1); m = xa.sum(1)
            fire = (h == m) & (m > 0)
            if fire.sum() > 3 * k:                      # fall back to k-WTA on h / m when the rule floods
                score = np.where(m > 0, h / np.maximum(m, 1), 0) + 1e-6 * rng.random(N)
                fire = np.zeros(N, dtype=bool); fire[np.argsort(-score)[:k]] = True
            tp = fire[P].sum(); prec = tp / max(1, fire.sum()); rec = tp / k
            f1s.append(0 if tp == 0 else 2 * prec * rec / (prec + rec))
        out[M] = round(float(np.mean(f1s)), 3)
    return k, out


if __name__ == '__main__':
    Ms = [250, 500, 1000, 2000, 4000, 8000, 16000]
    res = {}
    print('fixed fan-in C=1000, fixed fraction a=0.01 (the v3 rule; s=1 is N=10,000):')
    for N in (2500, 5000, 10000, 20000):
        k, o = run(N, 1000, 0.01, Ms)
        cap = max([M for M in Ms if o[M] >= 0.9], default=0)
        res[f'fixedC_N{N}'] = dict(k=k, aC=0.01 * 1000, f1=o, cap=cap)
        print(f'  N={N:6d} k={k:4d} aC={0.01*1000:.0f}  F1 by M: {o}  capacity(F1>=0.9) ~ {cap}', flush=True)
    print('dense (C = N), fixed fraction a=0.01:')
    for N in (2500, 5000, 10000):
        k, o = run(N, N, 0.01, Ms)
        cap = max([M for M in Ms if o[M] >= 0.9], default=0)
        res[f'dense_N{N}'] = dict(k=k, f1=o, cap=cap)
        print(f'  N={N:6d} k={k:4d}  F1 by M: {o}  capacity ~ {cap}', flush=True)
    print('fan-in ~ sqrt(s), fraction ~ 1/sqrt(s) (aC fixed at 10):')
    for N, C, a in ((2500, 500, 0.02), (10000, 1000, 0.01), (20000, 1414, 0.00707)):
        k, o = run(N, C, a, Ms)
        cap = max([M for M in Ms if o[M] >= 0.9], default=0)
        res[f'sqrt_N{N}'] = dict(k=k, C=C, f1=o, cap=cap)
        print(f'  N={N:6d} C={C} k={k:4d}  F1 by M: {o}  capacity ~ {cap}', flush=True)
    json.dump(res, open('ca3_capacity.json', 'w'), indent=1)
