"""Neuro review toy 1c: the proposed CA3 fix point. Willshaw capacity ~ 0.1/f^2 episodes while C*f*cue >= ~5.
Design point f = 0.01 (k 100, C 1000) holds ~1,000 (ca3_capacity.out). Fix point: N 1e4, k 32 (f 0.0032), C 3,100
(c*k = 10, as at the design point) -> predicted ~10,000. 50 % cue, 3 k-WTA iterations. Run: python3 -I ca3_sparse.py"""
import numpy as np
N, C, k = 10000, 3100, 32
rng = np.random.default_rng(2)
idx = np.stack([rng.choice(N, C, replace=False) for _ in range(N)]).astype(np.int16)
W = np.zeros((N, C), dtype=bool); pats = []
def kwta(h):
    th = np.partition(h, N - k)[N - k]; on = h >= th
    if on.sum() > k:
        keep = rng.choice(np.flatnonzero(on), k, replace=False); on = np.zeros(N, bool); on[keep] = True
    return on
def drive(x):
    return (W & x[idx]).sum(1).astype(np.float32) + 1e-3 * rng.random(N)
for p in range(1, 15001):
    a = np.zeros(N, bool); a[rng.choice(N, k, replace=False)] = True
    pats.append(np.flatnonzero(a)); W[pats[-1]] |= a[idx[pats[-1]]]
    if p in (2500, 5000, 10000, 15000):
        ov = []
        for j in rng.integers(0, p, 30):
            tgt = pats[j]; x = np.zeros(N, bool); x[rng.choice(tgt, k // 2, replace=False)] = True
            for _ in range(3):
                x = kwta(drive(x))
            ov.append(x[tgt].mean())
        print(f'  P={p:6d}: fill {W.mean():.3f}, recall overlap from 50 % cue {np.mean(ov):.2f}, synapses {N*C/1e6:.0f} M', flush=True)
