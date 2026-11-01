"""Neuro review toy 1b (local, < 300 MB, ~2 min): age-resolved recall of the design's CA3 (N 1e4, fan-in 1,000,
k 100, one-shot) with a palimpsest rule strong enough to keep the weight matrix at a steady fill (~0.1):
LTD p = 0.09 on (pre inactive, post active) synapses at each write (Amit & Fusi 1994 style bounded binary synapses).
Also: cued replay (cue = 30 % of the LAST stored episode, as awake replay starts from the current state) vs a
spontaneous random kick. 5,000 episodes stored (about half a bench life of event offsets).
Run: python3 -I ca3_palimp.py
"""
import numpy as np

N, C, k, P, LTD = 10000, 1000, 100, 5000, 0.09
rng = np.random.default_rng(1)
idx = np.stack([rng.choice(N, C, replace=False) for _ in range(N)]).astype(np.int16)
W = np.zeros((N, C), dtype=bool)
pats = []


def kwta(h):
    th = np.partition(h, N - k)[N - k]
    on = h >= th
    if on.sum() > k:
        keep = rng.choice(np.flatnonzero(on), k, replace=False); on = np.zeros(N, bool); on[keep] = True
    return on


def drive(x):
    return (W & x[idx]).sum(1).astype(np.float32) + 1e-3 * rng.random(N)


for p in range(P):
    a = np.zeros(N, bool); a[rng.choice(N, k, replace=False)] = True
    pats.append(np.flatnonzero(a)); post = pats[-1]
    pre_on = a[idx[post]]
    W[post] &= ~((~pre_on) & (rng.random(pre_on.shape) < LTD))
    W[post] |= pre_on
print(f'fill {W.mean():.3f} after {P} episodes')
for age in (10, 100, 300, 600, 1000, 2000, 4000):
    ov = []
    for _ in range(30):
        j = P - 1 - age + int(rng.integers(-5, 6)); tgt = pats[j]
        x = np.zeros(N, bool); x[rng.choice(tgt, k // 2, replace=False)] = True
        for _ in range(3):
            x = kwta(drive(x))
        ov.append(x[tgt].mean())
    print(f'  age {age:5d}: recall overlap from a 50 % cue {np.mean(ov):.2f}  (>= 0.9 on {np.mean(np.array(ov) >= 0.9):.2f})')
for cf in (0.2, 0.3):
    ov = []
    for _ in range(30):
        tgt = pats[P - 1 - 100 + int(rng.integers(-5, 6))]
        x = np.zeros(N, bool); x[rng.choice(tgt, int(cf * k), replace=False)] = True
        for _ in range(3):
            x = kwta(drive(x))
        ov.append(x[tgt].mean())
    print(f'  age 100, cue {int(cf * 100)} % (theta_rec 0.2 accepts such cues): recall overlap {np.mean(ov):.2f}')
Parr = np.stack(pats).astype(np.int32)
for name, frac in (('random kick', 0.0), ('cue 30 % of the last episode', 0.3)):
    best_ok, which = 0, []
    for _ in range(30):
        x = np.zeros(N, bool)
        if frac == 0:
            x[rng.choice(N, k, replace=False)] = True
        else:
            x[rng.choice(pats[-1], int(frac * k), replace=False)] = True
        for _ in range(5):
            x = kwta(drive(x))
        ovs = x[Parr].sum(1) / k
        best_ok += ovs.max() >= 0.9; which.append(int(np.argmax(ovs)))
    print(f'  replay from {name}: clean (>= 0.9 with a stored episode) {best_ok / 30:.2f}; settled on the last episode {np.mean(np.array(which) == P - 1):.2f}')
