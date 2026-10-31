# -*- coding: utf-8 -*-
"""correctness + speed of the torch/CUDA random layer against the numpy one (same seed, same contexts).
usage: python3 test_torch.py [n=300]"""
import os, sys, time, json
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
sys.path.insert(1, os.path.join(HERE, '..', 'synth'))   # repo: the unchanged ARCH_v2 core and its bench_copy
import numpy as np
import torch
from randlayer import RandMix
from randtorch import TorchRandMix
from rules import pair_keys
from randlayer import PAIRS
A = dict(a.split('=', 1) for a in sys.argv[1:])
n = int(A.get('n', 300))
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
if dev == 'cuda':
    torch.cuda.set_per_process_memory_fraction(0.05)
rng = np.random.default_rng(0)


def ctx():
    P = list(rng.choice(60, int(rng.integers(2, 7)), replace=False))
    g = {'P': P, 'L': [int(rng.integers(-1, 60))], 'N': [int(rng.integers(0, 7))], 'FAM': [int(rng.integers(0, 4))],
         'P1': [P[0]], 'PL': [P[-1]], 'CP': [int(rng.choice([-2, P[0]]))], 'CN': [-2], 'CC': [int(rng.integers(-2, 60))],
         'F': [('r', int(b)) for b in rng.choice(240, 32, replace=False)], 'A': [int(rng.integers(-2, 3))],
         'VR': [P[0]] if rng.random() < 0.5 else [], 'M': [int(rng.integers(0, 3))]}
    if not g['VR']:
        g.pop('VR')
    return pair_keys(g, PAIRS)


out = dict(device=dev, torch=torch.__version__, numpy=np.__version__, cases={})
for scale, alpha, train in ((0.25, 0.01, 0), (0.25, 0.01, 1), (1.0, 0.01, 0), (4.0, 0.01, 0), (16.0, 0.0025, 0)):
    kw = dict(scale=scale, alpha=alpha, train=train, code='rand', every=50)
    a, b = RandMix(**kw), TorchRandMix(device=dev, **kw)
    C = [ctx() for _ in range(n)]
    eq, jac, ta, tb = 0, [], 0.0, 0.0
    for i, k in enumerate(C):
        a.cache = {}; b.cache = {}
        t0 = time.time(); x = a.units(k); ta += time.time() - t0
        t0 = time.time(); y = b.units(k); tb += time.time() - t0
        if train:
            a.learn_event(); b.learn_event()
        eq += int(np.array_equal(x, y))
        jac.append(len(np.intersect1d(x, y)) / max(len(np.union1d(x, y)), 1))
    if dev == 'cuda':
        torch.cuda.synchronize()
    r = dict(N=a.N, active=round(a.stats['active_sum'] / a.stats['calls'], 1), identical=round(eq / n, 4),
             jaccard_min=round(min(jac), 4), numpy_ms=round(ta / n * 1000, 3), torch_ms=round(tb / n * 1000, 3),
             w_maxdiff=float(np.abs(a.w - b.w).max()), hebb_batches=a.stats['hebb_batches'])
    out['cases'][f'scale{scale}_alpha{alpha}_train{train}'] = r
    print(json.dumps(r), flush=True)
    del a, b
json.dump(out, open(os.path.join(HERE, 'out', 'test_torch.json'), 'w'), indent=1)
print(json.dumps(out))
