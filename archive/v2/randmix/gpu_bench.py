# -*- coding: utf-8 -*-
"""GPU form of the random layer for many agents at once (the port's inner loop), timed on CUDA.
For B agents (each its own random layer: fan-in table (B, N, 2, f), weights) and one event per agent:
  activity: x (B, D+1) 0/1 -> s = (x[b, lines] * w).sum(-1) -> eligibility -> per-population k-WTA by one batched sort
            (units stored population by population; rank = sorted position - population offset) -> mask (B, N)
  readout:  drive = sum of GEN rows of the active units (top-K gather, K = sum of k_t, padded)
  learning: NLMS step on the active rows and the instar step on the active units' synapses (indexed add)
Populations, sizes and k_t are randlayer.RandMix's (MIX_COUNT allocation, scale 1 / 4 / 16).  Inputs are random
contexts with the bench's statistics (5 held items x 16 code lines, 32 eye lines, single-valued groups).
usage: python3 gpu_bench.py [reps=50]"""
import os, sys, time, json
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(1, os.path.join(HERE, '..', 'synth'))   # repo: the unchanged ARCH_v2 core and its bench_copy
import numpy as np
import torch
from randlayer import RandMix

A = dict(a.split('=', 1) for a in sys.argv[1:])
reps = int(A.get('reps', 50))
dev = 'cuda'
torch.cuda.set_per_process_memory_fraction(float(A.get('memfrac', 0.25)))
V1 = 257
out = dict(gpu=torch.cuda.get_device_name(0), torch=torch.__version__, rows=[])
rng = np.random.default_rng(0)


def contexts(m, B):
    X = torch.zeros(B, m.D + 1, device=dev)
    for b in range(B):
        lines = []
        P = rng.choice(60, 5, replace=False)
        for i in P:
            lines += list(m.boff['P'] + m._item_code(int(i)))
        for g in ('L', 'P1', 'PL', 'CP', 'CN', 'CC', 'VR'):
            lines += list(m.boff[g] + m._item_code(int(rng.integers(0, 60))))
        for g, vals in (('N', 7), ('FAM', 4), ('A', 4), ('M', 3)):
            o = m.boff[g] + int(rng.integers(vals)) * m.k_s
            lines += list(range(o, o + m.k_s))
        lines += list(m.boff['F'] + rng.choice(240, 32, replace=False))
        X[b, torch.tensor(lines, device=dev)] = 1.0
    return X


for scale, alpha in ((1.0, 0.01), (4.0, 0.01), (16.0, 0.0025), (16.0, 0.01)):
    m = RandMix(scale=scale, alpha=alpha, code='rand')
    N = m.N
    for B in (1, 16, 64):
        if B * N * 6 > 1.2e8:          # keep the per-job GPU memory small
            continue
        lines = torch.from_numpy(m.lines.astype(np.int64)).to(dev).unsqueeze(0).expand(B, -1, -1, -1).contiguous()
        w = torch.ones(B, N, 2, m.fan, device=dev)
        w[:, torch.from_numpy(m.direct).to(dev), 1, :] = 0
        typ = torch.from_numpy(m.type_of.astype(np.int64)).to(dev)
        direct = torch.from_numpy(m.direct).to(dev)
        tie = torch.from_numpy(m.tie.astype(np.float64)).to(dev)
        kt = torch.from_numpy(m.kt.astype(np.int64)).to(dev)
        toff = torch.from_numpy(m.toff.astype(np.int64)).to(dev)
        ar = torch.arange(N, device=dev).unsqueeze(0).expand(B, -1).contiguous()
        gen = torch.zeros(B, N, V1, device=dev, dtype=torch.float16) if B * N * V1 * 2 < 5e9 else None
        X = contexts(m, B)
        lines_c = torch.where(lines >= 0, lines, torch.full_like(lines, m.D))

        def act():
            s = (torch.gather(X, 1, lines_c.view(B, -1)).view(B, N, 2, m.fan) * w).sum(-1)
            el = (s[..., 0] >= 1) & ((s[..., 1] >= 1) | direct)
            sc = (s[..., 0] + s[..., 1]).double()
            key = typ.double() * 1000.0 + torch.where(el, 100.0 - sc - tie, torch.full_like(sc, 500.0))
            o = torch.argsort(key, dim=1)
            pos = torch.empty_like(o)
            pos.scatter_(1, o, ar)
            return el & (pos - toff[typ] < kt[typ])

        K = int(m.kt.sum())                                                # upper bound of the active count
        bidx = torch.arange(B, device=dev).unsqueeze(1)

        thr = torch.full((len(m.Nt),), 3.0, device=dev, dtype=torch.float64)

        def act_thr():                                                     # GPU-friendly alternative: a per-population
            s = (torch.gather(X, 1, lines_c.view(B, -1)).view(B, N, 2, m.fan) * w).sum(-1)   # threshold (homeostatic
            el = (s[..., 0] >= 1) & ((s[..., 1] >= 1) | direct)          # in a real run), no sort: elementwise only
            return el & ((s[..., 0] + s[..., 1]).double() + tie >= thr[typ])

        def readout(mask):
            if gen is None:
                return None
            val, idx = torch.topk(mask.half(), K, dim=1)                    # the active units, padded to K
            rows = gen[bidx, idx]                                          # (B, K, V+1) gather of the active rows
            return (rows.float() * val.float().unsqueeze(-1)).sum(1), idx, val

        def learn(mask, r):
            if gen is None:
                return
            d, idx, val = r
            err = torch.zeros(B, V1, device=dev); err[:, 3] = 1.0
            err = err - d.clamp(0, 1)
            n = val.float().sum(1, keepdim=True).clamp(min=1)
            step = (0.35 / n).unsqueeze(-1) * err.unsqueeze(1) * val.float().unsqueeze(-1)   # (B, K, V+1)
            gen[bidx, idx] += step.half()                                  # NLMS on the active rows only
            li = lines_c[bidx, idx]                                        # (B, K, 2, f) synapses of the active units
            xs = torch.gather(X, 1, li.view(B, -1)).view(B, K, 2, m.fan)
            wa = w[bidx, idx]
            w[bidx, idx] = wa + 0.02 * val.float()[..., None, None] * (xs - wa)   # instar on the active units

        for _ in range(3):
            mk = act(); dd = readout(mk); learn(mk, dd)
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(reps):
            mk = act()
        torch.cuda.synchronize(); ta = (time.time() - t0) / reps
        t0 = time.time()
        for _ in range(reps):
            mt = act_thr()
        torch.cuda.synchronize(); tt = (time.time() - t0) / reps
        t0 = time.time()
        for _ in range(reps):
            mk = act(); dd = readout(mk); learn(mk, dd)
        torch.cuda.synchronize(); tf = (time.time() - t0) / reps
        r = dict(scale=scale, alpha=alpha, N=N, B=B, active=float(mk.sum(1).float().mean()), act_ms=round(ta * 1000, 3), act_thr_ms=round(tt * 1000, 3),
                 act_readout_learn_ms=round(tf * 1000, 3) if gen is not None else None,
                 agent_events_per_s=round(B / tf) if gen is not None else round(B / ta),
                 gpu_mem_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2))
        out['rows'].append(r)
        print(json.dumps(r), flush=True)
        del lines, w, gen, X, lines_c
        torch.cuda.empty_cache()
os.makedirs(os.path.join(HERE, 'out'), exist_ok=True)
json.dump(out, open(os.path.join(HERE, 'out', 'gpu_bench.json'), 'w'), indent=1)
print(json.dumps(out))
