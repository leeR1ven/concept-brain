"""Lemma 1 (fixed layers): how does the overlap of two binary codes change through one layer made by the generator?
Each unit takes K inputs with weights U(0.5, 1.5) and fires above its own threshold, set so that it fires at rate
rho_out on its own input (ARCH 二.4); the last layer instead lights the k most driven units of the layer (one threshold
for all). Inputs i.i.d. with density rho_in; two codes x, x' with P(x'=1 | x=1) = q (their overlap; sizes equal).
For one unit everything is exact: its 2^K input subsets, their drives, its threshold, and the 4^K joint states of
(x, x') on its inputs. Averaged over many units (weights drawn). numpy only.
Overlap of two codes = shared / sqrt(size x size) = P(both fire) / rate."""
import itertools
import json
import sys

import numpy as np

K = 6
SUB = np.array(list(itertools.product([0, 1], repeat=K)), dtype=np.int8)          # [64, K] input subsets


def joint_table(rho, q):
    """probabilities of the 4^K joint input states, as (index of x subset, index of x' subset, probability)"""
    b = rho * (1 - q) / (1 - rho)                     # P(x'=1 | x=0), keeps the density of x'
    p = {(1, 1): rho * q, (1, 0): rho * (1 - q), (0, 1): (1 - rho) * b, (0, 0): (1 - rho) * (1 - b)}
    st = np.array(list(itertools.product([0, 1], repeat=2 * K)), dtype=np.int8)   # [4096, 2K]
    x, y = st[:, :K], st[:, K:]
    pr = np.ones(len(st))
    for k in range(K):
        for (u, v), pv in p.items():
            pr = np.where((x[:, k] == u) & (y[:, k] == v), pr * pv, pr)
    w2 = 1 << np.arange(K)[::-1]
    return x @ w2, y @ w2, pr


def own_threshold_fire(drive, prob, rate):
    """which subsets fire, with the unit's own threshold: fires on about `rate` of its input (code's tie rule)"""
    order = np.argsort(drive)
    d, pv = drive[order], prob[order]
    tail = 1.0 - np.cumsum(pv)                        # P(drive > d_i) for sorted i
    fire = np.zeros(len(drive), bool)
    i = int(np.argmin(np.abs(tail - rate)))           # threshold at d_i: fire if drive > d_i
    fire[order[i + 1:]] = True
    return fire


def layer_map(rho_in, rho_out, q_in, units=4000, global_k=False, seed=0):
    rng = np.random.default_rng(seed)
    W = rng.uniform(0.5, 1.5, (units, K))
    D = W @ SUB.T                                     # [units, 64] drives
    psub = rho_in ** SUB.sum(1) * (1 - rho_in) ** (K - SUB.sum(1))
    if global_k:                                      # one threshold for the whole layer (the k most driven)
        o = np.argsort(D.ravel())
        flat = D.ravel()[o]
        tail = 1.0 - np.cumsum(np.tile(psub, units)[o]) / units          # P(drive > flat[i]) over the layer
        F = D > flat[int(np.argmin(np.abs(tail - rho_out)))]
    else:
        F = np.stack([own_threshold_fire(D[u], psub, rho_out) for u in range(units)])
    ix, iy, pr = joint_table(rho_in, q_in)
    rate = (F * psub).sum(1).mean()
    both = (F[:, ix] & F[:, iy]) @ pr
    return float(rate), float(both.mean() / max(rate, 1e-12))


def main():
    """argv: optional JSON {"dens": {...}, "inputs": {...}}: densities per layer and the input overlaps per kind"""
    arg = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}
    dens = arg.get('dens', {'input': 0.0735, 'L2': 0.2543, 'L3': 0.4495, 'L4': 0.0438})   # newborn s=4, merged
    inputs = arg.get('inputs', {'noise': 0.672, 'voice': 0.401, 'context': 0.524, 'other': 0.253})
    rows = {}
    for kind, q0 in inputs.items():
        q, r = q0, dens['input']
        out = {'input': round(q, 3)}
        for nm, glob in (('L2', False), ('L3', False), ('L4', True)):
            r_out, q = layer_map(r, dens[nm], q, global_k=glob)
            out[nm] = round(q, 3)
            r = dens[nm]
        rows[kind] = out
    print(json.dumps(rows))


if __name__ == '__main__':
    main()
