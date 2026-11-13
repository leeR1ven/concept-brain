"""Neuron morphology generator shared by every episode.

Builds one organic surface for a neuron: a soma with branching, tapering dendrites,
smoothly blended through a signed distance field and extracted with marching cubes.
Output is a cached .npz with verts, faces and `d` (path length from the soma, used by
the ignition wave), so a scene never has to regenerate it.

usage: python3 morph.py <outdir> <name> <seed> [--avoid x,y,z] [--h 0.012] [--extent 2.1]
"""
import argparse
import time

import numpy as np
from scipy.spatial import cKDTree
from skimage.measure import marching_cubes


def unit(v):
    return v / (np.linalg.norm(v) + 1e-12)


def fib_dirs(n, rng):
    """Evenly spread directions with jitter, so dendrites leave the soma all round."""
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    th = np.pi * (1 + 5**0.5) * i + rng.uniform(0, 6.28)
    d = np.stack([np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)], 1)
    return d + rng.normal(0, 0.16, d.shape)


class Tree:
    def __init__(self, rng, extent):
        self.rng, self.extent, self.segs = rng, extent, []  # (p0, r0, p1, r1, path)

    def branch(self, p, d, r, path, depth, budget):
        rng, step = self.rng, 0.055
        curl = unit(rng.normal(size=3))
        lump = rng.uniform(0, 6.28)
        s, next_fork = 0.0, rng.uniform(0.42, 0.75) * (0.8 ** depth)
        while s < budget and r > 0.0145:
            curl = unit(curl + rng.normal(0, 0.34, 3))
            d = unit(d + curl * 0.085 + rng.normal(0, 0.035, 3))
            q = p + d * step
            if np.linalg.norm(q) > self.extent:  # keep inside the grid, bending back
                d = unit(d - unit(q) * 0.5)
                q = p + d * step
            # slight beading along the branch, as real dendrites have
            bead = 1 + 0.10 * np.sin(s * 15 + lump) * np.sin(s * 6.1 + lump * 2)
            r2 = r * (1 - 0.021 * step / 0.055)
            self.segs.append((p, r * bead, q, r2 * bead, path + s))
            p, r, s = q, r2, s + step
            if s > next_fork and depth < 4 and r > 0.024:
                axis = unit(np.cross(d, rng.normal(size=3)))
                ang = rng.uniform(0.34, 0.68)
                for sign, frac in ((1, rng.uniform(0.66, 0.8)), (-1, rng.uniform(0.66, 0.8))):
                    nd = unit(d * np.cos(ang) + sign * axis * np.sin(ang))
                    self.branch(p, nd, r * frac, path + s, depth + 1, (budget - s) * rng.uniform(0.72, 1.0))
                return


def capsule_sdf(px, a, ra, b, rb):
    ab = b - a
    t = np.clip(((px - a) @ ab) / (ab @ ab + 1e-12), 0, 1)
    return np.linalg.norm(px - (a + t[..., None] * ab), axis=-1) - (ra + (rb - ra) * t)


def smin(a, b, k):
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0, 1)
    return b * (1 - h) + a * h - k * h * (1 - h)


def build(seed, avoid=(), h=0.012, extent=2.1, primaries=7, soma_r=0.46, cone=0.55):
    rng = np.random.default_rng(seed)
    t0 = time.time()
    tree = Tree(rng, extent)
    avoid = [unit(np.array(a, float)) for a in avoid]
    made = 0
    for d in fib_dirs(primaries + 5, rng):
        d = unit(d)
        if made >= primaries or any(d @ a > cone for a in avoid):
            continue
        made += 1
        r0 = rng.uniform(0.098, 0.145)
        # the trunk flares into the soma, like a real proximal dendrite
        tree.segs.append((d * soma_r * 0.72, r0 * 1.9, d * soma_r * 1.2, r0, 0.0))
        tree.branch(d * soma_r * 1.2, d, r0, 0.0, 0, rng.uniform(1.25, 2.0))
    print(f"seed {seed}: {len(tree.segs)} segments", flush=True)

    lim = extent + 0.3
    n = int(2 * lim / h) + 1
    F = np.full((n, n, n), 1.0, np.float32)
    ax = np.linspace(-lim, lim, n)
    ids = lambda c, m: (max(int((c - m + lim) / h), 0), min(int((c + m + lim) / h) + 2, n))

    # soma: a few overlapping spheres, so it is not a perfect ball
    for c, r in [(rng.normal(0, 0.05, 3), soma_r), (rng.normal(0, 0.16, 3), soma_r * 0.8),
                 (rng.normal(0, 0.16, 3), soma_r * 0.72)]:
        m = r + 0.5
        I = [ids(c[i], m) for i in range(3)]
        g = np.stack(np.meshgrid(ax[I[0][0]:I[0][1]], ax[I[1][0]:I[1][1]], ax[I[2][0]:I[2][1]], indexing="ij"), -1)
        s = F[I[0][0]:I[0][1], I[1][0]:I[1][1], I[2][0]:I[2][1]]
        s[:] = smin(s, np.linalg.norm(g - c, axis=-1) - r, 0.22)

    # thick segments first, so thin branches blend onto something solid
    segs = sorted(tree.segs, key=lambda x: -x[1])
    for a, ra, b, rb, _ in segs:
        m = max(ra, rb) + max(0.05, ra * 1.6)
        I = [ids(min(a[i], b[i]), m)[0] for i in range(3)], [ids(max(a[i], b[i]), m)[1] for i in range(3)]
        lo, hi = I
        g = np.stack(np.meshgrid(ax[lo[0]:hi[0]], ax[lo[1]:hi[1]], ax[lo[2]:hi[2]], indexing="ij"), -1)
        s = F[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
        s[:] = smin(s, capsule_sdf(g, a, ra, b, rb), max(0.018, ra * 0.55))
    print(f"field done {time.time() - t0:.0f}s, grid {n}^3", flush=True)

    v, f, _, _ = marching_cubes(F, 0.0, spacing=(h, h, h), allow_degenerate=False)
    v = (v - lim).astype(np.float32)
    v, f = taubin(v, f.astype(np.int32))

    # path length from the soma for every vertex, for the ignition wave
    mids = np.array([(a + b) / 2 for a, _, b, _, _ in tree.segs], np.float32)
    path = np.array([p for *_, p in tree.segs], np.float32)
    _, nn = cKDTree(mids).query(v, workers=1)
    dist = path[nn]
    dist[np.linalg.norm(v, axis=1) < soma_r * 1.25] = 0.0
    print(f"mesh {len(v)} verts {len(f)} tris, {time.time() - t0:.0f}s", flush=True)
    return v, f, dist


def taubin(v, f, it=6, lam=0.5, mu=-0.53):
    """Smoothing without shrinkage, to remove the marching cubes staircase."""
    n = len(v)
    idx = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    deg = np.bincount(idx[:, 0], minlength=n).astype(np.float32)[:, None] + 1e-6
    for i in range(it * 2):
        acc = np.zeros_like(v)
        for c in range(3):
            acc[:, c] = np.bincount(idx[:, 0], weights=v[idx[:, 1], c], minlength=n)
        v = v + (lam if i % 2 == 0 else mu) * (acc / deg - v)
    return v, f


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir")
    ap.add_argument("name")
    ap.add_argument("seed", type=int)
    ap.add_argument("--avoid", action="append", default=[])
    ap.add_argument("--h", type=float, default=0.012)
    ap.add_argument("--extent", type=float, default=2.1)
    ap.add_argument("--primaries", type=int, default=7)
    a = ap.parse_args()
    avoid = [tuple(float(x) for x in s.split(",")) for s in a.avoid]
    v, f, d = build(a.seed, avoid, a.h, a.extent, a.primaries)
    np.savez(f"{a.outdir}/{a.name}.npz", v=v, f=f, d=d)
