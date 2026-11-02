"""Birth geometry on the unit sheet (GPU, chunked; never an N x N matrix is kept):
local fixed fan-in tables, radius candidate lists, patches, and the reciprocal birth loop [P3, P4, P6, P7, P8]."""
import torch

from common import DEV

CH = 1024


def _gen(seed):
    g = torch.Generator(device=DEV)
    g.manual_seed(int(seed))
    return g


def local_fanin(pos_pre, pos_post, fan, r, seed):
    """[P3] each post unit takes `fan` random pre units within radius r (nearest ones if fewer are within r).
    Returns idx [n_post, fan] (int64) and weights U(0.5, 1.5) normalised to sum 1 per post unit."""
    g = _gen(seed)
    out = []
    for a in range(0, len(pos_post), CH):
        d2 = torch.cdist(pos_post[a:a + CH], pos_pre).pow(2)
        key = torch.where(d2 < r * r, torch.rand(d2.shape, generator=g, device=DEV), 1.0 + d2)
        out.append(torch.topk(key, fan, dim=1, largest=False).indices)
    idx = torch.cat(out)
    w = 0.5 + torch.rand(idx.shape, generator=g, device=DEV)
    return idx, w / w.sum(1, keepdim=True)


def radius_lists(pos_pre, pos_post, r, exclude_self=False):
    """[P7, P8] for every pre unit, all post units within radius r, padded with the sink index len(pos_post)."""
    n_post = len(pos_post)
    counts = []
    for a in range(0, len(pos_pre), CH):
        m = torch.cdist(pos_pre[a:a + CH], pos_post) < r
        if exclude_self:
            ar = torch.arange(a, min(a + CH, len(pos_pre)), device=DEV)
            m[ar - a, ar] = False
        counts.append(m.sum(1))
    cmax = int(torch.cat(counts).max())
    idx = torch.full((len(pos_pre), cmax), n_post, dtype=torch.int32, device=DEV)
    for a in range(0, len(pos_pre), CH):
        m = torch.cdist(pos_pre[a:a + CH], pos_post) < r
        if exclude_self:
            ar = torch.arange(a, min(a + CH, len(pos_pre)), device=DEV)
            m[ar - a, ar] = False
        order = torch.sort((~m).to(torch.int8), dim=1, stable=True).indices[:, :cmax]
        sel = torch.gather(m, 1, order)
        idx[a:a + CH] = torch.where(sel, order, torch.full_like(order, n_post)).to(torch.int32)
    return idx


def random_lists(n_pre, n_post, width, seed, exclude_self=False):
    """[H3] random potential partners (global, no topography), width per pre unit, without repeats per row."""
    g = _gen(seed)
    out = []
    for a in range(0, n_pre, CH):
        rows = min(CH, n_pre - a)
        key = torch.rand((rows, n_post), generator=g, device=DEV)
        if exclude_self:
            ar = torch.arange(a, a + rows, device=DEV)
            key[ar - a, ar] = 2.0
        out.append(torch.topk(key, width, dim=1, largest=False).indices.to(torch.int32))
    return torch.cat(out)


def patches(pos, size, seed):
    """[P4] Voronoi patches of about `size` units; returns patch id per unit and the number of patches"""
    g = _gen(seed)
    n_p = max(1, len(pos) // size)
    cent = torch.rand((n_p, 2), generator=g, device=DEV)
    pid = torch.cat([torch.cdist(pos[a:a + CH], cent).argmin(1) for a in range(0, len(pos), CH)])
    return pid, n_p


def reciprocal_loop(idx43, w43, idx32, w32, idx21, w21, n1, fan, seed):
    """[P6] msg 84 'G returns to A': each last-layer unit returns to the `fan` L1 copies with the largest path
    weight into it through L3 and L2. Returns by-pre idx [n4, fan] and weights U(0.5, 1.5) normalised to 1."""
    g = _gen(seed)
    out = []
    for a in range(0, idx43.shape[0], 256):
        i3 = idx43[a:a + 256]                                   # [c, F]
        p3 = w43[a:a + 256]
        i2 = idx32[i3]                                          # [c, F, F]
        p2 = p3[:, :, None] * w32[i3]
        i1 = idx21[i2]                                          # [c, F, F, F]
        p1 = p2[:, :, :, None] * w21[i2]
        dense = torch.zeros((i3.shape[0], n1), device=DEV)
        dense.scatter_add_(1, i1.reshape(i3.shape[0], -1), p1.reshape(i3.shape[0], -1))
        out.append(torch.topk(dense, fan, dim=1).indices)
    idx = torch.cat(out).to(torch.int32)
    w = 0.5 + torch.rand(idx.shape, generator=g, device=DEV)
    return idx, w / w.sum(1, keepdim=True)


def random_local_loop(pos4, pos1, fan, r, seed):
    """[P6] alternative birth loop: `fan` random L1 copies within radius r of each last-layer unit"""
    idx, w = local_fanin(pos1, pos4, fan, r, seed)             # by-post on the L4 side = by-pre for L4 -> L1
    return idx.to(torch.int32), w
