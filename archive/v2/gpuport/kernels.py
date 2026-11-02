# -*- coding: utf-8 -*-
"""GPU primitives of the ARCH_v2 port.

seqsum(W, rows)      out[r, c] = sum_k W[rows[r, k], c] added SEQUENTIALLY in k order in fp32 (rows < 0 skipped).
                     numpy's axis-0 reduction of W[A] (Delta.drive, Templates.match) is exactly this sequential sum
                     (checked: local numpy 1.26 and server numpy 2.1), so the port's drives are bit-identical to the
                     reference.  Triton kernel on CUDA; a slow torch loop elsewhere (CPU tests).
HashTable            per-agent open-addressing table key(int64) -> unit id; ids are assigned in creation order
                     (key-slot order within an event), exactly as MixDict assigns them; collisions are resolved
                     (linear probing), so no two conjunctions ever share a unit.
"""
from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
    HAVE_TRITON = True
except Exception:                       # noqa: BLE001
    HAVE_TRITON = False

if HAVE_TRITON:
    @triton.jit(do_not_specialize=['K', 'stride_r'])
    def _seqsum_kernel(W, ROWS, OUT, K, C, stride_w, stride_r, BLOCK_C: tl.constexpr):
        r = tl.program_id(0)
        cb = tl.program_id(1)
        cols = cb * BLOCK_C + tl.arange(0, BLOCK_C)
        cm = cols < C
        acc = tl.zeros([BLOCK_C], dtype=tl.float32)
        for k in range(0, K):
            row = tl.load(ROWS + r * stride_r + k)
            ok = row >= 0
            w = tl.load(W + row.to(tl.int64) * stride_w + cols, mask=cm & ok, other=0.0)
            acc = acc + w
        tl.store(OUT + r * C + cols, acc, mask=cm)


if HAVE_TRITON:
    @triton.jit(do_not_specialize=['K', 'stride_r'])
    def _learn_kernel(W, ROWS, CAND, TG, COEF, K, stride_r, V: tl.constexpr, C: tl.constexpr, BLOCK: tl.constexpr):
        """one learning event (R2) per program: drive = sequential fp32 sum of the active rows (numpy order), copy
        routes P, N, C added in f64 and rounded (numpy 1.26 scalar semantics), NLMS step on GEN (cols 0..V) and the
        three gates (cols V+1..V+3), then W[row] += step for every active row (rows unique per event)."""
        r = tl.program_id(0)
        cols = tl.arange(0, BLOCK)
        cm = cols < C
        acc = tl.zeros([BLOCK], dtype=tl.float32)
        for k in range(0, K):
            row = tl.load(ROWS + r * stride_r + k)
            w = tl.load(W + row.to(tl.int64) * C + cols, mask=cm & (row >= 0), other=0.0)
            acc = acc + w
        tg = tl.load(TG + r)
        cg = tl.load(COEF + r * 2)
        c2 = tl.load(COEF + r * 2 + 1)
        total = acc
        step = tl.zeros([BLOCK], dtype=tl.float32)
        for j in range(3):
            gd = tl.sum(tl.where(cols == V + 1 + j, acc, 0.0), 0)
            pg = tl.minimum(tl.maximum(gd, 0.0), 1.0)
            cj = tl.load(CAND + r * 3 + j)
            ok = (cj >= 0) & (cj < V)
            cur = tl.sum(tl.where(cols == cj, total, 0.0), 0)
            nw = (cur.to(tl.float64) + pg.to(tl.float64)).to(tl.float32)
            total = tl.where((cols == cj) & ok, nw, total)
            tj = tl.where(cj == tg, 1.0, 0.0)
            e2 = tj - pg
            s2 = tl.where((cj >= 0) & (tl.abs(e2) > 1e-4), c2 * e2, 0.0)
            step = tl.where(cols == V + 1 + j, s2, step)
        p = tl.minimum(tl.maximum(total, 0.0), 1.0)
        tv = tl.where(cols == tg, 1.0, 0.0)
        err = tv - p
        keep = ((tv > 0) | (p > 1e-4)) & (tl.abs(err) > 1e-4) & (cols <= V)
        step = tl.where(keep, cg * err, step)
        for k in range(0, K):
            row = tl.load(ROWS + r * stride_r + k)
            m = cm & (row >= 0)
            ptr = W + row.to(tl.int64) * C + cols
            w = tl.load(ptr, mask=m, other=0.0)
            tl.store(ptr, w + step, mask=m)


def learn_fused(W, rows, cand, tg, coef, V):
    """W (R, V+4) fp32; rows (n, K) global rows (-1 skip; unique within an event); cand (n, 3); tg (n,);
    coef (n, 2) fp32 = (eta*g/|A|, eta_gate*g/|A|).  In place."""
    n, K = rows.shape
    _learn_kernel[(n,)](W, rows.contiguous(), cand.contiguous(), tg.contiguous(), coef.contiguous(), K, rows.stride(0),
                        V=V, C=W.shape[1], BLOCK=512, enable_fp_fusion=False)


def seqsum(W2: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
    """W2: (R, C) fp32 contiguous; rows: (n, K) int64 (global row ids, -1 = skip, in summation order) -> (n, C) fp32"""
    n, K = rows.shape
    C = W2.shape[1]
    if n == 0:
        return torch.zeros(0, C, dtype=W2.dtype, device=W2.device)
    if W2.is_cuda and HAVE_TRITON:
        rows = rows.contiguous()
        out = torch.empty(n, C, dtype=torch.float32, device=W2.device)
        BC = 64 if C <= 64 else 128
        grid = (n, triton.cdiv(C, BC))
        _seqsum_kernel[grid](W2, rows, out, K, C, W2.stride(0), rows.stride(0), BLOCK_C=BC)
        return out
    out = torch.zeros(n, C, dtype=torch.float32, device=W2.device)
    for k in range(K):
        rk = rows[:, k]
        ok = rk >= 0
        if bool(ok.any()):
            out[ok] = out[ok] + W2[rk[ok]]
    return out


MULT = -7046029254386353131          # 0x9E3779B97F4A7C15 as signed int64 (Fibonacci hashing)
EMPTY = -1


class HashTable:
    """B independent tables of M = 2^logM slots (open addressing, linear probing, no deletion).  The flat buffers
    have one extra sink slot that receives the padding of masked scatters (never probed)."""

    def __init__(self, B, logM, device, probes=8):
        self.B, self.logM, self.M = B, logM, 1 << logM
        self.dev = device
        self.key_f = torch.full((B * self.M + 1,), EMPTY, dtype=torch.int64, device=device)
        self.uid_f = torch.full((B * self.M + 1,), -1, dtype=torch.int64, device=device)
        self.n = torch.zeros(B, dtype=torch.int64, device=device)        # units created per agent
        self.R = probes
        self.ar = torch.arange(probes, device=device)

    @property
    def key(self):
        return self.key_f[:self.B * self.M].view(self.B, self.M)

    @property
    def uid(self):
        return self.uid_f[:self.B * self.M].view(self.B, self.M)

    def clone(self):
        h = HashTable.__new__(HashTable)
        h.__dict__.update(self.__dict__)
        h.key_f, h.uid_f, h.n = self.key_f.clone(), self.uid_f.clone(), self.n.clone()
        return h

    def _h(self, keys):
        return ((keys * MULT) >> (64 - self.logM)) & (self.M - 1)

    def _probe(self, rows, keys, start):
        """rows (n,) agent ids; keys (n, K); start (n, K) first probe position -> (uid of a hit or -1, flat position of
        the first empty slot, resolved mask)."""
        M, R = self.M, self.R
        pos = (start[..., None] + self.ar) & (M - 1)                       # (n, K, R)
        flat = rows[:, None, None] * M + pos
        tk = self.key_f[flat]
        hit = tk == keys[..., None]
        emp = tk == EMPTY
        # a key is present iff it is met before the first empty slot (no deletion)
        first_emp = torch.where(emp, self.ar, R).min(-1).values            # (n, K)
        hit_r = torch.where(hit, self.ar, R).min(-1).values
        found = hit_r < first_emp
        fh = torch.gather(flat, 2, hit_r.clamp(max=R - 1)[..., None])[..., 0]
        uid = torch.where(found, self.uid_f[fh], torch.full_like(keys, -1))
        has_emp = first_emp < R
        fe = torch.gather(flat, 2, first_emp.clamp(max=R - 1)[..., None])[..., 0]
        return uid, fe, found | has_emp

    def lookup(self, rows, keys, valid, create):
        """rows (n,) long; keys (n, K) int64 >= 0; valid (n, K) bool -> uid (n, K) (-1 = none), lost (n, K) (new keys
        whose slot was taken by another new key this event: the caller places them with place_losers).
        create: absent valid keys get new ids n[b], n[b]+1, ... in key-slot order (MixDict's creation order)."""
        start = self._h(keys)
        uid, fe, resolved = self._probe(rows, keys, start)
        bad = valid & ~resolved
        if bool(bad.any()):                                                # long probe chain (rare): walk further
            uid, fe = self._slow(rows, keys, valid, start)
        uid = torch.where(valid, uid, torch.full_like(uid, -1))
        if not create:
            return uid, torch.zeros_like(valid)
        new = valid & (uid < 0)
        rank = torch.cumsum(new.to(torch.int64), 1) - 1
        nid = self.n[rows][:, None] + rank
        uid = torch.where(new, nid, uid)
        self.n.index_add_(0, rows, new.sum(1))
        sink = self.B * self.M
        tgt = torch.where(new, fe, torch.full_like(fe, sink))
        self.key_f.scatter_reduce_(0, tgt.reshape(-1), torch.where(new, keys, torch.full_like(keys, EMPTY)).reshape(-1),
                                   reduce='amax')
        won = new & (self.key_f[tgt] == keys)
        self.uid_f.scatter_(0, torch.where(won, tgt, torch.full_like(tgt, sink)).reshape(-1), uid.reshape(-1))
        self.key_f[sink] = EMPTY
        return uid, new & ~won

    def place_losers(self, rows, keys, uid, lost):
        """new keys that lost a contested empty slot: walk on to the next empty slot (sequential, rare)."""
        M = self.M
        b = rows[:, None].expand_as(keys)[lost]
        k, u = keys[lost], uid[lost]
        p = self._h(k)
        for _ in range(M):
            if not len(k):
                break
            tk = self.key_f[b * M + p]
            hit = tk == k
            emp = tk == EMPTY
            if bool(emp.any()):
                # one claim round: the max key wins each contested empty slot
                f = torch.where(emp, b * M + p, torch.full_like(p, self.B * M))
                self.key_f.scatter_reduce_(0, f, torch.where(emp, k, torch.full_like(k, EMPTY)), reduce='amax')
                won = emp & (self.key_f[f] == k)
                self.uid_f.scatter_(0, torch.where(won, f, torch.full_like(f, self.B * M)), u)
                self.key_f[self.B * M] = EMPTY
                done = won | hit
            else:
                done = hit
            k, u, b, p = k[~done], u[~done], b[~done], (p[~done] + 1) & (M - 1)

    def _slow(self, rows, keys, valid, start):
        uid = torch.full_like(keys, -1)
        fe = torch.full_like(keys, -1)
        todo = valid.clone()
        p = start.clone()
        b = rows[:, None].expand_as(keys)
        for _ in range(self.M):
            if not bool(todo.any()):
                break
            f = b * self.M + p
            tk = self.key_f[f]
            hit = todo & (tk == keys)
            uid = torch.where(hit, self.uid_f[f], uid)
            emp = todo & (tk == EMPTY)
            fe = torch.where(emp, f, fe)
            todo = todo & ~hit & ~emp
            p = (p + 1) & (self.M - 1)
        return uid, fe


def topk_mask_canon(d: torch.Tensor, k: int) -> torch.Tensor:
    """k-WTA with exact ties broken by unit index (the canonical rule = numpy argsort(-d, kind='stable')[:k])."""
    order = torch.sort(-d, dim=1, stable=True).indices[:, :k]
    m = torch.zeros_like(d, dtype=torch.bool)
    m.scatter_(1, order, True)
    return m
