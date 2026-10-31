# -*- coding: utf-8 -*-
"""Minimal batched RWKV-7 ("x070", Goose) in plain torch, for streaming a never-reset state.

Per layer the time-mix state is a matrix S (per head, value x key) updated by the generalised delta rule
    S <- S diag(w_t) - (S kk_t) (a_t * kk_t)^T + v_t k_t^T ,   y_t = S r_t
(w = decay, kk = normalised key, a = in-context learning rate): a local error-correcting Hebbian fast-weight memory.
Projections are computed in parallel over a chunk; only the S recurrence loops over time.
Checked against the official `rwkv` pip package (see check_rwkv.py).
"""
import torch
import torch.nn.functional as F


class RWKV7:
    def __init__(self, path, device='cuda', dtype=torch.float32):
        sd = torch.load(path, map_location='cpu')
        self.dev, self.dt = device, dtype
        self.w = {k: v.to(device=device, dtype=dtype) for k, v in sd.items()}
        w = self.w
        self.L = max(int(k.split('.')[1]) for k in w if k.startswith('blocks.')) + 1
        self.C = w['emb.weight'].shape[1]
        rk = w['blocks.0.att.r_k']
        self.H, self.N = (rk.shape[0], rk.shape[1]) if rk.dim() == 2 else (self.C // 64, 64)
        for k in list(w):                                  # flatten (1,1,C) mixing vectors
            if w[k].dim() == 3 and w[k].shape[0] == 1 and w[k].shape[1] == 1:
                w[k] = w[k].view(-1)
        # layer 0 ln0 folded into the embedding (as the official inference code does)
        self.emb = F.layer_norm(w['emb.weight'], (self.C,), w['blocks.0.ln0.weight'], w['blocks.0.ln0.bias'])

    def zero_state(self, B):
        st = []
        for _ in range(self.L):
            st.append([torch.zeros(B, self.C, device=self.dev, dtype=self.dt),                     # att shift
                       torch.zeros(B, self.H, self.N, self.N, device=self.dev, dtype=torch.float32),  # S (value x key)
                       torch.zeros(B, self.C, device=self.dev, dtype=self.dt)])                    # ffn shift
        return st

    @staticmethod
    def clone_state(st):
        return [[t.clone() for t in s] for s in st]

    @torch.no_grad()
    def forward(self, idx, state, return_hidden=False, need_logits=True):
        """idx (B, T) long; state from zero_state (updated in place). Returns logits (B, T, V) (or None)."""
        w = self.w
        B, T = idx.shape
        C, H, N = self.C, self.H, self.N
        x = self.emb[idx]
        v_first = None
        for i in range(self.L):
            p = f'blocks.{i}.'
            st = state[i]
            xx = F.layer_norm(x, (C,), w[p + 'ln1.weight'], w[p + 'ln1.bias'])
            prev = torch.cat([st[0].unsqueeze(1), xx[:, :-1]], 1)
            st[0] = xx[:, -1].clone()
            d = prev - xx
            xr, xw, xk, xv, xa, xg = (xx + d * w[p + 'att.x_' + c] for c in 'rwkvag')
            r = xr @ w[p + 'att.receptance.weight'].T
            ww = w[p + 'att.w0'] + torch.tanh(xw @ w[p + 'att.w1']) @ w[p + 'att.w2']
            ww = torch.exp(-0.606531 * torch.sigmoid(ww.float()))        # decay in (0.545, 1)
            k = xk @ w[p + 'att.key.weight'].T
            v = xv @ w[p + 'att.value.weight'].T
            if i == 0:
                v_first = v
            else:
                v = v + (v_first - v) * torch.sigmoid(w[p + 'att.v0'] + (xv @ w[p + 'att.v1']) @ w[p + 'att.v2'])
            a = torch.sigmoid(w[p + 'att.a0'] + (xa @ w[p + 'att.a1']) @ w[p + 'att.a2'])
            g = torch.sigmoid(xg @ w[p + 'att.g1']) @ w[p + 'att.g2']
            kk = F.normalize((k * w[p + 'att.k_k']).view(B, T, H, N), dim=-1).view(B, T, C)
            k = k * (1 + (a - 1) * w[p + 'att.k_a'])
            y = self._wkv(st, r, ww, k, v, -kk, kk * a, B, T)
            y = F.group_norm(y.view(B * T, C), H, w[p + 'att.ln_x.weight'], w[p + 'att.ln_x.bias'], eps=64e-5).view(B, T, C)
            rk = (r.view(B, T, H, N) * k.view(B, T, H, N) * w[p + 'att.r_k'].view(H, N)).sum(-1, keepdim=True)
            y = y + (rk * v.view(B, T, H, N)).view(B, T, C)
            x = x + (y * g) @ w[p + 'att.output.weight'].T
            xx = F.layer_norm(x, (C,), w[p + 'ln2.weight'], w[p + 'ln2.bias'])
            prev = torch.cat([st[2].unsqueeze(1), xx[:, :-1]], 1)
            st[2] = xx[:, -1].clone()
            kf = xx + (prev - xx) * w[p + 'ffn.x_k']
            kf = torch.relu(kf @ w[p + 'ffn.key.weight'].T) ** 2
            x = x + kf @ w[p + 'ffn.value.weight'].T
        h = F.layer_norm(x, (C,), w['ln_out.weight'], w['ln_out.bias'])
        logits = (h @ w['head.weight'].T) if need_logits else None
        return (logits, h) if return_hidden else logits

    def _wkv(self, st, r, w, k, v, a, b, B, T):
        H, N = self.H, self.N
        S = st[1]
        r, w, k, v, a, b = (z.float().view(B, T, H, N) for z in (r, w, k, v, a, b))
        out = torch.empty(B, T, H, N, device=r.device, dtype=torch.float32)
        for t in range(T):
            sa = torch.einsum('bhij,bhj->bhi', S, a[:, t])                       # S a  (value)
            S = S * w[:, t].unsqueeze(2) + sa.unsqueeze(3) * b[:, t].unsqueeze(2) + v[:, t].unsqueeze(3) * k[:, t].unsqueeze(2)
            out[:, t] = torch.einsum('bhij,bhj->bhi', S, r[:, t])
        st[1] = S
        return out.view(B, T, H * N).to(self.dt)


# ---------------------------------------------------------------------------------------------- Triton recurrence
try:
    import triton
    import triton.language as tl

    @triton.jit
    def _wkv7_kernel(r_ptr, w_ptr, k_ptr, v_ptr, a_ptr, b_ptr, s_ptr, y_ptr, T, H: tl.constexpr, N: tl.constexpr):
        pid = tl.program_id(0)                 # one program per (batch, head, value row i)
        i = pid % N
        bh = pid // N
        bb = bh // H
        h = bh % H
        j = tl.arange(0, N)
        s_off = (bh * N + i) * N + j
        S = tl.load(s_ptr + s_off)
        for t in range(T):
            base = ((bb * T + t) * H + h) * N
            at = tl.load(a_ptr + base + j)
            wt = tl.load(w_ptr + base + j)
            bt = tl.load(b_ptr + base + j)
            kt = tl.load(k_ptr + base + j)
            rt = tl.load(r_ptr + base + j)
            vt = tl.load(v_ptr + base + i)
            sa = tl.sum(S * at, axis=0)
            S = S * wt + sa * bt + vt * kt
            tl.store(y_ptr + base + i, tl.sum(S * rt, axis=0))
        tl.store(s_ptr + s_off, S)

    def wkv_triton(S, r, w, k, v, a, b):
        """S (B,H,N,N) fp32 updated in place; r..b (B,T,H,N) fp32 contiguous -> y (B,T,H,N)."""
        B, T, H, N = r.shape
        y = torch.empty_like(r)
        _wkv7_kernel[(B * H * N,)](r, w, k, v, a, b, S, y, T, H=H, N=N)
        return y
    HAVE_TRITON = True
except Exception as e:  # noqa: BLE001
    HAVE_TRITON = False
    TRITON_ERR = repr(e)


def _wkv_fast(self, st, r, w, k, v, a, b, B, T):
    H, N = self.H, self.N
    if not (HAVE_TRITON and self.use_triton):
        return RWKV7._wkv_loop(self, st, r, w, k, v, a, b, B, T)
    r, w, k, v, a, b = (z.float().reshape(B, T, H, N).contiguous() for z in (r, w, k, v, a, b))
    S = st[1].contiguous()
    y = wkv_triton(S, r, w, k, v, a, b)
    st[1] = S
    return y.view(B, T, H * N).to(self.dt)


RWKV7._wkv_loop = RWKV7._wkv
RWKV7._wkv = _wkv_fast
RWKV7.use_triton = True
