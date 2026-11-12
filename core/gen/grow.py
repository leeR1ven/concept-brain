"""The generator (docs/GENERATOR.md): neurons in 3D space, each with a receive side and a send side, excitatory or
inhibitory; one wiring formula for all; each neuron's own threshold keeps its own rate; competition by the inhibitory
neurons that grow with the rest. No rule says which kind of neuron connects to which.

Wiring (GENERATOR 二.3). Each neuron j draws K_j inputs i with probability proportional to
    exp(-|d| / lam_i) * exp(ks_i (cos(d, u_i) - 1)) * exp(kr_j (cos(d, u_j) - 1)),   d = p_j - p_i,
i.e. j near i (exponential distance rule, lam = the sender's axon reach), j on i's send side, i on j's receive side
(axon and dendrite overlap). A neuron with ks = kr = 0 (an interneuron) has no preferred side.
Every per-neuron number (reach, side preference, fan-in, rate, inhibitory or not) comes from the parameter fields of
the genome at the neuron's position; the fields are the only place where areas differ."""
import math

import numpy as np
import torch

DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def unit(v):
    return v / v.norm(dim=-1, keepdim=True).clamp_min(1e-9)


class Net:
    """a grown network. Fields: pos [n,3], u [n,3] (send / receive direction), inh [n] bool, rate [n] (own target
    rate), idx [n,K] (-1 = none), w [n,K] (>= 0; the sign comes from the sender), n_in (input neurons first)."""

    def __init__(self, pos, u, inh, rate, idx, w, n_in, g_inh, area):
        self.pos, self.u, self.inh, self.rate = pos, u, inh, rate
        self.idx, self.w, self.n_in, self.g_inh, self.area = idx, w, n_in, g_inh, area
        self.n = len(pos)
        self.sign = torch.where(inh, -g_inh, 1.0).float()
        self.theta = torch.zeros(self.n, device=pos.device)
        self.state = torch.zeros(self.n, dtype=torch.bool, device=pos.device)

    def drive(self, act):
        """act: bool [B, n] -> drive [B, n] (inputs' own drive is not used)"""
        a = torch.cat([act.float() * self.sign, torch.zeros(act.shape[0], 1, device=act.device)], 1)
        ix = torch.where(self.idx >= 0, self.idx, torch.full_like(self.idx, self.n))
        return (a[:, ix] * self.w[None]).sum(-1)

    def step(self, x_in, act, adapt=0.0):
        """one tick for a batch: x_in bool [B, n_in] (the organ's bits now), act bool [B, n] (the state a tick ago).
        Every neuron reads the others as they were a tick ago (one tick per link). adapt > 0: each neuron moves its
        own threshold toward its own rate (GENERATOR 二.4)."""
        d = self.drive(act)
        new = d > self.theta[None]
        new[:, :self.n_in] = x_in
        if adapt:
            r = new[:, self.n_in:].float().mean(0)
            self.theta[self.n_in:] += adapt * (r - self.rate[self.n_in:]) * d[:, self.n_in:].abs().mean(0).clamp_min(1e-3)
        return new


def grow(n_in_pos, pos, fields, seed=0, chunk=512, cut=4.0):
    """n_in_pos: [n_in, 3] input neuron positions (the organ surface); pos: [n, 3] the other neurons.
    fields(pos) -> dict of per-neuron tensors: 'dir' [n,3], 'lam', 'ks', 'kr', 'K', 'rate', 'inh' (bool).
    Input neurons send forward (their dir from fields too) and have no inputs."""
    g = torch.Generator(device='cpu').manual_seed(seed)
    P = torch.cat([n_in_pos, pos]).float().to(DEV)
    n_in, n = len(n_in_pos), len(n_in_pos) + len(pos)
    F = fields(P)
    u = unit(F['dir'] + 0.15 * torch.randn(n, 3, generator=g).to(DEV))
    inh = F['inh'].clone()
    inh[:n_in] = False
    lam, ks, kr = F['lam'], F['ks'], F['kr']
    K = F['K'].long()
    K[:n_in] = 0
    Kmax = int(K.max())
    idx = torch.full((n, Kmax), -1, dtype=torch.long, device=DEV)
    for a in range(n_in, n, chunk):
        b = min(a + chunk, n)
        d = P[None, a:b, :] - P[:, None, :]                      # [n, c, 3]: p_j - p_i
        d = d.transpose(0, 1)                                    # [c, n, 3]
        r = d.norm(dim=-1).clamp_min(1e-9)
        cs = (d * u[None, :, :]).sum(-1) / r                     # j on i's send side
        cr = (d * u[a:b, None, :]).sum(-1) / r                   # i on j's receive side
        lw = -r / lam[None, :] + ks[None, :] * (cs - 1) + kr[a:b, None] * (cr - 1)
        lw[r > cut * lam[None, :]] = -float('inf')
        ar = torch.arange(a, b, device=DEV)
        lw[ar - a, ar] = -float('inf')
        gum = -torch.log(-torch.log(torch.rand(lw.shape, generator=g).to(DEV).clamp_min(1e-12)))
        top = (lw + gum).topk(Kmax, dim=1)
        ok = torch.isfinite(lw.gather(1, top.indices)) & (torch.arange(Kmax, device=DEV)[None, :] < K[a:b, None])
        idx[a:b] = torch.where(ok, top.indices, torch.full_like(top.indices, -1))
    w = torch.where(idx >= 0, 0.5 + torch.rand(idx.shape, generator=g).to(DEV), torch.zeros(idx.shape, device=DEV))
    net = Net(P, u, inh, F['rate'], idx, w, n_in, F.get('g_inh', 1.0), F.get('area'))
    return net


def thermometer(e, levels):
    """e: float [B, C] in 0..1 -> bool [B, C * levels * 2] thermometer pairs (exactly half on)"""
    th = (torch.arange(levels, device=e.device).float() + 0.5) / levels
    up = e[..., None] >= th
    return torch.stack([up, ~up], -1).reshape(e.shape[0], -1)
