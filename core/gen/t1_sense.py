"""Generator test 1 (GENERATOR 四.1): grow one sensory area from the field rules alone and check that
  - the structure is layered and runs forward by itself (few backward / sideways links; hop depth follows position);
  - activity stays bounded: each depth fires near its own rate, with no runaway or silence (competition only by the
    grown inhibitory neurons, no top-k);
  - similar inputs light similar neurons deep inside, unrelated ones do not.
usage: python -m core.gen.t1_sense [key=value ...]   (CPU or GPU; about a minute at the defaults)"""
import json
import sys
import time

import torch

from .grow import DEV, grow, thermometer

P = dict(C=24, levels=8, L=24.0, W=24.0, H=16.0, dens=1.0, lam_e=1.5, lam_i=1.0, k_side=3.0, K_e=12, K_i=16,
         p_inh=0.2, rate_in=0.20, rate_out=0.05, g_inh=1.5, birth=1500, adapt=0.05, hold=10, B=48, seed=0)
for a in sys.argv[1:]:
    k, v = a.split('=', 1)
    P[k] = json.loads(v)
torch.manual_seed(P['seed'])
C, LV = P['C'], P['levels']
n_in = C * LV * 2
# the organ surface: channel along y, level and pair along z, at x = 0
yy = torch.arange(C).repeat_interleave(LV * 2).float() * P['W'] / C
zz = torch.arange(LV * 2).repeat(C).float() * P['H'] / (LV * 2)
in_pos = torch.stack([torch.zeros(n_in) - 0.5, yy, zz], 1)
n = int(P['L'] * P['W'] * P['H'] * P['dens'])
pos = torch.rand(n, 3) * torch.tensor([P['L'], P['W'], P['H']])


def fields(Q):
    """the genome's parameter fields for this test: one sensory area along +x"""
    m = len(Q)
    g = torch.Generator(device='cpu').manual_seed(P['seed'] + 1)
    inh = (torch.rand(m, generator=g) < P['p_inh']).to(Q.device)
    x = Q[:, 0].clamp(0, P['L']) / P['L']
    f = lambda e, i: torch.where(inh, torch.full((m,), float(i), device=Q.device), torch.full((m,), float(e), device=Q.device))  # noqa: E731
    return dict(dir=torch.tensor([1.0, 0.0, 0.0], device=Q.device).repeat(m, 1) * (~inh)[:, None].float() +
                torch.randn(m, 3, device=Q.device) * inh[:, None].float(),
                lam=f(P['lam_e'], P['lam_i']), ks=f(P['k_side'], 0.0), kr=f(P['k_side'], 0.0),
                K=f(P['K_e'], P['K_i']), inh=inh,
                rate=P['rate_in'] * (P['rate_out'] / P['rate_in']) ** x, g_inh=P['g_inh'], area=None)


t0 = time.time()
net = grow(in_pos, pos, fields, seed=P['seed'])
tgw = time.time() - t0
# ---- structure
src, tgt = net.idx, torch.arange(net.n, device=DEV)[:, None].expand_as(net.idx)
m = (src >= 0) & (~net.inh[tgt]) & (~net.inh[src.clamp_min(0)])     # excitatory -> excitatory links
d = net.pos[tgt[m]] - net.pos[src[m]]
dx = d[:, 0] / d.norm(dim=1).clamp_min(1e-9)
depth = torch.full((net.n,), -1, dtype=torch.long, device=DEV)
depth[:net.n_in] = 0
front = torch.zeros(net.n, dtype=torch.bool, device=DEV)
front[:net.n_in] = True
h = 0
while front.any() and h < 100:
    h += 1
    a = torch.cat([front.float(), torch.zeros(1, device=DEV)])
    hit = (a[torch.where(net.idx >= 0, net.idx, net.n)].sum(1) > 0) & (depth < 0)
    depth[hit] = h
    front = hit
ex = ~net.inh
ex[:net.n_in] = False
ok = ex & (depth >= 0)
corr = torch.corrcoef(torch.stack([depth[ok].float(), net.pos[ok, 0]]))[0, 1].item()
print('STRUCTURE', json.dumps(dict(neurons=net.n, inputs=net.n_in, grow_s=round(tgw, 1),
                                   backward=round((dx < 0).float().mean().item(), 4),
                                   sideways=round((dx.abs() < 0.3).float().mean().item(), 4),
                                   reached=round((depth[ex] >= 0).float().mean().item(), 3), max_depth=int(depth.max()),
                                   depth_x_corr=round(corr, 3))), flush=True)


# ---- birth: spontaneous changing input, each neuron finds its own threshold
def sounds(Bn):
    """smooth random spectra in 0..1 (C channels)"""
    e = torch.rand(Bn, C, device=DEV)
    k = torch.tensor([0.25, 0.5, 0.25], device=DEV)[None, None, :]
    return torch.nn.functional.conv1d(e[:, None, :], k, padding=1)[:, 0].clamp(0, 1)


net.theta[net.n_in:] = net.drive(torch.rand(8, net.n, device=DEV) < 0.1).mean(0)[net.n_in:]
act = torch.zeros(P['B'], net.n, dtype=torch.bool, device=DEV)
x = thermometer(sounds(P['B']), LV)
for t in range(P['birth']):
    if t % P['hold'] == 0:
        x = thermometer(sounds(P['B']), LV)
    act = net.step(x, act, adapt=P['adapt'])
xs = net.pos[:, 0]
bins = [(0, 6), (6, 12), (12, 18), (18, 24)]


def rates(A):
    out = {}
    for lo, hi in bins:
        mm = ex & (xs >= lo) & (xs < hi)
        out[f'{lo}-{hi}'] = round(A[:, mm].float().mean().item(), 3)
    return out


def settle(x, ticks=None):
    a = torch.zeros(x.shape[0], net.n, dtype=torch.bool, device=DEV)
    hist = []
    for _ in range(ticks or P['hold'] * 2):
        a = net.step(x, a)
        hist.append(a)
    return a, hist


e0 = sounds(P['B'])
A, hist = settle(thermometer(e0, LV))
print('RATES target', {f'{lo}-{hi}': round(P['rate_in'] * (P['rate_out'] / P['rate_in']) ** (((lo + hi) / 2) / P['L']), 3)
                       for lo, hi in bins}, 'got', rates(A), flush=True)
flip = (torch.stack(hist[-4:]).float().mean(0) * (1 - torch.stack(hist[-4:]).float().mean(0)) > 0).float()
print('STABILITY share of excitatory units still flipping in the last 4 ticks', round(flip[:, ex].mean().item(), 3), flush=True)
res = {}
for eps in (0.02, 0.05, 0.1, 0.3, 'new'):
    e1 = sounds(P['B']) if eps == 'new' else (e0 + eps * (torch.rand_like(e0) - 0.5) * 2).clamp(0, 1)
    A1, _ = settle(thermometer(e1, LV))
    row = {}
    for lo, hi in bins:
        mm = ex & (xs >= lo) & (xs < hi)
        a, b = A[:, mm].float(), A1[:, mm].float()
        row[f'{lo}-{hi}'] = round(((a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1)).clamp_min(1e-9)).mean().item(), 3)
    res[str(eps)] = row
print('SIMILARITY (cosine of the codes; input change eps)', json.dumps(res), f'({time.time() - t0:.0f} s)', flush=True)
