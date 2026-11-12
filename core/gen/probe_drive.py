"""Check (Fable 11-12 round 9): at birth, during speech, is the concept cells' net drive (basal - somatic inhibition)
above zero, and which cells project forward? usage: GEN_ARGS="..." python -m core.gen.probe_drive"""
import json

import torch

from . import t2_concept as T

T.birth()
REC = []
_d3 = T.drives3


def rec(eff):
    out = _d3(eff)
    REC.append(out)
    return out


T.drives3 = rec
B = 16
S = [T.stream(range(15000 + b * 4, 15000 + (b + 1) * 4), 9200, False) for b in range(B)]
Tn = min(len(s) for s in S)
act = torch.zeros(B, T.N, dtype=torch.bool, device=T.DEV)
T.reset(B)
ce = (T.area == 2) & ~T.inh
ci = (T.area == 2) & T.inh
net, inh_, bas_, aud_, spk_e, spk_i = [], [], [], [], [], []
TA, TN, TS, TL = [], [], [], []                          # per tick, every stream: auditory drive, net drive, spikes, label
for t in range(Tn):
    REC.clear()
    snd = torch.tensor([S[b][t][2] >= 0 for b in range(B)], device=T.DEV)
    T.SUB = []
    act = T.step(T.organ_in([(S[b][t][0], S[b][t][1]) for b in range(B)]), act)
    sub = torch.stack(T.SUB).float()                     # [steps, B, N]
    T.SUB = None
    TA.append(torch.stack([r[3] for r in REC]).mean(0)[:, ce])
    TN.append((torch.stack([r[0] for r in REC]).mean(0) - torch.stack([r[1] for r in REC]).mean(0))[:, ce])
    TS.append(sub.sum(0)[:, ce])
    TL.append([S[b][t][2] for b in range(B)])
    if snd.any():
        bas = torch.stack([r[0] for r in REC]).mean(0)[snd]
        inh = torch.stack([r[1] for r in REC]).mean(0)[snd]
        aud = torch.stack([r[3] for r in REC]).mean(0)[snd]
        net.append((bas - inh)[:, ce])
        inh_.append(inh[:, ce]); bas_.append(bas[:, ce]); aud_.append(aud[:, ce])
        spk_e.append(sub[:, snd][:, :, ce].sum(0)); spk_i.append(sub[:, snd][:, :, ci].sum(0))
net, inh_, bas_, aud_ = (torch.cat(x) for x in (net, inh_, bas_, aud_))
th = T.theta[ce]
srcs = T.idx[ce]
from_a = (T.area[srcs.clamp_min(0)] == 1) & (srcs >= 0)
print('NET DRIVE (concept E, speech ticks)', json.dumps(dict(
    basal=round(bas_.mean().item(), 4), auditory=round(aud_.mean().item(), 4), inhib=round(inh_.mean().item(), 4),
    net=round(net.mean().item(), 4), share_net_pos=round((net.mean(0) > 0).float().mean().item(), 3),
    theta_mean=round(th.mean().item(), 4), share_net_above_theta=round((net.mean(0) > th).float().mean().item(), 3),
    spikes_e_per_tick=round(torch.cat(spk_e).mean().item(), 4), spikes_i_per_tick=round(torch.cat(spk_i).mean().item(), 4),
    rate_target_e=float(T.P['rate_c']), aud_inputs=round(from_a.float().sum(1).mean().item(), 2))))
pa = (T.area == 1) & ~T.inh
print('AUDITORY E that project', round((T.proj & pa).float().sum().item() / pa.float().sum().item(), 3),
      'mean depth of projecting', round(T.depth[T.proj & pa].mean().item(), 3))
# who feeds the concept area from the auditory side: depth histogram of the sources
d = T.depth[srcs[from_a]]
print('DEPTH of auditory sources of concept E', json.dumps([round(x, 3) for x in torch.histc(d, 5, 0, 1).div(d.numel()).tolist()]))
T.drives3 = _d3
idn = T.identity_by_stage()
ids = T.identity_by_stage(mode='sight')
print('NEWBORN SIGHT IDENTITY', json.dumps({k: v for k, v in ids.items() if k in ('visual area', 'concept', 'concept silent')}))
print('NEWBORN IDENTITY', json.dumps({k: v for k, v in idn.items() if k in ('auditory deep', 'auditory shallow', 'concept', 'concept silent')}))
print('THETA by area', json.dumps({nm: round(T.theta[(T.area == k) & ~T.inh].mean().item(), 4) for nm, k in (('auditory', 1), ('concept', 2), ('visual', 3))}))


def ncm(Cs, L):
    """nearest class mean identity of per-hearing sums (lag 0, the held ticks), 60 commonest syllables, half/half"""
    C = torch.stack(Cs, 1)                                 # [B, T, n]
    Lb = np.array(L)                                       # [T, B]
    X, Y = [], []
    for b in range(B):
        for t in range(Tn - T.P['hold']):
            y = Lb[t, b]
            if y >= 0 and (t == 0 or Lb[t - 1, b] != y):
                X.append(C[b, t:t + T.P['hold']].sum(0))
                Y.append(int(y))
    X = torch.stack(X)
    Y = torch.as_tensor(Y, device=T.DEV)
    top = torch.argsort(torch.bincount(Y), descending=True)[:60]
    k = torch.isin(Y, top)
    X, Y = X[k], Y[k]
    X = X - X.mean(0, keepdim=True)                        # each unit's usual level removed (drives are all positive)
    half = torch.rand(len(Y), generator=torch.Generator(device='cpu').manual_seed(3)).to(T.DEV) < 0.5
    cls = [c for c in top.tolist() if bool((half & (Y == c)).any())]
    M = torch.stack([X[half & (Y == c)].mean(0) for c in cls])
    Mn = M / M.norm(dim=1, keepdim=True).clamp_min(1e-9)
    te = (~half) & torch.isin(Y, torch.as_tensor(cls, device=T.DEV))
    Xn = X[te] / X[te].norm(dim=1, keepdim=True).clamp_min(1e-9)
    pos_ = {c: i for i, c in enumerate(cls)}
    ti = torch.as_tensor([pos_[int(c)] for c in Y[te].tolist()], device=T.DEV)
    return round(((Xn @ Mn.T).argmax(1) == ti).float().mean().item(), 3)


import numpy as np
print('SPARSITY', json.dumps(dict(concept_e_share_firing_per_tick=round(torch.stack(TS).gt(0).float().mean().item(), 3))))
print('CONCEPT E IDENTITY FROM', json.dumps(dict(auditory_drive=ncm(TA, TL), net_drive=ncm(TN, TL), spikes=ncm(TS, TL))))
