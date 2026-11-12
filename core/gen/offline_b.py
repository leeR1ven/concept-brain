"""Offline check (Fable 11-11 round 6): is the generated auditory code good enough for concepts? Record the generated
brain's auditory spike counts per syllable hearing (sound only, no learning in the brain), then run the OLD bench rule b
(stay_bench came=b: graded check with each unit's usual learned push removed, top-k, a clean oracle teacher present on a
share of hearings) on those features, and measure the concept code's identity by sound alone before and after.
Bench reference on the old core's sound layer 1: identity 0.82 -> 0.90-0.93, A->B 0.47 -> 0.67-0.78.
usage: GEN_ARGS="..." python -m core.gen.offline_b [stage=deep|shallow|all] [n_tr=600] [n_te=150] [stdp=0]"""
import json
import os
import sys
import time

import numpy as np
import torch

from . import t2_concept as T

DEV = T.DEV
opt = dict(counts=1, stage='all', n_tr=600, n_te=150, n_learn=1200, stdp=0, NC=8000, K=80, fan=24, p_see=0.5, step=0.15,
           lag=2, seed=0)
for a in sys.argv[1:]:
    k, v = a.split('=', 1)
    opt[k] = json.loads(v) if v[0].isdigit() or v[0] in '-[' else v
t0 = time.time()
if not (opt.get('cache') and os.path.exists(str(opt.get('cache')))):
    T.birth()
if opt['stdp'] and not (opt.get('cache') and os.path.exists(str(opt.get('cache')))):   # optional: let the sensory areas learn first (STDP only)
    T.P['eta_b'] = 0.0
    T.run_streams(int(opt['stdp']), 0, 7700, True, True)
area, inh = T.area, T.inh
sel = (area == 1) & ~inh
if opt['stage'] == 'ear':                                 # the ear's own fibres: what the organ gives
    sel = torch.arange(T.N, device=DEV) < T.n_ear
elif opt['stage'] != 'all':
    sel = sel & ((T.depth >= 0.5) if opt['stage'] == 'deep' else (T.depth < 0.5))
fidx = torch.nonzero(sel).squeeze(1)


def record(d0, n_d, tag):
    """per tick spike counts of the auditory units [B, T, nf] and the labels; the per-hearing features are built later
    for each lag (the first ticks of a syllable, from lag on, over its held ticks)"""
    B = 16
    per = max(1, n_d // B)
    S = [T.stream(range(d0 + b * per, d0 + (b + 1) * per), tag, False) for b in range(B)]
    Tn = min(len(s) for s in S)
    act = torch.zeros(B, T.N, dtype=torch.bool, device=DEV)
    T.reset(B)
    F = []
    for t in range(Tn):
        T.SUB = []
        act = T.step(T.organ_in([(S[b][t][0], S[b][t][1]) for b in range(B)]), act)
        F.append(torch.stack(T.SUB).float().sum(0)[:, fidx])
    T.SUB = None
    return torch.stack(F, 1), [[S[b][t][2] for t in range(Tn)] for b in range(len(S))]


def hearings(rec, lag):
    F, L = rec
    hold = T.P['hold']
    X, Y = [], []
    for b in range(F.shape[0]):
        lab = L[b]
        for t in range(F.shape[1] - hold - lag):
            if lab[t] >= 0 and (t == 0 or lab[t - 1] != lab[t]):
                X.append(F[b, t + lag:t + lag + int(opt.get('win', hold))].sum(0))
                Y.append(lab[t])
    return torch.stack(X), np.array(Y)


cache = opt.get('cache')
if cache and os.path.exists(str(cache)):                # the same recording for every variant of the rule (minutes)
    Rtr, Rte, Rl = torch.load(str(cache), weights_only=False)
    Rtr, Rte, Rl = [(r[0].to(DEV), r[1]) for r in (Rtr, Rte, Rl)]
else:
    Rtr, Rte, Rl = record(0, opt['n_tr'], 7701), record(15000, opt['n_te'], 9201), record(opt['n_tr'], opt['n_learn'], 7702)
    if cache:
        torch.save([(r[0].cpu(), r[1]) for r in (Rtr, Rte, Rl)], str(cache))


def raw_id(Xa, Ya, Xb, Yb):
    cnt_ = np.bincount(Ya)
    top_ = [int(c) for c in np.argsort(-cnt_)[:100]]
    M, ks = [], []
    for c in top_:
        m = Ya == c
        if m.sum() >= 2:
            M.append(Xa[torch.as_tensor(m, device=DEV)].float().mean(0))
            ks.append(c)
    M = torch.stack(M)
    Mn = M / M.norm(dim=1, keepdim=True).clamp_min(1e-9)
    pos_ = {c: i for i, c in enumerate(ks)}
    m = np.array([y in pos_ for y in Yb])
    X = Xb[torch.as_tensor(m, device=DEV)].float()
    X = X / X.norm(dim=1, keepdim=True).clamp_min(1e-9)
    ti = torch.as_tensor([pos_[y] for y in Yb[m]], device=DEV)
    return round(((X @ Mn.T).argmax(1) == ti).float().mean().item(), 3)


best, best_lag = -1, 0
for lag in range(4):
    a_, ya = hearings(Rtr, lag)
    b_, yb = hearings(Rte, lag)
    r = raw_id(a_, ya, b_, yb)
    print(f'RAW counts, lag {lag}: identity {r}', flush=True)
    if r > best:
        best, best_lag = r, lag
Xtr, Ytr = hearings(Rtr, best_lag)
Xte, Yte = hearings(Rte, best_lag)
Xl, Yl = hearings(Rl, best_lag)
# a linear ceiling (Fable round 7): a supervised ridge readout on the same features, 100 syllables
def ridge_id(Xa, Ya, Xb, Yb, lam=10.0):
    cnt_ = np.bincount(Ya)
    top_ = [int(c) for c in np.argsort(-cnt_)[:100]]
    pos_ = {c: i for i, c in enumerate(top_)}
    ma = np.array([y in pos_ for y in Ya])
    mb = np.array([y in pos_ for y in Yb])
    A = Xa[torch.as_tensor(ma, device=DEV)].float()
    mu_, sd_ = A.mean(0), A.std(0).clamp_min(1e-6)
    A = (A - mu_) / sd_
    Yo = torch.zeros(len(A), len(top_), device=DEV)
    Yo[torch.arange(len(A)), torch.as_tensor([pos_[y] for y in Ya[ma]], device=DEV)] = 1.0
    W = torch.linalg.solve(A.T @ A + lam * torch.eye(A.shape[1], device=DEV), A.T @ Yo)
    Bm = (Xb[torch.as_tensor(mb, device=DEV)].float() - mu_) / sd_
    ti = torch.as_tensor([pos_[y] for y in Yb[mb]], device=DEV)
    return round(((Bm @ W).argmax(1) == ti).float().mean().item(), 3)


print('LINEAR CEILING (ridge, 100 syllables): counts', ridge_id(Xtr, Ytr, Xte, Yte), 'binary',
      ridge_id(Xtr > 0, Ytr, Xte > 0, Yte), flush=True)
if opt.get('sparse'):                                   # lit = above the unit's own usual count (ARCH 二.4, input side)
    m_ = Xtr.float().mean(0)
    Xtr, Xte, Xl = Xtr > m_, Xte > m_, Xl > m_
else:
    Xtr, Xte, Xl = Xtr > 0, Xte > 0, Xl > 0               # rule b reads lit / not lit, as on the bench
print('LIT share of the features used by rule b', round(Xtr.float().mean().item(), 3), flush=True)
print('RECORDED', json.dumps(dict(features=len(fidx), train=len(Ytr), test=len(Yte), learn=len(Yl),
                                  lit=round(Xtr.float().mean().item(), 3))), f'({time.time() - t0:.0f} s)', flush=True)
# ---- the old bench rule b on these features
g = torch.Generator(device='cpu').manual_seed(int(opt['seed']))
nf, NC, K = Xtr.shape[1], opt['NC'], opt['K']
bi = torch.randint(0, nf, (NC, opt['fan']), generator=g).to(DEV)
bw = (0.5 + torch.rand(NC, opt['fan'], generator=g)).to(DEV)
F = torch.zeros(nf, NC, device=DEV)
F.index_put_((bi.reshape(-1), torch.arange(NC, device=DEV).repeat_interleave(opt['fan'])), bw.reshape(-1), accumulate=True)
D0 = Xtr.float() @ F
usual = D0.mean(0)
win = float(torch.topk(D0 - usual[None], K, dim=1).values.mean())
E = torch.zeros(nf, NC, device=DEV)
I = torch.zeros(nf, NC, device=DEV)
MASK = None
if opt.get('kl'):                                       # each unit may learn from only kl random features (as in the brain)
    MASK = torch.zeros(nf, NC, device=DEV)
    MASK.scatter_(0, torch.argsort(torch.rand(nf, NC, generator=g), dim=0)[:int(opt['kl'])].to(DEV), 1.0)
MU = torch.zeros(NC, device=DEV)
cnt = np.bincount(np.concatenate([Ytr, Yl]))
cands = [int(x) for x in np.argsort(-cnt)[:300]]
col = {c: i for i, c in enumerate(cands)}
SETS = torch.zeros(len(cands), NC, dtype=torch.bool, device=DEV)
for i in range(len(cands)):
    SETS[i, torch.randperm(NC, generator=g)[:K].to(DEV)] = True


def lit(Dm):
    v, i = torch.topk(Dm, K, dim=1)
    return torch.zeros_like(Dm, dtype=torch.bool).scatter_(1, i, v > 0)


def Wl():
    W = E - I
    if MASK is not None:
        W = W * MASK
        if opt.get('ffi'):
            return (W - W.sum(0, keepdim=True) / MASK.sum(0, keepdim=True).clamp_min(1.0)) * MASK
    if opt.get('ffi'):                                  # feed-forward inhibition: only the push above the mean weight
        W = W - W.mean(0, keepdim=True)
    return W


def encode(X):
    return lit((X.float() @ F - usual[None]) / win + (X.float() @ Wl() - (0 if opt.get('ffi') else MU[None])))


def identity():
    Ctr, Cte = encode(Xtr).float(), encode(Xte).float()
    top = [c for c in cands[:100]]
    M, ks = [], []
    for c in top:
        m = Ytr == c
        if m.sum() >= 2:
            M.append(Ctr[torch.as_tensor(m, device=DEV)].mean(0))
            ks.append(c)
    M = torch.stack(M)
    Mn = M / M.norm(dim=1, keepdim=True).clamp_min(1e-9)
    pos = {c: i for i, c in enumerate(ks)}
    m = np.array([y in pos for y in Yte])
    Tn = Cte[torch.as_tensor(m, device=DEV)]
    Tn = Tn / Tn.norm(dim=1, keepdim=True).clamp_min(1e-9)
    ti = torch.as_tensor([pos[y] for y in Yte[m]], device=DEV)
    sc = Tn @ Mn.T
    own = sc.gather(1, ti[:, None]).mean().item()
    return dict(identity=round(((sc.argmax(1) == ti).float().mean().item()), 3), own=round(own, 3),
                other=round(sc.mean().item(), 3))


# nearest class mean on the raw features, for reference
def raw_identity():
    top = cands[:100]
    M, ks = [], []
    for c in top:
        m = Ytr == c
        if m.sum() >= 2:
            M.append(Xtr[torch.as_tensor(m, device=DEV)].float().mean(0))
            ks.append(c)
    M = torch.stack(M)
    Mn = M / M.norm(dim=1, keepdim=True).clamp_min(1e-9)
    pos = {c: i for i, c in enumerate(ks)}
    m = np.array([y in pos for y in Yte])
    X = Xte[torch.as_tensor(m, device=DEV)].float()
    X = X / X.norm(dim=1, keepdim=True).clamp_min(1e-9)
    ti = torch.as_tensor([pos[y] for y in Yte[m]], device=DEV)
    return round(((X @ Mn.T).argmax(1) == ti).float().mean().item(), 3)


print(f'RAW binary at the best lag {best_lag}, nearest class mean (100 syllables):', raw_identity(), flush=True)
print('RULE B 0', json.dumps(identity()), flush=True)
rng = np.random.default_rng(1)
for ep in range(int(opt.get('passes', 3))):
    for n in range(len(Yl)):
        x = Xl[n].float()
        a = torch.nonzero(x).squeeze(1)
        if a.numel() == 0:
            continue
        if opt.get('ffi'):
            W_ = Wl()
            q = W_[a].sum(0)
            tot = ((x @ F) - usual) / win + q
        else:
            q = E[a].sum(0) - I[a].sum(0)
            MU += (q - MU) / 1000.0
            tot = ((x @ F) - usual) / win + (q - MU)
        on = lit(tot[None])[0]
        came = on
        y = int(Yl[n])
        if y in col and rng.random() < opt['p_see']:
            came = on | SETS[col[y]]
        pred = tot.clamp(0.0, 1.0)
        pred = torch.where(came | (tot >= 1.0), pred, torch.zeros_like(pred))
        d = (0.5 * opt['step'] / a.numel()) * (came.float() - pred)[None]
        if MASK is not None:
            d = d * MASK[a] * (nf / float(opt['kl']))   # the same total step shared over the links a unit has
        E[a] = (E[a] + d).clamp_min(0.0)
        I[a] = (I[a] - d).clamp_min(0.0)
    print('RULE B after', ep + 1, 'passes', json.dumps(identity()), f'({time.time() - t0:.0f} s)', flush=True)
