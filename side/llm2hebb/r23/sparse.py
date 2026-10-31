# -*- coding: utf-8 -*-
"""Test 1b (the designer's hypothesis): is the interference of Hebbian fact writes caused by DENSE keys?

A sparse feature layer is inserted in parallel with layer l's MLP output projection:
    residual += W_down k  +  W_new s(k),      s(k) = sparse feature code of the MLP hidden activity k
s(k): standardised k -> random projection to N units -> k-WTA (keep the top a*N values, rectified), or the
      dense projection (no k-WTA), or the features of a top-k sparse autoencoder trained here on ordinary text.
W_new starts at zero and is written fact by fact with the delta rule (one shot, no back-prop):
    W_new += (v* - W_new s) s^T / (s^T s),   v* = scale * |h_l| * u_name / |u_name|   (output-embedding direction)
sites: 'pred' = the positions that predict the two name tokens; 'both' = also the subject's last token (value =
       mean direction of the two name tokens).
Sequential writes 10 -> 10k facts with R3's tests; plus the mean key overlap between different facts, dense vs sparse.
usage: python sparse.py code=rp|dense|sae N=65536 frac=0.02 site=pred layer=18 scale=1 n=10000 tag=..
"""
import os, sys, json, time, math, random
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import torch
import torch.nn.functional as F
import common as C
import facts as FA

A = dict(a.split('=', 1) for a in sys.argv[1:])
OUT = A.get('out', os.path.join(HERE, 'out', 'sparse.jsonl'))
os.makedirs(os.path.dirname(OUT), exist_ok=True)
torch.cuda.set_per_process_memory_fraction(float(A.get('memfrac', 0.22)))
torch.backends.cuda.matmul.allow_tf32 = True
torch.manual_seed(0)
T0 = time.time()
CODE, NU, FRAC = A.get('code', 'rp'), int(A.get('N', 65536)), float(A.get('frac', 0.02))
SITE, LAYER, SCALE, NMAX = A.get('site', 'pred'), int(A.get('layer', 18)), float(A.get('scale', 1.0)), int(A.get('n', 10000))
TAG = A.get('tag', f'{CODE}{NU}_{FRAC}_{SITE}')
tok, model = C.load_model()
for p in model.parameters():
    p.requires_grad_(False)
LAYERS = model.model.layers
PAD = tok.pad_token_id if tok.pad_token_id is not None else 0
DOWN = LAYERS[LAYER].mlp.down_proj


def log(d):
    d['t'] = round(time.time() - T0, 1)
    print(json.dumps(d, ensure_ascii=False), flush=True)
    with open(OUT, 'a') as f:
        f.write(json.dumps(d, ensure_ascii=False) + '\n')


def pad(seqs):
    L = max(len(s) for s in seqs)
    X = torch.full((len(seqs), L), PAD, dtype=torch.long)
    att = torch.zeros((len(seqs), L), dtype=torch.long)
    for i, s in enumerate(seqs):
        X[i, :len(s)] = torch.tensor(s)
        att[i, :len(s)] = 1
    return X.to(C.DEV), att.to(C.DEV)


dialogs = C.load_dialogs()
probes = C.Probes(tok, model, dialogs[-400:], n=150)
NAME_TOK = set(t for c in C.NAME_CH for t in C.enc(tok, c))
neigh = FA.make(tok, 300, seed=4242)


@torch.no_grad()
def collect_k(seqs, bs=32, maxtok=None):
    cap = {}
    h = DOWN.register_forward_pre_hook(lambda m, i: cap.__setitem__('k', i[0]))
    out, n = [], 0
    try:
        for b in range(0, len(seqs), bs):
            X, att = pad(seqs[b:b + bs])
            model.model(input_ids=X, attention_mask=att)
            m = att.bool().clone(); m[:, 0] = False
            out.append(cap['k'][m].half()); n += out[-1].shape[0]
            if maxtok and n >= maxtok:
                break
    finally:
        h.remove()
    return torch.cat(out)


# ------------------------------------------------------------------------------------------ the sparse code
text_seqs = [C.enc(tok, ''.join(C.dialog_lines(d)))[:96] for d in dialogs[:12000]]
KT = collect_k(text_seqs, maxtok=int(A.get('sae_tokens', 200000)) if CODE == 'sae' else 30000)
MU = KT.float().mean(0)
SD = KT.float().std(0).clamp_min(1e-4)


def std(k):
    return ((k.float() - MU) / SD).clamp(-6, 6)


KK = max(1, int(round(FRAC * NU)))
if CODE in ('rp', 'dense'):
    g = torch.Generator(device='cpu').manual_seed(7)
    R = (torch.randn(NU, MU.numel(), generator=g) / math.sqrt(MU.numel())).to(C.DEV)
elif CODE == 'sae':                       # top-k sparse autoencoder (Gao et al. 2024 style) on ordinary-text activations
    KS = int(A.get('sae_k', 64))
    X = std(KT)
    nval = X.shape[0] // 20
    Xv, Xt = X[:nval], X[nval:]
    d = X.shape[1]
    Wd = torch.randn(d, NU, device=C.DEV); Wd /= Wd.norm(dim=0, keepdim=True)
    Wd = Wd.requires_grad_(True)
    We = Wd.detach().T.clone().requires_grad_(True)
    be = torch.zeros(NU, device=C.DEV, requires_grad=True)
    bd = Xt.mean(0).clone().requires_grad_(True)
    opt = torch.optim.Adam([Wd, We, be, bd], lr=float(A.get('sae_lr', 3e-4)))

    def sae_enc(x, k):
        a = (x - bd) @ We.T + be
        v, i = a.topk(k, -1)
        return torch.zeros_like(a).scatter_(-1, i, F.relu(v))
    t = time.time()
    for step in range(int(A.get('sae_steps', 3000))):
        xb = Xt[torch.randint(0, Xt.shape[0], (4096,), device=C.DEV)]
        z = sae_enc(xb, KS)
        loss = ((z @ Wd.T + bd - xb) ** 2).sum(-1).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        with torch.no_grad():
            Wd /= Wd.norm(dim=0, keepdim=True)
    with torch.no_grad():
        zv = sae_enc(Xv, KS)
        fve = 1 - float(((zv @ Wd.T + bd - Xv) ** 2).sum() / ((Xv - Xv.mean(0)) ** 2).sum())
        alive = float((zv > 0).any(0).float().mean())
    log(dict(info='sae', N=NU, k=KS, tokens=int(Xt.shape[0]), steps=step + 1, fve_heldout=round(fve, 4),
             alive_frac_heldout=round(alive, 4), secs=round(time.time() - t, 1)))
    KK = int(A.get('code_k', KS))
    We, be, bd = We.detach(), be.detach(), bd.detach()
    del KT, X, Xt, Xv


def encode(k):
    """(..., d_ff) -> (..., NU) sparse (rectified top-KK) or dense code."""
    x = std(k)
    if CODE == 'sae':
        a = (x - bd) @ We.T + be
    else:
        a = x @ R.T
    if CODE == 'dense':
        return a
    v, i = a.topk(KK, -1)
    return torch.zeros_like(a).scatter_(-1, i, F.relu(v))


W_NEW = torch.zeros(model.config.hidden_size, NU, device=C.DEV)
NW = [0]


def hook(m, inp, out):
    if NW[0] == 0:
        return out
    s = encode(inp[0])
    return out + (s @ W_NEW.T).to(out.dtype)


DOWN.register_forward_hook(hook)


# ------------------------------------------------------------------------------------------ facts and tests
@torch.no_grad()
def eval_facts(FF, form='rewrite', bs=64):
    if not FF:
        return None
    ok = []
    for b in range(0, len(FF), bs):
        S = [f['p'][form] + [f['nt'][0]] for f in FF[b:b + bs]]
        X, att = pad(S)
        h = model.model(input_ids=X, attention_mask=att).last_hidden_state
        ln = torch.tensor([len(s) for s in S], device=C.DEV)
        ix = torch.arange(len(S), device=C.DEV)
        z1 = model.lm_head(h[ix, ln - 2]).argmax(-1)
        z2 = model.lm_head(h[ix, ln - 1]).argmax(-1)
        t = torch.tensor([f['nt'] for f in FF[b:b + bs]], device=C.DEV)
        ok.append(((z1 == t[:, 0]) & (z2 == t[:, 1])).float())
    return round(float(torch.cat(ok).mean()), 4)


@torch.no_grad()
def next_argmax(seqs, bs=64):
    out = []
    for b in range(0, len(seqs), bs):
        S = seqs[b:b + bs]
        X, att = pad(S)
        h = model.model(input_ids=X, attention_mask=att).last_hidden_state
        ln = torch.tensor([len(s) for s in S], device=C.DEV)
        out.append(model.lm_head(h[torch.arange(len(S), device=C.DEV), ln - 1]).argmax(-1))
    return torch.cat(out)


neigh_seqs = [f['p']['rewrite'] for f in neigh]
neigh0 = next_argmax(neigh_seqs)
F_ = FA.make(tok, NMAX, seed=1, exclude=set(f['subj'] for f in neigh))      # same facts as R3 / tests 1-2


@torch.no_grad()
def site_keys(FF, form='rewrite', bs=64):
    """k and |h| at the subject's last token and at the two name-predicting positions."""
    cap = {}
    h1 = DOWN.register_forward_pre_hook(lambda m, i: cap.__setitem__('k', i[0]))
    h2 = LAYERS[LAYER].register_forward_hook(lambda m, i, o: cap.__setitem__('h', o[0] if isinstance(o, tuple) else o))
    K, Hn = [], []
    try:
        for b in range(0, len(FF), bs):
            fb = FF[b:b + bs]
            S = [f['p'][form] + f['nt'] for f in fb]
            X, att = pad(S)
            model.model(input_ids=X, attention_mask=att)
            ix = torch.arange(len(fb), device=C.DEV)
            p0 = torch.tensor([len(f['p'][form]) - 1 for f in fb], device=C.DEV)
            ps = torch.tensor([len(f['p'][form]) - len(f['p']['rewrite']) + f['subj_last'] for f in fb], device=C.DEV)
            K.append(torch.stack([cap['k'][ix, ps], cap['k'][ix, p0], cap['k'][ix, p0 + 1]], 1).half())
            Hn.append(torch.stack([cap['h'][ix, ps], cap['h'][ix, p0], cap['h'][ix, p0 + 1]], 1).float().norm(dim=-1))
    finally:
        h1.remove(); h2.remove()
    return torch.cat(K), torch.cat(Hn)                       # (n, 3, d_ff), (n, 3)


def overlap_stats(Kr, Kp):
    """mean similarity between different facts' keys (same / other relation) and between a fact's own rewrite and
    paraphrase keys; dense = cosine of standardised keys, sparse = cosine of codes and Jaccard of active sets."""
    r = random.Random(3)
    n = Kr.shape[0]
    pairs = [(r.randrange(n), r.randrange(n)) for _ in range(3000)]
    pairs = [(a, b) for a, b in pairs if a != b]
    res = {}
    for site, si in (('pred', 1), ('subj', 0)):
        xd = std(Kr[:, si]); xd = xd / xd.norm(dim=-1, keepdim=True)
        xp = std(Kp[:, si]); xp = xp / xp.norm(dim=-1, keepdim=True)
        sd = encode(Kr[:, si]); sp = encode(Kp[:, si])
        sdn = sd / sd.norm(dim=-1, keepdim=True).clamp_min(1e-9); spn = sp / sp.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        act, actp = sd > 0, sp > 0
        for kind in ('same_rel', 'other_rel'):
            pp = [(a, b) for a, b in pairs if (F_[a]['rel'] == F_[b]['rel']) == (kind == 'same_rel')][:1000]
            a = torch.tensor([x for x, _ in pp], device=C.DEV); b = torch.tensor([y for _, y in pp], device=C.DEV)
            res[f'{site}_{kind}_dense_cos'] = round(float((xd[a] * xd[b]).sum(-1).mean()), 4)
            res[f'{site}_{kind}_code_cos'] = round(float((sdn[a] * sdn[b]).sum(-1).mean()), 4)
            if CODE != 'dense':
                inter = (act[a] & act[b]).sum(-1).float(); uni = (act[a] | act[b]).sum(-1).float()
                res[f'{site}_{kind}_jaccard'] = round(float((inter / uni).mean()), 4)
        res[f'{site}_selfpara_dense_cos'] = round(float((xd * xp).sum(-1).mean()), 4)
        res[f'{site}_selfpara_code_cos'] = round(float((sdn * spn).sum(-1).mean()), 4)
    return res


@torch.no_grad()
def checkpoint(n, extra):
    a = next_argmax(neigh_seqs)
    log(dict(mode='sparse', tag=TAG, code=CODE, N=NU, active=KK, frac=round(KK / NU, 5), site=SITE, layer=LAYER,
             scale=SCALE, n=n, new_rewrite=eval_facts(F_[max(0, n - 100):n]),
             new_para=eval_facts(F_[max(0, n - 100):n], 'para'), new_qa=eval_facts(F_[max(0, n - 100):n], 'qa'),
             first100=eval_facts(F_[:min(100, n)]), all=eval_facts(F_[:n]) if n <= 3000 or n == NMAX else None,
             neigh_same=round(float((a == neigh0).float().mean()), 4),
             neigh_bleed=round(float(np.mean([int(t) in NAME_TOK for t in a.tolist()])), 4), **probes.all(model), **extra))


if __name__ == '__main__':
    log(dict(info='start', args=A, base=probes.all(model), active=KK))
    t = time.time()
    K, Hn = site_keys(F_)
    Kp, _ = site_keys(F_[:2000], 'para')
    log(dict(info='overlap', tag=TAG, code=CODE, frac=round(KK / NU, 5), **overlap_stats(K[:2000], Kp)))
    U = model.lm_head.weight.float()
    U = U / U.norm(dim=-1, keepdim=True)
    t_items = time.time() - t
    done, tw = 0, 0.0
    for c in (10, 30, 100, 300, 1000, 3000, 10000):
        if c > NMAX:
            break
        t = time.time()
        with torch.no_grad():
            for b0 in range(done, c, 256):
                b1 = min(c, b0 + 256)
                S = encode(K[b0:b1])                                        # (m, 3, NU)
                for i in range(b0, b1):
                    f = F_[i]
                    items = [(1, U[f['nt'][0]]), (2, U[f['nt'][1]])]
                    if SITE == 'both':
                        items.append((0, (U[f['nt'][0]] + U[f['nt'][1]]) / math.sqrt(2)))
                    for si, u in items:
                        s = S[i - b0, si]
                        v = SCALE * Hn[i, si] * u
                        err = v - W_NEW @ s
                        W_NEW += torch.outer(err, s) / (s @ s).clamp_min(1e-9)
                        NW[0] += 1
        torch.cuda.synchronize()
        tw += time.time() - t
        done = c
        checkpoint(c, dict(ms_per_fact_write=round(1000 * tw / done, 3), W_norm=round(float(W_NEW.norm()), 2)))
    log(dict(info='end', gpu_gb=round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)))
