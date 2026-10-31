# -*- coding: utf-8 -*-
"""R3: one-shot Hebbian (delta-rule, rank-one) fact writes into one MLP layer of a frozen pretrained LLM, no
back-propagation at write time; sequential writes 10 -> 10,000 facts; compared with sequential fine-tuning.

write rule (one item = key k -> target v*):   W_down <- W_down + (v* - W_down k) kt^T / (k^T kt)
  kt = k            pure delta rule ("Hebbian": presynaptic k x postsynaptic error)
  kt = C^-1 k       ROME's whitened key (C = E[k k^T] over ordinary text: a fixed presynaptic decorrelation)
key k   : input of down_proj in layer l, at site 'last' (the position that predicts the name token) or 'subj' (the
          subject's last token, as in ROME).
value v*: 'icl'  : W0 k + g * (h_l(fact in context) - h_l(no context))  -- the model's own in-context representation
                   (forward passes only; h_l = residual output of layer l at the same site)
          'unemb': W0 k + s * |h_l| * u_t / |u_t|  -- the target token's (tied) output embedding (site 'last' only)
usage: python r3.py mode=dev | mode=seq layer=.. site=.. value=.. g=.. s=.. cov=I|C n=10000 tag=..
       python r3.py mode=ft params=L|all layer=.. lr=.. n=1000 tag=..
"""
import os, sys, json, time, math
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import torch
import common as C
import facts as FA

A = dict(a.split('=', 1) for a in sys.argv[1:])
MODE = A.get('mode', 'dev')
OUT = A.get('out', os.path.join(HERE, 'out', 'r3.jsonl'))
os.makedirs(os.path.dirname(OUT), exist_ok=True)
torch.cuda.set_per_process_memory_fraction(float(A.get('memfrac', 0.22)))
torch.manual_seed(0)
T0 = time.time()
tok, model = C.load_model()
for p in model.parameters():
    p.requires_grad_(False)
LAYERS = model.model.layers
PAD = tok.pad_token_id if tok.pad_token_id is not None else 0


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


@torch.no_grad()
def capture(seqs, layer, pos, bs=128):
    """k (input of down_proj) and h (residual output) of `layer` at position pos[i] of seqs[i]."""
    cap = {}
    h1 = LAYERS[layer].mlp.down_proj.register_forward_pre_hook(lambda m, i: cap.__setitem__('k', i[0]))
    h2 = LAYERS[layer].register_forward_hook(lambda m, i, o: cap.__setitem__('h', o[0] if isinstance(o, tuple) else o))
    K, H = [], []
    try:
        for b in range(0, len(seqs), bs):
            X, att = pad(seqs[b:b + bs])
            model.model(input_ids=X, attention_mask=att)
            ix = torch.arange(X.shape[0], device=C.DEV)
            pp = torch.tensor(pos[b:b + bs], device=C.DEV)
            K.append(cap['k'][ix, pp].float())
            H.append(cap['h'][ix, pp].float())
    finally:
        h1.remove(); h2.remove()
    return torch.cat(K), torch.cat(H)


@torch.no_grad()
def eval_facts(F, form='rewrite', bs=256):
    """teacher-forced exact match of both name tokens."""
    if not F:
        return None
    seqs = [f['p'][form] + [f['nt'][0]] for f in F]
    ok = []
    for b in range(0, len(seqs), bs):
        S = seqs[b:b + bs]
        X, att = pad(S)
        h = model.model(input_ids=X, attention_mask=att).last_hidden_state
        ln = torch.tensor([len(s) for s in S], device=C.DEV)
        ix = torch.arange(len(S), device=C.DEV)
        z1 = model.lm_head(h[ix, ln - 2]).argmax(-1)
        z2 = model.lm_head(h[ix, ln - 1]).argmax(-1)
        t1 = torch.tensor([f['nt'][0] for f in F[b:b + bs]], device=C.DEV)
        t2 = torch.tensor([f['nt'][1] for f in F[b:b + bs]], device=C.DEV)
        ok.append(((z1 == t1) & (z2 == t2)).float())
    return round(float(torch.cat(ok).mean()), 4)


@torch.no_grad()
def next_argmax(seqs, bs=256):
    out = []
    for b in range(0, len(seqs), bs):
        S = seqs[b:b + bs]
        X, att = pad(S)
        h = model.model(input_ids=X, attention_mask=att).last_hidden_state
        ln = torch.tensor([len(s) for s in S], device=C.DEV)
        out.append(model.lm_head(h[torch.arange(len(S), device=C.DEV), ln - 1]).argmax(-1))
    return torch.cat(out)


dialogs = C.load_dialogs()
probes = C.Probes(tok, model, dialogs[-400:], n=150)
NAME_TOK = set(t for c in C.NAME_CH for t in C.enc(tok, c))
neigh = FA.make(tok, 300, seed=4242)
neigh_seqs = [f['p']['rewrite'] for f in neigh]
neigh0 = next_argmax(neigh_seqs)


def neigh_stats():
    a = next_argmax(neigh_seqs)
    bleed = float(np.mean([int(t) in NAME_TOK for t in a.tolist()]))
    return dict(neigh_same=round(float((a == neigh0).float().mean()), 4), neigh_bleed=round(bleed, 4))


_COV = {}


def cov_inv(layer):
    if layer not in _COV:
        seqs = [C.enc(tok, ''.join(C.dialog_lines(d)))[:96] for d in dialogs[:600]]
        cap = {}
        hk = LAYERS[layer].mlp.down_proj.register_forward_pre_hook(lambda m, i: cap.__setitem__('k', i[0]))
        Cm = None
        n = 0
        with torch.no_grad():
            for b in range(0, len(seqs), 64):
                X, att = pad(seqs[b:b + 64])
                model.model(input_ids=X, attention_mask=att)
                k = cap['k'][att.bool()].double()
                Cm = k.T @ k if Cm is None else Cm + k.T @ k
                n += k.shape[0]
        hk.remove()
        Cm = Cm / n
        lam = 1e-3 * float(torch.trace(Cm)) / Cm.shape[0]
        _COV[layer] = torch.linalg.inv(Cm + lam * torch.eye(Cm.shape[0], device=C.DEV, dtype=torch.float64)).float()
        log(dict(info='cov', layer=layer, n=n))
    return _COV[layer]


def items_for(F, layer, site, value, g=1.0, s=1.0):
    """list of (k, v*) write items for facts F, computed with the ORIGINAL weights (k is exactly invariant to writes
    in this layer's down_proj; v* is the target output of this layer's MLP)."""
    W0 = LAYERS[layer].mlp.down_proj.weight.float()
    ctx, pos, ictx, ipos, tgt = [], [], [], [], []
    for f in F:
        pre = C.enc(tok, f"{f['subj']}叫{f['name']}。\n")
        if site == 'last':
            for j in range(2):
                c = f['p']['rewrite'] + f['nt'][:j]
                ctx.append(c); pos.append(len(c) - 1)
                ictx.append(pre + c); ipos.append(len(pre) + len(c) - 1)
                tgt.append(f['nt'][j])
        else:
            c = f['p']['rewrite']
            ctx.append(c); pos.append(f['subj_last'])
            ictx.append(pre + c); ipos.append(len(pre) + f['subj_last'])
            tgt.append(f['nt'][0])
    k, h = capture(ctx, layer, pos)
    v0 = k @ W0.T
    if value == 'icl':
        _, hi = capture(ictx, layer, ipos)
        v = v0 + g * (hi - h)
    else:
        u = model.lm_head.weight[torch.tensor(tgt, device=C.DEV)].float()
        u = u / u.norm(dim=-1, keepdim=True)
        v = v0 + s * h.norm(dim=-1, keepdim=True) * u
    per = 2 if site == 'last' else 1
    return [(k[i * per:(i + 1) * per], v[i * per:(i + 1) * per]) for i in range(len(F))]


@torch.no_grad()
def write(layer, K, V, Ci=None):
    """Ci: None (pure delta rule), a matrix (C^-1 whitening), or an int s (sparse key: only the s most active
    presynaptic units are written, a sparse-coding variant)."""
    W = LAYERS[layer].mlp.down_proj.weight
    for k, v in zip(K, V):
        if Ci is None:
            kt = k
        elif isinstance(Ci, int):
            kt = torch.zeros_like(k)
            ix = k.abs().topk(Ci).indices
            kt[ix] = k[ix]
        else:
            kt = Ci @ k
        err = v - W.float() @ k
        W += (torch.outer(err, kt) / torch.dot(k, kt)).to(W.dtype)


def dev():
    devF = FA.make(tok, 40, seed=999, exclude=set(f['subj'] for f in neigh))
    for layer in [int(x) for x in A.get('layers', '3,6,9,12,15,18').split(',')]:
        Wsave = LAYERS[layer].mlp.down_proj.weight.detach().clone()
        for site, value, g, s in [('last', 'icl', 1.0, 0), ('last', 'icl', 2.0, 0), ('last', 'unemb', 0, 0.5),
                                  ('last', 'unemb', 0, 2.0), ('subj', 'icl', 1.0, 0), ('subj', 'icl', 3.0, 0)]:
            items = items_for(devF, layer, site, value, g, s)
            for cov in ('I', 'C'):
                Ci = cov_inv(layer) if cov == 'C' else None
                res = {f: [] for f in ('rewrite', 'para', 'qa')}
                nsame, nbleed = [], []
                for f, (K, V) in zip(devF, items):
                    write(layer, K, V, Ci)
                    for fm in res:
                        res[fm].append(eval_facts([f], fm))
                    if len(nsame) < 10:
                        ns = neigh_stats(); nsame.append(ns['neigh_same']); nbleed.append(ns['neigh_bleed'])
                    LAYERS[layer].mlp.down_proj.weight.copy_(Wsave)
                log(dict(mode='dev', layer=layer, site=site, value=value, g=g, s=s, cov=cov,
                         **{fm: round(float(np.mean(v)), 3) for fm, v in res.items()},
                         neigh_same=round(float(np.mean(nsame)), 3), neigh_bleed=round(float(np.mean(nbleed)), 3)))


CKPT = [10, 30, 100, 300, 1000, 3000, 10000]


def checkpoint(F, n, extra):
    r = dict(n=n, new_rewrite=eval_facts(F[max(0, n - 100):n]), new_para=eval_facts(F[max(0, n - 100):n], 'para'),
             new_qa=eval_facts(F[max(0, n - 100):n], 'qa'), first100=eval_facts(F[:min(100, n)]),
             all=eval_facts(F[:n]), **neigh_stats(), **probes.all(model), **extra)
    log(r)


def seq():
    layer, site, value = int(A['layer']), A.get('site', 'last'), A.get('value', 'icl')
    g, s, cov, N = float(A.get('g', 1)), float(A.get('s', 1)), A.get('cov', 'I'), int(A.get('n', 10000))
    F = FA.make(tok, N, seed=int(A.get('seed', 1)), exclude=set(f['subj'] for f in neigh))
    t = time.time()
    items = items_for(F, layer, site, value, g, s)
    t_items = time.time() - t
    Ci = cov_inv(layer) if cov == 'C' else (int(cov[1:]) if cov.startswith('S') else None)
    base = dict(mode='seq', tag=A.get('tag', ''), layer=layer, site=site, value=value, g=g, s=s, cov=cov)
    log(dict(**base, n=0, **neigh_stats(), **probes.all(model)))
    done, tw = 0, 0.0
    for c in CKPT:
        if c > N:
            break
        t = time.time()
        for K, V in items[done:c]:
            write(layer, K, V, Ci)
        torch.cuda.synchronize()
        tw += time.time() - t
        done = c
        checkpoint(F, c, dict(base, ms_per_fact_write=round(1000 * tw / done, 3),
                              ms_per_fact_items=round(1000 * t_items / len(F), 3)))


def ft():
    """sequential fine-tuning baseline: per fact, Adam steps on CE(name | rewrite prompt) until both tokens are argmax
    (max `steps`), parameters = down_proj of `layer` (L) or every decoder layer (all)."""
    layer, which, lr, N = int(A.get('layer', 6)), A.get('params', 'L'), float(A['lr']), int(A.get('n', 1000))
    steps = int(A.get('steps', 10))
    F = FA.make(tok, N, seed=int(A.get('seed', 1)), exclude=set(f['subj'] for f in neigh))
    params = [LAYERS[layer].mlp.down_proj.weight] if which == 'L' else [p for l in LAYERS for p in l.parameters()]
    for p in params:
        p.requires_grad_(True)
    opt = torch.optim.Adam(params, lr=lr)
    base = dict(mode='ft', tag=A.get('tag', ''), params=which, layer=layer, lr=lr, steps=steps)
    log(dict(**base, n=0, **neigh_stats(), **probes.all(model)))
    done, tw, nsteps = 0, 0.0, 0
    for c in CKPT:
        if c > N:
            break
        t = time.time()
        for f in F[done:c]:
            x = torch.tensor([f['p']['rewrite'] + f['nt']], device=C.DEV)
            n0 = len(f['p']['rewrite'])
            for _ in range(steps):
                logits = model(x).logits[0, n0 - 1:n0 + 1].float()
                tgt = torch.tensor(f['nt'], device=C.DEV)
                if bool((logits.argmax(-1) == tgt).all()):
                    break
                loss = torch.nn.functional.cross_entropy(logits, tgt)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                nsteps += 1
        torch.cuda.synchronize()
        tw += time.time() - t
        done = c
        model.eval()
        checkpoint(F, c, dict(base, ms_per_fact=round(1000 * tw / done, 1), steps_per_fact=round(nsteps / done, 2),
                              gpu_gb=C.gpu_mem()))


# ------------------------------------------------------------------------------------------ episodic store (no weight change)
@torch.no_grad()
def capture_kz(seqs, layer, pos, bs=128):
    """key (down_proj input of `layer`) and the frozen model's argmax at position pos[i]."""
    cap = {}
    h1 = LAYERS[layer].mlp.down_proj.register_forward_pre_hook(lambda m, i: cap.__setitem__('k', i[0]))
    K, Z = [], []
    try:
        for b in range(0, len(seqs), bs):
            X, att = pad(seqs[b:b + bs])
            hs = model.model(input_ids=X, attention_mask=att).last_hidden_state
            ix = torch.arange(X.shape[0], device=C.DEV)
            pp = torch.tensor(pos[b:b + bs], device=C.DEV)
            K.append(cap['k'][ix, pp].float())
            Z.append(model.lm_head(hs[ix, pp]).argmax(-1))
    finally:
        h1.remove()
    return torch.cat(K), torch.cat(Z)


def knn():
    """one-shot key -> next-token store on the same keys as the R3 writes (layer `layer`, predicting position);
    the network is frozen; retrieval overrides the prediction when the best cosine >= theta."""
    layer, N = int(A.get('layer', 18)), int(A.get('n', 10000))
    thetas = [float(x) for x in A.get('thetas', '0.8,0.9,0.95').split(',')]
    F = FA.make(tok, N, seed=int(A.get('seed', 1)), exclude=set(f['subj'] for f in neigh))
    # centring vector: mean key over ordinary text
    seqs = [C.enc(tok, ''.join(C.dialog_lines(d)))[:96] for d in dialogs[:200]]
    cap = {}
    hk = LAYERS[layer].mlp.down_proj.register_forward_pre_hook(lambda m, i: cap.__setitem__('k', i[0]))
    ks = []
    with torch.no_grad():
        for b in range(0, len(seqs), 64):
            X, att = pad(seqs[b:b + 64])
            model.model(input_ids=X, attention_mask=att)
            ks.append(cap['k'][att.bool()].float())
    hk.remove()
    mu = torch.cat(ks).mean(0)

    def norm(k):
        k = k - mu
        return k / k.norm(dim=-1, keepdim=True)
    ctx, pos, tgt = [], [], []
    for f in F:
        for j in range(2):
            c = f['p']['rewrite'] + f['nt'][:j]
            ctx.append(c); pos.append(len(c) - 1); tgt.append(f['nt'][j])
    t = time.time()
    SK, _ = capture_kz(ctx, layer, pos)
    SK = norm(SK)
    ST = torch.tensor(tgt, device=C.DEV)
    t_items = time.time() - t

    def acc(FF, form, n, th):
        if not FF:
            return None
        sq, ps = [], []
        for f in FF:
            s_ = f['p'][form] + [f['nt'][0]]
            sq += [s_, s_]; ps += [len(s_) - 2, len(s_) - 1]
        k, z = capture_kz(sq, layer, ps)
        sim = norm(k) @ SK[:2 * n].T
        best, ix = sim.max(-1)
        pred = torch.where(best >= th, ST[:2 * n][ix], z)
        want = torch.tensor([t for f in FF for t in f['nt']], device=C.DEV)
        ok = (pred == want).view(-1, 2).all(-1).float()
        return round(float(ok.mean()), 4)

    nseq = [f['p']['rewrite'] for f in neigh]
    nk, nz = capture_kz(nseq, layer, [len(s_) - 1 for s_ in nseq])
    nsim = norm(nk) @ SK.T
    base = dict(mode='knn', layer=layer, ms_per_fact_items=round(1000 * t_items / len(F), 3))
    for n in [c for c in CKPT if c <= N]:
        for th in thetas:
            best, ix = nsim[:, :2 * n].max(-1)
            npred = torch.where(best >= th, ST[:2 * n][ix], nz)
            log(dict(base, n=n, theta=th, new_rewrite=acc(F[max(0, n - 100):n], 'rewrite', n, th),
                     new_para=acc(F[max(0, n - 100):n], 'para', n, th), new_qa=acc(F[max(0, n - 100):n], 'qa', n, th),
                     first100=acc(F[:min(100, n)], 'rewrite', n, th),
                     neigh_same=round(float((npred == neigh0).float().mean()), 4),
                     neigh_bleed=round(float(np.mean([int(t_) in NAME_TOK for t_ in npred.tolist()])), 4)))
    # false retrievals on ordinary text (corpus probe positions), at the largest n
    X, Mk, att = probes.X, probes.M, probes.att
    fired = {th: 0 for th in thetas}
    tot = 0
    with torch.no_grad():
        for b in range(0, X.shape[0], 10):
            hk = LAYERS[layer].mlp.down_proj.register_forward_pre_hook(lambda m, i: cap.__setitem__('k', i[0]))
            model.model(input_ids=X[b:b + 10], attention_mask=att[b:b + 10])
            hk.remove()
            k = norm(cap['k'][Mk[b:b + 10]].float())
            best = (k @ SK.T).max(-1).values
            for th in thetas:
                fired[th] += int((best >= th).sum())
            tot += k.shape[0]
    log(dict(base, n=N, corpus_false_fire={str(th): round(fired[th] / tot, 5) for th in thetas}, corpus_positions=tot))


if __name__ == '__main__':
    log(dict(info='start', mode=MODE, args=A, n_know=len(probes.know), base=probes.all(model)))
    {'dev': dev, 'seq': seq, 'ft': ft, 'knn': knn}[MODE]()
    log(dict(info='end', gpu_gb=C.gpu_mem()))
