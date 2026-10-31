# -*- coding: utf-8 -*-
"""R2: local learning rules on the slow weights of a pretrained LLM (Qwen2.5-0.5B), vs back-propagation.

Nodes = the residual stream after every decoder block (x_0 = embeddings, fixed; x_L feeds norm + tied head).
  BP      : the true gradient.
  PC      : predictive coding / inference learning. Energy F = sum_l 1/2|x_l - f_l(x_{l-1})|^2 + beta*CE(x_L).
            Inference: x_l -= gamma dF/dx_l for T steps (needs only block l's error and the next block's
            Jacobian-transpose of ITS error: local message passing); weights: dW_l = (eta/beta) J_W f_l^T eps_l
            (block-local). With the fixed-prediction assumption (PC-FPA, gamma=1, T=L) it equals BP exactly
            (Song et al. 2020; Millidge et al. 2022).
  LL      : layer-local delta rule on each block's MLP output weights W_down only, no messages between layers:
            dW_down^l = eta * e_l a_l^T, e_l = -dCE(head(norm(x_l)))/dx_l (a fixed local read-out per layer).
  DFA     : the same delta rule with a random fixed feedback e_l = R_l W_U^T (y - p) (direct feedback alignment).
usage: python r2.py mode=cos | mode=online method=bp|pc|bpdown|ll|adam lr=.. [T=48 gamma=0.5 beta=..]
"""
import os, sys, json, time, math, inspect, random
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import torch
import torch.nn.functional as F
import common as C

A = dict(a.split('=', 1) for a in sys.argv[1:])
MODE = A.get('mode', 'cos')
OUT = A.get('out', os.path.join(HERE, 'out', 'r2.jsonl'))
os.makedirs(os.path.dirname(OUT), exist_ok=True)
torch.cuda.set_per_process_memory_fraction(float(A.get('memfrac', 0.22)))
torch.manual_seed(0)
T0 = time.time()
tok, model = C.load_model()
for p in model.parameters():
    p.requires_grad_(False)
LAYERS = model.model.layers
NL = len(LAYERS)
_SIG = inspect.signature(LAYERS[0].forward).parameters
IDX_DOWN = [i for i, (n, _) in enumerate(LAYERS[0].named_parameters()) if n == 'mlp.down_proj.weight'][0]


def log(d):
    d['t'] = round(time.time() - T0, 1)
    print(json.dumps(d, ensure_ascii=False), flush=True)
    with open(OUT, 'a') as f:
        f.write(json.dumps(d, ensure_ascii=False) + '\n')


class Seq:
    """one unpadded sequence: embeddings + rotary + per-block call (causal sdpa, no mask)."""

    def __init__(self, ids):
        self.ids = torch.tensor([ids], device=C.DEV)
        self.x0 = model.model.embed_tokens(self.ids).detach()
        self.pos = torch.arange(len(ids), device=C.DEV)[None]
        self.kw = dict(position_ids=self.pos, attention_mask=None)
        if 'position_embeddings' in _SIG:
            self.kw['position_embeddings'] = model.model.rotary_emb(self.x0, self.pos)
        self.tgt = self.ids[0, 1:]

    def f(self, l, x):
        o = LAYERS[l](x, **self.kw)
        return o[0] if isinstance(o, tuple) else o

    def loss(self, xL):
        z = model.lm_head(model.model.norm(xL))[0, :-1].float()
        return F.cross_entropy(z, self.tgt, reduction='mean')


def block_params(l, which):
    if which == 'down':
        return [LAYERS[l].mlp.down_proj.weight]
    return [p for p in LAYERS[l].parameters()]


def bp_grads(s, which='all'):
    ps = [p for l in range(NL) for p in block_params(l, which)]
    for p in ps:
        p.requires_grad_(True)
    x = s.x0
    for l in range(NL):
        x = s.f(l, x)
    loss = s.loss(x)
    g = torch.autograd.grad(loss, ps)
    for p in ps:
        p.requires_grad_(False)
    out, i = [], 0
    for l in range(NL):
        n = len(block_params(l, which))
        out.append([-gi for gi in g[i:i + n]])      # descent direction
        i += n
    return out, float(loss)


def pc_update(s, T, gamma, beta, which='all', fpa=False):
    """returns per-block descent directions (scaled by 1/beta, comparable to -grad) and the final inference state."""
    with torch.no_grad():
        xbar = [s.x0]
        for l in range(NL):
            xbar.append(s.f(l, xbar[-1]))
    x = [t.clone() for t in xbar]
    for _ in range(T):
        mus, gin = [], []
        for l in range(NL):                       # mu_{l+1} = f_l(x_l) ; Jacobians at the current x (or xbar: FPA)
            xi = (xbar[l] if fpa else x[l]).detach().requires_grad_(l > 0)
            mus.append((xi, s.f(l, xi)))
        eps = [None] + [x[l + 1] - (xbar[l + 1] if fpa else mus[l][1].detach()) for l in range(NL)]
        xl = (xbar[NL] if fpa else x[NL]).detach().requires_grad_(True)
        gL = torch.autograd.grad(beta * s.loss(xl), xl)[0]
        newx = list(x)
        for l in range(1, NL + 1):
            d = eps[l].clone()
            if l < NL:
                xi, mu = mus[l]
                jt = torch.autograd.grad(mu, xi, grad_outputs=eps[l + 1], retain_graph=False)[0]
                d = d - jt
            else:
                d = d + gL
            newx[l] = (x[l] - gamma * d).detach()
        x = newx
    out = []
    for l in range(NL):
        ps = block_params(l, which)
        for p in ps:
            p.requires_grad_(True)
        xin = (xbar[l] if fpa else x[l]).detach()
        mu = s.f(l, xin)
        e = x[l + 1] - (xbar[l + 1] if fpa else mu.detach())
        g = torch.autograd.grad(mu, ps, grad_outputs=e)
        for p in ps:
            p.requires_grad_(False)
        out.append([gi / beta for gi in g])
    return out


def local_delta(s, kind='ll', R=None):
    """layer-local delta rule on W_down: dW = e_l a_l^T (summed over positions, / n positions)."""
    cap = {}
    hooks = [LAYERS[l].mlp.down_proj.register_forward_pre_hook(
        lambda m, i, l=l: cap.__setitem__(l, i[0].detach())) for l in range(NL)]
    xs = []
    with torch.no_grad():
        x = s.x0
        for l in range(NL):
            x = s.f(l, x)
            xs.append(x)
    for h in hooks:
        h.remove()
    n = s.tgt.shape[0]
    out = []
    with torch.no_grad():
        zf = model.lm_head(model.model.norm(xs[-1]))[0, :-1].float()
        err_out = F.one_hot(s.tgt, zf.shape[-1]).float() - zf.softmax(-1)           # y - p   (n, V)
    for l in range(NL):
        if kind == 'll':
            xl = xs[l].detach().requires_grad_(True)
            e = -torch.autograd.grad(s.loss(xl), xl)[0][0]                           # (T, d), already / n
        else:
            with torch.no_grad():
                e = torch.zeros_like(xs[l][0])
                e[:-1] = (err_out @ model.lm_head.weight.float()) @ R[l].T / n
        a = cap[l][0]
        out.append([e.T @ a])                                                       # (d, d_ff) like W_down
    return out


def cos(a, b):
    a = torch.cat([t.flatten().float() for t in a]); b = torch.cat([t.flatten().float() for t in b])
    return float(a @ b / (a.norm() * b.norm() + 1e-30))


def mode_cos():
    dialogs = C.load_dialogs()
    facts = C.make_facts(tok, 0, C.SUBJ[:8])
    lines = [f['line'] for f in facts] + [C.dialog_lines(d)[0] for d in dialogs[:8]]
    nl = int(A.get('nl', 16))
    lines = (lines[:nl // 2] + lines[8:8 + nl - nl // 2]) if nl < 16 else lines
    g = torch.Generator(device='cpu').manual_seed(1)
    d = model.config.hidden_size
    R = [torch.linalg.qr(torch.randn(d, d, generator=g))[0].to(C.DEV) for _ in range(NL)]
    cfgs = [('pc', dict(T=T, gamma=gm, beta=b)) for T in (6, 24, 48, 96) for gm in (0.5,) for b in (0.1,)] + \
           [('pc', dict(T=96, gamma=0.2, beta=0.1)), ('pc', dict(T=48, gamma=0.5, beta=1.0)),
            ('pc', dict(T=48, gamma=0.5, beta=0.01)), ('pcfpa', dict(T=NL, gamma=1.0, beta=1.0))]
    if 'pcs' in A:                       # e.g. pcs=48:0.1:0.1,192:0.03:0.1
        cfgs = [('pc', dict(T=int(a), gamma=float(b), beta=float(c))) for a, b, c in
                (x.split(':') for x in A['pcs'].split(','))]
    rows = {}
    for text in lines:
        s = Seq(C.enc(tok, text))
        gb_all, _ = bp_grads(s, 'all')
        gb_down, _ = bp_grads(s, 'down')
        for name, kw in cfgs:
            t = time.time()
            u = pc_update(s, kw['T'], kw['gamma'], kw['beta'], 'all', fpa=(name == 'pcfpa'))
            key = f"{name}_T{kw['T']}_g{kw['gamma']}_b{kw['beta']}"
            rows.setdefault(key, []).append(([cos(u[l], gb_all[l]) for l in range(NL)],
                                             [cos([u[l][IDX_DOWN]], gb_down[l]) for l in range(NL)],
                                             time.time() - t,
                                             [float(torch.cat([z.flatten() for z in u[l]]).norm() / torch.cat([z.flatten() for z in gb_all[l]]).norm()) for l in range(NL)]))
        for kind in (('ll', 'dfa') if A.get('local', '1') == '1' else ()):
            t = time.time()
            u = local_delta(s, kind, R)
            rows.setdefault(kind, []).append((None, [cos(u[l], gb_down[l]) for l in range(NL)], time.time() - t, None))
    for key, v in rows.items():
        r = dict(mode='cos', method=key, n=len(v), secs_per_line=round(float(np.mean([x[2] for x in v])), 3))
        if v[0][0] is not None:
            r['cos_block'] = [round(float(np.mean([x[0][l] for x in v])), 3) for l in range(NL)]
            r['norm_ratio'] = [round(float(np.mean([x[3][l] for x in v])), 3) for l in range(NL)]
        r['cos_down'] = [round(float(np.mean([x[1][l] for x in v])), 3) for l in range(NL)]
        r['cos_down_mean'] = round(float(np.mean(r['cos_down'])), 3)
        if 'cos_block' in r:
            r['cos_block_mean'] = round(float(np.mean(r['cos_block'])), 3)
        log(r)


# ------------------------------------------------------------------------------------------ online stream
def mode_online():
    method, lr = A.get('method', 'bp'), float(A['lr'])
    T, gamma, beta = int(A.get('T', 48)), float(A.get('gamma', 0.5)), float(A.get('beta', 0.1))
    nf, reps = int(A.get('nf', 30)), int(A.get('reps', 3))
    dialogs = C.load_dialogs()
    probes = C.Probes(tok, model, dialogs[-400:], n=150)
    facts = C.make_facts(tok, int(A.get('seed', 0)), C.SUBJ[:nf])
    rng = random.Random(int(A.get('seed', 0)) + 5)
    chat = [l for d in dialogs[:3000] for l in C.dialog_lines(d)]
    rng.shuffle(chat)
    ctx_line = '乙：哈哈好的\n'
    which = 'down' if method in ('bpdown', 'll', 'dfa') else 'all'
    params = [p for l in range(NL) for p in block_params(l, which)]
    opt = None
    if method == 'adam':
        for p in params:
            p.requires_grad_(True)
        opt = torch.optim.Adam(params, lr=lr)
    d = model.config.hidden_size
    g = torch.Generator(device='cpu').manual_seed(1)
    R = [torch.linalg.qr(torch.randn(d, d, generator=g))[0].to(C.DEV) for _ in range(NL)] if method == 'dfa' else None

    @torch.no_grad()
    def fact_acc(form):
        ok = []
        for f in facts:
            p = C.enc(tok, C.forms(f, ctx_line)[form])
            x = torch.tensor([p + f['ntok'][:1]], device=C.DEV)
            z = model(x).logits[0, -2:].argmax(-1).tolist()
            ok.append(int(z == f['ntok']))
        return round(float(np.mean(ok)), 4)

    def evaluate(tag, extra=None):
        model.eval()
        log(dict(mode='online', method=method, lr=lr, T=T if method == 'pc' else None, beta=beta if method == 'pc' else None,
                 tag=tag, cloze=fact_acc('cloze'), qa=fact_acc('qa'), **probes.all(model), **(extra or {})))

    def update(text):
        s = Seq(C.enc(tok, text))
        if method == 'adam':
            x = s.x0
            for l in range(NL):
                x = s.f(l, x)
            loss = s.loss(x)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            return
        if method in ('bp', 'bpdown'):
            u, _ = bp_grads(s, which)
        elif method == 'pc':
            u = pc_update(s, T, gamma, beta, 'all')
        else:
            u = local_delta(s, method, R)
        with torch.no_grad():
            for l in range(NL):
                for p, du in zip(block_params(l, which), u[l]):
                    p += lr * du.to(p.dtype)

    evaluate('p0')
    ci, nup, tu = 0, 0, 0.0
    for r in range(reps):
        order = list(range(nf))
        rng.shuffle(order)
        for i in order:
            for text in [facts[i]['line'], chat[ci]]:
                t = time.time()
                update(text)
                torch.cuda.synchronize()
                tu += time.time() - t
                nup += 1
            ci += 1
        if r in (0, reps - 1):
            evaluate(f'p{r + 1}', dict(ms_per_update=round(1000 * tu / nup, 1), gpu_gb=C.gpu_mem()))
    for j in range(int(A.get('delay', 100))):
        update(chat[ci]); ci += 1
    evaluate(f'p{reps}_delay', dict(ms_per_update=round(1000 * tu / nup, 1)))


if __name__ == '__main__':
    log(dict(info='start', mode=MODE, args=A))
    {'cos': mode_cos, 'online': mode_online}[MODE]()
    log(dict(info='end', gpu_gb=C.gpu_mem()))
