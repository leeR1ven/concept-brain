# -*- coding: utf-8 -*-
"""Test 1: an entity-keyed, DG-style sparse episodic store next to a FROZEN Qwen2.5-0.5B.

One fact = one heard statement '<person>的<rel>叫<name>。' stored in one shot as four keys + two values:
  person key  : entity-layer residual at the person's last token
  subject key : entity-layer residual at the subject's last token ('...的小狗')
  site keys   : site-layer (18) residual at the two positions that predict the name tokens  -> values t1, t2
Each key is standardised (corpus mean/std per dimension), then either DG-coded (random expansion to N units +
k-WTA, binary; match = overlap / k) or kept dense (unit vector; match = cosine).
Retrieval at a position: candidate facts need person match >= tp AND subject match >= te somewhere in the text so
far; the best candidate fires if its site key j matches the current position >= ts, and then its token j replaces
the model's prediction.  The network is never changed.
usage: python store.py mode=dev | mode=main layer=.. coder=dg|dense tp=.. te=.. ts=.. n=10000
"""
import os, sys, json, time, math, random
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import torch
import common as C
import facts as FA

A = dict(a.split('=', 1) for a in sys.argv[1:])
MODE = A.get('mode', 'dev')
OUT = A.get('out', os.path.join(HERE, 'out', 'store.jsonl'))
os.makedirs(os.path.dirname(OUT), exist_ok=True)
torch.cuda.set_per_process_memory_fraction(float(A.get('memfrac', 0.22)))
T0 = time.time()
tok, model = C.load_model()
for p in model.parameters():
    p.requires_grad_(False)
LAYERS = model.model.layers
PAD = tok.pad_token_id if tok.pad_token_id is not None else 0
LS = int(A.get('site_layer', 18))
NDG, KDG = int(A.get('N', 16384)), int(A.get('k', 32))


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
def forward(seqs, layers):
    """residual outputs of `layers` (dict layer -> (B,T,d)), model argmax (B,T), attention mask."""
    cap = {}
    hs = [LAYERS[l].register_forward_hook(lambda m, i, o, l=l: cap.__setitem__(l, o[0] if isinstance(o, tuple) else o))
          for l in layers]
    try:
        X, att = pad(seqs)
        last = model.model(input_ids=X, attention_mask=att).last_hidden_state
        z = model.lm_head(last).argmax(-1)
    finally:
        for h in hs:
            h.remove()
    return {l: cap[l].float() for l in layers}, z, att


dialogs = C.load_dialogs()
probes = C.Probes(tok, model, dialogs[-400:], n=150)
neigh = FA.make(tok, 300, seed=4242)
STATS = {}


def stats(layer):
    if layer not in STATS:
        seqs = [C.enc(tok, ''.join(C.dialog_lines(d)))[:96] for d in dialogs[:300]]
        s1 = s2 = None
        n = 0
        for b in range(0, len(seqs), 32):
            H, _, att = forward(seqs[b:b + 32], [layer])
            m = att.bool().clone()
            m[:, 0] = False                                            # skip the attention-sink token
            h = H[layer][m].double()
            s1 = h.sum(0) if s1 is None else s1 + h.sum(0)
            s2 = (h * h).sum(0) if s2 is None else s2 + (h * h).sum(0)
            n += h.shape[0]
        mu = s1 / n
        sd = (s2 / n - mu * mu).clamp_min(1e-8).sqrt()
        STATS[layer] = (mu.float(), sd.float())
    return STATS[layer]


class Coder:
    def __init__(self, layer, dense, seed):
        self.mu, self.sd = stats(layer)
        self.dense = dense
        g = torch.Generator(device='cpu').manual_seed(seed)
        self.R = None if dense else torch.randn(NDG, self.mu.numel(), generator=g).to(C.DEV)

    def __call__(self, h):
        z = ((h - self.mu) / self.sd).clamp(-6, 6)
        z = z / z.norm(dim=-1, keepdim=True)
        if self.dense:
            return z.half()
        a = z @ self.R.T
        out = torch.zeros(*a.shape, dtype=torch.half, device=C.DEV)
        return out.scatter_(-1, a.topk(KDG, -1).indices, 1.0)

    def sim(self, q, s):                        # q (..., D), s (M, D) -> (..., M) in [0, 1]
        r = q @ s.T
        return r.float() / KDG if not self.dense else r.float()


def person_last(f):
    pi, si = C.enc(tok, f['person']), C.enc(tok, f['subj'])
    return len(pi) - 1 if si[:len(pi)] == pi and len(pi) < len(si) else f['subj_last']


class Store:
    def __init__(self, layer, dense):
        self.le, self.dense = layer, dense
        self.ce = Coder(layer, dense, 11)
        self.cs = Coder(LS, dense, 12)

    @torch.no_grad()
    def build(self, F, bs=64):
        """one-shot write of every fact (keys from the heard statement itself)."""
        P, E, S0, S1, V = [], [], [], [], []
        for b in range(0, len(F), bs):
            FF = F[b:b + bs]
            seqs = [f['p']['rewrite'] + f['nt'] for f in FF]
            H, _, _ = forward(seqs, sorted({self.le, LS}))
            ix = torch.arange(len(FF), device=C.DEV)
            pl = torch.tensor([person_last(f) for f in FF], device=C.DEV)
            sl = torch.tensor([f['subj_last'] for f in FF], device=C.DEV)
            s0 = torch.tensor([len(f['p']['rewrite']) - 1 for f in FF], device=C.DEV)
            P.append(self.ce(H[self.le][ix, pl])); E.append(self.ce(H[self.le][ix, sl]))
            S0.append(self.cs(H[LS][ix, s0])); S1.append(self.cs(H[LS][ix, s0 + 1]))
            V.append(torch.tensor([f['nt'] for f in FF], device=C.DEV))
        self.P, self.E, self.S = torch.cat(P), torch.cat(E), [torch.cat(S0), torch.cat(S1)]
        self.V = torch.cat(V)

    @torch.no_grad()
    def analyze(self, seqs, evals, n, topc=5, bs=32):
        """evals[i] = list of (pos, j) (j = -1: either site). For every eval point returns the model argmax and the
        top-`topc` candidate facts with their person / subject / site matches (entity matches are prefix maxima)."""
        out = []
        for b in range(0, len(seqs), bs):
            S = seqs[b:b + bs]
            H, z, att = forward(S, sorted({self.le, LS}))
            m = att.bool()
            qe = self.ce(H[self.le])
            pe = self.ce.sim(qe, self.P[:n]) * m[..., None]                         # (B,T,n)
            ee = self.ce.sim(qe, self.E[:n]) * m[..., None]
            del qe
            pe = torch.cummax(pe, 1).values
            ee = torch.cummax(ee, 1).values
            sc = self.cs(H[LS])                                                     # (B,T,D)
            for i in range(len(S)):
                for pos, j in evals[b + i]:
                    tot = pe[i, pos] + ee[i, pos]
                    k = min(topc, n)
                    cand = tot.topk(k).indices
                    s0 = self.cs.sim(sc[i, pos][None], self.S[0][:n][cand])[0]
                    s1 = self.cs.sim(sc[i, pos][None], self.S[1][:n][cand])[0]
                    out.append(dict(z=int(z[i, pos]), cand=cand.tolist(), P=pe[i, pos, cand].tolist(),
                                    E=ee[i, pos, cand].tolist(), s0=s0.tolist(), s1=s1.tolist(), j=j))
        return out

    def decide(self, r, tp, te, ts):
        """-> (prediction, fired)"""
        for c, p, e, a0, a1 in zip(r['cand'], r['P'], r['E'], r['s0'], r['s1']):
            if p >= tp and e >= te:
                if r['j'] == 0 and a0 >= ts:
                    return int(self.V[c, 0]), True
                if r['j'] == 1 and a1 >= ts:
                    return int(self.V[c, 1]), True
                if r['j'] == -1 and max(a0, a1) >= ts:
                    return -1, True
                return r['z'], False                 # best entity candidate, but this is not its answer position
        return r['z'], False


def fact_queries(F, form):
    seqs = [f['p'][form] + [f['nt'][0]] for f in F]
    return seqs, [[(len(s) - 2, 0), (len(s) - 1, 1)] for s in seqs]


def other_rel(F, written):
    out = []
    r = random.Random(5)
    for f in F:
        for _ in range(20):
            rel = r.choice(FA.REL)
            if rel != f['rel'] and f"{f['person']}的{rel}" not in written:
                out.append(C.enc(tok, f"{f['person']}的{rel}叫"))
                break
    return out


def corpus_queries():
    seqs, ev = [], []
    for i in range(probes.X.shape[0]):
        L = int(probes.att[i].sum())
        seqs.append(probes.X[i, :L].tolist())
        ev.append([(t, -1) for t in range(1, L)])
    return seqs, ev


CQ = None


def measure(st, F, n, cfgs, written):
    """returns {cfg: metrics} for the store holding F[:n]."""
    global CQ
    new = F[max(0, n - 100):n]
    first = F[:min(100, n)]
    res = {}
    A_ = {}
    for name, FF, form in [('new_rewrite', new, 'rewrite'), ('new_para', new, 'para'), ('new_qa', new, 'qa'),
                           ('first100', first, 'rewrite')]:
        s, e = fact_queries(FF, form)
        A_[name] = (st.analyze(s, e, n), [t for f in FF for t in f['nt']])
    ns = [f['p']['rewrite'] for f in neigh]
    A_['leak_other'] = st.analyze(ns, [[(len(s) - 1, 0)] for s in ns], n)
    os_ = other_rel(F[:min(n, 300)], written)
    A_['leak_samep'] = st.analyze(os_, [[(len(s) - 1, 0)] for s in os_], n)
    if CQ is None:
        CQ = corpus_queries()
    A_['corpus'] = st.analyze(CQ[0], CQ[1], n)
    for cfg in cfgs:
        tp, te, ts = cfg
        m = {}
        for name in ('new_rewrite', 'new_para', 'new_qa', 'first100'):
            rs, want = A_[name]
            pred = [st.decide(r, tp, te, ts)[0] for r in rs]
            ok = np.array([int(a == b) for a, b in zip(pred, want)]).reshape(-1, 2).all(1)
            m[name] = round(float(ok.mean()), 4)
        m['leak_other'] = round(float(np.mean([st.decide(r, tp, te, ts)[1] for r in A_['leak_other']])), 4)
        m['leak_samep'] = round(float(np.mean([st.decide(r, tp, te, ts)[1] for r in A_['leak_samep']])), 4)
        m['corpus_fire'] = round(float(np.mean([st.decide(r, tp, te, ts)[1] for r in A_['corpus']])), 5)
        res[cfg] = m
    return res


def dev():
    devF = FA.make(tok, 1000, seed=777, exclude=set(f['subj'] for f in neigh))
    written = set(f['subj'] for f in devF)
    grid_dg = [(a, b, c) for a in (0.0, 0.2, 0.3, 0.4, 0.5, 0.6) for b in (0.2, 0.3, 0.4, 0.5, 0.6) for c in (0.1, 0.2, 0.3, 0.4)]
    grid_dn = [(a, b, c) for a in (0.0, 0.5, 0.6, 0.7, 0.8, 0.9) for b in (0.5, 0.6, 0.7, 0.8, 0.9) for c in (0.4, 0.5, 0.6, 0.7)]
    for layer in [int(x) for x in A.get('layers', '4,8,12,16').split(',')]:
        for dense in (False, True):
            st = Store(layer, dense)
            st.build(devF)
            res = measure(st, devF, len(devF), grid_dn if dense else grid_dg, written)
            for cfg, m in res.items():
                log(dict(mode='dev', layer=layer, coder='dense' if dense else 'dg', tp=cfg[0], te=cfg[1], ts=cfg[2], **m))


def score(r):
    pen = 10 * (max(0, r['leak_other'] - 0.02) + max(0, r['leak_samep'] - 0.05) + max(0, r['corpus_fire'] - 0.001)
                + max(0, 0.95 - r['new_rewrite']))
    return r['new_para'] + r['new_qa'] - pen


def main():
    if 'layer' not in A:                          # pick from the dev log
        rows = [json.loads(l) for l in open(A.get('devlog', OUT)) if '"mode": "dev"' in l and '"tp"' in l]
        coder = A.get('coder', 'dg')
        b = max([r for r in rows if r['coder'] == coder], key=score)
        layer, tp, te, ts = b['layer'], b['tp'], b['te'], b['ts']
        log(dict(info='picked', coder=coder, layer=layer, tp=tp, te=te, ts=ts, dev=b))
    else:
        coder, layer = A.get('coder', 'dg'), int(A['layer'])
        tp, te, ts = float(A['tp']), float(A['te']), float(A['ts'])
    N = int(A.get('n', 10000))
    F = FA.make(tok, N, seed=int(A.get('seed', 1)), exclude=set(f['subj'] for f in neigh))
    written = set(f['subj'] for f in F)
    st = Store(layer, coder == 'dense')
    t = time.time()
    st.build(F)
    torch.cuda.synchronize()
    ms = 1000 * (time.time() - t) / N
    for n in [c for c in (10, 30, 100, 300, 1000, 3000, 10000) if c <= N]:
        m = measure(st, F, n, [(tp, te, ts)], written)[(tp, te, ts)]
        extra = {}
        if n in (1000, 10000):
            s, e = fact_queries(F[:n], 'rewrite')
            rs = st.analyze(s, e, n)
            pred = [st.decide(r, tp, te, ts)[0] for r in rs]
            want = [t_ for f in F[:n] for t_ in f['nt']]
            extra['all'] = round(float(np.array([int(a == b) for a, b in zip(pred, want)]).reshape(-1, 2).all(1).mean()), 4)
        log(dict(mode='main', coder=coder, layer=layer, tp=tp, te=te, ts=ts, n=n, ms_per_fact_write=round(ms, 3),
                 bytes_per_fact=(4 * KDG * 2 if coder == 'dg' else 4 * 896 * 2), **m, **extra))


if __name__ == '__main__':
    log(dict(info='start', mode=MODE, args=A, base=probes.all(model)))
    {'dev': dev, 'main': main}[MODE]()
    log(dict(info='end', gpu_gb=round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)))
