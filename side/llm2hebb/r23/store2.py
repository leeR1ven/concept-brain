# -*- coding: utf-8 -*-
"""Test 1, second version: entity-keyed DG store with TOKEN-level entity keys (several keys per fact).

Store v1 (store.py) keyed each fact on the contextual residual at the person's / subject's LAST token. Its keys of the
same subject in different sentences were less alike than keys of different subjects in the same sentence (and the
subject sat at position 0, the attention-sink token). v2:
  * every sequence gets a neutral one-token prefix ('\n'), so no content token is at position 0;
  * entity keys = DG codes (random expansion + k-WTA) of EVERY token of the subject ('王|子涵|的|小狗'), at entity
    layer le (0 = input embeddings, else the residual after block le-1);
  * a fact is recognised when a fraction >= te of its subject-token keys each find a match >= tt somewhere in the text
    so far; it then fires at a position whose layer-18 code matches the fact's site key j >= ts (value = name token j).
The network is frozen.  usage: python store2.py mode=dev | mode=main [le= tt= te= ts=] n=10000
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
OUT = A.get('out', os.path.join(HERE, 'out', 'store2.jsonl'))
os.makedirs(os.path.dirname(OUT), exist_ok=True)
torch.cuda.set_per_process_memory_fraction(float(A.get('memfrac', 0.22)))
T0 = time.time()
tok, model = C.load_model()
for p in model.parameters():
    p.requires_grad_(False)
LAYERS = model.model.layers
PAD = tok.pad_token_id if tok.pad_token_id is not None else 0
LS = 18
NDG, KDG = int(A.get('N', 16384)), int(A.get('k', 32))
PRE = C.enc(tok, '\n')
TT = [0.5, 0.8]


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
    """layers: entity layer indices (0 = embeddings, l = output of block l-1) and 'site' (output of block LS)."""
    cap = {}
    hs = []
    for l in layers:
        mod = model.model.embed_tokens if l == 0 else (LAYERS[LS] if l == 'site' else LAYERS[l - 1])
        hs.append(mod.register_forward_hook(lambda m, i, o, l=l: cap.__setitem__(l, o[0] if isinstance(o, tuple) else o)))
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
        seqs = [PRE + C.enc(tok, ''.join(C.dialog_lines(d)))[:95] for d in dialogs[:300]]
        s1 = s2 = None
        n = 0
        for b in range(0, len(seqs), 32):
            H, _, att = forward(seqs[b:b + 32], [layer])
            m = att.bool().clone(); m[:, 0] = False
            h = H[layer][m].double()
            s1 = h.sum(0) if s1 is None else s1 + h.sum(0)
            s2 = (h * h).sum(0) if s2 is None else s2 + (h * h).sum(0)
            n += h.shape[0]
        mu = s1 / n
        STATS[layer] = (mu.float(), (s2 / n - mu * mu).clamp_min(1e-8).sqrt().float())
    return STATS[layer]


class Coder:
    def __init__(self, layer, seed):
        self.mu, self.sd = stats(layer)
        g = torch.Generator(device='cpu').manual_seed(seed)
        self.R = torch.randn(NDG, self.mu.numel(), generator=g).to(C.DEV)

    def __call__(self, h):
        z = ((h - self.mu) / self.sd).clamp(-6, 6)
        z = z / z.norm(dim=-1, keepdim=True)
        a = z @ self.R.T
        return torch.zeros(*a.shape, dtype=torch.half, device=C.DEV).scatter_(-1, a.topk(KDG, -1).indices, 1.0)

    @staticmethod
    def sim(q, s):
        return (q @ s.T).float() / KDG


class Store:
    def __init__(self, le):
        self.le = le
        self.ce = Coder(le, 11)
        self.cs = Coder('site', 12)

    @torch.no_grad()
    def build(self, F, bs=64):
        TK, own, S0, S1, V, cnt = [], [], [], [], [], []
        for b in range(0, len(F), bs):
            FF = F[b:b + bs]
            seqs = [PRE + f['p']['rewrite'] + f['nt'] for f in FF]
            H, _, _ = forward(seqs, [self.le, 'site'])
            for i, f in enumerate(FF):
                ns = f['subj_last'] + 1
                TK.append(self.ce(H[self.le][i, len(PRE):len(PRE) + ns]))
                own += [b + i] * ns
                cnt.append(ns)
            ix = torch.arange(len(FF), device=C.DEV)
            s0 = torch.tensor([len(PRE) + len(f['p']['rewrite']) - 1 for f in FF], device=C.DEV)
            S0.append(self.cs(H['site'][ix, s0])); S1.append(self.cs(H['site'][ix, s0 + 1]))
            V.append(torch.tensor([f['nt'] for f in FF], device=C.DEV))
        self.TK, self.own = torch.cat(TK), torch.tensor(own, device=C.DEV)
        self.cnt = torch.tensor(cnt, device=C.DEV, dtype=torch.float)
        self.rows = np.concatenate([[0], np.cumsum(cnt)])
        self.S, self.V = [torch.cat(S0), torch.cat(S1)], torch.cat(V)

    @torch.no_grad()
    def analyze(self, seqs, evals, n, topc=3, bs=16):
        R = int(self.rows[n])
        own = self.own[:R]
        out = []
        for b in range(0, len(seqs), bs):
            S = seqs[b:b + bs]
            H, z, att = forward(S, [self.le, 'site'])
            m = att.bool()
            ov = Coder.sim(self.ce(H[self.le]), self.TK[:R]) * m[..., None]               # (B,T,R)
            ovc = torch.cummax(ov, 1).values
            fr = {}
            for ti, tt in enumerate(TT):
                f_ = torch.zeros(*ovc.shape[:2], n, device=C.DEV)
                f_.index_add_(2, own, (ovc >= tt).float())
                fr[ti] = f_ / self.cnt[:n]
            ssum = torch.zeros(*ovc.shape[:2], n, device=C.DEV).index_add_(2, own, ovc) / self.cnt[:n]
            del ov, ovc
            sc = self.cs(H['site'])
            for i in range(len(S)):
                for pos, j in evals[b + i]:
                    key = fr[0][i, pos] * 2 + fr[1][i, pos] * 2 + ssum[i, pos]
                    cand = key.topk(min(topc, n)).indices
                    out.append(dict(z=int(z[i, pos]), cand=cand.tolist(),
                                    fr=[fr[t][i, pos, cand].tolist() for t in range(len(TT))],
                                    s0=Coder.sim(sc[i, pos][None], self.S[0][:n][cand])[0].tolist(),
                                    s1=Coder.sim(sc[i, pos][None], self.S[1][:n][cand])[0].tolist(), j=j))
        return out

    def decide(self, r, ti, te, ts):
        for c, f, a0, a1 in zip(r['cand'], r['fr'][ti], r['s0'], r['s1']):
            if f >= te:
                if r['j'] == 0 and a0 >= ts:
                    return int(self.V[c, 0]), True
                if r['j'] == 1 and a1 >= ts:
                    return int(self.V[c, 1]), True
                if r['j'] == -1 and max(a0, a1) >= ts:
                    return -1, True
                return r['z'], False
        return r['z'], False


def fact_queries(F, form):
    seqs = [PRE + f['p'][form] + [f['nt'][0]] for f in F]
    return seqs, [[(len(s) - 2, 0), (len(s) - 1, 1)] for s in seqs]


def other_rel(F, written):
    out, r = [], random.Random(5)
    for f in F:
        for _ in range(20):
            rel = r.choice(FA.REL)
            if rel != f['rel'] and f"{f['person']}的{rel}" not in written:
                out.append(PRE + C.enc(tok, f"{f['person']}的{rel}叫"))
                break
    return out


CQ = None


def measure(st, F, n, cfgs, written):
    global CQ
    new, first = F[max(0, n - 100):n], F[:min(100, n)]
    A_ = {}
    for name, FF, form in [('new_rewrite', new, 'rewrite'), ('new_para', new, 'para'), ('new_qa', new, 'qa'),
                           ('first100', first, 'rewrite')]:
        s, e = fact_queries(FF, form)
        A_[name] = (st.analyze(s, e, n), [t for f in FF for t in f['nt']])
    ns = [PRE + f['p']['rewrite'] for f in neigh]
    A_['leak_other'] = st.analyze(ns, [[(len(s) - 1, 0)] for s in ns], n)
    os_ = other_rel(F[:min(n, 300)], written)
    A_['leak_samep'] = st.analyze(os_, [[(len(s) - 1, 0)] for s in os_], n)
    ms_ = []
    for f in F[max(0, n - 100):n]:
        sq = PRE + C.enc(tok, f"{f['subj']}今天很开心，我们一起去公园玩了。")
        ms_.append((sq, [(t, -1) for t in range(len(PRE) + f['subj_last'] + 1, len(sq))]))
    A_['mention'] = st.analyze([a for a, _ in ms_], [b for _, b in ms_], n)
    qs_ = [PRE + f['p']['qa'] for f in F[max(0, n - 100):n]]
    A_['qa_other_pos'] = st.analyze(qs_, [[(t, -1) for t in range(len(PRE) + 1, len(q) - 1)] for q in qs_], n)
    if CQ is None:
        CQ = ([], [])
        for i in range(probes.X.shape[0]):
            L = int(probes.att[i].sum())
            CQ[0].append(PRE + probes.X[i, :L].tolist())
            CQ[1].append([(t, -1) for t in range(2, L + 1)])
    A_['corpus'] = st.analyze(CQ[0], CQ[1], n)
    res = {}
    for cfg in cfgs:
        ti, te, ts = cfg
        m = {}
        for name in ('new_rewrite', 'new_para', 'new_qa', 'first100'):
            rs, want = A_[name]
            pred = [st.decide(r, ti, te, ts)[0] for r in rs]
            m[name] = round(float(np.array([int(a == b) for a, b in zip(pred, want)]).reshape(-1, 2).all(1).mean()), 4)
        for name in ('leak_other', 'leak_samep'):
            m[name] = round(float(np.mean([st.decide(r, ti, te, ts)[1] for r in A_[name]])), 4)
        m['corpus_fire'] = round(float(np.mean([st.decide(r, ti, te, ts)[1] for r in A_['corpus']])), 5)
        m['mention_fire'] = round(float(np.mean([st.decide(r, ti, te, ts)[1] for r in A_['mention']])), 4)
        m['qa_nonanswer_fire'] = round(float(np.mean([st.decide(r, ti, te, ts)[1] for r in A_['qa_other_pos']])), 4)
        res[cfg] = m
    return res


GRID = [(ti, te, ts) for ti in range(len(TT)) for te in (0.75, 1.0) for ts in (0.0, 0.1, 0.2, 0.3, 0.4)]


def score(r):
    pen = 10 * (max(0, r['leak_other'] - 0.02) + max(0, r['leak_samep'] - 0.05) + max(0, r['corpus_fire'] - 0.001)
                + max(0, 0.95 - r['new_rewrite']))
    return r['new_para'] + r['new_qa'] - pen


def dev():
    devF = FA.make(tok, 1000, seed=777, exclude=set(f['subj'] for f in neigh))
    written = set(f['subj'] for f in devF)
    for le in [int(x) for x in A.get('layers', '0,2,4,8').split(',')]:
        st = Store(le)
        st.build(devF)
        for cfg, m in measure(st, devF, len(devF), GRID, written).items():
            log(dict(mode='dev', le=le, tt=TT[cfg[0]], te=cfg[1], ts=cfg[2], **m))


def main():
    if 'le' in A:
        le, tt, te, ts = int(A['le']), float(A['tt']), float(A['te']), float(A['ts'])
    else:
        rows = [json.loads(l) for l in open(A.get('devlog', OUT)) if '"mode": "dev"' in l and '"tt"' in l]
        b = max(rows, key=score)
        le, tt, te, ts = b['le'], b['tt'], b['te'], b['ts']
        log(dict(info='picked', le=le, tt=tt, te=te, ts=ts, dev=b))
    N = int(A.get('n', 10000))
    F = FA.make(tok, N, seed=int(A.get('seed', 1)), exclude=set(f['subj'] for f in neigh))
    written = set(f['subj'] for f in F)
    st = Store(le)
    t = time.time()
    st.build(F)
    torch.cuda.synchronize()
    ms = 1000 * (time.time() - t) / N
    tss = [float(x) for x in A.get('tss', str(ts)).split(',')]
    cfgs = [(TT.index(tt), te, x) for x in tss]
    for n in [c for c in (10, 30, 100, 300, 1000, 3000, 10000) if c <= N]:
        res = measure(st, F, n, cfgs, written)
        rs_all = None
        for cfg in cfgs:
            m = res[cfg]
            if n in (1000, 10000):
                if rs_all is None:
                    s, e = fact_queries(F[:n], 'rewrite')
                    rs_all = st.analyze(s, e, n)
                pred = [st.decide(r, *cfg)[0] for r in rs_all]
                want = [t_ for f in F[:n] for t_ in f['nt']]
                m['all'] = round(float(np.array([int(a == b) for a, b in zip(pred, want)]).reshape(-1, 2).all(1).mean()), 4)
            log(dict(mode='main', le=le, tt=tt, te=te, ts=cfg[2], n=n, ms_per_fact_write=round(ms, 3),
                     keys_per_fact=round(float(st.cnt.mean()) + 2, 2), **m))


if __name__ == '__main__':
    log(dict(info='start', mode=MODE, args=A))
    {'dev': dev, 'main': main}[MODE]()
    log(dict(info='end', gpu_gb=round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)))
