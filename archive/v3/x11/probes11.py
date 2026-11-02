"""X11 probes (scoring only; world labels never reach the brain).

codes():        each toned syllable's code in PFC L1 = the sound copies whose chain input RISES (the learning rule's
                post event) in the ticks after the syllable's onset, measured on the newborn's fixed sound area
                (time-parallel, no returns; the sound area never learns, so the codes hold for the whole life).
heldout_eval(): frozen brain (learning off, neuron states restored after) hears a FIXED set of held-out dialogues
                (LCCC test split, fixed voices / pauses / noise). For every syllable occurrence (not the first of a
                dialogue) with onset tick t0:
                  PFC prediction  = sound copies lit INTERNALLY in L1 (x_int, from L4(t-1)) on ticks t0+1..t0+3,
                                    before any of that syllable's sound can have reached L4;
                  PFC late (diagnostic) = the same on t0+2..t0+4, the last ticks before the syllable's own sound
                                    can reach L4 (they see all of the previous syllable through A1);
                  hippocampus     = sound copies lit internally in its copy layer H1 (h_int) on ticks t0+1..t0+2
                                    (its own prospective window);
                  heard (oracle)  = the copies that actually rose on t0+1..t0+3 (readout ceiling).
                Each is ranked against every syllable's code by cosine: top-1 / top-5 (toned and toneless), empty
                share; plus per-tick J(x_int, risen copies) on speech ticks.
ngram_eval():   unigram / bigram / trigram count models (and bigram from the syllable before last) trained on the
                same dialogues the brain has heard, scored on the same occurrences: directly (top-k of the
                distribution) and through the codes (predicted code = sum_s p(s) code_s, ranked like the brain).
Activity:       Window (per 10 dialogues, learning on) and the eval's counts per layer."""
import numpy as np
import torch

from common import DEV
from voice import ADULTS
from world11 import script_dialogue

NAMES = None


def n_sound_copies(b):
    return int(sum(len(ix) for ix in b.chains.sound_idx))


class Frozen:
    def __init__(self, life):
        self.life = life

    def __enter__(self):
        b = self.life.brain
        self.was = b.learning
        b.learning = False
        self.snap = b.snapshot()
        self.rng = self.life.coch.rng.bit_generator.state
        return self.snap

    def __exit__(self, *a):
        b = self.life.brain
        b.restore(self.snap)
        b.learning = self.was
        self.life.coch.rng.bit_generator.state = self.rng


# ---------------------------------------------------------------------------------------------- time-parallel sound
def sound_states(b, A0):
    """A0 bool [T, n0] (DEV) -> list of bool [T, n_l]; A1(t) = f(A0(t)), Al(t) = f(A(l-1)(t-1)); no returns"""
    snd = b.sound
    prev, out = A0.float(), []
    for l, ly in enumerate(snd.layers):
        T = prev.shape[0]
        d = torch.empty((T, ly.n_post), dtype=torch.bool, device=DEV)
        for a in range(0, T, 1024):
            d[a:a + 1024] = ly.drive_batch(prev[a:a + 1024]) > snd.thr[l]
        if l > 0:
            d = torch.cat([torch.zeros_like(d[:1]), d[:-1]])
        out.append(d)
        prev = d.float()
    return out


def chain_rises(b, states):
    """-> (L1ext sound part [T, nsnd], rises [T, nsnd]); L1ext(t) = sources(t-1)"""
    X = torch.cat([st[:, ix] for st, ix in zip(states, b.chains.sound_idx)], 1)
    X = torch.cat([torch.zeros_like(X[:1]), X[:-1]])
    R = X & ~torch.cat([torch.zeros_like(X[:1]), X[:-1]])
    return X, R


def token_stream(life, sids, unigram, n_ctx, seed):
    """tokens: (context syllable, target, next syllable), 3 ticks each, same voice; every target in every adult voice
    x n_ctx contexts; contexts drawn from the unigram distribution. Returns A0 [T, n0] bool, labels."""
    rng = np.random.default_rng(seed)
    p = unigram / unigram.sum()
    coch = life.coch
    st = coch.rng.bit_generator.state
    coch.rng = np.random.default_rng(seed + 1)
    A0, lab, vox, rep = [], [], [], []
    for vi, v in enumerate(ADULTS):
        for r in range(n_ctx):
            for s in sids:
                c1, c2 = rng.choice(len(p), 2, p=p)
                off = rng.uniform(-1, 1)
                for x in (c1, s, c2):
                    A0 += [coch.a0([(int(x), k, v, off, 0.0)]) for k in range(3)]
                lab.append(s)
                vox.append(vi)
                rep.append(r)
    coch.rng = np.random.default_rng(0)
    coch.rng.bit_generator.state = st
    return torch.as_tensor(np.array(A0), device=DEV), np.array(lab), np.array(vox), np.array(rep)


def codes(life, sids, unigram, n_ctx=2, seed=777):
    """-> dict(code3 [n_sids, nsnd] float, code2, n_tokens); windows relative to the target onset t0"""
    b = life.brain
    A0, lab, _, _ = token_stream(life, sids, unigram, n_ctx, seed)
    states = sound_states(b, A0)
    _, R = chain_rises(b, states)
    del states
    pos = {s: i for i, s in enumerate(sids)}
    li = torch.as_tensor([pos[s] for s in lab], device=DEV)
    t0 = torch.arange(len(lab), device=DEV) * 9 + 3
    w3 = R[t0 + 1].float() + R[t0 + 2].float() + R[t0 + 3].float()
    w2 = R[t0 + 1].float() + R[t0 + 2].float()
    w4 = R[t0 + 2].float() + R[t0 + 3].float() + R[t0 + 4].float()
    n = torch.zeros(len(sids), device=DEV).index_add_(0, li, torch.ones(len(lab), device=DEV))
    c3 = torch.zeros((len(sids), R.shape[1]), device=DEV).index_add_(0, li, w3) / n[:, None]
    c2 = torch.zeros((len(sids), R.shape[1]), device=DEV).index_add_(0, li, w2) / n[:, None]
    c4 = torch.zeros((len(sids), R.shape[1]), device=DEV).index_add_(0, li, w4) / n[:, None]
    return dict(code3=c3, code2=c2, code_late=c4, sids=np.asarray(sids), tokens=int(len(lab)),
                mean_code_size3=float((c3 >= 0.5).sum(1).float().mean()))


# ---------------------------------------------------------------------------------------------- ranking
def rank_scores(P, C, sids, true, bases_of=None):
    """P [n, d] predictions, C [k, d] codes; true: indices into sids. Cosine, random tie-break. Empty P = miss."""
    g = torch.Generator(device=DEV)
    g.manual_seed(5)
    Pn = P / P.norm(dim=1, keepdim=True).clamp_min(1e-9)
    Cn = C / C.norm(dim=1, keepdim=True).clamp_min(1e-9)
    S = Pn @ Cn.T + 1e-6 * torch.rand((P.shape[0], C.shape[0]), generator=g, device=DEV)
    tt = torch.as_tensor(true, device=DEV)
    st = S.gather(1, tt[:, None])
    rank = (S > st).sum(1)
    empty = P.sum(1) == 0
    rank = torch.where(empty, torch.full_like(rank, C.shape[0]), rank)
    out = dict(top1=float((rank == 0).float().mean()), top5=float((rank < 5).float().mean()),
               mrr=float((1.0 / (rank.float() + 1)).mean()), empty=float(empty.float().mean()))
    if bases_of is not None:                                   # toneless: best tone per base
        bo = torch.as_tensor(bases_of, device=DEV)
        nb = int(bo.max()) + 1
        SB = torch.full((P.shape[0], nb), -9.0, device=DEV).scatter_reduce(1, bo[None].expand(P.shape[0], -1), S,
                                                                             'amax')
        tb = bo[tt]
        sb = SB.gather(1, tb[:, None])
        rb = (SB > sb).sum(1)
        rb = torch.where(empty, torch.full_like(rb, nb), rb)
        out.update(toneless_top1=float((rb == 0).float().mean()), toneless_top5=float((rb < 5).float().mean()))
    return out, rank


# ---------------------------------------------------------------------------------------------- held-out playback
def heldout_plan(test, n_dlg, seed):
    stream, occ = [], []
    for d in range(n_dlg):
        plan = script_dialogue(test, d, seed)
        base = len(stream)
        hist = []
        for i, x in enumerate(plan):
            if x is not None and x[1] == 0:
                if hist:
                    occ.append((base + i, x[0], hist[-1], hist[-2] if len(hist) > 1 else -1, d, x[6], x[7],
                                hist[-3] if len(hist) > 2 else -1))
                hist.append(x[0])
        stream += plan
    stream += [None] * 6
    return stream, occ


def heldout_eval(life, test, n_dlg, cd, seed=4242):
    b = life.brain
    nsnd = n_sound_copies(b)
    stream, occ = heldout_plan(test, n_dlg, seed)
    T = len(stream)
    pos = {s: i for i, s in enumerate(cd['sids'])}
    occ = [o for o in occ if o[1] in pos]
    n = len(occ)
    o3 = np.full(T + 4, -1, np.int64)
    o2 = np.full(T + 4, -1, np.int64)
    o4 = np.full(T + 6, -1, np.int64)
    for j, o in enumerate(occ):
        o3[o[0] + 1:o[0] + 4] = j
        o2[o[0] + 1:o[0] + 3] = j
        o4[o[0] + 2:o[0] + 5] = j        # diagnostic late window: the last 3 ticks before the syllable's sound
    P3 = torch.zeros((n, nsnd), device=DEV)                                     # can reach L4 (t0+2 .. t0+4)
    P4 = torch.zeros((n, nsnd), device=DEV)
    H2 = torch.zeros((n, nsnd), device=DEV)
    O3 = torch.zeros((n, nsnd), device=DEV)
    jacc = torch.zeros(4, device=DEV)               # sum J pfc, n pfc, sum J hip, n hip
    names = b.count_names()
    csum = torch.zeros((2, len(names)), device=DEV)
    cn = [0, 0]
    with Frozen(life):
        life.coch.rng = np.random.default_rng(seed)
        m = np.zeros(24, bool)
        hip = b.hip is not None
        for t, x in enumerate(stream):
            a0 = life.coch.a0([] if x is None else [x[:5]])
            b.step(torch.as_tensor(a0, device=DEV), 0.0)
            xi = b.pfc.x_int[:nsnd]
            ri = b.pfc.L1ext.rose()[:nsnd]
            hi = b.hip.h_int[:nsnd] if hip else None
            if o3[t] >= 0:
                P3[o3[t]] += xi.float()
                O3[o3[t]] += ri.float()
            if o4[t] >= 0:
                P4[o4[t]] += xi.float()
            if hip and o2[t] >= 0:
                H2[o2[t]] += hi.float()
            sp = x is not None
            if sp:
                u = (xi | ri).sum()
                jacc[0] += (xi & ri).sum() / u.clamp_min(1)
                jacc[1] += (u > 0).float()
                if hip:
                    u2 = (hi | ri).sum()
                    jacc[2] += (hi & ri).sum() / u2.clamp_min(1)
                    jacc[3] += (u2 > 0).float()
            csum[int(sp)] += b.counts().float()
            cn[int(sp)] += 1
        life.coch.rng = np.random.default_rng(0)
    true = np.array([pos[o[1]] for o in occ])
    bases_of = np.asarray(cd['sids']) // 5
    res = dict(n_occ=n, n_dialogues=n_dlg, ticks=T)
    res['pfc'], rk = rank_scores(P3, cd['code3'], cd['sids'], true, bases_of)
    res['pfc']['mean_lit_per_window'] = float(P3.sum(1).mean() / 3)
    ti = np.array([o[6] == 0 for o in occ])                 # first syllable of a turn (after a pause)
    rk = rk.cpu().numpy()
    res['pfc']['top1_turn_initial'] = float((rk[ti] == 0).mean()) if ti.any() else None
    res['pfc']['top1_within_turn'] = float((rk[~ti] == 0).mean())
    res['pfc_late'], rk_late = rank_scores(P4, cd['code_late'], cd['sids'], true, bases_of)
    if b.hip is not None:
        res['hip'], rk_hip = rank_scores(H2, cd['code2'], cd['sids'], true, bases_of)
        res['hip']['mean_lit_per_window'] = float(H2.sum(1).mean() / 2)
    resp = np.array([o[5] >= 1 and o[6] == 0 for o in occ])   # response: first syllable of a reply turn
    res['response'] = dict(n=int(resp.sum()))
    for nm, r_ in (('pfc', rk), ('pfc_late', rk_late.cpu().numpy()),
                   ('hip', rk_hip.cpu().numpy() if b.hip is not None else None)):
        if r_ is not None and resp.any():
            res['response'][nm] = dict(top1=float((r_[resp] == 0).mean()), top5=float((r_[resp] < 5).mean()))
    res['heard_oracle'], _ = rank_scores(O3, cd['code3'], cd['sids'], true, bases_of)
    jv = jacc.cpu().numpy()
    res['tick_J_pfc'] = float(jv[0] / max(jv[1], 1))
    res['tick_J_hip'] = float(jv[2] / max(jv[3], 1)) if b.hip is not None else None
    cs = csum.cpu().numpy()
    res['counts'] = {nm: dict(silence=float(cs[0, i] / max(cn[0], 1)), speech=float(cs[1, i] / max(cn[1], 1)))
                     for i, nm in enumerate(names)}
    return res, occ


# ---------------------------------------------------------------------------------------------- count models
def sequences(dlgs, n):
    """dialogue syllable sequences (turns joined; the model sees the whole dialogue as one stream)"""
    return [np.concatenate(dlgs.turns(d)).astype(np.int64) for d in range(min(n, dlgs.n))]


def ngram_eval(train, n_train, occ, cd, K):
    """count models on the first n_train training dialogues; scored on the held-out occurrences"""
    seqs = sequences(train, n_train)
    allsyl = np.concatenate(seqs)
    uni = np.bincount(allsyl, minlength=K).astype(np.float64)
    big_k = np.concatenate([s[:-1] * K + s[1:] for s in seqs if len(s) > 1])
    tri_k = np.concatenate([(s[:-2] * K + s[1:-1]) * K + s[2:] for s in seqs if len(s) > 2])
    lag_k = np.concatenate([s[:-2] * K + s[2:] for s in seqs if len(s) > 2])
    lag3_k = np.concatenate([(s[:-3] * K + s[1:-2]) * K + s[3:] for s in seqs if len(s) > 3])
    reply0 = np.array([t[0] for d in range(min(n_train, train.n)) for t in train.turns(d)[1:] if len(t)],
                      np.int64)                                # first syllables of reply turns heard so far
    r0_order = np.argsort(-np.bincount(reply0, minlength=K), kind='stable')

    def table(keys):
        u, c = np.unique(keys, return_counts=True)
        return u, c

    big, tri, lag, lag3 = table(big_k), table(tri_k), table(lag_k), table(lag3_k)

    def follow(tab, ctx):
        u, c = tab
        lo, hi = np.searchsorted(u, [ctx * K, ctx * K + K])
        return u[lo:hi] - ctx * K, c[lo:hi]

    uni_order = np.argsort(-uni, kind='stable')
    sids = np.asarray(cd['sids'])
    pos = {s: i for i, s in enumerate(sids)}
    res = {}
    KINDS = ('unigram', 'bigram', 'trigram', 'bigram_lag2', 'trigram_lag2', 'unigram_reply_initial')
    hits = {k: np.zeros((len(occ), 2), bool) for k in KINDS}
    Pmix = np.zeros((len(occ), len(sids)), np.float32)
    puni = uni[sids] / max(uni[sids].sum(), 1)
    for j, (t0, s, p1, p2, d, turn, ps, p3) in enumerate(occ):
        lists = {k: [] for k in KINDS}
        b_n, b_c = follow(big, p1)
        lists['bigram'] = list(b_n[np.argsort(-b_c, kind='stable')][:5])
        if p2 >= 0:
            t_n, t_c = follow(tri, p2 * K + p1)
            lists['trigram'] = list(t_n[np.argsort(-t_c, kind='stable')][:5])
            l_n, l_c = follow(lag, p2)
            lists['bigram_lag2'] = list(l_n[np.argsort(-l_c, kind='stable')][:5])
        if p3 >= 0:
            m_n, m_c = follow(lag3, p3 * K + p2)
            lists['trigram_lag2'] = list(m_n[np.argsort(-m_c, kind='stable')][:5])
        for k in ('trigram', 'bigram', 'trigram_lag2', 'bigram_lag2'):   # back off to lower orders to fill 5
            fill = lists['bigram'] if k == 'trigram' else lists['bigram_lag2'] if k == 'trigram_lag2' else []
            for x in list(fill) + list(uni_order[:10]):
                if len(lists[k]) >= 5:
                    break
                if x not in lists[k]:
                    lists[k].append(x)
        lists['unigram'] = list(uni_order[:5])
        for k, L in lists.items():
            hits[k][j] = (len(L) > 0 and L[0] == s, s in L[:5])
        pb = np.zeros(len(sids))                                   # interpolated distribution for the code readout
        for x, c in zip(b_n, b_c):
            if x in pos:
                pb[pos[x]] += c
        pb = pb / max(pb.sum(), 1)
        pt = np.zeros(len(sids))
        if p2 >= 0:
            for x, c in zip(t_n, t_c):
                if x in pos:
                    pt[pos[x]] += c
            pt = pt / max(pt.sum(), 1)
        Pmix[j] = 0.6 * pt + 0.3 * pb + 0.1 * puni if pt.sum() > 0 else 0.75 * pb + 0.25 * puni
    lists_r0 = list(r0_order[:5])
    for j, o in enumerate(occ):
        hits['unigram_reply_initial'][j] = (lists_r0[0] == o[1], o[1] in lists_r0)
    resp = np.array([o[5] >= 1 and o[6] == 0 for o in occ])
    res['response'] = {k: dict(top1=float(h[resp, 0].mean()), top5=float(h[resp, 1].mean()))
                       for k, h in hits.items()} if resp.any() else {}
    res['response']['n'] = int(resp.sum())
    for k, h in hits.items():
        res[k] = dict(top1=float(h[:, 0].mean()), top5=float(h[:, 1].mean()))
    true = np.array([pos[o[1]] for o in occ])
    Pc = torch.as_tensor(Pmix, device=DEV) @ cd['code3']
    res['interp_via_codes'], _ = rank_scores(Pc, cd['code3'], sids, true, sids // 5)
    Pu = torch.as_tensor(np.tile(puni.astype(np.float32), (len(occ), 1)), device=DEV) @ cd['code3']
    res['unigram_via_codes'], _ = rank_scores(Pu, cd['code3'], sids, true, sids // 5)
    res['train_dialogues'] = len(seqs)
    res['train_syllables'] = int(len(allsyl))
    return res


# ---------------------------------------------------------------------------------------------- activity windows
class Activity:
    """per-window (10 dialogues) mean counts in silence / speech and the max fraction, learning on (a1L style)"""

    def __init__(self, b, window=10):
        self.b, self.window = b, window
        self.names = b.count_names()
        k = len(self.names)
        self.acc = torch.zeros((2, k), device=DEV)
        self.mx = torch.zeros(k, device=DEV)
        self.n = [0, 0]
        self.wins = []
        self.d0 = None
        sizes = dict(L1=b.n1, L2=b.pfc.n, L3=b.pfc.n, L4=b.pfc.n, Hs=b.hip.nh if b.hip is not None else 1)
        self.size = torch.tensor([sizes.get(nm, 1e9) for nm in self.names], device=DEV)

    def __call__(self, life):
        v = self.b.counts().float()
        sp = int(life.world.adult_speaking())
        self.acc[sp] += v
        self.mx = torch.maximum(self.mx, v)
        self.n[sp] += 1
        d = life.world.n_ex
        if self.d0 is None:
            self.d0 = d
        if d - self.d0 >= self.window:
            self.flush(d)

    def flush(self, d):
        if sum(self.n) == 0:
            return
        a = self.acc.cpu().numpy()
        mx = (self.mx / self.size).cpu().numpy()
        n = self.n
        mean = (a[0] + a[1]) / max(sum(n), 1)
        ix = {nm: i for i, nm in enumerate(self.names)}
        w = dict(dialogue=int(d), L1=float(mean[ix['L1']]), L4=float(mean[ix['L4']]), A4=float(mean[ix['A4']]),
                 L1int=float(mean[ix['L1int']]), L2_L1=float(mean[ix['L2']] / max(mean[ix['L1']], 1)),
                 L3_L2=float(mean[ix['L3']] / max(mean[ix['L2']], 1)),
                 L4_L3=float(mean[ix['L4']] / max(mean[ix['L3']], 1)),
                 Hs=float(mean[ix['Hs']]) if 'Hs' in ix else None,
                 L1_speech=float(a[1, ix['L1']] / max(n[1], 1)), L1_silence=float(a[0, ix['L1']] / max(n[0], 1)),
                 max_frac=float(max(mx[ix[k]] for k in ('L1', 'L2', 'L3', 'L4', 'Hs') if k in ix)))
        kh = self.b.c['kh']
        w['ok'] = bool(all(0.5 <= w[k] <= 2.0 for k in ('L2_L1', 'L3_L2', 'L4_L3')) and w['max_frac'] <= 0.5 and
                       (w['Hs'] is None or 0.5 * kh <= w['Hs'] <= 2.0 * kh))
        self.wins.append(w)
        self.acc.zero_()
        self.mx.zero_()
        self.n = [0, 0]
        self.d0 = d

    def summary(self, since=None):
        ws = [w for w in self.wins if since is None or w['dialogue'] > since]
        if not ws:
            return {}
        out = dict(windows=len(ws), share_ok=float(np.mean([w['ok'] for w in ws])))
        for k in ('L1', 'L1int', 'L4', 'A4', 'Hs', 'L2_L1', 'L3_L2', 'L4_L3', 'max_frac'):
            v = [w[k] for w in ws if w[k] is not None]
            if v:
                out[k] = dict(first=v[0], last=v[-1], mean=float(np.mean(v)), max=float(np.max(v)))
        return out
