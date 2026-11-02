"""Generation-0 hand wiring [G1] and the speech-calibration rules [M6]. Registry entry G1:
  from: chain-bearing units of sound layer A1 that reliably fire for syllable x in the four adult voices
  to:   the force neurons of x's 4 muscles
  how:  non-negative least squares (ridge 0.3, X1f method) from the measured newborn codes to the muscle sets
  where: the ACQUIRED (plastic, decaying) part, gamma_motor, concave-slowed decay; G1's gain g_G1 is set so that a
         full adult-voice code drives its own force neurons at g1_rel x the release level against tonic blocking.
Also measures every syllable's adult-voice code over all chain-bearing sound units (for the DIAGNOSTIC forced
inner-hearing hook of stage b; never used in the main arms)."""
import numpy as np
import torch

from common import DEV
from world_adapter import ADULTS, NSYL, OWN


def measure_codes(brain, coch, reps=2, pre=4, post=8, voices=ADULTS):
    """Sound area alone, from silence, no returns. P[s, v, j]: firing probability of chain source j (sound part)
    while syllable s is heard in voice v (each layer read at its own latency)."""
    snd, ch = brain.sound, brain.chains
    n_sound = sum(len(ix) for ix in ch.sound_idx)
    P = torch.zeros((NSYL, len(voices), n_sound), device=DEV)
    zero_ret = [torch.zeros(n, device=DEV) for n in snd.n]
    saved = [x.clone() for x in snd.state]
    for s in range(NSYL):
        for vi, v in enumerate(voices):
            for _ in range(reps):
                snd.state = [torch.zeros_like(x) for x in saved]
                seq = [[]] * pre + [[('s', s, v, 1.0)]] * 3 + [[]] * post
                hist = []
                for srcs in seq:
                    a0 = torch.as_tensor(coch.a0(srcs), device=DEV)
                    st = snd.step(a0, zero_ret)
                    hist.append(torch.cat([st[l][ix] for l, ix in enumerate(ch.sound_idx)]))
                H = torch.stack(hist).float()
                acc = torch.zeros(n_sound, device=DEV)
                a = 0
                for l, ix in enumerate(ch.sound_idx):                      # layer l lags the input by l ticks
                    acc[a:a + len(ix)] = H[pre + l: pre + l + 3, a:a + len(ix)].mean(0)
                    a += len(ix)
                P[s, vi] += acc / reps
    snd.state = saved
    return P


def select_codes(P, p_rel=0.5):
    """code(s): every sound chain source that syllable s reliably lights (fires in >= p_rel of s's ticks, mean over
    the adult voices). Units shared with other syllables are kept: NNLS sorts out the sharing (X1f's A1 map used
    the shared content features the same way)."""
    M = P.mean(1)
    return [torch.nonzero(M[s] >= p_rel).squeeze(1) for s in range(M.shape[0])]


def nnls(A, B, lam=0.3, it=3000):
    """min |A W - B|^2 + lam^2 |W|^2, W >= 0 (projected gradient; no n x n matrix is formed)"""
    L = torch.linalg.matrix_norm(A, ord=2) ** 2 + lam ** 2
    W = torch.zeros((A.shape[1], B.shape[1]), device=DEV)
    for _ in range(it):
        W = (W - (A.T @ (A @ W - B) + lam ** 2 * W) / L).clamp_min(0.0)
    return W


def g1_quality(Wm, codes, n0, need, w_ref):
    """how well a sound A1 -> muscle map says each syllable from its forced code, through two-layer weights:
    share of syllables whose 4 most driven muscles are exactly its own, and the median own-4th / best-other ratio"""
    out = Wm.sum(1).clamp_min(w_ref)
    inn = Wm.sum(0).clamp_min(w_ref)
    Wn = Wm / torch.sqrt(out[:, None] * inn[None, :])
    exact, margin = [], []
    for s, cd in enumerate(codes):
        x = torch.zeros(n0, device=DEV)
        x[cd[cd < n0]] = 1.0
        d = x @ Wn
        own = need[s] > 0
        exact.append(bool(set(torch.topk(d, 4).indices.tolist()) == set(torch.nonzero(own).squeeze(1).tolist())))
        margin.append((d[own].min() / d[~own].max().clamp_min(1e-9)).item())
    return float(np.mean(exact)), float(np.median(margin))


def wire_g1(brain, coch, ph):
    """[G1] + [M6]. Returns a registry dict.
    G1's A matrix (what each syllable lights on sound A1, the input of the least-squares map) is chosen among three
    forms by the map's own quality on the forced codes (share of syllables whose 4 most driven muscles are exactly
    its own, then the margin): binary at p >= 0.5, binary at p >= 0.8, or the firing probabilities themselves."""
    c, mot, ch = brain.c, brain.motor, brain.chains
    P = measure_codes(brain, coch)
    codes = select_codes(P)
    n0 = len(ch.sound_idx[0])                                   # layer-A1 sources are the first segment
    src = ch.sound_idx[0]
    need = torch.as_tensor(ph.need_mat, dtype=torch.float32, device=DEV)            # [34, 24]
    M = P.mean(1)[:, :n0]
    # [G1, approved 11-02] keep only A1 units that the child's OWN voice does not light: lit in < own_cut of the
    # ticks of every own syllable (the own voice then drives no G1 input at all)
    Pown = measure_codes(brain, coch, voices=(OWN,))[:, 0, :n0]                      # [34, n0]
    own_lit = Pown.max(0).values >= c['g1_own_cut']
    old = (M >= 0.5)
    kept = old & ~own_lit[None, :]
    keep_stats = dict(own_cut=c['g1_own_cut'], units_lit_by_own_voice=int(own_lit.sum()), a1_sources=n0,
                      kept_per_syllable=kept.sum(1).tolist(), old_per_syllable=old.sum(1).tolist(),
                      below_third=[int(s) for s in range(M.shape[0]) if kept[s].sum() < old[s].sum() / 3])
    M = M * (~own_lit)[None, :].float()
    variants = {'bin0.5': (M >= 0.5).float(), 'bin0.8': (M >= 0.8).float(), 'prob': M}
    codes_g1 = [cd[(cd >= n0) | ~own_lit[cd.clamp_max(n0 - 1)]] for cd in codes]      # quality on kept A1 units
    qual = {}
    best = None
    for k, A in variants.items():
        Wk = nnls(A, need)
        q = g1_quality(Wk, codes_g1, n0, need, c['w_ref'])
        qual[k] = q
        if best is None or q > best[0]:
            best = (q, k, Wk)
    Wm = best[2]
    Wf = Wm[:, mot.muscle_of]                                                       # [n0, 24 s]
    mot.set_g1(src, Wf)
    # [M6] gain: own-muscle force drive of a full adult code = g1_rel x release level against tonic blocking
    rel = c['theta_m'] + c['b_block'] * mot.n_tonic / (8.0 * mot.s)
    mot.G1.gain = 1.0
    own, pooled = [], []
    for s, cd in enumerate(codes):
        act = cd[cd < n0]
        if act.numel() == 0:
            continue
        d = mot.G1.drive(act)
        m = torch.as_tensor(ph.need_mat[s], device=DEV)[mot.muscle_of]
        own.append(d[m].median())
        pooled.append(d.sum())
    m0 = torch.stack(own).median()
    gain = float(c['g1_rel'] * rel / m0)
    mot.G1.gain = gain
    p_sharp = float(torch.stack(pooled).median()) * gain
    j = torch.arange(4 * mot.s, device=DEV, dtype=torch.float32)
    mot.ladder = p_sharp * (1.0 + c['ladder_step'] * 4.0 * (j + 1) / (4 * mot.s))  # recruits 1x..2x a sharp drive
    sizes = [int(x.numel()) for x in codes]
    l1 = [int((x < n0).sum()) for x in codes]
    return dict(set='G1', own_voice=keep_stats, variant=best[1], quality=qual, links=int((Wf > 1e-6).sum()), gain=gain,
                release_level=rel, p_sharp=p_sharp,
                code_size_min=min(sizes), code_size_median=float(np.median(sizes)), a1_code_min=min(l1),
                a1_code_median=float(np.median(l1)), syllables_without_a1_code=int(sum(1 for x in l1 if x == 0))), codes


def forced_mask(brain, units):
    """DIAGNOSTIC: bool mask over PFC copies for a set of sound chain sources (indices into the sound part)"""
    m = torch.zeros(brain.n1, dtype=torch.bool, device=DEV)
    m[units] = True
    return m
