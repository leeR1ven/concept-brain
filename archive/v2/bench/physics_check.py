# -*- coding: utf-8 -*-
"""Physics check: are chars distinguishable at ear layer 0 (FEAT0), and is articulation -> sound lawful?
For each char: the FEAT0 code of the full-window tick (3rd tick of the char, said by 'other' after silence).
Reports pairwise Jaccard (lexicon chars, all pairs; corpus chars, random pairs), nearest-neighbour identity of the
own-voice code against the other-voice codes, and corr(sound distance, shared muscles).
usage: python3 physics_check.py [physics] [seed]"""
import sys, json
import numpy as np
sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__)))
from physics import Voice, Stream, corpus_chars
from frontend import Ear0, feat0
from world import World, LEXICON_CHARS


def codes(v, ear, F0, chars, who='other'):
    out = np.zeros((len(chars), len(F0)), bool)
    none = (None, None, False)
    for i, c in enumerate(chars):
        st = Stream(v.F, v.T, 2)
        for k in range(3):
            src = (c, v.hear(c, who), k == 0)
            win = st.tick((src, none, none) if who != 'self' else (none, none, src))
        out[i] = ear.respond(ear.encode(win))[F0]
    return out


def jac(A, B):
    inter = A.astype(np.float32) @ B.T.astype(np.float32)
    na = A.sum(1)[:, None]; nb = B.sum(1)[None, :]
    return inter / np.maximum(na + nb - inter, 1)


def run(physics='random', seed=0):
    w = World('B', seed=seed, physics=physics)
    v = w.voice
    ear = Ear0(seed)
    F0 = feat0(ear, w.warmup_windows())
    lex = sorted(LEXICON_CHARS)
    C = codes(v, ear, F0, lex)
    J = jac(C, C); np.fill_diagonal(J, np.nan)
    rng = np.random.default_rng(1)
    allc = corpus_chars()
    samp = [allc[i] for i in rng.choice(len(allc), 600, replace=False)]
    C2 = codes(v, ear, F0, samp)
    J2 = jac(C2, C2); np.fill_diagonal(J2, np.nan)
    # own voice vs other voice: nearest neighbour among the lexicon (identity preserved?)
    Cs = codes(v, ear, F0, lex, who='self')
    nn = np.nanargmax(jac(Cs, C), 1)
    # lawfulness
    need = np.array([[m in v.need[c] for m in range(32)] for c in samp], np.float32)
    snd = np.stack([v.sound[c].ravel() for c in samp]).astype(np.float32)
    ii = rng.integers(0, len(samp), 20000); jj = rng.integers(0, len(samp), 20000)
    keep = ii != jj
    ds = np.abs(snd[ii] - snd[jj]).sum(1)[keep]; dm = (need[ii] * need[jj]).sum(1)[keep]
    j3 = J2[ii[keep], jj[keep]]
    shared = {k: float(np.nanmean(j3[dm == k])) if (dm == k).any() else None for k in range(4)}
    out = dict(physics=physics, seed=seed, n_feat0=len(F0), lexicon_chars=len(lex),
               lex_jaccard_mean=float(np.nanmean(J)), lex_jaccard_max=float(np.nanmax(J)),
               lex_pairs_over_0_8=int(np.nansum(J > 0.8) // 2), lex_pairs_over_0_9=int(np.nansum(J > 0.9) // 2),
               corpus600_jaccard_mean=float(np.nanmean(J2)), corpus600_jaccard_max=float(np.nanmax(J2)),
               corpus600_pairs_over_0_9=int(np.nansum(J2 > 0.9) // 2),
               own_voice_nn_correct=float(np.mean(nn == np.arange(len(lex)))),
               corr_sound_distance_vs_shared_muscles=float(np.corrcoef(ds, dm)[0, 1]),
               layer0_jaccard_by_shared_muscles=shared,
               mean_energy=float(np.mean([v.sound[c].sum() for c in samp])))
    return out


def full_vocab(physics='random', seed=0, chunk=500):
    """worst case over all corpus chars (chunked: memory-safe)."""
    w = World('B', seed=seed, physics=physics)
    ear = Ear0(seed); F0 = feat0(ear, w.warmup_windows())
    allc = corpus_chars()
    C = codes(w.voice, ear, F0, allc).astype(np.float32)
    n = len(allc); nC = C.sum(1)
    mx, over9, over95, ident = 0.0, 0, 0, 0
    hist = np.zeros(20, np.int64)
    for a in range(0, n, chunk):
        inter = C[a:a + chunk] @ C.T
        J = inter / np.maximum(nC[a:a + chunk, None] + nC[None, :] - inter, 1)
        rows = np.arange(a, min(a + chunk, n))
        J[rows - a, rows] = 0.0
        J = np.triu(J, k=a + 1 - a) if False else J
        mask = np.arange(n)[None, :] > rows[:, None]               # each pair once
        Jp = J[mask]
        mx = max(mx, float(Jp.max())); over9 += int((Jp > 0.9).sum()); over95 += int((Jp > 0.95).sum())
        ident += int((Jp >= 0.999).sum())
        hist += np.histogram(Jp, bins=20, range=(0, 1))[0]
    grids = {}
    dup = 0
    for c in allc:
        k = w.voice.sound[c].tobytes(); dup += k in grids; grids[k] = c
    return dict(physics=physics, seed=seed, chars=n, max_jaccard=mx, pairs_over_0_9=over9, pairs_over_0_95=over95,
                identical_codes=ident, identical_sound_grids=dup,
                jaccard_hist_0_05=[int(x) for x in hist])


if __name__ == '__main__':
    if len(sys.argv) > 3 and sys.argv[3] == 'full':
        print(json.dumps(full_vocab(sys.argv[1], int(sys.argv[2])), ensure_ascii=False))
        sys.exit()
    ph = sys.argv[1] if len(sys.argv) > 1 else 'random'
    sd = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    print(json.dumps(run(ph, sd), ensure_ascii=False, indent=1))
