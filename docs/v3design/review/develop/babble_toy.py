"""Developmental review toy (local, < 200 MB, ~1 min): can ONLINE babbling teach an inverse vocal map good enough to
say the lexicon (ARCH_v3_core L9: '>= 90 % of lexicon chars bound by 20 % of life'), with physics='lawful' (W6)?

Uses the bench's own Voice (v2/bench/physics.py, imported read-only) with physics='lawful': each muscle has a fixed
3-cell pattern; the sound of any set = clip(sum of patterns).  The mouth rule is the real one (4 of a char's muscles =
clear char, 3 = mumbled char, else garble), over the full 3,970-char mouth or mouth_vocab='lexicon' (141 chars).

Learner = the core's M2 prediction form used as an inverse map (sound -> muscles): p = clip(x W, 0, 1),
dW = eta x^T (t - p) / |x|_1, trained on N random 4-muscle babbles heard in the OWN voice (bench 'self' offset).
Production of a heard lexicon char (partner voice 'other', or 'other2'): the 4 muscles with the largest p.
Also reported: (i) random babbling alone (how many lexicon chars a random babble hits), (ii) a ridge least-squares
inverse (best linear map), (iii) an oracle decoder that knows the 32 muscle patterns (information bound).
Run: python3 -I babble_toy.py
"""
import sys, itertools, json
import numpy as np

sys.path.insert(0, 'v2/bench')
from physics import Voice, corpus_chars       # noqa: E402  (read-only import)
from world import LEXICON_CHARS               # noqa: E402

chars = corpus_chars()
lex = sorted(LEXICON_CHARS)
out = {}
for seed in (0, 1, 2):
    v = Voice(chars, seed=seed, physics='lawful')
    rng = np.random.default_rng(100 + seed)
    nm = v.n_muscles

    def feat(g):
        x = np.asarray(g, float).ravel() / 10.0
        return x

    def produce(W, g):
        p = feat(g) @ W
        return tuple(sorted(int(m) for m in np.argsort(-p)[:4]))

    def score(decoder, voice_who, vocab):
        if vocab == 'lexicon':
            v.set_mouth_vocab(LEXICON_CHARS)
        ok = 0
        for c in lex:
            s = decoder(v.hear(c, voice_who))
            ok += v.decode(s) == c
        if vocab == 'lexicon':                    # restore the full mouth
            v.NEED = v._NEED_full; v._mouth_cache = {}
            del v._chars_dec
        return ok / len(lex)

    res = {}
    # (i) random babbling alone: distinct lexicon chars produced (clear or mumbled) by N random babbles
    for vocab in ('full', 'lexicon'):
        if vocab == 'lexicon':
            v.set_mouth_vocab(LEXICON_CHARS)
        r2 = np.random.default_rng(7 + seed); hit = set(); curve = {}
        for n in range(1, 20001):
            s = tuple(sorted(int(m) for m in r2.choice(nm, 4, replace=False)))
            c = v.decode(s)
            if c in LEXICON_CHARS:
                hit.add(c)
            if n in (300, 1000, 3000, 10000, 20000):
                curve[n] = round(len(hit) / len(lex), 3)
        if vocab == 'lexicon':
            v.NEED = v._NEED_full; v._mouth_cache = {}; del v._chars_dec
        res[f'random_babble_{vocab}'] = curve
    # (ii) delta-rule inverse map from N own-voice babbles
    Wd = np.zeros((48, nm)); X = []; T = []
    n_done = 0
    for N in (100, 300, 1000, 3000):
        while n_done < N:
            s = tuple(sorted(int(m) for m in rng.choice(nm, 4, replace=False)))
            x = feat(v.hear_set(s)); t = np.zeros(nm); t[list(s)] = 1
            for _ in range(1):
                p = np.clip(x @ Wd, 0, 1)
                Wd += 0.35 * np.outer(x, t - p) / max(x.sum(), 1e-9)
            X.append(x); T.append(t); n_done += 1
        Xa, Ta = np.array(X), np.array(T)
        Wr = np.linalg.solve(Xa.T @ Xa + 1.0 * np.eye(48), Xa.T @ Ta)
        res[f'delta_N{N}'] = {who: {voc: round(score(lambda g: produce(Wd, g), who, voc), 3)
                                    for voc in ('full', 'lexicon')} for who in ('other', 'other2')}
        res[f'ridge_N{N}'] = {who: {voc: round(score(lambda g: produce(Wr, g), who, voc), 3)
                                    for voc in ('full', 'lexicon')} for who in ('other', 'other2')}
    # (iii) oracle: knows the 32 patterns; picks the 4-set whose clipped sum best matches the heard grid (exhaustive)
    sets = list(itertools.combinations(range(nm), 4))
    S = np.stack([np.clip(v.mpat[list(s)].sum(0), 0, 10).ravel() for s in sets]).astype(np.int16)

    def oracle(g):
        d = np.abs(S - np.asarray(g, np.int16).ravel()[None, :]).sum(1)
        return sets[int(np.argmin(d))]
    res['oracle'] = {who: round(score(oracle, who, 'full'), 3) for who in ('other', 'other2')}
    out[seed] = res
    print('seed', seed, json.dumps(res), flush=True)
json.dump(out, open('babble_toy.json', 'w'), indent=1)
