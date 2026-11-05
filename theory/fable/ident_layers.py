"""Where is the identification ceiling (forward computation only, newborn)? The same syllables, voices and split as
codes_multi.py, recorded at three levels: the ear's raw input bits (the 3 ticks of the syllable, concatenated), the
copies (ticks 1..4) and the last layer (ticks 4..6). Nearest-class-mean identification per level, both weightings,
and identify-then-lookup. ear=off removes the ear's per-tick jitter and noise (world-model choices, not the
architecture); the corpus's per-turn F0 offset and level variation stay.
usage (from a directory holding core/ and data/):  python3 ident_layers.py [S=300] [R=3] [ear=on|off] [key=value]"""
import json
import sys
import time

import numpy as np
import torch

from core.config import config
from core.corpus_life import CorpusLife

PRE = 12
TRAIN_VOICES = ('C1', 'C2', 'D1')
TEST_VOICE = 'D2'


class NoNoise:
    def uniform(self, a, b, n):
        return np.ones(n)

    def normal(self, m, s, n):
        return np.zeros(n)


def cos_sim(A, B):
    An = A / np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-9)
    Bn = B / np.maximum(np.linalg.norm(B, axis=1, keepdims=True), 1e-9)
    return An @ Bn.T


def main():
    over = {k: (v if v in ('on', 'off') else json.loads(v)) for k, v in (a.split('=', 1) for a in sys.argv[1:])}
    S, R, ear = over.pop('S', 300), over.pop('R', 3), over.pop('ear', 'on')
    t0 = time.time()
    life = CorpusLife(config(**{'levels_per_s': 4, 'n_brake': 20, 'birth_dialogues': 80, **over}), 0, 'data')
    b, p = life.brain, life.brain.pfc
    if ear == 'off':
        life.coch.rng = NoNoise()
    D = life.train
    s = D.syl.astype(np.int64)
    keep = np.ones(len(s) - 1, bool)
    st = D.turn_off[(D.turn_off > 0) & (D.turn_off < len(s))]
    keep[st - 1] = False
    f = np.bincount(s)
    syl = np.argsort(-f)[:S]
    ix = -np.ones(int(s.max()) + 1, np.int64)
    ix[syl] = np.arange(S)
    a_, b_ = ix[s[:-1][keep]], ix[s[1:][keep]]
    ok = (a_ >= 0) & (b_ >= 0)
    N = np.zeros((S, S))
    np.add.at(N, (a_[ok], b_[ok]), 1.0)
    piA = N.sum(1) / N.sum()
    best = N.argmax(1)
    rng = np.random.default_rng(11)

    def hear(x, v):
        off, lev = rng.uniform(-3.0, 1.0), rng.uniform(-2.0, 2.0)
        rec = []
        srcs = [None] * PRE + [(int(x), k, v, off + rng.normal(0, 0.3), lev) for k in range(3)] + [None] * 6
        raw = np.concatenate([life.coch.a0([q]) for q in srcs[PRE:PRE + 3]]).astype(np.float32)

        def r(t):
            if PRE + 1 <= t <= PRE + 6:
                rec.append((p.L1ext.on.clone(), p.L4.on.clone()))
        life.play(srcs, rec=r)
        E = torch.stack([q[0] for q in rec[0:4]]).any(0).numpy().astype(np.float32)
        A = torch.stack([q[1] for q in rec[3:6]]).any(0).numpy().astype(np.float32)
        return raw, E, A

    levels = ('input bits', 'copies', 'last layer')
    train = {l: [] for l in levels}
    tests = {'fresh': {l: [] for l in levels}, 'voice4': {l: [] for l in levels}}
    tC = {'fresh': [], 'voice4': []}
    for i, x in enumerate(syl):
        exs = [hear(x, v) for v in TRAIN_VOICES for _ in range(R)]
        for l, q in zip(levels, zip(*exs)):
            train[l].append(np.mean(q, 0))                 # class mean (the raw bits are not stored)
        for v in TRAIN_VOICES:
            for l, q in zip(levels, hear(x, v)):
                tests['fresh'][l].append(q)
            tC['fresh'].append(i)
        for _ in range(2):
            for l, q in zip(levels, hear(x, TEST_VOICE)):
                tests['voice4'][l].append(q)
            tC['voice4'].append(i)
        if i % 50 == 0:
            print(f'{i}/{S} {time.time() - t0:.0f}s', flush=True)
    out = dict(S=S, R_per_voice=R, ear=ear, unigram_top1_weighted=round(float((piA * (N.sum(0).argmax() == best)).sum()), 3))
    for l in levels:
        mu = np.stack(train[l])
        rep = dict(size=round(float(mu.sum(1).mean()), 1))
        for tname in tests:
            Tq = np.stack(tests[tname][l])
            c = np.array(tC[tname])
            ident = cos_sim(Tq, mu).argmax(1)
            w = piA[c] / piA[c].sum()
            rep[tname] = dict(identified_unweighted=round(float((ident == c).mean()), 3),
                              identified_weighted=round(float((w * (ident == c)).sum()), 3),
                              lookup_top1_weighted=round(float((w * (best[ident] == best[c])).sum()), 3))
        out[l] = rep
    print('IDENT', json.dumps(dict(**out, secs=round(time.time() - t0))), flush=True)


if __name__ == '__main__':
    main()
