"""Forward-only sweep of the sound area's fixed parameters (the coordinator's request): where between the ear's bits
(identification 81%) and the copies (38%) is the syllable's identity lost, and which birth parameter recovers it?
One configuration per run (config overrides as key=value). Same split as ident_layers.py: S syllables, 3 training
voices x R hearings (class means), 1 fresh hearing per training voice + 2 of the fourth voice as tests. Reported:
identification (nearest class mean; weighted by cue frequency / unweighted) at the ear's bits, at the copies (all,
and each sound layer's copies alone), at the last layer; the copy counts per sound layer; and the first-layer
density (the designer asked for the PFC to stay under 5 % active).
usage (from a directory holding core/ and data/):  python3 sweep_sound.py [S=150] [R=2] [key=value ...]"""
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


def cos_sim(A, B):
    An = A / np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-9)
    Bn = B / np.maximum(np.linalg.norm(B, axis=1, keepdims=True), 1e-9)
    return An @ Bn.T


def main():
    over = {k: json.loads(v) for k, v in (a.split('=', 1) for a in sys.argv[1:])}
    S, R = over.pop('S', 150), over.pop('R', 2)
    t0 = time.time()
    c = config(**{'levels_per_s': 4, 'n_brake': 20, 'birth_dialogues': 80, **over})
    life = CorpusLife(c, 0, 'data')
    b, p, ch = life.brain, life.brain.pfc, life.brain.chains
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
    rng = np.random.default_rng(11)
    n1 = b.n1
    seg = {l: (a, z) for kind, l, a, z in ch.segs if kind == 'sound'}

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

    levels = ['input bits', 'copies', 'last layer'] + [f'copies of sound layer {l + 1}' for l in sorted(seg)]

    def split(q):
        raw, E, A = q
        return [raw, E, A] + [E[a:z] for l, (a, z) in sorted(seg.items())]

    train = {l: [] for l in levels}
    tests = {'fresh': {l: [] for l in levels}, 'voice4': {l: [] for l in levels}}
    tC = {'fresh': [], 'voice4': []}
    per_hearing = []
    for i, x in enumerate(syl):
        exs = [split(hear(x, v)) for v in TRAIN_VOICES for _ in range(R)]
        per_hearing += [e[1].sum() for e in exs]
        for l, q in zip(levels, zip(*exs)):
            train[l].append(np.mean(q, 0))
        for v in TRAIN_VOICES:
            for l, q in zip(levels, split(hear(x, v))):
                tests['fresh'][l].append(q)
            tC['fresh'].append(i)
        for _ in range(2):
            for l, q in zip(levels, split(hear(x, TEST_VOICE))):
                tests['voice4'][l].append(q)
            tC['voice4'].append(i)
        if i % 50 == 0:
            print(f'{i}/{S} {time.time() - t0:.0f}s', flush=True)
    out = dict(over=over, S=S, R=R, n1=n1, copies_per_sound_layer=[int(len(ch.sound_idx[l])) for l in range(len(ch.sound_idx))],
               birth_density_L1ext=round(float(b.birth_density), 4),
               copies_lit_per_hearing=round(float(np.mean(per_hearing)), 1),
               L1ext_density_while_heard=round(float(np.mean(per_hearing)) / n1 / 4, 4))   # union of 4 ticks / 4
    for l in levels:
        mu = np.stack(train[l])
        if mu.shape[1] == 0:
            out[l] = 'no units'
            continue
        rep = dict(size=round(float(mu.sum(1).mean()), 1))
        for tname in tests:
            Tq = np.stack(tests[tname][l])
            cc = np.array(tC[tname])
            ident = cos_sim(Tq, mu).argmax(1)
            w = piA[cc] / piA[cc].sum()
            rep[tname] = dict(w=round(float((w * (ident == cc)).sum()), 3), u=round(float((ident == cc).mean()), 3))
        out[l] = rep
    print('SWEEP', json.dumps(dict(**out, secs=round(time.time() - t0))), flush=True)


if __name__ == '__main__':
    main()
