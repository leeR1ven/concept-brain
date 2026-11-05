"""Many hearings per syllable (forward computation only, newborn): the S commonest syllables, each heard alone after
quiet R times in each of three adult voices (C1, C2, D1) with the corpus's own per-turn variation (F0 offset, level),
plus held-out hearings: R_test fresh ones per training voice and R_voice in the fourth voice (D2).
Saved per hearing: the copies lit (ticks 1..4 after onset) and the last-layer units lit (ticks 4..6), bit-packed.
usage (from a directory holding core/ and data/):  python3 codes_multi.py out.npz [S=300] [R=6] [key=value]"""
import json
import sys
import time

import numpy as np
import torch

from core.config import config
from core.corpus_life import CorpusLife
from core.world import phon

PRE = 12
TRAIN_VOICES = ('C1', 'C2', 'D1')
TEST_VOICE = 'D2'


def main():
    out = sys.argv[1]
    over = {k: json.loads(v) for k, v in (a.split('=', 1) for a in sys.argv[2:] if '=' in a)}
    S, R = over.pop('S', 300), over.pop('R', 6)
    R_test, R_voice = over.pop('R_test', 2), over.pop('R_voice', 4)
    t0 = time.time()
    life = CorpusLife(config(**{'levels_per_s': 4, 'n_brake': 20, 'birth_dialogues': 80, **over}), 0, 'data')
    b, p = life.brain, life.brain.pfc
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
    bases = phon.load_inventory()
    names = [phon.toned_name(bases, int(x)) for x in syl]
    rng = np.random.default_rng(11)
    n1 = b.n1

    def hear(x, v):
        off, lev = rng.uniform(-3.0, 1.0), rng.uniform(-2.0, 2.0)     # a turn's F0 offset and level, as in the corpus
        rec = []

        def r(t):
            if PRE + 1 <= t <= PRE + 6:
                rec.append((p.L1ext.on.clone(), p.L4.on.clone()))
        life.play([None] * PRE + [(int(x), k, v, off + rng.normal(0, 0.3), lev, 0.0)[:5] for k in range(3)]
                  + [None] * 6, rec=r)
        E = torch.stack([q[0] for q in rec[0:4]]).any(0).numpy()       # ticks 1..4
        A = torch.stack([q[1] for q in rec[3:6]]).any(0).numpy()       # ticks 4..6
        return np.packbits(np.stack([E, A]), axis=-1)

    train = np.zeros((S, len(TRAIN_VOICES), R, 2, (n1 + 7) // 8), np.uint8)
    test = np.zeros((S, len(TRAIN_VOICES), R_test, 2, (n1 + 7) // 8), np.uint8)
    voice = np.zeros((S, R_voice, 2, (n1 + 7) // 8), np.uint8)
    for i, x in enumerate(syl):
        for vi, v in enumerate(TRAIN_VOICES):
            for k in range(R):
                train[i, vi, k] = hear(x, v)
            for k in range(R_test):
                test[i, vi, k] = hear(x, v)
        for k in range(R_voice):
            voice[i, k] = hear(x, TEST_VOICE)
        if i % 25 == 0:
            print(f'{i}/{S} {time.time() - t0:.0f}s', flush=True)
    np.savez_compressed(out, syl=syl, names=np.array(names), N=N, n1=n1, train=train, test=test, voice=voice,
                        covered=float(ok.mean()), budget=p.budget, secs=time.time() - t0)
    print('MULTI saved', out, f'{time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
