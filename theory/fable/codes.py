"""Forward computation only (no learning): the newborn's codes, tick by tick, for the readout and stability theory.
For the S commonest syllables, heard alone after quiet (voice C1 twice: train / test with different ear noise; voice
D1 once), and for the first S_ctx syllables also heard right after another syllable (running speech, no gap):
  ext[h, s, t]   first-layer copies lit from outside, tick t of the window (t = 0 is the tick after sound onset)
  l4[h, s, t]    last-layer units lit                      push[h, s, t]  last-layer push (float32, ticks 3..7)
  e1[s, t]       thought drive on every copy (train hearing, ticks 4..7)
Bigram counts N over the S syllables, the readout's possible links (idx), and the thought budget are saved with them.
usage (from a directory holding core/ and data/):  python3 codes.py out.npz [S=600] [S_ctx=120] [key=value]"""
import json
import sys
import time

import numpy as np
import torch

from core.config import config
from core.corpus_life import CorpusLife

PRE = 12          # quiet ticks before the sound
WIN = 11          # window: ticks -1 .. 9 relative to sound onset (index t = tick + 1)


def main():
    out = sys.argv[1]
    over = {k: json.loads(v) for k, v in (a.split('=', 1) for a in sys.argv[2:] if '=' in a)}
    S, S_ctx = over.pop('S', 600), over.pop('S_ctx', 120)
    t0 = time.time()
    life = CorpusLife(config(**{'levels_per_s': 4, 'n_brake': 20, 'birth_dialogues': 80, **over}), 0, 'data')
    b, p, c = life.brain, life.brain.pfc, life.c
    D = life.train
    s = D.syl.astype(np.int64)
    keep = np.ones(len(s) - 1, bool)
    st = D.turn_off[(D.turn_off > 0) & (D.turn_off < len(s))]
    keep[st - 1] = False                                      # no pair across a turn boundary
    f = np.bincount(s)
    syl = np.argsort(-f)[:S]
    ix = -np.ones(int(s.max()) + 1, np.int64)
    ix[syl] = np.arange(S)
    a_, b_ = ix[s[:-1][keep]], ix[s[1:][keep]]
    ok = (a_ >= 0) & (b_ >= 0)
    N = np.zeros((S, S))
    np.add.at(N, (a_[ok], b_[ok]), 1.0)
    n1 = b.n1

    def hear(seq, v='C1', want_push=False):
        """play seq (3 ticks per syllable, no gap) after quiet; record the window around the LAST syllable's onset"""
        rec = []
        onset = PRE + 3 * (len(seq) - 1)

        def r(t):
            if onset - 1 <= t <= onset + WIN - 2:
                d4 = (p.ff[2].drive(p.L3.prev) + p.L44.drive(p.L4.prev) - p.usual4).float()
                rec.append((p.L1ext.on.clone(), p.L4.on.clone(), p.x_int.clone(), d4, p.e1.clone()))
        life.play([None] * PRE + [(int(x), k, v, 0.0, 0.0) for x in seq for k in range(3)] + [None] * (WIN - 2),
                  rec=r)
        E = np.packbits(torch.stack([q[0] for q in rec]).numpy(), axis=1)
        A4 = np.packbits(torch.stack([q[1] for q in rec]).numpy(), axis=1)
        XI = np.packbits(torch.stack([q[2] for q in rec]).numpy(), axis=1)
        P = torch.stack([q[3] for q in rec[4:9]]).numpy() if want_push else None   # ticks 3..7
        E1 = torch.stack([q[4] for q in rec[5:9]]).numpy() if want_push else None  # ticks 4..7
        return E, A4, XI, P, E1

    ext, l4, xi, push, e1 = [], [], [], [], []
    for i, x in enumerate(syl):
        rows = [hear([x], 'C1', True), hear([x], 'C1', True), hear([x], 'D1', False)]
        ext.append([q[0] for q in rows])
        l4.append([q[1] for q in rows])
        xi.append([q[2] for q in rows])
        push.append([rows[0][3], rows[1][3]])
        e1.append(rows[0][4])
        if i % 50 == 0:
            print(f'{i}/{S} {time.time() - t0:.0f}s', flush=True)
    ctx_ext, ctx_l4 = [], []
    for i in range(S_ctx):
        rows = [hear([syl[(i + 7) % S], syl[i]]), hear([syl[(i + 13) % S], syl[i]])]
        ctx_ext.append([q[0] for q in rows])
        ctx_l4.append([q[1] for q in rows])
    np.savez_compressed(out, syl=syl, N=N, n1=n1, ext=np.array(ext), l4=np.array(l4), xi=np.array(xi),
                        push=np.array(push, np.float32), e1=np.array(e1, np.float32),
                        ctx_ext=np.array(ctx_ext), ctx_l4=np.array(ctx_l4),
                        idx=p.L41.idx.cpu().numpy(), n_post=p.L41.n_post, loop_idx=p.loop.idx.cpu().numpy(),
                        loop_w=p.loop.w.cpu().numpy(), ext_usual=p.ext_usual, budget=p.budget,
                        theta1=c['theta1'], context=c['context'], eta=c['eta'], w_ref=c['w_ref'],
                        cap44=p.L44.cap, pre=PRE, win=WIN, secs=time.time() - t0)
    print('CODES saved', out, f'{time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
