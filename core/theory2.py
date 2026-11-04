"""Theory part 2, exact: what the PFC's readout would have learned after hearing the whole corpus, computed, not trained.
Hebbian learning counts: a link last-layer i -> first-layer copy j grows each time i is lit while j comes on next. If
the codes stay as at birth (the cap on the last layer's own links), then after the corpus
    count(i, j) = sum over pairs (A, B) of N(A, B) [i in code(A)] [j in copies(B)],   only for i -> j that exist,
and each link's weight is g(count), g from the rule (dw = eta / (1 + w) per event: w = sqrt(1 + 2 eta n) - 1).
Cue: another hearing of A (the ear differs). Each copy's push minus its usual level (the push averaged over contexts
as often as they occur); a syllable's score = the mean over its own copies. Is the best-scored syllable the one that
most often follows A?
  code(A)    last-layer units lit when B would come (A heard after quiet, the ticks B starts in)
  copies(B)  first-layer copies B lights from outside, heard alone after quiet
usage: python -u -m core.theory2 <ckpt or 'newborn'> [S=600] [g=sqrt|lin] [key=value]"""
import json
import sys
import time

import numpy as np
import torch

from .config import config
from .corpus_life import CorpusLife

PRE = 15


def main():
    src = sys.argv[1]
    over = {k: json.loads(v) for k, v in (a.split('=', 1) for a in sys.argv[2:] if '=' in a)}
    S, gname = over.pop('S', 600), over.pop('g', 'sqrt')
    save = over.pop('save', None)
    t0 = time.time()
    if src == 'newborn':
        life = CorpusLife(config(**{'levels_per_s': 4, 'n_brake': 20, 'birth_dialogues': 80, **over}), 0, 'data')
    else:
        life = torch.load(src, weights_only=False)['life']
    b, p, c = life.brain, life.brain.pfc, life.c
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
    cover = float(ok.mean())

    def hear(x, v='C1'):
        """copies x lights; last-layer units lit in the ticks B would start; and the last layer's push there"""
        rec = []

        def r(t):
            d4 = p.ff[2].drive(p.L3.prev) + p.L44.drive(p.L4.prev) - p.usual4      # what ranked the last layer
            rec.append((p.L1ext.on.clone(), p.L4.on.clone(), d4.half()))
        life.play([None] * PRE + [(int(x), k, v, 0.0, 0.0) for k in range(3)] + [None] * 6, rec=r)
        E = torch.stack([q[0] for q in rec])
        A4 = torch.stack([q[1] for q in rec])
        P4 = torch.stack([q[2] for q in rec])
        return E[PRE:PRE + 6].any(0) & ~E[PRE - 1], A4[PRE + 3:PRE + 6].any(0), P4[PRE + 3:PRE + 6]

    cp, cd, cd2, pu, pu2 = [], [], [], [], []
    for x in syl:
        e1, a1, q1 = hear(x)
        _, a2, q2 = hear(x)
        cp.append(e1)
        cd.append(a1)
        cd2.append(a2)
        pu.append(q1)
        pu2.append(q2)
    if save:
        np.savez_compressed(save, cp=torch.stack(cp).numpy(), cd=torch.stack(cd).numpy(),
                            cd2=torch.stack(cd2).numpy(), N=N, idx=p.L41.idx.cpu().numpy(), n_post=p.L41.n_post,
                            push=torch.stack(pu).numpy(), push2=torch.stack(pu2).numpy(),
                            eta=c['eta'], syl=syl)
    Cp = torch.stack(cp).float()                                   # [S, n1] copies
    Cd = torch.stack(cd).float()                                   # [S, n4] codes (training)
    Cd2 = torch.stack(cd2).float()                                 # [S, n4] codes (test: the ear differs)
    L = p.L41
    R = torch.zeros(L.n_pre, L.n_post + 1, dtype=torch.bool)
    R.scatter_(1, L.idx.cpu(), True)
    R = R[:, :-1]                                                  # which links exist
    Nt = torch.as_tensor(N, dtype=torch.float32)
    W = Cd.T @ (Nt @ Cp)                                           # [n4, n1] counts
    eta = c['eta']
    if gname == 'sqrt':
        W.mul_(2 * eta).add_(1).sqrt_().sub_(1)
    else:
        W.mul_(eta)
    W.mul_(R)
    pi = torch.as_tensor(N.sum(1) / N.sum(), dtype=torch.float32)  # how often each context occurs
    Dr = Cd2 @ W                                                   # [S, n1] push after each cue
    usual = pi @ Dr
    E = Dr - usual
    score = (E @ Cp.T) / Cp.sum(1).clamp_min(1)                    # [cue, syllable] mean over its copies
    best = N.argmax(1)
    pred = score.argmax(1).numpy()
    rank = (score > score[np.arange(S), best][:, None]).sum(1).numpy()
    w = N.sum(1) / N.sum()
    kp = float((Cd2 @ R.float()).sum(1).mean() / R.shape[1])       # lit units reaching one copy, on average
    out = dict(over=over, S=S, g=gname, corpus_covered=round(cover, 3),
               top1=round(float((w * (pred == best)).sum()), 3), top5=round(float((w * (rank < 5)).sum()), 3),
               unigram_top1=round(float((w * (N.sum(0).argmax() == best)).sum()), 3),
               mean_rank=round(float((w * rank).sum()), 1), code_size=round(float(Cd.sum(1).mean()), 1),
               copies=round(float(Cp.sum(1).mean()), 1), kp=round(kp, 1), secs=round(time.time() - t0))
    print('THEORY2', src, json.dumps(out), flush=True)


if __name__ == '__main__':
    main()
