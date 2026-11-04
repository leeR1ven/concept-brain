"""Theory, no brain run: can the Hebbian readout (PFC last layer -> next first-layer input) pick B after A?
The last layer's state for a context c is a set of units; a unit of A's state is also in another context's state with
probability o (overlap). A link i -> X grows with every time X followed a context whose state held i. With each copy's
usual level subtracted (二.4), what remains for X after A is (n(A,X) - o n(A,X)) plus the fluctuation of which other
contexts happen to share unit i: variance o(1-o) sum_c n(c,X)^2 per unit, averaged over the Kp units of A's state that
reach X's copy. Signal-to-noise for B against its worst competitor X:
   SNR = (n(A,B) - n(A,X)) (1-o) / sqrt((var_B + var_X) / Kp)
Counts n are bigram counts (previous syllable = context) in the training dialogues, scaled to N dialogues heard.
With a brain given, o and Kp are measured on it instead of assumed: it hears a few dialogues (nothing learned), its
last-layer states are kept per tick; o = the share of one tick's lit units also lit at another tick at least 10
ticks away (different contexts), averaged; Kp = lit last-layer units per tick x the share of them that reach one copy.
usage: python -u -m core.snr [ckpt or 'newborn' [key=value]]"""
import json
import sys

import numpy as np
import torch

from .world import phon
from .world.corpus import Dialogues


def measure(src, over):
    from .config import config
    from .corpus_life import CorpusLife
    if src == 'newborn':
        life = CorpusLife(config(**{'levels_per_s': 4, 'n_brake': 20, 'birth_dialogues': 80, **over}), 0, 'data')
    else:
        life = torch.load(src, weights_only=False)['life']
    b, p = life.brain, life.brain.pfc
    b.learning = False
    S, E = [], []
    n0 = life.env.n_ex
    while life.env.n_ex - n0 < 4:
        life.tick()
        S.append(p.L4.on.clone())
        E.append(float(p.L1ext.on.sum()))
    S = torch.stack(S).float()
    E = np.array(E)
    loud = torch.as_tensor(E > np.median(E), device=S.device)            # ticks with sound
    X = S[loud]
    k = X.sum(1)
    G = X @ X.T                                                          # shared lit units between ticks
    T = len(X)
    idx = torch.nonzero(loud).squeeze(1)
    far = (idx[:, None] - idx[None, :]).abs() >= 10
    o = float((G / k[:, None].clamp_min(1))[far].mean())
    L = p.L41
    reach = float((L.idx < L.n_post).sum(1).float().mean()) / L.n_post  # share of copies one last-layer unit reaches
    return dict(o=round(o, 3), lit_per_tick=round(float(k.mean()), 1), reach=round(reach, 4),
                Kp=round(float(k.mean()) * reach, 1), ticks=T)


def main():
    meas = None
    if len(sys.argv) > 1:
        over = {k: json.loads(v) for k, v in (a.split('=', 1) for a in sys.argv[2:] if '=' in a)}
        meas = measure(sys.argv[1], over)
    D = Dialogues.load('data/train.npz')
    s = D.syl.astype(np.int64)
    keep = np.ones(len(s) - 1, bool)
    st = D.turn_off[(D.turn_off > 0) & (D.turn_off < len(s))]
    keep[st - 1] = False
    n = int(s.max()) + 1
    M = np.zeros((n, n))
    np.add.at(M, (s[:-1][keep], s[1:][keep]), 1.0)
    per_dlg = 1.0 / D.n                                    # counts per dialogue heard
    sq = (M ** 2).sum(0)                                   # sum_c n(c,X)^2 per X
    bases = phon.load_inventory()
    name = lambda x: bases[x // 5] + str(x % 5 + 1)
    off = M.copy()
    np.fill_diagonal(off, 0)
    order = np.argsort(-off, axis=None)
    pairs, used = [], set()
    for k in order:
        a, b = divmod(int(k), n)
        if b not in used:
            pairs.append((a, b))
            used.add(b)
        if len(pairs) == 20:
            break
    out = []
    grid = [(meas['o'], meas['Kp'])] if meas else [(o, Kp) for o in (0.05, 0.15, 0.4) for Kp in (31, 126)]
    N = 20000                                              # (the ratio does not depend on N)
    f = N * per_dlg
    for o, Kp in grid:
        snr = []
        for a, b in pairs:
            var = o * (1 - o) * (sq - M[a] ** 2) * f * f            # fluctuation per unit, per candidate X
            sig = (M[a, b] - M[a]) * (1 - o) * f
            z = sig / np.sqrt((var[b] + var) / Kp + 1e-12)
            z[b] = np.inf
            snr.append(float(z.min()))
        out.append(dict(overlap=o, Kp=Kp, median_snr=round(float(np.median(snr)), 2),
                        pairs_snr_over_3=int((np.array(snr) > 3).sum())))
    names = [(name(a), name(b), round(M[a, b] / M[a].sum(), 2)) for a, b in pairs]
    print('SNR', sys.argv[1] if len(sys.argv) > 1 else '-',
          json.dumps(dict(measured=meas, dialogues_in_corpus=D.n, pairs=names, table=out)), flush=True)


if __name__ == '__main__':
    main()
