"""Quick check of a newborn (minutes). Prints one number per item:
  1 counts per layer (silence / speech) and the PFC layer ratios
  2 does every layer still tell syllables apart (cross-voice decoding of 34 syllables; same minus different cosine)
  3 does the PFC hold a heard line (cross-voice decoding of the 32 questions from L4, 0 / 6 / 12 ticks after the end)
  4 thinking a syllable: is it said (clear / garble / silent)
  5 one-shot memory: a question and answer heard once (learning on), then 300 ticks of other dialogue, then the
    question again: does the hippocampus light the answer's sound copies
Usage: python -m core.check [seed] [key=value ...]"""
import json
import sys

import numpy as np
import torch

from .config import config
from .life import Life, log
from .net import DEV
from .world import world as W
from .world.adapter import C1, C2, D1, NSYL, SID, SYL, line_script


def cos_matrix(A, B):
    A = A / A.norm(dim=1, keepdim=True).clamp_min(1e-9)
    B = B / B.norm(dim=1, keepdim=True).clamp_min(1e-9)
    return A @ B.T


def decode(T, X):
    """templates T [k, n], tests X [k, n] of the same k classes: share decoded right, same - different cosine"""
    S = cos_matrix(X, T)
    acc = float((S.argmax(1) == torch.arange(len(S), device=DEV)).float().mean())
    d = S.diag().mean()
    off = (S.sum() - S.diag().sum()) / (S.numel() - len(S))
    return round(acc, 3), round(float(d - off), 3)


def counts(life, n=20):
    b = life.brain
    names = b.count_names()
    rows, lab = [], []
    b.learning = False
    target = life.env.w.n_ex + n
    while life.env.w.n_ex < target:
        life.tick()
        rows.append(b.counts().float())
        lab.append(life.env.adult_speaking())
    R, lab = torch.stack(rows), torch.as_tensor(lab, device=DEV)
    out = {k: [round(float(R[~lab, i].mean()), 1), round(float(R[lab, i].mean()), 1)] for i, k in enumerate(names)}
    m = R.mean(0)
    ix = {k: i for i, k in enumerate(names)}
    out['ratios L2/L1 L3/L2 L4/L3'] = [round(float(m[ix[a]] / m[ix[b_]].clamp_min(1)), 2)
                                      for a, b_ in (('L2', 'L1'), ('L3', 'L2'), ('L4', 'L3'))]
    return out


def layer_patterns(life, script, wins):
    """mean activity of L1ext, L2, L3, L4 over each layer's own window of ticks"""
    p = life.brain.pfc
    rec = {k: [] for k in wins}
    life.play(script, mute=True, rec=lambda t: [rec[k].append(getattr(p, k).on.float()) for k in wins])
    return {k: torch.stack(rec[k])[a:b].mean(0) for k, (a, b) in wins.items()}


def distinct(life):
    wins = {'L1ext': (5, 11), 'L2': (6, 12), 'L3': (7, 13), 'L4': (8, 14)}
    pats = {v: [layer_patterns(life, [None] * 4 + [(x, v, 1.0)] * 3 + [None] * 12, wins) for x in range(NSYL)]
            for v in (C1, D1)}
    return {k: decode(torch.stack([q[k] for q in pats[C1]]), torch.stack([q[k] for q in pats[D1]])) for k in wins}


def hold(life):
    qs = [W.question(f, x) for f in range(4) for x in range(8)]
    out = {}
    pats = {}
    for v in (C1, D1):
        pats[v] = []
        for q in qs:
            script = [None] * 4 + line_script(q, v) + [None] * 16
            end = 4 + 3 * len(q)
            rec = []
            life.play(script, mute=True, rec=lambda t: rec.append(life.brain.pfc.L4.on.float()))
            H = torch.stack(rec)
            pats[v].append([H[end + 3 + lag: end + 6 + lag].mean(0) for lag in (0, 6, 12)])
    for i, lag in enumerate((0, 6, 12)):
        out[f'lag {lag}'] = decode(torch.stack([p[i] for p in pats[C1]]), torch.stack([p[i] for p in pats[D1]]))
    return out


def say(life):
    res = []
    for x in range(NSYL):
        syl, ev = life.play([None] * 19, think={6 + k: [x] for k in range(3)})
        res.append('silent' if not syl else 'clear' if syl == [SYL[x]] and all(e[2] == 's' for e in ev)
                   else 'garble' if '~' in syl else 'other')
    return {k: round(res.count(k) / NSYL, 2) for k in ('clear', 'garble', 'other', 'silent')}


def recall(life, n_pairs=8, gap_dialogues=3):
    """one-shot: Q (C2) + pause + A (D1) heard once with learning on; then other dialogue; then Q again (frozen)"""
    w = life.env.w
    pairs = sorted(w.held)[:n_pairs]
    b = life.brain
    hit, rank, ctl_rank, hint = [], [], [], []
    codes = life.codes

    def probe(q):
        rec = []
        life.play([None] * 4 + line_script(q, C2) + [None] * 14, mute=True,
                  rec=lambda t: rec.append(b.hip.h_int.clone()))
        return torch.stack(rec)[4 + 3 * len(q):].any(0)
    for i, (f, x) in enumerate(pairs):
        q, a = W.question(f, x), W.answer(f, x, w.props)
        target = SID[a[0]]
        ctl = pairs[(i + 1) % len(pairs)]
        life.play([None] * 4 + line_script(q, C2) + [None] * 6 + line_script(a, D1) + [None] * 4, mute=True,
                  learn=True)
        b.learning = True
        life.run_dialogues(gap_dialogues)
        b.learning = False
        h = probe(q)
        hint.append(int(h.sum()))
        score = torch.stack([h[cd].float().mean() if cd.numel() else torch.zeros((), device=DEV) for cd in codes])
        hit.append(float(score[target]))
        rank.append(int((score > score[target]).sum()) + 1)
        hc = probe(W.question(*ctl))                                   # never heard together: a control
        sc = torch.stack([hc[cd].float().mean() if cd.numel() else torch.zeros((), device=DEV) for cd in codes])
        ctl_rank.append(int((sc > sc[SID[W.answer(*ctl, w.props)[0]]]).sum()) + 1)
    return dict(h_int_after_Q=hint, answer_share_lit=[round(v, 2) for v in hit], answer_rank=rank,
                control_rank=ctl_rank, chance_rank=NSYL / 2)


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 and '=' not in sys.argv[1] else 0
    over = {k: json.loads(v) for k, v in (a.split('=', 1) for a in sys.argv[1:] if '=' in a)}
    c = config(**over)
    life = Life(c, seed)
    out = dict(config=over, seed=seed, registry=life.registry)
    for name, fn in (('counts', counts), ('distinct', distinct), ('hold', hold), ('say', say), ('recall', recall)):
        out[name] = fn(life)
        log(name, json.dumps(out[name], ensure_ascii=False))
    print(json.dumps(out, ensure_ascii=False))


if __name__ == '__main__':
    main()
