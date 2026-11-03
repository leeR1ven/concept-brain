"""What does the model think after hearing a sentence? Loads a saved life (train_corpus ckpt), plays held-out
sentences to it (frozen, state restored), and translates what is lit inside back into syllables: the PFC's
internally lit copies (its thinking range) and the hippocampus's recall, in three windows after the sentence.
Each syllable's code = the sound copies it lights when heard; a window's content = the syllables whose codes are
most covered (beyond what that syllable usually gets). Also prints the real reply, for judging by eye.
Usage: python -u -m core.think <ckpt> [n_sentences]"""
import sys
from collections import Counter

import numpy as np
import torch

from .net import DEV
from .world import phon
from .world.voice import ADULTS


def main():
    life = torch.load(sys.argv[1], weights_only=False)['life']
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    b, bases = life.brain, phon.load_inventory()
    name = lambda s: phon.toned_name(bases, int(s))
    freq = Counter(life.train.syl[:400000].tolist())
    vocab = [s for s, _ in freq.most_common(400)]
    codes = {s: life.code(s) for s in vocab}
    sid = list(codes)
    print(f'{len(sid)} most common syllables as the vocabulary for reading thoughts', flush=True)

    def read(H):
        sc = torch.stack([H[codes[s]].float().mean() if codes[s].numel() else torch.zeros((), device=DEV)
                          for s in sid])
        return sc

    D = life.heldout
    items = []
    for d in range(D.n):
        tr = D.turns(d)
        if len(tr) < 2 or not (4 <= len(tr[0]) <= 12):
            continue
        prompt, reply = tr[0], tr[1]
        src = [None] * 4 + [(int(x), k, ADULTS[0], 0.0, 0.0) for x in prompt for k in range(3)] + [None] * 12
        rec = {'pfc': [], 'hip': []}
        life.play(src, rec=lambda t: (rec['pfc'].append(b.pfc.x_int.clone()), rec['hip'].append(b.hip.h_int.clone())))
        e = len(src) - 12
        sc = {k: [read(torch.stack(rec[k])[e + a:e + z].any(0)) for a, z in ((0, 4), (4, 8), (8, 12))] for k in rec}
        items.append((prompt, reply, sc))
        if len(items) >= n:
            break
    base = {k: torch.stack([w for _, _, sc in items for w in sc[k]]).mean(0) for k in ('pfc', 'hip')}
    for prompt, reply, sc in items:                          # beyond what each syllable usually gets
        print(f'\nheard:   {" ".join(name(s) for s in prompt)}\nreply:   {" ".join(name(s) for s in reply)}')
        for k in ('pfc', 'hip'):
            parts = []
            for w in sc[k]:
                top = torch.topk(w - base[k], 6)
                parts.append(' '.join(name(sid[i]) for i, v in zip(top.indices.tolist(), top.values.tolist())
                                      if v > 0) or '-')
            print(f'{"thinks:" if k == "pfc" else "recalls:"} ' + '  |  '.join(parts))


if __name__ == '__main__':
    main()
