"""Talk to a saved life: say sentences to it one after another (its state carries over, learning off) and see what
it thinks after each. Sentences are toned pinyin, separated by '/', e.g.
    python -u -m core.talk <ckpt> "ni3 hao3 / ni3 chi1 fan4 le5 ma5 / jin1 tian1 tian1 qi4 zen3 me5 yang4"
After each sentence, 12 quiet ticks: what the PFC thinks (its internally lit copies) and what the hippocampus
recalls, read back as syllables (the most common 400 syllables, beyond each one's usual level) in 3 windows."""
import sys
from collections import Counter

import torch

from .net import DEV
from .world import phon
from .world.voice import ADULTS


def main():
    life = torch.load(sys.argv[1], weights_only=False)['life']
    b, bases = life.brain, phon.load_inventory()
    idx = {bb: i for i, bb in enumerate(bases)}
    name = lambda s: phon.toned_name(bases, int(s))
    sid = [s for s, _ in Counter(life.train.syl[:400000].tolist()).most_common(400)]
    codes = {s: life.code(s) for s in sid}

    def read(H):
        return torch.stack([H[codes[s]].float().mean() if codes[s].numel() else torch.zeros((), device=DEV)
                            for s in sid])

    def listen(sylls, keep=False):
        src = [None] * 4 + [(x, k, ADULTS[1], 0.0, 0.0) for x in sylls for k in range(3)] + [None] * 12
        rec = {'pfc': [], 'hip': []}
        snap = None if keep else b.snapshot()
        was, b.learning = b.learning, False
        for x in src:
            b.step(torch.as_tensor(life.coch.a0([] if x is None else [x]), device=DEV), 0.0)
            rec['pfc'].append(b.pfc.x_int.clone())
            rec['hip'].append(b.hip.h_int.clone())
        b.learning = was
        if snap is not None:
            b.restore(snap)
        e = len(src) - 12
        return {k: [read(torch.stack(rec[k])[e + a:e + z].any(0)) for a, z in ((0, 4), (4, 8), (8, 12))]
                for k in rec}

    base_items = []                                   # each syllable's usual level: 20 held-out sentences
    D = life.heldout
    for d in range(D.n):
        tr = D.turns(d)
        if tr and 3 <= len(tr[0]) <= 12:
            base_items.append(listen([int(x) for x in tr[0]]))
        if len(base_items) >= 20:
            break
    base = {k: torch.stack([w for it in base_items for w in it[k]]).mean(0) for k in ('pfc', 'hip')}
    for sent in sys.argv[2].split('/'):
        sylls = []
        for tok in sent.split():
            tone = int(tok[-1]) if tok[-1].isdigit() else 5
            sylls.append(idx[tok.rstrip('12345')] * 5 + tone - 1)
        sc = listen(sylls, keep=True)                  # the conversation goes on: state is kept
        print(f'\nsaid:    {sent.strip()}')
        for k in ('pfc', 'hip'):
            for label, sub in (('', 0.0), (' (above usual)', base[k])):
                parts = []
                for w in sc[k]:
                    top = torch.topk(w - sub, 6)
                    parts.append(' '.join(name(sid[i]) for i, v in zip(top.indices.tolist(), top.values.tolist())
                                          if v > 0) or '-')
                print(f'{"thinks" if k == "pfc" else "recalls"}{label}: ' + '  |  '.join(parts), flush=True)


if __name__ == '__main__':
    main()
