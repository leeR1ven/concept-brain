"""A long life in the big dialogue world, watched as it goes (same output as train.py).
Usage: python -u -m core.train_corpus <data dir> [dialogues] [key=value ...]   e.g. s=4"""
import json
import sys
import time

import torch

from .config import config
from .corpus_life import CorpusLife
from .life import log
from .net import DEV


def main():
    data = sys.argv[1]
    rest = sys.argv[2:]
    n = int(rest[0]) if rest and '=' not in rest[0] else 100000
    over = {k: json.loads(v) for k, v in (a.split('=', 1) for a in rest if '=' in a)}
    every, probe, seed = over.pop('every', 4000), over.pop('probe', 2000), over.pop('seed', 0)
    # the big world's ear: 4 thermometer levels per channel x s; its brake lines need 20 x s lit for full drive
    c = config(**{'levels_per_s': 4, 'n_brake': 20, 'birth_dialogues': 80, **over})
    life = CorpusLife(c, seed, data)
    b, w = life.brain, life.env
    names = b.count_names() + ['L4 stuck %', 'Hs stuck %']
    print('PROBE', 0, json.dumps(life.reply_ranks()), flush=True)
    acc, k, t, start, nxt = torch.zeros(len(names), device=DEV), 0, 0, w.n_ex, probe
    h4, hs = [], []
    b.learning = True
    t0 = time.time()
    while w.n_ex - start < n:
        life.tick()
        h4 = (h4 + [b.pfc.L4.on.clone()])[-4:]
        hs = (hs + [b.hip.Hs.on.clone()])[-4:]
        st = [(h[-1] & h[0]).sum() / h[-1].sum().clamp_min(1) * 100 if len(h) == 4 else torch.zeros((), device=DEV)
              for h in (h4, hs)]
        acc += torch.cat([b.counts().float(), torch.stack(st).float()])
        k += 1
        t += 1
        if t % every == 0:
            print('COUNT', w.n_ex - start, f'{(time.time() - t0) / t * 1000:.0f} ms/tick',
                  json.dumps({nm: round(v) for nm, v in zip(names, (acc / k).tolist())}), flush=True)
            acc.zero_()
            k = 0
        if w.n_ex - start >= nxt:
            print('PROBE', nxt, json.dumps(life.reply_ranks()), flush=True)
            nxt += probe
    before = life.reply_ranks()
    for tb in b.hip.plastic():
        tb.w.zero_()
    b.hip.usual.zero_()
    print('HIPPOCAMPUS CLEARED', json.dumps(dict(before=before, after=life.reply_ranks())), flush=True)
    log('done')


if __name__ == '__main__':
    main()
