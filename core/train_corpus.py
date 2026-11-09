"""A long life in the big dialogue world, watched as it goes (same output as train.py). With ckpt=<path> the whole
life is saved every save_min minutes and a later run with the same ckpt continues it (chained jobs).
Usage: python -u -m core.train_corpus <data dir> [dialogues] [key=value ...]   e.g. s=4 ckpt="/x/life.pt"
"""
import json
import os
import sys
import time

import torch

from .config import config
from .corpus_life import CorpusLife
from .world.corpus import CurriculumWorld, Dialogues, common_order
from .life import log
from .net import DEV


def main():
    data = sys.argv[1]
    rest = sys.argv[2:]
    n = int(rest[0]) if rest and '=' not in rest[0] else 100000
    over = {k: json.loads(v) for k, v in (a.split('=', 1) for a in rest if '=' in a)}
    every, probe, seed = over.pop('every', 4000), over.pop('probe', 2000), over.pop('seed', 0)
    ckpt, save_min = over.pop('ckpt', None), over.pop('save_min', 120)
    load = over.pop('load', ckpt)                             # continue from this save (default: ckpt itself)
    pool, grow = over.pop('pool', 0), over.pop('grow_every', 5000)
    B = over.pop('batch', 1)
    save_at = list(over.pop('save_at', []))                   # extra saves <ckpt>_<n>.pt at these dialogue counts
    grow_concept, text = over.pop('grow_concept', 0), over.pop('text', None)
    if ckpt and os.path.exists(ckpt if load is None else load):
        load = load or ckpt
        st = torch.load(load, weights_only=False)
        life, start, nxt = st['life'], st['start'], st['next_probe']
        log(f'continued from {load}: {life.env.n_ex - start} dialogues so far')
        for k_, v_ in config(**{'levels_per_s': 4, 'n_brake': 20}).items():   # parameters added since the save
            life.c.setdefault(k_, v_)
        if pool and not isinstance(life.env, CurriculumWorld):  # from now on: common dialogues first, repeated
            old = life.env
            life.env = CurriculumWorld(life.train, life.coch, life.seed, common_order(life.train), pool, grow,
                                       start=old.i)
            life.env.n_ex, life.env.t = old.n_ex, old.t
            log(f'curriculum: the {pool} most ordinary dialogues first, pool doubling every {grow} dialogues')
    else:
        # the big world's ear: 4 thermometer levels per channel x s; its brake lines need 20 x s lit for full drive
        c = config(**{'levels_per_s': 4, 'n_brake': 20, 'birth_dialogues': 80, **over})
        life = CorpusLife(c, seed, data)
        start, nxt = life.env.n_ex, probe
        print('PROBE', 0, json.dumps(life.reply_ranks()), flush=True)
        print('NEXT', 0, json.dumps(life.next_ranks()), flush=True)
    if grow_concept and getattr(life.brain, 'concept', None) is None:
        from .grow import add_concept                        # the shared concept area grows into the living brain
        add_concept(life)
        log(f'concept area added at {life.env.n_ex - start} dialogues')
    if text and not getattr(life, 'text_added', False):       # read-aloud text joins the dialogues (1 per 3)
        from .world.corpus import concat, mixed_order
        tx = Dialogues.load(text)
        life.order = mixed_order(life.train, tx)
        life.train, life.text_added = concat(life.train, tx), True
        log(f'text added: {tx.n} items; now {life.train.n} items, one text item after every 3 dialogues')
    if B > 1:                                       # many children, one set of connections (core/batch.py)
        return run_batch(life, B, n, start, nxt, every, probe, ckpt, save_min, grow, save_at)
    b, w = life.brain, life.env
    names = b.count_names() + ['L4 stuck %', 'Hs stuck %']
    acc, k, t = torch.zeros(len(names), device=DEV), 0, 0
    saved_at = time.time()
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
            print('NEXT', nxt, json.dumps(life.next_ranks()), flush=True)
            nxt += probe
        if ckpt and time.time() - saved_at > save_min * 60:
            torch.save(dict(life=life, start=start, next_probe=nxt), ckpt + '.tmp')
            os.replace(ckpt + '.tmp', ckpt)
            saved_at = time.time()
            log(f'saved at {w.n_ex - start} dialogues')
    before = life.reply_ranks()
    for tb in b.hip.plastic():
        tb.w.zero_()
    b.hip.usual.zero_()
    print('HIPPOCAMPUS CLEARED', json.dumps(dict(before=before, after=life.reply_ranks())), flush=True)
    log('done')


def run_batch(life, B, n, start, nxt, every, probe, ckpt, save_min, grow, save_at=()):
    from .batch import Batch
    env = life.env
    base = env.n_ex
    pool = getattr(life, 'pool_now', None) or (env.pool(env.i) if isinstance(env, CurriculumWorld) else 200)
    order = getattr(life, 'order', None)
    order = common_order(life.train) if order is None else order
    worlds = [CurriculumWorld(life.train, life.coch, life.seed, order, pool,
                              max(1, grow // B), start=base * 7 + 1000003 * (k + 1)) for k in range(B)]
    bt = Batch(life, worlds)
    log(f'{B} children overhear side by side on one set of connections (pool {pool}, doubling every {grow} '
        f'dialogues in all)')
    names = life.brain.count_names() + ['L4 stuck %', 'Hs stuck %']
    acc, k, t, saved_at, t0 = torch.zeros(len(names), device=DEV), 0, 0, time.time(), time.time()
    h4, hs = [], []
    while base + bt.n_ex() - start < n:
        bt.tick()
        h4 = (h4 + [bt.L4.on[0].clone()])[-4:]
        hs = (hs + [bt.Hs.on[0].clone()])[-4:]
        st = [(h[-1] & h[0]).sum() / h[-1].sum().clamp_min(1) * 100 if len(h) == 4 else torch.zeros((), device=DEV)
              for h in (h4, hs)]
        acc += torch.cat([bt.counts(), torch.stack(st).float()])
        k += 1
        t += 1
        done = base + bt.n_ex() - start
        if t % every == 0:
            print('COUNT', done, f'{(time.time() - t0) / t * 1000:.0f} ms/tick for {B}',
                  json.dumps({nm: round(v) for nm, v in zip(names, (acc / k).tolist())}), flush=True)
            acc.zero_()
            k = 0
        if done >= nxt:
            print('PROBE', nxt, json.dumps(life.reply_ranks()), flush=True)
            print('NEXT', nxt, json.dumps(life.next_ranks()), flush=True)
            nxt += probe
        if ckpt and save_at and done >= save_at[0]:
            env.n_ex = base + bt.n_ex()
            torch.save(dict(life=life, start=start, next_probe=nxt), ckpt[:-3] + f'_{save_at[0]}.pt')
            log(f'saved a copy at {done} dialogues')
            save_at = save_at[1:]
        if ckpt and time.time() - saved_at > save_min * 60:
            env.n_ex = base + bt.n_ex()                 # the dialogues heard in all, kept with the life
            life.pool_now = worlds[0].pool(worlds[0].i)   # how far the pool has grown, kept with the life
            torch.save(dict(life=life, start=start, next_probe=nxt), ckpt + '.tmp')
            os.replace(ckpt + '.tmp', ckpt)
            saved_at = time.time()
            log(f'saved at {done} dialogues')
    if ckpt:                                            # and once more at the end
        env.n_ex = base + bt.n_ex()
        life.pool_now = worlds[0].pool(worlds[0].i)
        torch.save(dict(life=life, start=start, next_probe=nxt), ckpt + '.tmp')
        os.replace(ckpt + '.tmp', ckpt)
        log(f'saved at {base + bt.n_ex() - start} dialogues')
    log('done')


if __name__ == '__main__':
    main()
