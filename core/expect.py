"""Did it learn the regularities of what it hears? (The test used for infants before they can speak: Saffran, Aslin &
Newport 1996 - after listening, they react differently to the familiar order of the same syllables; in the brain the
mismatch response to sounds that break a learned sequence.) The PFC learns "what comes next" (ARCH 三 前额叶), so
before a sound arrives some of its copies are already lit from inside (thought, or recalled by the hippocampus). When
it arrives, the share of its newly lit copies that were lit inside (in the same tick, predicted from the tick before,
or one tick earlier) is what the brain itself expected. Nothing is read out by an outside reader, and nothing assumes what it should think: only the same sounds in
two orders are compared, so a newborn shows no difference, and a larger difference = more learned.
  heard   lines heard most in training, their own order vs the same syllables shuffled
  novel   never-heard lines, own order vs shuffled (the regularities carried over to new lines)
  reply   in dialogues heard in training, the real reply after a turn vs the reply of another dialogue
          (the same replies in both, only the pairing differs)
  once    never-heard lines, half heard once (learning on, as in life): expected share again minus before, for the
          once-heard half minus for the other half; soon, and after 20 more dialogues
  drill   learning by repetition (also the check of the test itself): 20 never-heard lines each heard `reps` times;
          the change of their own-order-minus-shuffled gap, against 20 control lines (drilled / control / difference)
Each: mean expected share (in %) real / control, their difference, and its standard error over the items.
usage: python -u -m core.expect <ckpt or 'newborn'> [only=drill] [reps=20] [lead_drill=4] [lead_test=4] [seed=k] [g44=x] [l41_scale=f] [usual0=1] [hip_scale=f] [key=value]"""
import json
import sys
import time

import numpy as np
import torch

from .config import config
from .corpus_life import CorpusLife
from .exam import first_lines
from .net import DEV
from .world.corpus import common_order

N = 40


def counts(b):
    """this tick: external copies newly on; of them, lit inside (PFC thought / hippocampal recall) now (what was
    predicted for this tick from the tick before) or one tick earlier"""
    p, h = b.pfc, b.hip
    rose = p.L1ext.rose()
    tp, th = p.x_int | p.L1.prev, h.h_int | h.H1.prev
    return torch.stack([rose.sum(), (rose & tp).sum(), (rose & th).sum(), (rose & (tp | th)).sum()]).float()


def listen(life, srcs, w, learn=False):
    """the summed counts over ticks w (slice); frozen (states restored) unless learn"""
    b, rec = life.brain, []
    if learn:
        was, b.learning = b.learning, True
        for x in srcs:
            b.step(torch.as_tensor(life.coch.a0([] if x is None else [x]), device=DEV), 0.0)
            rec.append(counts(b))
        b.learning = was
    else:
        life.play(srcs, rec=lambda t: rec.append(counts(b)))
    return torch.stack(rec)[w].sum(0).cpu().numpy()


def line_srcs(ln, v, lead=4):
    return [None] * lead + [(x, k, v, 0.0, 0.0) for x in ln for k in range(3)] + [None] * 4


def share(c):
    """expected share (%) by PFC thought, hippocampal recall, either"""
    return 100.0 * c[1:] / max(1.0, c[0])


def compare(a, b):
    """a, b: [items, 3] shares; mean real, mean control, mean difference, its standard error"""
    d = a - b
    return {k: [round(float(a[:, i].mean()), 2), round(float(b[:, i].mean()), 2), round(float(d[:, i].mean()), 2),
                round(float(d[:, i].std() / np.sqrt(len(d))), 2)] for i, k in enumerate(('pfc', 'hip', 'any'))}


def main():
    src = sys.argv[1]
    over = {k: json.loads(v) for k, v in (a.split('=', 1) for a in sys.argv[2:] if '=' in a)}
    seed = over.pop('seed', 0)                      # another draw of shuffles and drilled lines
    only = over.pop('only', None)                   # 'drill': only the check of learning by repetition
    reps = over.pop('reps', 20)
    lead_drill, lead_test = over.pop('lead_drill', 4), over.pop('lead_test', 4)   # quiet ticks before each line
    t0 = time.time()
    if src == 'newborn':
        life = CorpusLife(config(**{'levels_per_s': 4, 'n_brake': 20, 'birth_dialogues': 80, **over}), 0, 'data')
    else:
        life = torch.load(src, weights_only=False)['life']
        if 'g44' in over:                           # probe: the last layer's internal push (set at birth)
            life.brain.pfc.L44.gain = over['g44']
        if 'l41_scale' in over:                     # probe: the learned "what comes next" links scaled
            life.brain.pfc.L41.w.mul_(over['l41_scale'])
        if over.get('usual0'):                      # probe: every PFC copy's / last-layer unit's usual level back to 0
            p = life.brain.pfc
            p.usual.zero_()
            p.usual4.zero_()
        if 'hip_scale' in over:                     # probe: the hippocampus's learned past scaled
            for tb in life.brain.hip.plastic():
                tb.w.mul_(over['hip_scale'])
        life.c.update({k: v for k, v in over.items() if k in life.c})
    rng = np.random.default_rng(7 + seed)
    order = getattr(life, 'order', None)
    order = common_order(life.train) if order is None else order

    def shuffled(ln):
        for _ in range(20):
            s = [ln[i] for i in rng.permutation(len(ln))]
            if s != ln:
                return s
        return None

    def drill_test():
        """20 drilled and 20 control never-heard lines. Per line its gap (own order minus shuffled, voices C1 and D2),
        before and after each drilled line is heard `reps` times (voice C1, learning on, the lines mixed). The gap's
        change for the drilled lines minus for the control lines is what repetition taught."""
        pool = first_lines(life.heldout, range(life.heldout.n), 2 * N + 50 * (seed + 1), lo=4)[2 * N + 50 * seed:]
        items = [(ln, sh) for ln, sh in ((ln, shuffled(ln)) for ln in pool) if sh is not None][:2 * N // 2]

        def gaps():
            g = []
            for ln, sh in items:
                wi = slice(lead_test + 3, lead_test + 3 * len(ln) + 3)
                g.append(np.mean([share(listen(life, line_srcs(ln, v, lead_test), wi)) -
                                  share(listen(life, line_srcs(sh, v, lead_test), wi)) for v in ('C1', 'D2')], 0))
            return np.array(g)
        g0 = gaps()
        drilled = items[0::2]
        for _ in range(reps):
            for i in rng.permutation(len(drilled)):
                listen(life, line_srcs(drilled[i][0], 'C1', lead_drill) + [None] * 4, slice(0, 1), learn=True)
        d = gaps() - g0
        return compare(d[0::2], d[1::2])

    out = {}
    if only == 'drill':
        print('EXPECT', src, json.dumps(dict(over=over, reps=reps, lead=[lead_drill, lead_test], res={'drill': drill_test()},
                                             secs=round(time.time() - t0))), flush=True)
        return
    for nm, lines in (('heard', first_lines(life.train, order, N, lo=4)),
                      ('novel', first_lines(life.heldout, range(life.heldout.n), N, lo=4))):
        R, C = [], []
        for ln in lines:
            sh = shuffled(ln)
            if sh is None:
                continue
            w = slice(7, 4 + 3 * len(ln) + 3)               # from the 2nd syllable to 3 ticks after the last
            for v in ('C1', 'D2'):
                R.append(share(listen(life, line_srcs(ln, v), w)))
                C.append(share(listen(life, line_srcs(sh, v), w)))
        out[nm] = compare(np.array(R), np.array(C))

    pairs = []
    for d in order:
        tr = life.train.turns(int(d))
        for j in range(len(tr) - 1):
            if 2 <= len(tr[j]) <= 10 and len(tr[j + 1]) >= 2:
                pairs.append(([int(x) for x in tr[j]], [int(x) for x in tr[j + 1][:4]]))
                break
        if len(pairs) >= N:
            break
    R, C = [], []
    for i, (pr, rp) in enumerate(pairs):
        for reply in (rp, pairs[(i + 1) % len(pairs)][1]):
            srcs = [None] * 4 + [(x, k, 'C1', 0.0, 0.0) for x in pr for k in range(3)] + [None] * 6 + \
                [(x, k, 'D1', 0.0, 0.0) for x in reply for k in range(3)] + [None] * 3
            e = 4 + 3 * len(pr) + 6
            (R if reply is rp else C).append(share(listen(life, srcs, slice(e, len(srcs)))))
    out['reply'] = compare(np.array(R), np.array(C))

    lines = first_lines(life.heldout, range(life.heldout.n), 2 * N, lo=4)[N:]
    w = [slice(7, 4 + 3 * len(ln) + 3) for ln in lines]

    def all_shares():
        return np.array([share(listen(life, line_srcs(ln, 'C1'), wi)) for ln, wi in zip(lines, w)])
    before = all_shares()
    for i in range(0, len(lines), 2):                     # the once-heard half, heard as in life
        listen(life, line_srcs(lines[i], 'C1') + [None] * 4, slice(0, 1), learn=True)
    soon = all_shares()
    was, life.brain.learning = life.brain.learning, True
    n0 = life.env.n_ex
    while life.env.n_ex - n0 < 20:
        life.tick()
    life.brain.learning = was
    later = all_shares()
    for when, X in (('once_soon', soon), ('once_later', later)):
        g = X - before
        out[when] = compare(g[0::2], g[1::2][:len(g[0::2])])
    out['drill'] = drill_test()
    print('EXPECT', src, json.dumps(dict(over=over, res=out, items={'heard': N, 'reply': len(pairs), 'once': len(lines)},
                                         secs=round(time.time() - t0))), flush=True)


if __name__ == '__main__':
    main()
