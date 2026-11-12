"""Frozen checks of the concept-space syllable pools (core/speak3.py) on a save, inside the body.
1. Injection in concept space (Fable a40 B6): the push of one syllable's code at strengths around the bar; mixtures of
   two; a flat mix of ten; the brake (an adult voice heard). The muscles go through the mouth: said right / another
   syllable / a garble / nothing.
2. Cut-line test: the first part of a never-heard line, then silence with the mouth on (its own voice heard back);
   the first syllable said vs the line's next one; own order vs the same syllables shuffled; the babbler.
usage: python -m core.gen.pool_check <ckpt> [n=80] [share=0.1]"""
import collections
import json
import sys

import numpy as np
import torch


def main(ckpt, n=80, share=0.1, quiet=6, min_ratio=1.2):
    import core.corpus_life  # noqa: F401
    from ..exam import first_lines
    from ..net import DEV
    from ..speak3 import install_concept_pools
    from ..world.voice import ADULTS
    life = torch.load(ckpt, weights_only=False)['life']
    b, mot = life.brain, life.brain.motor
    b.learning = False
    cp = install_concept_pools(life, share=share, min_ratio=min_ratio)
    sids = mot.cpool_sids
    code = (cp.W > 0)
    snap = b.snapshot()
    silent = torch.as_tensor(life.with_sight(life.coch.a0([])), device=DEV)
    loud = torch.as_tensor(life.with_sight(life.coch.a0([(int(sids[0]), -1, ADULTS[0], 0.0, 0.0)])), device=DEV)
    zf, zs = torch.zeros_like(b.ret_force), torch.zeros_like(b.ret_stop)
    n1 = b.pfc.e1.shape[0]

    def say(e1, a0):
        b.restore(snap)
        mot.e1 = e1
        mus = mot.step(a0, zf, zs).cpu().numpy()
        y = life.mouth.say(mus)
        return None if y is None else ('garble' if y == 'garble' else int(y))

    def inj(parts):
        e1 = torch.zeros(n1, device=DEV)
        for i, s in parts:
            e1[cp.ca:cp.cb] += s * cp.bar * code[i].float()
        return e1
    rng = np.random.default_rng(3)
    rows = rng.choice(len(sids), 40, replace=False)
    out = {}

    def tally(name, cases):
        k = collections.Counter()
        for e1, a0, want in cases:
            y = say(e1, a0)
            k['nothing' if y is None else 'garble' if y == 'garble' else 'right' if y == want else 'other'] += 1
        out[name] = {kk: round(v / len(cases), 3) for kk, v in sorted(k.items())}
        print('INJECT', name, json.dumps(out[name]), flush=True)
    for s in (0.5, 0.9, 1.2, 2.0, 4.0):
        tally(f'one syllable at {s} x bar', [(inj([(i, s)]), silent, sids[i]) for i in rows])
    for a in (0.7, 0.6, 0.5):
        tally(f'two syllables {a}/{1 - a:.1f} (x 2 bar)', [(inj([(i, 2 * a), (j, 2 * (1 - a))]), silent, sids[i])
                                                       for i, j in zip(rows[:20], rows[20:])])
    tally('flat mix of ten (each 2 x bar)', [(inj([(int(j), 2.0) for j in rng.choice(len(sids), 10, replace=False)]),
                                              silent, -1) for _ in range(20)])
    tally('one syllable at 4 x bar, brake on (an adult voice)', [(inj([(i, 4.0)]), loud, sids[i]) for i in rows])
    b.restore(snap)
    # 2. cut-line test inside the body
    cnt = np.bincount(life.train.syl[:200000].astype(np.int64))
    common = int(np.argmax(cnt))
    lines = first_lines(life.heldout, range(life.heldout.n), n, lo=4)
    cases = []
    for ln in lines:
        k = int(rng.integers(2, len(ln)))
        pre = list(ln[:k])
        sh = pre
        for _ in range(10):
            sh = [pre[i] for i in rng.permutation(len(pre))]
            if sh != pre:
                break
        cases.append((pre, sh, int(ln[k])))
    res = {'cases': len(cases), 'babbler says the next': round(float(np.mean([t == common for _, _, t in cases])), 3)}
    for nm, idx in (('own order', 0), ('shuffled', 1)):
        said, wins = [], 0
        for case in cases:
            b.restore(snap)
            own, ys = None, []
            srcs = [None] * 4 + [x for s_ in case[idx] for x in life.syl(s_, 'C1')] + [None] * quiet
            for t, x in enumerate(srcs):
                a0 = life.coch.a0(([] if x is None else [x]) + ([] if own is None else [(own, -1, 'own', 0.0, 0.0)]))
                mus = b.step(torch.as_tensor(life.with_sight(a0), device=DEV), 0.0).cpu().numpy()
                y = life.mouth.say(mus)
                own = y if (y is not None and y != 'garble') else None
                if t >= len(srcs) - quiet:
                    ys.append(y)
                    wins += cp.winner >= 0
            said.append([y for y in ys if y is not None])
        res[nm] = dict(said_anything=round(float(np.mean([bool(s) for s in said])), 3),
                       garbles=round(float(np.mean([sum(y == 'garble' for y in s) for s in said])), 3),
                       first_is_next=round(float(np.mean([bool(s) and s[0] == cs[2] for s, cs in zip(said, cases)])), 3),
                       next_anywhere=round(float(np.mean([cs[2] in s for s, cs in zip(said, cases)])), 3),
                       pool_wins_per_case=round(wins / len(cases), 2))
        print('CUT', nm, json.dumps(res[nm]), flush=True)
    b.restore(snap)
    print('POOLCHECK', json.dumps(dict(inject=out, cut=res)), flush=True)


if __name__ == '__main__':
    kw = dict(a.split('=', 1) for a in sys.argv[2:])
    main(sys.argv[1], n=int(kw.get('n', 80)), share=float(kw.get('share', 0.1)), min_ratio=float(kw.get('min_ratio', 1.2)))
