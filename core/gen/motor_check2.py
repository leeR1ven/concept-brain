"""Frozen checks of the motor area after grow.rebuild_1113 (before any training). A thought is a first-layer copy lit
from inside (PFC.force_int for one tick); everything after it runs as in life: returns prepare force neurons, the
settling decides, the mouth says what the muscles do (exact pattern match), the own voice is heard back.
usage: python -m core.gen.motor_check2 <ckpt> [n=40]"""
import collections
import json
import sys

import numpy as np
import torch


def main(ckpt, n=40, no_mask=0):
    import core.corpus_life  # noqa: F401
    from ..grow import rebuild_1113
    from ..net import DEV
    from ..speak import concept_codes, patterns72
    from ..world.voice import ADULTS
    life = torch.load(ckpt, weights_only=False)['life']
    if not getattr(life, 'rebuilt_1113', False):
        rebuild_1113(life)
    b, p, mot = life.brain, life.brain.pfc, life.brain.motor
    b.learning = False
    if no_mask:                                           # intentions compete in the thought budget like any thought
        p.motor_mask = None
    segs = {(k, l): (a, bb) for k, l, a, bb in b.chains.segs}
    f0, f1 = segs[('force', 0)]
    ca, cb = segs[('concept', 0)]
    sids = list(mot.settle_sids)
    P = patterns72(life.body)
    exact = {P[s].tobytes(): s for s in range(len(P))}
    rng = np.random.default_rng(4)
    test = [int(x) for x in rng.choice(sids, n, replace=False)]
    code, _, _ = concept_codes(life, test)
    snap = b.snapshot()
    fl = []                                               # background: force copies lit inside on quiet ticks
    for _ in range(12):
        b.step(torch.as_tensor(life.with_sight(life.coch.a0([])), device=DEV), 0.0)
        fl.append((int(p.x_int[f0:f1].sum()), int(p.x_int.sum()), int((b.ret_force > 0).sum())))
    b.restore(snap)
    print('BACKGROUND quiet ticks: force copies thought / all thoughts / force neurons prepared by returns', fl[4:], flush=True)

    def force_copies(s, share=1.0):
        ch = np.nonzero(P[s])[0]
        ch = ch[:max(1, int(round(share * len(ch))))] if share < 1 else ch
        m = torch.zeros(p.n, dtype=torch.bool, device=DEV)
        fn = torch.nonzero(torch.isin(mot.muscle_of, torch.as_tensor(ch, device=DEV))).squeeze(1)
        m[f0 + fn] = True
        return m

    def run(thought=None, heard=None, ticks=6, brake=False):
        """the thought lit for one tick (tick 0); returns what the mouth said on each tick (exact syllable, 'garble',
        None) - the own voice is heard back"""
        b.restore(snap)
        out, own = [], None
        for t in range(ticks):
            p.force_int = thought if (thought is not None and t == 0) else None
            src = ([] if heard is None or t != 0 else [(heard, -1, ADULTS[0], 0.0, 0.0)]) + \
                  ([(sids[0], -1, ADULTS[1], 0.0, 0.0)] if brake else []) + \
                  ([] if own is None else [(own, -1, 'own', 0.0, 0.0)])
            mus = b.step(torch.as_tensor(life.with_sight(life.coch.a0(src)), device=DEV), 0.0).cpu().numpy()
            if not mus.any():
                out.append(None)
                own = None
            else:
                y = exact.get(mus.astype(bool).tobytes(), 'garble')
                out.append(y)
                own = None if y == 'garble' else y
        p.force_int = None
        b.restore(snap)
        return out
    R = {}

    def tally(name, cases):
        k = collections.Counter()
        for out, want in cases:
            said = [y for y in out if y is not None]
            if not said:
                k['nothing'] += 1
            elif 'garble' in said:
                k['garble'] += 1
            elif said[0] == want:
                k['right' + (' (then repeated)' if len(said) > 1 else '')] += 1
            else:
                k['other'] += 1
        R[name] = {kk: round(v / len(cases), 3) for kk, v in sorted(k.items())}
        print('CHECK', name, json.dumps(R[name], ensure_ascii=False), flush=True)
    for sh in (1.0, 0.5, 0.15):
        tally(f'1 a syllable intended (its force copies thought, {sh:.0%} of its channels)',
              [(run(force_copies(s, sh)), s) for s in test])
    tally('2 two syllables intended together (union)',
          [(run(force_copies(s) | force_copies(t)), s) for s, t in zip(test[:n // 2], test[n // 2:])])
    for sh in (1.0, 0.5, 0.25):
        def cthought(i, sh=sh):
            m = torch.zeros(p.n, dtype=torch.bool, device=DEV)
            idx = torch.nonzero(code[i]).squeeze(1)
            idx = idx[:max(1, int(round(sh * idx.numel())))]
            m[ca + idx] = True
            return m
        tally(f'3 a syllable thought as a concept ({sh:.0%} of its code) -> speech link', [(run(cthought(i)), s) for i, s in enumerate(test)])
    tally('4 intended while an adult speaks (brake)', [(run(force_copies(s), brake=True), s) for s in test])
    tally('5 heard (an adult says it), nothing thought: must stay silent', [(run(None, heard=s), -1) for s in test])
    print('MOTORCHECK', json.dumps(R, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    kw = dict(a.split('=', 1) for a in sys.argv[2:])
    main(sys.argv[1], n=int(kw.get('n', 40)), no_mask=int(kw.get('no_mask', 0)))
