"""The cut-line test (Fable a39 C4), frozen: the first part of a never-heard line, then silence with the mouth on (its
own voice heard the next tick); what does the child say? Against the same syllables shuffled, and the babbler.
With the tendency speech link (core/speak2.py) and the settling pools installed in this process only.
usage: python -m core.gen.cut_check <ckpt> [n=60] [gains=1,2,4]"""
import json
import sys

import numpy as np
import torch


def main(ckpt, n=60, gains=(1.0, 2.0, 4.0)):
    import core.corpus_life  # noqa: F401
    from ..exam import first_lines
    from ..net import DEV
    from ..speak2 import grade_scale, install_settle, install_tendency_speech
    life = torch.load(ckpt, weights_only=False)['life']
    b = life.brain
    lk = install_tendency_speech(life, cal_dialogues=20)
    grade_scale(lk, b.motor.m)
    install_settle(life)
    g0 = lk.gain
    rng = np.random.default_rng(5)
    lines = first_lines(life.heldout, range(life.heldout.n), n, lo=4)
    cnt = np.bincount(life.train.syl[:200000].astype(np.int64))
    common = int(np.argmax(cnt))
    cases = []
    for ln in lines:
        k = int(rng.integers(2, len(ln)))
        pre = ln[:k]
        sh = list(pre)
        for _ in range(10):
            sh = [pre[i] for i in rng.permutation(len(pre))]
            if sh != pre:
                break
        cases.append((pre, sh, ln[k]))

    def run(prefix, quiet=6):
        was, b.learning = b.learning, False
        snap = b.snapshot()
        own, said = None, []
        srcs = [None] * 4 + [x for s_ in prefix for x in life.syl(s_, 'C1')] + [None] * quiet
        for x in srcs:
            a0 = life.coch.a0(([] if x is None else [x]) + ([] if own is None else [(own, -1, 'own', 0.0, 0.0)]))
            mus = b.step(torch.as_tensor(life.with_sight(a0), device=DEV), 0.0).cpu().numpy()
            y = life.mouth.say(mus)
            said.append(y)
            own = y if (y is not None and y != 'garble') else None
        b.restore(snap)
        b.learning = was
        return [y for y in said[-quiet:] if y is not None]

    out = {'cases': len(cases), 'babbler says the next': round(float(np.mean([t == common for _, _, t in cases])), 3)}
    for g in gains:
        lk.gain = g0 * g
        res = {}
        for nm, idx in (('own order', 0), ('shuffled', 1)):
            S = [run(c[idx]) for c in cases]
            first = [s[0] if s else None for s in S]
            res[nm] = dict(said_anything=round(float(np.mean([bool(s) for s in S])), 3),
                           first_is_next=round(float(np.mean([f == c[2] for f, c in zip(first, cases)])), 3),
                           next_anywhere=round(float(np.mean([c[2] in s for s, c in zip(S, cases)])), 3),
                           garbles=round(float(np.mean([sum(y == 'garble' for y in s) for s in S])), 3),
                           examples=[x for s in S[:12] for x in s[:1] if x != 'garble'][:8])
        out[f'gain x{g}'] = res
        print('CUT', json.dumps(out), flush=True)


if __name__ == '__main__':
    kw = dict(a.split('=', 1) for a in sys.argv[2:])
    main(sys.argv[1], n=int(kw.get('n', 60)),
         gains=tuple(float(x) for x in kw.get('gains', '1,2,4').split(',')))
