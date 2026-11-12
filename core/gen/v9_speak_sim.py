"""Proof of principle on the long-trained v9 (Fable a38 2b / a39 C4): its PFC's tendency drives the same speech
machinery as core/speak2.py (the tendency link wired from v9's present concept codes to the 72 speech channels,
graded force copies, the settling pools), outside its body (v9's mouth is the older 24-channel one). Cut-line test:
the first part of a never-heard line, then silence; on the silent ticks, which syllable wins (or none), against the
next syllable; own order vs the same syllables shuffled; the babbler. Frozen.
usage (in a checkout of 49e444b with core/speak.py, core/speak2.py): python -m core.gen.v9_speak_sim <ckpt> [n=80]"""
import json
import sys

import numpy as np
import torch


def main(ckpt, n=80, top_share=0.2, quiet=4, mode='muscle'):
    import core.corpus_life  # noqa: F401
    from ..exam import first_lines
    from ..gen0 import nnls
    from ..net import DEV
    from ..speak import patterns72
    from ..speak2 import SettlePools, TendencyLink
    from ..world.voice import ADULTS
    life = torch.load(ckpt, weights_only=False)['life']
    b, p, c = life.brain, life.brain.pfc, life.c
    segs = {k: (a, bb) for k, l, a, bb in b.chains.segs}
    ca, cb = segs['concept']
    lag = 1 if c.get('concept_sync') else 2
    cnt = np.bincount(life.train.syl[:200000].astype(np.int64), minlength=len(life.body.pat))
    sids = [int(x) for x in np.argsort(-cnt)[:600] if cnt[x] > 0]
    FC = torch.zeros((len(sids), cb - ca), device=DEV)        # present codes: heard alone, >= half of the voices
    for i, x in enumerate(sids):
        for v in ADULTS:
            rr = []
            life.play([None] * 4 + life.syl(x, v) + [None] * 2, rec=lambda t: rr.append(p.L1ext.on.clone()))
            FC[i] += torch.stack(rr[4 + lag:4 + lag + 2])[:, ca:cb].any(0).float() / len(ADULTS)
    code = FC >= 0.5
    share = code.float().mean(0)
    code_f = code & (share <= c.get('speak_unit_max', 0.05))[None, :]
    m = 8
    muscle_of = torch.arange(72 * m, device=DEV) // m
    need = torch.as_tensor(patterns72(life.body)[sids], dtype=torch.float32, device=DEV)
    W = nnls(code_f.float(), need)
    lk = TendencyLink(ca, cb, 0, 72 * m, 72 * m, muscle_of, W[:, muscle_of])
    lk.grade = 1.0 / (0.5 + 1.5 * (torch.arange(72 * m, device=DEV) % m).float() / (m - 1))
    pools = SettlePools(need.bool()[:, muscle_of])
    theta = c['theta1']
    # the gain: theta at the (1 - top_share) quantile of the drive's top value, on never-heard dialogue
    from ..world.corpus import script_dialogue
    tops = []
    for di in range(min(20, life.heldout.n)):
        src = script_dialogue(life.heldout, di, np.random.default_rng([life.seed, 9311, di]),
                              tuple(c.get('gap_turn', (4, 10))), 8, 1, 1)
        life.play([None if x is None else x[:5] for x in src],
                  rec=lambda t: tops.append(float((lk.drive(p.e1) / lk.grade).max())))
    g0 = theta / max(float(np.quantile(np.array(tops), 1.0 - top_share)), 1e-9)
    size = code.float().sum(1).clamp_min(1.0)

    def support(e1):                                      # each syllable pool reads the PFC's expectation through
        return (code.float() @ e1[ca:cb].clamp_min(0.0)) / size   # its own concept code (concept-space pools)
    if mode == 'concept':
        S1 = []
        for di in range(min(20, life.heldout.n)):
            src = script_dialogue(life.heldout, di, np.random.default_rng([life.seed, 9311, di]),
                                  tuple(c.get('gap_turn', (4, 10))), 8, 1, 1)
            life.play([None if x is None else x[:5] for x in src],
                      rec=lambda t: S1.append(torch.topk(support(p.e1), 2).values.cpu().numpy()))
        S1 = np.array(S1)
        print('SUPPORT top / second quantiles', np.round(np.quantile(S1[:, 0], [0.5, 0.8, 0.95]), 4).tolist(),
              np.round(np.quantile(S1[:, 0] / np.maximum(S1[:, 1], 1e-9), [0.5, 0.8, 0.95]), 3).tolist(), flush=True)
    rng = np.random.default_rng(5)
    lines = first_lines(life.heldout, range(life.heldout.n), n, lo=4)
    common = int(np.argmax(cnt))
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
    out = {'cases': len(cases), 'babbler says the next': round(float(np.mean([t == common for _, _, t in cases])), 3)}
    levels = (1.0, 2.0, 4.0, 8.0) if mode == 'muscle' else (0.5, 0.8, 0.9, 0.95)
    for gm in levels:
        lk.gain = g0 * gm
        if mode == 'concept':                             # speak only when the top pool is in the most confident
            thr_ = float(np.quantile(S1[:, 0], gm))       # (1 - gm) of ticks and clearly ahead of the second
            ratio_ = float(np.quantile(S1[:, 0] / np.maximum(S1[:, 1], 1e-9), 0.5))
        res = {}
        for nm, idx in (('own order', 0), ('shuffled', 1)):
            said = []
            for case in cases:
                W_ = []
                srcs = [None] * 4 + [x for s_ in case[idx] for x in life.syl(s_, 'C1')] + [None] * quiet

                def r_(t):
                    # the push at a tick is made before that tick's copies arrive, from the state a tick earlier:
                    # the expectation of the syllable that would have come right after the cut is the push at
                    # (first silent tick) + lag; the later silent ticks already expect what follows silence
                    if mode == 'concept' and t != len(srcs) - quiet + lag:
                        return
                    if t >= len(srcs) - quiet:                # the silent ticks after the cut
                        if mode == 'concept':
                            sp = support(p.e1)
                            v2 = torch.topk(sp, 2)
                            ok = float(v2.values[0]) >= thr_ and float(v2.values[0]) >= ratio_ * float(v2.values[1])
                            W_.append(sids[int(v2.indices[0])] if ok else None)
                            return
                        lit = lk.drive(p.e1) > theta
                        w = pools.settle(lit)
                        W_.append(None if w < 0 else sids[w])
                life.play(srcs, rec=r_)
                said.append([x for x in W_ if x is not None])
            res[nm] = dict(said_anything=round(float(np.mean([bool(s) for s in said])), 3),
                           first_is_next=round(float(np.mean([bool(s) and s[0] == cs[2] for s, cs in zip(said, cases)])), 3),
                           next_anywhere=round(float(np.mean([cs[2] in s for s, cs in zip(said, cases)])), 3))
        out[(f'gain x{gm}' if mode == 'muscle' else f'speak above the {gm:.0%} quantile')] = res
        print('V9SPEAK', json.dumps(out), flush=True)


if __name__ == '__main__':
    kw = dict(a.split('=', 1) for a in sys.argv[2:])
    main(sys.argv[1], n=int(kw.get('n', 80)), mode=kw.get('mode', 'muscle'))
