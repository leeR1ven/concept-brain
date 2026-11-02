# -*- coding: utf-8 -*-
"""x2 read-only analysis: what the awake-recall pool teaches.  Decodes the hippocampal cells that awake recall can
replay (live, valid NEXT link, as in core.awake_recall) into text (the agent's own syllable categories, named by
hearing every corpus char in the same world's voice), and classifies each cue -> NEXT pair:
  next_partnerQ  the NEXT line is a partner prompt (a line the partner opens an exchange with)
  next_other     anything else (an answer, a recast, a statement)
split by era (born before clock0 = tutored training, after = untutored use).
usage: python3 an_pool.py <pickle> <seed> <clock0> [more pickles with the same seed/clock0 ...]"""
import os, sys, gzip, pickle, collections
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from agent import BENCH                                    # noqa: E402
sys.path.insert(0, BENCH)
import numpy as np                                         # noqa: E402
from world import World                                    # noqa: E402
from physics import Stream                                 # noqa: E402

QW = set('什么吗哪几谁')


def cat_map(agent, voice, chars):
    m = {}
    for ch in chars:
        st = Stream(voice.F, voice.T, voice.T // 3, 3, voice.levels)
        e = agent._ear_copy(); none = (None, None, False); g = voice.hear(ch, 'other'); out = None
        for k in range(3):
            out = e.step(st.tick(((ch, g, k == 0), none, none)))
        j, s = agent.Z.best(out[agent.F0])
        if s >= agent.theta_z:
            m.setdefault(int(j), ch)
    return m


def main():
    seed, clock0 = int(sys.argv[2]), int(sys.argv[3])
    for path in [sys.argv[1]] + sys.argv[4:]:
        with gzip.open(path, 'rb') as f:
            obj = pickle.load(f)
        ag, prompts = obj['agent'], obj['prompts']
        w = World("C", seed=seed, tutor="recast", eye_layout='retino')
        from world import LEXICON_CHARS
        chars = sorted(set(''.join(prompts)) | set(LEXICON_CHARS))
        cm = cat_map(ag, w.voice, chars)
        txt = lambda items: ''.join(cm.get(int(i), '?') for i in items if 0 <= int(i) < ag.core.c.V)
        core, h = ag.core, ag.core.hip
        n = h.n
        nx = core.next_of[:n]
        ok = nx >= 0
        ok[ok] = h.born[nx[ok]] == core.next_born[:n][ok]
        ok &= h.sig[:n] > 0.05
        old = h.born[:n] < clock0
        res = {}
        ex = {}
        for era, msk in (('old', ok & old), ('new', ok & ~old)):
            cnt = collections.Counter()
            pairs = collections.Counter()
            for e in np.flatnonzero(msk):
                cue = txt(h.content[e]['items'])
                nxt = txt(h.content[int(nx[e])]['items'])
                cq = 'cueQ' if (cue in prompts or set(cue) & QW) else 'cueS'
                kq = 'next_partnerQ' if (nxt in prompts or set(nxt) & QW) else 'next_other'
                cnt[cq + '>' + kq] += 1
                pairs[(cue, nxt)] += 1
            res[era] = dict(n=int(msk.sum()), **cnt)
            ex[era] = pairs.most_common(8)
        print(f"== {os.path.basename(path)} done_n {obj.get('done_n')} hip n {n} clock {h.clock} cats named {len(cm)}")
        for era in ('old', 'new'):
            print(f"  pool {era}: {res[era]}")
            print(f"     top pairs: {ex[era]}")


if __name__ == '__main__':
    main()
