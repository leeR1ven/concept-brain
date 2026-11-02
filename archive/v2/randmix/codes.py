# -*- coding: utf-8 -*-
"""How items enter the random layer: template similarity vs code overlap (acoustic fly-hash codes), on a trained RL
agent (out/agents/<tag>_s<seed>.pkl).  For every pair of coded items: Jaccard of the binarised Z templates and the
overlap |c_i & c_j| / k_c of their codes.  Also: for the fault test's 'similar-sounding' substitutes (largest cosine
of the heard sound grid): their cosine.
usage: python3 codes.py tag=<tag> seed=<s>"""
import os, sys, pickle, json
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(1, os.path.join(HERE, '..', 'synth'))   # repo: the unchanged ARCH_v2 core and its bench_copy
import numpy as np                    # noqa: E402
import rcore                          # noqa: E402,F401
from agent import BENCH               # noqa: E402
sys.path.insert(0, BENCH)
import world as world_mod             # noqa: E402
from world import World               # noqa: E402

A = dict(a.split('=', 1) for a in sys.argv[1:])
tag, seed = A['tag'], int(A['seed'])
ag = pickle.load(open(os.path.join(HERE, 'out', 'agents', f'{tag}_s{seed}.pkl'), 'rb'))
m = ag.core.mix
Z = ag.Z
items = sorted(i for i in m.codes if 0 <= i < Z.n)
T = {i: Z.W[:, i] >= 0.5 for i in items}
jt, co = [], []
for a in range(len(items)):
    for b in range(a + 1, len(items)):
        i, j = items[a], items[b]
        jt.append((T[i] & T[j]).sum() / max((T[i] | T[j]).sum(), 1))
        co.append(len(np.intersect1d(m.codes[i], m.codes[j])) / m.k_c)
jt, co = np.array(jt), np.array(co)
bins = [0, 0.2, 0.3, 0.4, 0.5, 0.6, 1.01]
tab = []
for lo, hi in zip(bins[:-1], bins[1:]):
    s = (jt >= lo) & (jt < hi)
    tab.append([lo, hi, int(s.sum()), round(float(co[s].mean()), 3) if s.any() else None])
# the fault test's similar-sounding substitutes, through the agent's own ear (Z) and codes
w = World('B', seed=seed, eye_layout='retino')
v = w.voice
lex = sorted(world_mod.LEXICON_CHARS)
G = {c: v.hear(c, 'other').astype(np.float64).ravel() for c in lex}


sims = []
for c in lex:
    best, bc = -1, None
    for d in lex:
        if d != c:
            s = float(G[c] @ G[d] / (np.linalg.norm(G[c]) * np.linalg.norm(G[d]) + 1e-9))
            if s > best:
                best, bc = s, d
    sims.append((c, bc, best))
out = dict(tag=tag, seed=seed, n_items=len(items), pairs=len(jt), template_jaccard_mean=round(float(jt.mean()), 3),
           code_overlap_mean=round(float(co.mean()), 3), corr=round(float(np.corrcoef(jt, co)[0, 1]), 3),
           by_template_jaccard=tab, similar_cos_mean=round(float(np.mean([s for _, _, s in sims])), 3),
           similar_cos_q=[round(float(q), 3) for q in np.quantile([s for _, _, s in sims], [0.1, 0.5, 0.9])])
print(json.dumps(out, ensure_ascii=False))
json.dump(out, open(os.path.join(HERE, 'out', f'codes_{tag}_s{seed}.json'), 'w'), ensure_ascii=False)
