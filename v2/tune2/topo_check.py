# -*- coding: utf-8 -*-
"""Is the input continuous on the ear's sheet?  Positions of the syllable categories (T2Agent: mean layer-0 coordinate
of the first exemplar) vs (a) the next syllable inside a heard line, (b) the first syllable of the next line given any
syllable of the line before (what the readout must predict), (c) random pairs.  Also the radius that reaches a given
share of the items (calibration of loc_r).
usage: python3 topo_check.py seed=0 nA=300 nB=300 out=out/topo_s0.json"""
import os, sys, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from agent_t2 import T2Agent                       # noqa: E402
from agent import BENCH                            # noqa: E402
sys.path.insert(0, BENCH)
from world import World                            # noqa: E402
from run_j import TUNE_BEST, THINK_FINAL           # noqa: E402
from run_t2 import ARCH_V2                         # noqa: E402

A = dict(a.split("=", 1) for a in sys.argv[1:])
seed, nA, nB = int(A.get('seed', 0)), int(A.get('nA', 300)), int(A.get('nB', 300))
kw = dict(TUNE_BEST); kw.update(THINK_FINAL); kw.update(ARCH_V2)
wA = World('A', seed=seed, tutor='recast', eye_layout='retino')
ag = T2Agent(wA, seed=seed, **kw)
lines = []
orig = ag.core._ct_input
def spy(rec, surprise=None):
    lines.append([i for i in rec['items'] if 0 <= i < ag.core.c.V])
    return orig(rec, surprise=surprise)
ag.core._ct_input = spy
wA.run(ag, nA); ag.babble(wA.voice, 35960)
wB = World('BC', seed=seed, tutor='recast', prev=wA, eye_layout='retino')
wB.run(ag, nB)
pos = ag.core.item_pos[:ag.core.c.V]
known = np.flatnonzero(~np.isnan(pos[:, 0]))
P = pos[known]
D = np.sqrt(((P[:, None] - P[None]) ** 2).sum(-1))
iu = np.triu_indices(len(known), 1)
pd = D[iu]
cnt = np.zeros(ag.core.c.V);
for L in lines:
    for i in L:
        cnt[i] += 1
def dist(a, b):
    return float(np.sqrt(((pos[a] - pos[b]) ** 2).sum()))
within = [dist(a, b) for L in lines for a, b in zip(L[:-1], L[1:]) if a != b]
nxt = []
for L0, L1 in zip(lines[:-1], lines[1:]):
    if L0 and L1 and L1[0] not in L0:
        nxt.append(min(dist(a, L1[0]) for a in set(L0)))
rng = np.random.default_rng(0)
p = cnt[known] / cnt[known].sum()
ra = rng.choice(known, size=20000, p=p); rb = rng.choice(known, size=20000, p=p)
rnd = [dist(a, b) for a, b in zip(ra, rb) if a != b]
# random 'min over a line' baseline: the same line lengths, random items
rnd_min = []
for L0, L1 in zip(lines[:-1], lines[1:]):
    if L0 and L1 and L1[0] not in L0:
        S = rng.choice(known, size=len(set(L0)), p=p)
        rnd_min.append(min(dist(a, L1[0]) for a in S))
fr = {}
for r in (0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.2):
    fr[r] = round(float((D <= r).sum(1).mean() / len(known)), 4)        # share of items within r (incl. itself)
q = lambda x: [round(float(v), 4) for v in np.percentile(x, [10, 25, 50, 75, 90])]
out = dict(seed=seed, n_items=int(len(known)), n_lines=len(lines), pairwise=q(pd), within_line_next=q(within),
           next_line_first_min=q(nxt), random_pairs=q(rnd), random_min_same_len=q(rnd_min),
           share_within_r=fr, frac_next_in_r={r: round(float(np.mean(np.array(within) <= r)), 4) for r in fr},
           frac_answer_in_r={r: round(float(np.mean(np.array(nxt) <= r)), 4) for r in fr},
           frac_random_in_r={r: round(float(np.mean(np.array(rnd) <= r)), 4) for r in fr},
           frac_randmin_in_r={r: round(float(np.mean(np.array(rnd_min) <= r)), 4) for r in fr})
print(json.dumps(out, indent=1))
json.dump(out, open(A.get('out', os.path.join(HERE, 'out', f'topo_s{seed}.json')), 'w'), indent=1)
