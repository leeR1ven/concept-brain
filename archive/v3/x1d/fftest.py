# standalone: similarity transfer of the birth feed-forward stack L1 -> L2 -> L3 (exact top-k, no recurrence)
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'x1c'))   # repo: core3.py, world.py, run.py from ../x1c
from world import Physics, World, render, NSYL, C1
from core3 import Brain
from run import birth_rates, parse
ph = Physics(); w = World(ph, 0); rates = birth_rates(ph, w, 2000)
def topk(x, k):
    o = np.zeros(len(x), bool); o[np.argsort(-x)[:k]] = True; return o
def jac(a, b): return (a & b).sum() / max(1, (a | b).sum())
for spec in sys.argv[1:]:
    kv = {k: parse(v) for k, v in (a.split('=') for a in spec.split(','))}
    b = Brain(ph, 0, rates=rates, **kv); c = b.c
    rng = np.random.default_rng(1)
    L = {1: [], 2: [], 3: []}
    for s in range(NSYL):
        for rep in range(3):
            snd = render(ph, [('s', s, C1, 1.0)], rng)
            a0 = np.concatenate([snd, ~snd]); l1 = np.zeros(b.n1, bool); l1[:b.na1] = a0[b.send]
            x2 = b.W12[l1].sum(0) + 1e-3 * rng.random(c['n2']); l2 = topk(x2, c['k2'])
            x3 = b.W23[l2].sum(0) + 1e-3 * rng.random(c['n3']); l3 = topk(x3, c['k3'])
            for k, v in ((1, l1), (2, l2), (3, l3)): L[k].append((s, v))
    out = []
    for k in (1, 2, 3):
        same = np.mean([jac(a, bb) for i, (s, a) in enumerate(L[k]) for (t, bb) in L[k][i + 1:] if s == t])
        cross = np.mean([jac(a, bb) for i, (s, a) in enumerate(L[k]) for (t, bb) in L[k][i + 1:] if s != t])
        out.append(f'L{k} same {same:.2f} cross {cross:.2f}')
    print(spec, '|', ' | '.join(out), '| L1 active %.1f' % np.mean([v.sum() for _, v in L[1]]), flush=True)
