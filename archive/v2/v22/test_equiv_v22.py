# -*- coding: utf-8 -*-
"""AgentV22 with every new key at its default == AgentV21 (this folder's copy, v21final semantics), bit for bit, on the
ARCH_v2.1 final config (mouth output, replies, GEN weights, hippocampus).  A second check runs AgentV22 with rm=1
(c_flat) to make sure the random layer runs inside the v2.1 core (no equality expected).
usage: python3 test_equiv_v22.py [nB=40] [rm=1]"""
import os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import _paths  # noqa: E402,F401  (repo: v2.2's own agent_v21 / rules / randlayer before ../v21 and ../synth)
from agent_v21 import AgentV21
from agent_v22 import AgentV22
from agent import BENCH
sys.path.insert(0, BENCH)
from world import World
from run_j import TUNE_BEST, THINK_FINAL
A = dict(a.split('=', 1) for a in sys.argv[1:])
nB = int(A.pop('nB', 40)); do_rm = int(A.pop('rm', 0))
FINAL = dict(rebind=1, ct=True, init_min_tau=10, hip_rho=2.0, ctx=False)
V21 = dict(ral=1, ral_other=1, say_rec=1, say_loc=1, scene_held=1, val_veto=0.2, dip=1, act_T=60, soc_feats=1, rec_fix=1)
def run(cls, extra):
    kw = dict(TUNE_BEST); kw.update(THINK_FINAL); kw.update(FINAL); kw.update(V21); kw.update(extra)
    wA = World('A', seed=0, tutor='recast', eye_layout='retino')
    ag = cls(wA, seed=0, **kw)
    wA.run(ag, 20); ag.babble(wA.voice, 35960)
    w = World('B', seed=0, tutor='recast', prev=wA, eye_layout='retino'); w.run(ag, nB)
    w2 = World('C', seed=0, tutor='recast', prev=w, eye_layout='retino'); w2.run(ag, nB)
    reps = [r.get('reply') for r in w.log + w2.log]
    return reps, ag.core.gen.W[:ag.core.mix.n].copy(), ag.core.mix.n, ag.core.hip.n, ag
t0 = time.time()
if not do_rm:
    r1, W1, m1, h1, a1 = run(AgentV21, {})
    r2, W2, m2, h2, a2 = run(AgentV22, {})
    same = r1 == r2 and m1 == m2 and h1 == h2 and W1.shape == W2.shape and np.array_equal(W1, W2)
    print(f'replies {len(r1)} identical {r1 == r2} | mix {m1} {m2} | hip {h1} {h2} | GEN identical {W1.shape == W2.shape and np.array_equal(W1, W2)}')
    print('EQUIVALENT' if same else 'DIFFERENT', f'{time.time() - t0:.0f}s', a2.core.ral_stats, a2.core.val_stats)
else:
    rm = dict(train=1, prune=1, age=20000, regrow=1, alloc='flat')
    r2, W2, m2, h2, a2 = run(AgentV22, dict(rm=rm))
    acc = np.mean([r.get('reply') in (r.get('acceptable') or []) for r in []]) if False else None
    print('rm run ok', f'{time.time() - t0:.0f}s', 'units', m2, 'GEN', W2.shape, a2.core.mix.snapshot('t'), a2.core.ral_stats)
    print('replies (last 20):', r2[-20:])
