# -*- coding: utf-8 -*-
"""Bench identity: AgentH with the hearing keys ON == AgentV21 (v2.1 final config), bit for bit, on the clean bench
(mouth output = replies, GEN weights, MIX and hippocampus counts, syllable categories), plus the hearing counters
(relocks / back-tracks / gated ticks must be 0; max sound run < h_stuck).
usage: python3 test_equiv_h.py [nA=60 nB=60] [h=h_gate:1,h_back:1,...] [noise=0]"""
import os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'v21')); sys.path.insert(1, os.path.join(HERE, '..', 'synth')); sys.path.insert(0, HERE)
from agent_v21 import AgentV21                                 # noqa: E402
from agent_h import AgentH                                     # noqa: E402
from agent import BENCH                                        # noqa: E402
sys.path.insert(0, BENCH)
from world import World                                        # noqa: E402
from run_v21 import TUNE_BEST, THINK_FINAL, ARCH_V2            # noqa: E402

V21FINAL = dict(ral=1, ral_other=1, say_rec=1, say_loc=1, scene_held=1, val_veto=0.2, dip=1, act_T=60, soc_feats=1,
                rec_fix=1)
A = dict(a.split('=', 1) for a in sys.argv[1:])
nA, nB = int(A.pop('nA', 60)), int(A.pop('nB', 60))
noise = float(A.pop('noise', 0))
hk = {k: (float(v) if '.' in v else int(v)) for k, v in (kv.split(':') for kv in
      A.pop('h', 'h_gate:1,h_back:1,h_relock:1,h_vbound:1,h_nocreate:1').split(',') if kv)}


def run(cls, extra):
    kw = dict(TUNE_BEST); kw.update(THINK_FINAL); kw.update(ARCH_V2); kw.update(V21FINAL); kw.update(extra)
    wA = World('A', seed=0, tutor='recast', eye_layout='retino', noise=noise)
    ag = cls(wA, seed=0, **kw)
    wA.run(ag, nA); ag.babble(wA.voice, 35960)
    w = World('B', seed=0, tutor='recast', prev=wA, eye_layout='retino', noise=noise); w.run(ag, nB)
    w2 = World('C', seed=0, tutor='recast', prev=w, eye_layout='retino', noise=noise); w2.run(ag, nB)
    reps = [r.get('reply') for r in w.log + w2.log]
    return reps, ag.core.gen.W[:ag.core.mix.n].copy(), ag.core.mix.n, ag.core.hip.n, ag


t0 = time.time()
r1, W1, m1, h1, a1 = run(AgentV21, {})
r2, W2, m2, h2, a2 = run(AgentH, hk)
same = r1 == r2 and m1 == m2 and h1 == h2 and W1.shape == W2.shape and np.array_equal(W1, W2) and a1.Z.n == a2.Z.n \
    and np.array_equal(a1.Z.W[:, :a1.Z.n], a2.Z.W[:, :a2.Z.n])
print(f'noise {noise} keys {hk} | replies {len(r1)} identical {r1 == r2} | mix {m1} {m2} | hip {h1} {h2} | '
      f'cats {a1.Z.n} {a2.Z.n} | GEN identical {W1.shape == W2.shape and np.array_equal(W1, W2)}')
print('hearing counters', a2.hstats, 'gate', None if a2.gate is None else
      dict(mu=a2.gate.mu, nu=a2.gate.nu, **a2.gate.stats))
print('lines', a1.stats['lines'], a2.stats['lines'], 'items', a1.stats['items'], a2.stats['items'])
print('EQUIVALENT' if same else 'DIFFERENT', f'{time.time() - t0:.0f}s')
