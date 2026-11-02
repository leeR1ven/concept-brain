# -*- coding: utf-8 -*-
"""x2 equivalence: AgentX2 with every new key off (and with stab_track=1, which only keeps use traces) must equal
ARCH_v2.2's AgentV22 FROM THE v22 FOLDER (original code) bit for bit on the v2.2 final config (c_flat + gated veto):
replies, GEN weights, gates, gateR, hippocampus, the random layer's weights / lines.  Each side runs in its own process.
usage: python3 test_equiv_x2.py [nB=40] [extra keys k=v ...]          (runs both sides and compares)
       python3 test_equiv_x2.py side=v22|x2 out=f.npz [nB=40] [k=v ...] (one side)"""
import os, sys, subprocess, time, hashlib
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
V22 = os.path.join(HERE, '..', 'v22')
A = dict(a.split('=', 1) for a in sys.argv[1:])
side = A.pop('side', None)
nB = int(A.pop('nB', 40))
FINAL = dict(rebind=1, ct=True, init_min_tau=10, hip_rho=2.0, ctx=False)
V21 = dict(ral=1, ral_other=1, say_rec=1, say_loc=1, scene_held=1, val_veto=0.2, dip=1, act_T=60, soc_feats=1, rec_fix=1)
V22K = dict(veto_R1=1, veto_eps=0.1, veto_W=40, veto_gate=0.15, veto_fast=0.04)
RM = dict(train=1, prune=1, age=20000, regrow=1, alloc='flat')


def one(side, outp, extra):
    d = V22 if side == 'v22' else HERE
    sys.path.insert(0, d)
    from agent import BENCH
    sys.path.insert(0, BENCH)
    from world import World
    from run_j import TUNE_BEST, THINK_FINAL
    if side == 'v22':
        from agent_v22 import AgentV22 as Cls
    else:
        from agent_x2 import AgentX2 as Cls
    kw = dict(TUNE_BEST); kw.update(THINK_FINAL); kw.update(FINAL); kw.update(V21); kw.update(V22K)
    for k, v in extra.items():
        kw[k] = float(v) if '.' in v else int(v)
    wA = World('A', seed=0, tutor='recast', eye_layout='retino')
    ag = Cls(wA, seed=0, rm=dict(RM), **kw)
    wA.run(ag, 20); ag.babble(wA.voice, 35960)
    w = World('B', seed=0, tutor='recast', prev=wA, eye_layout='retino'); w.run(ag, nB)
    w2 = World('C', seed=0, tutor='recast', prev=w, eye_layout='retino'); w2.run(ag, nB)
    reps = np.array([str(r.get('reply')) for r in w.log + w2.log])
    c = ag.core
    np.savez(outp, reps=reps, gen=c.gen.W[:c.mix.n], gate=c.gate.W[:c.mix.n], gateR=c.gateR.W[:c.mix.n],
             sig=c.hip.sig[:c.hip.n], keys=c.hip.keys[:c.hip.n], mixw=c.mix.w, lines=c.mix.lines,
             crit=c.critic.v[:c.mix.n], n=np.array([c.mix.n, c.hip.n, c.stats['learn_events'], c.mix.ev]))
    print(side, 'ok', len(reps), 'learn events', c.stats['learn_events'], 'U' if getattr(c.gen, 'U', None) is not None else '')


if side:
    one(side, A.pop('out'), A)
    sys.exit(0)
t0 = time.time()
ex = [f'{k}={v}' for k, v in A.items()]
fa, fb = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out', 'eq_v22.npz'), \
         os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out', 'eq_x2.npz')
if not os.path.exists(fa):
    subprocess.run([sys.executable, __file__, 'side=v22', f'out={fa}', f'nB={nB}'], check=True)
subprocess.run([sys.executable, __file__, 'side=x2', f'out={fb}', f'nB={nB}'] + ex, check=True)
a, b = np.load(fa), np.load(fb)
same = {k: bool(np.array_equal(a[k], b[k])) for k in a.files}
print('per array identical:', same)
print('EQUIVALENT' if all(same.values()) else 'DIFFERENT', f'{time.time() - t0:.0f}s', 'extra', ex)
