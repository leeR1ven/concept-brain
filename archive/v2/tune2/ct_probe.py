# -*- coding: utf-8 -*-
"""Continuous-thought dynamics on the real bench stream (SIMULATED): after each heard line, the held items P vs the
line; the subthreshold residue of the previous line; P at the onset of a recast (after the own reply) vs the question.
usage: python3 ct_probe.py [lam=0.3] [tau=0] [nA=150] [nB=200]"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np
from agent_synth import SynthAgent
from agent import BENCH
sys.path.insert(0, BENCH)
from world import World
from run_j import TUNE_BEST, THINK_FINAL
A = dict(a.split('=', 1) for a in sys.argv[1:])
lam, tau = float(A.get('lam', 0.3)), float(A.get('tau', 0))
kw = dict(TUNE_BEST); kw.update(THINK_FINAL); kw.update(ct=True, ct_lam=lam, ct_tau=tau, rebind=1)
wA = World('A', seed=0, tutor='recast'); ag = SynthAgent(wA, seed=0, **kw)
wA.run(ag, int(A.get('nA', 150))); ag.babble(wA.voice, 35960)
core = ag.core
rec = dict(lines=0, eq=0, sizes=[], residue=[], recast_ctx=[], recast_n=0)
orig_input = core._ct_input
prev_items = [None]
def hooked(r, surprise=None):
    before = core.T_a.copy()
    orig_input(r, surprise)
    items = [i for i in r['items'] if 0 <= i < core.c.V]
    held = set(np.flatnonzero(core.T_a >= core.c.ct_theta).tolist())
    rec['lines'] += 1; rec['eq'] += int(held == set(items)); rec['sizes'].append(len(held))
    if prev_items[0] is not None:
        old = [i for i in prev_items[0] if i not in set(items)]
        if old:
            rec['residue'].append(float(np.mean(core.T_a[old])))
    prev_items[0] = items
core._ct_input = hooked
own_end = [None]
orig_finish = core.own_finish
def fin(st, t_end=None):
    own_end[0] = (list(st.get('items', [])), [int(i) for i in np.flatnonzero(core.T_a >= core.c.ct_theta)])
    return orig_finish(st, t_end=t_end)
core.own_finish = fin
orig_begin = core.begin_line
def bl(eye, att):
    st = orig_begin(eye, att)
    if own_end[0] is not None and own_end[0][0] and not core._replaying:
        rec['recast_n'] += 1
        rec['recast_ctx'].append(int(set(st.get('Pbag') or st['P']) == set(own_end[0][1])))
        own_end[0] = None
    return st
core.begin_line = bl
wB = World('B', seed=0, tutor='recast', prev=wA); wB.run(ag, int(A.get('nB', 200)))
print(f"lam {lam} tau {tau}: lines {rec['lines']}, held set == the new line's items {rec['eq'] / max(rec['lines'], 1):.3f}, "
      f"held items mean {np.mean(rec['sizes']):.2f}, residue of line n-1 (subthreshold activation) {np.mean(rec['residue']):.3f}, "
      f"first heard line after an own reply: context == the thought at the reply's end {np.mean(rec['recast_ctx']):.3f} (n {rec['recast_n']})")
