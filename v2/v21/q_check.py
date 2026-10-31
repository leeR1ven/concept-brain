# -*- coding: utf-8 -*-
"""Does the learned question detector (ral_q) find the question words?  A short life; weights per syllable read
through the agent's own ear categories.  usage: python3 q_check.py [ticks=200000] [world=life|staged]"""
import os, sys, time, json
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'synth'))  # repo: unchanged ARCH_v2 core + bench_copy
from agent_v21 import AgentV21
from agent import BENCH
sys.path.insert(0, BENCH)
from life import Life
from physics import Stream
from run_v21 import TUNE_BEST, ARCH_V2
from life_synth import THINK_LIFE
A = dict(a.split('=', 1) for a in sys.argv[1:])
ticks = int(float(A.get('ticks', 2e5)))
kw = dict(TUNE_BEST); kw.update(ARCH_V2); kw.update(ct_assoc=0.5, wander=50, wander_after=200); kw.update(THINK_LIFE)
kw.update(ral=1, ral_other=1, ral_q=1)
L = Life(seed=5, tutor='recast', eye_layout='retino')
ag = AgentV21(L, seed=5, **kw); ag.babble(L.voice, 35960)
t0 = time.time(); L.run(ag, ticks=ticks)
def cat(ch):
    v = L.voice; st = Stream(v.F, v.T, v.T // 3, 3, v.levels); e = ag._ear_copy(); none = (None, None, False)
    g = v.hear(ch, 'other'); out = None
    for k in range(3):
        out = e.step(st.tick(((ch, g, k == 0), none, none)))
    j, m = ag.Z.best(out[ag.F0])
    return int(j) if m >= ag.theta_z else None
core = ag.core
print(f'{time.time() - t0:.0f}s', {k: core.ral_stats[k] for k in ('q_pos', 'q_neg', 'q_kept', 'answered', 'contingent')})
for ch in '什么吗哪几谁是不色个边在好吃要你这我叫再见晚安香蕉苹果红两':
    c = cat(ch)
    print(ch, c, None if c is None else round(core.q_w.get(c, 0.0), 3), end=' | ')
print()
for line in ('这是什么', '香蕉', '在左边', '红色', '两个', '好吃', '要香蕉', '你要香蕉吗', '有几个香蕉', '香蕉在哪里', '你好', '再见', '是', '不是', '对是香蕉'):
    cs = [cat(ch) for ch in line]
    print(line, round(core.q_p([c for c in cs if c is not None]), 2), end=' | ')
print()
