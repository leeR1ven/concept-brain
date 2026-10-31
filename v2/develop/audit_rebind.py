import os
import sys, pickle, numpy as np
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import agent as AG
from agent import DevAgent, BENCH
sys.path.insert(0, BENCH)
import world as BW
ag = pickle.load(open(sys.argv[1], "rb"))
ag.__class__ = AG.DevAgent; ag.theta_bind = 0.85
v = BW.World("B", seed=0).voice
ag.cat2prog, ag.prog2cat = {}, {}
for j in range(ag.Z.n):
    ag._bind_from_memory(j, ag.Z.W[:, j] >= 0.5)
lab = {}
for ch in sorted(set("".join(BW.OBJECTS + BW.SLOTS + BW.COLOURS + BW.COUNTS + BW.NAMES + BW.NOVEL) + "这是什么在哪里你要吗我好不吃很有几颜的谢客气晚安再见拜早上呀对叫刚才看宝")):
    j, m = ag.Z.best(ag.own_pattern(v, ch))
    if m >= 0.75: lab.setdefault(j, ch)
bad = [(lab[j], v.decode(tuple(sorted(int(x) for x in ag.body.prog[y].nonzero()[0])))) for j, y in ag.cat2prog.items() if j in lab
       and v.decode(tuple(sorted(int(x) for x in ag.body.prog[y].nonzero()[0]))) != lab[j]]
print(f"re-bound {len(ag.cat2prog)} of {ag.Z.n} categories from the babbling memory ({len(ag.bab_set)} sounds); wrong: {bad}")
