# privileged audit: does each category's program produce the category's own char?
import os
import sys, pickle, numpy as np
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from agent import DevAgent, BENCH
sys.path.insert(0, BENCH)
import world as BW
ag = pickle.load(open(sys.argv[1], "rb"))
v = BW.World("B", seed=0).voice
cat_char = {}
for ch in v.chars:
    pass
# label each category by the corpus char whose OTHER-voice aligned pattern matches it best (privileged)
import itertools
lab = {}
for ch in sorted(set("".join(BW.OBJECTS + BW.SLOTS + BW.COLOURS + BW.COUNTS + BW.NAMES + BW.NOVEL) + "这是什么在哪里你要吗我好不吃很有几颜的谢客气晚安再见拜早上呀对叫刚才看宝")):
    j, m = ag.Z.best(ag.own_pattern(v, ch))
    if m >= 0.75: lab.setdefault(j, ch)
bad = []
for j, y in ag.cat2prog.items():
    said = v.decode(tuple(sorted(int(x) for x in ag.body.prog[y].nonzero()[0])))
    if j in lab and said != lab[j]: bad.append((lab[j], said))
print(f"bound categories {len(ag.cat2prog)}, labelled {sum(1 for j in ag.cat2prog if j in lab)}, wrong program: {len(bad)} {bad[:20]}")
