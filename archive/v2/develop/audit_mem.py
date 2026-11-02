import os
import sys, pickle, numpy as np, collections
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from agent import DevAgent, BENCH
sys.path.insert(0, BENCH)
import world as BW
ag = pickle.load(open(sys.argv[1], "rb"))
v = BW.World("B", seed=0).voice
print("bab_pat", ag.bab_pat.shape, "bab_set", len(ag.bab_set), "distinct sets", len(set(map(tuple, ag.bab_set))), "stats", ag.stats)
dec = [v.decode(tuple(sorted(int(x) for x in s))) for s in ag.bab_set[:50]]
print("first entries decode to:", dec[:20], "types", type(ag.bab_set[0]), ag.bab_set[0])
