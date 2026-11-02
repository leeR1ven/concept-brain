# Drift in free use (bench): learning ON, nobody teaches (probe_frac=1: every exchange is an untutored free reply).
# Replies to a fixed probe schedule before vs after N untutored dialogues; accuracy and first-char entropy.
import os, sys, copy, pickle, collections, numpy as np
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from agent import DevAgent, BENCH
sys.path.insert(0, BENCH)
from world import World
from metrics import evaluate
ag = pickle.load(open(sys.argv[1], "rb")); st = sys.argv[2] if len(sys.argv) > 2 else "C"; N = int(sys.argv[3]) if len(sys.argv) > 3 else 600
def probe(a):
    a = copy.deepcopy(a); a.learning = False
    w = World(st, seed=0, tutor="recast", probe_frac=1.0, held_frac=0.3, ex_per_dialogue=(1, 1)); w.run(a, 200)
    return [r["reply"] for r in w.log], evaluate(w.log)[st]["ALL"]["acc"]
r0, a0 = probe(ag)
ag.learning = True
w = World(st, seed=0, tutor="recast", probe_frac=1.0, held_frac=0.0)       # untutored free use, learning on
w.run(ag, N)
r1, a1 = probe(ag)
same = np.mean([x == y for x, y in zip(r0, r1)])
H = lambda rs: (lambda p: float(-(p * np.log2(p)).sum()))(np.array(list(collections.Counter(x[:1] for x in rs).values()), float) / len(rs))
print(f"free use {N} untutored {st} dialogues with learning on: identical replies {same:.2f}; acc {a0:.2f} -> {a1:.2f}; "
      f"first-char entropy {H(r0):.2f} -> {H(r1):.2f}")
