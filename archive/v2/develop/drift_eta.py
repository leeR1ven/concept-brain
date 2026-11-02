# Drift in untutored free use vs the learning rate (tuning): C probes before/after 300 untutored dialogues.
import sys, copy, pickle
sys.dont_write_bytecode = True
sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__)))
from agent import DevAgent, BENCH
sys.path.insert(0, BENCH)
from world import World
from ablate import run_one
ag = pickle.load(open(sys.argv[1], "rb"))
base = run_one(ag, "full", {}, "C", 80, 0)["ALL"][0]
for scale in (1.0, 0.3, 0.1):
    a2 = copy.deepcopy(ag); a2.learning = True
    a2.core.gen.eta *= scale; a2.core.gate.eta *= scale
    w = World("C", seed=0, tutor="recast", probe_frac=1.0, held_frac=0.0); w.run(a2, 300)
    after = run_one(a2, "full", {}, "C", 80, 0)["ALL"][0]
    print(f"learning rates x{scale}: C probes {base} -> {after} after 300 untutored dialogues", flush=True)
