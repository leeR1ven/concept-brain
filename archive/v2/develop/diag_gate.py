# Privileged intervention: open the completion gate (and close the prompt/next gates) whenever a completion exists.
# If one-shot naming then works, recall + programs are fine and the gate's learning is the only bottleneck.
import sys, pickle, numpy as np
sys.dont_write_bytecode = True
sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__)))
import core as C
from agent import DevAgent, BENCH
sys.path.insert(0, BENCH)
from world import World
from metrics import evaluate
for mode in ("as learned", "completion gate forced open"):
    ag = pickle.load(open(sys.argv[1], "rb")); core = ag.core
    ag.learning = True; core.c.learn_listen = False
    if mode != "as learned":
        orig = core._drives
        def drives(st, orig=orig, core=core):
            A, d, gd, total, cands = orig(st)
            cc = cands.get('C', (None, -1))[0]
            if cc is not None and 0 <= cc < core.c.V and st['fam'] >= 1:
                for k, j in (('P', 0), ('N', 1)):
                    cand = cands.get(k, (None, -1))[0]
                    if cand is not None and 0 <= cand < core.c.V:
                        total[cand] -= core.c.gain * np.clip(gd[j], 0, 1)
                total[cc] += core.c.gain * (1 - np.clip(gd[2], 0, 1))
            return A, d, gd, total, cands
        core._drives = drives
    w = World("D", seed=0, tutor="recast", probe_frac=1.0, held_frac=0.3)
    w.run(ag, 150)
    r = evaluate(w.log)["D"]
    print(f"{mode:28s}: oneshot {r['oneshot']['acc']:.2f} (n={r['oneshot']['n']})  name {r['name']['acc']:.2f}  check {r['check']['acc']:.2f}  before {r['before']['acc']:.2f}", flush=True)
