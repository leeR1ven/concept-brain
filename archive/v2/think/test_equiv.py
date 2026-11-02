# develop agent vs ThinkAgent(policy v1, think_k=3, ior off): the replies must be identical (the thought = begin_line)
import os, sys, time
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from agent import DevAgent, BENCH
from agent_think import ThinkAgent
sys.path.insert(0, BENCH)
from world import World
nA, nB = int(sys.argv[1]), int(sys.argv[2])
out = {}
for name in ('dev', 'think_v1'):
    t0 = time.time()
    w = World('A', seed=0, tutor='recast')
    ag = DevAgent(w, seed=0, floor=0.3) if name == 'dev' else ThinkAgent(w, seed=0, floor=0.3, policy='v1', think_k=3, ior=False)
    w.run(ag, nA); ag.babble(w.voice, 35960)
    w2 = World('B', seed=0, tutor='recast', prev=w); w2.run(ag, nB)
    out[name] = [(r['prompt'], r['reply'], r.get('latency'), r.get('after_recast_reply')) for r in w.log + w2.log]
    print(name, len(out[name]), f"{time.time()-t0:.0f}s", flush=True)
a, b = out['dev'], out['think_v1']
diff = [i for i in range(min(len(a), len(b))) if a[i] != b[i]]
print('records', len(a), len(b), 'differences', len(diff), 'first', [(a[i], b[i]) for i in diff[:3]])
