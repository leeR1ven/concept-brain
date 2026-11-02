# -*- coding: utf-8 -*-
"""Diagnostic (review2/develop): what does the final seed-0 agent SAY on held-out (never-taught) frame x filler probes?
Loads synth/out/agent_final_s0.pkl (read only), runs frozen C and D probe worlds (probe_frac 1, held_frac 0.3, as
run_j.probe), and dumps every reply with its prompt, acceptable set and held flag."""
import collections, copy, json, os, pickle, resource, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
SYN = os.path.join(HERE, '..', '..', '..', 'v2', 'synth')
sys.path.insert(0, SYN); sys.path.insert(0, os.path.join(SYN, 'bench_copy'))
from world import World            # noqa
import agent_synth                 # noqa  (classes for unpickling)
t0 = time.time()
with open(os.path.join(SYN, 'out', 'agent_final_s0.pkl'), 'rb') as f:
    agent = pickle.load(f)
print('loaded', round(time.time() - t0), 's, maxrss MB', resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024, flush=True)
n = int(sys.argv[1]) if len(sys.argv) > 1 else 80
out = {}
for stage in sys.argv[2:] or ['C', 'B', 'D']:   # no deepcopy (RAM): C and B frozen first, D (index on) last
    ag = agent
    if stage in 'BC':
        ag.learning = False
    else:
        ag.learning = True; ag.core.c.learn_listen = False
    w = World(stage, seed=0, tutor='recast', probe_frac=1.0, held_frac=0.3, eye_layout='retino')
    w.run(ag, n)
    recs = [dict(task=r['task'], held=bool(r.get('held_out')), prompt=r['prompt'], reply=r['reply'],
                 acc=r['reply'] in (r.get('acceptable') or []), acceptable=r.get('acceptable'))
            for r in w.log if r.get('kind') == 'probe']
    out[stage] = recs
    by = collections.defaultdict(list)
    for r in recs:
        by[(r['task'], r['held'])].append(r)
    for k in sorted(by):
        rs = by[k]
        c = collections.Counter(r['reply'] for r in rs)
        print(stage, k, 'n', len(rs), 'acc %.2f' % (sum(r['acc'] for r in rs) / len(rs)), c.most_common(8), flush=True)
    print('maxrss MB', resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024, round(time.time() - t0), 's', flush=True)
with open(os.path.join(HERE, 'heldout_replies.json'), 'w') as f:
    json.dump(out, f, ensure_ascii=False, indent=0)
