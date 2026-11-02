# -*- coding: utf-8 -*-
"""Diagnostic (review2/develop): comprehension vs production on held-out items ("looking while listening").
For prompts that name an object without pointing (where: X在哪里, colour: X是什么颜色, count: 有几个X), record the slot
the agent's reply thought attends after its comparator step (k=2) and compare it with X's true slot; split by
held-out (frame x filler never taught) and by whether the reply was acceptable.  Final seed-0 agent, frozen."""
import collections, json, os, pickle, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
SYN = os.path.join(HERE, '..')  # v21 copy: code from arch/v2/v21; agent pickle from $AGENT (default synth's seed-0 final agent)
PKL = os.environ.get('AGENT', os.path.join(HERE, '..', '..', 'synth', 'out', 'agent_final_s0.pkl'))
sys.path.insert(0, SYN); sys.path.insert(0, os.path.join(SYN, 'bench_copy'))
sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'synth'))  # repo: unchanged ARCH_v2 core + bench_copy
from world import World            # noqa
import agent_synth, agent_v21      # noqa
with open(PKL, 'rb') as f:
    agent = pickle.load(f)
agent.learning = False
core = agent.core
rec = []
orig = core.think_step
def spy(th):
    e = orig(th)
    if th['k'] == 2:
        rec.append((agent.t, th['kind'], th['st'].get('att'), th['st'].get('att_tutor')))
    return e
core.think_step = spy
n = int(sys.argv[1]) if len(sys.argv) > 1 else 120
res = collections.defaultdict(lambda: [0, 0, 0, 0])   # n, attended X's slot, acc, (attended & acc)
for stage in ('B', 'C'):
    w = World(stage, seed=0, tutor='recast', probe_frac=1.0, held_frac=0.3, eye_layout='retino')
    off = agent.t - 0
    w.run(agent, n)
    for r in w.log:
        if r.get('kind') != 'probe' or r['task'] not in ('where', 'colour', 'count') or not r.get('item'):
            continue
        X = r['item'][1]
        slots = [i for i, it in enumerate(r['scene'] or ()) if it is not None and it[0] == X]
        if not slots:
            continue
        tp = r['t_prompt_end'] + off
        th = [x for x in rec if tp - 1 <= x[0] <= tp + 8 and x[1] == 'reply' and x[2] is not None]
        att = th[0][2] if th else None
        acc = r['reply'] in r['acceptable']
        k = (r['task'], 'held' if r.get('held_out') else 'seen')
        a = res[k]; a[0] += 1; a[1] += int(att == slots[0]); a[2] += int(acc); a[3] += int(att == slots[0] and acc)
for k in sorted(res):
    n_, at, ac, both = res[k]
    print(k, f'n {n_}  attends X (comprehension) {at / n_:.2f}  reply acceptable {ac / n_:.2f}  both {both / n_:.2f}  (chance attend 0.33)')
json.dump({f'{a}|{b}': v for (a, b), v in res.items()}, open(os.path.join(HERE, os.environ.get('OUTJ', 'comprehend.json')), 'w'))
