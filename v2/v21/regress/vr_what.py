# -*- coding: utf-8 -*-
"""Diagnostic (review2/develop): on 'what' probes (这是什么 + pointing), does visual recognition (VR) at the pointed slot
already hold the object's word (comprehension lexicon), for held-out objects too?  The categories of the object's name
are read from the agent's own detections of the partner saying X (heard while listening), so no world label is used
for the check except to group the records.  Final seed-0 agent, frozen."""
import collections, os, pickle, sys
HERE = os.path.dirname(os.path.abspath(__file__))
SYN = os.path.join(HERE, '..')  # v21 copy: code from arch/v2/v21; agent pickle from $AGENT (default synth's seed-0 final agent)
PKL = os.environ.get('AGENT', os.path.join(HERE, '..', '..', 'synth', 'out', 'agent_final_s0.pkl'))
sys.path.insert(0, SYN); sys.path.insert(0, os.path.join(SYN, 'bench_copy'))
sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'synth'))  # repo: unchanged ARCH_v2 core + bench_copy
from world import World            # noqa
import agent_synth, agent_v21      # noqa
agent = pickle.load(open(PKL, 'rb'))
agent.learning = False
core = agent.core
rec = []
orig = core.think_step
def spy(th):
    e = orig(th)
    if th['k'] == 3 and th['kind'] == 'reply':
        rec.append((agent.t, tuple(sorted(int(x) for x in (th['st'].get('VR') if th['st'].get('VR') is not None else [])))))
    return e
core.think_step = spy
# the categories of each object name: detect the partner's voice saying it (agent's own perception, frozen)
from physics import Stream
names = {}
w = World('B', seed=0, tutor='recast', probe_frac=1.0, held_frac=0.3, eye_layout='retino')
for X in ['苹果', '香蕉', '西瓜', '葡萄', '饼干', '牛奶', '皮球', '汽车', '小狗', '小猫', '小鸟', '鸭子']:
    cats = []
    for ch in X:
        f = agent.own_pattern(w.voice, ch) if False else None
        st = Stream(w.voice.F, w.voice.T, w.voice.T // 3, 3, w.voice.levels)
        e = agent._ear_copy(); none = (None, None, False); g = w.voice.hear(ch, 'other'); out = None
        for k in range(3):
            out = e.step(st.tick(((ch, g, k == 0), none, none)))
        j, m = agent.Z.best(out[agent.F0])
        cats.append(int(j) if m >= agent.theta_z else None)
    names[X] = tuple(cats)
off = agent.t
w.run(agent, int(sys.argv[1]) if len(sys.argv) > 1 else 150)
res = collections.defaultdict(lambda: [0, 0, 0])
for r in w.log:
    if r.get('kind') != 'probe' or r['task'] != 'what':
        continue
    X = r['item'][1]; tp = r['t_prompt_end'] + off
    vr = [v for (t, v) in rec if tp - 1 <= t <= tp + 8]
    vr = vr[0] if vr else ()
    has = all(c in vr for c in names[X] if c is not None)
    k = 'held' if r.get('held_out') else 'seen'
    a = res[k]; a[0] += 1; a[1] += int(has and len(vr) > 0); a[2] += int(r['reply'] in r['acceptable'])
for k, (n, h, acc) in sorted(res.items()):
    print(f'what {k}: n {n}  VR at the pointed slot holds the name {h / n:.2f}  reply acceptable {acc / n:.2f}')
