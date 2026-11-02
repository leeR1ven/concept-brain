# -*- coding: utf-8 -*-
"""Diagnostic (review2/develop): what does drift in untutored use look like?  The final seed-0 agent runs run_j's drift
protocol (C world, probe_frac 1.0, held_frac 0, learning on, 300 dialogues), then frozen C probes as run_j.probe.
Wrong replies are classified: silent / a question or partner line (contains 什么 吗 哪 几 不 or equals a prompt heard
in C) / another task's answer form / other."""
import collections, json, os, pickle, sys
HERE = os.path.dirname(os.path.abspath(__file__))
SYN = os.path.join(HERE, '..')  # v21 copy: code from arch/v2/v21; agent pickle from $AGENT (default synth's seed-0 final agent)
PKL = os.environ.get('AGENT', os.path.join(HERE, '..', '..', 'synth', 'out', 'agent_final_s0.pkl'))
sys.path.insert(0, SYN); sys.path.insert(0, os.path.join(SYN, 'bench_copy'))
sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'synth'))  # repo: unchanged ARCH_v2 core + bench_copy
from world import World            # noqa
import agent_synth, agent_v21      # noqa
agent = pickle.load(open(PKL, 'rb'))
Q = set('什么吗哪几') | {'不'}
def kind(r, prompts):
    rep = r['reply']
    if rep in r['acceptable']:
        return 'ok'
    if not rep:
        return 'silent'
    if rep in prompts or any(ch in Q for ch in rep) or rep.endswith('好不好吃'):
        return 'question/partner line'
    return 'other answer'
def run(stage_kw, n, learning):
    agent.learning = learning
    w = World('C', seed=0, tutor='recast', eye_layout='retino', **stage_kw)
    w.run(agent, n)
    return [r for r in w.log if r.get('kind') == 'probe']
before = run(dict(probe_frac=1.0, held_frac=0.3), 80, False)
drift = run(dict(probe_frac=1.0, held_frac=0.0), 300, True)
after = run(dict(probe_frac=1.0, held_frac=0.3), 80, False)
prompts = {r['prompt'] for r in before + drift + after}
out = {}
for name, rs in (('before', before), ('drift first 100', drift[:len(drift) // 3]), ('drift last 100', drift[-len(drift) // 3:]),
                 ('after (frozen)', after)):
    c = collections.Counter(kind(r, prompts) for r in rs)
    n = len(rs)
    out[name] = {k: round(v / n, 3) for k, v in c.items()}
    wrong = collections.Counter(r['reply'] for r in rs if kind(r, prompts) == 'question/partner line')
    print(f'{name:16s} n {n:3d} ' + '  '.join(f'{k} {v / n:.2f}' for k, v in sorted(c.items())), '| top partner-line replies', wrong.most_common(6), flush=True)
# per task after drift
bt = collections.defaultdict(collections.Counter)
for r in after:
    bt[r['task']][kind(r, prompts)] += 1
for t, c in sorted(bt.items()):
    print('after', t, dict(c))
json.dump(out, open(os.path.join(HERE, 'drift_replies.json'), 'w'), ensure_ascii=False)
