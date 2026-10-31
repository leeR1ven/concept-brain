# distinct MIX keys per type (pair schema) in the saved ARCH_v2 agent (synth/out/agent_final_s0.pkl, read only)
import os, sys, pickle, collections, json
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, 'bench_copy'))
import numpy as np
ag = pickle.load(open(os.path.join(HERE, '..', 'synth', 'out', 'agent_final_s0.pkl'), 'rb'))
mix = ag.core.mix
cnt = collections.Counter(); use = collections.Counter(); vals = collections.defaultdict(set)
gen = ag.core.gen
for k, u in mix.ids.items():
    t = k[0] if len(k) == 2 else (k[0], k[2])
    cnt[t] += 1; use[t] += float(gen.use[u]) if u < len(gen.use) else 0
    if len(k) == 2: vals[k[0]].add(k[1])
out = {('|'.join(t) if isinstance(t, tuple) else t): [c, round(use[t] / max(c, 1), 1)] for t, c in cnt.most_common()}
print(json.dumps(out, ensure_ascii=False))
print('total', sum(cnt.values()), 'n', mix.n)
print({g: len(v) for g, v in vals.items()})
print('V used', ag.Z.n, 'templates active bits', [int((ag.Z.W[:, j] >= 0.5).sum()) for j in range(5)], 'NF', ag.Z.W.shape)
json.dump(out, open(os.path.join(HERE, 'out', 'mix_keys_final_s0.json'), 'w'), ensure_ascii=False)
