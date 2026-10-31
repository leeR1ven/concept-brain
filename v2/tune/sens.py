# sensitivity table: each one-at-a-time config vs the base of its rung (same seed), with the seed noise of the base
# usage: python3 sens.py PREFIX (r1_ | r2_) [metric J]
import sys, glob, os, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze import load, score
pre = sys.argv[1]; met = sys.argv[2] if len(sys.argv) > 2 else 'J'
recs = load(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out', 'runs', pre + '*.json'))
S = [score(r) for r in recs]
base = [s for s in S if s['tag'] == pre + 'base']
b0 = [s for s in base if s['seed'] == 0]
if not b0:
    sys.exit('no base seed 0')
b0 = b0[0]
sd = float(np.std([s[met] for s in base], ddof=1)) if len(base) > 1 else float('nan')
print(f"base {pre}base seed0 {met} {b0[met]:.3f}; base over seeds {[round(s[met], 3) for s in base]} sd {sd:.3f}")
keys = ['B1', 'hB1', 'DR'] if pre.startswith('r1') else ['B1', 'C2', 'B3', 'C3', 'D3', 'RET', 'H', 'DR', 'check', 'name', 'oneshot']
rows = []
for s in S:
    if s['seed'] != 0 or s['tag'] == pre + 'base':
        continue
    rows.append((s[met] - b0[met], s))
rows.sort(key=lambda x: -x[0])
for d, s in rows:
    cfg = next(r for r in recs if r['tag'] == s['tag'] and r['seed'] == 0)['cfg']
    print(f"{d:+.3f}  {s['tag']:22s} {met} {s[met]:.3f} viol {s['viol']} | " + " ".join(
        f"{k} {s[k]:.2f}" if isinstance(s.get(k), (int, float)) else f"{k} -" for k in keys))
