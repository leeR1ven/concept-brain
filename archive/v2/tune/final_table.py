# per-failure table: mean (sd) over seeds for each tag
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze import load, score
tags = sys.argv[1].split(',')
recs = load(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out', 'runs', '*.json'))
cols = [('J', 'J'), ('B3', 'B end'), ('C3', 'C end'), ('D3', 'D end'), ('RB', 'B3-B1'), ('RC', 'C3-C2'), ('H', 'held-out'),
        ('hB', 'held B'), ('hC', 'held C'), ('hD', 'held D'), ('DRb', 'C pre-drift'), ('DR', 'C post-drift'),
        ('colour', 'colour'), ('count', 'count'), ('check', 'check'), ('name', 'name'), ('oneshot', 'one-shot')]
print('| config | seeds | ' + ' | '.join(c[1] for c in cols) + ' | mech viol |')
print('|' + '---|' * (len(cols) + 3))
for t in tags:
    S = [score(r) for r in recs if r['tag'] == t and r['R'] == 3]
    if not S:
        continue
    for s in S:
        s['RB'] = s['B3'] - s['B1']; s['RC'] = s['C3'] - s['C2']
    cells = []
    for k, _ in cols:
        v = [s[k] for s in S if isinstance(s.get(k), (int, float))]
        cells.append('-' if not v else (f"{np.mean(v):.2f}" + (f" ({np.std(v, ddof=1):.2f})" if len(v) > 1 else "")))
    print(f"| {t} | {','.join(str(s['seed']) for s in sorted(S, key=lambda x: x['seed']))} | " + ' | '.join(cells)
          + f" | {sum(s['viol'] for s in S)} |")
