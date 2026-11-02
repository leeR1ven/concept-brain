# ritual / colour thoughts in stage C: release step, confidence, gain, outcome (matched by prompt end)
import sys, pickle, collections, numpy as np
p = sys.argv[1]
L = pickle.load(open(p + '_log.pkl', 'rb'))
TH = pickle.load(open(p + '_thoughts.pkl', 'rb'))
by_open = collections.defaultdict(list)
for t in TH:
    if t['kind'] == 'reply':
        by_open[t['t_open']].append(t)
for task in ('ritual', 'colour', 'offer'):
    rs = [r for r in L if r['stage'] == 'C' and r['task'] == task and r['kind'] in ('probe', 'recast') and 2300 <= r['dlg'] < 4300]
    rows = []
    for r in rs:
        t = None
        for d in range(4):
            if by_open.get(r['t_prompt_end'] + d):
                t = by_open[r['t_prompt_end'] + d][0]; break
        if t is None:
            continue
        rows.append((r['reply'] in r['acceptable'], bool(r['reply']), t['outcome'], t['tau_rel'], t['k_rel'], t['e0'], t['e_rel'], t['G'], t['p_rel']))
    print(task, len(rs), 'matched', len(rows))
    print('  outcomes', collections.Counter((a, o) for a, s, o, *_ in rows).most_common(8))
    G = [x[7] for x in rows]; e0 = [x[5] for x in rows if x[5] is not None]
    print('  G mean %.2f  e0 mean %.2f' % (np.mean(G), np.mean(e0)))
    sil = [x for x in rows if not x[1]]
    print('  silent examples', [(o, tr, kr, round(e0 or 0, 2), round(G, 2), pr) for _, _, o, tr, kr, e0, er, G, pr in sil[:6]])
    bad = [x for x in rows if x[1] and not x[0]]
    print('  wrong examples', [(tr, kr, round(e0 or 0, 2), round(er or 0, 2), round(G, 2)) for _, _, o, tr, kr, e0, er, G, pr in bad[:6]])
