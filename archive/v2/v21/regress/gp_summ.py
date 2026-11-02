import json, sys, collections
import numpy as np
G = collections.defaultdict(list)
for l in open(sys.argv[1] if len(sys.argv) > 1 else 'gp_dev.jsonl'):
    r = json.loads(l)
    k = (r['variant'], r.get('tutor'), json.dumps(r.get('ralkw') or {}, sort_keys=True))
    G[k].append(r)
for k, R in G.items():
    fh = [r['frozen']['held'] for r in R]; fs = [r['frozen']['seen'] for r in R]
    lh = [r['ALL']['held'] for r in R]; ls = [r['ALL']['seen'] for r in R]
    pl = [r['frozen']['held_cls'].get('partner_line', 0) for r in R]
    print(f"{k[0]:5s} {k[1]:6s} {k[2]:45s} seeds {[r['seed'] for r in R]} frozen seen {np.mean(fs):.2f} held {np.mean(fh):.2f} "
          f"{[round(x, 2) for x in fh]} | live seen {np.mean([x for x in ls if x is not None]):.2f} held {np.mean(lh):.2f} | pl {np.mean(pl):.2f}")
