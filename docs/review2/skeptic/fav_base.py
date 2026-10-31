# token-weighted base rate for the 'favourite thoughts' measure: share of the 20 most frequent heard chars among
# all heard char tokens (partner + other2 speech) in the ARCH_v2 life log (seed 5).  Read-only.
import pickle, collections, sys, json
p = sys.argv[1]
log = pickle.load(open(p, 'rb'))
cnt = collections.Counter()
kinds = collections.Counter()
for r in log:
    kinds[(r.get('type'), r.get('kind'))] += 1
    for k in ('prompt', 'recast', 'model_reply'):
        t = r.get(k)
        if isinstance(t, str) and r.get('type') != 'init':
            cnt.update(t)
tot = sum(cnt.values())
top = cnt.most_common(20)
share = sum(v for _, v in top) / tot
print('records', len(log), 'distinct chars', len(cnt), 'tokens', tot)
print('top20', ''.join(c for c, _ in top))
print('token share of top20 = %.3f ; type share = %.3f' % (share, 20 / len(cnt)))
print('kinds', dict(kinds.most_common(12)))
