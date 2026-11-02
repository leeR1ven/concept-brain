"""Count how much speech an agent hears in the bench's staged J protocol, in one life, and per world2 day
(Silent agent; recast tutor as run_j). Read-only use of the repo bench."""
import sys, time, collections
sys.path.insert(0, 'v2/bench')
from world import World
from baselines import Silent

def heard_chars(rec):
    n = 0; lines = 0
    for k in ('prompt', 'model_reply', 'recast'):
        v = rec.get(k)
        if v:
            n += len(v); lines += 1
    return n, lines

seed = 0
ag = Silent()
tot = collections.Counter()
t0 = time.time()
wA = World('A', seed=seed, tutor='recast'); wA.run(ag, 300)
prev = wA
logs = [('A', wA.log)]
for st in ['B', 'BC', 'BCDD']:
    w = World(st, seed=seed, tutor='recast', prev=prev); w.run(ag, 1000); prev = w
    logs.append((st, w.log))
ticks = prev.t
grand = 0; glines = 0
obj_mentions = collections.Counter()
OBJ = ['苹果','香蕉','饼干','牛奶','西瓜','葡萄','小狗','小猫','皮球','汽车','小鸟','鸭子']
for st, lg in logs:
    c = 0; L = 0; kinds = collections.Counter()
    for r in lg:
        n, l = heard_chars(r); c += n; L += l; kinds[r.get('kind')] += 1
        for k in ('prompt', 'model_reply', 'recast'):
            v = r.get(k) or ''
            for o in OBJ:
                obj_mentions[o] += v.count(o)
    grand += c; glines += L
    print(f'stage {st}: records {len(lg)}, heard lines {L}, heard syllables {c}, kinds {dict(kinds)}')
print(f'staged protocol total: heard lines {glines}, syllables {grand}, world ticks {ticks}, {time.time()-t0:.1f}s')
print('object-name exposures:', dict(obj_mentions), 'mean', sum(obj_mentions.values())/len(OBJ))

# one life, 10^6 ticks
from life import Life
L = Life(seed=5, tutor='recast')
lg = L.run(Silent(), ticks=1_000_000)
c = 0; nl = 0; init = 0
kinds = collections.Counter()
for r in lg:
    if r.get('type') == 'ex':
        n, l = heard_chars(r); c += n; nl += l; kinds[r.get('kind')] += 1
print(f'life 10^6 ticks (seed 5, recast): ex records {sum(kinds.values())}, heard lines {nl}, syllables {c}, kinds {dict(kinds)}, ticks {L.t}')
