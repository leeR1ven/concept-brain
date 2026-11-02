# Babbling coverage under the bench's random physics: how many babbles until every char needed by stages A-D is
# produced?  (a) random 4-muscle sets with replacement (a naive infant), (b) one pass over all sets in random order.
import os, sys, itertools, collections, numpy as np
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "bench_copy"))
from physics import Voice, corpus_chars
import world as BW
v = Voice(corpus_chars(), seed=0)
need = set()
for o in BW.OBJECTS + BW.COLOURS + BW.COUNTS + BW.SLOTS + BW.NAMES + BW.NOVEL: need |= set(o)
for q, acc in BW.RITUALS: need |= set(q) | set("".join(acc))
need |= set("这是在要我好吃很有的不对叫刚才吗哪里什么看宝")
sets = list(itertools.combinations(range(32), 4))
dec = [v.decode(s) for s in sets]
cnt = collections.Counter(dec)
ones = sum(1 for c in need if cnt.get(c, 0) == 1)
print(f"needed chars (stages A-D): {len(need)}; produced by exactly one of 35,960 sets: {ones}; by >= 10 sets: "
      f"{sum(1 for c in need if cnt.get(c, 0) >= 10)}; distinct producible chars overall: {len(cnt)}")
rng = np.random.default_rng(0)
marks = [5000, 20000, 35960, 100000, 200000]
for mode in ("random with replacement", "one pass, no repeats"):
    found = set(); out = []
    order = rng.permutation(len(sets))
    for n in range(1, marks[-1] + 1):
        i = int(rng.integers(len(sets))) if mode.startswith("random") else int(order[(n - 1) % len(sets)])
        c = dec[i]
        if c in need: found.add(c)
        if n in marks: out.append(f"{n}: {len(found)}/{len(need)}")
    print(f"{mode:24s} " + "  ".join(out))
