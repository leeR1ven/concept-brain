import json, sys, numpy as np
from collections import Counter, defaultdict
lines = [json.loads(l) for l in open("/root/work/corpus/dialogs.jsonl", encoding="utf-8")]
rs = np.random.default_rng(1); rs.shuffle(lines)
text = lambda d: "。".join(d) + "。"
test = [text(d) for d in lines[-1000:]]
for ntr in (5000, 30000):
    train = [text(d) for d in lines[:ntr]]
    cnt = [defaultdict(Counter) for _ in range(4)]
    for s in train:
        for j in range(1, len(s)):
            for o in range(1, 4):
                if j - o >= 0: cnt[o][s[j-o:j]][s[j]] += 1
    res = {}
    for order in (1, 2, 3):
        hit = n = 0
        for s in test:
            for j in range(1, len(s)):
                pred = None
                for o in range(min(order, j), 0, -1):     # back off to shorter context
                    c = cnt[o].get(s[j-o:j])
                    if c: pred = c.most_common(1)[0][0]; break
                hit += pred == s[j]; n += 1
        res[order] = round(hit / n, 3)
    print(f"{ntr} dialogues: next-char top-1 with context of 1/2/3 chars: {res[1]} / {res[2]} / {res[3]}  (n={n})")
