"""Designer's idea: top layer at t-1 makes a Hebbian link to the input at t (earlier -> later).
A: top layer depends only on the current input (layers 1->2->3 feedforward).
B: top layer also gets its own previous state (a loop) -> conjunctive context code.
Readout = Hebbian counts top(t-1)_i -> char(t), normalised per source neuron."""
import json, sys, numpy as np
rng = np.random.default_rng(0)
N, K = 2000, 40

def kwta(v, k=K):
    out = np.zeros_like(v, dtype=bool); out[np.argpartition(-v, k)[:k]] = True; return out

def make(V, loop):
    Win = rng.random((N, V)) < 0.05          # fixed random input -> hidden (stand-in for layers 1-2)
    Wrec = (rng.random((N, N)) < 0.02) if loop else None
    def step(c, prev):
        v = Win[:, c].astype(float) + rng.random(N) * 1e-3
        if loop and prev is not None:
            v = v + 0.6 * (Wrec[:, prev].sum(1) / K)   # previous top state feeds back
        return kwta(v)
    return step

def run(seqs_train, seqs_test, V, loop, focus=None):
    step = make(V, loop)
    C = np.zeros((N, V))
    for s in seqs_train:
        prev = None; top_prev = None
        for c in s:
            if top_prev is not None:
                C[top_prev, c] += 1                  # earlier (top at t-1) -> later (input at t)
            top = step(c, None if top_prev is None else np.flatnonzero(top_prev)); top_prev = top
    P = C / (C.sum(1, keepdims=True) + 1.0)
    hit = n = 0
    for s in seqs_test:
        top_prev = None
        for j, c in enumerate(s):
            if top_prev is not None and (focus is None or focus(s, j)):
                hit += int(np.argmax(P[top_prev].sum(0)) == c); n += 1
            top_prev = step(c, None if top_prev is None else np.flatnonzero(top_prev))
    return hit / max(n, 1), n

# toy: the char after 爱 depends on the char two steps back
chars = list("我你他她爱。")
idx = {c: i for i, c in enumerate(chars)}
pairs = [("我", "你"), ("你", "我"), ("他", "她"), ("她", "他")]
toy = [[idx[a], idx["爱"], idx[b], idx["。"]] for a, b in pairs] * 50
after_love = lambda s, j: j > 0 and s[j - 1] == idx["爱"]
for loop in (False, True):
    acc, n = run(toy, toy[:4], len(chars), loop, after_love)
    print(f"toy  {'B loop' if loop else 'A no loop'}: predict char after 爱: {acc:.2f} (n={n})")

# real corpus: next-char prediction on held-out dialogues
lines = [json.loads(l) for l in open("/root/work/corpus/dialogs.jsonl", encoding="utf-8")]
rs = np.random.default_rng(1); rs.shuffle(lines)
text = lambda d: "。".join(d) + "。"
vocab = sorted({ch for d in lines for ch in text(d)}); vi = {c: i for i, c in enumerate(vocab)}
enc = lambda d: [vi[c] for c in text(d)]
ntr = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
train, test = [enc(d) for d in lines[:ntr]], [enc(d) for d in lines[-1000:]]
for loop in (False, True):
    rng = np.random.default_rng(0)
    acc, n = run(train, test, len(vocab), loop)
    print(f"corpus {ntr} dialogues {'B loop' if loop else 'A no loop'}: next-char top-1 {acc:.3f} (n={n})")
