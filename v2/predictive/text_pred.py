# -*- coding: utf-8 -*-
"""Text-level check of the predictor (the coordinator's protocol, coord/ngram.py + coord/rnn_idea.py): next-char top-1 on
the last 1,000 shuffled dialogues (rng 1), lines joined by '。'.  The learner is ZP's rule (PR: signed, error-driven,
normalised, one online pass) with ARCH_v2-P's context codes instead of Hebbian counts:
  x1 / x2   the previous char / the one before (localist)
  R         the designer's recurrent predictor: R(t) = kWTA(Win c(t) + 0.6 Wrec R(t-1)), N 2000, k 40 (fixed random)
  Gline     the current line so far: union of fixed 8-unit codes of its unigrams + bigrams (2,048)
  Ghold     the previous line's Gline (the held prompt)
Usage: python3 text_pred.py N_TRAIN 'x1;x1,x2;x1,x2,R;x1,x2,gline,ghold;x1,x2,R,gline,ghold'"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.environ.get('PRED_HERE', os.path.dirname(os.path.abspath(__file__))))
from pr import Predictor, F32, topk_mask  # noqa: E402

CORPUS = next(p for p in ('/root/work/corpus/dialogs.jsonl', os.path.expanduser('~/v2lab/data/dialogs.jsonl')) if os.path.exists(p))
lines = [json.loads(l) for l in open(CORPUS, encoding='utf-8')]
rs = np.random.default_rng(1); rs.shuffle(lines)
text = lambda d: '。'.join(d) + '。'  # noqa: E731
vocab = sorted({ch for d in lines for ch in text(d)}); vi = {c: i for i, c in enumerate(vocab)}
V = len(vocab); END = vi['。']
ntr = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
variants = [v.split(',') for v in (sys.argv[2] if len(sys.argv) > 2 else 'x1;x1,x2;x1,x2,R;x1,x2,gline,ghold;x1,x2,R,gline,ghold').split(';')]
train = [[vi[c] for c in text(d)] for d in lines[:ntr]]
test = [[vi[c] for c in text(d)] for d in lines[-1000:]]
NG, M, NR, KR = 2048, 8, 2000, 40
rng0 = np.random.default_rng(0)
Win = rng0.random((NR, V)) < 0.05
Wrec = rng0.random((NR, NR)) < 0.02
codes = {}


def code(key):
    c = codes.get(key)
    if c is None:
        c = codes[key] = np.random.default_rng([7] + list(key)).choice(NG, M, replace=False)
    return c


def norm(x):
    x = x.astype(F32); n = float(np.sqrt((x * x).sum())); return x / n if n > 0 else x


def run(use, eta=0.5):
    P = Predictor(V, eta=eta, theta=1.0, bias=True)
    for nm, n in (('x1', V), ('x2', V), ('R', NR), ('gline', NG), ('ghold', NG)):
        if nm in use:
            P.add(nm, n)

    def stream(seqs, learn):
        hit = n = 0
        for s in seqs:
            prev1 = prev2 = -1; Rs = np.zeros(NR, bool); gl = np.zeros(NG, bool); gh = np.zeros(NG, bool); last = -1
            for j, c in enumerate(s):
                if j > 0:
                    xs = {}
                    if 'x1' in use: z = np.zeros(V, F32); z[prev1] = 1; xs['x1'] = z
                    if 'x2' in use and prev2 >= 0: z = np.zeros(V, F32); z[prev2] = 1; xs['x2'] = z
                    if 'R' in use: xs['R'] = norm(Rs)
                    if 'gline' in use: xs['gline'] = norm(gl)
                    if 'ghold' in use: xs['ghold'] = norm(gh)
                    p = P.predict(xs)
                    if not learn:
                        hit += int(np.argmax(P.d) == c); n += 1
                    else:
                        y = np.zeros(V, F32); y[c] = 1; P.learn(y)
                # update the context with char c
                if 'R' in use:
                    v = Win[:, c].astype(F32) + rng0.random(NR).astype(F32) * 1e-3
                    if Rs.any():
                        v = v + 0.6 * Wrec[:, Rs].sum(1) / KR
                    Rs = topk_mask(v, KR)
                if c == END:
                    gh = gl.copy(); gl = np.zeros(NG, bool); last = -1
                else:
                    gl[code((1, c))] = True
                    if last >= 0:
                        gl[code((2, last, c))] = True
                    last = c
                prev2, prev1 = prev1, c
        return hit / max(n, 1), n
    t0 = time.time()
    stream(train, True)
    acc, n = stream(test, False)
    return acc, n, time.time() - t0


for use in variants:
    acc, n, sec = run(set(use))
    print(f'{ntr} dialogues, PR predictor, context {"+".join(use):24s}: next-char top-1 {acc:.3f} (n={n}, {sec:.0f}s)', flush=True)
