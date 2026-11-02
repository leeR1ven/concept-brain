"""composition of the deep-layer class codes: which context's randomly indexed units each filler's code holds.
python3 -I diag6.py <concept.py args>   (runs the full life, then analyses the final network)"""
import sys, os, json, numpy as np; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import concept as C
cfg, outp = C.parse(sys.argv[1:])
tl, info = C.build_stream(cfg, verbose=False)
A = C.Area2(cfg, info['feat_A'], info['feat_E'])
asm, cix = info['asm'], info['cix']
eye = np.zeros(0, np.int64); aud = None
for t, ev in tl:
    if 'eye' in ev: eye = ev['eye'].astype(np.int64) + C.N_A
    if 'end' in ev: aud = None
    if 'ch' in ev: aud = ev['ch']
    A.step(t, eye if aud is None else np.union1d(asm[aud], eye))
k2 = int(round(cfg['f2'] * A.N2))
ctx = '吃玩要是个的把你'
idx = {}
for c in ctx:
    x1 = A.c1.probe([asm[cix[c]].astype(np.int64)])[0]
    idx[c] = np.argsort(-A.K.push(x1))[:k2]
out = {}
for X in C.OBJECTS:
    x = A.probe([asm[cix[X[0]]].astype(np.int64)])[0]
    x2 = x[x >= A.N1] - A.N1
    fr = {c: round(len(np.intersect1d(x2, idx[c])) / max(1, len(x2)), 3) for c in ctx}
    fr['none'] = round(1 - len(np.intersect1d(x2, np.unique(np.concatenate(list(idx.values()))))) / max(1, len(x2)), 3)
    out[X] = fr
    print(X, 'held' if X in info['held'] else '    ', ' '.join(f'{c} {v:.2f}' for c, v in fr.items()), flush=True)
print('chance per set', round(k2 / A.N2, 3))
if outp:
    json.dump(dict(cfg=cfg, comp=out, held=sorted(info['held'])), open(outp, 'w'), ensure_ascii=False)
