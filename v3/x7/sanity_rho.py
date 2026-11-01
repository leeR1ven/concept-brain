"""Sanity check: on smooth natural-image-like level maps the harness should reproduce local3d's high rho."""
import os, sys
import numpy as np
import sens
rng = np.random.default_rng(5)
def smooth(n):
    g = rng.random((n, sens.F + 4, sens.TW + 4))
    k = np.ones((5, 5)) / 25
    out = np.zeros((n, sens.F, sens.TW))
    for i in range(5):
        for j in range(5):
            out += g[:, i:i + sens.F, j:j + sens.TW] / 25
    out = (out - out.min()) / (out.max() - out.min())
    return np.clip((out * 10).astype(int), 0, 9)
X = smooth(150)
for tg in [(0.4, 0.3, 0.2, 0.1), (0.3, 0.15, 0.08, 0.04)]:
    sens.TARGET = tg
    for arm in ('dom1', 'eq4', 'eq7'):
        layers = sens.build(arm, 0)
        th = sens.calibrate(sens.encode(X[:50]), layers)
        hs = sens.forward(sens.encode(X[50:]), layers, th)
        Y = X[50:]
        iu = np.triu_indices(len(Y), 1)
        din = np.abs(Y[:, None] - Y[None]).sum((2, 3))[iu].astype(float)
        rr = []
        for h in hs:
            hf = h.astype(np.float32); c = hf.sum(1); ham = (c[:, None] + c[None] - 2 * hf @ hf.T)[iu]
            rr.append(round(sens.spearman(din, ham.astype(float)), 3))
        print(tg, arm, 'rho per layer', rr, 'counts', [int(h.sum(1).mean()) for h in hs])
