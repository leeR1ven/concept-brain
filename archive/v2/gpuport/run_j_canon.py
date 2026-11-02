# -*- coding: utf-8 -*-
"""The numpy reference's staged J protocol (ref/run_j.py, unchanged) with ONE change: the DG k-WTA breaks exact ties
by unit index (stable sort) instead of np.argpartition (platform-dependent; see report).  Same CLI as run_j.py."""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'synth'))
import rules  # noqa: E402


def dg(self, inputs):
    d = self.tie.copy()
    for g, x in inputs.items():
        if g in self.idx and x is not None and x.any():
            d += self.wg[g] * x[self.idx[g]].sum(1)
    code = np.zeros(self.n_dg, bool)
    code[np.argsort(-d, kind='stable')[:self.k_dg]] = True
    return code


rules.Hippo.dg = dg


def _sum312(xs):
    """CPython 3.12's sum() over floats (Neumaier); identical to sum() on 3.12, replicates it on 3.11."""
    xs = list(xs)
    if not xs:
        return 0
    f = 0 + xs[0]; c = 0.0
    for x in xs[1:]:
        t = f + x
        if abs(f) >= abs(x):
            c += (f - t) + x
        else:
            c += (x - t) + f
        f = t
    if c and np.isfinite(c):
        f += c
    return f


import agent_think  # noqa: E402


def _p(self, keys, bias=None):
    b = self.init if bias is None else bias
    return float(np.clip(b + _sum312(self.w.get(k, 0.0) for k in keys), 0.0, 1.0))


agent_think.SparsePR.p = _p
import run_j  # noqa: E402
run_j.main()
