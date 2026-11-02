# -*- coding: utf-8 -*-
"""Bench regression driver: v21's run_v21.py protocol (frozen by default; metric code untouched) with
  * the canonical DG tie rule (gpuport/run_j_canon.py: stable index sort in the DG k-WTA; CPython 3.12 sum() in the
    turn learners' SparsePR.p), so runs are comparable across machines;
  * AgentH in place of AgentV21 (h_* keys accepted; with no h_* key this is v2.1 itself).
Same CLI as run_v21.py, e.g.  python3 run_h_canon.py tag=x seed=0 ral=1 ... h_gate=1 h_conf=1 [w_noise=1.0]"""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'v21')); sys.path.insert(1, os.path.join(HERE, '..', 'synth'))
sys.path.insert(0, HERE)
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
import run_v21  # noqa: E402
from agent_h import AgentH, typed_h  # noqa: E402
_typed = run_v21.typed
run_v21.typed = lambda k, v: typed_h(k, v, _typed)
run_v21.AgentV21 = AgentH
if __name__ == '__main__':
    run_v21.main()
