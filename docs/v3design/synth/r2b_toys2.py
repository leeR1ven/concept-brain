"""r2b check 2: the same as r2b_toys part 1, but the readout votes only over units that HAVE edges (relative
competition, as a global threshold picks the target that stands out), for SFH (count code, designer rule) and SW."""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from r2b_toys import SFHAgent, HebbAgent, CT    # noqa: E402

def rel_pred(self, F):
    d = np.zeros(self.V); n = 0
    for f in F:
        w = self.W.get(f)
        if w is None:
            continue
        s = w.sum()
        if s > 0:
            d += w / s; n += 1
    if n == 0:
        return d
    d /= n
    return d / np.sqrt(np.maximum(self.inS, 1e-9) / max(self.inS.max(), 1e-9))

class SFHrel(SFHAgent):
    pred = rel_pred
class SWrel(HebbAgent):
    pred = rel_pred

out = {}
for wname, kw in [('W1', dict()), ('realistic mix, care continues 50 %', dict(g_quick=0.15, q_reask=0.30, gu_jit=0.5, p_tut=0.5))]:
    for name, cls in [('SFH-rel (count code, designer rule)', SFHrel), ('SW-rel (handover, designer rule)', SWrel)]:
        CT.Agent = cls
        rows = [CT.run('SW', s, **kw) for s in range(5)]
        out[f'{wname} | {name}'] = {k: [round(float(np.mean([r[k][i] for r in rows])), 2) for i in range(3)] for k in rows[0]}
        print(wname, '|', name, out[f'{wname} | {name}'], flush=True)
json.dump(out, open('r2b_toys2.json', 'w'), indent=1)
