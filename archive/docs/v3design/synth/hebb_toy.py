"""Revision-2 check (local, < 100 MB, ~1-2 min): the DESIGNER's cortical rule vs the delta rule, inside the anchored
(latest x age, SW handover) context of age_toy.

Designer's rule (DESIGNER_SPEC #14, #15, #17, #19; char-neuron-network; chain-neuron plastic2):
  - Hebbian, event-gated: only at a heard onset; context features (earlier, still-held) -> the onset items (later);
  - concave, uncapped growth: dw = eta / (1 + w / w_ref)^pow   (no weakening by absence);
  - one global decay of plastic edges: w *= gamma every event; edges < 1e-3 deleted;
  - two-layer weights: propagation = strength / out-strength of the source, / sqrt(in-strength of the target).
Delta rule (r1 / bench R2): p = clip(sum w), dw = eta (t - p)/|F|.
Measures: (1) untutored stability under W1 and under the realistic caregiver mix with care continuing (develop's toy);
          (2) correction cost: after 600 formats, 4 answers change and are re-taught n times (formats only);
              share of changed items answered with the NEW answer, and of unchanged items still right.
Run: python3 -I hebb_toy.py
"""
import os, sys, json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import age_toy                                     # noqa: E402  (also injects AgeAgent into caregiver_toy)
from age_toy import CT, AgeAgent, drift_toy        # noqa: E402

ETA_H, WREF, POW, GAMMA = 0.1, 1.0, 1.0, 0.9995


class HebbAgent(AgeAgent):
    def __init__(self, toks, variant):
        super().__init__(toks, 'SW')
        self.inS = np.zeros(self.V)

    def pred(self, F):
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
        d /= len(F)
        return d / np.sqrt(np.maximum(self.inS, 1e-9) / max(self.inS.max(), 1e-9))

    def learn(self, F, items):
        for f in list(self.W):                      # one global decay of plastic edges
            self.W[f] *= GAMMA
        self.inS *= GAMMA
        if not F:
            return
        for f in F:
            w = self.W.get(f)
            if w is None:
                w = self.W[f] = np.zeros(self.V)
            for it in items:
                j = self.ix[it]
                dw = ETA_H / (1.0 + w[j] / WREF) ** POW
                w[j] += dw; self.inS[j] += dw


def stability():
    out = {}
    for wname, kw in [('W1 as specified', dict()),
                      ('realistic mix, care continues 50 %', dict(g_quick=0.15, q_reask=0.30, gu_jit=0.5, p_tut=0.5))]:
        out[wname] = {}
        for name, cls in [('delta (SW)', AgeAgent), ('designer Hebbian (SW)', HebbAgent)]:
            CT.Agent = cls
            rows = [CT.run('SW', s, **kw) for s in range(5)]
            out[wname][name] = {k: [round(float(np.mean([r[k][i] for r in rows])), 2) for i in range(3)] for k in rows[0]}
            print(wname, '|', name, out[wname][name])
    CT.Agent = AgeAgent
    return out


def correction(cls, seed, n_list=(1, 2, 4, 8, 16)):
    rng = np.random.default_rng(seed)
    NF, NX, ANS = drift_toy.NF, drift_toy.NX, drift_toy.ANS
    toks = [f'f{i}' for i in range(NF)] + [f'x{i}' for i in range(NX)] + ['ok']
    at = [[f'a{f}_{k}' for k in range(ANS[f])] for f in range(NF)]
    toks += [t for r in at for t in r]
    table = {(f, x): at[f][rng.integers(ANS[f])] for f in range(NF) for x in range(NX)}
    ag = cls(toks, 'SW'); now = 0

    def fmt(f, x):
        nonlocal now
        now += 15; ag.hear([f'f{f}', f'x{x}'], now, 6); qend = now + 6
        ag.hear([table[(f, x)]], qend + 10, 3); now = qend + 13

    def answer(f, x):
        st = (ag.L, ag.O, ag.last_other)
        ag.hear([f'f{f}', f'x{x}'], 10 ** 7, 6, learn=False); tau, tok = ag.decide(10 ** 7 + 6)
        ag.L, ag.O, ag.last_other = st
        return tok

    for _ in range(600):
        fmt(int(rng.integers(NF)), int(rng.integers(NX)))
    keys = list(table); idx = rng.choice(len(keys), 4, replace=False); changed = [keys[i] for i in idx]
    for (f, x) in changed:                          # the world changes 4 answers
        alts = [a for a in at[f] if a != table[(f, x)]]
        table[(f, x)] = alts[int(rng.integers(len(alts)))]
    res, done = {}, 0
    for n in n_list:
        while done < n:
            for (f, x) in changed:
                fmt(f, x)
            for _ in range(8):                      # ordinary life goes on in between
                f, x = keys[int(rng.integers(len(keys)))]
                if (f, x) not in changed:
                    fmt(f, x)
            done += 1
        new_ok = np.mean([answer(f, x) == table[(f, x)] for (f, x) in changed])
        old_ok = np.mean([answer(f, x) == table[(f, x)] for (f, x) in keys if (f, x) not in changed])
        res[n] = (round(float(new_ok), 2), round(float(old_ok), 2))
    return res


if __name__ == '__main__':
    st = stability()
    cor = {}
    for name, cls in [('delta (SW)', AgeAgent), ('designer Hebbian (SW)', HebbAgent)]:
        rows = [correction(cls, s) for s in range(5)]
        cor[name] = {n: (round(float(np.mean([r[n][0] for r in rows])), 2), round(float(np.mean([r[n][1] for r in rows])), 2))
                     for n in rows[0]}
        print('correction |', name, '(re-teachings: changed-right, unchanged-right)', cor[name])
    json.dump({'stability': st, 'correction': {k: {str(n): v for n, v in d.items()} for k, d in cor.items()}},
              open('hebb_toy.json', 'w'), indent=1)
