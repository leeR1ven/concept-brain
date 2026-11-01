"""Revision-2 check (local, < 100 MB, ~1 min): can the 'age' of the latest event come WITHOUT dedicated time neurons?

Three age codes for the anchored context (features always carry the latest event's items, as in drift_toy's ST):
  ST  soft log bins [2^k, 2^(k+2))  (r1's code; equal to 'fading units + inhibition from a faster-fading unit')
  SF  fading assembly, NO competition: unit k of the held event stays on until age 2^(k+1)  -> nested sets;
      the number of units still on encodes age (designer spec 2: strength = number of active neurons)
  SW  fading + k-WTA handover (M1 only): unit k has drive d_k = 0.7^k and fatigue rate 2^-k; the top-2 units of
      d_k * exp(-tau * 2^-k) win.  Fast units win first, slower ones take over at log-spaced ages: emergent windows.
World and learner: the develop reviewer's caregiver_toy.run (read-only import) with drift_toy's event-level delta rule
(learning only at heard onsets = the designer's 'just turned on' rule).  5 seeds; 600 formats, then 3,000 untutored.
Run: python3 -I age_toy.py
"""
import os, sys, json, itertools, math
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'review', 'develop'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import drift_toy                                   # noqa: E402
import caregiver_toy as CT                         # noqa: E402  (read-only: only its run() is used)

K = 9


def sw_winners(tau):
    act = [(0.7 ** k) * math.exp(-tau * 2.0 ** -k) for k in range(K)]
    return sorted(range(K), key=lambda k: -act[k])[:2]


def sf_units(tau):
    return [k for k in range(K) if tau < 2 ** (k + 1)] or [K - 1]


class AgeAgent(drift_toy.Agent):
    def feats(self, now):
        if self.v not in ('SF', 'SW'):
            return super().feats(now)
        if self.L is None:
            return []
        src, g, toff = self.L; g = sorted(g)
        og = sorted(self.O[1]) if self.O is not None else []
        osrc = self.O[0] if self.O is not None else None
        tau = max(1, now - toff)
        B = sf_units(tau) if self.v == 'SF' else sw_winners(tau)
        F = [('Lt', src, a, k) for a in g for k in B]
        F += [('LLt', src, a, b, k) for a, b in itertools.combinations(g, 2) for k in B]
        F += [('LOt', src, a, k, osrc, o) for a in g for k in B for o in og]
        return F


CT.Agent = AgeAgent                                # inject the new age codes into the reviewer's world


def main():
    print('SW windows (age -> winning units):', {t: sw_winners(t) for t in (1, 3, 5, 10, 16, 30, 45, 90, 180)})
    print('SF units still on:', {t: sf_units(t) for t in (1, 5, 10, 45, 90)})
    worlds = [('W1 as specified', dict()),
              ('moves on 10 %', dict(g_quick=0.10)),
              ('realistic mix (move-on 15 %, re-ask 30 %, jitter 50 %), care continues 50 %',
               dict(g_quick=0.15, q_reask=0.30, gu_jit=0.5, p_tut=0.5))]
    res = {}
    for wname, kw in worlds:
        res[wname] = {}
        print('==', wname)
        for v in ('ST', 'SF', 'SW'):
            rows = [CT.run(v, s, **kw) for s in range(5)]
            keys = list(rows[0].keys())
            agg = {k: [round(float(np.mean([r[k][i] for r in rows])), 2) for i in range(3)] for k in keys}
            res[wname][v] = agg
            print(f'  {v}:', '  '.join(f'{k}: acc {a[0]:.2f} echo {a[1]:.2f} silent {a[2]:.2f}' for k, a in agg.items()))
    json.dump(res, open('age_toy.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
