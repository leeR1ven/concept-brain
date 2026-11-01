"""Revision-2b checks of the coordinator's mapping (local, < 100 MB, a few minutes).
Part 1 (mapping point 1): age = number of a fading assembly's units still on, under the DESIGNER's rule.
  SFH: learning uses only units that JUST changed: the held event's units that fell within the last half of its age
       (the falling front, L_k in (tau/2, tau]); recall (spreading) uses every unit still on (L_k > tau): nested.
  Compared with SW + designer Hebbian (hebb_toy) and ST + delta, in W1 (30 % silence) and the realistic mix.
Part 2 (mapping point 3): release by reward-scaled Hebbian links against default blocking, no TD critic.
  Events: arrival {onset, person}; empty-room object appearance {onset, object_k}. A release at an arrival before the
  partner's 30-tick greeting is answered (r = 1); a release in the empty room is not (r = 0).
  Arms: (A) reward-only Hebbian + global decay; (B) signed: r - rbar, positive -> drive links grow, negative ->
  blocking links grow (the 'dip' as a blocking connection); both concave. Measures per tenth of life: first-greeting
  rate at arrivals, empty-room releases.
Run: python3 -I r2b_toys.py
"""
import os, sys, json, math
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import itertools
import hebb_toy                                    # noqa: E402
from hebb_toy import HebbAgent, CT                 # noqa: E402


class SFHAgent(HebbAgent):
    def _f(self, now, recall):
        if self.L is None:
            return []
        src, g, toff = self.L; g = sorted(g)
        og = sorted(self.O[1]) if self.O is not None else []
        osrc = self.O[0] if self.O is not None else None
        tau = max(1, now - toff)
        if recall:
            K = [k for k in range(9) if 2 ** (k + 1) > tau] or [8]
        else:
            K = [k for k in range(9) if tau / 2 < 2 ** (k + 1) <= tau] or [0]
        F = [('Lt', src, a, k) for a in g for k in K]
        F += [('LLt', src, a, b, k) for a, b in itertools.combinations(g, 2) for k in K]
        F += [('LOt', src, a, k, osrc, o) for a in g for k in K for o in og]
        return F

    def feats(self, now):
        return self._f(now, recall=True)

    def hear(self, items, now, dur, learn=True):
        if learn:
            self.learn(self._f(now, recall=False), items)
        self.O, self.L = self.L, ('o', list(items), now + dur)
        self.last_other = list(items)


def part1():
    out = {}
    for wname, kw in [('W1 as specified', dict()),
                      ('realistic mix, care continues 50 %', dict(g_quick=0.15, q_reask=0.30, gu_jit=0.5, p_tut=0.5))]:
        CT.Agent = SFHAgent
        rows = [CT.run('SW', s, **kw) for s in range(5)]
        out[wname] = {k: [round(float(np.mean([r[k][i] for r in rows])), 2) for i in range(3)] for k in rows[0]}
        print('part1', wname, '| SFH (count code + designer rule):', out[wname])
    return out


def part2(arm, seed, tenths=10, n_arr=100, n_obj=160, eta=0.3, gamma=0.999, noise=0.25, block=0.6):
    rng = np.random.default_rng(seed)
    W = {}; B = {}; rbar = 0.5
    res = []
    for t in range(tenths):
        greet_first = empty_rel = 0
        evs = ['arr'] * n_arr + ['obj'] * n_obj
        rng.shuffle(evs)
        for ev in evs:
            feats = ['onset', 'person'] if ev == 'arr' else ['onset', f'obj{int(rng.integers(5))}']
            drive = sum(W.get(f, 0.0) for f in feats) - sum(B.get(f, 0.0) for f in feats)
            release = drive + noise * rng.standard_normal() > block
            for f in list(W):                         # one global decay of plastic edges
                W[f] *= gamma
            for f in list(B):
                B[f] *= gamma
            if not release:
                continue
            r = 1.0 if ev == 'arr' else 0.0
            if ev == 'arr':
                greet_first += 1
            else:
                empty_rel += 1
            if arm == 'A':                            # reward-only Hebbian
                for f in feats:
                    W[f] = W.get(f, 0.0) + eta * r / (1.0 + W.get(f, 0.0))
            else:                                     # signed: r - rbar
                d = r - rbar
                rbar += 0.05 * (r - rbar)
                for f in feats:
                    if d > 0:
                        W[f] = W.get(f, 0.0) + eta * d / (1.0 + W.get(f, 0.0))
                    else:
                        B[f] = B.get(f, 0.0) + eta * (-d) / (1.0 + B.get(f, 0.0))
        res.append((round(greet_first / n_arr, 2), empty_rel))
    return res


if __name__ == '__main__':
    p1 = part1()
    p2 = {}
    for arm in ('A', 'B'):
        rows = [part2(arm, s) for s in range(5)]
        p2[arm] = [(round(float(np.mean([r[t][0] for r in rows])), 2), round(float(np.mean([r[t][1] for r in rows])), 1))
                   for t in range(10)]
        print('part2 arm', arm, '(per tenth: first-greeting rate at arrivals, empty-room releases)', p2[arm])
    json.dump({'part1': p1, 'part2': p2}, open('r2b_toys.json', 'w'), indent=1)
