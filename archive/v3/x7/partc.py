"""X7 Part C 28a: whole-syllable competition in the motor area (HARVEST #28a / h3 #8).

Practice: the child repeats random syllables through the hand-wired (generation-0) sound->muscle map with motor noise
and occasional cross-talk; muscle->muscle links learn by the one rule (co-rise strengthens both ways; A rising after B
weakens A->B; falls change nothing), inhibitory partners by X1c's partner form.  Probe: garble per utterance with and
without these learned links in the competition (one settling pass).
"""
import json
import sys

import numpy as np

import os as _os, sys as _sys   # repo: world.py from ../x1c, voice.py from ../../world
_H = _os.path.dirname(_os.path.abspath(__file__))
for _d in (_os.path.join(_H, '..', 'x1c'), _os.path.join(_H, '..', '..', 'world'), _H):
    if _d not in _sys.path:
        _sys.path.insert(0, _d)
from babble import Motor, Rule, standin, classify, M, L
from world import Physics, NSYL

ETA_V, ETA_VCO = 0.004, 0.01


def s_conc(w):
    return np.maximum(1.0 / (1.0 + w / L['w_ref']), L['floor'])


class Partners:
    def __init__(self, n):
        self.V = np.zeros((n, n), np.float32)          # V[a, b]: inhibition from a onto b
        self.prev = np.zeros(n, bool)

    def step(self, mus):
        R = np.nonzero(mus & ~self.prev)[0]
        if len(R):
            off = np.nonzero(~mus)[0]
            ix = np.ix_(R, off)
            self.V[ix] += ETA_V * s_conc(self.V[ix])
            on = np.nonzero(mus)[0]
            ix2 = np.ix_(on, R)
            self.V[ix2] = np.maximum(self.V[ix2] - ETA_VCO, 0.0)
            np.fill_diagonal(self.V, 0.0)
        self.prev = mus.copy()

    def decay(self):
        self.V *= L['dec']
        self.V[self.V < L['prune']] = 0.0


def drive_for(ph, S, s, rng, sigma, cross):
    a = ph.content[s].astype(np.float32)
    d = M['g_ret'] * (a @ S[:ph.n_c])
    if cross is not None:
        d = d + 0.7 * (ph.content[cross].astype(np.float32) @ S[:ph.n_c])
    if sigma > 0:
        d = d + sigma * rng.standard_normal(len(d)).astype(np.float32)
    return np.maximum(d, 0.0)


def practice(ph, S, seed, n_ep=2000):
    rng = np.random.default_rng(seed)
    mot = Motor(ph, seed)
    E = Rule(ph.n_mus, ph.n_mus, L['eta'], L['eta'], mask_diag=True)
    P = Partners(ph.n_mus)
    t = 0
    stats = dict(correct=0, garble=0, wrong=0, silent=0)
    for ep in range(n_ep):
        s = int(rng.integers(NSYL))
        cross = int(rng.integers(NSYL)) if rng.random() < 0.3 else None
        cls = []
        for k in range(3 + int(rng.integers(3, 9))):
            if k < 3:
                d = drive_for(ph, S, s, rng, 0.3, cross)
            else:
                d = np.zeros(ph.n_mus, np.float32)
            mus = mot.step(d, None, brake_on=False)
            if k < 3:
                cls.append(classify(ph, ph.decode(mus), s))
            E.step(mus, mus)
            P.step(mus)
            t += 1
            if t % L['dec_every'] == 0:
                E.decay()
                P.decay()
        c = max(set(cls), key=cls.count)
        stats['correct' if c in ('clear', 'mumble') else c] += 1
    return E.W, P.V, stats


def lateral_fn(E, V, g, use_e=True, use_v=True):
    En = E / max(E.sum(1).max(), 1e-9)
    Vn = V / max(V.sum(1).max(), 1e-9)

    def f(mus):
        m = mus.astype(np.float32)
        x = np.zeros(len(m), np.float32)
        if use_e:
            x += g * (m @ En)
        if use_v:
            x -= g * (m @ Vn)
        return x
    return f


def probe(ph, S, seed, cond, lateral):
    mot = Motor(ph, seed + 500)
    rng = np.random.default_rng(seed + 700)
    out = dict(correct=0, garble=0, wrong=0, silent=0)
    gt = 0
    n = 0
    for s in range(NSYL):
        for rep in range(3):
            mot.reset()
            cross = None
            sigma = 0.0
            if cond == 'noise':
                sigma = 0.3
            elif cond == 'cross':
                sigma = 0.15
                cross = int(rng.integers(NSYL - 1))
                cross += cross >= s
            cls = []
            for k in range(3):
                d = drive_for(ph, S, s, rng, sigma, cross)
                mus = mot.step(d, None, brake_on=False, lateral=lateral)
                cls.append(classify(ph, ph.decode(mus), s))
            gt += sum(c == 'garble' for c in cls)
            ns = [c for c in cls if c != 'silent']
            n += 1
            if not ns:
                out['silent'] += 1
                continue
            vals, cnt = np.unique(ns, return_counts=True)
            best = vals[cnt == cnt.max()]
            r = best[0] if len(best) == 1 else ns[0]
            out['correct' if r in ('clear', 'mumble') else r] += 1
    res = {k: v / n for k, v in out.items()}
    res['garble_ticks'] = gt / (3 * n)
    return res


def run(seed):
    ph = Physics()
    S = standin(ph)
    E, V, st = practice(ph, S, seed)
    need = ph.need_mat.astype(np.float32)
    same = (need.T @ need) > 0                      # muscle pairs that share a syllable
    np.fill_diagonal(same, False)
    off = ~same
    np.fill_diagonal(off, False)
    res = dict(seed=seed, practice=st, E_same=float(E[same].mean()), E_other=float(E[off].mean()),
               V_same=float(V[same].mean()), V_other=float(V[off].mean()), probes={})
    arms = {'none': None}
    for g in (0.15, 0.3, 0.6):
        arms[f'EV_{g}'] = lateral_fn(E, V, g)
    arms['E_0.3'] = lateral_fn(E, V, 0.3, use_v=False)
    arms['V_0.3'] = lateral_fn(E, V, 0.3, use_e=False)
    for name, lat in arms.items():
        res['probes'][name] = {c: probe(ph, S, seed, c, lat) for c in ('clean', 'noise', 'cross')}
    return res


if __name__ == '__main__':
    seeds = [int(s) for s in sys.argv[1].split(',')]
    R = []
    for sd in seeds:
        r = run(sd)
        R.append(r)
        print(sd, r['practice'], 'E same/other %.3f/%.3f V same/other %.3f/%.3f' % (r['E_same'], r['E_other'], r['V_same'], r['V_other']))
        for name, pr in r['probes'].items():
            print('  ', name, {c: (round(v['correct'], 2), round(v['garble'], 2)) for c, v in pr.items()}, flush=True)
    json.dump(R, open(sys.argv[2], 'w'))
