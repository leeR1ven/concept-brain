"""X7 Part C 28b: syllable pace from force-grade neurons rising in turn (no timer), HARVEST #28b / h3 #9.

Units: per muscle m a muscle neuron mu_m and 3 force-grade neurons f_m1..f_m3 (birth ramp: f_mk is on while mu_m has
been on for >= k ticks).  Practice: a 4-syllable word (X1c syllables with disjoint muscle sets), 3 ticks per syllable,
rest 6 ticks, 300 repetitions.  Learned links unit -> mu by the one rule in two forms:
  held  (ARCH 11-01): B rises at t: A on at t, t-1 or t-2 -> A->B up;  A rises at t: B on at t-1 or t-2 -> A->B down.
  onset ("only just-changed units"): B rises at t: A rose at t or t-1 -> up;  A rises at t: B rose at t-1 -> down.
Free running: syllable i is held (ASSUMED: by its inner hearing) until the next one starts; its force ramp runs; the
next syllable starts when the learned input to its muscles reaches phi x the input it got at its taught onset
(+ Gaussian noise).  Measure the onset-to-onset interval (syllable length).
"""
import json
import sys

import numpy as np

import os as _os, sys as _sys   # repo: world.py from ../x1c, voice.py from ../../world
_H = _os.path.dirname(_os.path.abspath(__file__))
for _d in (_os.path.join(_H, '..', 'x1c'), _os.path.join(_H, '..', '..', 'world'), _H):
    if _d not in _sys.path:
        _sys.path.insert(0, _d)
from world import Physics, NSYL

G = 3
L = dict(eta=0.02, w_ref=0.5, floor=0.01)


def s_conc(w):
    return np.maximum(1.0 / (1.0 + w / L['w_ref']), L['floor'])


def pick_word(ph, rng):
    while True:
        w = rng.choice(NSYL, 4, replace=False)
        sets = [ph.need[s] for s in w]
        if all(not (sets[i] & sets[j]) for i in range(4) for j in range(i + 1, 4)):
            return [int(x) for x in w]


def units(mus_age):
    """mus_age: ticks each muscle has been on (0 = off) -> unit vector [mu (24), f1 (24), f2 (24), f3 (24)]."""
    on = mus_age > 0
    return np.concatenate([on] + [mus_age >= k for k in range(1, G + 1)])


def run(seed, form, n_rep=300):
    ph = Physics()
    rng = np.random.default_rng(seed)
    word = pick_word(ph, rng)
    nm = ph.n_mus
    nu = nm * (G + 1)
    W = np.zeros((nu, nm), np.float32)
    age = np.zeros(nm, int)
    hist_u = [np.zeros(nu, bool)] * 3          # unit states at t-1, t-2, t-3
    hist_m = [np.zeros(nm, bool)] * 3
    for rep in range(n_rep):
        plan = []
        for s in word:
            plan += [s] * 3
        plan += [None] * 6
        for s in plan:
            mus = ph.need_mat[s].copy() if s is not None else np.zeros(nm, bool)
            age = np.where(mus, age + 1, 0)
            u = units(age)
            u1, u2, u3 = hist_u
            m1, m2, m3 = hist_m
            riseB = np.nonzero(mus & ~m1)[0]
            riseA = np.nonzero(u & ~u1)[0]
            if form == 'held':
                pre = np.nonzero(u | u1 | u2)[0]
                postd = np.nonzero(m1 | m2)[0]
            else:
                pre = np.nonzero((u & ~u1) | (u1 & ~u2))[0]
                postd = np.nonzero(m1 & ~m2)[0]
            if len(riseB) and len(pre):
                ix = np.ix_(pre, riseB)
                W[ix] += L['eta'] * s_conc(W[ix])
            if len(riseA) and len(postd):
                ix = np.ix_(riseA, postd)
                W[ix] = np.maximum(W[ix] - L['eta'] * s_conc(W[ix]), 0.0)
            hist_u = [u, u1, u2]
            hist_m = [mus, m1, m2]
    # learned link strength per grade, previous syllable -> next syllable's muscles
    prof = {}
    for a, b in zip(word[:-1], word[1:]):
        ma, mb = sorted(ph.need[a]), sorted(ph.need[b])
        for g in range(G + 1):
            rows = [g * nm + m for m in ma]
            prof.setdefault(g, []).append(float(W[np.ix_(rows, mb)].mean()))
    prof = {['mu', 'f1', 'f2', 'f3'][g]: float(np.mean(v)) for g, v in prof.items()}
    # input time course to the next syllable's muscles while the previous one runs its ramp
    course = []
    for a, b in zip(word[:-1], word[1:]):
        mb = sorted(ph.need[b])
        c = []
        for k in range(1, 6):                      # previous syllable on for k ticks (k = 1 is its onset tick)
            ag = np.zeros(nm, int)
            ag[sorted(ph.need[a])] = k
            c.append(float(W[units(ag)][:, mb].sum(0).mean()))
        course.append(c)
    course = np.array(course)
    ref = course[:, 2:3]                           # input after 3 ticks = the input at the taught onset
    norm = course / np.maximum(ref, 1e-9)
    # free running with a threshold phi of the taught onset input, plus noise
    lens = {}
    nrng = np.random.default_rng(seed + 5)
    for phi in (0.4, 0.6, 0.8):
        L_ = []
        for trial in range(200):
            for i in range(3):
                for k in range(1, 7):              # next onset at tick k+1 after the previous onset
                    x = norm[i, min(k, 5) - 1] + 0.15 * nrng.standard_normal()
                    if x >= phi:
                        L_.append(k)
                        break
                else:
                    L_.append(7)
        lens[phi] = dict(mean=float(np.mean(L_)), sd=float(np.std(L_)), share3=float(np.mean(np.array(L_) == 3)),
                         share_pm1=float(np.mean(np.abs(np.array(L_) - 3) <= 1)))
    return dict(seed=seed, form=form, word=word, grade_link=prof, course=norm.mean(0).tolist(), lens=lens)


if __name__ == '__main__':
    R = []
    for seed in [int(s) for s in sys.argv[1].split(',')]:
        for form in ('held', 'onset'):
            r = run(seed, form)
            R.append(r)
            print(seed, form, 'links', {k: round(v, 3) for k, v in r['grade_link'].items()},
                  'input course (k=1..5, /onset)', [round(x, 2) for x in r['course']],
                  'len', {p: (round(v['mean'], 2), round(v['sd'], 2), round(v['share3'], 2)) for p, v in r['lens'].items()},
                  flush=True)
    json.dump(R, open(sys.argv[2], 'w'))
