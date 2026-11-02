"""X7 Part B (HARVEST #26): can babbling teach "heard sound -> the same muscles" under the designer's rule?

Body/world: X1c `world.py` Physics + render (copied unchanged).  Motor area: X1c `core3._motor` (tonic blocking neurons
win by default, divisive gain, release by sharpness relative to the max), copied; the birth-given stand-in Mmap is
removed and replaced by a learned matrix W (auditory feature on-units -> muscles).

Rule (ARCH sec. 2.8 with the 11-01 held-neuron form), binary units, every tick t:
  * B (muscle) rises at t: every A on at t, t-1 or t-2 (just rose with it / held / just fell) -> A->B += eta * s(w)
  * A rises at t: every B on at t-1 or t-2 (B came before A, held or just fell)           -> A->B -= eta_m * s(w)
  * a fall changes nothing.  s(w) = max(1/(1+w/w_ref), 0.01); w >= 0; one global decay x0.998 / 100 ticks, prune.
Arms: base (1-tick own-voice delay), same (same-tick own hearing), redup (bababa), imit (imitation game).
"""
import json
import sys
import time

import numpy as np

import os as _os, sys as _sys   # repo: world.py from ../x1c, voice.py from ../../world
_H = _os.path.dirname(_os.path.abspath(__file__))
for _d in (_os.path.join(_H, '..', 'x1c'), _os.path.join(_H, '..', '..', 'world'), _H):
    if _d not in _sys.path:
        _sys.path.insert(0, _d)
from world import Physics, render, NSYL, OWN, C1

M = dict(gamma_m=0.1, theta_m=0.22, c_bm=0.5, rho=0.6, tonic=1.0, c_mb=1.2, g_brake=3.0, near=0.6, noise_b=0.15,
         c_sm=1.0, g_dir=0.5, g_ret=1.0, n_b=8)
L = dict(eta=0.02, w_ref=0.5, floor=0.01, dec=0.998, dec_every=100, prune=1e-3, win=2)


def nrm(W, a=0.5):
    """two-layer weights (core3._nrm): propagation = W / (out^a in^(1-a)), total mass kept."""
    if W.sum() <= 0:
        return W.copy()
    rs, cs = W.sum(1), W.sum(0)
    rs_f = np.maximum(rs, 0.3 * np.median(rs[rs > 0]))
    cs_f = np.maximum(cs, 0.3 * np.median(cs[cs > 0]))
    P = W / (rs_f[:, None] ** a * cs_f[None, :] ** (1 - a))
    return P * (W.sum() / max(P.sum(), 1e-9))


class Motor:
    """X1c motor area without the PFC: d = drive; blocking (tonic) neurons and phasic stop neurons; release by sharpness."""

    def __init__(self, ph, seed):
        r = np.random.default_rng(seed + 31)
        self.ph = ph
        self.nm = ph.n_mus
        self.th_b = M['tonic'] + r.uniform(-0.45, 0.45, M['n_b'])
        self.th_s = 1.0 + r.uniform(-0.2, 0.2, M['n_b'])
        self.rng = np.random.default_rng(seed + 32)
        self.voice_lines = [np.arange(ph.n_c + v * ph.n_vu, ph.n_c + (v + 1) * ph.n_vu) for v in range(1, 5)]
        self.reset()

    def reset(self):
        self.mus = np.zeros(self.nm, bool)

    def step(self, d, heard=None, brake_on=True, lateral=None):
        dn = d / (1.0 + M['gamma_m'] * d.sum())
        brake = 0.0
        if brake_on and heard is not None:
            for vl in self.voice_lines:
                if heard[vl].mean() >= M['near']:
                    brake = M['g_brake']
        noise = M['noise_b'] * self.rng.standard_normal(M['n_b'])
        blocks = M['tonic'] - M['c_mb'] * self.mus.sum() / 4.0 + noise > self.th_b
        stops = brake + noise > self.th_s
        thr = M['theta_m'] + M['c_bm'] * blocks.mean() + M['c_sm'] * stops.mean()
        mus = (dn > thr) & (dn >= M['rho'] * dn.max()) if dn.max() > 0 else np.zeros(self.nm, bool)
        if lateral is not None and mus.any():                  # Part C: one settling pass with learned M<->M links
            d2 = dn + lateral(mus)
            mus = (d2 > thr) & (d2 >= M['rho'] * d2.max()) if d2.max() > 0 else np.zeros(self.nm, bool)
        self.mus = mus
        return mus


class Rule:
    """the one rule on W[A -> B], with 2-tick histories of A and B."""

    def __init__(self, nA, nB, eta, eta_m, mask_diag=False):
        self.W = np.zeros((nA, nB), np.float32)
        self.eta, self.eta_m = eta, eta_m
        self.A1 = np.zeros(nA, bool)
        self.A2 = np.zeros(nA, bool)
        self.B1 = np.zeros(nB, bool)
        self.B2 = np.zeros(nB, bool)
        self.diag = mask_diag
        self.n_plus = 0.0
        self.n_minus = 0.0

    def s(self, w):
        return np.maximum(1.0 / (1.0 + w / L['w_ref']), L['floor'])

    def step(self, A, B):
        W = self.W
        riseB = np.nonzero(B & ~self.B1)[0]
        riseA = np.nonzero(A & ~self.A1)[0]
        if len(riseB):
            pre = np.nonzero(A | self.A1 | self.A2)[0]
            if len(pre):
                ix = np.ix_(pre, riseB)
                d = self.eta * self.s(W[ix])
                if self.diag:
                    d = d * (pre[:, None] != riseB[None, :])
                W[ix] += d
                self.n_plus += d.sum()
        if len(riseA):
            post = np.nonzero(self.B1 | self.B2)[0]
            if len(post):
                ix = np.ix_(riseA, post)
                d = self.eta_m * self.s(W[ix])
                if self.diag:
                    d = d * (riseA[:, None] != post[None, :])
                W[ix] = np.maximum(W[ix] - d, 0.0)
                self.n_minus += d.sum()
        self.A2, self.A1 = self.A1, A.copy()
        self.B2, self.B1 = self.B1, B.copy()

    def decay(self):
        self.W *= L['dec']
        self.W[self.W < L['prune']] = 0.0


class LawfulPhysics(Physics):
    """sensitivity (HARVEST #26 asked for 'lawful physics'): a garbled gesture sounds like a blend of the syllables it
    partly forms: every syllable with >= 2 of its 4 muscles active contributes a fixed share hit/4 of its content units.
    Clear and mumbled syllables are exactly as in X1c."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        r = np.random.default_rng(777)
        self.order = [r.permutation(np.nonzero(self.content[s])[0]) for s in range(NSYL)]

    def garble_content(self, key):
        mus = np.zeros(self.n_mus, bool)
        mus[list(key)] = True
        hit = self.need_mat[:, mus].sum(1)
        v = np.zeros(self.n_c, bool)
        for s in np.nonzero(hit >= 2)[0]:
            v[self.order[s][:int(round(self.k_syl * hit[s] / 4))]] = True
        return v


def standin(ph):
    S = np.zeros((ph.n_a, ph.n_mus), np.float32)
    S[:ph.n_c] = ph.content.T.astype(np.float32) @ ph.need_mat.astype(np.float32) / ph.k_syl
    return S


def classify(ph, mouth, s):
    if mouth is None:
        return 'silent'
    if mouth[0] == 'g':
        return 'garble'
    if mouth[1] == s:
        return 'clear' if mouth[0] == 's' else 'mumble'
    return 'wrong'


def probe(ph, P, seed, mode, lateral=None):
    """every syllable: content units on 3 ticks; read the muscles.  Returns shares."""
    mot = Motor(ph, seed + 500)
    rng = np.random.default_rng(seed + 600)
    out = dict(clear=0, mumble=0, wrong=0, garble=0, silent=0)
    garble_ticks = 0
    for s in range(NSYL):
        mot.reset()
        cls = []
        for k in range(3):
            if mode == 'inner':
                a = np.zeros(ph.n_a, bool)
                a[:ph.n_c] = ph.content[s]
                d = M['g_ret'] * (a.astype(np.float32) @ P)
            else:
                a = render(ph, [('s', s, C1, 1.0)], rng)
                d = M['g_dir'] * (a.astype(np.float32) @ P)
            mus = mot.step(d, a, brake_on=False, lateral=lateral)
            cls.append(classify(ph, ph.decode(mus), s))
        garble_ticks += sum(c == 'garble' for c in cls)
        ns = [c for c in cls if c != 'silent']
        if not ns:
            out['silent'] += 1
            continue
        vals, cnt = np.unique(ns, return_counts=True)
        best = vals[cnt == cnt.max()]
        res = best[0] if len(best) == 1 else ns[0]
        out[res] += 1
    r = {k: v / NSYL for k, v in out.items()}
    r['correct'] = r['clear'] + r['mumble']
    r['garble_ticks'] = garble_ticks / (3 * NSYL)
    return r


def diag(ph, W):
    true = (ph.content.T.astype(np.float32) @ ph.need_mat.astype(np.float32)) > 0     # (150, 24)
    Wc = W[:ph.n_c]
    D = ph.content.astype(np.float32) @ Wc                                           # syllable -> muscle drive
    top = np.argsort(-D, 1)[:, :4]
    hit = np.array([ph.need_mat[s, top[s]].sum() for s in range(NSYL)])
    return dict(w_true=float(Wc[true].mean()), w_other=float(Wc[~true].mean()),
                w_voice=float(W[ph.n_c:].mean()), w_sum=float(W.sum()),
                top4_exact=float((hit == 4).mean()), top4_hits=float(hit.mean()))


def readout(W, mass_c, n_c, how='content'):
    """propagation matrix for the probe: two-layer weights, then a fixed overall gain set so that the content rows
    carry the stand-in's mass ('content', amended main) or the whole matrix does ('total', as first planned)."""
    P = nrm(W)
    ref = P[:n_c].sum() if how == 'content' else P.sum()
    return P * (mass_c / max(ref, 1e-9))


def probes(ph, W, seed, mass_c):
    out = {}
    for how in ('content', 'total'):
        P = readout(W, mass_c, ph.n_c, how)
        out[how] = dict(inner=probe(ph, P, seed, 'inner'), hear=probe(ph, P, seed, 'hear'))
    P = readout(W, mass_c, ph.n_c, 'content')
    best = dict(inner=(0, None), hear=(0, None))
    for g in (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0):
        for mode in ('inner', 'hear'):
            r = probe(ph, g * P, seed, mode)
            if r['correct'] > best[mode][0] or best[mode][1] is None:
                best[mode] = (r['correct'], g)
    out['oracle_gain'] = best
    return out


def life(arm, seed, n_babble=2000, ratio=1.0, checkpoints=(500, 1000, 2000), closed=False, p_imit=0.5, part_c=False,
         phys='x1c'):
    ph = Physics() if phys == 'x1c' else LawfulPhysics()
    rng = np.random.default_rng(seed)
    mot = Motor(ph, seed)
    rule = Rule(ph.n_a, ph.n_mus, L['eta'], ratio * L['eta'])
    mm = Rule(ph.n_mus, ph.n_mus, L['eta'], ratio * L['eta'], mask_diag=True) if part_c else None
    co = np.zeros((ph.n_mus, ph.n_mus)) if part_c else None
    S = standin(ph)
    mass = S[:ph.n_c].sum()
    heard = np.zeros(ph.n_a, bool)          # what the ear delivers this tick
    pending = np.zeros(ph.n_a, bool)        # sound produced this tick, heard next tick (1-tick delay)
    t = 0
    res = dict(arm=arm, seed=seed, ratio=ratio, closed=closed, cps={})
    stats = dict(babbles=0, clear=0, mumble=0, garble=0, imitations=0)
    queue = []                              # per-tick plan: (babble muscle set or None, caregiver syllable or None)

    def plan_babble():
        k = int(rng.integers(3, 6))
        ms = np.zeros(ph.n_mus, bool)
        ms[rng.choice(ph.n_mus, k, replace=False)] = True
        reps = 3 if arm == 'redup' else 1
        q = []
        for r in range(reps):
            q += [(ms, None, 'b')] * 3
            if r < reps - 1:
                q += [(None, None, 'gap')]
        rest = int(rng.integers(3, 9))
        q += [(None, None, 'rest')] * rest
        return ms, q

    cur_ms, queue = plan_babble()
    bab_decodes = []
    P = None
    while stats['babbles'] < n_babble:
        if not queue:
            stats['babbles'] += 1
            if stats['babbles'] in checkpoints:
                pr = probes(ph, rule.W, seed, mass)
                cp = dict(inner=pr['content']['inner'], hear=pr['content']['hear'], total=pr['total'],
                          oracle=pr['oracle_gain'], **diag(ph, rule.W),
                          n_plus=rule.n_plus, n_minus=rule.n_minus, stats=dict(stats))
                if part_c:
                    cp['mm'] = mm.W.copy().tolist()
                res['cps'][stats['babbles']] = cp
                if stats['babbles'] >= n_babble:
                    break
            cur_ms, queue = plan_babble()
            bab_decodes = []
        ms, cg, tag = queue.pop(0)
        drive = np.zeros(ph.n_mus, np.float32)
        if ms is not None:
            drive[ms] = 2.0                                    # random babbling drive (must beat the blocking)
        if closed and P is not None:
            drive = drive + M['g_dir'] * (heard.astype(np.float32) @ P)
        mus = mot.step(drive, heard, brake_on=True)
        mouth = ph.decode(mus)
        if tag == 'b':
            bab_decodes.append(mouth)
        srcs = []
        if mouth is not None:
            srcs.append((mouth[0], mouth[1], OWN, 1.0))
        if cg is not None:
            srcs.append(('s', cg, C1, 1.0))
        snd = render(ph, srcs, rng)
        if arm == 'same':
            heard_now = snd                                    # own voice heard in the same tick as the muscles
        else:
            heard_now = pending                                # heard one tick after it is produced
            pending = snd
            if cg is not None:                                 # the caregiver's voice is not delayed by our body
                pass
        rule.step(heard_now, mus)
        if part_c:
            mm.step(mus, mus)
            co += np.outer(mus, mus)
        heard = heard_now
        t += 1
        if t % L['dec_every'] == 0:
            rule.decay()
            if part_c:
                mm.decay()
            if closed:
                Pn = nrm(rule.W)
                P = Pn * (mass / max(Pn.sum(), 1e-9))
        # end of a babble's sound: count it; imitation game
        if tag == 'b' and (not queue or queue[0][2] != 'b'):
            kinds = [m for m in bab_decodes if m is not None]
            if kinds:
                kd = kinds[len(kinds) // 2]
                stats[{'s': 'clear', 'm': 'mumble', 'g': 'garble'}[kd[0]]] += 1
                if arm == 'imit' and kd[0] in ('s', 'm') and rng.random() < p_imit and queue and queue[0][2] == 'rest':
                    stats['imitations'] += 1
                    lat = int(rng.integers(1, 3))
                    q = [(None, None, 'rest')] * 2                      # she waits 2 ticks
                    q += [(None, kd[1], 'cg')] * 3                       # she says the syllable clearly
                    q += [(None, None, 'rest')] * (lat - 1)              # the child repeats 1-2 ticks after she stops
                    q += [(cur_ms, None, 'rep')] * 3                     # the child repeats the same gesture
                    q += [(None, None, 'rest')] * int(rng.integers(3, 9))
                    queue = q
    res['ticks'] = t
    res['stats'] = stats
    if part_c:
        res['co'] = co.tolist()
    return res


def standin_probe(seed, phys='x1c'):
    ph = Physics() if phys == 'x1c' else LawfulPhysics()
    S = standin(ph)
    return dict(inner=probe(ph, S, seed, 'inner'), hear=probe(ph, S, seed, 'hear'))


if __name__ == '__main__':
    arms = sys.argv[1].split(',')
    seeds = [int(s) for s in sys.argv[2].split(',')]
    ratio = float(sys.argv[3])
    out = sys.argv[4]
    phys = sys.argv[5] if len(sys.argv) > 5 else 'x1c'
    nbab = int(sys.argv[6]) if len(sys.argv) > 6 else 2000
    closed = len(sys.argv) > 7 and sys.argv[7] == 'closed'
    cps = tuple(c for c in (500, 1000, 2000, 5000) if c <= nbab)
    allr = []
    for seed in seeds:
        for arm in arms:
            t0 = time.time()
            r = life(arm, seed, ratio=ratio, closed=closed, phys=phys, n_babble=nbab, checkpoints=cps)
            r['phys'] = phys
            r['secs'] = time.time() - t0
            allr.append(r)
            last = r['cps'][max(r['cps'])]
            print(arm, seed, ratio, phys, 'secs %.1f' % r['secs'], 'stats', r['stats'],
                  {cp: (round(v['inner']['correct'], 2), round(v['hear']['correct'], 2), round(v['inner']['garble'], 2))
                   for cp, v in r['cps'].items()},
                  'w_true %.4f w_other %.4f top4 %.2f hits %.2f oracle %s' % (last['w_true'], last['w_other'],
                  last['top4_exact'], last['top4_hits'], last['oracle']), flush=True)
    allr.append(dict(arm='standin', phys=phys, **{str(s): standin_probe(s, phys) for s in seeds}))
    with open(out, 'w') as f:
        json.dump(allr, f, default=float)
