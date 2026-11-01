"""X3: the designer's hippocampus on a continuous tick-by-tick stream of chain copies (local numpy, 1 process).

Stream: stream.json (the v2.2 bench Life world, recorded by record_stream.py; 120 days, 2.58 M ticks, 12,087 lines).
  Each tick the copied areas show
    AUD  : 2,048 binary units; a heard char = a fixed random 24-unit assembly (lineage/physics seed), 3 ticks per char
    VOICE:   256 units; the speaker ('other', 'other2') = an 8-unit set, on while that speaker talks
    EYE  : 1,024 units; the world's eye bits (the chain rule drops always-on units: none is on > 95 % of the time)
  Hc = the 1:1 chain copies of these 3,328 units inside the hippocampus.
  Probe episodes go into silent daytime gaps: novel 6-char lines of familiar chars, said by a caregiver in whatever
  scene the eye shows. They are the unique episodes whose recall is scored against age.

Hippocampus (the designer's form):
  * binary units, its own population; storage by Hebbian links among its own units; no per-memory cells, no records,
    no time cells (IDEAS #24: time = order);
  * the designer's rule (chain-neuron plastic2 / ana PlasticRule): only units that just changed take part;
      falling -> rising : w += eta_seq * g(w)      rising <-> rising: w += eta_co * g(w)
      g(w) = 1 / (1 + w / w_ref) ** pow            (concave, uncapped growth)
  * one global decay with the hippocampus's own gamma: every T ticks all plastic links w <- gamma * w; links below
    w_min are deleted; birth links are exempt.
  Arms:
    'copy'   : the storage population is Hc itself (dense Hc x Hc potential links), no recode;
    'recode' : Hc -> Hs by fixed random birth links; Hs = N_H units, a global threshold keeps about k on (k-WTA is
               the #21 stand-in for the quantile threshold + inhibitory pool); plastic links Hs <-> Hs and
               Hs -> Hc (co-rising), the readout that carries recall out to the PFC;
    'recode_rec': as 'recode', plus the population's own random birth recurrent links (designer #5: born full of
               random connections; exempt from decay), so a moment's code also depends on the previous code: order
               becomes context without any time cell.
  The Hs state is recomputed when its input changes (event-driven shortcut; constant input = settled state).
Recall (no learning): the probe's first 3 moments are replayed through the chain copies with only a fraction f of
  each moment's Hc units on (f = 1, 0.5, 0.2); then the hippocampus runs on its own links for 3 steps; each step's Hc
  AUD readout is decoded to the best char assembly; score = fraction of chars 4..6 recalled in order.
Run: python3 -I x3_hippo.py <experiment>
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
N_A, K_A, N_V, K_V, N_E = 2048, 24, 256, 8, 1024
O_V, O_E = N_A, N_A + N_V
N_HC = N_A + N_V + N_E
SYL = 3
LP, NCUE = 6, 3


def load_stream():
    d = json.load(open(os.path.join(HERE, 'stream.json')))
    return d['events'], d['ticks']


# ----------------------------------------------------------------------------------------------- the world stream
class World:
    def __init__(self, ev, t_end, seed_phys=7, seed_probe=11, probe_every=2000, kinds=None, spread_gap=3000,
                 n_spread=4):
        rng = np.random.default_rng(seed_phys)                       # lineage physics seed: fixed codes
        chars = sorted({ch for e in ev if e[1] == 'line' for ch in e[3]})
        self.chars = chars
        self.cix = {c: i for i, c in enumerate(chars)}
        self.asm = np.stack([np.sort(rng.choice(N_A, K_A, replace=False)) for _ in chars]).astype(np.int32)
        self.voice = {w: np.sort(rng.choice(N_V, K_V, replace=False)).astype(np.int32) + O_V
                      for w in ('other', 'other2', 'self')}
        self.t_end = t_end
        lines = [e for e in ev if e[1] == 'line' and e[0] < t_end]
        eyes = [(e[0], np.asarray(e[4], np.int32) + O_E) for e in ev if e[0] < t_end]
        self.eye_t = np.array([t for t, _ in eyes], np.int64)
        self.eye_v = [v for _, v in eyes]
        self.eye_change = np.array([eyes[i][0] for i in range(1, len(eyes))
                                    if not np.array_equal(eyes[i][1], eyes[i - 1][1])], np.int64)
        utt = [(e[0], e[2], e[3], 'world', -1) for e in lines]
        prng = np.random.default_rng(seed_probe)
        kinds = kinds or ['u', 'u', 'pS', 'u', 'pV', 'rB', 'u', 'pD', 'rS']
        self.probes = []
        starts = [u[0] for u in utt]
        ends = [u[0] + SYL * len(u[2]) for u in utt]
        gaps = [(ends[i], starts[i + 1]) for i in range(len(utt) - 1)]
        next_t, ki = 1500, 0
        pend = []
        dur = SYL * LP

        def eye_at(t):
            i = np.searchsorted(self.eye_t, t, side='right') - 1
            return self.eye_v[i] if i >= 0 else np.zeros(0, np.int32)

        def quiet(t0, t1):
            i = np.searchsorted(self.eye_change, t0, side='left')
            return i >= len(self.eye_change) or self.eye_change[i] > t1

        def new_text(k=LP, avoid=''):
            pool = [c for c in chars if c not in avoid]
            return ''.join(prng.choice(pool, k, replace=False))

        for g0, g1 in gaps:
            if g1 - g0 < dur + 30:
                continue
            t = g0 + 15
            if pend and pend[0][0] <= t and len(eye_at(t)) and quiet(t, t + dur + 2):
                tmin, pid, who, need_change = pend[0]
                P = self.probes[pid]
                ok = True
                if need_change:
                    t_a = self.probes[P['pair']]['t'][0]
                    ok = np.searchsorted(self.eye_change, t_a) < np.searchsorted(self.eye_change, t)
                if ok:
                    pend.pop(0)
                    P['t'].append(t); P['who_t'].append(who)
                    utt.append((t, who, P['text'], 'probe', pid))
                    continue
            if t < next_t or g1 - g0 < 3 * dur + 60 or not len(eye_at(t)) or not quiet(t, t + 3 * dur + 40):
                continue
            kind = kinds[ki % len(kinds)]; ki += 1; next_t = t + probe_every
            if kind in ('u', 'rB', 'rS'):
                P = dict(kind=kind, text=new_text(), t=[t], who_t=['other'], pair=-1)
                pid = len(self.probes); self.probes.append(P)
                utt.append((t, 'other', P['text'], 'probe', pid))
                if kind == 'rB':                                      # 3 back-to-back repetitions, same scene
                    for r in (1, 2):
                        tr = t + (dur + 6) * r; P['t'].append(tr); P['who_t'].append('other')
                        utt.append((tr, 'other', P['text'], 'probe', pid))
                if kind == 'rS':                                      # repetitions spread over time and scenes
                    for r in range(1, n_spread):
                        pend.append((t + spread_gap * r, pid, 'other', False))
                    pend.sort(key=lambda x: x[0])
            else:                                                     # similar pair: shared 3-char prefix
                a = new_text()
                b = a[:NCUE] + new_text(LP - NCUE, avoid=a)
                ia = len(self.probes)
                PA = dict(kind=kind, text=a, t=[t], who_t=['other'], pair=ia + 1)
                PB = dict(kind=kind, text=b, t=[], who_t=[], pair=ia)
                self.probes += [PA, PB]
                utt.append((t, 'other', a, 'probe', ia))
                if kind in ('pS', 'pV'):
                    tb = t + dur + 10; who = 'other' if kind == 'pS' else 'other2'
                    PB['t'].append(tb); PB['who_t'].append(who)
                    utt.append((tb, who, b, 'probe', ia + 1))
                else:                                                 # pD: B after the next scene change
                    pend.append((t + 100, ia + 1, 'other', True)); pend.sort(key=lambda x: x[0])
        utt.sort(key=lambda u: u[0])
        self.utt = utt
        # chain copies exist only for feature units (ARCHITECTURE.md 2.3): never-on / always-on units get none
        eye_on = np.zeros(N_E, bool)
        for v in self.eye_v:
            eye_on[v - O_E] = True
        self.feat = np.unique(np.concatenate([self.asm.ravel(), self.voice['other'], self.voice['other2'],
                                              np.flatnonzero(eye_on) + O_E])).astype(np.int32)
        self.n_sp = int(0.05 * len(self.feat))                       # spare stable IDs for children of splits
        self.pert = None

    def set_perturb(self, t_p, frac, rng):
        """slow split / delete of the copied areas (stable IDs, as in the chain repo): at t_p a fraction of the
        feature units is deleted (IDs kept, dead) and the same number of children is added with new IDs N_HC + j;
        a child takes 2 of its parent's inputs, so it is on whenever its parent is on (ASSUMED)"""
        n = int(round(frac * len(self.feat)))
        dead = np.zeros(N_HC + self.n_sp, bool)
        dead[rng.choice(self.feat, n, replace=False)] = True
        alive = self.feat[~dead[self.feat]]
        parent = rng.choice(alive, min(n, self.n_sp), replace=False)
        self.pert = dict(t=t_p, dead=dead, parent=parent, n=n)

    def apply(self, hc, t):
        if self.pert is None or t < self.pert['t'] or len(hc) == 0:
            return hc
        p = self.pert
        m = np.zeros(N_HC + self.n_sp, bool); m[hc] = True
        kids = N_HC + np.flatnonzero(m[p['parent']])
        hc = hc[~p['dead'][hc]]
        return np.concatenate([hc, kids]).astype(np.int32)

    def alive_asm(self, c, t):
        a = self.asm[c]
        if self.pert is None or t < self.pert['t']:
            return a
        return a[~self.pert['dead'][a]]

    def eye_at(self, t):
        i = np.searchsorted(self.eye_t, t, side='right') - 1
        return self.eye_v[i] if i >= 0 else np.zeros(0, np.int32)

    def changes(self):
        """yield (t, Hc active set, info) at every tick where Hc changes"""
        ev_t = set(self.eye_t.tolist())
        aud = {}
        for t0, who, text, src, pid in self.utt:
            for i, ch in enumerate(text):
                aud[t0 + SYL * i] = (self.cix[ch], who, pid, i)
            aud.setdefault(t0 + SYL * len(text), None)
        times = sorted(ev_t | set(aud.keys()))
        ei = -1; cur_eye = np.zeros(0, np.int32); cur_a = None
        for t in times:
            if t >= self.t_end:
                break
            while ei + 1 < len(self.eye_t) and self.eye_t[ei + 1] <= t:
                ei += 1; cur_eye = self.eye_v[ei]
            if t in aud:
                cur_a = aud[t]
            parts = [cur_eye]
            if cur_a is not None:
                c, who, pid, pos = cur_a
                parts = [self.asm[c], self.voice[who], cur_eye]
            yield t, self.apply(np.unique(np.concatenate(parts)).astype(np.int32), t), cur_a

    def moments(self, P, rep=-1, t_now=0):
        t0 = P['t'][rep]; who = P['who_t'][rep]
        eye = self.eye_at(t0)
        ms = [np.unique(np.concatenate([self.asm[self.cix[ch]], self.voice[who], eye])) for ch in P['text']]
        return self.apply(eye, t_now), [self.apply(m, t_now) for m in ms]

    def decode(self, aud_units, t=0):
        """the char whose (surviving) assembly the recalled source-area units cover best; None below half"""
        if len(aud_units) == 0:
            return None
        m = np.zeros(N_A, bool); m[aud_units] = True
        ok = np.ones(N_A, bool)
        if self.pert is not None and t >= self.pert['t']:
            ok = ~self.pert['dead'][:N_A]
        ov = (m[self.asm] & ok[self.asm]).sum(1)
        n = ok[self.asm].sum(1)
        j = int(np.argmax(ov / np.maximum(n, 1)))
        return j if ov[j] >= n[j] / 2 else None

    def fidelity(self, aud_units, c, t=0):
        a = self.alive_asm(c, t)
        return float(np.isin(a, aud_units).mean()) if len(a) else 0.0


# ----------------------------------------------------------------------------------------------- links
class Links:
    """CSR by pre, fixed fan-out of potential partners (birth anatomy). plastic: weights 0 at birth, effective weight
    = v * scale, the global decay multiplies scale lazily, links below w_min count as deleted. fixed: birth weights."""

    def __init__(self, n_pre, n_post, fan, rng, cfg, no_self=False, fixed=False):
        self.n_post = n_post
        dt = np.int32 if n_post > 32000 else np.int16
        if fan >= n_post - 1:
            fan = n_post - 1
            base = np.arange(n_post - 1)
            self.post = np.stack([np.where(base >= i, base + 1, base) for i in range(n_pre)]).astype(dt)
        else:
            self.post = rng.integers(0, n_post, size=(n_pre, fan)).astype(dt)
            if no_self and n_pre == n_post:
                sm = self.post == np.arange(n_pre)[:, None]
                self.post[sm] = ((self.post[sm].astype(np.int64) + 1) % n_post).astype(dt)
        self.fixed = fixed
        self.v = (rng.uniform(0.05, 0.20, (n_pre, fan)).astype(np.float32) if fixed
                  else np.zeros((n_pre, fan), np.float32))
        self.scale = 1.0
        self.c = cfg

    def g(self, w):
        return 1.0 / (1.0 + w / self.c['w_ref']) ** self.c['pow']

    def learn(self, pre, post_mask, eta):
        if len(pre) == 0 or eta == 0:
            return
        P = self.post[pre]
        m = post_mask[P]
        if not m.any():
            return
        sub = self.v[pre]
        w = sub * self.scale
        w = np.where(w < self.c['w_min'], 0.0, w)
        self.v[pre] = np.where(m, (w + eta * self.g(w)) / self.scale, sub)

    def depress(self, pre, post_mask, eta):
        """reverse order (ARCHITECTURE.md 2.8): B before A weakens A -> B; pre = rising A, post = falling B"""
        if len(pre) == 0 or eta == 0:
            return
        P = self.post[pre]
        m = post_mask[P]
        if not m.any():
            return
        sub = self.v[pre]
        w = sub * self.scale
        self.v[pre] = np.where(m, np.maximum(0.0, w - eta * self.g(w)) / self.scale, sub)

    def prop_norm(self, blk=500):
        """two-layer weights: propagation = strength / sqrt(out-strength of pre x in-strength of post)"""
        n = len(self.v)
        self.s_out = np.zeros(n, np.float32); s_in = np.zeros(self.n_post, np.float64)
        for b in range(0, n, blk):
            w = self.v[b:b + blk] * self.scale
            w[w < self.c['w_min']] = 0.0
            self.s_out[b:b + blk] = w.sum(1)
            s_in += np.bincount(self.post[b:b + blk].ravel(), weights=w.ravel(), minlength=self.n_post)
        self.s_out += 1e-6
        self.s_in = (s_in + 1e-6).astype(np.float32)

    def drive_prop(self, pre):
        if len(pre) == 0:
            return np.zeros(self.n_post, np.float32)
        w = self.v[pre] * self.scale
        w = np.where(w < self.c['w_min'], 0.0, w)
        P = self.post[pre]
        p = w / np.sqrt(self.s_out[pre][:, None] * self.s_in[P])
        return np.bincount(P.ravel(), weights=p.ravel(), minlength=self.n_post).astype(np.float32)

    def decay(self, n):
        if n <= 0 or self.fixed:
            return
        self.scale *= self.c['gamma'] ** n
        if self.scale < 1e-4:
            for b in range(0, len(self.v), 500):
                blk = self.v[b:b + 500]
                blk *= self.scale
                blk[blk < self.c['w_min']] = 0.0
            self.scale = 1.0

    def drive(self, pre):
        if len(pre) == 0:
            return np.zeros(self.n_post, np.float32)
        w = self.v[pre].ravel() * self.scale
        if not self.fixed:
            w[w < self.c['w_min']] = 0.0
        return np.bincount(self.post[pre].ravel(), weights=w, minlength=self.n_post).astype(np.float32)

    def n_live(self, blk=500):
        return int(sum((self.v[b:b + blk] * self.scale >= self.c['w_min']).sum() for b in range(0, len(self.v), blk)))


class Birth:
    """random fan-in Hc -> Hs from birth, CSR by pre. ARCHITECTURE.md 3: links into and inside a deep area are born
    random and are trained: w = w_birth + dw, where only the learned part dw decays (one global decay, the
    hippocampus's gamma) and the birth part never decays. eta_in = 0 keeps the recode fixed."""

    def __init__(self, n_h, cA, cV, cE, rng, cfg=None, feat=None, n_sp=0):
        feat = np.arange(N_HC) if feat is None else feat
        fa, fv, fe = feat[feat < O_V], feat[(feat >= O_V) & (feat < O_E)], feat[feat >= O_E]
        # links from units that have no copy do not exist: each block's fan-in shrinks with its feature share
        cA = max(1, int(round(cA * len(fa) / N_A))); cV = max(1, int(round(cV * len(fv) / N_V)))
        cE = max(1, int(round(cE * len(fe) / N_E)))
        pre = np.concatenate([rng.choice(fa, (n_h, cA)), rng.choice(fv, (n_h, cV)),
                              rng.choice(fe, (n_h, cE))], 1).ravel()
        post = np.repeat(np.arange(n_h), cA + cV + cE)
        if n_sp:                                          # birth links of future children's copies (new stable IDs)
            fan = int(round(n_h * cA / len(fa)))
            pre = np.concatenate([pre, np.repeat(N_HC + np.arange(n_sp), fan)])
            post = np.concatenate([post, rng.integers(0, n_h, n_sp * fan)])
        w = rng.uniform(0.05, 0.20, len(pre)).astype(np.float32)
        o = np.argsort(pre, kind='stable')
        self.post, self.w = post[o].astype(np.int32), w[o]
        self.ptr = np.searchsorted(pre[o], np.arange(N_HC + n_sp + 1))
        self.n_h = n_h
        self.dv = np.zeros_like(self.w); self.scale = 1.0; self.c = cfg
        self.learned = False

    def idx(self, hc):
        return np.concatenate([np.arange(self.ptr[j], self.ptr[j + 1]) for j in hc]) if len(hc) else np.zeros(0, int)

    def weights(self, idx):
        w = self.w[idx]
        if self.learned:
            dw = self.dv[idx] * self.scale
            w = w + np.where(np.abs(dw) < self.c['w_min'], 0.0, dw)
        return w

    def drive(self, hc):
        if len(hc) == 0:
            return np.zeros(self.n_h, np.float32)
        idx = self.idx(hc)
        return np.bincount(self.post[idx], weights=np.maximum(self.weights(idx), 0.0),
                           minlength=self.n_h).astype(np.float32)

    def learn(self, pre, post_mask, eta, sign=1.0):
        if len(pre) == 0 or eta == 0:
            return
        idx = self.idx(pre)
        m = post_mask[self.post[idx]]
        if not m.any():
            return
        idx = idx[m]
        w = np.maximum(self.weights(idx), 0.0)
        g = 1.0 / (1.0 + w / self.c['w_ref']) ** self.c['pow']
        dw = self.dv[idx] * self.scale + sign * eta * g
        dw = np.maximum(dw, -self.w[idx])                  # the total weight cannot go below 0
        self.dv[idx] = dw / self.scale
        self.learned = True

    def decay(self, n):
        if n <= 0 or not self.learned:
            return
        self.scale *= self.c['gamma'] ** n
        if self.scale < 1e-4:
            self.dv *= self.scale; self.scale = 1.0
            self.dv[np.abs(self.dv) < self.c['w_min']] = 0.0


def kwta(d, k, rng):
    """top-k (used only by the experimenter's AUD decoder, and as the alpha = 1 limit of Thr)"""
    pos = np.flatnonzero(d > 0)
    if len(pos) <= k:
        return np.sort(pos)
    dd = d[pos] + 1e-6 * rng.random(len(pos))
    return np.sort(pos[np.argpartition(-dd, k)[:k]])


class Thr:
    """the area's global threshold (ARCHITECTURE.md 2.1): it tracks the drive quantile that would leave a fraction f
    on, smoothed with rate alpha per update, so the number on varies with the signal (a weak or absent signal turns
    few or none on); gain control = the inhibitory pool's ceiling: at most `ceil` x the set-point count can be on
    (neuro review #8 "at most k" form). No fixed quota. alpha = 1 and ceil = 1 is the top-k limit."""

    def __init__(self, f, alpha, theta=None, ceil=2.0):
        self.f, self.a, self.theta, self.ceil = f, alpha, theta, ceil

    def fire(self, d, rng):
        n = len(d)
        k = max(1, int(round(self.f * n)))
        q = float(np.partition(d, n - k)[n - k])
        self.theta = q if self.theta is None else self.theta + self.a * (q - self.theta)
        dd = d + 1e-7 * rng.random(n)
        on = np.flatnonzero((dd >= max(self.theta, 1e-9)) & (d > 0))
        kmax = int(self.ceil * k)
        if len(on) > kmax:
            on = np.sort(on[np.argpartition(-dd[on], kmax)[:kmax]])
        return on


# ----------------------------------------------------------------------------------------------- the hippocampus
class Hippo:
    def __init__(self, cfg, seed=3, feat=None, n_sp=0):
        self.c = cfg
        self.dead_aud = None
        self.dead_hc = None
        self.rng = np.random.default_rng(seed)
        self.arm = cfg['arm']
        lc = dict(w_ref=cfg['w_ref'], pow=cfg['pow'], w_min=cfg['w_min'], gamma=cfg['gamma'])
        if self.arm.startswith('recode'):
            n = cfg['N_H']
            self.birth = Birth(n, cfg['cA'], cfg['cV'], cfg['cE'], np.random.default_rng(seed + 1), lc, feat, n_sp)
            self.rec = Links(n, n, cfg['C'], np.random.default_rng(seed + 2), lc, no_self=True)
            self.out = Links(n, N_A + N_V, cfg['C_out'], np.random.default_rng(seed + 3), lc)
            self.brec = (Links(n, n, cfg['cB'], np.random.default_rng(seed + 4), lc, no_self=True, fixed=True)
                         if self.arm == 'recode_rec' else None)
            self.n = n
        else:
            self.rec = Links(N_HC + n_sp, N_HC + n_sp, cfg['C'], np.random.default_rng(seed + 2), lc, no_self=True)
            self.n = N_HC + n_sp
            self.brec = None
        self.prev = np.zeros(0, np.int64)
        self.prev_hc = np.zeros(0, np.int64)
        self.t_dec = 0
        self.thr = Thr(cfg['k'] / self.n, cfg['alpha'], ceil=cfg['ceil'])

    def encode(self, hc, prev, thr):
        """the population's state when its input changes: input through the birth links, plus the previous state
        through the birth recurrence (rho) and through the learned links' propagation weights (lam): one dynamics for
        listening and for cued recall"""
        if not self.arm.startswith('recode'):
            return hc
        d = self.birth.drive(hc)
        if len(prev):
            if self.brec is not None:
                r = self.brec.drive(prev)
                d = d + self.c['rho'] * (r - r.mean())    # gain control: inhibition grows with the total activity
            if self.c['lam'] and getattr(self.rec, 's_out', None) is not None:
                d = d + self.c['lam'] * self.rec.drive_prop(prev)
        return thr.fire(d, self.rng)

    def step(self, t, hc):
        nd = t // self.c['T'] - self.t_dec
        if nd > 0:
            self.rec.decay(nd)
            if self.arm.startswith('recode'):
                self.out.decay(nd); self.birth.decay(nd)
            self.t_dec += nd
        self.n_step = getattr(self, 'n_step', 0) + 1
        if self.c['lam'] and self.n_step % self.c['norm_every'] == 0:
            self.rec.prop_norm()
        x = self.encode(hc, self.prev, self.thr)
        rising = np.setdiff1d(x, self.prev, assume_unique=True)
        falling = np.setdiff1d(self.prev, x, assume_unique=True)
        rm = np.zeros(self.n, bool); rm[rising] = True
        if self.c['eta_rev'] and len(falling):
            fm = np.zeros(self.n, bool); fm[falling] = True
            self.rec.depress(rising, fm, self.c['eta_rev'])
        self.rec.learn(falling, rm, self.c['eta_seq'])
        self.rec.learn(rising, rm, self.c['eta_co'])
        if self.arm.startswith('recode') and self.c['eta_in']:
            hrise = np.setdiff1d(hc, self.prev_hc, assume_unique=True)
            hfall = np.setdiff1d(self.prev_hc, hc, assume_unique=True)
            self.birth.learn(hrise, rm, self.c['eta_in'])                 # co-rising input -> code
            self.birth.learn(hfall, rm, self.c['eta_in'])                 # input that just ended -> code (order)
            if len(falling):
                fm = np.zeros(self.n, bool); fm[falling] = True
                self.birth.learn(hrise, fm, self.c['eta_in'], sign=-1.0)  # code on before the input: weakens
        if self.arm.startswith('recode'):
            hr = np.setdiff1d(hc, self.prev_hc, assume_unique=True)
            hr = hr[hr < N_A + N_V]
            if len(hr):
                hm = np.zeros(N_A + N_V, bool); hm[hr] = True
                self.out.learn(rising, hm, self.c['eta_out'])
        self.prev = x; self.prev_hc = hc
        return x

    def set_time(self, t, W):
        on = W.pert is not None and t >= W.pert['t']
        self.dead_aud = W.pert['dead'][:N_A] if on else None
        self.dead_hc = W.pert['dead'] if on else None

    def prepare_recall(self):
        if self.c['twolayer'] or self.c['lam']:
            self.rec.prop_norm()

    def aud_of(self, x):
        """recall reaches the source area (AUD) through the hippocampus's return chains; deleted units cannot fire"""
        if not self.arm.startswith('recode'):
            a = x[x < N_A]
            return a if self.dead_aud is None else a[~self.dead_aud[a]]
        d = self.out.drive(x)[:N_A]
        if self.dead_aud is not None:
            d[self.dead_aud] = 0.0
        return kwta(d, K_A, self.rng)

    def free(self, x, steps):
        xs = []
        a = self.c['alpha']
        if self.arm.startswith('recode'):
            th = Thr(self.c['k'] / self.n, a, ceil=self.c['ceil'])
        else:
            th = [Thr(K_A / N_A, a, ceil=self.c['ceil']), Thr(K_V / N_V, a, ceil=self.c['ceil']),
                  Thr(55 / N_E, a, ceil=self.c['ceil'])]
        for _ in range(steps):
            d = self.rec.drive_prop(x) if self.c['twolayer'] else self.rec.drive(x)
            if self.arm.startswith('recode'):
                x = th.fire(d, self.rng)
            else:
                x = np.concatenate([th[0].fire(d[:N_A], self.rng), th[1].fire(d[O_V:O_E], self.rng) + O_V,
                                    th[2].fire(d[O_E:], self.rng) + O_E])
            x = x[d[x] > 0]
            if self.dead_hc is not None and not self.arm.startswith('recode'):
                x = x[~self.dead_hc[x]]
            xs.append(x)
        return xs

    def present(self, eye, moments, frac, rng):
        """replay a cue through the chain copies (no learning); returns the state after the last cue moment"""
        def sub(s):
            return np.sort(rng.choice(s, max(1, int(round(frac * len(s)))), replace=False)) if len(s) else s
        th = Thr(self.thr.f, self.thr.a, self.thr.theta, self.thr.ceil)
        x = self.encode(sub(eye), np.zeros(0, np.int64), th)
        for m in moments:
            x = self.encode(sub(m), x, th)
        return x


# ----------------------------------------------------------------------------------------------- tests
def recall(H, W, P, frac, rng, ncue=NCUE, steps=None, mode='store', t_now=0, fid=None):
    """mode 'store': the cue is a random fraction of the stored hippocampal pattern of moment ncue (the reviewers'
    form, pattern completion + sequence inside the hippocampus); mode 'input': the first ncue moments are replayed
    through the chain copies with a fraction of each moment's copy units on (end to end, through the recode)"""
    steps = steps or LP - ncue
    if mode == 'store':
        code = P['code'][ncue - 1]
        x = np.sort(rng.choice(code, max(1, int(round(frac * len(code)))), replace=False)) if len(code) else code
    else:
        eye, ms = W.moments(P, t_now=t_now)
        x = H.present(eye, ms[:ncue], frac, rng)
    xs = H.free(x, steps)
    auds = [H.aud_of(x) for x in xs]
    got = [W.decode(a, t_now) for a in auds]
    tgt = [W.cix[c] for c in P['text'][ncue:ncue + steps]]
    if fid is not None:
        fid.append(float(np.mean([W.fidelity(a, c, t_now) for a, c in zip(auds, tgt)])))
    return got, tgt


def test_all(H, W, t_now, fracs, rng, n_recent=30, n_old=30):
    res = []
    H.set_time(t_now, W)
    H.prepare_recall()
    probes = [(i, P) for i, P in enumerate(W.probes) if P['t'] and P['t'][-1] + SYL * LP < t_now and 'code' in P
              and all(c is not None for c in P['code'])]
    old = probes[:-n_recent]
    if len(old) > n_old:
        old = [old[j] for j in sorted(rng.choice(len(old), n_old, replace=False))]
    for i, P in old + probes[-n_recent:]:
        age = t_now - (P['t'][-1] + SYL * LP)
        row = dict(i=i, kind=P['kind'], age=int(age), n_exp=len(P['t']),
                   pert=int(W.pert is not None and P['t'][-1] < W.pert['t'] <= t_now))
        for f in fracs:
            fid = []
            got, tgt = recall(H, W, P, f, rng, t_now=t_now, fid=fid)
            row[f'r{int(f * 100)}'] = float(np.mean([g == t for g, t in zip(got, tgt)]))
            row[f'fid{int(f * 100)}'] = fid[0]
            if f in IN_FRACS:
                g2, t2 = recall(H, W, P, f, rng, mode='input', t_now=t_now)
                row[f'i{int(f * 100)}'] = float(np.mean([g == t for g, t in zip(g2, t2)]))
            if P['pair'] >= 0 and W.probes[P['pair']]['t']:
                Q = W.probes[P['pair']]
                row[f'sep_right{int(f * 100)}'] = float(got[0] == tgt[0])
                row[f'sep_other{int(f * 100)}'] = float(got[0] == W.cix[Q['text'][NCUE]])
        res.append(row)
    return res


def kicks(H, W, t_now, rng, n=50, steps=4):
    H.set_time(t_now, W)
    H.prepare_recall()
    probe_bi, world_bi = set(), set()
    for P in W.probes:
        if P['t'] and P['t'][-1] < t_now:
            probe_bi |= {(W.cix[a], W.cix[b]) for a, b in zip(P['text'], P['text'][1:])}
    for t0, who, text, src, pid in W.utt:
        if t0 < t_now and src == 'world':
            world_bi |= {(W.cix[a], W.cix[b]) for a, b in zip(text, text[1:])}
    out = dict(n=n, silent=0, decoded=0, world_seq=0, probe_seq=0)
    for _ in range(n):
        x0 = (np.sort(rng.choice(H.n, H.c['k'], replace=False)) if H.arm.startswith('recode')
              else np.sort(rng.choice(N_A, K_A, replace=False)))
        dec = [W.decode(H.aud_of(x), t_now) for x in H.free(x0, steps)]
        if all(d is None for d in dec):
            out['silent'] += 1; continue
        out['decoded'] += 1
        pairs = [(a, b) for a, b in zip(dec, dec[1:]) if a is not None and b is not None]
        if any(p in probe_bi for p in pairs):
            out['probe_seq'] += 1
        elif any(p in world_bi for p in pairs):
            out['world_seq'] += 1
    return out


_EV = {}


def run(cfg, t_end, test_at, fracs=(1.0, 0.5, 0.2), seed=0, do_kicks=False, verbose=True):
    t0c = time.time()
    if 'ev' not in _EV:
        _EV['ev'] = load_stream()
    ev, _ = _EV['ev']
    W = World(ev, t_end, probe_every=cfg.get('probe_every', 2000), kinds=cfg.get('kinds'),
              seed_probe=11 + seed)
    rng = np.random.default_rng(100 + seed)
    if cfg.get('pert_frac'):
        W.set_perturb(int(cfg['pert_t']), cfg['pert_frac'], np.random.default_rng(900 + seed))
    H = Hippo(cfg, seed=3 + seed, feat=W.feat, n_sp=W.n_sp)
    tests, kick_res = [], []
    ti = n_ch = 0
    for t, hc, info in W.changes():
        while ti < len(test_at) and t >= test_at[ti]:
            tests += [dict(r, t_test=int(test_at[ti])) for r in test_all(H, W, test_at[ti], fracs, rng)]
            if do_kicks:
                kick_res.append(dict(t=int(test_at[ti]), **kicks(H, W, test_at[ti], rng)))
            ti += 1
        H.set_time(t, W)
        x = H.step(t, hc); n_ch += 1
        if info is not None and info[2] >= 0:
            P = W.probes[info[2]]
            P.setdefault('code', [None] * LP)[info[3]] = x
    while ti < len(test_at):
        tests += [dict(r, t_test=int(test_at[ti])) for r in test_all(H, W, test_at[ti], fracs, rng)]
        if do_kicks:
            kick_res.append(dict(t=int(test_at[ti]), **kicks(H, W, test_at[ti], rng)))
        ti += 1
    import resource
    out = dict(tests=tests, kicks=kick_res, n_probes=len(W.probes), live=H.rec.n_live(),
               secs=round(time.time() - t0c, 1), n_change=n_ch,
               rss_mb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024))
    if verbose:
        print(f'   [{cfg["arm"]}] {n_ch} change ticks, {len(W.probes)} probes, live links {out["live"]}, '
              f'{out["secs"]} s, max RSS {out["rss_mb"]} MB', flush=True)
    return out


IN_FRACS = (1.0, 0.5)
AGE_BINS = [0, 1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000, 500000, 1000000]


def summarize(tests, key='r50', kinds=('u',), n_exp=None):
    rows = [r for r in tests if r['kind'] in kinds and key in r and (n_exp is None or r['n_exp'] in n_exp)]
    out = []
    for lo, hi in zip(AGE_BINS, AGE_BINS[1:]):
        v = [r[key] for r in rows if lo <= r['age'] < hi]
        if v:
            out.append((lo, hi, len(v), round(float(np.mean(v)), 3)))
    return out


def horizon(tests, key='r50', kinds=('u',), thr=0.9):
    h = 0
    for lo, hi, n, m in summarize(tests, key, kinds):
        if m >= thr and n >= 3:
            h = hi
        else:
            break
    return h


BASE = dict(arm='recode_rec', N_H=10000, k=64, C=2000, C_out=1024, cA=200, cV=40, cE=60, cB=400, rho=1.0,
            eta_seq=2.0, eta_co=0.2, eta_rev=2.0, eta_out=2.0, alpha=0.3, ceil=2.0, twolayer=True, lam=0.0, norm_every=200, eta_in=0.0, pert_frac=0.0, pert_t=0, probe_every=2000, w_ref=1.0, pow=1.0, w_min=0.01, gamma=0.99, T=200)

def parse_over(args):
    o = {}
    for a in args:
        k, v = a.split('=')
        b = BASE.get(k)
        o[k] = (v == '1') if isinstance(b, bool) else (type(b)(v) if b is not None else (int(v) if v.isdigit() else v))
    return o


if __name__ == '__main__':
    exp = sys.argv[1] if len(sys.argv) > 1 else 'smoke'
    if exp == 'smoke':
        ov = parse_over([a for a in sys.argv[2:] if not a.startswith('t_end=')])
        t_end = int(([a.split('=')[1] for a in sys.argv[2:] if a.startswith('t_end=')] or [60000])[0])
        cfg = dict(BASE, **ov)
        r = run(cfg, t_end, list(range(3000, t_end + 1, 1500)), do_kicks=False)
        for key in ('r100', 'r50', 'r20', 'i100', 'i50'):
            print(cfg['arm'], key, summarize(r['tests'], key))
