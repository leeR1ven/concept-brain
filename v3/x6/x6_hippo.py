"""X6: harvest items 17, 18, 19, 21, 22 on the X3 hippocampus.

x3_hippo.py is imported unchanged. With all new options at their defaults this is the X3 base.
New options (cfg keys):
  beta      chain-input gain on the birth drive (item 18); 1.0 = X3
  unified   True = no mode switch (item 18): every update, also recall, is
            d = beta*birth(input) + rho*(brec(prev) - mean) + lam*rec_prop(prev); recall keeps the silent input
  win       learning window in ticks for the sequence term (item 17 alternative); <= 2 = X3 (just-fallen units only)
  pause, iv, step_pause : the Q-A probes of item 17 (pause ticks, inner-voice fraction, stepped pause)
World kinds: X3 kinds plus 'qa' (item 17 triples). Facts (item 22) are passed as a plan.
"""
import os
import sys
import time
import resource

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'x345'))   # repo: x3_hippo.py and stream.json from ../x345
import x3_hippo as X  # noqa: E402
from x3_hippo import N_A, K_A, N_V, N_E, O_V, O_E, N_HC, SYL, LP, NCUE, Thr, kwta  # noqa: E402,F401

PET_NAMES = ['豆豆', '花花', '球球', '星星', '白白', '月月', '熊熊', '果果']
PETS = ['小狗', '小猫', '鸭子', '小鸟']
QA_PAUSE_MAX = 30


class World6(X.World):
    """X3's world with two extra probe kinds. Built like X3.World (same physics seed, same assemblies)."""

    def __init__(self, ev, t_end, seed_phys=7, seed_probe=11, probe_every=2000, kinds=None, spread_gap=3000,
                 n_spread=4, pause=12, iv_frac=0.0, step_pause=True, n_pool=6, facts=None):
        rng = np.random.default_rng(seed_phys)
        chars = sorted({ch for e in ev if e[1] == 'line' for ch in e[3]})
        self.chars = chars
        self.cix = {c: i for i, c in enumerate(chars)}
        self.asm = np.stack([np.sort(rng.choice(N_A, K_A, replace=False)) for _ in chars]).astype(np.int32)
        self.voice = {w: np.sort(rng.choice(X.N_V, X.K_V, replace=False)).astype(np.int32) + O_V
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
        ivrng = np.random.default_rng(seed_probe + 1000)
        kinds = kinds or ['u', 'u', 'pS', 'u', 'pV', 'rB', 'u', 'pD', 'rS']
        self.probes = []
        self.extra = {}                       # tick -> AUD units of a stepped pause moment (inner voice or empty)
        self.pause = pause
        self.n_pm = pause // SYL              # pause moments (the line end + the silent moments)
        pc = prng.choice(len(chars), 2 * n_pool, replace=False)
        self.pool = [chars[pc[2 * i]] + chars[pc[2 * i + 1]] for i in range(n_pool)]
        self.pool_of = {chars[pc[j]]: j // 2 for j in range(2 * n_pool)}
        q_chars = [c for c in chars if c not in self.pool_of]
        starts = [u[0] for u in utt]
        ends = [u[0] + SYL * len(u[2]) for u in utt]
        gaps = [(ends[i], starts[i + 1]) for i in range(len(utt) - 1)]
        next_t, ki = 1500, 0
        pend = []
        dur = SYL * LP
        fpend = sorted(facts or [], key=lambda f: f[0])          # (t_min, pet, name)
        self.facts = []

        def eye_at(t):
            i = np.searchsorted(self.eye_t, t, side='right') - 1
            return self.eye_v[i] if i >= 0 else np.zeros(0, np.int32)

        def quiet(t0, t1):
            i = np.searchsorted(self.eye_change, t0, side='left')
            return i >= len(self.eye_change) or self.eye_change[i] > t1

        def new_text(k=LP, avoid='', pool=None):
            pool = [c for c in (pool or chars) if c not in avoid]
            return ''.join(prng.choice(pool, k, replace=False))

        qa_dur = 3 * (dur + QA_PAUSE_MAX + 2 * SYL + 15)           # same schedule whatever the pause
        for g0, g1 in gaps:
            t = g0 + 15
            if fpend and fpend[0][0] <= t and g1 - g0 >= 3 * 7 + 60 and len(eye_at(t)) and quiet(t, t + 30):
                _, pet, name = fpend.pop(0)
                text = '我的' + pet + '叫' + name
                P = dict(kind='fact', text=text, pet=pet, name=name, t=[t], who_t=['other'], pair=-1)
                pid = len(self.probes); self.probes.append(P); self.facts.append(pid)
                utt.append((t, 'other', text, 'probe', pid))
                continue
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
            kind = kinds[ki % len(kinds)]
            if kind == 'qa':
                if g1 - g0 < qa_dur + 40 or not quiet(t, t + qa_dur + 20):
                    continue                                        # wait for a long enough quiet gap
                ki += 1; next_t = t + probe_every
                tt, mates = t, []
                ans3 = prng.choice(n_pool, 3, replace=False)
                for j in range(3):
                    q = new_text(pool=q_chars)
                    a = self.pool[int(ans3[j])]
                    te = tt + SYL * LP
                    tA = te + pause
                    U = np.unique(self.asm[[self.cix[c] for c in q]].ravel())
                    perm = ivrng.permutation(U)
                    iv = np.sort(perm[:int(round(iv_frac * len(U)))]).astype(np.int32)
                    P = dict(kind='qa', text=q, ans=a, ans_i=int(ans3[j]), t=[tt], who_t=['other'], pair=-1, tA=tA,
                             iv=iv, te=te)
                    pid = len(self.probes); self.probes.append(P); mates.append(pid)
                    utt.append((tt, 'other', q, 'probe', pid))
                    utt.append((tA, 'other', a, 'ans', pid))
                    if step_pause:
                        for m in range(self.n_pm):
                            self.extra[te + SYL * m] = (iv, pid, m)
                    tt = tA + SYL * 2 + 15
                for pid in mates:
                    self.probes[pid]['mates'] = [self.probes[p]['ans_i'] for p in mates if p != pid]
                continue
            ki += 1; next_t = t + probe_every
            if kind in ('u', 'rB', 'rS'):
                P = dict(kind=kind, text=new_text(), t=[t], who_t=['other'], pair=-1)
                pid = len(self.probes); self.probes.append(P)
                utt.append((t, 'other', P['text'], 'probe', pid))
                if kind == 'rB':
                    for r in (1, 2):
                        tr = t + (dur + 6) * r; P['t'].append(tr); P['who_t'].append('other')
                        utt.append((tr, 'other', P['text'], 'probe', pid))
                if kind == 'rS':
                    for r in range(1, n_spread):
                        pend.append((t + spread_gap * r, pid, 'other', False))
                    pend.sort(key=lambda x: x[0])
            else:
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
                else:
                    pend.append((t + 100, ia + 1, 'other', True)); pend.sort(key=lambda x: x[0])
        utt.sort(key=lambda u: u[0])
        self.utt = utt
        eye_on = np.zeros(N_E, bool)
        for v in self.eye_v:
            eye_on[v - O_E] = True
        self.feat = np.unique(np.concatenate([self.asm.ravel(), self.voice['other'], self.voice['other2'],
                                              np.flatnonzero(eye_on) + O_E])).astype(np.int32)
        self.n_sp = int(0.05 * len(self.feat))
        self.pert = None

    def changes(self):
        """yield (t, Hc active set, info) at every tick where Hc changes, and at every stepped pause moment.
        info = (char, who, pid, pos): pos < 100 probe line char, 100+i answer char, 200+m pause moment"""
        ev_t = set(self.eye_t.tolist())
        aud = {}
        for t0, who, text, src, pid in self.utt:
            for i, ch in enumerate(text):
                aud[t0 + SYL * i] = (self.cix[ch], who, pid, i + (100 if src == 'ans' else 0))
            aud.setdefault(t0 + SYL * len(text), None)
        times = sorted(ev_t | set(aud.keys()) | set(self.extra.keys()))
        ei = -1; cur_eye = np.zeros(0, np.int32); cur_a = None
        for t in times:
            if t >= self.t_end:
                break
            while ei + 1 < len(self.eye_t) and self.eye_t[ei + 1] <= t:
                ei += 1; cur_eye = self.eye_v[ei]
            if t in aud:
                cur_a = aud[t]
            parts = [cur_eye]
            info = cur_a
            if cur_a is not None:
                c, who, pid, pos = cur_a
                parts = [self.asm[c], self.voice[who], cur_eye]
            elif t in self.extra:
                iv, pid, m = self.extra[t]
                parts = [iv, cur_eye]
                info = (None, None, pid, 200 + m)
            yield t, self.apply(np.unique(np.concatenate(parts)).astype(np.int32), t), info

    def line_moments(self, text, who, eye):
        return [np.unique(np.concatenate([self.asm[self.cix[ch]]] + ([self.voice[who]] if who else []) + [eye]))
                .astype(np.int32) for ch in text]


class Links6(X.Links):
    """X3 links; with track = True the two-layer sums S_out, S_in are kept current at every learning step
    (ARCHITECTURE / HARVEST 1.6: "two-layer weights computed every tick from the current totals"). X3 recomputed
    them every 200 updates, which only mattered at recall (lam = 0 while listening). A full recompute every
    norm_every updates still removes the drift from lazily pruned links."""
    track = False

    def _eff(self, rows):
        w = self.v[rows] * self.scale
        w[w < self.c['w_min']] = 0.0
        return w

    def _upd(self, pre, old):
        d = self._eff(pre) - old
        self.s_out[pre] += d.sum(1)
        self.s_in += np.bincount(self.post[pre].ravel(), weights=d.ravel(), minlength=self.n_post).astype(np.float32)
        np.maximum(self.s_in, 1e-6, out=self.s_in); np.maximum(self.s_out, 1e-6, out=self.s_out)

    def learn(self, pre, post_mask, eta):
        if not self.track or len(pre) == 0 or eta == 0:
            return super().learn(pre, post_mask, eta)
        old = self._eff(pre)
        super().learn(pre, post_mask, eta)
        self._upd(pre, old)

    def depress(self, pre, post_mask, eta):
        if not self.track or len(pre) == 0 or eta == 0:
            return super().depress(pre, post_mask, eta)
        old = self._eff(pre)
        super().depress(pre, post_mask, eta)
        self._upd(pre, old)

    def decay(self, n):
        super().decay(n)
        if self.track and n > 0:
            f = np.float32(self.c['gamma'] ** n)
            self.s_out *= f; self.s_in *= f


class Hippo6(X.Hippo):
    def __init__(self, cfg, seed=3, feat=None, n_sp=0):
        super().__init__(cfg, seed, feat, n_sp)
        self.fall_t = np.full(self.n, -10 ** 9, np.int64)
        self.rec.__class__ = Links6
        self.rec.track = bool(cfg['lam']) and not cfg.get('stale', False)
        if cfg['lam']:
            self.rec.prop_norm()

    # one update rule; X3 = beta 1, lam 0 while listening
    def encode(self, hc, prev, thr, lam=None):
        lam = self.c['lam'] if lam is None else lam
        d = self.c['beta'] * self.birth.drive(hc)
        if len(prev):
            if self.brec is not None:
                r = self.brec.drive(prev)
                d = d + self.c['rho'] * (r - r.mean())
            if lam and getattr(self.rec, 's_out', None) is not None:
                d = d + lam * self.rec.drive_prop(prev)
        return thr.fire(d, self.rng)

    def step(self, t, hc):
        nd = t // self.c['T'] - self.t_dec
        if nd > 0:
            self.rec.decay(nd); self.out.decay(nd); self.birth.decay(nd)
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
        if self.c['win'] > 2:
            self.fall_t[falling] = t
            on = np.zeros(self.n, bool); on[x] = True
            pre = np.flatnonzero((self.fall_t >= t - self.c['win']) & ~on)
            self.rec.learn(pre, rm, self.c['eta_seq'])
        else:
            self.rec.learn(falling, rm, self.c['eta_seq'])
        self.rec.learn(rising, rm, self.c['eta_co'])
        if self.c.get('held') and len(rising):
            # ARCHITECTURE 2.8 / HARVEST 1.1: a unit that stays on while B rises counts once, as "A before B"
            held = np.intersect1d(self.prev, x, assume_unique=True)
            self.rec.learn(held, rm, self.c['eta_seq'])
        hr = np.setdiff1d(hc, self.prev_hc, assume_unique=True)
        hr = hr[hr < N_A + N_V]
        if len(hr):
            hm = np.zeros(N_A + N_V, bool); hm[hr] = True
            self.out.learn(rising, hm, self.c['eta_out'])
        self.prev = x; self.prev_hc = hc
        return x

    def new_thr(self):
        return Thr(self.thr.f, self.thr.a, self.thr.theta, self.thr.ceil)

    def present6(self, eye, moments, frac, rng, th=None, start=None, init=None):
        """a cue heard through the chain copies. start 'now' (default): the hippocampus's own state at the test time
        (usually the small post-line state); 'x3': X3's start, one eye-only update from an empty state without the
        recurrent gain control, which turns on about 2k units and sends the replayed codes down another path"""
        def sub(s):
            return np.sort(rng.choice(s, max(1, int(round(frac * len(s)))), replace=False)) if len(s) else s
        th = th or self.new_thr()
        start = start or self.c.get('start', 'now')
        x = self.encode(sub(eye), np.zeros(0, np.int64), th) if start == 'x3' else self.prev
        if init is not None:                       # ideal re-hearing: the exact state and threshold of the encoding
            x = init[0]; th.theta = init[1]
        for m in moments:
            x = self.encode(sub(m), x, th)
        return x, th

    def run_free(self, x, steps, th=None, quiet=None):
        """X3 recall (learned links only) or, if unified, the same update as listening with the silent input"""
        if not self.c['unified']:
            return self.free(x, steps)
        th = th or self.new_thr()
        quiet = np.zeros(0, np.int32) if quiet is None else quiet
        xs = []
        lam = self.c.get('lam_rec')
        for _ in range(steps):
            x = self.encode(quiet, x, th, lam=lam)
            xs.append(x)
        return xs


# ----------------------------------------------------------------------------------------------- tests
def qk(d, k):
    return float(np.partition(d, len(d) - k)[len(d) - k]) if len(d) >= k else 0.0


def recall6(H, W, P, frac, rng, mode, quiet_kind, t_now, start=None):
    steps = LP - NCUE
    eye = W.eye_at(P['t'][-1])
    quiet = {'eye': eye, 'dark': np.zeros(0, np.int32), 'now': W.eye_at(t_now)}[quiet_kind]
    th = H.new_thr()
    if mode == 'store':
        code = P['code'][NCUE - 1]
        x = np.sort(rng.choice(code, max(1, int(round(frac * len(code)))), replace=False)) if len(code) else code
    else:
        ms = W.line_moments(P['text'][:NCUE], P['who_t'][-1], eye)
        x, th = H.present6(eye, ms, frac, rng, th, start=start)
    xs = H.run_free(x, steps, th, quiet)
    auds = [H.aud_of(z) for z in xs]
    got = [W.decode(a) for a in auds]
    tgt = [W.cix[c] for c in P['text'][NCUE:NCUE + steps]]
    fid = float(np.mean([W.fidelity(a, c) for a, c in zip(auds, tgt)]))
    hs = float(np.mean([len(np.intersect1d(z, P['code'][NCUE + j])) / max(1, len(P['code'][NCUE + j]))
                        for j, z in enumerate(xs)]))
    H.last_got = got
    return float(np.mean([g == t for g, t in zip(got, tgt)])), fid, hs


def test_all6(H, W, t_now, rng, fracs=(1.0, 0.5), in_fracs=(1.0,), n_recent=30, n_old=30, quiets=('eye',),
              kinds=('u', 'rB', 'rS', 'pS', 'pV', 'pD'), min_age=0, max_age=None, x3start=True):
    res = []
    H.prepare_recall()
    probes = [(i, P) for i, P in enumerate(W.probes) if P['kind'] in kinds and P['t'] and
              P['t'][-1] + SYL * LP < t_now and 'code' in P and all(c is not None for c in P['code'])]
    old = probes[:-n_recent] if n_recent else probes
    if len(old) > n_old:
        old = [old[j] for j in sorted(rng.choice(len(old), n_old, replace=False))]
    elif n_old == 0:
        old = []
    for i, P in old + (probes[-n_recent:] if n_recent else []):
        age = t_now - (P['t'][-1] + SYL * LP)
        if age < min_age or (max_age is not None and age > max_age):
            continue
        row = dict(i=i, kind=P['kind'], age=int(age), n_exp=len(P['t']), t_test=int(t_now))
        for q in quiets:
            sfx = '' if q == 'eye' else q[0]
            for f in fracs:
                r, fid, hs = recall6(H, W, P, f, rng, 'store', q, t_now)
                row[f'r{int(f * 100)}{sfx}'] = r; row[f'fid{int(f * 100)}{sfx}'] = fid; row[f'hs{int(f * 100)}{sfx}'] = hs
                if P['pair'] >= 0 and W.probes[P['pair']]['t'] and f == 0.5 and q == 'eye':
                    Q = W.probes[P['pair']]; g0 = H.last_got[0]
                    row['sep_right'] = float(g0 == W.cix[P['text'][NCUE]])
                    row['sep_other'] = float(g0 == W.cix[Q['text'][NCUE]])
            for f in in_fracs:
                r, fid, hs = recall6(H, W, P, f, rng, 'input', q, t_now)
                row[f'i{int(f * 100)}{sfx}'] = r
                if x3start and q == 'eye':
                    row[f'i{int(f * 100)}x'] = recall6(H, W, P, f, rng, 'input', q, t_now, start='x3')[0]
        res.append(row)
    return res


def first_pool(W, decs):
    for s, d in enumerate(decs):
        if d is not None and W.chars[d] in W.pool_of:
            return W.pool_of[W.chars[d]], s
    return -1, -1


def test_qa(H, W, t_now, rng, age_lo=500, age_hi=12000):
    H.prepare_recall()
    out = []
    for i, P in enumerate(W.probes):
        if P['kind'] != 'qa' or 'code' not in P:
            continue
        age = t_now - (P['tA'] + 2 * SYL)
        if not (age_lo <= age <= age_hi):
            continue
        eye = W.eye_at(P['t'][0])
        ms = W.line_moments(P['text'], 'other', eye)
        row = dict(i=i, age=int(age), t_test=int(t_now), ans=P['ans_i'], mates=P['mates'])
        pin = np.unique(np.concatenate([P['iv'], eye])).astype(np.int32)
        code = P['code'][-1]
        for cue in ('hear', 'store', 'ideal'):
            for cont in ('F', 'P'):
                # cue: re-hear Q through the copies, or 50 % of Q's stored last-char code (X3's r50 form), or an
                # ideal re-hearing (from the exact encoding-time state and threshold: isolates storage from replay)
                if cue == 'hear':
                    x, th = H.present6(eye, ms, 1.0, rng)
                elif cue == 'ideal':
                    x, th = H.present6(eye, ms, 1.0, rng, init=P['enc'])
                else:
                    x = np.sort(rng.choice(code, max(1, len(code) // 2), replace=False)); th = H.new_thr()
                # continuation F: the hippocampus runs on its own; P: the pause input is there again (the scene,
                # plus the inner voice in the inner-voice arms), then it runs on its own
                if cont == 'P':
                    for _ in range(W.n_pm):
                        x = H.encode(pin, x, th)
                    xs = H.run_free(x, 4, th, eye)
                else:
                    xs = H.run_free(x, W.n_pm + 4, th, eye)
                dec = [W.decode(H.aud_of(z)) for z in xs]
                a, st = first_pool(W, dec)
                tag = {'hear': '', 'store': 'S', 'ideal': 'I'}[cue] + cont
                row.update({tag + '_first': a, tag + '_step': st})
                if 'acode' in P and P['acode'][0] is not None:
                    ac = P['acode'][0]
                    row[tag + '_hsA'] = float(max(len(np.intersect1d(z, ac)) / max(1, len(ac)) for z in xs))
        out.append(row)
    return out


def test_fact(H, W, t_now, rng, P, delay):
    H.prepare_recall()
    eye_now, eye_f = W.eye_at(t_now), W.eye_at(P['t'][0])
    n1 = W.cix[P['name'][0]]
    row = dict(pet=P['pet'], name=P['name'], delay=int(delay), t_test=int(t_now))
    for tag, eye, cue, steps in (('R1', eye_now, '我的' + P['pet'] + '叫', 2), ('R1s', eye_f, '我的' + P['pet'] + '叫', 2),
                                 ('R2', eye_now, '我的' + P['pet'] + '叫什么', 4)):
        ms = W.line_moments(cue, 'other', eye)
        x, th = H.present6(eye, ms, 1.0, rng)
        xs = H.run_free(x, steps, th, eye)
        dec = [W.decode(H.aud_of(z)) for z in xs]
        row[tag] = [-1 if d is None else int(d) for d in dec]
        row[tag + '_ok'] = int(dec[0] == n1) if tag != 'R2' else int(n1 in dec)
        row[tag + '_full'] = int(dec[0] == n1 and dec[1] == n1)
    # ideal re-hearing of 我的X叫 (exact encoding-time state and threshold, the fact's scene)
    x, th = H.present6(eye_f, W.line_moments('我的' + P['pet'] + '叫', 'other', eye_f), 1.0, rng, init=P['enc'])
    dec = [W.decode(H.aud_of(z)) for z in H.run_free(x, 2, th, eye_f)]
    row['I1'] = [-1 if d is None else int(d) for d in dec]; row['I1_ok'] = int(dec[0] == n1)
    row['I1_full'] = int(dec[0] == n1 and dec[1] == n1)
    # storage diagnostic: 50 % of the stored code of the 叫 moment (X3's r50 form), 2 free steps
    code = P['code'][4]
    x = np.sort(rng.choice(code, max(1, len(code) // 2), replace=False))
    dec = [W.decode(H.aud_of(z)) for z in H.run_free(x, 2, H.new_thr(), eye_f)]
    row['S50'] = [-1 if d is None else int(d) for d in dec]
    row['S50_ok'] = int(dec[0] == n1); row['S50_full'] = int(dec[0] == n1 and dec[1] == n1)
    return row


def heard_sets(W, t_now):
    within, cross = set(), set()
    prev = None
    for t0, who, text, src, pid in W.utt:
        t1 = t0 + SYL * len(text)
        if t1 > t_now:
            break
        ids = [W.cix[c] for c in text]
        within |= set(zip(ids, ids[1:]))
        if prev is not None and t0 - prev[0] < 30:
            cross.add((prev[1], ids[0]))
        prev = (t1, ids[-1])
    return within | cross


def kicks6(H, W, t_now, rng, n=100, steps=4, recent=3000):
    H.prepare_recall()
    heard = heard_sets(W, t_now)
    lines = [u for u in W.utt if t_now - recent <= u[0] and u[0] + SYL * len(u[2]) <= t_now and len(u[2]) >= 3]
    eye = W.eye_at(t_now)
    out = {a: dict(n=0, silent=0, single=0, real=0, chimera=0, anyheard=0, wrong=0, wrong_in=0, wrong_tc=0, right=0)
           for a in ('random', 'cued')}
    if not lines:
        return out
    for _ in range(n):
        u = lines[int(rng.integers(len(lines)))]
        T = u[2][:2]; tlast = W.cix[T[-1]]
        for arm in ('random', 'cued'):
            if arm == 'random':
                x0 = np.sort(rng.choice(H.n, H.c['k'], replace=False))
                xs = H.run_free(x0, steps, None, eye)
            else:
                ms = W.line_moments(T, None, eye)
                x, th = H.present6(eye, ms, 1.0, rng)
                xs = H.run_free(x, steps, th, eye)
            dec = [W.decode(H.aud_of(z)) for z in xs]
            seq = [d for d in dec if d is not None]
            o = out[arm]; o['n'] += 1
            if not seq:
                o['silent'] += 1; continue
            pairs = list(zip(seq, seq[1:]))
            bad = sum(p not in heard for p in pairs)
            tc_bad = int((tlast, seq[0]) not in heard)
            if not pairs:
                o['single'] += 1
            elif bad:
                o['chimera'] += 1
            else:
                o['real'] += 1
            if any(p in heard for p in pairs):
                o['anyheard'] += 1
            o['wrong_in'] += bad; o['wrong_tc'] += tc_bad; o['wrong'] += bad + tc_bad
            if arm == 'cued' and dec[0] is not None and dec[0] == W.cix[u[2][2]]:
                o['right'] += 1
    return out


def drive_scales(H, W, hc_speech, P):
    """for item 18: k-th largest chain drive in speech and k-th largest learned (prop) drive from a stored code"""
    H.rec.prop_norm()
    k = H.c['k']
    qb = qk(H.c['beta'] * H.birth.drive(hc_speech), k)
    qr = [qk(H.rec.drive_prop(P['code'][j]), k) for j in range(LP - 1)]
    return qb, float(np.mean(qr))


# ----------------------------------------------------------------------------------------------- run
_EV = {}

BASE6 = dict(X.BASE, start='now', held=False, beta=1.0, unified=False, win=0, pause=12, iv=0.0, step_pause=True, probe_every=1500,
             kinds=['u'])


def run6(cfg, t_end, seed=0, test_every=10000, tests=('all',), test_kw=None, kick_at=(), fact_plan=None,
         fact_delays=(1000, 2000, 5000, 10000), scale_at=(), verbose=True, log=None):
    t0c = time.time()
    if 'ev' not in _EV:
        _EV['ev'] = X.load_stream()
    ev, _ = _EV['ev']
    W = World6(ev, t_end, probe_every=cfg['probe_every'], kinds=cfg['kinds'], seed_probe=11 + seed,
               pause=cfg['pause'], iv_frac=cfg['iv'], step_pause=cfg['step_pause'], facts=fact_plan)
    rng = np.random.default_rng(100 + seed)
    H = Hippo6(cfg, seed=3 + seed, feat=W.feat, n_sp=W.n_sp)
    sched = []
    for t in range(test_every, t_end + 1, test_every):
        for kind in tests:
            sched.append((t, kind, None))
    for t in kick_at:
        sched.append((t, 'kick', None))
    for t in scale_at:
        sched.append((t, 'scale', None))
    for pid in W.facts:
        for d in fact_delays:
            sched.append((W.probes[pid]['t'][0] + SYL * 7 + d, 'fact', (pid, d)))
    sched.sort(key=lambda s: (s[0], s[1]))
    res = dict(all=[], qa=[], kick=[], fact=[], scale=[])
    tk = test_kw or {}
    last_speech = None

    def do(t, kind, arg):
        if kind == 'all':
            res['all'] += test_all6(H, W, t, rng, **tk.get('all', {}))
        elif kind == 'qa':
            res['qa'] += test_qa(H, W, t, rng)
        elif kind == 'kick':
            res['kick'].append(dict(t=int(t), **kicks6(H, W, t, rng, **tk.get('kick', {}))))
        elif kind == 'fact':
            res['fact'].append(test_fact(H, W, t, rng, W.probes[arg[0]], arg[1]))
        elif kind == 'scale':
            Ps = [P for P in W.probes if P['kind'] == 'u' and 'code' in P and all(c is not None for c in P['code'])
                  and P['t'][-1] < t]
            if Ps and last_speech is not None:
                qb, qr = zip(*[drive_scales(H, W, last_speech, P) for P in Ps[-10:]])
                res['scale'].append(dict(t=int(t), q_birth=float(np.mean(qb)), q_rec=float(np.mean(qr))))
        if log:
            log(f'  test {kind} t={t} {time.time() - t0c:.0f}s')

    si = n_ch = 0
    for t, hc, info in W.changes():
        while si < len(sched) and t >= sched[si][0]:
            do(*sched[si]); si += 1
        if info is not None and info[2] >= 0 and info[3] == 0:
            W.probes[info[2]]['enc'] = (H.prev.copy(), H.thr.theta)
        x = H.step(t, hc); n_ch += 1
        if info is not None and info[2] >= 0:
            P = W.probes[info[2]]; pos = info[3]
            if pos < 100:
                P.setdefault('code', [None] * len(P['text']))[pos] = x
                last_speech = hc
            elif pos < 200:
                P.setdefault('acode', [None] * len(P['ans']))[pos - 100] = x
            else:
                P.setdefault('pcode', [None] * W.n_pm)[pos - 200] = x
        elif info is not None:
            last_speech = hc
    while si < len(sched):
        do(*sched[si]); si += 1
    out = dict(res, n_probes=len(W.probes), live=H.rec.n_live(), secs=round(time.time() - t0c, 1), n_change=n_ch,
               rss_mb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024),
               pool=W.pool, chars=W.chars, n_qa=sum(P['kind'] == 'qa' for P in W.probes),
               facts=[dict(pet=W.probes[p]['pet'], name=W.probes[p]['name'], t=W.probes[p]['t'][0]) for p in W.facts])
    if verbose:
        print(f'   {n_ch} change ticks, {len(W.probes)} probes, live {out["live"]}, {out["secs"]} s, '
              f'RSS {out["rss_mb"]} MB', flush=True)
    return out
