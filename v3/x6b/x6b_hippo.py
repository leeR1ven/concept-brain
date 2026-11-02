"""X6b: an innate listening signal (basal-forebrain acetylcholine analogue) on the X6 / X3 hippocampus.

x3_hippo.py (v3/x345) and x6_hippo.py (v3/x6) are imported unchanged. With ls = None this is X6's harness.

The listening population ("LS"): M = 64 binary units, birth-wired, never learning. Each reads c_in = 300 random units
of the auditory INPUT layer (the externally heard char assembly + the speaker's voice units). A unit is on when at
least one of its inputs is on. L = the fraction of LS units on: graded with how much is heard (one heard char:
L ~ 0.98; nothing heard: 0), content-free, no timer. Like the innate brake (ARCHITECTURE 2.7) it reads only the input
layer, so the inner voice (return chains -> AUD feature units) does not turn it on.

Its action (cfg 'ls' = f, the suppression factor): the hippocampus's recurrent drive -- the birth recurrence AND the
learned links -- is multiplied by s = 1 - (1 - f) * L. While someone speaks nearby s ~ f; in silence s = 1.
One update rule for listening, silence and recall (no store / recall switch):
    d = birth(input) + s * [ rho * (brec(prev) - mean) + lam * rec_prop(prev) ]
"""
import os
import sys
import time
import resource

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
for _x in ('x6', 'x345'):   # repo: x6_hippo.py from ../x6, x3_hippo.py (and its stream.json) from ../x345
    _p = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', _x)
    if _p not in sys.path:
        sys.path.append(_p)
import x3_hippo as X  # noqa: E402
import x6_hippo as H6  # noqa: E402
from x3_hippo import N_A, N_V, O_V, O_E, SYL, LP, NCUE  # noqa: E402,F401

N_AUD_IN = N_A + N_V


class World6b(H6.World6):
    def changes(self):
        """World6.changes plus ext = the externally heard auditory input units (empty in silence and in the inner-voice
        pause moments: return chains do not reach the input layer)"""
        ev_t = set(self.eye_t.tolist())
        aud = {}
        for t0, who, text, src, pid in self.utt:
            for i, ch in enumerate(text):
                aud[t0 + SYL * i] = (self.cix[ch], who, pid, i + (100 if src == 'ans' else 0))
            aud.setdefault(t0 + SYL * len(text), None)
        times = sorted(ev_t | set(aud.keys()) | set(self.extra.keys()))
        ei = -1; cur_eye = np.zeros(0, np.int32); cur_a = None
        empty = np.zeros(0, np.int32)
        for t in times:
            if t >= self.t_end:
                break
            while ei + 1 < len(self.eye_t) and self.eye_t[ei + 1] <= t:
                ei += 1; cur_eye = self.eye_v[ei]
            if t in aud:
                cur_a = aud[t]
            parts = [cur_eye]
            info = cur_a
            ext = empty
            if cur_a is not None:
                c, who, pid, pos = cur_a
                ext = np.concatenate([self.asm[c], self.voice[who]]).astype(np.int32)
                parts = [self.asm[c], self.voice[who], cur_eye]
            elif t in self.extra:
                iv, pid, m = self.extra[t]
                parts = [iv, cur_eye]
                info = (None, None, pid, 200 + m)
            yield t, self.apply(np.unique(np.concatenate(parts)).astype(np.int32), t), info, ext


class Listen:
    """the innate listening population (birth-wired, no learning)"""

    def __init__(self, rng, M=64, c_in=300):
        self.inp = np.zeros((M, N_AUD_IN), bool)
        for i in range(M):
            self.inp[i, rng.choice(N_AUD_IN, c_in, replace=False)] = True

    def level(self, ext):
        if len(ext) == 0:
            return 0.0
        return float(self.inp[:, ext].any(1).mean())


class Hippo6b(H6.Hippo6):
    def __init__(self, cfg, seed=3, feat=None, n_sp=0):
        super().__init__(cfg, seed, feat, n_sp)
        self.ls = Listen(np.random.default_rng(seed + 7)) if cfg.get('ls') is not None else None
        self._L = 0.0
        self.stat = dict(n_sp=0, L_sp=0.0, on_sp=0, n_si=0, on_si=0)

    def L_of(self, ext):
        return self.ls.level(ext) if self.ls is not None else 0.0

    def gain(self, L):
        f = self.c.get('ls')
        return 1.0 if f is None else 1.0 - (1.0 - f) * L

    def encode(self, hc, prev, thr, lam=None, L=None):
        lam = self.c['lam'] if lam is None else lam
        L = self._L if L is None else L
        s = self.gain(L)
        d = self.c['beta'] * self.birth.drive(hc)
        if len(prev):
            if self.brec is not None:
                r = self.brec.drive(prev)
                d = d + s * self.c['rho'] * (r - r.mean())
            if lam and getattr(self.rec, 's_out', None) is not None:
                d = d + s * lam * self.rec.drive_prop(prev)
        return thr.fire(d, self.rng)

    def step(self, t, hc, L=0.0, heard=False):
        self._L = L
        try:
            x = super().step(t, hc)
        finally:
            self._L = 0.0
        st = self.stat
        if heard:
            st['n_sp'] += 1; st['L_sp'] += L; st['on_sp'] += len(x)
        else:
            st['n_si'] += 1; st['on_si'] += len(x)
        return x

    def present6(self, eye, moments, frac, rng, th=None, start=None, init=None, heard=True):
        """a heard cue through the chain copies; the listening population sees the cue's auditory input units"""
        def sub(s):
            return np.sort(rng.choice(s, max(1, int(round(frac * len(s)))), replace=False)) if len(s) else s
        th = th or self.new_thr()
        start = start or self.c.get('start', 'now')
        x = self.encode(sub(eye), np.zeros(0, np.int64), th) if start == 'x3' else self.prev
        if init is not None:
            x = init[0]; th.theta = init[1]
        for m in moments:
            mm = sub(m)
            x = self.encode(mm, x, th, L=self.L_of(mm[mm < N_AUD_IN]) if heard else 0.0)
        return x, th


# ----------------------------------------------------------------------------------------------- extra tests
def ov(a, b):
    """overlap of a re-heard code a with the stored code b: shared units / the larger of the two sets"""
    return len(np.intersect1d(a, b)) / max(1, len(a), len(b))


def rehear(H, W, text, who, eye, codes, other=None, xs_out=None):
    """re-hear a whole line from the hippocampus's current state (no learning); overlap with the stored code per char.
    other: the stored codes of a different line (specificity control: a locked-in network gives high overlap with
    everything). The hippocampus's own random generator (tie noise) is saved and restored, so the life continues
    exactly as without this test."""
    st = H.rng.bit_generator.state
    x, th = H.prev, H.new_thr()
    out, oth = [], []
    for j, m in enumerate(W.line_moments(text, who, eye)):
        x = H.encode(m, x, th, L=H.L_of(m[m < N_AUD_IN]))
        out.append(round(ov(x, codes[j]), 3))
        if other is not None and j < len(other):
            oth.append(round(ov(x, other[j]), 3))
    H.rng.bit_generator.state = st
    return (out, oth) if other is not None else out


def test_all6b(H, W, t_now, rng, **kw):
    res = H6.test_all6(H, W, t_now, rng, **kw)
    eye_now = W.eye_at(t_now)
    prevP = None
    for row in res:
        P = W.probes[row['i']]
        eye_own = W.eye_at(P['t'][-1])
        if prevP is not None:                                                           # same scene + control
            row['ov'], row['ovx'] = rehear(H, W, P['text'], P['who_t'][-1], eye_own, P['code'], other=prevP['code'])
        else:
            row['ov'] = rehear(H, W, P['text'], P['who_t'][-1], eye_own, P['code'])
        prevP = P
        row['ovn'] = rehear(H, W, P['text'], P['who_t'][-1], eye_now, P['code'])         # the scene at test time
        # diagnostics of the stored codes: persistence = overlap of consecutive chars' codes (units that do not change
        # do not learn under the rule); readout = the stored code's return to AUD decodes to its own char
        cs = P['code']
        row['persist'] = round(float(np.mean([ov(a, b) for a, b in zip(cs, cs[1:])])), 3)
        st = H.rng.bit_generator.state
        row['rd'] = round(float(np.mean([W.decode(H.aud_of(c)) == W.cix[ch] for c, ch in zip(cs, P['text'])])), 3)
        if H.c['unified'] and len(cs[NCUE - 1]):
            # diagnostic: the same r50 cue, recalled with X3's dynamics (learned links only, no input, no birth
            # recurrence): separates what was stored from the no-switch recall dynamics
            rr = np.random.default_rng(1000003 * int(t_now) + row['i'])
            c3 = cs[NCUE - 1]
            xs = H.free(np.sort(rr.choice(c3, max(1, len(c3) // 2), replace=False)), LP - NCUE)
            got = [W.decode(H.aud_of(z)) for z in xs]
            row['r50x3'] = float(np.mean([g == W.cix[ch] for g, ch in zip(got, P['text'][NCUE:])]))
        H.rng.bit_generator.state = st
        row['same_scene'] = int(np.array_equal(eye_own, eye_now))
    return res


def test_qa_b(H, W, t_now, rng):
    res = H6.test_qa(H, W, t_now, rng)
    for row in res:
        P = W.probes[row['i']]
        row['ovQ'] = rehear(H, W, P['text'], 'other', W.eye_at(P['t'][0]), P['code'])
    return res


def test_fact_b(H, W, t_now, rng, P, delay):
    if len(P['code'][4]) == 0:                       # guard: X6's S50 diagnostic needs a non-empty stored code
        P = dict(P, code=list(P['code'])); P['code'][4] = np.zeros(1, np.int64)
    row = H6.test_fact(H, W, t_now, rng, P, delay)
    eye_f, eye_now = W.eye_at(P['t'][0]), W.eye_at(t_now)
    row['ov7'] = rehear(H, W, P['text'], 'other', eye_f, P['code'])                      # whole line, same scene
    row['ov7n'] = rehear(H, W, P['text'], 'other', eye_now, P['code'])                   # whole line, test scene
    row['same_scene'] = int(np.array_equal(eye_f, eye_now))
    return row


# ----------------------------------------------------------------------------------------------- run
_EV = {}
BASE6b = dict(H6.BASE6, ls=None)


def run6b(cfg, t_end, seed=0, test_every=10000, tests=('all',), test_kw=None, kick_at=(), fact_plan=None,
          fact_delays=(1000, 2000, 5000, 10000), scale_at=(), verbose=True, log=None):
    t0c = time.time()
    if 'ev' not in _EV:
        _EV['ev'] = X.load_stream()
    ev, _ = _EV['ev']
    W = World6b(ev, t_end, probe_every=cfg['probe_every'], kinds=cfg['kinds'], seed_probe=11 + seed,
                pause=cfg['pause'], iv_frac=cfg['iv'], step_pause=cfg['step_pause'], facts=fact_plan)
    rng = np.random.default_rng(100 + seed)
    H = Hippo6b(cfg, seed=3 + seed, feat=W.feat, n_sp=W.n_sp)
    sched = []
    for t in range(test_every, t_end + 1, test_every):
        for kind in tests:
            sched.append((t, kind, None))
    for t in scale_at:
        sched.append((t, 'scale', None))
    for pid in W.facts:
        for d in fact_delays:
            sched.append((W.probes[pid]['t'][0] + SYL * 7 + d, 'fact', (pid, d)))
    sched.sort(key=lambda s: (s[0], s[1]))
    res = dict(all=[], qa=[], kick=[], fact=[], scale=[])
    tk = test_kw or {}
    last = {'speech': None}

    def do(t, kind, arg):
        if kind == 'all':
            res['all'] += test_all6b(H, W, t, rng, **tk.get('all', {}))
        elif kind == 'qa':
            res['qa'] += test_qa_b(H, W, t, rng)
        elif kind == 'fact':
            res['fact'].append(test_fact_b(H, W, t, rng, W.probes[arg[0]], arg[1]))
        elif kind == 'scale':
            Ps = [P for P in W.probes if P['kind'] == 'u' and 'code' in P and all(c is not None for c in P['code'])
                  and P['t'][-1] < t]
            if Ps and last['speech'] is not None:
                qb, qr = zip(*[H6.drive_scales(H, W, last['speech'], P) for P in Ps[-10:]])
                res['scale'].append(dict(t=int(t), q_birth=float(np.mean(qb)), q_rec=float(np.mean(qr))))
        if log:
            log(f'  test {kind} t={t} {time.time() - t0c:.0f}s')

    si = n_ch = 0
    for t, hc, info, ext in W.changes():
        while si < len(sched) and t >= sched[si][0]:
            do(*sched[si]); si += 1
        if info is not None and info[2] >= 0 and info[3] == 0:
            W.probes[info[2]]['enc'] = (H.prev.copy(), H.thr.theta)
        x = H.step(t, hc, L=H.L_of(ext), heard=len(ext) > 0); n_ch += 1
        if info is not None and info[2] >= 0:
            P = W.probes[info[2]]; pos = info[3]
            if pos < 100:
                P.setdefault('code', [None] * len(P['text']))[pos] = x
                last['speech'] = hc
            elif pos < 200:
                P.setdefault('acode', [None] * len(P['ans']))[pos - 100] = x
            else:
                P.setdefault('pcode', [None] * W.n_pm)[pos - 200] = x
        elif info is not None:
            last['speech'] = hc
    while si < len(sched):
        do(*sched[si]); si += 1
    st = H.stat
    out = dict(res, n_probes=len(W.probes), live=H.rec.n_live(), secs=round(time.time() - t0c, 1), n_change=n_ch,
               rss_mb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024),
               pool=W.pool, chars=W.chars, n_qa=sum(P['kind'] == 'qa' for P in W.probes),
               facts=[dict(pet=W.probes[p]['pet'], name=W.probes[p]['name'], t=W.probes[p]['t'][0]) for p in W.facts],
               ls_stat=dict(L_speech=st['L_sp'] / max(1, st['n_sp']), on_speech=st['on_sp'] / max(1, st['n_sp']),
                            on_silence=st['on_si'] / max(1, st['n_si']), n_speech=st['n_sp'], n_silence=st['n_si']))
    if verbose:
        print(f'   {n_ch} change ticks, {len(W.probes)} probes, live {out["live"]}, {out["secs"]} s, '
              f'RSS {out["rss_mb"]} MB, {out["ls_stat"]}', flush=True)
    return out
