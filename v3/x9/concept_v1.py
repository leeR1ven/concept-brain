"""X9: the abstract concept area (ARCHITECTURE 2.2, open item 14) built from existing parts, offline module test.

Stream: the recorded bench Life stream (stream.json, x345) + the v2/meaning class dialogues (w2 frames with the bench's
own 12 objects) inserted into silent daytime gaps; held fillers never heard in the taste frame (see PLAN.md 0).
Sensory feature neurons (through 1:1 chain copies): AUD = X3's char assemblies (2,048 units, 24 per char, physics seed
7, X3's char order), EYE = the 1,024 world bits (only units that are ever on have copies).

The concept area (one population, binary units):
  birth   : local random fan-in, a few comparable inputs from AUD and from EYE (never decays), plus silent potential
            synapses (birth weight 0) that can grow; a born random recurrence (arm Tb / Tbl)
  drive   : two-layer weights, push = w / sqrt(out-total(pre) * in-total(post)) on the total (birth + learned) weight;
            d = ff(current input) + beta * rec(previous concept state)
  count   : X3's Thr (smoothed quantile, ceiling 2x) in life; the instant quantile at test
  rule    : the one rule (ARCH 2.8, 11-01 form) on every plastic link, at every update (state recomputed when the input
            changes: char onset, line end, eye change; X3's event-driven shortcut):
              B rises while A is on (rising / held / just off)  -> A->B += eta * g(w), g = 1 / (1 + w / w_ref) ** pow
              A rises after B was on                            -> A->B -= eta_rev * g(w)   (total weight >= 0)
              B falling changes nothing
  decay   : one global decay of the learned parts (gamma per T ticks), links below w_min deleted
Arms: B0 (beta 0), Tb (born random recurrence), Tl (learned temporal links), Tbl (both).
Run: python3 -I concept.py arm=Tbl seed=0 [key=value ...] [out=file.json]
"""
import json
import os
import sys
import time

import numpy as np

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.environ.get('BENCH', os.path.join(HERE, '..', '..', 'v2', 'bench'))   # repo: v2/bench
sys.path.insert(0, BENCH)
from world import Curriculum, OBJECTS, _rng  # noqa: E402

N_A, K_A, N_E = 2048, 24, 1024
N_IN = N_A + N_E
SYL = 3
FOODS = OBJECTS[:6]
TOYS = OBJECTS[6:]                       # 小狗 小猫 皮球 汽车 小鸟 鸭子
TOYS_HELDABLE = ['皮球', '汽车', '鸭子']    # toys sharing no char with another toy (no sound leak of class)
FRAMES = {
    'offer': (lambda x: '你要' + x + '吗', lambda x: '要' + x),
    'eat': (lambda x: '你吃' + x + '吗', lambda x: '吃' + x),
    'play': (lambda x: '你玩' + x + '吗', lambda x: '玩' + x),
    'taste': (lambda x: x + '好不好吃', lambda x: '好吃' if x in FOODS else '不能吃'),
}
ALLOWED = {'offer': OBJECTS, 'eat': FOODS, 'play': TOYS, 'taste': OBJECTS}

CFG = dict(arm='Tbl', seed=0, N=2000, f=0.06, alpha=0.3, ceil=2.0,
           cA=40, cE=6, w_lo=0.5, w_hi=1.0, pA=0.25, pE=0.25,           # birth fan-in and potential synapses
           cR=40, r_lo=0.5, r_hi=1.0, pR=0.25,                          # born recurrence and its potential partners
           beta=0.1, g_E=0.07,
           eta_ff=0.05, eta_ff_off=0.05, eta_ff_rev=0.05,               # feed-forward rates (on / just off / reverse)
           eta_seq=0.05, eta_co=0.02, eta_rev=0.05,                     # recurrent rates
           w_ref=1.0, pow=1.0, w_min=0.01, gamma_ff=0.9995, gamma_rec=0.9995, T=200,
           n_ins=1200, ins_gap=1500, lean=0, nostream=0, frac=1.0,
           ckpt='0.333,0.667,0.833,1.0', n_top=80)


# ------------------------------------------------------------------------------------------------- the stream
def build_stream(cfg, verbose=True):
    """returns (timeline, info): timeline = sorted list of (t, kind, payload); kinds 'eye' (indices), 'ch' (char
    index), 'end' (line end); info = chars, assemblies, eye codes, held sets, counts"""
    d = json.load(open(os.path.join(HERE, 'stream.json')))
    ev = d['events']
    t_end = int(d['ticks'] * cfg['frac'])
    seed = cfg['seed']
    rng = np.random.default_rng(1000 + seed)
    held_f = [str(x) for x in rng.choice(FOODS, 2, replace=False)]
    held_t = [str(x) for x in rng.choice(TOYS_HELDABLE, 2, replace=False)]
    held = set(held_f + held_t)
    # X3's char order and assemblies (physics seed 7), then the new chars from the same rng
    chars = sorted({ch for e in ev if e[1] == 'line' for ch in e[3]})
    prng = np.random.default_rng(7)
    asm = [np.sort(prng.choice(N_A, K_A, replace=False)) for _ in chars]
    extra = sorted(set(''.join(FRAMES[f][0](x) + FRAMES[f][1](x) for f in FRAMES for x in ALLOWED[f])) - set(chars))
    for c in extra:
        chars.append(c); asm.append(np.sort(prng.choice(N_A, K_A, replace=False)))
    cix = {c: i for i, c in enumerate(chars)}
    cur = Curriculum(0)
    rp = _rng(0, 14)
    person = np.concatenate([rp.choice(N_E, 8, replace=False), rp.choice(N_E, 8, replace=False)])

    def drop(text):
        """base-stream lines removed for the held fillers: taste-frame lines (and, lean, every class frame)"""
        for x in held:
            if x in text and ('好吃' in text):
                return True
            if cfg['lean'] and x in text and text.startswith('我的'):
                return True
        return False

    base = [e for e in ev if e[0] < t_end]
    n_drop = sum(1 for e in base if e[1] == 'line' and drop(e[3]))
    lines = [] if cfg['nostream'] else [(e[0], e[2], e[3]) for e in base if e[1] == 'line' and not drop(e[3])]
    eyes = [(e[0], np.asarray(e[4], np.int32)) for e in base]          # every event carries the eye at that time
    # silent daytime gaps for the class dialogues
    evs = sorted(base, key=lambda e: e[0])
    gaps = []
    for a, b in zip(evs, evs[1:]):
        ta = a[0] + (SYL * len(a[3]) if a[1] == 'line' else 0)
        if b[0] - ta >= 80 and len(a[4]) > 0:
            gaps.append((ta, b[0], np.asarray(a[4], np.int32)))
    elig, last = [], -10 ** 9
    for g in gaps:                                     # spaced >= ins_gap ticks apart, then a uniform sample
        if g[0] - last >= cfg['ins_gap']:
            elig.append(g); last = g[0]
    n_take = min(len(elig), int(round(cfg['n_ins'] * cfg['frac'])))
    sel = [elig[i] for i in np.sort(rng.choice(len(elig), n_take, replace=False))]
    ins_lines, ins_eyes, n_frame = [], [], {}
    for g0, g1, eye0 in sel:
        while True:
            fr = str(rng.choice(list(FRAMES)))
            pool = [x for x in ALLOWED[fr] if not (x in held and (fr == 'taste' or (cfg['lean'] and fr in ('eat', 'play'))))]
            if pool:
                break
        x = str(rng.choice(pool))
        q, a = FRAMES[fr][0](x), FRAMES[fr][1](x)
        s = int(rng.integers(3))
        scene = [None, None, None]
        scene[s] = (x, int(rng.integers(6)), int(rng.integers(3)))
        e_x = np.flatnonzero(cur.eye(tuple(scene), s)).astype(np.int32)
        pers = np.intersect1d(eye0, person)
        t0 = g0 + 10
        ins_eyes.append((t0, np.union1d(e_x, pers).astype(np.int32)))
        tq = t0 + 5
        ins_lines.append((tq, 'other', q))
        ta = tq + SYL * len(q) + 3
        ins_lines.append((ta, 'other2', a))
        te = ta + SYL * len(a) + 5
        assert te < g1, (te, g1)
        ins_eyes.append((te, eye0))
        n_frame[(fr, x)] = n_frame.get((fr, x), 0) + 1
    tl = []
    for t, v in eyes + ins_eyes:
        tl.append((t, 0, 'eye', v))
    for t, who, text in lines + ins_lines:
        for i, ch in enumerate(text):
            tl.append((t + SYL * i, 1, 'ch', cix[ch]))
        tl.append((t + SYL * len(text), 1, 'end', None))
    tl.sort(key=lambda r: (r[0], r[1]))
    # merge: one update per distinct time (eye first, then the sound at that tick)
    out = []
    for t, _, k, v in tl:
        if out and out[-1][0] == t:
            out[-1][1][k] = v
        else:
            out.append((t, {k: v}))
    cnt = {}
    for t, who, text in lines + ins_lines:
        for ch in text:
            cnt[ch] = cnt.get(ch, 0) + 1
    eye_on = np.zeros(N_E, bool)
    for _, v in eyes + ins_eyes:
        eye_on[v] = True
    info = dict(chars=chars, cix=cix, asm=asm, cur=cur, held_f=held_f, held_t=held_t, held=held,
                t_end=t_end, n_ins=len(sel), n_drop=n_drop, n_frame=n_frame, cnt=cnt,
                feat_A=np.unique(np.concatenate([asm[cix[c]] for c in cnt])).astype(np.int32),
                feat_E=np.flatnonzero(eye_on).astype(np.int32), n_lines=len(lines) + len(ins_lines))
    if verbose:
        print(f'stream: {len(out)} updates, {info["n_lines"]} lines ({len(ins_lines)} inserted, {n_drop} dropped), '
              f'held foods {held_f} toys {held_t}, feat A {len(info["feat_A"])} E {len(info["feat_E"])}', flush=True)
    return out, info


# ------------------------------------------------------------------------------------------------- the area
class Thr:
    """X3's count control: smoothed quantile threshold with an inhibitory ceiling (alpha = 1: instant quantile)"""

    def __init__(self, f, alpha, ceil=2.0, theta=None):
        self.f, self.a, self.ceil, self.theta = f, alpha, ceil, theta

    def fire(self, d, rng=None):
        n = len(d)
        k = max(1, int(round(self.f * n)))
        q = float(np.partition(d, n - k)[n - k])
        self.theta = q if self.theta is None else self.theta + self.a * (q - self.theta)
        dd = d + (1e-7 * rng.random(n) if rng is not None else 0.0)
        on = np.flatnonzero((dd >= max(self.theta, 1e-9)) & (d > 0))
        kmax = int(self.ceil * k)
        if len(on) > kmax:
            on = np.sort(on[np.argpartition(-dd[on], kmax)[:kmax]])
        return on


class Area:
    def __init__(self, cfg, feat_A, feat_E):
        self.c = c = cfg
        N = self.N = c['N']
        rng = np.random.default_rng(500 + c['seed'])
        self.rng = np.random.default_rng(600 + c['seed'])
        fe = feat_E + N_A
        # birth feed-forward: cA AUD + cE EYE comparable inputs per concept unit
        B = np.zeros((N_IN, N), np.float32)
        for j in range(N):
            a = rng.choice(feat_A, c['cA'], replace=False)
            e = rng.choice(fe, min(c['cE'], len(fe)), replace=False)
            B[a, j] = rng.uniform(c['w_lo'], c['w_hi'], len(a))
            B[e, j] = rng.uniform(c['w_lo'], c['w_hi'], len(e))
        self.B = B
        M = np.zeros((N_IN, N), bool)                      # potential synapses (birth anatomy; silent ones weight 0)
        M[feat_A] = rng.random((len(feat_A), N)) < c['pA']
        M[fe] = rng.random((len(fe), N)) < c['pE']
        M |= B > 0
        self.M = M
        self.V = np.zeros((N_IN, N), np.float32)           # learned part (unscaled); effective = V * s
        self.s = 1.0
        self.Bout = B.sum(1).astype(np.float64)
        # two-layer weights computed per source area (harvest h4 #10): in-totals of AUD and EYE lines kept apart
        self.Bin = np.stack([B[:N_A].sum(0), B[N_A:].sum(0)]).astype(np.float64)
        self.Vout = np.zeros(N_IN); self.Vin = np.zeros((2, N))
        # recurrence
        self.use_rb = c['arm'] in ('Tb', 'Tbl')
        self.use_rl = c['arm'] in ('Tl', 'Tbl')
        self.RB = np.zeros((N, N), np.float32)
        if self.use_rb:
            for j in range(N):
                k = rng.choice(N - 1, c['cR'], replace=False); k[k >= j] += 1
                self.RB[k, j] = rng.uniform(c['r_lo'], c['r_hi'], len(k))
        self.RM = rng.random((N, N)) < c['pR']
        np.fill_diagonal(self.RM, False)
        self.RM |= self.RB > 0
        self.RV = np.zeros((N, N), np.float32); self.rs = 1.0
        self.RBout = self.RB.sum(1).astype(np.float64); self.RBin = self.RB.sum(0).astype(np.float64)
        self.RVout = np.zeros(N); self.RVin = np.zeros(N)
        self.thr = Thr(c['f'], c['alpha'], c['ceil'])
        self.prev = np.zeros(0, np.int64); self.prev_in = np.zeros(0, np.int64)
        self.t_dec = 0; self.n_dec = 0
        self.n_on = []

    # ---- drive
    def g(self, w):
        return 1.0 / (1.0 + np.maximum(w, 0.0) / self.c['w_ref']) ** self.c['pow']

    def push(self, inp, a):
        if len(inp) == 0:
            return np.zeros(self.N)
        W = self.B[inp] + self.s * self.V[inp]
        sout = np.sqrt(np.maximum(self.Bout[inp] + self.s * self.Vout[inp], 1e-9))
        sin = np.sqrt(np.maximum(self.Bin[a] + self.s * self.Vin[a], 1e-9))
        return (W / sout[:, None]).sum(0) / sin

    def ff(self, inp):
        inp = np.asarray(inp, np.int64)
        a, e = inp[inp < N_A], inp[inp >= N_A]
        return self.push(a, 0) + self.c['g_E'] * self.push(e, 1)

    def rec(self, prev):
        if len(prev) == 0 or not (self.use_rb or self.use_rl) or self.c['beta'] == 0:
            return np.zeros(self.N)
        W = self.RB[prev] + self.rs * self.RV[prev]
        sout = self.RBout[prev] + self.rs * self.RVout[prev]
        sin = self.RBin + self.rs * self.RVin
        ok = sout > 1e-6
        if not ok.any():
            return np.zeros(self.N)
        r = (W[ok] / np.sqrt(sout[ok])[:, None]).sum(0) / np.sqrt(np.maximum(sin, 1e-6))
        return r

    def drive(self, inp, prev):
        return self.ff(inp) + self.c['beta'] * self.rec(prev)

    # ---- learning (the one rule)
    def _ff_update(self, pre, post, rate):
        """pre rows x post cols, rate per pre row (signed)"""
        if len(pre) == 0 or len(post) == 0:
            return
        ix = np.ix_(pre, post)
        m = self.M[ix]
        if not m.any():
            return
        b = self.B[ix]; v = self.V[ix]
        w = b + self.s * v
        dw = rate[:, None] * self.g(w) * m
        nv = np.maximum(w + dw, 0.0)                      # total weight never below 0
        dv = (nv - w) / self.s
        self.V[ix] = v + dv
        self.Vout[pre] += dv.sum(1)
        ia = pre < N_A
        self.Vin[0, post] += dv[ia].sum(0); self.Vin[1, post] += dv[~ia].sum(0)

    def _rec_update(self, pre, post, eta):
        if len(pre) == 0 or len(post) == 0 or eta == 0:
            return
        ix = np.ix_(pre, post)
        m = self.RM[ix]
        if not m.any():
            return
        b = self.RB[ix]; v = self.RV[ix]
        w = b + self.rs * v
        nv = np.maximum(w + eta * self.g(w) * m, 0.0)
        dv = (nv - w) / self.rs
        self.RV[ix] = v + dv
        self.RVout[pre] += dv.sum(1); self.RVin[post] += dv.sum(0)

    def decay(self, t):
        nd = t // self.c['T'] - self.t_dec
        if nd <= 0:
            return
        self.t_dec += nd
        self.s *= self.c['gamma_ff'] ** nd
        self.rs *= self.c['gamma_rec'] ** nd
        self.n_dec += nd
        if self.s < 0.5 or self.rs < 0.5:
            self.renorm()

    def renorm(self):
        """fold the global decay scale into the weights; delete learned links below w_min"""
        wm = self.c['w_min']
        for b0 in range(0, N_IN, 512):
            v = self.V[b0:b0 + 512]; v *= self.s
            v[(np.abs(v) < wm) & (self.B[b0:b0 + 512] == 0)] = 0.0
        self.s = 1.0
        self.Vout = self.V.sum(1).astype(np.float64)
        self.Vin = np.stack([self.V[:N_A].sum(0), self.V[N_A:].sum(0)]).astype(np.float64)
        self.RV *= self.rs; self.RV[self.RV < wm] = 0.0; self.rs = 1.0
        self.RVout = self.RV.sum(1).astype(np.float64); self.RVin = self.RV.sum(0).astype(np.float64)

    def step(self, t, inp):
        c = self.c
        self.decay(t)
        x = self.thr.fire(self.drive(inp, self.prev), self.rng)
        prev, pin = self.prev, self.prev_in
        rising = np.setdiff1d(x, prev, assume_unique=True)
        in_rise = np.setdiff1d(inp, pin, assume_unique=True)
        in_held = np.intersect1d(inp, pin, assume_unique=True)
        in_off = np.setdiff1d(pin, inp, assume_unique=True)
        # feed-forward: B (concept unit) rises while A (input) is on / just off -> A->B up
        pre = np.concatenate([in_rise, in_held, in_off]).astype(np.int64)
        rate = np.concatenate([np.full(len(in_rise) + len(in_held), c['eta_ff']),
                               np.full(len(in_off), c['eta_ff_off'])])
        self._ff_update(pre, rising, rate)
        # A (input) rises after B (concept unit) was on -> A->B down
        if c['eta_ff_rev'] and len(prev):
            self._ff_update(in_rise, prev, np.full(len(in_rise), -c['eta_ff_rev']))
        if self.use_rl:
            # B rises while A was on (held or just off) -> A->B up (sequence); co-rising -> mutual (co)
            self._rec_update(prev, rising, c['eta_seq'])
            if c['eta_co'] and len(rising) > 1:
                self._rec_update(rising, rising, c['eta_co'])
            if c['eta_rev'] and len(prev):
                self._rec_update(rising, prev, -c['eta_rev'])
        self.prev, self.prev_in = x, np.asarray(inp, np.int64)
        self.n_on.append(len(x))
        return x

    # ---- frozen probes (no learning)
    def probe(self, seq):
        """seq: list of input index arrays, presented after silence (no previous state); returns the states"""
        th = Thr(self.c['f'], 1.0, self.c['ceil'])
        prev = np.zeros(0, np.int64); xs = []
        for inp in seq:
            d = self.drive(inp, prev)
            th.theta = None
            x = th.fire(d)
            xs.append(x); prev = x
        return xs


# ------------------------------------------------------------------------------------------------- tests
def jac(a, b):
    if len(a) == 0 and len(b) == 0:
        return 0.0
    return len(np.intersect1d(a, b)) / max(1, len(np.union1d(a, b)))


def changed(a, b):
    return 1.0 - len(np.intersect1d(a, b)) / max(1, len(a), len(b))


def snapshot(A, info, rng_seed):
    asm, cix, cur = info['asm'], info['cix'], info['cur']
    words, first = {}, {}
    for X in OBJECTS:
        xs = A.probe([asm[cix[ch]].astype(np.int64) for ch in X])
        words[X] = np.union1d(xs[0], xs[1])
        first[X] = xs[0]
    top = sorted(info['cnt'], key=lambda ch: -info['cnt'][ch])[:A.c['n_top']]
    chars = {ch: A.probe([asm[cix[ch]].astype(np.int64)])[0] for ch in top}
    r = np.random.default_rng(rng_seed)
    draws = [(int(r.integers(3)), int(r.integers(6)), int(r.integers(3))) for _ in range(3)]
    see = {}
    for X in OBJECTS:
        see[X] = []
        for s, col, n in draws:
            scene = [None, None, None]; scene[s] = (X, col, n)
            e = np.flatnonzero(cur.eye(tuple(scene), None)).astype(np.int64) + N_A
            see[X].append(A.probe([e])[0])
    return dict(words=words, first=first, chars=chars, see=see)


def metrics(S, info, key='words', full=True):
    W = S[key]
    held = info['held']
    cls = {x: 0 for x in FOODS}; cls.update({x: 1 for x in TOYS})

    def share(a, b):
        return bool(set(a) & set(b))
    within, between = [], []
    for i, a in enumerate(OBJECTS):
        for b in OBJECTS[i + 1:]:
            if share(a, b):
                continue
            (within if cls[a] == cls[b] else between).append(jac(W[a], W[b]))

    def decode(h, pool):
        sc = []
        for k in (0, 1):
            m = [jac(W[h], W[y]) for y in pool if cls[y] == k and y != h and not share(h, y)]
            sc.append(np.mean(m) if m else 0.0)
        return int(np.argmax(sc) == cls[h] and sc[0] != sc[1]), sc[cls[h]] - sc[1 - cls[h]]
    seen = [x for x in OBJECTS if x not in held]
    hd = [decode(h, seen) for h in sorted(held)]
    loo = [decode(h, seen) for h in seen]
    out = dict(within=float(np.mean(within)), between=float(np.mean(between)),
               held_dec=float(np.mean([h[0] for h in hd])), held_margin=float(np.mean([h[1] for h in hd])),
               held_each={h: int(v[0]) for h, v in zip(sorted(held), hd)},
               seen_dec=float(np.mean([h[0] for h in loo])))
    if not full:
        return out
    # cross-modal
    mt, mm, top1 = [], [], []
    for d in range(len(S['see'][OBJECTS[0]])):
        for X in OBJECTS:
            m = jac(W[X], S['see'][X][d])
            o = [jac(W[X], S['see'][Y][d]) for Y in OBJECTS if Y != X]
            mt.append(m); mm.append(float(np.mean(o)))
            top1.append(float(m > max(o)))
    C = list(S['chars'].values())
    pj = [jac(C[i], C[j]) for i in range(len(C)) for j in range(i + 1, len(C))]
    out.update({'c1_' + k: v for k, v in metrics(S, info, 'first', False).items() if k != 'held_each'})
    return dict(out, xm_match=float(np.mean(mt)), xm_mis=float(np.mean(mm)), xm_top1=float(np.mean(top1)),
                collapse=float(np.mean(pj)), size_word=float(np.mean([len(v) for v in W.values()])),
                size_char=float(np.mean([len(v) for v in C])), size_see=float(np.mean([len(v) for vv in S['see'].values() for v in vv])))


def drift(S0, S1):
    w = [changed(S0['words'][x], S1['words'][x]) for x in OBJECTS]
    c = [changed(S0['chars'][k], S1['chars'][k]) for k in S0['chars'] if k in S1['chars']]
    return dict(drift_words=float(np.mean(w)), drift_chars=float(np.mean(c)),
                drift_words_jac=float(np.mean([1 - jac(S0['words'][x], S1['words'][x]) for x in OBJECTS])))


def run(cfg, verbose=True):
    t0 = time.time()
    tl, info = build_stream(cfg, verbose)
    A = Area(cfg, info['feat_A'], info['feat_E'])
    if verbose:
        print(f'area built {time.time() - t0:.1f} s', flush=True)
    ck = [float(x) for x in str(cfg['ckpt']).split(',')]
    ck_t = [int(f * info['t_end']) for f in ck]
    snaps, res = {}, []
    eye = np.zeros(0, np.int64); aud = None
    ci = 0
    for t, ev in tl:
        while ci < len(ck_t) and t >= ck_t[ci]:
            S = snapshot(A, info, 77 + cfg['seed'])
            snaps[ck[ci]] = S
            m = metrics(S, info); m['frac'] = ck[ci]; m['secs'] = round(time.time() - t0, 1)
            m['n_on_mean'] = float(np.mean(A.n_on[-5000:])) if A.n_on else 0.0
            res.append(m)
            if verbose:
                print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in m.items()},
                      ensure_ascii=False), flush=True)
            ci += 1
        if 'eye' in ev:
            eye = ev['eye'].astype(np.int64) + N_A
        if 'end' in ev:
            aud = None
        if 'ch' in ev:
            aud = ev['ch']
        inp = eye if aud is None else np.union1d(info['asm'][aud], eye)
        A.step(t, inp)
    while ci < len(ck_t):
        S = snapshot(A, info, 77 + cfg['seed'])
        snaps[ck[ci]] = S
        m = metrics(S, info); m['frac'] = ck[ci]; m['secs'] = round(time.time() - t0, 1)
        m['n_on_mean'] = float(np.mean(A.n_on[-5000:]))
        res.append(m)
        if verbose:
            print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in m.items()},
                             ensure_ascii=False), flush=True)
        ci += 1
    dr = {}
    if 0.667 in snaps and 1.0 in snaps:
        dr = drift(snaps[0.667], snaps[1.0])
    if 0.833 in snaps and 1.0 in snaps:
        dr.update({k + '_56': v for k, v in drift(snaps[0.833], snaps[1.0]).items()})
    import resource
    out = dict(cfg=cfg, res=res, drift=dr, held_f=info['held_f'], held_t=info['held_t'], n_ins=info['n_ins'],
               n_drop=info['n_drop'], n_updates=len(tl), secs=round(time.time() - t0, 1),
               rss_mb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024),
               n_on=dict(mean=float(np.mean(A.n_on)), p95=float(np.percentile(A.n_on, 95)),
                         ceil_frac=float(np.mean(np.array(A.n_on) >= int(cfg['ceil'] * round(cfg['f'] * cfg['N']))))))
    if verbose:
        print('drift', json.dumps({k: round(v, 4) for k, v in dr.items()}), 'secs', out['secs'], 'rss', out['rss_mb'],
              'n_on', json.dumps({k: round(v, 3) for k, v in out['n_on'].items()}), flush=True)
    return out


def parse(argv):
    cfg = dict(CFG)
    outp = None
    for a in argv:
        k, v = a.split('=', 1)
        if k == 'out':
            outp = v; continue
        b = CFG[k]
        cfg[k] = type(b)(v) if not isinstance(b, str) else v
    return cfg, outp


if __name__ == '__main__':
    cfg, outp = parse(sys.argv[1:])
    if cfg['arm'] == 'B0':
        cfg['beta'] = 0.0
    out = run(cfg)
    if outp:
        json.dump(out, open(outp, 'w'), ensure_ascii=False)
