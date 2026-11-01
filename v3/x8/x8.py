"""X8: do the designer's two correction mechanisms (ARCH 2.8) keep skills AND allow corrections?

Harness: X4 (x4_merge.py, copied unchanged and imported): lineage, AUD assemblies, PFC (birth fan-in + birth
recurrence + quantile threshold), PFC -> AUD prediction links, lessons, raising, readout. See PLAN.md.
New here: rule variants (concave on the way down: item 3), a fast store H (item 4) whose recall lights the same
AUD, the reward tool (item 5), judged asks, own voice (self-rehearsal), phases.
Run: python3 -I x8.py <arm> <w_own> <seed> [out.json] [key=value ...]
"""
import copy
import json
import os
import resource
import sys
import time
from collections import deque

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(1, os.path.join(HERE, '..', 'x345'))   # repo: the unchanged X4 harness (x4_merge.py)
import x4_merge as X                                      # noqa: E402
from x4_merge import N_A, K_A, V, N_P, F_P, N_AX, SYL, GAP, CA, CB, Thr, Lineage, answer, make_lessons  # noqa

EMPTY = np.zeros(0, np.int64)


def rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


# ----------------------------------------------------------------------------------------------- learned links
class Links:
    """links pre-population -> AUD: w = B (birth, never decays) + D (learned), w >= 0. One rule for all areas."""

    def __init__(self, B, c, eta, eta_co, gamma, fast_part=False):
        self.B = B.astype(np.float32)
        self.D = np.zeros_like(self.B)
        self.c, self.eta, self.eta_co, self.gamma = c, eta, eta_co, gamma
        self.bsum = self.B.sum(0, dtype=np.float64)
        self.si = self.bsum.copy()
        self.log = deque()
        self.fast = fast_part                     # item 12: D = F (fast, decays) + S (slow); self.D is F + S
        if fast_part:
            self.S = np.zeros_like(self.B)

    def g(self, w):
        return np.maximum(self.c['floor'], 1.0 / (1.0 + np.maximum(w, 0.0) / self.c['w_ref']) ** self.c['pow'])

    def _apply(self, rows, cols, dlt, t, log=True):
        ix = np.ix_(rows, cols)
        d0 = self.D[ix]
        new = np.maximum(d0 + dlt, -self.B[ix])
        delta = (new - d0).astype(np.float32)
        self.D[ix] = new
        self.si[cols] += delta.sum(0)
        if self.fast:                             # S never goes below what is left of the total
            Sx = self.S[ix]
            self.S[ix] = np.where(new < Sx, np.maximum(new, 0.0), Sx)
        if log and self.c['log']:
            self.log.append((t, rows, cols, delta))
        return delta

    def upd(self, rows, cols, eta, sign, t):
        if len(rows) == 0 or len(cols) == 0 or eta == 0:
            return
        ix = np.ix_(rows, cols)
        if sign > 0 or self.c['down_concave'] or self.c['down_weak_only']:
            step = eta * self.g(self.B[ix] + self.D[ix])
        else:
            step = np.full((len(rows), len(cols)), eta, np.float32)
        delta = self._apply(rows, cols, sign * step, t)
        if self.fast and sign > 0 and self.c['f2s_rep'] > 0:
            # item 12: repeated strengthening moves F into S: when F alone exceeds f2s_rep, a share moves
            F = self.D[ix] - self.S[ix]
            mv = np.where(F > self.c['f2s_rep'], self.c['f2s_frac'] * F, 0.0)
            self.S[ix] += mv

    def decay(self, nd):
        if nd <= 0:
            return
        if self.fast:                             # the one decay acts on F only (item 12)
            F = self.D - self.S
            gf = self.c['gamma_f']
            if self.c['down_concave']:
                F *= (1.0 - (1.0 - gf) * self.g(self.B + self.D)) ** nd
            else:
                F *= gf ** nd
            self.D = self.S + F
        elif self.c['down_concave']:
            k = 1.0 - self.gamma
            left = nd
            while left > 0:
                m = min(left, self.c['dec_chunk'])
                self.D *= (1.0 - k * self.g(self.B + self.D)) ** m
                left -= m
        else:
            self.D *= self.gamma ** nd
        small = np.abs(self.D) < self.c['w_min']
        self.D[small] = 0.0
        if self.fast:
            self.S[small] = 0.0
        self.si = self.bsum + self.D.sum(0, dtype=np.float64)

    def drive(self, x):
        """two-layer weights: strength / sqrt(out-strength of pre x in-strength of post)"""
        if len(x) == 0:
            return np.zeros(self.B.shape[1], np.float32)
        W = self.B[x] + self.D[x]
        so = W.sum(1) + 1e-6
        return (W / np.sqrt(so)[:, None]).sum(0) / np.sqrt(self.si + 1e-6)

    def prune(self, t, win):
        while self.log and t - self.log[0][0] >= win:
            self.log.popleft()

    def reward(self, t, gain, win, to_slow=False):
        """the reward tool: every change of the last win ticks is added again x gain (fading with age)"""
        self.prune(t, win)
        for t0, rows, cols, delta in list(self.log):
            f = gain * (1.0 - (t - t0) / win)
            d = self._apply(rows, cols, f * delta, t, log=False)
            if self.fast and to_slow:             # item 12: at a reward moment F of these synapses moves into S
                ix = np.ix_(rows, cols)
                F = self.D[ix] - self.S[ix]
                self.S[ix] += np.where(F > 0, F, 0.0)
        self.log.clear()

    def wipe(self):
        self.D[:] = 0.0
        if self.fast:
            self.S[:] = 0.0
        self.si = self.bsum.copy()
        self.log.clear()


# ----------------------------------------------------------------------------------------------- populations
class Pop:
    """binary population fed by AUD chain copies (dense birth fan-in) and its own birth recurrence (sparse)"""

    def __init__(self, B_in, rec_idx, rec_w, f, alpha, rho, rng, dense_rec=None):
        self.B_in, self.ri, self.rw, self.Brec = B_in, rec_idx, rec_w, dense_rec
        self.n = B_in.shape[1]
        self.f, self.alpha, self.rho, self.rng = f, alpha, rho, rng
        self.thr = Thr(f, alpha)
        self.theta_ref = None
        self.need_input = False

    def fire(self, aud, prev, thr=None, rng=None):
        thr = self.thr if thr is None else thr
        if self.need_input and len(aud) == 0:   # recurrence alone cannot reach threshold: silent without input
            return EMPTY
        d = self.B_in[aud].sum(0) if len(aud) else np.zeros(self.n, np.float32)
        if len(prev):
            if self.Brec is not None:
                r = self.Brec[prev].sum(0)
            else:
                r = np.bincount(self.ri[prev].ravel(), weights=self.rw[prev].ravel(), minlength=self.n)
            d = d + self.rho * (r - r.mean())
        return thr.fire(d, self.rng if rng is None else rng)

    def code(self, pats):
        th = Thr(self.f, self.alpha); th.theta = self.thr.theta if self.theta_ref is None else self.theta_ref
        rng = np.random.default_rng(12345)
        x = EMPTY
        for p in pats:
            x = self.fire(p, x, th, rng)
        return x


def hippo_birth(seed, n_h, fan_in=CA, fan_out=CB):
    r = np.random.default_rng(seed)
    B_in = np.zeros((N_AX, n_h), np.float32)
    for j in range(n_h):
        B_in[r.choice(N_A, fan_in, replace=False), j] = r.uniform(0.05, 0.20, fan_in)
    idx = np.zeros((n_h, fan_out), np.int64)
    for i in range(n_h):
        tg = r.choice(n_h - 1, fan_out, replace=False); tg[tg >= i] += 1
        idx[i] = tg
    w = r.uniform(0.05, 0.20, (n_h, fan_out)).astype(np.float32)
    B_out = np.zeros((n_h, N_AX), np.float32)
    B_out[:, :N_A] = r.uniform(0.0, 0.02, (n_h, N_A))
    return B_in, idx, w, B_out


# ----------------------------------------------------------------------------------------------- agent
class Agent:
    def __init__(self, lin, cfg, seed, hip_birth=None):
        c = self.c = cfg
        self.L = lin
        self.rng = np.random.default_rng(seed)
        B_ap = np.zeros((N_AX, N_P), np.float32); B_ap[:N_A] = lin.B_ap
        B_pa = np.zeros((N_P, N_AX), np.float32); B_pa[:, :N_A] = lin.B_pa
        self.pfc = Pop(B_ap, None, None, F_P, c['alpha'], c['rho'], self.rng, dense_rec=lin.B_pp)
        self.ctx = Links(B_pa, c, c['eta'], c['eta_co'], c['gamma'], fast_part=c.get('item12', False))
        self.hip = None
        if c['hip']:
            B_in, idx, w, B_out = hip_birth
            self.hpop = Pop(B_in, idx, w, c['f_h'], c['alpha'], c['rho_h'], np.random.default_rng(seed + 1))
            self.hpop.need_input = c['h_need_input']
            self.hip = Links(B_out, c, c['eta_h'], c['eta_co_h'], c['gamma_h'])
        self.M = np.zeros((V, N_AX), np.float32)
        for ch in range(V):
            self.M[ch, lin.asm[ch]] = 1.0 / K_A
        r = np.random.default_rng(777 + seed)
        self.own_order = [r.permutation(lin.asm[ch]) for ch in range(V)]
        self.prev_p = EMPTY; self.prev_a = EMPTY; self.prev_h = EMPTY
        self.t = 0; self.t_dec = 0
        self.theta = None
        self.n_overlap = [0, 0]

    def pattern(self, ch):
        return self.L.asm[ch]

    def own_pat(self, ch, w):
        k = max(1, int(round(w * K_A)))
        return np.sort(self.own_order[ch][:k])

    # ---- dynamics
    def _learn(self, L, x, prev, ra, fa):
        rp, fp = np.setdiff1d(x, prev), np.setdiff1d(prev, x)
        L.upd(fp, ra, L.eta, +1.0, self.t)
        L.upd(rp, ra, L.eta_co, +1.0, self.t)
        L.upd(rp, fa, L.eta, -1.0, self.t)
        L.prune(self.t, self.c['win'])

    def step(self, aud):
        c = self.c
        nd = self.t // c['T'] - self.t_dec
        if nd > 0:
            self.ctx.decay(nd)
            if self.hip is not None:
                self.hip.decay(nd)
            self.t_dec += nd
        x = self.pfc.fire(aud, self.prev_p)
        if len(self.prev_p) and len(aud):
            self.n_overlap[0] += len(np.intersect1d(x, self.prev_p)); self.n_overlap[1] += len(x)
        ra, fa = np.setdiff1d(aud, self.prev_a), np.setdiff1d(self.prev_a, aud)
        self._learn(self.ctx, x, self.prev_p, ra, fa)
        if self.hip is not None:
            h = self.hpop.fire(aud, self.prev_h)
            self._learn(self.hip, h, self.prev_h, ra, fa)
            self.prev_h = h
        self.prev_p, self.prev_a = x, aud

    def hear(self, chars):
        for ch in chars:
            self.step(self.pattern(ch)); self.t += SYL
        self.step(EMPTY); self.t += GAP

    def advance(self, ticks):
        """untouched time: nothing is heard; only the global decay acts"""
        self.t += int(ticks)
        nd = self.t // self.c['T'] - self.t_dec
        if nd > 0:
            self.ctx.decay(nd)
            if self.hip is not None:
                self.hip.decay(nd)
            self.t_dec += nd

    def reward(self, gain):
        self.ctx.reward(self.t, gain, self.c['win'], to_slow=self.c.get('item12', False))
        if self.hip is not None:
            self.hip.reward(self.t, gain, self.c['win'])

    def scores_live(self):
        sc_c = self.M @ self.ctx.drive(self.prev_p)
        sc_h = self.c['r_hip'] * (self.M @ self.hip.drive(self.prev_h)) if self.hip is not None else None
        return sc_c, sc_h

    def ask(self, q, w_own=0.0, expect=None, judged=False):
        """the adult says the question; then a silent answer moment (recall / own voice), then silence;
        judged: pleased -> reward; wrong or none -> (upset) + recast 'q a'. Returns (answer, recast?)."""
        c = self.c
        for ch in q:
            self.step(self.pattern(ch)); self.t += SYL
        need = judged or w_own > 0 or self.hip is not None
        ans = None; parts = []
        if need:
            sc_c, sc_h = self.scores_live()
            ans = answer(sc_c if sc_h is None else sc_c + sc_h, self.theta, c['sharp'])
            if sc_h is not None and ans is not None and sc_h[ans] >= self.theta:
                # inner hearing: AUD lights the winner of the summed return drive (cortex + fast store) when the
                # fast store's return to it reaches threshold (with no inner voice the cortex return alone stays
                # below the lighting threshold)
                parts.append(self.pattern(ans))
            if w_own > 0 and ans is not None:
                parts.append(self.own_pat(ans, w_own))
        aud = np.unique(np.concatenate(parts)) if parts else EMPTY
        self.step(aud); self.t += SYL
        if len(aud):                     # same step structure as a heard line: one silent step after the last sound
            self.step(EMPTY)
        recast = False
        if judged:
            if ans == expect:
                if c['pleased'] > 0:
                    self.reward(c['pleased'])
            else:
                if c['upset'] > 0 and ans is not None:
                    self.reward(-(1.0 + c['upset']))
                recast = True
        self.t += GAP
        if recast:
            self.hear(list(q) + [expect])
        return ans, recast

    # ---- tests (no learning)
    def test(self, items, mode):
        out = []
        for q, a in items:
            pats = [self.pattern(ch) for ch in q]
            sc = self.M @ self.ctx.drive(self.pfc.code(pats))
            if mode == 'beh' and self.hip is not None:
                sc = sc + self.c['r_hip'] * (self.M @ self.hip.drive(self.hpop.code(pats)))
            elif mode == 'hip':
                sc = self.c['r_hip'] * (self.M @ self.hip.drive(self.hpop.code(pats)))
            out.append(answer(sc, self.theta, self.c['sharp']))
        return out

    def wipe_hip(self):
        if self.hip is not None:
            self.hip.wipe(); self.prev_h = EMPTY


def clone(ag):
    """deep copy of an agent that shares the read-only birth wiring (never modified)"""
    memo = {id(ag.L): ag.L, id(ag.ctx.B): ag.ctx.B, id(ag.ctx.bsum): ag.ctx.bsum, id(ag.pfc.B_in): ag.pfc.B_in,
            id(ag.pfc.Brec): ag.pfc.Brec, id(ag.M): ag.M}
    if ag.hip is not None:
        for o in (ag.hip.B, ag.hip.bsum, ag.hpop.B_in, ag.hpop.ri, ag.hpop.rw):
            memo[id(o)] = o
    return copy.deepcopy(ag, memo)


def acc(ans, items):
    return float(np.mean([g == a for g, (q, a) in zip(ans, items)])) if items else float('nan')


# ----------------------------------------------------------------------------------------------- world streams
def filler_lines(rng, n, banned):
    out = []
    while len(out) < n:
        f = [int(v) for v in rng.choice(V, 3, replace=False)]
        if (f[0], f[1]) not in banned:
            out.append(f)
    return out


def stretch(ag, lessons, n_ask, n_fill, w_own, rng, banned, marks, log):
    items = [('ask', i) for i in rng.integers(0, len(lessons), n_ask)] + [('fill', f) for f in filler_lines(rng, n_fill, banned)]
    order = rng.permutation(len(items))
    for k, i in enumerate(order):
        kind, v = items[i]
        if kind == 'ask':
            ag.ask(lessons[v][0], w_own)
        else:
            ag.hear(v)
        if (k + 1) in marks:
            log.append(dict(lines=k + 1, ctx=acc(ag.test(lessons, 'ctx'), lessons),
                            beh=acc(ag.test(lessons, 'beh'), lessons)))


def corrections(ag, lessons, changed, w_own, rng, banned, R, control=False):
    """changed: list of (lesson index, new answer). Returns per-changed-lesson histories."""
    new = list(lessons)
    for i, a in changed:
        new[i] = (lessons[i][0], a)
    ch_items = [new[i] for i, _ in changed]
    unch_idx = [i for i in range(len(lessons)) if i not in {j for j, _ in changed}]
    unch = [lessons[i] for i in unch_idx]
    start_beh = ag.test(unch, 'beh'); start_ctx = ag.test(unch, 'ctx')
    known_b = [k for k, (g, (q, a)) in enumerate(zip(start_beh, unch)) if g == a]
    known_c = [k for k, (g, (q, a)) in enumerate(zip(start_ctx, unch)) if g == a]
    before_ch = ag.test([lessons[i] for i, _ in changed], 'beh')
    hist = [[] for _ in changed]                   # per round: (answer at the judged ask, recast?)
    rounds = []
    for r in range(R):
        items = [('c', k) for k in range(len(changed))] + [('u', int(v)) for v in rng.choice(len(unch), 8)] + \
                [('f', f) for f in filler_lines(rng, 20, banned)]
        for i in rng.permutation(len(items)):
            kind, v = items[i]
            if kind == 'c':
                q, a = ch_items[v]
                if control:
                    got, rc = ag.ask(q, w_own)
                else:
                    got, rc = ag.ask(q, w_own, a, judged=True)
                hist[v].append((got == a, rc, got == lessons[changed[v][0]][1]))
            elif kind == 'u':
                ag.ask(unch[v][0], w_own)
            else:
                ag.hear(v)
        eb = ag.test(ch_items, 'beh'); ec = ag.test(ch_items, 'ctx')
        ub = ag.test(unch, 'beh'); uc = ag.test(unch, 'ctx')
        rounds.append(dict(ch_beh=acc(eb, ch_items), ch_ctx=acc(ec, ch_items),
                           unch_beh=float(np.mean([ub[k] == unch[k][1] for k in known_b])) if known_b else None,
                           unch_ctx=float(np.mean([uc[k] == unch[k][1] for k in known_c])) if known_c else None))
    per = []
    for k, h in enumerate(hist):
        recasts_before, first = 0, None
        for (ok, rc, old) in h:
            if ok:
                first = recasts_before; break
            recasts_before += int(rc)
        stable = None
        for j in range(len(h)):
            if all(o for o, _, _ in h[j:]):
                stable = int(sum(rc for _, rc, _ in h[:j])); break
        per.append(dict(first=first if first is not None else 21, stable=stable if stable is not None else 21,
                        total_recasts=int(sum(rc for _, rc, _ in h)), said_old=int(sum(o for _, _, o in h)),
                        right_before=bool(before_ch[k] == lessons[changed[k][0]][1])))
    return dict(per=per, rounds=rounds, n_unch_known_beh=len(known_b), n_unch_known_ctx=len(known_c))


def new_learning(ag, items, w_own, rng, banned, R, n_fill=15):
    hist = [[] for _ in items]
    curve = []
    for r in range(R):
        seq = [('n', k) for k in range(len(items))] + [('f', f) for f in filler_lines(rng, n_fill, banned)]
        for i in rng.permutation(len(seq)):
            kind, v = seq[i]
            if kind == 'n':
                q, a = items[v]
                got, rc = ag.ask(q, w_own, a, judged=True)
                hist[v].append((got == a, rc))
            else:
                ag.hear(v)
        curve.append(dict(beh=acc(ag.test(items, 'beh'), items), ctx=acc(ag.test(items, 'ctx'), items)))
    first = []
    for h in hist:
        n, f = 0, None
        for ok, rc in h:
            if ok:
                f = n; break
            n += int(rc)
        first.append(f if f is not None else R + 1)
    return dict(first=first, curve=curve)


# ----------------------------------------------------------------------------------------------- one run
BASE = dict(X.BASE, fan_in=32, floor=0.01, down_concave=False, down_weak_only=False, hip=False, n_h=4096, f_h=0.01,
            eta_h=1.0, eta_co_h=0.033, rho_h=1.0, h_need_input=True, gamma_h=0.97, r_hip=1.0, pleased=0.0, upset=0.0, win=24, log=False,
            dec_chunk=5, item12=False, gamma_f=0.99, f2s_rep=0.5, f2s_frac=0.5)

ARMS = {
    'A0': dict(),
    'A3w': dict(down_weak_only=True),
    'A3': dict(down_concave=True),
    'A3R': dict(down_concave=True, pleased=2.0, log=True),
    'A4H': dict(hip=True),
    'A4': dict(hip=True, pleased=2.0, log=True),
    'A34H': dict(down_concave=True, hip=True),
    'A34': dict(down_concave=True, hip=True, pleased=2.0, log=True),
    'A34U': dict(down_concave=True, hip=True, pleased=2.0, upset=1.0, log=True),
    'A12': dict(item12=True, pleased=2.0, log=True),
    'A12U': dict(item12=True, pleased=2.0, upset=1.0, log=True),
}


def make_cfg(arm, over):
    c = dict(BASE, **ARMS[arm]); c.update(over)
    return c


def run(arm, w_own, seed, over=None, n_ask=3000, n_fill=3000, R=20, R_new=10, life=True, amnesia=True,
        do_new=True, verbose=True):
    t0 = time.time()
    c = make_cfg(arm, over or {})
    lin = Lineage(seed_lin=1 + seed)
    hb = hippo_birth(5000 + seed, c['n_h'], c['fan_in']) if c['hip'] else None
    rng = np.random.default_rng(50 + seed)
    S, own, C = make_lessons(rng, n_sib=2)
    lessons = S + own[0] + [(q, a[0]) for q, a in C]
    new_items = own[1]
    banned = {tuple(q) for q, a in lessons + new_items}
    # phase 1: X4 raising (identical stream to raise_sib for sibling 0)
    ag = Agent(lin, c, 100 * seed + 11, hb)
    r1 = np.random.default_rng(100 * seed + 11 + 1000)
    items = [list(q) + [a] for q, a in lessons] * c['n_rep']
    items += [list(r1.choice(V, 3, replace=False)) for _ in range(c['n_fill'])]
    for i in r1.permutation(len(items)):
        ag.hear([int(v) for v in items[i]])
    ag.pfc.theta_ref = ag.pfc.thr.theta
    if ag.hip is not None:
        ag.hpop.theta_ref = ag.hpop.thr.theta
    ag.theta = 0.3 * float(np.median([(ag.M @ ag.ctx.drive(ag.pfc.code([ag.pattern(ch) for ch in q]))).max()
                                      for q, a in lessons]))
    res = dict(arm=arm, w_own=w_own, seed=seed, over=over or {}, theta=ag.theta)
    p1c, p1b = ag.test(lessons, 'ctx'), ag.test(lessons, 'beh')
    known = [i for i, (g, (q, a)) in enumerate(zip(p1c, lessons)) if g == a]
    res['p1'] = dict(ctx=acc(p1c, lessons), beh=acc(p1b, lessons), own=acc(p1c[20:50], own[0]),
                     shared=acc(p1c[:20], S), hold_overlap=ag.n_overlap[1] and ag.n_overlap[0] / ag.n_overlap[1])
    if c['hip']:
        res['p1']['hip'] = acc(ag.test(lessons, 'hip'), lessons)

    def retention(a):
        g = a.test(lessons, 'ctx')
        return dict(ctx_abs=acc(g, lessons), ctx_ret=float(np.mean([g[i] == lessons[i][1] for i in known])) if known else None,
                    beh_abs=acc(a.test(lessons, 'beh'), lessons))
    # untouched life (decay only)
    if life:
        a2 = clone(ag)
        res['life'] = []
        for tk in (100_000, 300_000, 1_000_000):
            a2.advance(tk - (a2.t - ag.t))
            res['life'].append(dict(ticks=tk, **retention(a2)))
        del a2
    marks = {1500, 3000, 4500, 6000}
    if amnesia and c['hip']:
        a2 = clone(ag)
        a2.wipe_hip()
        lg = []
        stretch(a2, lessons, n_ask, n_fill, w_own, np.random.default_rng(3000 + seed), banned, marks, lg)
        res['amnesia'] = dict(curve=lg, **retention(a2))
        del a2
    lg = []
    stretch(ag, lessons, n_ask, n_fill, w_own, np.random.default_rng(3000 + seed), banned, marks, lg)
    res['stretch'] = dict(curve=lg, **retention(ag))
    if verbose:
        print(json.dumps(dict(arm=arm, w=w_own, seed=seed, p1=res['p1'], stretch=res['stretch']['ctx_ret'],
                              amn=res.get('amnesia', {}).get('ctx_ret'), secs=round(time.time() - t0))), flush=True)
    # phase 4 branch: adult new learning, and a fresh birth on the same stream
    if do_new:
        a2 = clone(ag)
        res['new_adult'] = new_learning(a2, new_items, w_own, np.random.default_rng(6000 + seed), banned, R_new)
        del a2
        fr = Agent(lin, c, 100 * seed + 11, hb)
        fr.theta = ag.theta
        fr.pfc.theta_ref = ag.pfc.theta_ref
        if fr.hip is not None:
            fr.hpop.theta_ref = ag.hpop.theta_ref
        res['new_fresh'] = new_learning(fr, new_items, w_own, np.random.default_rng(6000 + seed), banned, R_new)
        del fr
    # phase 3: corrections
    rc = np.random.default_rng(4000 + seed)
    idx = rc.choice(50, 4, replace=False)
    changed = []
    for i in idx:
        a = int(rc.integers(V - 1)); a += a >= lessons[i][1]
        changed.append((int(i), a))
    res['changed'] = changed
    a2 = clone(ag)
    ctl = corrections(a2, lessons, changed, w_own, np.random.default_rng(5000 + seed), banned, R, control=True)
    res['corr_control'] = dict(rounds=ctl['rounds'])
    del a2
    res['corr'] = corrections(ag, lessons, changed, w_own, np.random.default_rng(5000 + seed), banned, R)
    ag.wipe_hip()
    res['corr']['ctx_after_wipe'] = acc(ag.test([(lessons[i][0], a) for i, a in changed], 'ctx'),
                                        [(lessons[i][0], a) for i, a in changed])
    res['secs'] = round(time.time() - t0, 1)
    res['rss_mb'] = round(rss_mb())
    if verbose:
        print(json.dumps(dict(arm=arm, w=w_own, seed=seed, first=[p['first'] for p in res['corr']['per']],
                              unch=res['corr']['rounds'][-1], secs=res['secs'], rss=res['rss_mb'])), flush=True)
    return res


if __name__ == '__main__':
    arm, w_own, seed = sys.argv[1], float(sys.argv[2]), int(sys.argv[3])
    out = sys.argv[4] if len(sys.argv) > 4 else None
    over = {}
    kw = {}
    for kv in sys.argv[5:]:
        k, v = kv.split('=')
        v = json.loads(v)
        if k in ('n_ask', 'n_fill_s', 'R', 'R_new', 'life', 'amnesia', 'do_new'):
            kw['n_fill' if k == 'n_fill_s' else k] = v
        else:
            over[k] = v
    res = run(arm, w_own, seed, over, **kw)
    if out:
        json.dump(res, open(out, 'w'))
