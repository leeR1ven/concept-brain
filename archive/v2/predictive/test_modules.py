# -*- coding: utf-8 -*-
"""Module tests of ARCH_v2-P parts built from the ONE rule (pr.py).  Usage: python3 -B test_modules.py [M1 M2 M4 M5]
  M1 assemblies: PR with a STATE target on recurrent links (co-activity = target), third factor = 'remember this'.
  M2 hippocampus: DG expansion + CA3 assemblies (one-shot) + reinstatement of the stored line content; recall from a
     partial cue; recency among competing episodes (one global decay); capacity.
  M4 PFC working memory: gated load, leaky units + learned recurrence; hold through a reply; period-2 check; update.
  M5 drift: the held code of a fixed probe line while the recurrence keeps learning on 1,000 other lines.
Each prints numbers and PASS/FAIL against criteria fixed in advance."""
import sys
import time
import numpy as np
sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__)))
from pr import Predictor, Recurrent, topk_mask, F32  # noqa: E402

RES = []


def report(name, ok, msg):
    RES.append((name, ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {msg}", flush=True)


def jac(a, b):
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else 1.0


# ===================================================================================== M1 assemblies
def m1_run(N=2000, k=100, fanin=1000, M=100, eta=0.5, seed=0, n_test=40):
    rng = np.random.default_rng(seed)
    R = Recurrent(N, k, fanin, eta=eta, theta=1.0, lam_v=0.0, c_rec=1.0, seed=seed)
    pats = [rng.choice(N, k, replace=False) for _ in range(M)]

    def ext(idx, a=10.0):
        e = np.zeros(N, F32); e[idx] = a; return e
    for p in pats:
        for _ in range(2):
            R.step(ext(rng.choice(N, k, replace=False)))           # other input, no learning
        R.step(ext(p))                                              # onset (its source is the gap): not stored
        for _ in range(2):
            R.step(ext(p), learn=True)                              # held ticks: third factor on
    out = {}
    for name, group in (('oldest', pats[:n_test]), ('newest', pats[-n_test:])):
        comp, hold16 = [], []
        for p in group:
            for _ in range(2):
                R.step(ext(rng.choice(N, k, replace=False)))
            R.step(ext(rng.choice(p, k // 2, replace=False)))      # 50 % cue, one tick
            for _ in range(3):
                s = R.step(np.zeros(N, F32))
            comp.append(s[p].mean())
            for _ in range(13):
                s = R.step(np.zeros(N, F32))
            hold16.append(s[p].mean())
        out[name] = (float(np.mean(comp)), float(np.mean(np.array(comp) >= 0.9)), float(np.mean(hold16)))
    # update: hold A, cue B (50 %, one tick) -> switch?
    sw = []
    for i in range(20):
        A, Bp = pats[(-1 - i) % M], pats[(-1 - i - M // 2) % M]
        R.step(ext(A)); [R.step(np.zeros(N, F32)) for _ in range(3)]
        R.step(ext(rng.choice(Bp, k // 2, replace=False)))
        for _ in range(3):
            s = R.step(np.zeros(N, F32))
        sw.append(s[Bp].mean())
    gsw = []                                                         # gated update: the gate loads B in full (2 ticks)
    for i in range(20):
        A, Bp = pats[(-1 - i) % M], pats[(-1 - i - M // 2) % M]
        R.step(ext(A)); [R.step(np.zeros(N, F32)) for _ in range(3)]
        R.step(ext(Bp)); R.step(ext(Bp))
        for _ in range(3):
            s = R.step(np.zeros(N, F32))
        gsw.append(s[Bp].mean())
    sw = (float(np.mean(sw)), float(np.mean(gsw)))
    # spontaneous: random start -> best overlap with any stored pattern after 10 free ticks
    best = []
    P_sets = [set(p.tolist()) for p in pats]
    for i in range(20):
        R.step(ext(rng.choice(N, k, replace=False)))
        for _ in range(10):
            s = R.step(np.zeros(N, F32))
        on = set(np.flatnonzero(s).tolist())
        best.append(max(len(on & q) / k for q in P_sets))
    W = R.P.paths['rec'].W
    return out, sw, float(np.mean(best)), float(np.abs(W).max()), float(np.abs(W).sum(0).max())


def m1():
    rows = []
    for fanin, Ms in ((1000, (100, 300)), (100, (20, 100))):
        for M in Ms:
            t0 = time.time()
            out, sw, spont, wmax, wtot = m1_run(fanin=fanin, M=M)
            rows.append((fanin, M, out, sw, spont, wmax, wtot))
            print(f"   fan-in {fanin} M {M}: oldest compl {out['oldest'][0]:.2f} (>=0.9: {out['oldest'][1]:.2f}) hold@16 "
                  f"{out['oldest'][2]:.2f} | newest compl {out['newest'][0]:.2f} hold@16 {out['newest'][2]:.2f} | update by "
                  f"50% 1-tick cue {sw[0]:.2f}, by gated full load {sw[1]:.2f} | spontaneous best overlap {spont:.2f} | max|w| {wmax:.3f} max sum|w| per target {wtot:.2f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
    r = {(f, M): (o, sw, sp) for f, M, o, sw, sp, _, _ in rows}
    ok = (r[(1000, 100)][0]['oldest'][1] >= 0.95 and r[(1000, 100)][0]['oldest'][2] >= 0.9 and r[(1000, 100)][1][0] >= 0.8
          and r[(100, 100)][0]['oldest'][0] < 0.5)
    report('M1 assemblies from the PR state target (N 2000, k 100) [pre-set criteria]', ok,
           'criteria: fan-in 1000 / 100 patterns: >=95 % complete from 50 % cues, hold >= 0.9 at 16 ticks, update by a 50 % '
           '1-tick cue >= 0.8; fan-in 100 fails (capacity scales with fan-in); chance overlap 0.05')
    ok2 = (r[(1000, 100)][0]['oldest'][1] >= 0.95 and r[(1000, 100)][0]['oldest'][2] >= 0.9 and r[(1000, 100)][1][1] >= 0.8
           and r[(100, 100)][0]['oldest'][0] < 0.5)
    report('M1b same, update through the gate (full 2-tick load) [criterion revised after M1]', ok2,
           'a held attractor resists a partial distractor; the BG gate (full load) replaces it')


# ===================================================================================== M2 hippocampus
class Hippo:
    """EC (cortical context) -> DG (fixed sparse expansion, k-WTA) -> CA3 (assemblies: learned recurrent PR, state
    target, one-shot) -> reinstatement CA3 -> content (learned PR, one-shot).  One global decay of the learned CA3/out
    weights per stored episode (the designer's single clean-up) gives recency among competing episodes."""

    def __init__(self, n_ec, n_out, N=4000, k=40, fan=12, thr=3, eta=1.0, decay=0.0, rec=True, seed=0, c_rec=1.0):
        r = np.random.default_rng(seed)
        self.N, self.k, self.thr, self.decay, self.rec = N, k, thr, decay, rec
        self.inp = r.integers(0, n_ec, (N, fan))
        self.tie = r.random(N).astype(F32) * 1e-3
        self.CA3 = Recurrent(N, k, fanin=min(1000, N - 1), eta=eta, theta=1.0, c_rec=c_rec, seed=seed + 1)
        self.OUT = Predictor(n_out, eta=eta, theta=1.0, bias=False)
        self.OUT.add('ca3', N)

    def dg(self, ec):
        I = ec[self.inp].sum(1).astype(F32)
        idx = np.flatnonzero(I >= self.thr)
        s = np.zeros(self.N, bool)
        if len(idx) > self.k:
            idx = idx[np.argpartition(-(I[idx] + self.tie[idx]), self.k)[:self.k]]
        s[idx] = True
        return s

    def store(self, ec, content):
        d = self.dg(ec).astype(F32) * 10.0
        self.CA3.A.reset()
        self.CA3.step(d)                                   # onset
        self.CA3.step(d, learn=self.rec)                   # one held tick, third factor on (novel episode)
        x = self.CA3.A.out.copy() / np.sqrt(self.k)
        self.OUT.predict({'ca3': x}); self.OUT.learn(content.astype(F32))
        if self.decay > 0:
            for P in (self.CA3.P.paths['rec'], self.OUT.paths['ca3']):
                P.W *= (1 - self.decay)

    def recall(self, ec, ticks=3):
        d = self.dg(ec).astype(F32) * 10.0
        self.CA3.A.reset()
        self.CA3.step(d)
        for _ in range(ticks):
            self.CA3.step(np.zeros(self.N, F32) if self.rec else d)
        x = self.CA3.A.out.copy() / np.sqrt(self.k)
        return self.OUT.predict({'ca3': x}).copy()


def m2():
    rng = np.random.default_rng(5)
    n_ec, n_ch, m = 3072, 200, 8                     # EC = Ghold (2048) + eye (1024); content = chars of the line
    def episode():
        g = np.zeros(n_ec, bool)
        n_codes = rng.integers(6, 11)
        for _ in range(n_codes):
            g[rng.choice(2048, m, replace=False)] = True
        g[2048 + rng.choice(1024, 72, replace=False)] = True
        ch = np.zeros(n_ch, bool); ch[rng.choice(n_ch, 4, replace=False)] = True
        return g, ch
    rows = {}
    for rec, kk, eta in ((True, 40, 1.0), (True, 80, 0.5), (False, 40, 1.0)):
        for E in (100, 1000, 3000):
            H = Hippo(n_ec, n_ch, rec=rec, seed=1, k=kk, eta=eta)
            eps = [episode() for _ in range(E)]
            for g, ch in eps:
                H.store(g, ch)
            res = {}
            for nm, grp in (('oldest', eps[:50]), ('newest', eps[-50:])):
                full, part = [], []
                for g, ch in grp:
                    p = H.recall(g); top = set(np.argsort(-p)[:4]); full.append(len(top & set(np.flatnonzero(ch))) / 4)
                    gp = g.copy(); on = np.flatnonzero(gp[:2048]); gp[rng.choice(on, len(on) // 2, replace=False)] = False
                    p = H.recall(gp); top = set(np.argsort(-p)[:4]); part.append(len(top & set(np.flatnonzero(ch))) / 4)
                res[nm] = (float(np.mean(full)), float(np.mean(part)))
            rows[(rec, kk, E)] = res
            print(f"   CA3 recurrence {rec} k {kk} eta {eta} episodes {E}: content recall (top-4 chars) oldest full/half-cue "
                  f"{res['oldest'][0]:.2f}/{res['oldest'][1]:.2f}  newest {res['newest'][0]:.2f}/{res['newest'][1]:.2f}", flush=True)
    # recency: 5 competing episodes share the same partial cue (e.g. 'my name is N' in 5 dialogues); the newest wins?
    win = {}
    for decay in (0.0, 0.002):
        H = Hippo(n_ec, n_ch, rec=True, seed=2, decay=decay)
        hits = 0
        for trial in range(40):
            base, _ = episode()
            names = rng.choice(n_ch, 5, replace=False)
            for j, nmc in enumerate(names):
                g = base.copy(); g[rng.choice(2048, 16, replace=False)] = True     # same frame, a different name code
                ch = np.zeros(n_ch, bool); ch[nmc] = True
                H.store(g, ch)
                for _ in range(20):
                    H.store(*episode())                                         # unrelated episodes in between
            p = H.recall(base)
            hits += int(np.argmax(p) == names[-1])
        win[decay] = hits / 40
    print(f"   recency among 5 competing episodes (cue = their shared part): newest recalled {win[0.0]:.2f} without decay, "
          f"{win[0.002]:.2f} with one global decay 0.002 per episode (chance 0.20)", flush=True)
    best_rec = max((rows[(True, 40, 1000)], rows[(True, 80, 1000)]), key=lambda r: r['oldest'][1])
    ok = best_rec['oldest'][0] >= 0.9 and best_rec['oldest'][1] > rows[(False, 40, 1000)]['oldest'][1] and win[0.002] >= 0.8
    report('M2 hippocampal index (DG + CA3 + reinstatement), one exposure', ok,
           'criteria: 1000 episodes, oldest recalled >= 0.9 from the full cue; CA3 recurrence improves half-cue recall; '
           'with one global decay the newest of competing episodes wins >= 0.8')


# ===================================================================================== M4 PFC working memory
def m4():
    rng = np.random.default_rng(7)
    N, k = 2048, 80
    def line():
        g = np.zeros(N, bool)
        for _ in range(rng.integers(6, 11)):
            g[rng.choice(N, 8, replace=False)] = True
        return g
    out = {}
    for name, lam_v, c_rec, learn in (('leaky + learned recurrence', 0.9, 1.0, True),
                                      ('leaky, no recurrence', 0.9, 0.0, False),
                                      ('no leak, learned recurrence', 0.0, 1.0, True)):
        R = Recurrent(N, k, fanin=1000, eta=0.5, theta=1.0, lam_v=lam_v, c_rec=c_rec, seed=3)
        lines = [line() for _ in range(300)]
        holds, p2, upd, dec = [], [], [], []
        for i, g in enumerate(lines):
            drive = g.astype(F32) * 3.0
            R.A.v[:] = 0                           # the BG gate's load = reset of the held potential + new input
            R.step(drive, learn=learn)            # load (BG gate open at the line end), 2 ticks
            s0 = R.step(drive, learn=learn).copy()
            seq = [s0]
            for t in range(60):                    # hold through a reply: gate closed, weak distractor input leaks in
                dis = np.zeros(N, F32); dis[rng.choice(N, k, replace=False)] = 0.3
                seq.append(R.step(dis).copy())
            holds.append(jac(seq[0], seq[60]))
            j1 = np.mean([jac(seq[t], seq[t + 1]) for t in range(10, 58)])
            j2 = np.mean([jac(seq[t], seq[t + 2]) for t in range(10, 58)])
            p2.append(j1 >= j2 - 1e-9)
            if i > 0:
                upd.append(jac(s0, prev))
            prev = seq[60]
            # decode: which of the last 20 lines is held (overlap of the held code with each line's input)?
            if i >= 20:
                cand = lines[i - 19:i + 1]
                sc = [(seq[60] & c).sum() for c in cand]
                dec.append(int(np.argmax(sc)) == 19)
        out[name] = (float(np.mean(holds)), float(np.mean(p2)), float(np.mean(upd)), float(np.mean(dec)))
        print(f"   {name:30s}: hold J(load, +60) {out[name][0]:.2f}  no-period-2 {out[name][1]:.2f}  "
              f"update J(new load, old hold) {out[name][2]:.2f}  decode held line (20-way) {out[name][3]:.2f}", flush=True)
    return_early = True
    o = out['leaky + learned recurrence']
    ok = o[0] >= 0.9 and o[1] >= 0.99 and o[2] <= 0.3 and o[3] >= 0.95
    report('M4 PFC working memory (gated load = reset + input, leak, learned recurrence)', ok,
           'criteria (weak distractor input during the hold): hold J >= 0.9 over 60 ticks; no period-2 in >= 99 % of '
           'holds; a new load replaces the old (J <= 0.3); decode >= 0.95')
    return
    # the old failure: a pure 2-layer synchronous loop A <-> B without leak or self-hold
    A = np.zeros(N, bool); A[rng.choice(N, k, replace=False)] = True
    WAB = rng.random((N, N)).astype(F32) < 0.02
    WBA = rng.random((N, N)).astype(F32) < 0.02
    B = topk_mask(A.astype(F32) @ WAB + rng.random(N) * 1e-3, k)
    seqA = []
    a, b = A, np.zeros(N, bool)
    for t in range(40):
        a, b = topk_mask(b.astype(F32) @ WBA + 1e-3 * rng.random(N), k) if t else A, topk_mask(a.astype(F32) @ WAB + 1e-3 * rng.random(N), k)
        seqA.append(a)
    j1 = np.mean([jac(seqA[t], seqA[t + 1]) for t in range(10, 38)]); j2 = np.mean([jac(seqA[t], seqA[t + 2]) for t in range(10, 38)])
    print(f"   2-layer synchronous loop, no leak: J(t,t+1) {j1:.2f} vs J(t,t+2) {j2:.2f} (period 2 if the second is larger)", flush=True)
    o = out['leaky + learned recurrence']
    ok = o[0] >= 0.9 and o[1] >= 0.99 and o[2] <= 0.3 and o[3] >= 0.95
    report('M4 PFC working memory (gated load, leak, learned recurrence)', ok,
           'criteria (weak distractor input during the hold): hold J >= 0.9 over 60 ticks; no period-2 in >= 99 % of holds; a new load replaces the old (J <= 0.3); '
           'decode >= 0.95')


# ===================================================================================== M5 drift
def m5():
    rng = np.random.default_rng(9)
    N, k = 2048, 80
    def line():
        g = np.zeros(N, bool)
        for _ in range(rng.integers(6, 11)):
            g[rng.choice(N, 8, replace=False)] = True
        return g
    R = Recurrent(N, k, fanin=1000, eta=0.5, theta=1.0, lam_v=0.9, c_rec=1.0, seed=4)
    probe = line()
    def held(g, learn=False):
        d = g.astype(F32) * 3.0
        R.step(d, learn=learn); R.step(d, learn=learn)
        for _ in range(20):
            s = R.step(np.zeros(N, F32))
        return s.copy()
    for _ in range(50):                            # the probe line is familiar (heard 50x among others)
        held(probe, learn=True); held(line(), learn=True)
    c0 = held(probe)
    js = []
    for i in range(1000):
        held(line(), learn=True)
        if (i + 1) % 250 == 0:
            js.append(jac(c0, held(probe)))
    print(f"   held code of a fixed familiar line vs its code before: Jaccard after 250/500/750/1000 other lines "
          f"{'/'.join(f'{j:.2f}' for j in js)}", flush=True)
    report('M5 drift of the PFC code under continued learning', js[-1] >= 0.8, 'criterion: Jaccard >= 0.8 after 1,000 lines')


# ===================================================================================== M6 reverberation checklist
def bench_lines():
    sys.path.insert(0, __import__('os').path.join(__import__('os').path.dirname(__import__('os').path.abspath(__file__)), 'bench_snapshot'))
    import world as W
    L = set()
    for X in W.OBJECTS:
        L |= {'这是什么', X, '这是' + X, X + '在哪里', '你要' + X + '吗', '要' + X, X + '好不好吃', X + '是什么颜色',
              '有几个' + X, '这是' + X + '吗', '在左边', '在中间', '在右边'}
    for S in W.SLOTS: L.add(S + '是什么')
    for C in W.COLOURS: L.add(C)
    for N in W.NAMES: L |= {'我叫' + N, '你叫' + N}
    for q, acc in W.RITUALS: L |= {q, *acc}
    return sorted(L)


def line_code(line, N=2048, m=8, seed=0):
    g = np.zeros(N, bool); prev = None
    for ch in line:
        g[np.random.default_rng([seed, 1, ord(ch)]).choice(N, m, replace=False)] = True
        if prev is not None:
            g[np.random.default_rng([seed, 2, ord(prev), ord(ch)]).choice(N, m, replace=False)] = True
        prev = ch
    return g


def m6(sp_beta=0.0):
    rng = np.random.default_rng(11)
    lines = bench_lines(); codes = [line_code(l) for l in lines]
    N = 2048
    print(f"   {len(lines)} distinct bench lines; input codes: mean active {np.mean([c.sum() for c in codes]):.0f}, "
          f"mean pairwise Jaccard {np.mean([jac(codes[i], codes[j]) for i in range(60) for j in range(i + 1, 60)]):.3f}", flush=True)
    res = {}
    for name, learn_hold in (('learning at load only', False), ('learning ON during the hold too', True)):
        R = Recurrent(N, 120, fanin=1000, eta=0.5, theta=1.0, lam_v=0.9, c_rec=1.0, seed=5, sp_beta=sp_beta)
        def load(g, learn=True):
            R.A.v[:] = 0
            d = g.astype(F32) * 3.0
            R.step(d, learn=learn); return R.step(d, learn=learn).copy()
        def hold(T, learn=False):
            out = []
            for _ in range(T):
                dis = np.zeros(N, F32); dis[rng.choice(N, 120, replace=False)] = 0.3
                out.append(R.step(dis, learn=learn).copy())
            return out
        probe = codes[3]
        for i in range(1500):                                  # a stream of lines, each loaded and held 30 ticks
            g = codes[int(rng.integers(len(codes)))]
            load(g); hold(30, learn=learn_hold)
        load(probe, learn=False); p0 = hold(60)[-1]
        # (1) generic-state index: overlap of held states of different lines vs their input overlap
        idx = rng.choice(len(codes), 40, replace=False)
        held = []
        for i in idx:
            load(codes[i], learn=False); held.append(hold(60)[-1])
        hj = np.mean([jac(held[a], held[b]) for a in range(40) for b in range(a + 1, 40)])
        ij = np.mean([jac(codes[idx[a]], codes[idx[b]]) for a in range(40) for b in range(a + 1, 40)])
        # (3) duration: 20-way decode of the held line after T ticks of silence (+ distractors)
        dur = {}
        for T in (10, 60, 120, 240, 480):
            ok = 0
            for trial in range(40):
                cand = rng.choice(len(codes), 20, replace=False)
                load(codes[cand[0]], learn=False); s = hold(T)[-1]
                ok += int(np.argmax([(s & codes[c]).sum() / codes[c].sum() for c in cand]) == 0)
            dur[T] = ok / 40
        # (4) replacement and (5) period-2 index
        rep, p2 = [], []
        for trial in range(40):
            a, b = codes[int(rng.integers(len(codes)))], codes[int(rng.integers(len(codes)))]
            load(a, learn=False); ha = hold(30)
            load(b, learn=False); hb = hold(30)
            rep.append(jac(hb[-1], ha[-1]) - jac(b, a))             # extra overlap beyond the inputs' own overlap
            seq = ha
            p2.append(np.mean([jac(seq[t], seq[t + 2]) - jac(seq[t], seq[t + 1]) for t in range(5, 27)]))
        # (6) drift: the probe's held code before vs after 1,000 more lines (with the same learning regime)
        for i in range(1000):
            load(codes[int(rng.integers(len(codes)))]); hold(30, learn=learn_hold)
        load(probe, learn=False); p1 = hold(60)[-1]
        res[name] = dict(generic=hj, input=ij, dur=dur, rep=float(np.mean(rep)), p2=float(np.mean(p2)), drift=jac(p0, p1))
        print(f"   {name}: generic index (held-state Jaccard between different lines) {hj:.3f} vs input {ij:.3f} | "
              f"20-way decode after silence " + ' '.join(f"{T}:{v:.2f}" for T, v in dur.items()) +
              f" | replacement: extra overlap new-vs-old {np.mean(rep):+.3f} | period-2 index {np.mean(p2):+.4f} | "
              f"drift J(probe before, after 1k lines) {jac(p0, p1):.2f}", flush=True)
    # (2) how many lines can be held at once: load L lines without reset (accumulating), k = 120 x L
    cap = {}
    for L in (1, 2, 3, 4, 6):
        R = Recurrent(N, 60 * L, fanin=1000, eta=0.5, theta=1.0, lam_v=0.97, c_rec=1.0, seed=6)
        hits = []
        for trial in range(30):
            R.A.v[:] = 0
            cand = rng.choice(len(codes), 20, replace=False); loaded = cand[:L]
            for i in loaded:
                d = codes[i].astype(F32) * 3.0
                R.step(d, learn=True); R.step(d, learn=True)
            for _ in range(60):
                dis = np.zeros(N, F32); dis[rng.choice(N, 120, replace=False)] = 0.3
                s = R.step(dis)
            sc = np.array([(s & codes[c]).sum() / codes[c].sum() for c in cand])
            hits.append(len(set(np.argsort(-sc)[:L]) & set(range(L))) / L)
        cap[L] = float(np.mean(hits))
    print('   held at once (load L lines in a row, no reset, k = 60 L, leak 0.97): fraction of the L lines among the top-L '
          'of 20 after 60 ticks: ' + ' '.join(f"L={L}:{v:.2f}" for L, v in cap.items()), flush=True)
    r = res['learning ON during the hold too']
    ok = (r['generic'] <= r['input'] + 0.05 and r['dur'][120] >= 0.9 and r['rep'] <= 0.05 and r['p2'] <= 0.0
          and r['drift'] >= 0.8)
    report(f'M6 reverberation checklist (learning on during holds; set point beta {sp_beta})', ok,
           'criteria: held states no more alike than their inputs (+0.05); 20-way decode >= 0.9 after 120 ticks of '
           'silence; clean replacement (extra overlap <= 0.05); period-2 index <= 0; drift J >= 0.8 after 1k lines')


if __name__ == '__main__':
    which = sys.argv[1:] or ['M1', 'M2', 'M4', 'M5']
    for w in which:
        t0 = time.time()
        {'M1': m1, 'M2': m2, 'M4': m4, 'M5': m5, 'M6': m6, 'M6b': lambda: m6(0.5)}[w]()
        print(f'   ({w}: {time.time() - t0:.0f}s)', flush=True)
    print(f"SUMMARY: {sum(ok for _, ok in RES)}/{len(RES)} passed")
