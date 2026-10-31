# -*- coding: utf-8 -*-
"""Unit tests of the ONE rule (PR, pr.py) on scripted spike trains.  Each test prints numbers and PASS/FAIL against a
criterion fixed before running.  Usage: OMP_NUM_THREADS=1 python3 -B test_rule.py > test_rule.out"""
import sys
import numpy as np
sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__)))
from pr import Predictor, Trace, F32  # noqa: E402

RES = []


def report(name, ok, msg):
    RES.append((name, ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {msg}", flush=True)


def count_rule(w, eta=0.1, w_ref=0.2):
    """net.py forward increment (concave, no ceiling) for comparison."""
    return w + eta / (1 + w / w_ref)


# ------------------------------------------------------------------ T1 probability estimate, T2 predicted -> stop
def t1_t2():
    rng = np.random.default_rng(0)
    out = []
    for q in (0.2, 0.5, 0.9):
        P = Predictor(2, eta=0.05, theta=1.0, bias=False)          # targets: Y, other
        P.add('S', 1)
        ps = []
        for tr in range(3000):
            P.predict({'S': np.ones(1, F32)})
            ps.append(P.p[0])
            y = np.array([1, 0], F32) if rng.random() < q else np.array([0, 1], F32)
            P.learn(y)
        out.append((q, float(np.mean(ps[-1000:]))))
    ok = all(abs(a - b) < 0.05 for a, b in out)
    report('T1 weight = P(target | source)', ok,
           '  '.join(f"q={q:.1f} -> p={p:.3f}" for q, p in out) + '   (criterion |p-q| < 0.05; net.py counts have no such fixed point)')
    # T2: deterministic; weight after 100 vs 1000 events
    P = Predictor(1, eta=0.3, theta=1.0, bias=False); P.add('S', 1)
    w_at = {}
    wc = 0.0; wc_at = {}
    for tr in range(1, 1001):
        P.predict({'S': np.ones(1, F32)}); P.learn(np.ones(1, F32))
        wc = count_rule(wc)
        if tr in (10, 100, 1000):
            w_at[tr] = float(P.paths['S'].W[0, 0]); wc_at[tr] = wc
    ok = abs(w_at[100] - w_at[1000]) < 1e-6 and abs(w_at[1000] - 1.0) < 1e-3
    report('T2 predicted -> no more strengthening', ok,
           f"PR w after 10/100/1000 events {w_at[10]:.4f}/{w_at[100]:.4f}/{w_at[1000]:.4f}; "
           f"net.py count {wc_at[10]:.3f}/{wc_at[100]:.3f}/{wc_at[1000]:.3f} (keeps growing)")


# ------------------------------------------------------------------ T3 predicted but absent -> weaken; learned inhibition
def t3():
    rng = np.random.default_rng(1)
    res = {}
    for signed in (True, False):
        P = Predictor(2, eta=0.1, theta=1.0, bias=False)            # Y, N
        P.add('ctx', 2)                                              # sources A, C
        for tr in range(4000):
            withC = rng.random() < 0.5
            x = np.array([1, 1 if withC else 0], F32)
            P.predict({'ctx': x})
            y = np.array([0, 1], F32) if withC else np.array([1, 0], F32)
            P.learn(y)
            if not signed:
                np.maximum(P.paths['ctx'].W, 0, out=P.paths['ctx'].W)
        pa = P.predict({'ctx': np.array([1, 0], F32)})[0]
        pac = P.predict({'ctx': np.array([1, 1], F32)})[0]
        res[signed] = (float(pa), float(pac), float(P.paths['ctx'].W[1, 0]))
    s, e = res[True], res[False]
    ok = s[0] > 0.9 and s[1] < 0.1 and s[2] < 0 and e[1] > 0.3
    report('T3 predicted-but-absent weakens below 0 (learned inhibition)', ok,
           f"signed: p(Y|A)={s[0]:.2f} p(Y|A+C)={s[1]:.2f} w(C->Y)={s[2]:+.2f};  excitatory-only: p(Y|A)={e[0]:.2f} "
           f"p(Y|A+C)={e[1]:.2f} (the exception cannot be stored)")


# ------------------------------------------------------------------ T4 constant source -> base rate, no runaway
def t4():
    rng = np.random.default_rng(2)
    nT = 10                                    # 10 possible targets; Y = 0 occurs with base rate 0.1 at events
    P = Predictor(nT, eta=0.05, theta=1.0, bias=False)
    P.add('const', 1); P.add('ctx', 20)
    hist = []
    for tr in range(6000):
        c = rng.integers(20)
        x = np.zeros(20, F32); x[c] = 1
        P.predict({'const': np.ones(1, F32), 'ctx': x})
        tgt = c % nT if rng.random() < 0.7 else rng.integers(nT)        # context predicts its class 70 %
        y = np.zeros(nT, F32); y[tgt] = 1
        P.learn(y)
        hist.append(float(np.abs(P.paths['const'].W).max()))
    wc = P.paths['const'].W[0]
    ok = hist[-1] < 0.5 and abs(hist[-1] - hist[3000]) < 0.1
    report('T4 constant source -> bounded base-rate weight', ok,
           f"|w(const)| max over targets: {hist[999]:.3f} at 1k, {hist[2999]:.3f} at 3k, {hist[-1]:.3f} at 6k events; "
           f"mean w(const) {wc.mean():+.3f} (base rate 0.10 shared with the context weights)")


# ------------------------------------------------------------------ T5 held source -> target onset; T7 reverse order
def t5_t7():
    res = {}
    for name, S_on, Y_t in (('held S 0..9, Y@6', range(0, 10), 6), ('fresh S@5, Y@6', [5], 6),
                            ('reverse Y@6, S 7..9', range(7, 10), 6)):
        P = Predictor(2, eta=0.2, theta=1.0, bias=False)            # Y and a competitor event N (at tick 15)
        P.add('S', 1)
        tr_S = Trace(1, 0.0)
        for trial in range(200):
            for t in range(20):
                xprev = tr_S.x.copy()
                tr_S.step([1.0 if t in S_on else 0.0])
                if t == Y_t:                                         # event tick of the group: Y happened
                    P.predict({'S': xprev}); P.learn(np.array([1, 0], F32))
                if t == 15:                                          # another event: N happened
                    P.predict({'S': xprev}); P.learn(np.array([0, 1], F32))
        # test: prediction of Y at its tick given S's script
        tr_S.reset(); pY = 0.0
        for t in range(20):
            xprev = tr_S.x.copy(); tr_S.step([1.0 if t in S_on else 0.0])
            if t == Y_t:
                pY = float(P.predict({'S': xprev})[0])
        res[name] = (pY, float(P.paths['S'].W[0, 0]))
    ok = res['held S 0..9, Y@6'][0] > 0.9 and res['fresh S@5, Y@6'][0] > 0.9 and res['reverse Y@6, S 7..9'][1] == 0.0
    report('T5/T7 held source learns the onset; reverse order gets nothing', ok,
           '  '.join(f"{k}: p(Y)={a:.2f} w={b:+.2f}" for k, (a, b) in res.items()) +
           '   (net.py exact: held 0.000, fresh 0.712, reverse 0.000 -- skeptic S5)')


# ------------------------------------------------------------------ T6 delays: trace + distractors
def t6():
    rows = []
    for lam in (0.0, 0.5, 0.8):
        accs = []
        for k in (1, 2, 4, 6):
            rng = np.random.default_rng(10 + k)
            nD = 20
            P = Predictor(2, eta=0.1, theta=1.0, bias=True)          # targets: Y, N
            P.add('S', 1); P.add('D', nD)
            trS, trD = Trace(1, lam), Trace(nD, lam)
            for trial in range(600):
                trS.reset(); trD.reset()
                cueS = rng.random() < 0.5                             # S present in half of the trials
                dist = {int(rng.integers(1, max(k, 2))): int(rng.integers(nD))} if k > 1 else {}
                for t in range(k + 1):
                    xs = {'S': trS.x.copy(), 'D': trD.x.copy()}
                    if t == k:
                        P.predict(xs)
                        P.learn(np.array([1, 0], F32) if cueS else np.array([0, 1], F32))
                    trS.step([1.0 if (t == 0 and cueS) else 0.0])
                    dd = np.zeros(nD, F32)
                    if t in dist:
                        dd[dist[t]] = 1
                    trD.step(dd)
            # test: S at 0, nothing else, Y-prediction at k; and no S
            out = []
            for cue in (True, False):
                trS.reset(); trD.reset(); p = 0
                for t in range(k + 1):
                    xs = {'S': trS.x.copy(), 'D': trD.x.copy()}
                    if t == k:
                        p = float(P.predict(xs)[0])
                    trS.step([1.0 if (t == 0 and cue) else 0.0]); trD.step(np.zeros(nD, F32))
                out.append(p)
            accs.append(out[0] - out[1])                               # discrimination S vs no S
        rows.append((lam, accs))
    ok = rows[0][1][0] > 0.8 and rows[0][1][2] < 0.1 and rows[2][1][2] > 0.5
    report('T6 credit across a delay needs the trace', ok,
           '  '.join(f"lam={lam}: " + '/'.join(f"{a:.2f}" for a in accs) for lam, accs in rows) +
           '   (p(Y|S) - p(Y|no S) at delay 1/2/4/6 ticks, a random distractor in between)')


# ------------------------------------------------------------------ T8 third factor: freeze, reward, self-reinforcement
def t8():
    # (a) M = 0 freezes
    P = Predictor(2, eta=0.5, bias=False); P.add('S', 1)
    P.predict({'S': np.ones(1, F32)}); P.learn(np.array([1, 0], F32), M=0.0)
    froze = float(np.abs(P.paths['S'].W).sum()) == 0.0
    # (b) contextual bandit: 8 contexts, 4 actions, reward 1 iff action = ctx % 4.  The actor is the predictor whose
    #     target is its OWN produced action; M = delta = r - V(ctx) (critic = PR unit with target r).
    def run(mode, seed=0, n=4000):
        rng = np.random.default_rng(seed)
        A = Predictor(4, eta=0.2, bias=False); A.add('c', 8)
        C = Predictor(1, eta=0.2, bias=False); C.add('c', 8)
        hits, acts = [], []
        for tr in range(n):
            c = rng.integers(8); x = np.zeros(8, F32); x[c] = 1
            p = A.predict({'c': x}); v = C.predict({'c': x})[0]
            z = (p + 0.05) ** 2; z = z / z.sum()                      # "say the prediction" (sampled)
            a = int(rng.choice(4, p=z)); r = float(a == c % 4)
            y = np.zeros(4, F32); y[a] = 1
            if mode == 'M=1 (own output as target)':
                A.learn(y, M=1.0)
            elif mode == 'M=r':
                A.learn(y, M=r)
            else:
                A.learn(y, M=r - v); C.learn(np.array([r], F32))
            hits.append(r); acts.append(a)
        return float(np.mean(hits[-1000:])), np.bincount(acts[-1000:], minlength=4) / 1000
    out = {m: run(m) for m in ('M=1 (own output as target)', 'M=r', 'M=delta=r-V')}
    ok = froze and out['M=delta=r-V'][0] > 0.9 and out['M=1 (own output as target)'][0] < 0.5
    report('T8 third factor: M=0 freezes; own targets need the RPE', ok,
           f"M=0 froze: {froze};  " + '  '.join(f"{m}: reward {h:.2f}" for m, (h, _) in out.items()) +
           '  (chance 0.25; M=1 = self-reinforcement drift, the minimal verifier\'s 你->的/了)')


# ------------------------------------------------------------------ T9 one-shot storage capacity on sparse codes
def t9():
    rng = np.random.default_rng(3)
    N, k, nC = 2000, 40, 200
    rows = []
    for M in (100, 300, 1000, 3000):
        codes = [rng.choice(N, k, replace=False) for _ in range(M)]
        lab = rng.integers(nC, size=M)
        res = {}
        for rule in ('PR eta=1', 'PR exc-only', 'count'):
            P = Predictor(nC, eta=1.0, theta=1.0, bias=False); P.add('h', N)
            for m in range(M):
                x = np.zeros(N, F32); x[codes[m]] = 1 / np.sqrt(k)
                y = np.zeros(nC, F32); y[lab[m]] = 1
                if rule == 'count':
                    P.paths['h'].W[codes[m], lab[m]] += 1.0
                else:
                    P.predict({'h': x}); P.learn(y)
                    if rule == 'PR exc-only':
                        np.maximum(P.paths['h'].W, 0, out=P.paths['h'].W)
            ok_n = 0
            for m in range(M):
                x = np.zeros(N, F32); x[codes[m]] = 1 / np.sqrt(k)
                P.predict({'h': x})
                ok_n += int(np.argmax(P.d) == lab[m])
            res[rule] = ok_n / M
        rows.append((M, res))
    ok = rows[1][1]['PR eta=1'] > 0.95 and rows[3][1]['PR eta=1'] >= rows[3][1]['count']
    report('T9 one-shot storage on sparse codes (N 2000, k 40, 200 labels)', ok,
           '  '.join(f"M={M}: " + ', '.join(f"{r} {v:.2f}" for r, v in res.items()) for M, res in rows))


# ------------------------------------------------------------------ T9b correlated codes + skewed (Zipf) labels
def t9b():
    rng = np.random.default_rng(4)
    N, k, nC = 2000, 40, 200
    core = rng.choice(N, 200, replace=False)                      # a shared pool: half of every code comes from it
    rows = []
    zipf = 1.0 / np.arange(1, nC + 1); zipf /= zipf.sum()
    for M in (300, 1000, 3000):
        codes = [np.r_[rng.choice(core, k // 2, replace=False), rng.choice(N, k // 2, replace=False)] for _ in range(M)]
        lab = rng.choice(nC, size=M, p=zipf)
        res = {}
        for rule in ('PR eta=1', 'PR eta=0.3 x3', 'PR exc-only', 'count'):
            P = Predictor(nC, eta=1.0, theta=1.0, bias=False); P.add('h', N)
            reps = 3 if 'x3' in rule else 1
            for rep in range(reps):
                for m in range(M):
                    x = np.zeros(N, F32); x[codes[m]] = 1 / np.sqrt(k)
                    y = np.zeros(nC, F32); y[lab[m]] = 1
                    if rule == 'count':
                        P.paths['h'].W[codes[m], lab[m]] += 1.0
                    else:
                        P.predict({'h': x}); P.learn(y, eta=0.3 if 'x3' in rule else 1.0)
                        if rule == 'PR exc-only':
                            np.maximum(P.paths['h'].W, 0, out=P.paths['h'].W)
            ok_n = 0
            for m in range(M):
                x = np.zeros(N, F32); x[codes[m]] = 1 / np.sqrt(k)
                P.predict({'h': x})
                ok_n += int(np.argmax(P.d) == lab[m])
            res[rule] = ok_n / M
        rows.append((M, res))
    ok = rows[1][1]['PR eta=1'] > rows[1][1]['count'] + 0.1
    report('T9b one-shot storage, correlated codes (half from a shared core) + Zipf labels', ok,
           '  '.join(f"M={M}: " + ', '.join(f"{r} {v:.2f}" for r, v in res.items()) for M, res in rows) +
           '   (criterion: PR beats counts by > 0.1 at M=1000)')


if __name__ == '__main__':
    t1_t2(); t3(); t4(); t5_t7(); t6(); t8(); t9(); t9b()
    print(f"SUMMARY: {sum(ok for _, ok in RES)}/{len(RES)} passed")
