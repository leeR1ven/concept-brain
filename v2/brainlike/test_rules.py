# Unit tests of the three rule families on scripted spike trains.  python3 -B test_rules.py > test_rules.out
import numpy as np
from rules import OrderPath, AssocPath, DecisionPath

R = []


def check(name, ok, info):
    R.append(ok)
    print(("PASS " if ok else "FAIL ") + name + " :: " + info)


# ------------------------------------------------------------------ R1 order rule
def run_seq(path, seq, gap=3, M=1.0):
    """seq of unit ids (or None for silence), one event every `gap` ticks."""
    for u in seq:
        for k in range(gap):
            on = [u] if (k == 0 and u is not None) else []
            path.step(on, M=M)


p = OrderPath(5, eta_f=1.0, eta_b=1.0, W=4)
run_seq(p, [0, 1])
check("R1a one-shot A->B", p.w[0, 1] > 0.99 and p.w[1, 0] == 0, f"w01={p.w[0,1]:.3f} w10={p.w[1,0]:.3f}")
p = OrderPath(5, W=4); run_seq(p, [1, 0])
check("R1b reverse order never learns A->B", p.w[0, 1] == 0, f"w01={p.w[0,1]:.3f} w10={p.w[1,0]:.3f}")
p = OrderPath(5, W=4); run_seq(p, [0, 1, 0, 1, 0, 1])
check("R1c ping-pong A,B,A,B: no two-way loop", min(p.w[0, 1], p.w[1, 0]) < 0.01,
      f"w01={p.w[0,1]:.3f} w10={p.w[1,0]:.3f} (the last order wins, the reverse is erased)")
p = OrderPath(5, W=4)
for t in range(6):
    p.step([0, 1] if t == 0 else [])
check("R1d same-tick onsets do not link", p.w.max() == 0, f"max w={p.w.max():.3f}")
p = OrderPath(5, W=4); run_seq(p, [0, None, None, 1])
check("R1e outside the window (9 ticks) no link", p.w[0, 1] == 0, f"w01={p.w[0,1]:.3f}")
p = OrderPath(5, W=4); run_seq(p, [0, 1], M=0.0)
check("R1f gate M=0 blocks learning", p.w.max() == 0, f"max w={p.w.max():.3f}")
# repeated syllable: 好不好吃 -> echo after 好 is 吃 (back term erases 好->不 when 不->好 follows)
p = OrderPath(5, W=4); H, BU, CH = 0, 1, 2
run_seq(p, [H, BU, H, CH])
check("R1g 好不好吃: successor(好)=吃, successor(不)=好", p.w[H].argmax() == CH and p.w[BU].argmax() == H and p.w[H, BU] == 0,
      f"w[好]={np.round(p.w[H,:3],2)} w[不]={np.round(p.w[BU,:3],2)}")
# slow statistics with presynaptic normalisation: w_ab -> P(b | a)
rng = np.random.default_rng(0)
p = OrderPath(4, eta_f=0.02, eta_b=0.0, W=4, norm='pre', B=1.0)
seq = []
for _ in range(3000):
    seq += [0, (1 if rng.random() < 0.7 else 2), 3]
run_seq(p, seq)
check("R1h slow + pre-normalisation estimates P(next | a)", abs(p.w[0, 1] - 0.7) < 0.08 and abs(p.w[0, 2] - 0.3) < 0.08,
      f"w01={p.w[0,1]:.3f} (0.7) w02={p.w[0,2]:.3f} (0.3) rowsum={p.w[0].sum():.3f}")

# ------------------------------------------------------------------ R2 association rule
rng = np.random.default_rng(1)
# (i) one-shot, CA3 regime: sparse (2 %), one presentation per pattern; 4 patterns per unit on average, the budget
#     holds B/(K*eta) = 5 patterns per unit (beyond that: graceful palimpsest forgetting, by design)
N, K, NP = 2000, 40, 200
A = AssocPath(N, N, eta=0.005, B=1.0)
pats = [rng.choice(N, K, replace=False) for _ in range(NP)]
maxsum = 0.0
for P in pats:
    x = np.zeros(N, bool); x[P] = True
    A.learn(x, x); maxsum = max(maxsum, A.w.sum(0).max())
check("R2a per-target totals <= budget at EVERY learning tick", maxsum <= 1.0 + 1e-4, f"max column sum={maxsum:.4f} (B=1)")
def complete(A, P, N, K, steps=3):
    x = np.zeros(N, bool); x[P[:K // 2]] = True                 # half cue
    for _ in range(steps):
        d = A.current(x) + rng.uniform(0, 1e-4, N)
        y = np.zeros(N, bool); y[np.argpartition(-d, K)[:K]] = True; x = y
    return len(set(np.nonzero(x)[0]) & set(P)) / K
ov = [complete(A, P, N, K) for P in pats]
ok_old = sum(o >= 0.9 for o in ov[:40]); ok_new = sum(o >= 0.9 for o in ov[-40:])
check("R2b one-shot assemblies (200 patterns, 2% active) complete from a half cue", ok_old >= 36 and ok_new >= 36,
      f"oldest 40: {ok_old}/40, newest 40: {ok_new}/40, mean overlap {np.mean(ov):.2f} (chance 0.02)")
# (ii) slow concepts: 30 patterns at 5 %, each presented 12 times interleaved with noise, small eta
N, K = 1000, 50
A3 = AssocPath(N, N, eta=0.002, B=1.0)
pats3 = [rng.choice(N, K, replace=False) for _ in range(30)]
for rep in range(12):
    for P in rng.permutation(len(pats3)):
        x = np.zeros(N, bool); x[pats3[P]] = True
        x[rng.choice(N, 10)] = True                              # 20 % noise units
        A3.learn(x, x)
ov3 = [complete(A3, P, N, K) for P in pats3]
check("R2b' slow interleaved concepts (30 x 12, 20% noise) complete from a half cue", np.mean([o >= 0.9 for o in ov3]) >= 0.9,
      f"completed {sum(o>=0.9 for o in ov3)}/30, mean overlap {np.mean(ov3):.2f}")
T = AssocPath(50, 1, eta=0.1, B=1.0)
proto = rng.random(50) < 0.3
for _ in range(200):
    x = proto ^ (rng.random(50) < 0.05)
    T.learn_template(x, 0)
c = np.corrcoef(T.w[:, 0], proto)[0, 1]
check("R2c template form converges to the prototype, sum = B", c > 0.9 and abs(T.w[:, 0].sum() - 1.0) < 1e-4,
      f"corr={c:.3f} sum={T.w[:,0].sum():.4f}")
A2 = AssocPath(10, 2, eta=0.5, B=1.0); A2.learn(np.ones(10, bool), np.array([True, False]), M=0.0)
check("R2d gate M=0 blocks learning", A2.w.max() == 0, "")

# ------------------------------------------------------------------ R3 decision rule
D = DecisionPath(20, 3, eta=0.5)
x = np.zeros(20, bool); x[:5] = True
errs = []
for _ in range(20):
    D.tag(x); e = D.learn(np.array([1, 0, 0])); errs.append(abs(e[0]))
D.tag(x); p = D.drive()
check("R3a delta: prediction converges to the target", p[0] > 0.95 and abs(p[1]) < 0.05, f"p={np.round(p,3)}")
w_before = D.w.copy(); D.tag(x); D.learn(np.array([1, 0, 0]))
check("R3b already predicted -> (almost) no more strengthening", np.abs(D.w - w_before).max() < 0.01,
      f"max dw={np.abs(D.w-w_before).max():.4f}")
D.tag(x); D.learn(np.array([0, 0, 0]))
D.tag(x); p2 = D.drive()
check("R3c predicted but absent -> weakens", p2[0] < p[0] - 0.2, f"p before={p[0]:.3f} after={p2[0]:.3f}")
D2 = DecisionPath(20, 3, eta=0.5); D2.tag(x); D2.learn(np.array([1, 0, 0]), M=0.0)
check("R3d third factor M=0 -> no learning", D2.w.max() == 0, "")
# signed: context A -> y, context A+B -> not y (needs a negative weight from B)
D3 = DecisionPath(10, 1, eta=0.5)
xa = np.zeros(10, bool); xa[:3] = True; xab = xa.copy(); xab[3:6] = True
for _ in range(200):
    D3.tag(xa); D3.learn(np.array([1.0])); D3.tag(xab); D3.learn(np.array([0.0]))
D3.tag(xa); pa = D3.drive()[0]; D3.tag(xab); pab = D3.drive()[0]
check("R3e signed weights learn an exception (A -> y, A+B -> not y)", pa > 0.8 and pab < 0.2 and D3.w[3:6].min() < 0,
      f"p(A)={pa:.2f} p(A+B)={pab:.2f} min w(B)={D3.w[3:6].min():.2f}")
# delayed credit with a decaying trace: source held 10 ticks, target 5-20 ticks later, M on.  Functional measure:
# the prediction the trace makes at the target tick (p = w . e), relative to the target 1.
for lag in (5, 10, 20):
    res = {}
    for M in (1.0, 0.0):
        Dt = DecisionPath(4, 1, eta=0.5, lam=0.9)
        for tr in range(30):
            Dt.e[:] = 0
            for t in range(10 + lag):
                xs = np.zeros(4, bool); xs[0] = t < 10
                Dt.tag(xs)
                if t == 10 + lag - 1:
                    e_target = Dt.e.copy()
                    Dt.learn(np.array([1.0]), M=M)
        res[M] = float(e_target @ Dt.w[:, 0])
    check(f"R3f trace bridges a {lag}-tick delay (M on) and learns nothing with M off", res[1.0] > 0.5 and res[0.0] == 0,
          f"prediction at the target tick: M on {res[1.0]:.2f}, M off {res[0.0]:.2f}")

print(f"\n{sum(R)}/{len(R)} unit tests passed")
