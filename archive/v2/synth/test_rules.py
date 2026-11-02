# -*- coding: utf-8 -*-
"""Unit tests of every rule in rules.py (SIMULATED; numpy; seconds).  Prints PASS/FAIL with numbers."""
import sys, time
import numpy as np
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from rules import OneShotChain, Delta, MixDict, Hippo, Templates, Critic, Outstar, pair_keys, jaccard

RES = []


def check(name, ok, msg):
    RES.append((name, ok))
    print(f"{'PASS' if ok else 'FAIL'}  {name}: {msg}", flush=True)


rng = np.random.default_rng(0)

# ------------------------------------------------------------------ R1 one-shot chain: ordered copy incl. repeats
def copy_line(items):
    ch = OneShotChain(items); used = set(); out = []; last = None
    for _ in range(len(items) + 2):
        c, pos = ch.candidate(last, used)
        if c is None:
            break
        out.append(c); used.add(pos); last = c
    return out

ok = 0; N = 2000
for _ in range(N):
    L = int(rng.integers(2, 11)); items = list(rng.integers(0, 12, L))      # small vocab -> many repeats
    ok += copy_line(items) == items
check("R1 ordered copy", ok / N >= 0.99, f"{ok}/{N} random lines (len 2-10, vocab 12, repeats) copied exactly in order")
check("R1 repeats", copy_line(list("好不好吃")) == list("好不好吃") and copy_line(list("哈哈哈")) == list("哈哈哈"),
      f"好不好吃 -> {''.join(copy_line(list('好不好吃')))}, 哈哈哈 -> {''.join(copy_line(list('哈哈哈')))}")

# ------------------------------------------------------------------ R2 Delta
D = Delta(5, cap=100, eta=1.0)
A = np.array([3, 7, 9])
D.learn(A, np.eye(5)[2], D.drive(A))
check("R2 one-shot", abs(D.drive(A)[2] - 1) < 1e-5, f"eta=1: drive after one event {D.drive(A)[2]:.3f} (target 1)")
w0 = D.W.copy(); D.learn(A, np.eye(5)[2], D.drive(A))
check("R2 predicted -> no more strengthening", np.abs(D.W - w0).max() < 1e-6, f"max |dW| = {np.abs(D.W - w0).max():.2e}")
D2 = Delta(5, cap=100, eta=0.5); D2.learn(A, np.eye(5)[2], D2.drive(A)); d_before = D2.drive(A)[2]
D2.learn(A, np.zeros(5), D2.drive(A))
check("R2 predicted but absent -> weaken", D2.drive(A)[2] < d_before, f"drive {d_before:.3f} -> {D2.drive(A)[2]:.3f}")
w0 = D2.W.copy(); D2.learn(A, np.eye(5)[1], D2.drive(A), g=0.0)
check("R2 third factor g=0 -> no change", np.abs(D2.W - w0).max() == 0, "no weight changed")

# signed vs excitatory-only vs count rule: facts embedded in a stream of frequent-target noise (cf. compute 1.4)
def lifetime(mode, seed=0, n_fact=200, n_noise=8000, NI=3000, K=40, NO=60, eta=0.4, mature=0.0):
    r = np.random.default_rng(seed)
    pz = 1.0 / np.arange(1, NO + 1); pz /= pz.sum()
    D = Delta(NO, cap=NI, eta=eta, mature=mature)
    facts = [(r.choice(NI, K, replace=False), int(r.choice(NO, p=pz))) for _ in range(n_fact)]
    order = []
    for i in range(n_fact):
        order += [('f', i)] * 1
    stream = [('n', None)] * n_noise
    pos = sorted(r.choice(len(stream) + len(order), len(order), replace=False))
    seq = []; fi = iter(order); ni = iter(stream); P = set(pos)
    for t in range(len(stream) + len(order)):
        seq.append(next(fi) if t in P else next(ni))
    shared = r.choice(NI, 400, replace=False)                 # a shared core: codes overlap like real T codes
    for kind, i in seq:
        if kind == 'f':
            a, y = facts[i]
            for rep in range(3):                              # taught 3x in a row (a short episode)
                t = np.eye(NO)[y]
                if mode == 'count':
                    D.W[np.ix_(a, [y])] += 0.05; continue
                D.learn(a, t, D.drive(a))
                if mode == 'exc':
                    np.maximum(D.W, 0, out=D.W)
        else:
            a = np.r_[r.choice(shared, K // 2, replace=False), r.choice(NI, K // 2, replace=False)]
            y = int(r.choice(NO, p=pz)); t = np.eye(NO)[y]
            if mode == 'count':
                D.W[np.ix_(a, [y])] += 0.05; continue
            D.learn(a, t, D.drive(a))
            if mode == 'exc':
                np.maximum(D.W, 0, out=D.W)
    ok = [int(np.argmax(D.drive(a)) == y) for a, y in facts]
    return np.mean(ok), np.mean(ok[:n_fact // 4]), np.mean(ok[-n_fact // 4:])

t0 = time.time()
res = {m: lifetime(m) for m in ('signed', 'exc', 'count')}
mat = lifetime('signed', mature=20.0)
for m, (a, o, n) in res.items():
    print(f"      lifetime recall [{m:6s}] all {a:.2f}  oldest quarter {o:.2f}  newest quarter {n:.2f}")
print(f"      lifetime recall [signed+maturation 20] all {mat[0]:.2f}  oldest {mat[1]:.2f}  newest {mat[2]:.2f}  ({time.time()-t0:.0f}s)")
check("R2 signed weights beat excitatory-only and counts", res['signed'][0] > res['exc'][0] + 0.15 and res['signed'][0] > res['count'][0] + 0.15,
      f"signed {res['signed'][0]:.2f} vs exc {res['exc'][0]:.2f} vs count {res['count'][0]:.2f}")

# ------------------------------------------------------------------ R3 MixDict
M = MixDict(cap=50)
k1 = pair_keys({'P': [1, 2], 'L': [5]}, [('P', 'L')])
u1 = M.units(k1); u2 = M.units(k1)
check("R3 recruit-once", len(u1) == 5 and np.array_equal(u1, u2), f"{len(u1)} units (2 direct P + 1 direct L + 2 pairs), stable ids")
D3 = Delta(3, cap=50); D3.W[u1] = 1.0
for _ in range(100):
    M.units(pair_keys({'P': [9]}, []))
nd = M.prune(50, readouts=[D3])
check("R3 delete unused", nd == 5 and np.abs(D3.W[u1]).sum() == 0, f"pruned {nd} units unused for >50 events; their readout rows zeroed")

# ------------------------------------------------------------------ R4 Hippo
NB, NS = 300, 400
H = Hippo({'bag': NB, 'scene': NS}, n_dg=3000, k_dg=60, fan=(10, 10), cap=1000, seed=1, gamma=0.999, theta=0.2)
def rand_ep(r):
    b = np.zeros(NB, bool); b[r.choice(NB, 6, replace=False)] = True
    s = np.zeros(NS, bool); s[r.choice(NS, 24, replace=False)] = True
    return b, s
eps = [rand_ep(rng) for _ in range(800)]
codes = [H.dg({'bag': b, 'scene': s}) for b, s in eps]
for i, c in enumerate(codes):
    H.store(c, i)
hit = np.mean([H.recall(c)[0] == i for i, c in enumerate(codes)])
check("R4 one-shot exact recall", hit >= 0.99, f"{hit:.3f} of 800 episodes recalled from the exact cue after ONE storage")
# partial cue: 2 of 6 bag items replaced, scene unchanged; or scene half replaced
hp = []; hs = []
for i, (b, s) in enumerate(eps[:300]):
    b2 = b.copy(); on = np.nonzero(b)[0]; b2[on[:2]] = False; b2[rng.choice(NB, 2)] = True
    hp.append(H.recall(H.dg({'bag': b2, 'scene': s}))[0] == i)
    s2 = s.copy(); on = np.nonzero(s)[0]; s2[on[:12]] = False; s2[rng.choice(NS, 12)] = True
    hs.append(H.recall(H.dg({'bag': b, 'scene': s2}))[0] == i)
check("R4 partial-cue recall", np.mean(hp) >= 0.9 and np.mean(hs) >= 0.9, f"2/6 words changed: {np.mean(hp):.2f}; half the scene changed: {np.mean(hs):.2f}")
# pattern separation: input Jaccard vs DG Jaccard for near-identical inputs
ji, jd = [], []
for b, s in eps[:200]:
    b2 = b.copy(); on = np.nonzero(b)[0]; b2[on[0]] = False; b2[rng.choice(NB)] = True
    x1 = np.r_[b, s]; x2 = np.r_[b2, s]
    ji.append(jaccard(x1, x2)); jd.append(jaccard(H.dg({'bag': b, 'scene': s}), H.dg({'bag': b2, 'scene': s})))
check("R4 pattern separation", np.mean(jd) < np.mean(ji), f"input Jaccard {np.mean(ji):.2f} -> DG Jaccard {np.mean(jd):.2f} (one word changed)")
# decay half-life
H2 = Hippo({'bag': NB}, cap=10, gamma=0.999); H2.store(np.ones(H2.n_dg, bool), 0); H2.tick(693)
check("R4 slow decay", abs(H2.sig[0] - 0.5) < 0.01, f"strength after 693 steps at gamma 0.999: {H2.sig[0]:.3f} (half-life 693)")

# ------------------------------------------------------------------ R5 Templates (synthetic FEAT0-like codes)
NF = 1152
protos = [rng.random(NF) < 0.4 for _ in range(100)]
Z = Templates(NF, theta=0.8, rate=0.5)
for p in protos:
    Z.new(p.astype(np.float32))
def noisy(p, flip):
    q = p.copy(); i = rng.choice(NF, int(flip * NF), replace=False); q[i] = ~q[i]; return q
acc = np.mean([Z.best(noisy(p, 0.05))[0] == j for j, p in enumerate(protos)])
mm = np.mean([Z.best(noisy(p, 0.05))[1] for p in protos]); other = np.mean([Z.match(protos[(j + 1) % 100])[j] for j in range(100)])
check("R5 template detection", acc >= 0.99 and mm > 0.85 and other < 0.6, f"acc {acc:.2f}, own match {mm:.2f}, other char {other:.2f}")

# ------------------------------------------------------------------ R6 critic + reward-modulated actor (contextual bandit)
def bandit(use_reward, seed=0, n=3000):
    r = np.random.default_rng(seed)
    NC, NA = 20, 6
    good = r.integers(0, NA, NC)
    actor = Delta(NA, cap=NC, eta=0.5); cr = Critic(cap=NC, eta=0.3)
    hits = []
    for t in range(n):
        c = int(r.integers(NC)); A = np.array([c])
        d = actor.drive(A) + r.random(NA) * 0.3                         # exploration noise
        a = int(np.argmax(d))
        rew = 1.0 if a == good[c] else 0.0
        delta = cr.update(A, rew)
        if use_reward:
            # three-factor: eligibility = (source, chosen action); g = delta (dopamine-like RPE)
            actor.W[c, a] += 0.5 * delta
        hits.append(rew)
    return np.mean(hits[-500:])
b1, b0 = bandit(True), bandit(False)
check("R6 reward-only learning", b1 >= 0.9 and b0 < 0.35, f"last-500 accuracy with RPE third factor {b1:.2f}, without {b0:.2f} (chance 0.17)")

# ------------------------------------------------------------------ R7 outstar: exact conditional frequencies
O = Outstar(4, 50, m_min=0.0)
pt = rng.random(50)                                   # true P(target unit on | source 0 active)
for _ in range(400):
    O.learn(np.array([0]), (rng.random(50) < pt).astype(np.float32))
err = np.abs(O.W[0] - pt).max()
O2 = Outstar(4, 50, m_min=0.02)
for _ in range(400):
    O2.learn(np.array([1, 2]), (rng.random(50) < pt).astype(np.float32))   # two co-active sources: no credit split
err2 = max(np.abs(O2.W[1] - pt).max(), np.abs(O2.W[2] - pt).max())
check("R7 outstar estimates P(target | source)", err < 0.1 and err2 < 0.15,
      f"max |w - p| = {err:.3f} (1/n plasticity), {err2:.3f} (co-active sources, floor 0.02: no credit split)")

print(f"\n{sum(ok for _, ok in RES)}/{len(RES)} unit tests passed")
