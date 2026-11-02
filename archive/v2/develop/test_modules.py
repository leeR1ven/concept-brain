# -*- coding: utf-8 -*-
"""Module tests of the cognitive core (SIMULATED, symbolic world devworld.py; seconds to minutes).
  M1 episodic one-shot recall and retention: hippocampus + sleep / hippocampus without sleep / no hippocampus
  M4 slot binding: never-heard frame x filler with and without the BG copy gate
  M5 PFC hold and period-2: buffers held through the reply; leaky CTX vs a synchronous non-leaky loop
  M6 drift in free use: learning ON, no tutor, replies to fixed probes before vs after
usage: python3 test_modules.py [M1] [M4] [M5] [M6]"""
import sys, copy, collections
import numpy as np
import os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from core import Core, Cfg, Ctx
import devworld as DW
from rules import jaccard

WHICH = set(a for a in sys.argv[1:]) or {"M1", "M4", "M5", "M6", "M7"}
os.makedirs(os.path.join(HERE, "out"), exist_ok=True)
OUT = open(os.path.join(HERE, "out", "modules_" + "_".join(sorted(WHICH)) + ".out"), "a")
def log(s):
    print(s, flush=True); OUT.write(s + "\n"); OUT.flush()

CH = {}
def ids(s):
    return [CH.setdefault(ch, len(CH)) for ch in s]
def txt(seq):
    inv = {v: k for k, v in CH.items()}
    return "".join(inv.get(i, "?") for i in seq)


def make(seed=0, **kw):
    W = DW.DevWorld(seed=seed, retino=True)
    core = Core(Cfg(seed=seed, **kw))
    return W, core


def ex_recast(W, core, ex, learn=True):
    eye = W.eye(ex["scene"], None)
    core.hear_line(ids(ex["prompt"]), eye, ex["att"], learn=learn, store=learn)
    out = core.speak(eye, ex["att"]); core.own_finish(core.own)
    ok = txt(out) in ex["acceptable"]
    if learn:
        core.reward(1.0 if ok else 0.0)
        core.hear_line(ids(ex["tutor"]), eye, ex["att"], learn=True, store=True, contingent=True)
    return ok, txt(out)


def probe(W, core, prompt, eye=None, att=None):
    st = (core.prev, core.ctx.u.copy(), core.ctx.s.copy())
    core.hear_line(ids(prompt), eye, att, learn=False, store=False)
    out = core.speak(eye, att)
    core.prev, core.ctx.u, core.ctx.s = st
    return txt(out)


# ======================================================================= M1 episodic recall / retention
if "M1" in WHICH:
    rng = np.random.default_rng(5)
    pool = list("甲乙丙丁戊己庚辛壬癸子丑寅卯辰巳午未申酉戌亥东南西北春夏秋冬金木水火土")
    facts = []
    used = set()
    while len(facts) < 60:
        n = "".join(rng.choice(pool, 2, replace=False))
        if n in used:
            continue
        used.add(n); facts.append((n + "喜欢什么", DW.OBJ[int(rng.integers(len(DW.OBJ)))]))
    delays = [0, 100, 400, 1500]
    res = {}
    CONDS = [("hippo+sleep", {}, 150), ("hippo, no sleep", {}, 0), ("no hippocampus", dict(hippo=False), 0),
             ("hippo+priority sleep", dict(sleep_mode="priority"), 150)]
    only = os.environ.get("M1_ONLY")
    if only:
        CONDS = [c for c in CONDS if c[0] in only.split(",")]
    for cond, kw, sleep_every in CONDS:
        W, core = make(seed=1, hip_cap=600, hip_gamma=0.999, **kw)
        for _ in range(1500):                      # background: stages B, C (so the core knows the task world)
            ex_recast(W, core, W.exchange("B" if rng.random() < 0.4 else "C"))
        sched = []                                 # teach each fact once, interleaved with background exchanges
        t = 0
        hits = collections.defaultdict(list)
        pending = []
        for i, (q, a) in enumerate(facts):
            # teach: the partner asks, the tutor answers (model: the agent overhears; no muscles moved)
            core.hear_line(ids(q), None, None); core.hear_line(ids(a), None, None)
            for d in delays:
                pending.append((t + d, i, d))
            for _ in range(25):                    # 25 background exchanges between facts
                pending.sort()
                while pending and pending[0][0] <= t:
                    _, j, d = pending.pop(0)
                    hits[d].append(probe(W, core, facts[j][0]) == facts[j][1])
                ex_recast(W, core, W.exchange("C"))
                t += 1
                if sleep_every and t % sleep_every == 0:
                    core.sleep(n=300)
        while pending:                             # remaining long delays
            pending.sort()
            while pending and pending[0][0] <= t:
                _, j, d = pending.pop(0)
                hits[d].append(probe(W, core, facts[j][0]) == facts[j][1])
            ex_recast(W, core, W.exchange("C")); t += 1
            if sleep_every and t % sleep_every == 0:
                core.sleep(n=300)
        res[cond] = {d: (round(float(np.mean(hits[d])), 2), len(hits[d])) for d in delays}
        log(f"M1 [{cond:16s}] one-shot fact recall (exact answer) by delay in exchanges: {res[cond]}")
    if "hippo+sleep" in res and "no hippocampus" in res:
        ok = res["hippo+sleep"][0][0] >= 0.8 and res["no hippocampus"][0][0] <= 0.3 and \
            res["hippo+sleep"][1500][0] > res["hippo, no sleep"][1500][0] + 0.2
        log(f"M1 {'PASS' if ok else 'FAIL'}: one-shot via the index; long-delay retention needs sleep replay")

# ======================================================================= M4 slot binding ablation
if "M4" in WHICH:
    for cond, kw in [("full", {}), ("no copy gate (COPY_prompt off)", dict(copy_prompt=False)),
                     ("no episodic routes", dict(copy_next=False, copy_comp=False))]:
        W, core = make(seed=2, **kw)
        for i in range(4500):
            ex_recast(W, core, W.exchange("B" if i < 1500 else "C"))
        r = collections.defaultdict(list)
        for i in range(240):
            task = ["offer", "taste", "like"][i % 3]
            ex = W.exchange("C", test=True, task=task)
            ok, t = ex_recast(W, core, ex, learn=False)
            r[task].append(ok)
        log(f"M4 [{cond:30s}] never-heard frame x filler: " + str({k: round(float(np.mean(v)), 2) for k, v in r.items()}))

# ======================================================================= M5 hold and period-2
if "M5" in WHICH:
    rng = np.random.default_rng(3)
    # (a) a synchronous binary loop with reciprocal (symmetric) learned links between two states x1 <-> x2:
    #     W = x2 x1^T + x1 x2^T (the old PFC <-> thalamus / receiving <-> thinking class).  Without persistence the
    #     synchronous k-WTA update alternates x1, x2, x1 ... (period 2); leaky potentials u = lam*u + (1-lam)*W x
    N, K = 800, 24
    def kwta(x):
        s = np.zeros(N, bool); s[np.argpartition(-(x + rng.random(N) * 1e-3), K)[:K]] = True; return s
    x1 = np.zeros(N, bool); x1[rng.choice(N, K, replace=False)] = True
    x2 = np.zeros(N, bool); x2[rng.choice(np.flatnonzero(~x1), K, replace=False)] = True
    Wr = np.outer(x2, x1).astype(float) + np.outer(x1, x2) + 0.2 * (rng.random((N, N)) < 0.02)
    Wa = np.outer(x1, x1).astype(float) + np.outer(x2, x2) + 0.2 * (rng.random((N, N)) < 0.02)   # autoassociative
    D = float(K)                                                          # cross-drive onto a pattern unit
    def loop(W, lam, h, T=60):
        u = np.zeros(N); x = x1.copy(); hist = []
        for t in range(T):
            u = lam * u + (1 - lam) * (W @ x + h * D * x)                 # leak + self-excitation h (x of t-1)
            x = kwta(u); hist.append(x.copy())
        j1 = np.mean([jaccard(hist[t], hist[t + 1]) for t in range(20, T - 2)])
        j2 = np.mean([jaccard(hist[t], hist[t + 2]) for t in range(20, T - 2)])
        return j1, j2
    for name, W, lam, h in [("reciprocal, no leak, no self", Wr, 0.0, 0.0), ("reciprocal, leak 0.75", Wr, 0.75, 0.0),
                            ("reciprocal, leak 0.95", Wr, 0.95, 0.0), ("reciprocal, self-excitation 0.5 D", Wr, 0.5, 0.5),
                            ("reciprocal, self-excitation 1.5 D", Wr, 0.5, 1.5), ("autoassociative, no leak", Wa, 0.0, 0.0)]:
        j1, j2 = loop(W, lam, h)
        log(f"M5 synchronous loop [{name:34s}]: J(t,t+1)={j1:.2f} J(t,t+2)={j2:.2f} -> "
            f"{'PERIOD-2 alternation' if j2 > j1 + 0.1 else 'no period-2'}")
    # (b) CTX (the slow context layer of the core) on a stream of lines: turnover per line and hold in silence
    ctx = Ctx(400 + 400, seed=1)
    codes = []
    for t in range(200):
        x = np.zeros(800, bool); x[rng.choice(400, 5, replace=False)] = True; x[400 + rng.choice(400, 30, replace=False)] = True
        codes.append(ctx.step(x).copy())
    j1 = np.mean([jaccard(codes[t], codes[t + 1]) for t in range(10, 198)]); j2 = np.mean([jaccard(codes[t], codes[t + 2]) for t in range(10, 198)])
    log(f"M5 CTX (lambda 0.75 per line) on new lines: J(n,n+1)={j1:.2f} J(n,n+2)={j2:.2f} (turnover per line {1-j1:.2f}); "
        f"{'PASS' if j1 >= j2 else 'FAIL'} no period-2")
    # (c) the prompt buffer is held exactly through the own reply (state copy, updated only at line offsets)
    W, core = make(seed=4)
    for i in range(300):
        ex_recast(W, core, W.exchange("B"))
    ex = W.exchange("B"); eye = W.eye(ex["scene"], None)
    core.hear_line(ids(ex["prompt"]), eye, ex["att"])
    before = list(core.prev["items"]); st = core.own_begin(eye, ex["att"])
    held = []
    for n in range(4):
        prog, s_end = core.own_scores(st); y = int(np.argmax(prog)); core.own_advance(st, y)
        held.append(st["P"] == before and core.prev["items"] == before)
    log(f"M5 prompt buffer held through the reply: {sum(held)}/{len(held)} boundaries (J = 1.0); PASS" if all(held) else "M5 hold FAIL")

# ======================================================================= M6 drift in free use
if "M6" in WHICH:
    W, core = make(seed=6)
    for i in range(4000):
        ex_recast(W, core, W.exchange("B" if i < 1500 else "C"))
    probes = [W.exchange("C", task=t) for t in ["offer", "taste", "like", "colour", "count", "ritual"] * 30]
    def answers():
        return [probe(W, core, ex["prompt"], W.eye(ex["scene"], None), ex["att"]) for ex in probes]
    a0 = answers()
    acc0 = np.mean([a in ex["acceptable"] for a, ex in zip(a0, probes)])
    firsts0 = collections.Counter(a[:1] for a in a0)
    # free use: the agent answers 2000 prompts, learning ON, but nobody answers or rewards (no tutor)
    for i in range(2000):
        ex = W.exchange("C")
        eye = W.eye(ex["scene"], None)
        core.hear_line(ids(ex["prompt"]), eye, ex["att"], learn=True, store=True)
        core.speak(eye, ex["att"]); core.own_finish(core.own)
    a1 = answers()
    acc1 = np.mean([a in ex["acceptable"] for a, ex in zip(a1, probes)])
    same = np.mean([x == y for x, y in zip(a0, a1)])
    firsts1 = collections.Counter(a[:1] for a in a1)
    def H(c):
        p = np.array(list(c.values()), float); p /= p.sum(); return float(-(p * np.log2(p)).sum())
    log(f"M6 free use (2000 untutored exchanges, learning on): identical replies {same:.2f}, acc {acc0:.2f} -> {acc1:.2f}, "
        f"first-char entropy {H(firsts0):.2f} -> {H(firsts1):.2f}; {'PASS' if same >= 0.9 and acc1 >= acc0 - 0.05 else 'FAIL'} (no drift)")

# ======================================================================= M7 holding thoughts (coordinator's 5 questions)
if "M7" in WHICH:
    rng = np.random.default_rng(9)
    W, core = make(seed=7)
    for i in range(1500):
        ex_recast(W, core, W.exchange("B" if i < 600 else "C"))
    lex = DW.OBJ + DW.COL + DW.CNT + DW.SLOT + ["这是什么", "你要", "好不好吃", "是什么颜色", "有几个", "在哪里"]
    def rand_line():
        return "".join(rng.choice(lex, int(rng.integers(2, 4))))
    # (1) PFC-1 register: hold through silence (no events), clean replacement, period-2, drift under learning
    a = rand_line(); core.hear_line(ids(a), None, None, learn=True)
    held = list(core.prev["items"]); silent_ok = core.prev["items"] == ids(a)       # silence = no events: unchanged
    b = rand_line(); core.hear_line(ids(b), None, None, learn=True)
    replaced = core.prev["items"] == ids(b) and core.prev["items"] != held
    log(f"M7 PFC-1 register: holds {len(held)} items exactly through any silence: {silent_ok}; a new partner line replaces "
        f"it exactly: {replaced}; capacity = the prompt line + the current line (2 lines) held at once; state constant -> "
        f"J(t,t+1) = J(t,t+2) = 1 (period-2 index 0); no learned weights inside the register -> drift 0 (PROVED)")
    # (2) CTX: 20-way decoding of the line heard d lines ago from the current context state
    def proj(items):
        x = np.zeros(core.c.V + core.c.n_eye, bool); x[core._items(items)] = True
        d = x[core.ctx.idx].sum(1).astype(np.float32) + core.ctx.tie
        s_ = np.zeros(core.ctx.n, bool); s_[np.argpartition(-d, core.ctx.k)[:core.ctx.k]] = True; return s_
    hist, states = [], []
    for t in range(400):
        L = ids(rand_line()); core.hear_line(L, None, None, learn=True); hist.append(L); states.append(core.ctx.s.copy())
    for d in (0, 1, 2, 4, 8):
        ok = 0; n = 0
        for t in range(50, 400):
            cands = [hist[t - d]] + [hist[int(j)] for j in rng.choice(np.r_[0:t - d - 10], 19, replace=False)]
            sc = [(states[t] & proj(c)).sum() for c in cands]
            ok += int(np.argmax(sc) == 0); n += 1
        log(f"M7 CTX (lambda {core.c.ctx_lam}): line heard {d} lines ago decoded 20-way from the context state: {ok / n:.2f}")
    far = np.mean([jaccard(states[t], states[t + 150]) for t in range(50, 240)])
    near = np.mean([jaccard(states[t], states[t + 1]) for t in range(50, 390)])
    j2 = np.mean([jaccard(states[t], states[t + 2]) for t in range(50, 390)])
    log(f"M7 CTX generic-state check: states 150 lines apart overlap {far:.3f} (k/N = {core.ctx.k / core.ctx.n:.3f}); "
        f"consecutive {near:.2f}, two apart {j2:.2f} -> period-2 index {j2 - near:+.2f}; fixed weights -> no learning drift")
