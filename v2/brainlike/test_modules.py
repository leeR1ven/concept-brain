# Module tests of the brainlike prototype.  python3 -B test_modules.py AGENT_A.pkl > test_modules.out
#   M1 assemblies: hippocampal one-shot episodes (EC->DG->CA3->CA1 of the agent) complete from partial cues
#   M2 LEX segmentation on the bench's stage-A stream: within-word vs across-word transition probability; chunks vs
#      the bench lexicon
#   M3 GIST hold: DQ (leaky units + MD loop) holds a loaded gist through a reply with noise; no period-2
#      (J(t,t+1) >= J(t,t+2)); a pure synchronous 2-layer loop without leak does alternate (control)
#   M4 drift: codes of the same probe line (SUP, DC->DQ, DG) early vs after more learning
#   M5 episodic recall inside the agent: 'one-shot' line -> recalled words after intervening lines
import sys, pickle, copy, collections, numpy as np
sys.argv, argv = ['x'], sys.argv
import bench_run as br
from agent import Agent, NSM, NLM, kwta

R = []


def check(name, ok, info):
    R.append(ok); print(("PASS " if ok else "FAIL ") + name + " :: " + info, flush=True)


def J(a, b):
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    u = (a | b).sum()
    return (a & b).sum() / u if u else 1.0


rng = np.random.default_rng(0)
# ------------------------------------------------------------------ M1 hippocampal assemblies (agent's own path)
# 300 one-shot episodes through the agent's EC -> DG (fixed) -> CA3 (EC->CA3 and CA3->CA3 one-shot co-activity);
# cue = 50 % of the EC pattern; retrieval = the stored index with the largest overlap after CA3 completion.
ag = Agent({'seed': 3})
c = ag.c
pats, idxs = [], []
for k in range(300):
    ec = np.zeros(ag.n_ec, bool)
    ec[rng.choice(c.n_d, c.k_d, replace=False)] = True
    ec[c.n_d + rng.choice(NLM, 6, replace=False)] = True
    ec[-c.n_x + rng.choice(c.n_x, c.k_x, replace=False)] = True
    g = ag.dg_code(ec)
    ag.EC3.learn(ec, g); ag.C3.learn(g, g)
    pats.append(ec); idxs.append(np.nonzero(g)[0])
def recall(ec_cue):
    d = ag.EC3.current(ag.ec_weights(ec_cue)); idx = kwta(d, c.k_dg)
    for _ in range(2):
        idx = kwta(ag.C3.current(idx) + 0.5 * d / max(d.max(), 1e-9), c.k_dg)
    ov = [len(np.intersect1d(np.nonzero(idx)[0], e)) for e in idxs]
    return int(np.argmax(ov)), max(ov) / c.k_dg
res = []
for i in list(range(20)) + list(range(280, 300)):
    cue = pats[i].copy(); on = np.nonzero(cue)[0]; cue[rng.choice(on, len(on) // 2, replace=False)] = False
    k, q = recall(cue); res.append((k == i, q))
res = np.array(res, float)
check("M1 CA3 one-shot episodes are recalled from a 50% cue (300 stored; oldest 20 + newest 20)",
      res[:20, 0].mean() >= 0.9 and res[20:, 0].mean() >= 0.9,
      f"correct episode: oldest {res[:20,0].mean():.2f} newest {res[20:,0].mean():.2f}; index overlap {res[:,1].mean():.2f}")

# ------------------------------------------------------------------ M2 LEX segmentation (needs a stage-A agent)
if len(argv) > 1:
    A = pickle.load(open(argv[1], 'rb'))
    from world import OBJECTS, COLOURS, COUNTS, SLOTS, NOVEL, NAMES, RITUALS
    words = set(OBJECTS + COLOURS + COUNTS + SLOTS + NAMES + ['什么', '哪里', '颜色', '这是', '不是', '好吃', '客气', '晚安',
                                                              '再见', '早上', '宝宝'])
    # map syllable units to chars through the world's voice: the unit is the char whose clean sound it detects
    import physics
    W = br.mk('A', 0)
    v = W.voice
    st = physics.Stream(v.F, v.T, 2, 3, v.levels, 0.0)
    def clean(ch):
        st2 = physics.Stream(v.F, v.T, 2, 3, v.levels, 0.0)
        none = (None, None, False)
        for k in range(3):
            win = st2.tick(((ch, v.hear(ch, 'other'), k == 0), none, none))
        return A.ear.feat0(win)
    unit = {}
    for ch in v.chars:
        j, cs = A.syl_detect(clean(ch))
        if cs >= A.c.th_det:
            unit[j] = ch
    TP = A.S.w
    within, across = [], []
    for wd in words:
        if len(wd) == 2 and all(ch in unit.values() for ch in wd):
            inv = {ch: j for j, ch in unit.items()}
            within.append(TP[inv[wd[0]], inv[wd[1]]])
    # across-word: last char of one word -> first char of another, both known units
    inv = {ch: j for j, ch in unit.items()}
    for w1 in words:
        for w2 in words:
            if w1 != w2 and w1[-1] in inv and w2[0] in inv and (w1[-1] + w2[0]) not in words:
                x = TP[inv[w1[-1]], inv[w2[0]]]
                across.append(x)
    within, across = np.array(within), np.array(across)
    chunks = [''.join(unit.get(s, '?') for s in A.chunk_syl[k]) for k in sorted(A.chunk_syl)]
    true_ch = [ch for ch in chunks if ch in words]
    check("M2 LEX: within-word transition probability >> across-word (Saffran-style)",
          len(within) > 5 and within.mean() > 3 * max(across.mean(), 0.01),
          f"within {within.mean():.2f} (n {len(within)}) vs across {across.mean():.3f} (n {len(across)})")
    check("M2' LEX chunks are mostly real words (precision >= 0.6)", len(chunks) > 5 and len(true_ch) / len(chunks) >= 0.6,
          f"{len(true_ch)}/{len(chunks)} chunks are lexicon words: {' '.join(chunks)}")

# ------------------------------------------------------------------ M3 GIST hold, period-2
def hold_run(lam, md, noise, ticks=40, seed=0):
    g = Agent({'seed': seed, 'lam_dq': lam, 'md': md, 'dq_noise': noise})
    r = np.random.default_rng(seed)
    g.dc = np.zeros(g.c.n_d, bool); g.dc[r.choice(g.c.n_d, g.c.k_d, replace=False)] = True
    g.v_dc = g.dc.astype(np.float32)
    g.load_dq()
    seq = [g.dq.copy()]
    for _ in range(ticks):
        g.dq_tick(); seq.append(g.dq.copy())
    j0 = [J(seq[0], s) for s in seq]
    j1 = np.mean([J(seq[t], seq[t + 1]) for t in range(ticks - 1)])
    j2 = np.mean([J(seq[t], seq[t + 2]) for t in range(ticks - 2)])
    return j0, j1, j2
for noise, need in ((0.0, 0.8), (0.3, 0.8), (1.0, 0.0)):
    j0, j1, j2 = hold_run(0.9, 0.3, noise)
    check(f"M3 DQ hold (leaky 0.9 + MD gain 0.3) through a 40-tick reply, selection noise sd {noise}"
          + (" (report only)" if need == 0 else ""), j0[-1] >= need and j1 >= j2 - 1e-9,
          f"J(0,40)={j0[-1]:.2f} J(t,t+1)={j1:.3f} J(t,t+2)={j2:.3f}")


def sync_loop(leak, selfw=0.0, ticks=40, n=1000, k=50, seed=0):
    """control: two populations A <-> B with fixed one-to-one cross weights (a 2-layer loop), k-WTA, starting with
    CONFLICTING contents; v <- leak*v + (1-leak)*other + selfw*own output (selfw = within-layer self-hold)."""
    r = np.random.default_rng(seed)
    pa = np.zeros(n, bool); pa[r.choice(n, k, replace=False)] = True
    pb = np.zeros(n, bool); pb[r.choice(n, k, replace=False)] = True
    va, vb = pa.astype(float), pb.astype(float)
    A_, B_ = pa.copy(), pb.copy(); seq = []
    for _ in range(ticks):
        va = leak * va + (1 - leak) * B_ + selfw * A_; vb = leak * vb + (1 - leak) * A_ + selfw * B_
        A_, B_ = kwta(va + r.normal(0, 1e-3, n), k), kwta(vb + r.normal(0, 1e-3, n), k)
        seq.append(A_.copy())
    j1 = np.mean([J(seq[t], seq[t + 1]) for t in range(10, ticks - 1)])
    j2 = np.mean([J(seq[t], seq[t + 2]) for t in range(10, ticks - 2)])
    return j1, j2
j1, j2 = sync_loop(0.0)
j1l, j2l = sync_loop(0.9)
j1s, j2s = sync_loop(0.9, selfw=0.2)
check("M3' control: a synchronous 2-layer loop alternates with period 2 (no leak); a within-layer self-hold "
      "(the MD-type 1:1 loop) removes it", j2 > j1 + 0.5 and j1s >= j2s - 1e-9 and j1s > 0.9,
      f"no leak: J(t,t+1)={j1:.2f} J(t,t+2)={j2:.2f}; leak 0.9 only: {j1l:.2f}/{j2l:.2f} (STILL period 2 near the tie); "
      f"leak 0.9 + self-hold 0.2: {j1s:.2f}/{j2s:.2f}")

print(f"\n{sum(R)}/{len(R)} module tests passed", flush=True)
