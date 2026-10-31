# Loop-hold tests of the PFC (coordinator's checklist).  python3 -B test_hold.py AGENT.pkl WORLD.pkl > test_hold.out
# On a trained agent (learning frozen unless stated), 20 different partner lines with fixed scenes:
#   H1 generic-state index: mean Jaccard of the held DQ codes of DIFFERENT lines vs the random-pair level k/n
#   H2 duration in silence: 20-way nearest-neighbour decoding of the held line after 10 / 50 / 200 silent ticks
#      (selection noise sd 0 and 0.3)
#   H3 replacement: after line i is held, line j arrives -> DQ decodes as j; residual overlap with i
#   H4 period-2 index during the hold: J(t,t+1) - J(t,t+2) (>= 0 = no alternation)
#   H5 capacity: DQ holds exactly one line (by design); CTX keeps a blend of the last lines (decode line n-1 from CTX
#      after line n); the hippocampus keeps many (recall of each of the 20 lines from its words after all 20)
#   H6 drift with learning ON: the same 20 lines re-presented after 200 more B dialogues of learning
import sys, copy, pickle, numpy as np
argv = sys.argv; sys.argv = ['x']
import bench_run as br
from agent import kwta

a0 = pickle.load(open(argv[1], 'rb')); w0 = pickle.load(open(argv[2], 'rb'))
for k, v in dict(pred_log=[], line_no=0, blank_n=0, self_go=-100, att_src=None).items():
    if not hasattr(a0, k): setattr(a0, k, v)
from agent import DEFAULT
for k, v in DEFAULT.items(): a0.c.setdefault(k, v)
for kv in argv[3:]:
    if kv.startswith('quick'): continue
    k, v = kv.split('='); a0.c[k] = type(DEFAULT[k])(v)
QUICK = 'quick=1' in argv
if a0.c.get('word_assembly') and not hasattr(a0, 'word_code'):
    from agent import NLM
    rw = np.random.default_rng(a0.c.seed + 9)
    a0.word_code = np.zeros((NLM, 512), np.float32)
    for s in range(NLM):
        a0.word_code[s, rw.choice(512, 20, replace=False)] = 1.0
    idx, g = a0.sup_fan['lex']
    a0.sup_fan['lex'] = (np.random.default_rng(5).integers(0, 512, idx.shape), g)
rng = np.random.default_rng(3)
from world import OBJECTS
LINES = []
for i in range(20):
    objs = [OBJECTS[(i + k) % 12] for k in range(3)]
    sc = tuple((o, (i + k) % 6, (i + 2 * k) % 3) for k, o in enumerate(objs))
    text = [objs[0] + '在哪里', '你要' + objs[1] + '吗', '左边是什么', objs[2] + '是什么颜色', '有几个' + objs[0]][i % 5]
    LINES.append((text, sc))


def J(x, y):
    u = (x | y).sum()
    return float((x & y).sum() / u) if u else 1.0


def quiet(a, w, n=8, min_ticks=30, cap=120):
    """bench protocol (gap_answer): at least min_ticks of waiting, then until the agent has been silent n ticks."""
    for k in range(cap):
        if k >= min_ticks and not a.body.SPK and (a.t - a.t_mouth) > n:
            return
        w._tick()


def present(a, w, text, sc, silent=20, noise=0.0, record=()):
    """play one partner line in a fixed scene; return DQ at the load and during the following silence (the agent
    may answer meanwhile, as in the bench; DQ must hold the line through its own reply)."""
    a.c['dq_noise'] = noise
    w.agent = a
    quiet(a, w)
    w._set_scene(sc, None)
    w._silence(6)
    w._say('other', text)
    held = {}
    for k in range(1, silent + 1):
        w._tick()
        if k in record:
            held[k] = a.dq.copy()
    return a.dq.copy(), held


def frozen(a):
    b = copy.deepcopy(a); b.c['freeze'] = True; b.c['babble'] = False; b.c['babble_idle'] = False
    return b


w = br.mk('B', 0, prev=w0)
a = frozen(a0)
# reference codes: each line presented once, DQ 3 ticks after the line end
ref = []
for text, sc in LINES:
    _, h = present(a, w, text, sc, silent=3, record=(3,))
    ref.append(h[3])
R = np.array(ref)
pair = [J(R[i], R[j]) for i in range(20) for j in range(i + 1, 20)]
k_over_n = a.c.k_d / a.c.n_d
print(f"H1 generic-state index: mean Jaccard between held codes of different lines {np.mean(pair):.3f} "
      f"(max {np.max(pair):.3f}); random-pair level {k_over_n / (2 - k_over_n):.3f}; old PFC 0.47-0.79", flush=True)
# same line twice (reliability)
same = []
for text, sc in LINES[:10]:
    _, h = present(a, w, text, sc, silent=3, record=(3,))
    same.append(J(h[3], ref[LINES.index((text, sc))]))
print(f"   same line presented again: Jaccard {np.mean(same):.3f}", flush=True)


def decode(code):
    return int(np.argmax([J(code, r) for r in ref]))


for noise in ((0.0,) if QUICK else (0.0, 0.3)):
    hits = {10: [], 50: [], 200: []}; p2 = []
    if QUICK: LINES_ = LINES[:10]
    for i, (text, sc) in enumerate(LINES):
        b = frozen(a0)
        wb = br.mk('B', 0, prev=w0)
        # train-free: the frozen agent; silence after the line with the eye blanked (nothing new comes in)
        b.c['dq_noise'] = noise
        wb.agent = b
        quiet(b, wb); wb._set_scene(sc, None); wb._silence(6); wb._say('other', text)
        wb._set_scene(None, None)
        seq = []
        for k in range(1, 201):
            wb._tick(); seq.append(b.dq.copy())
            if k in hits:
                hits[k].append(decode(b.dq) == i)
        p2.append(np.mean([J(seq[t], seq[t + 1]) - J(seq[t], seq[t + 2]) for t in range(10, 198)]))
    print(f"H2 noise sd {noise}: 20-way decoding of the held line after 10 / 50 / 200 silent ticks: "
          f"{np.mean(hits[10]):.2f} / {np.mean(hits[50]):.2f} / {np.mean(hits[200]):.2f};  "
          f"H4 period-2 index J(t,t+1)-J(t,t+2) = {np.mean(p2):+.4f} (>= 0: no alternation)", flush=True)
if QUICK:
    sys.exit(0)
# H3 replacement
rep, resid = [], []
for i in range(20):
    j = (i + 7) % 20
    b = frozen(a0); wb = br.mk('B', 0, prev=w0)
    present(b, wb, *LINES[i], silent=10)
    d1 = b.dq.copy()
    present(b, wb, *LINES[j], silent=3)
    rep.append(decode(b.dq) == j); resid.append(J(b.dq, d1))
print(f"H3 replacement: a new partner line replaces the held one in {np.mean(rep):.2f} of 20 cases; residual overlap "
      f"with the old line {np.mean(resid):.3f}", flush=True)
# H5 capacity: CTX keeps a blend; hippocampus keeps all
b = frozen(a0); wb = br.mk('B', 0, prev=w0)
ctx_hits = []
ctx_ref = []
for i, (text, sc) in enumerate(LINES):
    present(b, wb, text, sc, silent=3)
    ctx_ref.append(b.ctx.copy())
# decode line n-1 from CTX after line n: nearest neighbour among CTX states recorded right after each line
for i in range(1, 20):
    c_now = ctx_ref[i]
    sims = [J(c_now, ctx_ref[k]) for k in range(20) if k != i]
    ks = [k for k in range(20) if k != i]
    ctx_hits.append(ks[int(np.argmax(sims))] == i - 1)
print(f"H5 capacity: DQ holds 1 line by design; CTX after line n is closest to CTX after line n-1 in "
      f"{np.mean(ctx_hits):.2f} of cases (chance 1/19 = 0.05): a slowly moving topic, not a list", flush=True)
# hippocampus: after all 20 lines (frozen = no encoding) -> use a learning copy for encoding only
b = copy.deepcopy(a0); b.c['babble'] = False; b.c['babble_idle'] = False
wb = br.mk('B', 0, prev=w0)
for text, sc in LINES:
    present(b, wb, text, sc, silent=3)
n_ep = len(b.episodes)
ok = 0
for i, (text, sc) in enumerate(LINES):
    # cue: the line's own words (from its stored episode) without the scene
    e = b.episodes[n_ep - 20 + i]
    ec = np.zeros(b.n_ec, bool); o = b.c.n_d
    from agent import NLM
    ec[o:o + NLM] = e['words']
    d = b.EC3.current(b.ec_weights(ec)); idx = kwta(d, b.c.k_dg)
    for _ in range(2):
        idx = kwta(b.C3.current(idx) + 0.5 * d / max(d.max(), 1e-9), b.c.k_dg)
    E = np.array([ep['idx'] for ep in b.episodes])
    ok += int(np.argmax(idx[E].sum(1))) == n_ep - 20 + i
print(f"H5' hippocampus: each of the 20 lines recalled from its words alone after all 20 were stored (among "
      f"{n_ep} episodes): {ok}/20", flush=True)
# H6 drift with learning ON
b = copy.deepcopy(a0); b.c['babble'] = False
wb = br.mk('B', 0, prev=w0)
wb.run(b, 200)
bf = frozen(b); wf = br.mk('B', 0, prev=wb)
dr = []
for i, (text, sc) in enumerate(LINES):
    _, h = present(bf, wf, text, sc, silent=3, record=(3,))
    dr.append(J(h[3], ref[i]))
hits = [int(np.argmax([J(bf_code, r) for r in ref])) == i for i, bf_code in enumerate([None] * 0)]
print(f"H6 drift with learning on: Jaccard of the held code of the same line before vs after 200 more dialogues of "
      f"learning {np.mean(dr):.3f} (min {np.min(dr):.3f})", flush=True)
