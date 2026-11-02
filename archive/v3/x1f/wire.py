"""Generation-0 hand-added connection sets (every set is listed in REGISTRY.md).

A1  speech basics, auditory content line -> muscle (ARCH 2.7, 2.11 step 1, 2.12): each sound's internal neurons (the L1
    chain copies of its content features, lit when the sound is heard or inner-heard) connect to the muscles that
    produce it.  Computed as the non-negative least-squares map (ridge 0.3) from every syllable's content code to its
    4-muscle set, so a full inner hearing drives exactly its own muscles (X1c's stand-in sums the muscle sets of every
    syllable sharing a feature and leaves the 4th and 5th muscle only 1.5x apart -> garble).  Replaces Mmap; used by
    both the PFC auditory return (inner hearing) and the direct auditory -> motor link, as Mmap was.
    The X1c body has no order or force grades inside a syllable (one 4-muscle set per syllable, decoded every tick), so
    the map is a set map, not a sequence map.

B/C (DIAGNOSTIC arm only, "hand-wired thinking"): see wire_thinking().
"""
import numpy as np


def nnls(A, b, it=4000):
    x = np.zeros(A.shape[1])
    L = np.linalg.norm(A, 2) ** 2
    for _ in range(it):
        x = np.maximum(x - (A.T @ (A @ x - b)) / L, 0)
    return x


def sound_action_map(ph, lam=0.3):
    C = ph.content.astype(float)
    N = ph.need_mat.astype(float)
    Ca = np.vstack([C, lam * np.eye(C.shape[1])])
    W = np.zeros((C.shape[1], N.shape[1]))
    for m in range(N.shape[1]):
        W[:, m] = nnls(Ca, np.concatenate([N[:, m], np.zeros(C.shape[1])]))
    return W


def wire_speech(brain, ph, lam=0.3):
    W = sound_action_map(ph, lam)
    cont = np.zeros((brain.n1, ph.n_c), np.float32)
    for k in range(brain.na1):
        if brain.l1_content[k]:
            cont[k, brain.l1_feat[k]] = 1.0
    brain.Mmap_standin = brain.Mmap
    brain.Mmap = (cont @ W).astype(np.float32)
    return dict(set='A1', area='auditory L1 content copies -> motor muscle neurons', n_links=int((brain.Mmap > 1e-6).sum()),
                total=float(brain.Mmap.sum()), lam=lam)


# ====================================================================== DIAGNOSTIC arm: hand-wired thinking (B) + responding (C)
VOICES = None


def measure_codes(brain, ph, props, seed, reps=2, p_rel=0.6):
    """per-voice middle-layer (L2) codes of every syllable, on the frozen newborn (learning off, mouth muted): each
    question and answer line is heard in each voice (C1, C2, D1, D2, own); firing probability of every L2 unit in the
    ticks 1-3 after the syllable's onset (L2 lags the input by 1 tick)."""
    from world import SID, render, OWN, C1, C2, D1, D2, question, answer
    V = [C1, C2, D1, D2, OWN]
    rng = np.random.default_rng(seed + 9100)
    snap = brain.snapshot()
    rec = {}
    for rep in range(reps):
        for f in range(4):
            for x in range(8):
                for line in (question(f, x), answer(f, x, props)):
                    for v in V:
                        brain.restore(snap)
                        seq = [None] * 4 + [SID[s] for s in line for _ in range(3)] + [None] * 4
                        L2 = []
                        for cg in seq:
                            brain.step(render(ph, [('s', cg, v, 1.0)] if cg is not None else [], rng), 0.0, learn=False)
                            L2.append(brain.l2.copy())
                        for i, s in enumerate(line):
                            on = 4 + 3 * i
                            for k in (1, 2, 3):
                                rec.setdefault((SID[s], v), []).append(L2[on + k])
    brain.restore(snap)
    return {k: np.mean(Ls, 0) for k, Ls in rec.items()}          # firing probability per (syllable, voice)


def select_codes(prob, p_rel=0.5, margin=0.3):
    """selective code of (s, v): units on in >= p_rel of s's ticks in voice v and at least `margin` more often than for
    any other syllable in the same voice (so a code unit says 'this syllable', not 'some speech')"""
    codes = {}
    voices = sorted({v for s, v in prob})
    for v in voices:
        keys = [k for k in prob if k[1] == v]
        P = np.array([prob[k] for k in keys])
        for i, k in enumerate(keys):
            other = np.delete(P, i, 0).max(0) if len(keys) > 1 else np.zeros(P.shape[1])
            codes[k] = np.nonzero((P[i] >= p_rel) & (P[i] - other >= margin))[0]
    return codes


def wire_thinking(brain, ph, props, base_pairs, seed, m=5, n_wm=4, th_wm=1.0, s_self=3.0, inh=3.5, a_x=5.5, a_h=0.6,
                  a_c=0.8, w_line=1.0, codes=None, p_rel=0.6, margin=0.1, only_t1=False, local31=False, say_pairs=None):
    """B: a working-memory (WM) part of the top layer = n_wm patches of the X3 patch-mean; its units get no direct
    L2 input (their L2->L3 links are removed) and fire at a fixed threshold th_wm on their lateral drive.  Assemblies of
    m WM units:  filler holds H_X (X = fillers of base pairs) and, per base pair (f, X) with answer a_1..a_n, answer
    steps Q1..Qn.  Links (H33, given as drive strengths):
        self links inside every assembly (it holds itself);
        inhibitory links from every assembly to the members of every other assembly that is not its successor
        ("rarely co-active neurons inhibit each other", ARCH 2.4) -> one held assembly at a time, the newest wins;
        heard copies of X -> H_X (strong: a heard filler takes over);
        H_X & code(last question syllable) -> Q1;  Qk & code(a_k) -> Q(k+1);  Qn & code(a_n) -> D_q ('answered', holds the
        question's topic and predicts nothing); each source alone stays below threshold, both together cross it.
        Links run one way: the successor inhibits its predecessor, so the thought moves along the fixed path.
    C: Qk -> L1 copies of a_k's content (inner hearing), so the answer is said by the A1 sound -> muscle map.  The
    learned top -> L1 links keep the M31 locality mask (ARCH 2.4 is about the Hebbian links); the hand-added ones reach
    all 10 copies (local31=False), because 5-unit assemblies reach only ~70 % of a syllable's copies through the mask
    and a partial inner hearing is often blocked by A2.  code(s) = the heard L1 content copies of syllable s.
    Returns a registry dict."""
    from world import SID, question, answer, FILL, OWN, C1, C2, D1, D2
    rng = np.random.default_rng(seed + 9200)
    n3 = brain.c['n3']
    lab = brain.PM3.argmax(1)
    sizes = np.bincount(lab, minlength=brain.PM3.shape[1])
    cent = np.array([brain.pos3[lab == p].mean(0) if sizes[p] else np.zeros(2) for p in range(len(sizes))])
    fillers = sorted({x for f, x in base_pairs})
    need = m * (len(fillers) + sum(len(answer(f, x, props)) + 1 for f, x in base_pairs))
    cand = [p for p in range(len(sizes)) if sizes[p] >= 15]
    pick = [max(cand, key=lambda p: sizes[p])]
    while len(pick) < len(cand) and (len(pick) < n_wm or sizes[pick].sum() < need):   # spread-out patches
        pick.append(max((p for p in cand if p not in pick), key=lambda p: min(np.sum((cent[p] - cent[q]) ** 2) for q in pick)))
    wm = np.nonzero(np.isin(lab, pick))[0]
    free = list(rng.permutation(wm))                           # non-overlapping assemblies
    if codes is None:
        codes = measure_codes(brain, ph, props, seed)
    codes = select_codes(codes, p_rel, margin)
    V = [C1, C2, D1, D2, OWN]
    W2 = np.zeros((brain.c['n2'], n3), np.float32)
    H13 = np.zeros((brain.n1, n3), np.float32)                  # hand-added heard L1 copies -> WM
    H33 = np.zeros((n3, n3), np.float32)
    H31 = np.zeros((n3, brain.n1), np.float32)
    asm, succ = {}, {}

    def new_asm(name):
        A = np.array(sorted(free[:m]))
        del free[:m]
        asm[name] = A
        succ[name] = set()
        return A

    used = np.nonzero(ph.content.any(0))[0]

    def feed_code(s, A, a):                                    # heard L1 content copies of s -> A
        own = np.nonzero(ph.content[s])[0]
        lines = np.nonzero(brain.l1_content & np.isin(brain.l1_feat, own))[0]
        oth = np.nonzero(brain.l1_content & np.isin(brain.l1_feat, np.setdiff1d(used, own)))[0]
        H13[np.ix_(lines, A)] += a / len(lines)

    def say(A, s):
        lines = np.nonzero(brain.l1_content & np.isin(brain.l1_feat, np.nonzero(ph.content[s])[0]))[0]
        M = brain.M31[np.ix_(A, lines)] if local31 else np.ones((len(A), len(lines)), np.float32)
        reach = M.sum(0)
        H31[np.ix_(A, lines)] += np.where(reach > 0, M * w_line / np.maximum(reach, 1), 0)
        return int((reach > 0).sum()), len(lines)

    for x in fillers:
        H = new_asm(f'H_{FILL[x]}')
        feed_code(SID[FILL[x]], H, a_x)
    trans, cover = [], []
    for (f, x) in base_pairs:
        Q, A = question(f, x), answer(f, x, props)
        prev, prev_s = f'H_{FILL[x]}', SID[Q[-1]]
        for k, s in enumerate(A):
            name = f'Q{k + 1}_{f}{x}'
            new_asm(name)
            trans.append((prev, prev_s, name))
            if say_pairs is None or (f, x) in say_pairs:       # C only for the pairs that are to be answered
                cover.append(say(asm[name], SID[s]))
            prev, prev_s = name, SID[s]
        new_asm(f'D_{f}{x}')                                   # 'answered': holds the topic, predicts nothing
        trans.append((prev, prev_s, f'D_{f}{x}'))
    for src_, s, dst in trans:
        succ[src_].add(dst)
    names = list(asm)
    for a_ in names:                                           # excitation: self links and transitions
        A = asm[a_]
        for i in A:
            for j in A:
                if i != j:
                    H33[i, j] += s_self / (len(A) - 1)
    for src_, s, dst in trans:
        H33[np.ix_(asm[src_], asm[dst])] += a_h / len(asm[src_])
        feed_code(s, asm[dst], a_c)
    # inhibition: WM neurons that are not wired together inhibit each other ("rarely co-active neurons inhibit
    # each other", ARCH 2.4); a full other assembly gives -inh
    sub = H33[np.ix_(wm, wm)]
    neg = (sub == 0) & ~np.eye(len(wm), dtype=bool)
    sub[neg] = -inh / m
    H33[np.ix_(wm, wm)] = sub
    np.fill_diagonal(H33, 0)
    brain.W23_wm_removed = brain.W23[:, wm].copy()
    brain.W23[:, wm] = W2[:, wm]                                # WM gets only the hand-added L2 feeds
    brain.B33[:, wm] = 0.0                                      # and no random birth lateral input
    brain._norm()
    brain.H33, brain.H31, brain.H13 = H33, H31, H13
    brain.l3 = brain.l3.copy()
    brain.l3[wm] = False                                       # the new WM part starts empty
    brain.pred = np.zeros_like(brain.pred)
    brain.l1 = brain.l1 & brain.ext_prev
    brain.wm, brain.asm, brain.th_wm = wm, asm, th_wm
    nc = {s: [len(codes.get((s, v), [])) for v in V] for s in range(len(ph.content))}
    used = sorted({SID[FILL[x]] for x in fillers} | {t[1] for t in trans})
    return dict(set='B+C (diagnostic)', wm_patches=[int(p) for p in pick], wm_units=int(len(wm)), n_assemblies=len(asm),
                assembly_size=m, memberships_per_wm_unit=round(float(sum(len(a) for a in asm.values()) / len(wm)), 2),
                H13_links=int((H13 != 0).sum()), H33_pos_links=int((H33 > 0).sum()), H33_neg_links=int((H33 < 0).sum()), H31_links=int((H31 > 0).sum()),
                W23_links_removed=int((brain.W23_wm_removed > 0).sum()),
                inner_hearing_lines_reached=round(float(np.mean([c[0] / c[1] for c in cover])), 3) if cover else None,
                n_path_pairs=len(base_pairs), n_say_pairs=len(base_pairs) if say_pairs is None else len(say_pairs),
                code_size_used_min=int(min(min(nc[s]) for s in used)), code_size_used_median=float(np.median([nc[s] for s in used])),
                params=dict(m=m, n_wm=n_wm, th_wm=th_wm, s_self=s_self, inh=inh, a_x=a_x, a_h=a_h, a_c=a_c, w_line=w_line,
                            p_rel=p_rel, margin=margin))
