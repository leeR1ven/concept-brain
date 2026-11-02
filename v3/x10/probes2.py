"""Probes for the 11-02 tests: fast PFC, one-shot, long-term knowledge, layer overlap, held-context share
(see PLAN addenda). World labels are used only for scoring."""
import numpy as np
import torch

import world as W
from common import DEV
from probes import Frozen, contexts, decode, jac, line_script, play, stability, tasks
from world_adapter import ADULTS, C1, C2, NSYL, SID, SYL

CHANCE = 1 / 32


def codes_ext(life, lags=(0, 6, 12, 24, 48), learn=False, n_ctx=4, seed=0):
    """a3 probe extended: 32 questions x n_ctx contexts; L4 (and Hs) at the given lags. learn=True: learning stays
    on within each trial and the full state (neuron states and every plastic weight) is restored between trials."""
    b = life.brain
    props, _, _ = tasks(life)
    rng = np.random.default_rng(seed + 9100)
    L = max(lags)
    S4, SN, SH, SI, key = [], [], [], [], []
    was = b.learning
    full = b.snapshot_full()
    b.learning = learn
    for f in range(4):
        for x in range(8):
            for ci, ctx in enumerate(contexts(rng, props)[:n_ctx]):
                b.restore_full(full)
                q = line_script(W.question(f, x), C1 if ci % 2 == 0 else C2)
                t_end = len(ctx) + len(q) - 1
                s4, sn, sh, si = {}, {}, {}, {}

                def rec(lf, t):
                    d = t - t_end
                    if d in lags:
                        s4[d] = lf.brain.pfc.L4.on.clone()
                        si[d] = torch.cat([lf.brain.pfc.x_int, lf.brain.pfc.L4.on])
                        if lf.brain.hip is not None:
                            sh[d] = lf.brain.hip.Hs.on.clone()
                    if d - 1 in lags:
                        sn[d - 1] = lf.brain.pfc.L4.on.clone()
                play(life, ctx + q + [None] * (L + 2), mute=True, rec=rec)
                S4.append(torch.stack([s4[lg] for lg in lags]))
                SI.append(torch.stack([si[lg] for lg in lags]))
                SN.append(torch.stack([sn[lg] for lg in lags]))
                if sh:
                    SH.append(torch.stack([sh[lg] for lg in lags]))
                key.append((f, x, ci, 0))
    b.restore_full(full)
    b.learning = was
    S4, SN, SI = torch.stack(S4), torch.stack(SN), torch.stack(SI)
    SH = torch.stack(SH) if SH else None
    q_of = lambda k: (k[0], k[1])
    n = len(lags)
    res = dict(lags=list(lags), learn=learn, chance=CHANCE)
    res['L4_q32'] = [decode(S4, key, i, q_of) for i in range(n)]
    res['L1int_L4_q32'] = [decode(SI, key, i, q_of) for i in range(n)]
    res['L4_filler'] = [float(np.mean([decode(S4, key, i, lambda k, f=f: k[1] if k[0] == f else None)
                                       for f in range(4)])) for i in range(n)]
    res['motion_J'] = [float(jac(S4[:, i], SN[:, i]).mean().item()) for i in range(n)]
    if SH is not None:
        res['Hs_q32'] = [decode(SH, key, i, q_of) for i in range(n)]
    below = [lg for lg, d in zip(lags, res['L4_q32']) if d <= CHANCE + 0.1]
    res['remembered_until'] = below[0] if below else f'> {L}'
    # input influence at lag 0: L4 after different questions (lower = the question changes L4 more)
    K = len(key)
    diff, same = [], []
    for i in range(K):
        for j in range(i + 1, K):
            if key[i][2] == key[j][2]:
                if (key[i][0], key[i][1]) != (key[j][0], key[j][1]):
                    diff.append(jac(S4[i, 0], S4[j, 0]).item())
            elif (key[i][0], key[i][1]) == (key[j][0], key[j][1]):
                same.append(jac(S4[i, 0], S4[j, 0]).item())
    res['J_lag0_different_questions_same_context'] = float(np.mean(diff))
    res['J_lag0_same_question_other_context'] = float(np.mean(same))
    return res


def answer_hit(life, q, a_first, voice=C1, after=6, learn=None):
    """hear q (learning as set, or `learn`); route 1 (PFC): is a_first's code the most driven by the internal L4
    drive in PFC L1? route 2 (hippocampus): is a_first's code the most lit among the hippocampus's internally lit
    copies (whose returns re-light the sound units)? ticks 1..after after q. Returns (pfc hit, hip hit, rel)."""
    b = life.brain
    qs = line_script(q, voice)
    t_end = len(qs) - 1
    drv, hip = [], []
    was = b.learning
    if learn is not None:
        b.learning = learn

    def rec(lf, t):
        if t_end < t <= t_end + after:
            drv.append(lf.brain.pfc.d1.clone())
            if lf.brain.hip is not None:
                hip.append(lf.brain.hip.h_int.float())
    play(life, qs + [None] * (after + 1), mute=True, rec=rec)
    b.learning = was
    D = torch.stack(drv).mean(0)
    per = torch.stack([D[cd].mean() if cd.numel() else torch.zeros((), device=DEV) for cd in life.codes])
    pfc_hit = int(per.argmax().item() == a_first)
    hip_hit = 0
    if hip:
        Hm = torch.stack(hip).mean(0)
        ph = torch.stack([Hm[cd].mean() if cd.numel() else torch.zeros((), device=DEV) for cd in life.codes])
        hip_hit = int(ph.max() > 0 and ph.argmax().item() == a_first)
    return pfc_hit, hip_hit, float(per[a_first] / per.max().clamp_min(1e-9))


def one_shot(life, Ds=(200, 500, 1000, 2000)):
    """each held-out pair heard once (adult C1 asks, adult C2 answers 3 ticks later), 12 silent ticks, then D ticks
    of the ordinary overheard world (learning on); then Q again (learning off): hit = A's first syllable top-driven.
    Full state restored between trials; control = the same without the one exposure."""
    b = life.brain
    props, _, held = tasks(life)
    full = b.snapshot_full()
    was = b.learning
    out = {}
    for D in Ds:
        for cond in ('heard_once', 'control'):
            if cond == 'control' and D != Ds[0]:
                continue
            hits, hhits, rel = [], [], []
            for f, x in held:
                b.restore_full(full)
                b.learning = True
                Q, A = W.question(f, x), W.answer(f, x, props)
                if cond == 'heard_once':
                    play(life, line_script(Q, C1) + [None] * 3 + line_script(A, C2) + [None] * 12, mute=True)
                t0 = b.t
                while b.t - t0 < D:
                    life.tick()
                h, hh, r = answer_hit(life, Q, SID[A[0]], learn=True)
                hits.append(h)
                hhits.append(hh)
                rel.append(r)
            out[f'{cond} D={D}'] = dict(pfc_route_hit=float(np.mean(hits)), hip_route_hit=float(np.mean(hhits)),
                                        either=float(np.mean([max(x, y) for x, y in zip(hits, hhits)])),
                                        answer_drive_rel_max=float(np.mean(rel)), n=len(hits))
    b.restore_full(full)
    b.learning = was
    return out


def long_term(life, n_pairs=5, reps=20, dialogues=300):
    """n held-out pairs heard `reps` times each early (interleaved, learning on), probed right after, then
    `dialogues` overheard dialogues with learning on (a1L windows), then probed again (continuous life)."""
    b = life.brain
    props, _, held = tasks(life)
    pairs = held[:n_pairs]
    rng = np.random.default_rng(life.seed + 9950)
    order = [p for p in pairs for _ in range(reps)]
    rng.shuffle(order)
    was = b.learning
    b.learning = True
    for f, x in order:
        play(life, line_script(W.question(f, x), C1) + [None] * 3 + line_script(W.answer(f, x, props), C2)
             + [None] * 8, mute=True)

    def probe():
        with Frozen(life):
            snap = b.snapshot()
            hits = []
            for f, x in pairs:
                b.restore(snap)
                h, hh, _ = answer_hit(life, W.question(f, x), SID[W.answer(f, x, props)[0]], learn=False)
                hits.append(max(h, hh))
        return float(np.mean(hits))
    before = probe()
    st = stability(life, dialogues)
    after = probe()
    b.learning = was
    return dict(pairs=len(pairs), reps=reps, hit_after_teaching=before, hit_after_dialogues=after, stability=st)


def layer_overlap(life, reps=2):
    """Jaccard between syllable codes: different syllables (same voice C1) and the same syllable in two voices,
    in sound A1..A4 (chained units), PFC L1ext, L2, L3, L4. Each population in its own latency window."""
    b = life.brain
    ch = b.chains
    win = {'A1': (0, 3), 'A2': (1, 4), 'A3': (2, 5), 'A4': (3, 6), 'L1ext': (1, 7), 'L2': (2, 8), 'L3': (3, 9),
           'L4': (4, 10)}
    codes = {}
    with Frozen(life):
        snap = b.snapshot()
        for x in range(NSYL):
            for v in (C1, C2):
                acc = {}
                for _ in range(reps):
                    b.restore(snap)
                    rec = []

                    def rd(lf, t):
                        p = lf.brain.pfc
                        st = lf.brain.sound.state
                        rec.append(dict(A1=st[0][ch.sound_idx[0]], A2=st[1][ch.sound_idx[1]], A3=st[2][ch.sound_idx[2]],
                                        A4=st[3][ch.sound_idx[3]], L1ext=p.L1ext.on.clone(), L2=p.L2.on.clone(),
                                        L3=p.L3.on.clone(), L4=p.L4.on.clone()))
                    play(life, [None] * 4 + [(x, v, 1.0)] * 3 + [None] * 8, mute=True, rec=rd)
                    for k, (lo, hi) in win.items():
                        m = torch.stack([r[k] for r in rec[4 + lo: 4 + hi]]).float().mean(0)
                        acc[k] = acc.get(k, 0) + m / reps
                        p0 = torch.stack([r[k] for r in rec[0:4]]).float().mean(0)
                        acc[k + '_pre'] = acc.get(k + '_pre', 0) + p0 / reps
                for k in win:
                    codes[(k, x, v)] = acc[k] >= 0.5
                    codes[(k + '_ev', x, v)] = (acc[k] >= 0.5) & (acc[k + '_pre'] < 0.5)   # evoked on top of the held state
    out = {}
    for k in win:
        diff = [jac(codes[(k, x, C1)], codes[(k, y, C1)]).item() for x in range(NSYL) for y in range(x + 1, NSYL)]
        same = [jac(codes[(k, x, C1)], codes[(k, x, C2)]).item() for x in range(NSYL)]
        size = float(np.mean([codes[(k, x, C1)].sum().item() for x in range(NSYL)]))
        ke = k + '_ev'
        dife = [jac(codes[(ke, x, C1)], codes[(ke, y, C1)]).item() for x in range(NSYL) for y in range(x + 1, NSYL)]
        same_e = [jac(codes[(ke, x, C1)], codes[(ke, x, C2)]).item() for x in range(NSYL)]
        out[k] = dict(J_different_syllables=float(np.mean(diff)), J_same_syllable_two_voices=float(np.mean(same)),
                      code_size=size, evoked_J_different=float(np.mean(dife)), evoked_J_same=float(np.mean(same_e)),
                      evoked_size=float(np.mean([codes[(ke, x, C1)].sum().item() for x in range(NSYL)])))
    return out


def held_share(life, n_dialogues=20):
    """share of internally lit L1 copies whose feature was lit by its chain within the last 1..12 ticks"""
    b = life.brain
    held, tot = [], []

    def rec(lf):
        p = lf.brain.pfc
        xi = p.x_int
        age = p.L1ext.age
        held.append((xi & (age >= 1) & (age <= 12)).sum())
        tot.append(xi.sum())
    life.run_dialogues(n_dialogues, rec)
    H = torch.stack(held).float().sum().item()
    T = torch.stack(tot).float().sum().item()
    return dict(internal_copies_per_tick=T / max(1, len(tot)), held_context_share=H / max(1.0, T))


def collapse_test(life, seed=0, T_in=3, T=9):
    """MC collapse test, no PFC involved (11-02). The brain's returns are 0/1 per force neuron, so the input is 0/1:
    a 'f_x / f_y' mixture lights a random share f_x of each of x's muscles' force neurons and f_y of y's (a shared
    neuron counts once). Input for T_in ticks, then free ticks. Classified by the decoded mouth: settles on the
    stronger / weaker syllable, other, garble, silent; garble on the first tick; garble ticks; ticks on after the
    input stopped."""
    b, ph = life.brain, life.ph
    mot = b.motor
    need = ph.need_mat
    m = mot.m
    rng = np.random.default_rng(seed + 9990)
    mus_of = mot.muscle_of.cpu().numpy()
    share = lambda x, y: int((need[x] & need[y]).sum())

    def pat(x, f):
        v = np.zeros(mot.nf, np.float32)
        for j in np.nonzero(need[x])[0]:
            v[j * m + rng.choice(m, int(round(f * m)), replace=False)] = 1.0
        return v
    conds = {'clean': [], '2of4': [], 'noise10': []}
    for x in range(NSYL):
        mu = np.nonzero(need[x])[0]
        conds['clean'].append((x, None, need[x][mus_of].astype(np.float32)))
        conds['2of4'].append((x, None, np.isin(mus_of, mu[:2]).astype(np.float32)))
        conds['noise10'].append((x, None, np.minimum(need[x][mus_of] + (rng.random(mot.nf) < 0.1), 1).astype(np.float32)))
        for sh in (0, 1, 2):
            ys = [y for y in range(NSYL) if y != x and share(x, y) == sh]
            y = int(rng.choice(ys))
            for fx, fy in ((0.6, 0.4), (0.5, 0.5), (0.9, 0.7), (0.8, 0.8)):
                conds.setdefault(f'{int(fx * 100)}/{int(fy * 100)} share{sh}', []).append(
                    (x, y, np.minimum(pat(x, fx) + pat(y, fy), 1.0)))
    out = {}
    with Frozen(life):
        snap = b.snapshot()
        for name, trials in conds.items():
            cnt = dict(stronger=0, weaker=0, other=0, garble=0, silent=0)
            onset_g = g_ticks = 0
            after = []
            for x, y, drive in trials:
                b.restore(snap)
                decs = []
                for t in range(T):
                    mot.inject = torch.as_tensor(drive, device=DEV) if t < T_in else None
                    a0 = life.probe_env.step(None, mute=True)
                    mm = b.step(torch.as_tensor(a0, device=DEV), 0.0).cpu().numpy()
                    decs.append(ph.decode(mm))
                mot.inject = None
                said = [d for d in decs if d is not None]
                syl = [d[1] for d in said if d[0] in ('s', 'm')]
                onset_g += int(decs[0] is not None and decs[0][0] == 'g')
                g_ticks += sum(d is not None and d[0] == 'g' for d in decs)
                last = max([i for i, d in enumerate(decs) if d is not None], default=-1)
                after.append(max(last - (T_in - 1), 0))
                if not said:
                    cnt['silent'] += 1
                elif not syl:
                    cnt['garble'] += 1
                else:
                    top = max(set(syl), key=syl.count)
                    cnt['stronger' if top == x else 'weaker' if top == y else 'other'] += 1
            n = len(trials)
            out[name] = dict(n=n, **{k: v / n for k, v in cnt.items()}, onset_garble=onset_g / n,
                             garble_ticks=g_ticks / n, ticks_on_after_input=float(np.mean(after)))
    return out


def tendency_profile(life, ticks=(5, 16)):
    """What does the PFC's speech tendency look like at the motor area? Inner hearing of x (forced x code in PFC L1
    on ticks 6-8, learning off); per tick the force neurons whose 1:1 return is lit. Per syllable s: the mean over
    s's 4 muscles of the share of that muscle's force neurons with a lit return. Per tick (ticks[0]..ticks[1]):
    x's share, the best rival's share, rivals with share >= 0.5, x ranked first, share of all force neurons lit.
    Baseline: the same ticks with nothing forced (the spontaneous returns)."""
    from probes import code_mask
    b, ph, mot = life.brain, life.ph, life.brain.motor
    need = torch.as_tensor(ph.need_mat, device=DEV).float()                         # [NSYL, 24]

    def run(x):
        rec = []
        force = None if x is None else {6 + k: code_mask(life, [x]) for k in range(3)}
        play(life, [None] * (ticks[1] + 1), mute=True, force=force,
             rec=lambda lf, t: rec.append(lf.brain.ret_force.float().clone()) if ticks[0] <= t <= ticks[1] else None)
        return torch.stack(rec)                                                     # [T, nf]

    def shares(R):
        per_mus = torch.zeros(R.shape[0], 24, device=DEV).index_add_(1, mot.muscle_of, R) / mot.m
        return per_mus @ need.T / 4.0                                               # [T, NSYL]
    rows = []
    with Frozen(life):
        snap = b.snapshot()
        b.restore(snap)
        base = run(None)
        for x in range(NSYL):
            if life.codes[x].numel() == 0:
                continue
            b.restore(snap)
            R = run(x)
            S = shares(R)
            riv = torch.cat([S[:, :x], S[:, x + 1:]], 1)
            rows.append(dict(x=S[:, x], best=riv.max(1).values, ge_half=(riv >= 0.5).sum(1).float(),
                             first=(S.argmax(1) == x).float(), total=R.mean(1)))
    T = list(range(ticks[0], ticks[1] + 1))
    out = {k: [round(float(v), 3) for v in torch.stack([r[k] for r in rows]).mean(0)] for k in rows[0]}
    out['tick'] = T
    out['baseline_total'] = [round(float(v), 3) for v in base.mean(1)]
    out['n'] = len(rows)
    return out


LAT = {'A1': 0, 'A2': 1, 'A3': 2, 'A4': 3, 'L1ext': 1, 'L2': 2, 'L3': 3, 'L4': 4}


def overlap_trace(life):
    """Syllable-code overlap per layer (coordinator 11-02): sound A1..A4 (chained units only), PFC L1ext, L2, L3, L4.
    A syllable lasts 3 ticks (slices); each layer is read at its latency (A1 0 ... L4 4 ticks): per slice and pooled
    over the 3 slices (on in >= 2 of 3). Jaccard: different syllables in the same voice (C1); the same syllable in
    two voices (C1, C2); syllable vs the silence state (the 4 ticks before onset). Also evoked codes (on in the
    syllable, off in silence). L4 for inner hearing (x's code forced in PFC L1 for 3 ticks, read at the same lag) vs
    heard x, and inner x vs inner y."""
    from probes import code_mask
    b = life.brain
    ch = b.chains

    def grab(lf):
        p, st = lf.brain.pfc, lf.brain.sound.state
        return dict(A1=st[0][ch.sound_idx[0]].clone(), A2=st[1][ch.sound_idx[1]].clone(),
                    A3=st[2][ch.sound_idx[2]].clone(), A4=st[3][ch.sound_idx[3]].clone(),
                    L1ext=p.L1ext.on.clone(), L2=p.L2.on.clone(), L3=p.L3.on.clone(), L4=p.L4.on.clone())
    rec_all = {}
    with Frozen(life):
        snap = b.snapshot()
        for x in range(NSYL):
            for v in (C1, C2, 'inner'):
                b.restore(snap)
                rec = []
                if v == 'inner':
                    play(life, [None] * 14, mute=True, force={4 + k: code_mask(life, [x]) for k in range(3)},
                         rec=lambda lf, t: rec.append(grab(lf)))
                else:
                    play(life, [None] * 4 + [(x, v, 1.0)] * 3 + [None] * 7, mute=True, rec=lambda lf, t: rec.append(grab(lf)))
                rec_all[(x, v)] = rec
    out = {}
    for k, lat in LAT.items():
        sl = lambda x, v, s: rec_all[(x, v)][4 + lat + s][k]
        pooled = lambda x, v: torch.stack([sl(x, v, s) for s in range(3)]).float().mean(0) >= 0.5
        sil = lambda x, v: torch.stack([rec_all[(x, v)][t][k] for t in range(4)]).float().mean(0) >= 0.5
        res = {}
        for lab, f in [('slice1', lambda x, v: sl(x, v, 0)), ('slice2', lambda x, v: sl(x, v, 1)),
                       ('slice3', lambda x, v: sl(x, v, 2)), ('pooled', pooled)]:
            codes = {(x, v): f(x, v) for x in range(NSYL) for v in (C1, C2)}
            diff = np.mean([jac(codes[(x, C1)], codes[(y, C1)]).item() for x in range(NSYL) for y in range(x + 1, NSYL)])
            same = np.mean([jac(codes[(x, C1)], codes[(x, C2)]).item() for x in range(NSYL)])
            vs_sil = np.mean([jac(codes[(x, C1)], sil(x, C1)).item() for x in range(NSYL)])
            res[lab] = dict(J_diff=round(float(diff), 3), J_same_voice2=round(float(same), 3),
                            J_vs_silence=round(float(vs_sil), 3),
                            size=round(float(np.mean([codes[(x, C1)].sum().item() for x in range(NSYL)])), 1))
        ev = {(x, v): pooled(x, v) & ~sil(x, v) for x in range(NSYL) for v in (C1, C2)}
        res['evoked'] = dict(J_diff=round(float(np.mean([jac(ev[(x, C1)], ev[(y, C1)]).item() for x in range(NSYL)
                                                          for y in range(x + 1, NSYL)])), 3),
                             J_same_voice2=round(float(np.mean([jac(ev[(x, C1)], ev[(x, C2)]).item() for x in range(NSYL)])), 3),
                             size=round(float(np.mean([ev[(x, C1)].sum().item() for x in range(NSYL)])), 1))
        res['n_units'] = int(sl(0, C1, 0).numel())
        out[k] = res
    inner = {x: torch.stack([rec_all[(x, 'inner')][4 + 4 + s]['L4'] for s in range(3)]).float().mean(0) >= 0.5
             for x in range(NSYL)}
    heard = {x: torch.stack([rec_all[(x, C1)][4 + 4 + s]['L4'] for s in range(3)]).float().mean(0) >= 0.5
             for x in range(NSYL)}
    out['L4_inner'] = dict(J_inner_vs_heard_same=round(float(np.mean([jac(inner[x], heard[x]).item() for x in range(NSYL)])), 3),
                           J_inner_diff=round(float(np.mean([jac(inner[x], inner[y]).item() for x in range(NSYL)
                                                             for y in range(x + 1, NSYL)])), 3),
                           J_inner_vs_heard_other=round(float(np.mean([jac(inner[x], heard[y]).item() for x in range(NSYL)
                                                                       for y in range(NSYL) if y != x])), 3))
    return out


def qa_link_stats(life, after=4):
    """Learned Q -> A links (coordinator 11-02; cf. X1e "同时亮的太多，学到的连接把所有先后顺序平均掉了"): for each
    question (frozen, voice C1), the L4 units on in the ticks 0..after-1 after it ends drive the learned L4 -> L1
    table (L41) into L1. Per question: the drive into the answer's first-syllable code vs the other syllables' codes
    (ratio to their mean; answer ranked first), and the raw L41 weight from those L4 units onto the answer's code
    copies vs onto other codes' copies. Seen and held-out pairs separately."""
    from probes import tasks
    b = life.brain
    pfc = b.pfc
    props, seen, held = tasks(life)
    code_of = torch.full((b.n1,), -1, dtype=torch.long, device=DEV)
    for s, cd in enumerate(life.codes):
        if cd.numel():
            code_of[cd] = s
    out = {}
    with Frozen(life):
        snap = b.snapshot()
        for lab, pairs in (('seen', seen), ('held', held)):
            ratios, firsts, wr = [], [], []
            for f, x in pairs:
                A = SID[W.answer(f, x, props)[0]]
                b.restore(snap)
                q = line_script(W.question(f, x), C1)
                t_end = len(q) - 1
                acc = []
                play(life, q + [None] * (after + 1), mute=True,
                     rec=lambda lf, t: acc.append(lf.brain.pfc.L4.on.clone()) if t_end < t <= t_end + after else None)
                act = torch.stack(acc).float().mean(0) >= 0.5
                d = pfc.L41.drive(act)
                per = torch.stack([d[cd].mean() if cd.numel() else torch.zeros((), device=DEV) for cd in life.codes])
                others = torch.cat([per[:A], per[A + 1:]])
                ratios.append(float(per[A] / others.mean().clamp_min(1e-9)))
                firsts.append(int(per.argmax().item() == A))
                rows = torch.nonzero(act).squeeze(1)
                ix = pfc.L41.idx[rows].long()
                w = pfc.L41.w[rows]
                tgt = torch.where(ix < b.n1, code_of[ix.clamp_max(b.n1 - 1)], torch.full_like(ix, -1))
                on_a = w[tgt == A]
                on_o = w[(tgt >= 0) & (tgt != A)]
                wr.append(float(on_a.mean() / on_o.mean().clamp_min(1e-12)) if on_a.numel() and on_o.numel() else float('nan'))
            out[lab] = dict(n=len(pairs), drive_ratio_answer_vs_others=float(np.nanmean(ratios)),
                            answer_ranked_first=float(np.mean(firsts)), weight_ratio_answer_vs_others=float(np.nanmean(wr)))
    return out


def next_syllable(life):
    """Does the PFC predict the next syllable of a heard line? Frozen; each of the 32 questions in voice C1. At the
    onset tick of syllable k (k >= 2), before its own chains arrive, the internal drive into PFC L1 averaged over each
    syllable's code copies (it comes from L4 one tick earlier): the heard next syllable ranked first, its mean rank
    (chance 17.5), and the same for the answer's first syllable at the tick after the question ends."""
    b = life.brain
    props, seen, held = tasks(life)
    firsts, ranks, a_first, a_rank = [], [], [], []
    with Frozen(life):
        snap = b.snapshot()
        for f in range(4):
            for x in range(8):
                Q = W.question(f, x)
                A = SID[W.answer(f, x, props)[0]]
                q = line_script(Q, C1)
                b.restore(snap)
                drv = []
                play(life, q + [None] * 2, mute=True, rec=lambda lf, t: drv.append(lf.brain.pfc.d1.clone()))
                per = lambda t: torch.stack([drv[t][cd].mean() if cd.numel() else torch.zeros((), device=DEV)
                                             for cd in life.codes])
                for k in range(1, len(Q)):
                    p = per(3 * k)
                    tgt = SID[Q[k]]
                    rank = int((p > p[tgt]).sum().item()) + 1
                    firsts.append(int(rank == 1))
                    ranks.append(rank)
                p = per(len(q))
                rank = int((p > p[A]).sum().item()) + 1
                a_first.append(int(rank == 1))
                a_rank.append(rank)
    return dict(next_in_question_first=float(np.mean(firsts)), next_in_question_mean_rank=float(np.mean(ranks)),
                answer_after_question_first=float(np.mean(a_first)), answer_mean_rank=float(np.mean(a_rank)),
                n=len(firsts), chance_first=1 / 34)


def speech_summary(world, a, b_):
    """the child's spontaneous utterances in overheard dialogues a < n <= b_: what, how clear, when"""
    from collections import Counter
    from world6 import overhear_mech
    out = overhear_mech(world, a, b_)
    utts = [u for u in world.utts if a < u['ex'] <= b_]
    n = max(1, out['n_dialogues'])
    clear = [u for u in utts if u['garble'] == 0 and u['mumble'] == 0]
    out.update(clear_utts_per_dlg=round(len(clear) / n, 4),
               clear_single_syllable_per_dlg=round(sum(len(u['syl']) == 1 for u in clear) / n, 4),
               mean_ticks=round(float(np.mean([u['ticks'] for u in utts])), 2) if utts else None,
               when=dict(Counter(u['state'] for u in utts)),
               top_clear_syllables=Counter(s for u in clear for s in u['syl']).most_common(8),
               examples=[' '.join(u['syl']) for u in utts[-12:]])
    return out
