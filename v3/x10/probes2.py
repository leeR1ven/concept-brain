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


def l4_tracking(life, n_lines=24, seed=0):
    """[fix A check] does L4 follow the input? Frozen. Templates: L4 code of each syllable heard alone (C1, on in >= 2
    of the 3 slices at lag 4). Running speech: n_lines random 3-syllable lines (C1 / C2) with 6-10 silent ticks between;
    each syllable's L4 slices (lag 4) decoded by the best Jaccard to the templates (current-syllable accuracy, chance
    1/34), and J(L4(t), L4(t-3)) over speech ticks (how much L4 moves per syllable); L4 count in speech and silence."""
    b = life.brain
    rng = np.random.default_rng(seed + 6100)
    tmpl = []
    with Frozen(life):
        snap = b.snapshot()
        for x in range(NSYL):
            b.restore(snap)
            rec = []
            play(life, [None] * 4 + [(x, C1, 1.0)] * 3 + [None] * 6, mute=True, rec=lambda lf, t: rec.append(lf.brain.pfc.L4.on.clone()))
            tmpl.append(torch.stack(rec[8:11]).float().mean(0) >= 0.5)
        T = torch.stack(tmpl)
        b.restore(snap)
        script, lab = [None] * 6, [None] * 6
        for _ in range(n_lines):
            v = C1 if rng.random() < 0.5 else C2
            for x in rng.integers(0, NSYL, 3):
                script += [(int(x), v, 1.0)] * 3
                lab += [int(x)] * 3
            gap = int(rng.integers(6, 11))
            script += [None] * gap
            lab += [None] * gap
        rec, hrec = [], []
        play(life, script, mute=True, rec=lambda lf, t: (rec.append(lf.brain.pfc.L4.on.clone()),
                                                       hrec.append(lf.brain.hip.Hs.on.clone() if lf.brain.hip is not None else None)))
    hits, jm, jh, c_sp, c_si = [], [], [], [], []
    for t in range(4, len(script)):
        l4 = rec[t]
        x = lab[t - 4]
        (c_sp if lab[t - 4] is not None else c_si).append(float(l4.sum()))
        if x is not None:
            inter = (T & l4[None]).sum(1).float()
            union = (T | l4[None]).sum(1).float().clamp_min(1)
            hits.append(int((inter / union).argmax().item() == x))
            if t >= 3:
                jm.append(jac(l4, rec[t - 3]).item())
                if hrec[t] is not None:
                    jh.append(jac(hrec[t], hrec[t - 3]).item())
    return dict(current_syllable_decoding=float(np.mean(hits)), J_L4_t_vs_t3_speech=float(np.mean(jm)),
                J_Hs_t_vs_t3_speech=float(np.mean(jh)) if jh else None,
                L4_count_speech=float(np.mean(c_sp)), L4_count_silence=float(np.mean(c_si)), n=len(hits))


def hip_recall(life, Ds=(300,), after=8, n_long=4):
    """[fix B check] a held-out Q -> A pair heard once (C1 asks, C2 answers 3 ticks later; learning on), D ticks of the
    ordinary overheard world, then Q again (learning on). For `after` ticks after Q: the hippocampus's internally lit
    copies (h_int): A's code share and its rank among all 34 codes and among the answer syllables; the answer's
    sound-area units re-lit (share of A's chained sound units on, vs the same probe without the exposure); h_int per
    tick. Full state restored between trials."""
    from probes import tasks
    b = life.brain
    props, seen, held = tasks(life)
    ans_set = sorted({SID[W.answer(f, x, props)[0]] for f in range(4) for x in range(8)})
    src = b.chains.fwd_src
    full = b.snapshot_full()
    was = b.learning
    out = {}
    for D in Ds:
        for cond in ('heard_once', 'control'):
            share, rank_all, rank_ans, relit, hcount, l1x, drv = [], [], [], [], [], [], {}
            for f, x in (held if D <= 1000 else held[:n_long]):     # long gaps: fewer pairs (time)
                b.restore_full(full)
                b.learning = True
                Q, A = W.question(f, x), W.answer(f, x, props)
                a = SID[A[0]]
                if cond == 'heard_once':
                    play(life, line_script(Q, C1) + [None] * 3 + line_script(A, C2) + [None] * 12, mute=True)
                t0 = b.t
                while b.t - t0 < D:
                    life.tick()
                qs = line_script(Q, C1)
                t_end = len(qs) - 1
                H, S, X, DR = [], [], [], []

                cd = life.codes[a]
                cd = cd[src[cd, 0] == 0]                                  # A's sound-chain copies -> their sound units
                groups = [(l, src[cd[src[cd, 1] == l], 2]) for l in range(len(b.sound.n))]

                def rec(lf, t):
                    if t_end < t <= t_end + after:
                        H.append(lf.brain.hip.h_int.float())
                        DR.append(lf.brain.hip.dh1)
                        on = torch.cat([lf.brain.sound.state[l][u].float() for l, u in groups if u.numel()]) \
                            if cd.numel() else torch.zeros(1, device=DEV)
                        S.append(on.mean())
                        X.append(lf.brain.pfc.L1ext.on[life.codes[a]].float().mean() if life.codes[a].numel()
                                 else torch.zeros((), device=DEV))
                play(life, qs + [None] * (after + 1), mute=True, rec=rec)
                Hm = torch.stack(H).mean(0)
                per = torch.stack([Hm[cd].mean() if cd.numel() else torch.zeros((), device=DEV) for cd in life.codes])
                share.append(float(per[a]))
                rank_all.append(int((per > per[a]).sum().item()) + 1)
                pa = per[ans_set]
                rank_ans.append(int((pa > per[a]).sum().item()) + 1)
                relit.append(float(torch.stack(S).mean()))
                l1x.append(float(torch.stack(X).mean()))
                th = life.c['theta_h1']
                for key, k_ in (('raw', 0), ('sub', 1)):
                    per_t = torch.stack([torch.stack([dr[k_][cd2].mean() if cd2.numel() else torch.zeros((), device=DEV)
                                                      for cd2 in life.codes]) for dr in DR])        # [ticks, 34]
                    mx = per_t.max(0).values
                    drv.setdefault(key, []).append((float(mx[a] / th), float(torch.cat([mx[:a], mx[a + 1:]]).max() / th),
                                                    float(torch.stack([dr[k_].max() for dr in DR]).max() / th)))
                hcount.append(float(torch.stack(H).sum(1).mean()))
            out[f'{cond} D={D}'] = dict(A_code_hint_share=float(np.mean(share)), A_rank_all34=float(np.mean(rank_all)),
                                        A_first_all34=float(np.mean([r == 1 for r in rank_all])),
                                        A_rank_among_answers=float(np.mean(rank_ans)), n_answers=len(ans_set),
                                        A_sound_units_relit=float(np.mean(relit)), A_L1ext_copies_lit=float(np.mean(l1x)),
                                        h_int_per_tick=float(np.mean(hcount)),
                                        H1_drive_over_theta={k: dict(A_code_mean=float(np.mean([v[0] for v in vs])),
                                                                     best_other_code=float(np.mean([v[1] for v in vs])),
                                                                     max_any_copy=float(np.mean([v[2] for v in vs])))
                                                             for k, vs in drv.items()},
                                        n=len(share))
    b.restore_full(full)
    b.learning = was
    return out


def hip_trace(life, Ds=(300, 3000), after=8, n_long=4):
    """[fix B, B4] where does recall fail? Held-out pairs, each heard once (C1 asks Q; 3 silent ticks; C2 answers A;
    12 silent ticks; learning on), D ticks of the overheard world, then Q again (learning on). Full state restored per
    trial; the never-exposed control replays the same without the exposure.
    Hs level: (1) re-hearing Q: tick-by-tick J of Hs with this trial's first-hearing Q sequence vs other trials' Q
    sequences; (2) the ticks after Q: J with this trial's Hs at A's onset vs other trials' A onsets; (3) in the first
    hearing's pause (3 ticks between Q and A): the share of Q's sound units lit (by returns: no input), and the share of
    Hs that belongs to the Hs units active during Q.
    H1 level (after the re-heard Q): H1t drive into A's H1 copies vs the best other code, raw and after the patch-mean
    subtraction, in units of theta_h1; out-strength of the active Hs rows; in-strength of A's copies vs other copies;
    the mean learned weight from the active Hs rows onto A's copies vs onto other codes' copies; h_int per tick; A's
    sound units and L1ext copies lit."""
    from probes import tasks
    b = life.brain
    hip = b.hip
    props, seen, held = tasks(life)
    src = b.chains.fwd_src
    th = life.c['theta_h1']
    code_of = torch.full((b.n1,), -1, dtype=torch.long, device=DEV)
    for s_, cd in enumerate(life.codes):
        if cd.numel():
            code_of[cd] = s_

    def sound_units(syls):
        cd = torch.cat([life.codes[s_] for s_ in syls]) if syls else torch.zeros(0, dtype=torch.long, device=DEV)
        cd = cd[src[cd, 0] == 0]
        return [(l, src[cd[src[cd, 1] == l], 2]) for l in range(len(b.sound.n))]

    def lit_share(groups):
        parts = [b.sound.state[l][u].float() for l, u in groups if u.numel()]
        return float(torch.cat(parts).mean()) if parts else 0.0

    full = b.snapshot_full()
    was = b.learning
    out = {}
    for D in Ds:
        pairs = held if D <= 1000 else held[:n_long]
        trials = {}
        for cond in ('heard_once', 'control'):
            rows = []
            for f, x in pairs:
                b.restore_full(full)
                b.learning = True
                Q, A = W.question(f, x), W.answer(f, x, props)
                a = SID[A[0]]
                qsyl = [SID[s_] for s_ in Q]
                qs, as_ = line_script(Q, C1), line_script(A, C2)
                nq = len(qs)
                first = dict(Hs=[], gap_sound=[], gap_hs_q=[])
                if cond == 'heard_once':
                    qgroups = sound_units(qsyl)
                    seqH = []

                    def rec1(lf, t):
                        seqH.append(hip.Hs.on.clone())
                        if nq <= t < nq + 3:
                            first['gap_sound'].append(lit_share(qgroups))
                    play(life, qs + [None] * 3 + as_ + [None] * 12, mute=True, rec=rec1)
                    first['Hs'] = seqH
                    qunion = torch.stack(seqH[2:nq + 2]).any(0)
                    for t in range(nq, nq + 3):
                        hs = seqH[t]
                        first['gap_hs_q'].append(float((hs & qunion).sum()) / max(1.0, float(hs.sum())))
                t0 = b.t
                while b.t - t0 < D:
                    life.tick()
                seqR, H, S, X, DR, act = [], [], [], [], [], []
                agroups = sound_units([a])

                def rec2(lf, t):
                    seqR.append(hip.Hs.on.clone())
                    if nq - 1 < t <= nq - 1 + after:
                        H.append(hip.h_int.float())
                        DR.append(hip.dh1)
                        S.append(lit_share(agroups))
                        X.append(b.pfc.L1ext.on[life.codes[a]].float().mean() if life.codes[a].numel() else torch.zeros((), device=DEV))
                        act.append(hip.Hs.on.clone())
                play(life, qs + [None] * (after + 1), mute=True, rec=rec2)
                r = dict(a=a, nq=nq, first=first, seqR=seqR)
                per = {}
                for key, k_ in (('raw', 0), ('sub', 1)):
                    pt = torch.stack([torch.stack([dr[k_][cd2].mean() if cd2.numel() else torch.zeros((), device=DEV)
                                                   for cd2 in life.codes]) for dr in DR]).max(0).values
                    per[key] = (float(pt[a] / th), float(torch.cat([pt[:a], pt[a + 1:]]).max() / th))
                r['drive'] = per
                rows_act = torch.nonzero(torch.stack(act).any(0)).squeeze(1)
                T_ = hip.H1t
                r['out_strength_active_rows'] = float(T_.out[rows_act].mean()) if rows_act.numel() else 0.0
                ca = life.codes[a]
                r['in_strength_A'] = float(T_.inn[ca].mean()) if ca.numel() else 0.0
                r['in_strength_all'] = float(T_.inn[:b.n1].mean())
                if rows_act.numel():
                    ix = T_.idx[rows_act].long()
                    w = T_.w[rows_act]
                    tgt = torch.where(ix < b.n1, code_of[ix.clamp_max(b.n1 - 1)], torch.full_like(ix, -1))
                    r['w_to_A'] = float(w[tgt == a].mean()) if bool((tgt == a).any()) else 0.0
                    r['w_to_other_codes'] = float(w[(tgt >= 0) & (tgt != a)].mean()) if bool(((tgt >= 0) & (tgt != a)).any()) else 0.0
                r['h_int_per_tick'] = float(torch.stack(H).sum(1).mean())
                Hm = torch.stack(H).mean(0)
                pc = torch.stack([Hm[cd2].mean() if cd2.numel() else torch.zeros((), device=DEV) for cd2 in life.codes])
                r['A_hint_share'] = float(pc[a])
                r['other_hint_share_max'] = float(torch.cat([pc[:a], pc[a + 1:]]).max())
                r['A_sound_relit'] = float(np.mean(S))
                r['A_L1ext_lit'] = float(torch.stack(X).mean())
                rows.append(r)
            trials[cond] = rows
        # Hs-level comparisons across trials (heard_once)
        ho = trials['heard_once']
        jq_own, jq_oth, ja_own, ja_oth = [], [], [], []
        for i, r in enumerate(ho):
            nq = r['nq']
            for j, r2 in enumerate(ho):
                L = min(nq, r2['nq'])
                jq = np.mean([jac(r['seqR'][t], r2['first']['Hs'][t]).item() for t in range(2, L)])
                k0, k1 = r2['nq'] + 3 + 2, nq + 3 + 2                     # Hs at A's onset (+2 ticks relay), same timing
                ja = np.mean([jac(r['seqR'][k1 + k], r2['first']['Hs'][k0 + k]).item() for k in range(3)
                              if k1 + k < len(r['seqR']) and k0 + k < len(r2['first']['Hs'])])
                (jq_own if i == j else jq_oth).append(jq)
                (ja_own if i == j else ja_oth).append(ja)
        summ = {}
        for cond, rows in trials.items():
            summ[cond] = dict(
                n=len(rows),
                H1_drive_over_theta={k: dict(A=float(np.mean([r['drive'][k][0] for r in rows])),
                                             best_other=float(np.mean([r['drive'][k][1] for r in rows]))) for k in ('raw', 'sub')},
                out_strength_active_Hs_rows=float(np.mean([r['out_strength_active_rows'] for r in rows])),
                in_strength_A_copies=float(np.mean([r['in_strength_A'] for r in rows])),
                in_strength_all_copies=float(np.mean([r['in_strength_all'] for r in rows])),
                w_active_Hs_to_A=float(np.mean([r.get('w_to_A', 0.0) for r in rows])),
                w_active_Hs_to_other_codes=float(np.mean([r.get('w_to_other_codes', 0.0) for r in rows])),
                h_int_per_tick=float(np.mean([r['h_int_per_tick'] for r in rows])),
                A_hint_share=float(np.mean([r['A_hint_share'] for r in rows])),
                other_hint_share_max=float(np.mean([r['other_hint_share_max'] for r in rows])),
                A_sound_relit=float(np.mean([r['A_sound_relit'] for r in rows])),
                A_L1ext_lit=float(np.mean([r['A_L1ext_lit'] for r in rows])))
        summ['heard_once'].update(
            Hs_rehear_J_own_Q=float(np.mean(jq_own)), Hs_rehear_J_other_Q=float(np.mean(jq_oth)),
            Hs_after_Q_J_own_A_onset=float(np.mean(ja_own)), Hs_after_Q_J_other_A_onsets=float(np.mean(ja_oth)),
            first_pause_Q_sound_units_lit=float(np.mean([np.mean(r['first']['gap_sound']) for r in ho])),
            first_pause_Hs_share_from_Q=float(np.mean([np.mean(r['first']['gap_hs_q']) for r in ho])))
        out[f'D={D}'] = summ
    b.restore_full(full)
    b.learning = was
    return out
