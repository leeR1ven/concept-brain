"""Frozen probes and diagnostics [PLAN 6]. Every probe: learning off, the brain's neuron states restored afterwards
(weights are untouched because learning is off). World labels are used only here, for scoring."""
from collections import Counter

import numpy as np
import torch

import world as W
from common import DEV
from world_adapter import ADULTS, C1, C2, SID, SYL, contexts, line_script


# ---------------------------------------------------------------------------------------------- driving a probe
def play(life, script, mute=False, rec=None, force=None):
    """script: list of adult sources per tick (None = silence). force: dict tick -> bool mask over PFC copies
    (DIAGNOSTIC forced inner hearing). rec(life, t) is called after every tick."""
    b, env = life.brain, life.probe_env
    m = np.zeros(24, bool)
    for t, adult in enumerate(script):
        b.force_int = None if force is None else force.get(t)
        a0 = env.step(None if mute else m, adult=adult, mute=mute)
        m = b.step(torch.as_tensor(a0, device=DEV), 0.0).cpu().numpy()
        life.last_mouth = m                                  # muscles output this tick (read by probes)
        if rec is not None:
            rec(life, t)
    b.force_int = None
    return m


class Frozen:
    """context manager: learning off, neuron states restored at exit"""

    def __init__(self, life):
        self.life = life

    def __enter__(self):
        b = self.life.brain
        self.was = b.learning
        b.learning = False
        self.snap = b.snapshot()
        return self.snap

    def __exit__(self, *a):
        b = self.life.brain
        b.restore(self.snap)
        b.learning = self.was


def tasks(life):
    w = life.env.w
    seen = [(f, x) for f in range(4) for x in range(8) if (f, x) not in w.held]
    held = sorted(w.held)
    return w.props, seen, held


# ---------------------------------------------------------------------------------------------- (a1) activity
def activity(life, n_dialogues):
    """run the world (learning as set) and record per-tick counts with the silence / adult-speech label"""
    b = life.brain
    rows, lab = [], []

    def rec(lf):
        rows.append(b.counts())
        lab.append(lf.env.adult_speaking())
    life.run_dialogues(n_dialogues, rec)
    C = torch.stack(rows).float().cpu().numpy()
    lab = np.array(lab)
    names = b.count_names()
    sizes = dict(zip([f'A{l + 1}' for l in range(len(b.sound.n))], b.sound.n))
    sizes.update(L1ext=b.n1, L1int=b.n1, L1=b.n1, L2=b.pfc.n, L3=b.pfc.n, L4=b.pfc.n, force=b.motor.nf,
                 stop=b.motor.ns)
    if b.hip is not None:
        sizes.update(H1=b.n1, H1int=b.n1, Hs=b.hip.nh)
    if getattr(b, 'touch', None) is not None:
        sizes.update({f'T{l + 1}': n for l, n in enumerate(b.touch.n)})
    out = {}
    for i, n in enumerate(names):
        sil, sp = C[~lab, i].mean(), C[lab, i].mean()
        out[n] = dict(silence=float(sil), speech=float(sp), ratio=float(sil / max(sp, 1e-9)),
                      min=float(C[:, i].min()), max=float(C[:, i].max()), frac_max=float(C[:, i].max() / sizes[n]))
    ix = {n: i for i, n in enumerate(names)}
    for a, bb in (('L2', 'L1'), ('L3', 'L2'), ('L4', 'L3')):
        for nm, m in (('silence', ~lab), ('speech', lab)):
            out[f'{a}/{bb} {nm}'] = float(C[m, ix[a]].mean() / max(C[m, ix[bb]].mean(), 1e-9))
    return out


def a1_pass(act, nA=4):
    """a1 (coordinator 11-02): A0 exactly half on (asserted every tick in Cochlea.code); PFC layer-to-layer ratios
    in [0.67, 1.5] in silence and in speech; no PFC or Hs layer ever empty or > 50 % on; silence L1 >= 0.2 x
    speech L1. Silence/speech ratios per area and layer are reported, not judged."""
    ok = all(0.67 <= act[f'{a}/{bb} {nm}'] <= 1.5 for a, bb in (('L2', 'L1'), ('L3', 'L2'), ('L4', 'L3'))
             for nm in ('silence', 'speech'))
    for n in ('L1', 'L2', 'L3', 'L4', 'Hs'):
        if n in act:
            ok = ok and act[n]['min'] > 0 and act[n]['frac_max'] <= 0.5
    ok = ok and act['L1']['silence'] >= 0.2 * act['L1']['speech']
    return bool(ok)


def stability(life, n_dialogues=300, window=10):
    """a1L (coordinator 11-02): learning on, n overheard dialogues. In every window of `window` dialogues the PFC
    layer ratios stay in [0.5, 2] and Hs's mean stays in [0.5, 2] x kh; no PFC/Hs layer is ever > 50 % on."""
    b = life.brain
    was = b.learning
    b.learning = True
    rows, dl, lab = [], [], []
    d0 = life.env.w.n_ex

    def rec(lf):
        rows.append(lf.brain.counts())
        dl.append(lf.env.w.n_ex)
        lab.append(lf.env.adult_speaking())
    life.run_dialogues(n_dialogues, rec)
    b.learning = was
    C = torch.stack(rows).float().cpu().numpy()
    dl = np.array(dl)
    lab = np.array(lab)
    names = b.count_names()
    budget = {}                                           # activity budget after 100 and 300 learning dialogues
    for k in (100, 300):
        m = (dl >= d0 + k - 10) & (dl < d0 + k)
        if m.sum() > 20:
            budget[k] = {n: dict(silence=float(C[m & ~lab, i].mean()) if (m & ~lab).any() else None,
                                 speech=float(C[m & lab, i].mean()) if (m & lab).any() else None)
                         for i, n in enumerate(names)}
    ix = {n: i for i, n in enumerate(b.count_names())}
    sizes = dict(L1=b.n1, L2=b.pfc.n, L3=b.pfc.n, L4=b.pfc.n, Hs=b.hip.nh if b.hip is not None else 1)
    wins = []
    for w0 in range(dl.min(), dl.max() + 1, window):
        m = (dl >= w0) & (dl < w0 + window)
        if m.sum() < 50:
            continue
        mean = C[m].mean(0)
        wmax = max(float(C[m][:, ix[n]].max() / sizes[n]) for n in sizes if n in ix)
        wins.append(dict(dialogue=int(w0 - d0), L2_L1=float(mean[ix['L2']] / max(mean[ix['L1']], 1)),
                         L3_L2=float(mean[ix['L3']] / max(mean[ix['L2']], 1)),
                         L4_L3=float(mean[ix['L4']] / max(mean[ix['L3']], 1)),
                         Hs=float(mean[ix['Hs']]) if 'Hs' in ix else None,
                         L4=float(mean[ix['L4']]), A4=float(mean[ix['A4']]),
                         L4_frac=float(mean[ix['L4']] / sizes['L4']), max_frac=wmax))
    fmax = {n: float(C[:, ix[n]].max() / sizes[n]) for n in sizes if n in ix}
    kh = b.c['kh']

    def win_ok(w):
        ok = all(0.5 <= w[k] <= 2.0 for k in ('L2_L1', 'L3_L2', 'L4_L3')) and w['max_frac'] <= 0.5
        return ok and (b.hip is None or 0.5 * kh <= w['Hs'] <= 2.0 * kh)
    for w in wins:
        w['ok'] = bool(win_ok(w))
    ok = all(w['ok'] for w in wins)                       # the 300-dialogue a1L: every window
    out = dict(pass_=bool(ok), windows=wins, frac_max=fmax, budget=budget,
               share_windows_ok=float(np.mean([w['ok'] for w in wins])) if wins else 0.0)
    if n_dialogues >= 1000:                               # extended a1L (pre-registered 11-02): >= 95 % of windows
        def mean_in(key, lo, hi):                         # ok, and no sustained rise over the last 500 dialogues
            v = [w[key] for w in wins if lo <= w['dialogue'] < hi]
            return float(np.mean(v)) if v else float('nan')
        trend = {}
        for key in ('L4', 'Hs', 'A4'):
            at500, last = mean_in(key, 490, 510), mean_in(key, n_dialogues - 100, n_dialogues)
            xs = [w['dialogue'] for w in wins if w['dialogue'] >= n_dialogues - 500]
            ys = [w[key] for w in wins if w['dialogue'] >= n_dialogues - 500]
            slope = float(np.polyfit(xs, ys, 1)[0] * 100) if len(xs) > 2 else float('nan')   # per 100 dialogues
            trend[key] = dict(at500=at500, last100=last, ratio=last / max(at500, 1e-9), slope_per100=slope)
        out['trend'] = trend
        out['pass_ext'] = bool(out['share_windows_ok'] >= 0.95 and trend['L4']['ratio'] <= 1.2
                               and trend['Hs']['ratio'] <= 1.2)
    return out


# ---------------------------------------------------------------------------------------------- (a2, a3) codes
def question_codes(life, n_ctx=4, n_noise=2, lags=12, seed=0):
    """32 questions x contexts x noise, mouth muted: PFC L4 (and Hs) states at lags 0..lags after the question."""
    b = life.brain
    props, _, _ = tasks(life)
    rng = np.random.default_rng(seed + 9100)
    L4, HS, key = [], [], []
    with Frozen(life):
        snap = b.snapshot()
        for f in range(4):
            for x in range(8):
                ctxs = contexts(rng, props)[:n_ctx]
                for ci, ctx in enumerate(ctxs):
                    for nz in range(n_noise):
                        b.restore(snap)
                        q = line_script(W.question(f, x), C1 if nz % 2 == 0 else C2)
                        script = ctx + [None] * 4 + q + [None] * lags
                        t_end = len(ctx) + 4 + len(q) - 1
                        st4, sth = [], []

                        def rec(lf, t):
                            if t >= t_end:
                                st4.append(lf.brain.pfc.L4.on.clone())
                                if lf.brain.hip is not None:
                                    sth.append(lf.brain.hip.Hs.on.clone())
                        play(life, script, mute=True, rec=rec)
                        L4.append(torch.stack(st4))
                        if sth:
                            HS.append(torch.stack(sth))
                        key.append((f, x, ci, nz))
    return torch.stack(L4), (torch.stack(HS) if HS else None), key


def decode(S, key, lag, classes):
    """nearest centroid, 2-fold over contexts (train on half the contexts, test on the others).
    classes: function key -> label (None = excluded). Returns accuracy."""
    X = S[:, lag].float()
    ctx = np.array([k[2] for k in key])
    lab = [classes(k) for k in key]
    use = np.array([l is not None for l in lab])
    labs = sorted({l for l in lab if l is not None})
    li = {l: i for i, l in enumerate(labs)}
    y = np.array([li[l] if l is not None else -1 for l in lab])
    cs = sorted(set(ctx))
    half = set(cs[:len(cs) // 2])
    acc = []
    for train in (half, set(cs) - half):
        tr = torch.as_tensor(np.array([c in train for c in ctx]) & use, device=DEV)
        te = torch.as_tensor(np.array([c not in train for c in ctx]) & use, device=DEV)
        yt = torch.as_tensor(y, device=DEV)
        cent = torch.stack([X[tr & (yt == i)].mean(0) if (tr & (yt == i)).any() else torch.full_like(X[0], 1e9)
                            for i in range(len(labs))])
        d = torch.cdist(X[te], cent)
        acc.append((d.argmin(1) == yt[te]).float().mean().item())
    return float(np.mean(acc))


def jac(a, b):
    a, b = a.bool(), b.bool()
    return ((a & b).sum(-1).float() / (a | b).sum(-1).clamp_min(1).float())


def newborn_codes(life, n_ctx=4, n_noise=2, lags=12):
    """a2 and a3 metrics"""
    L4, HS, key = question_codes(life, n_ctx, n_noise, lags)
    q_of = lambda k: (k[0], k[1])
    res = dict(chance=1 / 32)
    res['L4_decode'] = {lag: decode(L4, key, lag, q_of) for lag in (0, 3, 6, 9, 12) if lag <= lags}
    res['L4_filler_given_frame_lag12'] = float(np.mean([decode(L4, key, min(12, lags),
                                                               lambda k, f=f: k[1] if k[0] == f else None)
                                                        for f in range(4)]))
    if HS is not None:
        res['Hs_decode'] = {lag: decode(HS, key, lag, q_of) for lag in (0, 6, 12) if lag <= lags}
    same, cross = [], []
    K = len(key)
    for i in range(K):
        for j in range(i + 1, K):
            if key[i][2] == key[j][2]:
                continue
            if (key[i][0], key[i][1]) == (key[j][0], key[j][1]):
                same.append(jac(L4[i, 0], L4[j, 0]).item())
            elif key[i][0] == key[j][0]:
                cross.append(jac(L4[i, 0], L4[j, 0]).item())
    res['J_same_q_other_ctx'] = float(np.mean(same))
    res['J_other_q_same_frame'] = float(np.mean(cross))
    res['J_margin'] = res['J_same_q_other_ctx'] - res['J_other_q_same_frame']
    if HS is not None:                                     # the same measures on the hippocampal population
        hs_same, hs_cross = [], []
        for i in range(K):
            for j in range(i + 1, K):
                if key[i][2] == key[j][2]:
                    continue
                if (key[i][0], key[i][1]) == (key[j][0], key[j][1]):
                    hs_same.append(jac(HS[i, 0], HS[j, 0]).item())
                elif key[i][0] == key[j][0]:
                    hs_cross.append(jac(HS[i, 0], HS[j, 0]).item())
        res['Hs_J_same_q_other_ctx'] = float(np.mean(hs_same))
        res['Hs_J_other_q_same_frame'] = float(np.mean(hs_cross))
        res['Hs_J_margin'] = res['Hs_J_same_q_other_ctx'] - res['Hs_J_other_q_same_frame']
    res['motion_J_t_t1'] = float(jac(L4[:, 1:], L4[:, :-1]).mean().item())
    res['L4_count_lag'] = L4.float().sum(-1).mean(0).tolist()
    # a2 (revised 11-02, coordinator: ARCH 4.19 + designer "模型分不开相同的经历很正常，先不管"): the hippocampal part is
    # the reproducibility margin, not 32-way decoding (which is still reported as a number)
    res['a2_pass'] = bool(res['L4_decode'][0] >= 0.5 and res['J_margin'] >= 0.1
                          and (HS is None or res['Hs_J_margin'] >= 0.1))
    res['a3_pass'] = bool(res['L4_decode'].get(12, 0) >= 0.3 and res['L4_filler_given_frame_lag12'] >= 0.3
                          and res['motion_J_t_t1'] < 0.9)
    return res


# ---------------------------------------------------------------------------------------------- (b) speech basics
def code_mask(life, syls):
    m = torch.zeros(life.brain.n1, dtype=torch.bool, device=DEV)
    for s in syls:
        m[life.codes[s]] = True
    return m


def forced_say(life, syl_seq, adult_at=None, adult_line=None, free=10):
    """DIAGNOSTIC: force the inner hearing of each syllable for 3 ticks in turn; return what the child said."""
    env = life.probe_env
    T = 6 + 3 * len(syl_seq) + free
    script = [None] * T
    if adult_at is not None:
        for i, src in enumerate(adult_line):
            if adult_at + i < T:
                script[adult_at + i] = src
    force = {}
    for i, s in enumerate(syl_seq):
        for k in range(3):
            force[6 + 3 * i + k] = code_mask(life, [s])
    env.reset_track()
    play(life, script, force=force)
    return list(env.syl), list(env.events)


def speech_basics(life, n_overhear=1000):
    b = life.brain
    props, seen, held = tasks(life)
    rng = np.random.default_rng(life.seed + 9300)
    out = {}
    with Frozen(life) as snap:
        said = []
        loop = 0
        for s in range(len(SYL)):
            b.restore(snap)
            u, _ = forced_say(life, [SID[SYL[s]]])
            said.append(u == [SYL[s]])
            loop += int(u.count(SYL[s]) >= 2)
        out['b1_forced_syllable_said'] = float(np.mean(said))
        out['b3_own_voice_loop'] = loop / len(SYL)
        ok, gar = [], 0
        for f in range(4):
            for x in range(8):
                b.restore(snap)
                A = W.answer(f, x, props)
                u, _ = forced_say(life, [SID[a] for a in A])
                ok.append(u == A)
                gar += int('~' in u)
        out['b2_forced_answer_said'] = float(np.mean(ok))
        out['b1b2_garble'] = gar / 32
        echo = 0
        lines = 0
        for f in range(4):
            for x in range(8):
                for line in (W.question(f, x), W.answer(f, x, props)):
                    b.restore(snap)
                    life.probe_env.reset_track()
                    v = C1 if lines % 2 == 0 else C2
                    play(life, line_script(line, v) + [None] * 6)
                    echo += int(len(life.probe_env.syl) > 0)
                    lines += 1
        out['b4_echo_per_line'] = echo / lines
        mix_g = 0
        for i in range(len(SYL)):
            b.restore(snap)
            s1, s2 = rng.choice(len(SYL), 2, replace=False)
            life.probe_env.reset_track()
            play(life, [None] * 6 + [None] * 3 + [None] * 10,
                 force={6 + k: code_mask(life, [int(s1), int(s2)]) for k in range(3)})
            mix_g += int('~' in life.probe_env.syl)
        out['b5_mixture_garble'] = mix_g / len(SYL)
        stopped, n = 0, 0
        for f in range(4):
            for x in range(8):
                A = W.answer(f, x, props)
                if len(A) < 2:
                    continue
                b.restore(snap)
                line = line_script(W.question(int(rng.integers(4)), int(rng.integers(8))), C2)
                at = 6 + 2                                                  # the adult starts during syllable 1
                u, ev = forced_say(life, [SID[a] for a in A], adult_at=at, adult_line=line)
                late = [e for e in ev if e[0] > at + 6]                     # onsets beyond the next syllable
                stopped += int(len(late) == 0)
                n += 1
        out['b6_brake_stops'] = stopped / max(1, n)
    # b7: newborn while overhearing (learning on), then b1 again
    b.learning = True
    w = life.env.w
    u0, n0 = len(w.utts), w.n_ex
    life.run_dialogues(n_overhear)
    out['b7_utterances_per_dialogue'] = (len(w.utts) - u0) / max(1, w.n_ex - n0)
    b.learning = False
    with Frozen(life) as snap:
        said = []
        for s in range(len(SYL)):
            b.restore(snap)
            u, _ = forced_say(life, [s])
            said.append(u == [SYL[s]])
        out['b7_b1_after_overhearing'] = float(np.mean(said))
    out['pass'] = dict(b1=out['b1_forced_syllable_said'] >= 0.9 and out['b1b2_garble'] <= 0.05,
                       b2=out['b2_forced_answer_said'] >= 0.9, b3=out['b3_own_voice_loop'] <= 0.05,
                       b4=out['b4_echo_per_line'] <= 0.02, b5=out['b5_mixture_garble'] <= 0.1,
                       b6=out['b6_brake_stops'] >= 0.9,
                       b7=out['b7_utterances_per_dialogue'] <= 0.05 and out['b7_b1_after_overhearing'] >= 0.85)
    return out


# ---------------------------------------------------------------------------------------------- (c) question answering
def frame_baseline(props, seen):
    hit = 0
    maj = {}
    for f in range(4):
        ans = [tuple(W.answer(g, x, props)) for g, x in seen if g == f]
        a, k = Counter(ans).most_common(1)[0]
        maj[f] = a
        hit += k
    return hit / len(seen), maj


def qa_probe(life, wait=12, reps=2):
    """a caregiver (C1 or C2) asks each of the 32 questions after a short context and waits; the child's answer"""
    b = life.brain
    props, seen, held = tasks(life)
    base, maj = frame_baseline(props, seen)
    rec = []
    d3 = []
    with Frozen(life) as snap:
        for f in range(4):
            for x in range(8):
                A = W.answer(f, x, props)
                for r in range(reps):
                    b.restore(snap)
                    life.probe_env.reset_track()
                    q = line_script(W.question(f, x), C1 if r % 2 == 0 else C2)
                    script = [None] * 10 + q + [None] * wait
                    t_end = 10 + len(q) - 1
                    drv, rets = [], []

                    def rd(lf, t):
                        if t_end < t <= t_end + 6:
                            drv.append(lf.brain.pfc.d1.clone())
                            rets.append(lf.brain.ret_sound[0] > 0)
                    play(life, script, rec=rd)
                    u = list(life.probe_env.syl)
                    rec.append(dict(f=f, x=x, A=A, said=u, seen=(f, x) in seen,
                                    exact=u == A, first=len(u) > 0 and u[0] == A[0], silent=len(u) == 0,
                                    garble='~' in u, minority=tuple(A) != maj[f] if (f, x) in seen else None))
                    D = torch.stack(drv).mean(0)                                 # [n1] internal drive, lags 1-6
                    per = torch.stack([D[cd].mean() if cd.numel() else torch.zeros((), device=DEV)
                                       for cd in life.codes])
                    a1 = SID[A[0]]
                    wrong = torch.cat([per[:a1], per[a1 + 1:]])
                    n_a1 = len(b.chains.sound_idx[0])                            # d4: answer's A1 units re-lit
                    cd = life.codes[a1]
                    cd = cd[cd < n_a1]
                    units = b.chains.sound_idx[0][cd]
                    lit = torch.stack(rets).any(0)[units].float().mean().item() if cd.numel() else 0.0
                    d3.append(dict(own=per[a1].item(), best_wrong=wrong.max().item(),
                                   top=int(per.argmax().item() == a1), hub=int(per.argmax().item()), d4=lit))
    S = [r for r in rec if r['seen']]
    H = [r for r in rec if not r['seen']]
    mino = [r for r in S if r['minority']]
    acc = lambda R, k: float(np.mean([r[k] for r in R])) if R else None
    hubs = Counter(d['hub'] for d in d3)
    out = dict(frame_baseline=base,
               seen=dict(exact=acc(S, 'exact'), first=acc(S, 'first'), silent=acc(S, 'silent'),
                         garble=acc(S, 'garble'), wrong=float(np.mean([(not r['exact']) and not r['silent']
                                                                       for r in S]))),
               minority_exact=acc(mino, 'exact'), n_minority=len(mino) // max(1, reps),
               held=dict(exact=acc(H, 'exact'), first=acc(H, 'first'), silent=acc(H, 'silent')),
               d3=dict(own=float(np.mean([d['own'] for d in d3])), best_wrong=float(np.mean([d['best_wrong']
                                                                                             for d in d3])),
                       answer_top_driven=float(np.mean([d['top'] for d in d3])),
                       hub_share=hubs.most_common(1)[0][1] / len(d3), hub_syllable=SYL[hubs.most_common(1)[0][0]]),
               d4_answer_A1_units_relit=float(np.mean([d['d4'] for d in d3])),
               answers=[(r['f'], r['x'], ''.join(r['A']), ''.join(r['said'])) for r in rec])
    s = out['seen']
    out['pass'] = bool(out['minority_exact'] is not None and out['minority_exact'] >= 0.5
                       and s['exact'] >= base + 0.1 and s['wrong'] <= 0.25 and s['garble'] <= 0.1)
    return out


def table_stats(life):
    """d6: positive-pair fraction of the acquired tables; d10: birth-loop push vs strongest learned push"""
    b = life.brain
    out = {}
    for tb in b.plastic():
        valid = (tb.idx < tb.n_post)
        out[tb.name] = dict(pos_frac=float(((tb.w > 0) & valid).sum() / valid.sum().clamp_min(1)),
                            w_mean_pos=float(tb.w[tb.w > 0].mean()) if (tb.w > 0).any() else 0.0,
                            w_max=float(tb.w.max()))
    lp = b.pfc.loop.w[b.pfc.loop.w > 0].mean().item()
    L = b.pfc.L41
    if (L.w > 0).any():
        rows = torch.nonzero(L.w.max(1).values > 0).squeeze(1)[:2000]
        ix = L.idx[rows].long()
        p = L.gain * L.w[rows] / torch.sqrt(L.out[rows].clamp_min(L.c['w_ref'])[:, None] *
                                            L.inn[ix].clamp_min(L.c['w_ref']))
        v = p[L.w[rows] > 0]
        out['d10'] = dict(loop_link=lp, learned_link_p90=float(torch.kthvalue(v, max(1, int(0.9 * v.numel()))).values))
    else:
        out['d10'] = dict(loop_link=lp, learned_link_p90=0.0)
    return out


def return_share(life, n_dialogues=50):
    """d8: share of return-lit copies that come from the hippocampus (vs the PFC)"""
    b = life.brain
    hp, pf = [], []

    def rec(lf):
        pf.append(lf.brain.pfc.x_int.sum())
        if lf.brain.hip is not None:
            hp.append(lf.brain.hip.h_int.sum())
    life.run_dialogues(n_dialogues, rec)
    P = torch.stack(pf).float().sum().item()
    H = torch.stack(hp).float().sum().item() if hp else 0.0
    return dict(pfc_returns=P, hip_returns=H, hip_share=H / max(1.0, P + H))


def lateral_sign(life, ticks=1000, sample=500):
    """d7: effective lateral weight j -> i (learned L4<->L4 propagation minus the patch neuron's share) against the
    co-activity of the pair, over `ticks` ticks of the running world (learning as set)."""
    b = life.brain
    P = b.pfc
    rows = []

    def rec(lf):
        rows.append(lf.brain.pfc.L4.on.clone())
    t0 = b.t
    while b.t - t0 < ticks:
        life.tick()
        rec(life)
    S = torch.stack(rows).float()                                         # [T, n4]
    g = torch.Generator(device=DEV)
    g.manual_seed(7)
    J = torch.randperm(P.n, generator=g, device=DEV)[:sample]
    L = P.L44
    ix = L.idx[J].long()                                                  # [m, C]
    valid = ix < P.n
    ixc = ix.clamp_max(P.n - 1)
    p = L.gain * L.w[J] / torch.sqrt(L.out[J].clamp_min(L.c['w_ref'])[:, None] * L.inn[ix].clamp_min(L.c['w_ref']))
    p = p * valid
    pid, n_p = P.patch[3]
    cnt = torch.zeros(n_p, device=DEV).index_add_(0, pid, torch.ones(P.n, device=DEV))
    psum = torch.zeros((len(J), n_p), device=DEV).scatter_add_(1, pid[ixc] * valid, p)
    g_pm = P.c['g_pm'] if P.c['inhib'] == 'P1' else 0.0
    eff = p - g_pm * psum.gather(1, pid[ixc]) / cnt[pid[ixc]]
    co = (S[:, J].T @ S) / S.shape[0]                                    # [m, n4] joint firing probability
    rate = S.mean(0)
    lift = co.gather(1, ixc) / (rate[J][:, None] * rate[ixc]).clamp_min(1e-6)
    e, l = eff[valid & (rate[ixc] > 0)], lift[valid & (rate[ixc] > 0)]
    if e.numel() < 8:
        return dict(n=int(e.numel()))
    q = torch.quantile(l[:min(l.numel(), 1 << 23)], torch.tensor([0.25, 0.75], device=DEV))
    lo, hi = e[l <= q[0]], e[l >= q[1]]
    return dict(n=int(e.numel()), eff_low_coactivity=float(lo.mean()), eff_high_coactivity=float(hi.mean()),
                pass_=bool(lo.mean() < 0 < hi.mean()))


# ---------------------------------------------------------------------------------------------- (b) speech links
def classify(syls, events, x):
    """one utterance after a probe: 'clear' / 'mumble' (x said only as a mumble) / 'garble' / 'other' / 'silent'"""
    if not syls:
        return 'silent'
    if '~' in syls:
        return 'garble'
    if syls == [SYL[x]]:
        return 'clear' if any(e[2] == 's' for e in events) else 'mumble'
    return 'other'


def speech_links(life, reps=1):
    """b1 by inner hearing (x's sound copies lit internally in PFC L1) and by L4 holding an adult-voice x (an adult
    says x, then 12 silent ticks): shares said clearly / mumble / garble / other / silent, per probe kind."""
    b = life.brain
    out = {}
    with Frozen(life) as snap:
        for kind in ('inner', 'adult_hold'):
            cnt = Counter()
            for x in range(len(SYL)):
                for r in range(reps):
                    b.restore(snap)
                    life.probe_env.reset_track()
                    if kind == 'inner':
                        play(life, [None] * 6 + [None] * 3 + [None] * 12,
                             force={6 + k: code_mask(life, [x]) for k in range(3)})
                    else:
                        play(life, [None] * 6 + [(x, C1 if r % 2 == 0 else C2, 1.0)] * 3 + [None] * 12)
                    cnt[classify(list(life.probe_env.syl), list(life.probe_env.events), x)] += 1
            n = sum(cnt.values())
            out[kind] = {k: cnt[k] / n for k in ('clear', 'mumble', 'garble', 'other', 'silent')}
    return out


def force_link_stats(life, sel):
    """learned + wired L4 -> force-copy link of each syllable's L4 unit set (sel: bool [34, n4]): true/other weight
    ratio, top-4 exact, and margin = weakest own-muscle drive / strongest other-muscle drive (X7: ~1.5 needed)"""
    b, ph = life.brain, life.ph
    mot, pfc = b.motor, b.pfc
    seg = [sg for sg in b.chains.segs if sg[0] == 'force'][0]
    tabs = [pfc.L41] + ([pfc.G1P] if pfc.G1P is not None else [])
    exact, margin, ratio = [], [], []
    L = pfc.L41
    isf_all = (L.idx >= seg[2]) & (L.idx < seg[3])
    mus_all = (L.idx.long() - seg[2]).clamp(0, mot.nf - 1)
    learners = []
    for x in range(len(SYL)):
        ownx = torch.as_tensor(ph.need_mat[x], device=DEV)[mot.muscle_of[mus_all]] & isf_all
        learners.append(int(((L.w > 0.01) & ownx).any(1).sum()))
    for x in range(len(SYL)):
        if not bool(sel[x].any()):
            continue
        d = sum(t.drive(sel[x]) for t in tabs)[seg[2]:seg[3]]
        per = torch.zeros(24, device=DEV).index_add_(0, mot.muscle_of, d)
        own = torch.as_tensor(ph.need_mat[x], device=DEV)
        exact.append(set(torch.topk(per, 4).indices.tolist()) == set(torch.nonzero(own).squeeze(1).tolist()))
        margin.append(float(per[own].min() / per[~own].max().clamp_min(1e-9)))
        rows = torch.nonzero(sel[x]).squeeze(1)                    # raw L41 weights from x's units to force copies
        ix = pfc.L41.idx[rows].long()
        w = pfc.L41.w[rows]
        isf = (ix >= seg[2]) & (ix < seg[3])
        mus = (ix - seg[2]).clamp(0, mot.nf - 1)
        ownw = isf & own[mot.muscle_of[mus]]
        oth = isf & ~own[mot.muscle_of[mus]]
        ratio.append(float(w[ownw].mean() / w[oth].mean().clamp_min(1e-9)) if ownw.any() and oth.any() else float('nan'))
    return dict(l4_units_with_learned_weight_median=float(np.median(learners)),
                top4_exact=float(np.mean(exact)) if exact else 0.0, margin_median=float(np.median(margin)) if margin else 0.0,
                true_other_weight_ratio=float(np.nanmedian(ratio)) if ratio else float('nan'), n_syllables=len(exact))


# ---------------------------------------------------------------------------------------------- context test (11-02)
def context_test(life, arms=('K0', 'K1', 'K1c', 'K2'), lags=(0, 6, 12, 24, 48), reps=4):
    """Does related context make the question last longer in the PFC (and the hippocampus)? See PLAN addendum."""
    b = life.brain
    props, _, _ = tasks(life)
    L = max(lags)
    out = {}
    for arm in arms:
        rng = np.random.default_rng(life.seed + 9900)                  # same draws in every arm
        S4, SH, key = [], [], []
        with Frozen(life):
            snap = b.snapshot()
            for f in range(4):
                for x in range(8):
                    for r in range(reps):
                        b.restore(snap)
                        v = C1 if r % 2 == 0 else C2
                        nar = W.narration(x, props, rng)
                        x2 = int((x + 1 + rng.integers(7)) % 8)            # a different item, same narration form
                        nar2 = ([W.FILL[x2], props[x2]['col'], '的'] if len(nar) == 3
                                else [W.FILL[x2], props[x2]['size']])
                        if arm == 'K0':
                            ctx = contexts(rng, props)[r % 4]
                        elif arm == 'K1':
                            ctx = line_script(nar, v) + [None] * 4
                        elif arm == 'K1c':
                            ctx = line_script(nar2, v) + [None] * 4
                        else:
                            ctx = [None] * 20
                        q = W.question(f, x)
                        if arm == 'K2':                                # item = filler + its colour, same frame
                            i = q.index(W.FILL[x])
                            q = q[:i + 1] + [props[x]['col']] + q[i + 1:]
                        qs = line_script(q, v)
                        t_end = len(ctx) + len(qs) - 1
                        s4, sh = {}, {}

                        def rec(lf, t):
                            if t - t_end in lags:
                                s4[t - t_end] = lf.brain.pfc.L4.on.clone()
                                if lf.brain.hip is not None:
                                    sh[t - t_end] = lf.brain.hip.Hs.on.clone()
                            if t - t_end in [lg + 1 for lg in lags]:
                                s4[('next', t - t_end - 1)] = lf.brain.pfc.L4.on.clone()
                        play(life, ctx + qs + [None] * (L + 2), mute=True, rec=rec)
                        S4.append(torch.stack([s4[lg] for lg in lags] + [s4[('next', lg)] for lg in lags]))
                        if sh:
                            SH.append(torch.stack([sh[lg] for lg in lags]))
                        key.append((f, x, r, 0))
        S4 = torch.stack(S4)
        SH = torch.stack(SH) if SH else None
        q_of = lambda k: (k[0], k[1])
        n = len(lags)
        res = dict(lags=list(lags), chance=1 / 32)
        res['L4_q32'] = [decode(S4, key, i, q_of) for i in range(n)]
        res['L4_filler'] = [float(np.mean([decode(S4, key, i, lambda k, f=f: k[1] if k[0] == f else None)
                                           for f in range(4)])) for i in range(n)]
        res['L4_motion_J'] = [float(jac(S4[:, i], S4[:, n + i]).mean().item()) for i in range(n)]
        if SH is not None:
            res['Hs_q32'] = [decode(SH, key, i, q_of) for i in range(n)]
            res['Hs_filler'] = [float(np.mean([decode(SH, key, i, lambda k, f=f: k[1] if k[0] == f else None)
                                               for f in range(4)])) for i in range(n)]
        for pop in ('L4', 'Hs'):
            if pop + '_q32' in res:
                below = [lg for lg, d in zip(lags, res[pop + '_q32']) if d <= 1 / 32 + 0.1]
                res[pop + '_remembered_until'] = below[0] if below else f'> {L}'
        out[arm] = res
    return out
