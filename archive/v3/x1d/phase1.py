"""X1d phase 1: at birth (learn=False), is the PFC top-layer (L3) trajectory during the pause after a question
input-locked and reproducible?  Same question in different preceding contexts vs different questions; motion;
damage spreading.  Uses X1c's world.py and core3.py unchanged (copied).

  python3 -I phase1.py <name> [key=value ...]          -> out1/<name>.json
"""
import sys, os, json, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'x1c'))   # repo: core3.py, world.py, run.py from ../x1c
from world import Physics, World, render, question, answer, narration, SID, OWN, C1, C2  # noqa
from core4 import Brain
from run import birth_rates, parse, run_until

NOLEARN = dict(eta=0.0, eta_minus=0.0, eta33=0.0, eta_v=0.0, eta_v_co=0.0, eta_r=0.0, fast=False, hip=False)

LAGS = 17                      # lag 0 = L3 at the last question tick ... lag 16
_RATES = {}


def rates_for(ph, seed):
    if seed not in _RATES:
        _RATES[seed] = birth_rates(ph, World(ph, seed))
    return _RATES[seed]


class Runner:
    """feeds a caregiver schedule tick by tick; own mouth output is heard one tick later (as in run.probe)."""

    def __init__(self, b, ph, mouth=True):
        self.b, self.ph, self.mouth, self.mprev = b, ph, mouth, None

    def snap(self):
        return (self.b.snapshot(), self.mprev)

    def restore(self, s):
        self.b.restore(s[0])
        self.mprev = s[1]

    def run(self, sched, rng, rec=False):
        out = []
        for cg in sched:
            srcs = []
            if self.mouth and self.mprev is not None:
                srcs.append((self.mprev[0], self.mprev[1], OWN, 1.0))
            if cg is not None:
                srcs.append(('s', cg[0], cg[1], 1.0))
            mus = self.b.step(render(self.ph, srcs, rng), 0.0, learn=False)
            self.mprev = self.ph.decode(mus)
            if rec:
                out.append((self.b.l3.copy(), int(self.b.l1.sum()), int(self.b.l2.sum()), int(self.b.pred.sum()),
                            self.mprev is not None))
        return out


def line(syls, voice):
    return [(SID[s], voice) for s in syls for _ in range(3)]


def sil(n):
    return [None] * n


def context(props, rng, pairs):
    """a world-like preceding stream: lead-in silence, 1-2 earlier exchanges (Q, pause, A), optional narration, gap."""
    s = sil(int(rng.integers(4, 30)))
    for _ in range(int(rng.integers(1, 3))):
        f, x = pairs[int(rng.integers(len(pairs)))]
        v = C1 if rng.random() < 0.7 else C2
        s += line(question(f, x), v) + sil(12) + line(answer(f, x, props), v) + sil(int(rng.integers(8, 17)))
        if rng.random() < 0.4:
            x = int(rng.integers(8))
            s += line(narration(x, props, rng), v) + sil(int(rng.integers(5, 10)))
    return s


def jac(a, b):
    u = (a | b).sum(-1)
    return np.where(u > 0, (a & b).sum(-1) / np.maximum(u, 1), 0.0)     # empty vs empty counts as 0 (no information)


def motion_stats(P):
    P = np.asarray(P, bool)
    def jj(a, b):
        u = (a | b).sum(1)
        return np.where(u > 0, (a & b).sum(1) / np.maximum(u, 1), 1.0)   # X1c's convention for motion
    return dict(n_active=round(float(P.sum(1).mean()), 1), p_empty=round(float((P.sum(1) == 0).mean()), 3),
                turnover1=round(float(1 - jj(P[:-1], P[1:]).mean()), 3),
                turnover5=round(float(1 - jj(P[:-5], P[5:]).mean()), 3),
                stuck20=round(float((jj(P[:-20], P[20:]) > 0.9).mean()), 3), distinct=int(len({p.tobytes() for p in P})),
                ticks=len(P))


def born(bk, seed, birth_ex=300, care='format'):
    """birth period: the newborn hears the world for birth_ex exchanges with every Hebbian / value rate at 0, so only
    the activity regulation (homeostatic offsets, running means) settles.  Returns (ph, world, brain)."""
    ph = Physics()
    w = World(ph, seed, care=care)
    kw = dict(bk)
    keep = {k: kw.get(k, None) for k in NOLEARN}
    kw.update(NOLEARN)
    b = Brain(ph, seed, rates=rates_for(ph, seed), **kw)
    if birth_ex:
        run_until(b, w, birth_ex)
    from core3 import DEF
    for k in NOLEARN:                                          # give the real rates back (phase 2 continues from here)
        b.c[k] = keep[k] if keep[k] is not None else DEF[k]
    return ph, w, b


def measure(bk, seed=0, n_ctx=6, mouth=True, damage=(0.1, 0.25), n_dmg_ctx=2, long_sil=300, n_long=3, verbose=True,
            birth_ex=300, brain=None):
    t0 = time.time()
    if brain is None:
        ph, w, b = born(bk, seed, birth_ex)
    else:
        ph, w, b = brain
    R = Runner(b, ph, mouth)
    allq = [(f, x) for f in range(4) for x in range(8)]
    nq = len(allq)
    crng = np.random.default_rng(seed + 500)
    # warm-up so that the start state is not the all-zero state
    R.run(context(w.props, crng, allq), crng)
    base = R.snap()
    TR = np.zeros((nq, n_ctx, LAGS, b.c['n3']), bool)        # L3 trajectories in the pause
    ONSET = np.zeros((nq, n_ctx, b.c['n3']), bool)
    QEND = np.zeros((nq, n_ctx, b.c['n3']), bool)
    N3 = np.zeros((nq, n_ctx, LAGS))
    N3q = []
    P1 = np.zeros((nq, n_ctx, LAGS))
    N2 = np.zeros((nq, n_ctx, LAGS))
    MO = np.zeros((nq, n_ctx, LAGS))
    talk_q = 0
    ctx_snaps = []
    for c in range(n_ctx):
        R.restore(base)
        R.run(context(w.props, crng, allq), crng)
        cs = R.snap()
        ctx_snaps.append(cs)
        for qi, (f, x) in enumerate(allq):
            R.restore(cs)
            rng = np.random.default_rng(seed * 100003 + c * 1009 + qi)
            ONSET[qi, c] = b.l3
            rq = R.run(line(question(f, x), C1), rng, rec=True)
            N3q.append([r[0].sum() for r in rq])
            talk_q += any(r[4] for r in rq)
            QEND[qi, c] = rq[-1][0]
            rp = R.run(sil(LAGS - 1), rng, rec=True)
            seq = [rq[-1]] + rp
            for k, r in enumerate(seq):
                TR[qi, c, k] = r[0]
                N3[qi, c, k] = r[0].sum()
                N2[qi, c, k] = r[2]
                P1[qi, c, k] = r[3]
                MO[qi, c, k] = r[4]
    # ---- (a) same question, different contexts; (b) different questions
    frame = np.array([f for f, x in allq])
    fill = np.array([x for f, x in allq])
    same, cross, cross_sf, cross_of, dec, dec_frame, dec_fill = [], [], [], [], [], [], []
    iu = [(i, j) for i in range(n_ctx) for j in range(n_ctx) if i != j]
    for k in range(LAGS):
        S = TR[:, :, k]                                       # (q, c, n)
        # similarity tensor J[q, c, q', c']
        Sf = S.reshape(nq * n_ctx, -1).astype(np.float32)
        inter = Sf @ Sf.T
        cnt = Sf.sum(1)
        uni = cnt[:, None] + cnt[None, :] - inter
        J = np.where(uni > 0, inter / np.maximum(uni, 1), 0.0).reshape(nq, n_ctx, nq, n_ctx)
        # mean over c != c' for each (q, q')
        Mqq = np.zeros((nq, nq))
        for (i, j) in iu:
            Mqq += J[:, i, :, j]
        Mqq /= len(iu)
        same.append(float(np.diag(Mqq).mean()))
        off = ~np.eye(nq, dtype=bool)
        cross.append(float(Mqq[off].mean()))
        sf = (frame[:, None] == frame[None, :]) & off
        of = frame[:, None] != frame[None, :]
        cross_sf.append(float(Mqq[sf].mean()))
        cross_of.append(float(Mqq[of].mean()))
        # decoding: each (q, c) -> question q' with the highest mean J to q' in the other contexts
        ok = okf = okx = 0
        for qi in range(nq):
            for c in range(n_ctx):
                others = [cc for cc in range(n_ctx) if cc != c]
                sc = J[qi, c][:, others].mean(1)              # (q',)
                if sc.max() <= 0:
                    continue
                best = np.flatnonzero(sc == sc.max())
                pq = best[np.random.default_rng(qi * 7 + c).integers(len(best))]
                ok += pq == qi
                okf += frame[pq] == frame[qi]
                scf = np.where(frame == frame[qi], sc, -1)    # filler given the frame
                bf = np.flatnonzero(scf == scf.max())
                okx += fill[bf[np.random.default_rng(qi * 11 + c).integers(len(bf))]] == fill[qi]
        tot = nq * n_ctx
        dec.append(ok / tot)
        dec_frame.append(okf / tot)
        dec_fill.append(okx / tot)
    # decoding on windows one loop long (3 ticks: L3 -> L1 -> L2 -> L3), states concatenated
    dec_win, same_win, cross_win = [], [], []
    for k in range(LAGS - 2):
        S = np.concatenate([TR[:, :, k], TR[:, :, k + 1], TR[:, :, k + 2]], axis=2)
        Sf = S.reshape(nq * n_ctx, -1).astype(np.float32)
        inter = Sf @ Sf.T
        cnt = Sf.sum(1)
        uni = cnt[:, None] + cnt[None, :] - inter
        J = np.where(uni > 0, inter / np.maximum(uni, 1), 0.0).reshape(nq, n_ctx, nq, n_ctx)
        Mqq = np.zeros((nq, nq))
        for (i, j) in iu:
            Mqq += J[:, i, :, j]
        Mqq /= len(iu)
        off = ~np.eye(nq, dtype=bool)
        same_win.append(float(np.diag(Mqq).mean()))
        cross_win.append(float(Mqq[off].mean()))
        ok = 0
        for qi in range(nq):
            for c in range(n_ctx):
                others = [cc for cc in range(n_ctx) if cc != c]
                sc = J[qi, c][:, others].mean(1)
                if sc.max() <= 0:
                    continue
                best = np.flatnonzero(sc == sc.max())
                ok += best[np.random.default_rng(qi * 7 + c).integers(len(best))] == qi
        dec_win.append(ok / (nq * n_ctx))
    # onset (before the question) similarity across contexts: how different the starting states are
    on_same = float(np.mean([jac(ONSET[q, i], ONSET[q, j]) for q in range(nq) for (i, j) in iu]))
    # within-trajectory motion in the pause
    within1 = [float(np.mean(jac(TR[:, :, k], TR[:, :, k + 1]))) for k in range(LAGS - 1)]
    # ---- (d) damage spreading at question end, identical input afterwards
    dmg = {}
    for p in damage:
        curves = []
        ctrl = []
        for c in range(min(n_dmg_ctx, n_ctx)):
            for qi, (f, x) in enumerate(allq):
                R.restore(ctx_snaps[c])
                rng = np.random.default_rng(seed * 100003 + c * 1009 + qi)
                R.run(line(question(f, x), C1), rng)
                s0 = R.snap()
                rs = rng.bit_generator.state
                ref = [b.l3.copy()] + [r[0] for r in R.run(sil(LAGS - 1), rng, rec=True)]
                R.restore(s0)
                a = np.flatnonzero(b.l3)
                ina = np.flatnonzero(~b.l3)
                m = int(round(p * len(a)))
                pr = np.random.default_rng(qi + 31 * c + int(p * 1000))
                if m and len(ina) >= m:
                    l3 = b.l3.copy()
                    l3[pr.choice(a, m, replace=False)] = False
                    l3[pr.choice(ina, m, replace=False)] = True
                    b.l3 = l3
                rng.bit_generator.state = rs
                per = [b.l3.copy()] + [r[0] for r in R.run(sil(LAGS - 1), rng, rec=True)]
                curves.append([float(jac(ref[k], per[k])) for k in range(LAGS)])
        dmg[str(p)] = [round(float(v), 3) for v in np.mean(curves, 0)]
    # ---- (e) noise only: same context snapshot, same question, fresh render noise (L1 and L3 Jaccard)
    nz3, nz1 = [], []
    for qi, (f, x) in enumerate(allq):
        reps = []
        for rep in range(3):
            R.restore(ctx_snaps[0])
            rng = np.random.default_rng(777 + 13 * qi + rep)
            rq = R.run(line(question(f, x), C1), rng, rec=True)
            l1q = b.l1.copy()
            rp = R.run(sil(LAGS - 1), rng, rec=True)
            reps.append(([rq[-1][0]] + [r[0] for r in rp], l1q))
        for i in range(3):
            for j in range(i + 1, 3):
                nz3.append([float(jac(reps[i][0][k], reps[j][0][k])) for k in range(LAGS)])
                nz1.append(float(jac(reps[i][1], reps[j][1])))
    # ---- long silence after a question: motion statistics
    P = []
    lrng = np.random.default_rng(seed + 900)
    for c in range(min(n_long, n_ctx)):
        R.restore(ctx_snaps[c])
        f, x = allq[int(lrng.integers(nq))]
        R.run(line(question(f, x), C1), lrng)
        P += [r[0] for r in R.run(sil(long_sil), lrng, rec=True)]
    mo = motion_stats(P)
    r3 = lambda v: [round(float(x), 3) for x in v]
    res = dict(cfg=bk, seed=seed, n_ctx=n_ctx, mouth=mouth,
               same=r3(same), cross=r3(cross), cross_same_frame=r3(cross_sf), cross_other_frame=r3(cross_of),
               dec_q=r3(dec), dec_q_win3=r3(dec_win), same_win3=r3(same_win), cross_win3=r3(cross_win), dec_frame=r3(dec_frame), dec_fill_given_frame=r3(dec_fill),
               onset_same=round(on_same, 3), within_J1=r3(within1),
               n3_lag=r3(N3.mean((0, 1))), n3_lag_min=r3(N3.min((0, 1))), n3_lag_max=r3(N3.max((0, 1))),
               p_empty_lag=r3((N3 == 0).mean((0, 1))), n2_lag=r3(N2.mean((0, 1))), pred_lag=r3(P1.mean((0, 1))),
               mouth_lag=r3(MO.mean((0, 1))), n3_during_q=r3(np.mean(N3q, 0)), talk_during_q=round(talk_q / (nq * n_ctx), 3),
               damage=dmg, motion_long_silence=mo, noise_same=r3(np.mean(nz3, 0)), noise_L1_qend=round(float(np.mean(nz1)), 3), time_s=round(time.time() - t0, 1))
    if verbose:
        pr_curves(res)
    return res


def pr_curves(r):
    def fm(v, ks=(0, 1, 2, 3, 4, 6, 8, 10, 12, 14, 16)):
        return ' '.join(f'{v[k]:.2f}' for k in ks if k < len(v))
    print(f"cfg={r['cfg']} seed={r['seed']}  ({r['time_s']}s)")
    print('  lag            ', ' '.join(f'{k:>4d}' for k in (0, 1, 2, 3, 4, 6, 8, 10, 12, 14, 16)))
    for key in ('noise_same', 'same', 'cross', 'cross_same_frame', 'cross_other_frame', 'dec_q', 'dec_frame', 'dec_fill_given_frame',
                'n3_lag', 'p_empty_lag', 'n2_lag', 'pred_lag', 'mouth_lag'):
        print(f'  {key:<16}', fm(r[key]))
    print('  within_J1       ', fm(r['within_J1'] + [np.nan]))
    for p, v in r['damage'].items():
        print(f'  damage {p:<9}', fm(v))
    print('  noise_L1_qend %.2f' % r['noise_L1_qend'])
    print('  onset_same %.2f  talk_during_q %.2f  n3 during Q %s' % (r['onset_same'], r['talk_during_q'], r['n3_during_q']))
    print('  long silence', r['motion_long_silence'], flush=True)


if __name__ == '__main__':
    name = sys.argv[1]
    kv = {k: parse(v) for k, v in (a.split('=', 1) for a in sys.argv[2:])}
    seed = kv.pop('seed', 0)
    n_ctx = kv.pop('n_ctx', 6)
    mouth = kv.pop('mouth', True)
    bex = kv.pop('birth_ex', 300)
    res = measure(kv, seed=seed, n_ctx=n_ctx, mouth=mouth, birth_ex=bex)
    res['birth_ex'] = bex
    os.makedirs(os.path.join(HERE, 'out1'), exist_ok=True)
    json.dump(res, open(os.path.join(HERE, 'out1', f'{name}_s{seed}.json'), 'w'), indent=1, default=str)
