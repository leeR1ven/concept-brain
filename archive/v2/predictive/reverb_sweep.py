# -*- coding: utf-8 -*-
"""Parameter map of the reverberating PFC hold (the designer's loop), between M6 (collapse) and M6b (no hold).

Area: N 2048 binary units, k-WTA, leaky potential v(t) = lam v(t-1) + input, learned recurrence by the ONE rule (PR, state
target = co-activity), fan-in 1000.  Inputs: the realistic bench line codes (147 lines; union of 8-unit unigram + bigram
codes, frame chars shared), presented with a Zipf frequency (rank r with p ~ 1/(r+1)) so experience can shape basins.
Load (the BG gate at another speaker's line end) = partial reset of v + the line's input for 2 ticks; hold = no input,
weak random distractors (0.3 on k random units per tick).
Knobs: beta/tau_sp (slow homeostatic set point), gamma/tau_a (fatigue: long-on units tire), c_rec (recurrent gain),
amp/reset (strength of a new line, partial reset), idf (online normalisation of frequent input features), k (sparsity),
learn_first (learn on the load's first tick, whose source is the OLD thought), hold_learn (learning while holding).
Measures: (a) 20-way decode of the held line at 10/30/60/120 ticks; (b) collapse = mean Jaccard of held states of
different lines (vs their inputs); (c) drift J(s60, s500) and decode at 200/500; (d) favourites: frequency-rank
percentile of the nearest line after 500 silent ticks (0 = most frequent; uniform = 0.5) and how often it is a top-10 %
line; (e) replacement: extra overlap with the old thought after a new load + decode of the new line; (f) period-2 index.
Usage: python3 -B reverb_sweep.py BATCH   (prints one row per setting)"""
import sys
import time

import numpy as np

sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__)))
from pr import Recurrent, F32  # noqa: E402
from test_modules import bench_lines, line_code, jac  # noqa: E402

N = 2048
LINES = bench_lines()
CODES = [line_code(l, N) for l in LINES]
L = len(LINES)


class LoopPFC:
    def __init__(self, k=120, fanin=1000, eta=0.5, lam=0.9, c_rec=1.0, beta=0.0, tau_sp=20000.0, gamma=0.0, tau_a=200.0,
                 idf=0.0, tau_f=3000.0, seed=5, k_adapt=0.0, **_):
        self.R = Recurrent(N, k, fanin, eta, theta=1.0, lam_v=lam, c_rec=c_rec, seed=seed, sp_beta=beta, sp_tau=tau_sp)
        self.k = k
        self.k_adapt = k_adapt      # > 0: active count follows input strength: k = k_adapt x (number of driven units)
        self.gamma, self.tau_a = F32(gamma), F32(tau_a)
        self.a = np.zeros(N, F32)
        self.idf, self.tau_f = idf, tau_f
        self.f = np.full(N, 1.0 / 35, F32)            # online input-frequency trace of every input line (feature)

    def step(self, drive, learn=False):
        s = self.R.step(drive - self.gamma * self.a, learn=learn)
        if self.gamma > 0:
            self.a += (s.astype(F32) - self.a) / self.tau_a
        return s

    def load(self, code, amp=3.0, reset=1.0, learn_first=True, learn=True, track=True):
        w = np.ones(N, F32)
        if self.idf > 0:
            w = (float(self.f[code].mean()) / np.maximum(self.f, 1e-4)) ** self.idf
            w = np.clip(w, 0.1, 3.0).astype(F32)
        drive = (amp * code * w).astype(F32)
        if self.idf > 0 and track:
            self.f += (code.astype(F32) - self.f) / self.tau_f
        self.R.A.v *= F32(1.0 - reset)
        if self.k_adapt > 0:                           # the designer's "strong signals activate more neurons"
            self.R.A.k = int(max(8, round(self.k_adapt * float(code.sum()))))
        self.step(drive, learn=learn and learn_first)
        return self.step(drive, learn=learn)

    def hold(self, T, rng, learn=False, dis=0.3, record=None):
        out = []
        for t in range(T):
            d = np.zeros(N, F32); d[rng.choice(N, 120, replace=False)] = dis
            s = self.step(d, learn=learn)
            if record is None or (t + 1) in record:
                out.append(s.copy())
        return out


def score(s, c):
    return jac(s, CODES[c])          # Jaccard: a short line nested inside a longer one does not tie


def run(P, seed=0, n_train=1500, hold_train=30):
    rng = np.random.default_rng(seed)
    order = rng.permutation(L)                       # frequency rank of each line (rank 0 = most frequent)
    rank = np.empty(L, int); rank[order] = np.arange(L)
    pz = 1.0 / (np.arange(L) + 1.0); pz /= pz.sum()
    prob = pz[rank]
    m = LoopPFC(**P)
    ld = dict(amp=P.get('amp', 3.0), reset=P.get('reset', 1.0), learn_first=P.get('learn_first', True))
    t0 = time.time()
    for i in range(n_train):
        c = int(rng.choice(L, p=prob))
        m.load(CODES[c], **ld)
        m.hold(hold_train, rng, learn=P.get('hold_learn', False), record=set())
    # (a)(b)(f) 30 trials x 120 ticks, uniform lines
    T_dec = (10, 30, 60, 120)
    dec = {T: [] for T in T_dec}; states = {10: [], 60: []}; p2 = []; lines_used = []
    for tr in range(30):
        cand = rng.choice(L, 20, replace=False)
        m.load(CODES[cand[0]], learn=False, track=False, **{k: v for k, v in ld.items() if k != 'learn_first'})
        traj = m.hold(120, rng)
        for T in T_dec:
            sc = [score(traj[T - 1], c) for c in cand]
            dec[T].append(int(np.argmax(sc)) == 0)
        states[10].append(traj[9]); states[60].append(traj[59]); lines_used.append(cand[0])
        p2.append(np.mean([jac(traj[t], traj[t + 2]) - jac(traj[t], traj[t + 1]) for t in range(10, 117)]))
    def pairJ(S):
        return float(np.mean([jac(S[i], S[j]) for i in range(len(S)) for j in range(i + 1, len(S))]))
    gin = pairJ([CODES[c] for c in lines_used])
    # (c)(d) long silence: 12 trials x 500 ticks
    d200, d500, drift, fav_pct, fav_top, stay = [], [], [], [], [], []
    for tr in range(12):
        cand = rng.choice(L, 20, replace=False)
        m.load(CODES[cand[0]], learn=False, track=False, **{k: v for k, v in ld.items() if k != 'learn_first'})
        traj = m.hold(500, rng, record={60, 200, 500})
        s60, s200, s500 = traj
        d200.append(int(np.argmax([score(s200, c) for c in cand]) == 0))
        d500.append(int(np.argmax([score(s500, c) for c in cand]) == 0))
        drift.append(jac(s60, s500))
        near = int(np.argmax([score(s500, c) for c in range(L)]))
        stay.append(near == cand[0])
        fav_pct.append(rank[near] / (L - 1)); fav_top.append(rank[near] < L // 10)
    # (e) replacement
    extra, decB = [], []
    for tr in range(20):
        cand = rng.choice(L, 20, replace=False); A, B = cand[0], cand[1]
        m.load(CODES[A], learn=False, track=False, **{k: v for k, v in ld.items() if k != 'learn_first'})
        sA = m.hold(30, rng)[-1]
        m.load(CODES[B], learn=False, track=False, **{k: v for k, v in ld.items() if k != 'learn_first'})
        sB = m.hold(10, rng)[-1]
        extra.append(jac(sB, sA) - jac(CODES[B], CODES[A]))
        decB.append(int(np.argmax([score(sB, c) for c in cand]) == 1))
    return dict(dec={T: float(np.mean(v)) for T, v in dec.items()}, gen10=pairJ(states[10]), gen60=pairJ(states[60]),
                gin=gin, p2=float(np.mean(p2)), d200=float(np.mean(d200)), d500=float(np.mean(d500)),
                drift=float(np.mean(drift)), fav_pct=float(np.mean(fav_pct)), fav_top=float(np.mean(fav_top)),
                stay=float(np.mean(stay)), extra=float(np.mean(extra)), decB=float(np.mean(decB)), secs=time.time() - t0)


def row(name, r):
    d = r['dec']
    return (f"{name:34s} | dec10/30/60/120 {d[10]:.2f} {d[30]:.2f} {d[60]:.2f} {d[120]:.2f} | collapse J10 {r['gen10']:.2f} "
            f"J60 {r['gen60']:.2f} (inputs {r['gin']:.2f}) | silence: dec200 {r['d200']:.2f} dec500 {r['d500']:.2f} "
            f"J(60,500) {r['drift']:.2f} stay {r['stay']:.2f} fav-pct {r['fav_pct']:.2f} top10% {r['fav_top']:.2f} | "
            f"replace extra {r['extra']:+.2f} decB {r['decB']:.2f} | p2 {r['p2']:+.3f} | {r['secs']:.0f}s")


BASE = dict(k=120, lam=0.9, c_rec=1.0, beta=0.0, gamma=0.0, idf=0.0, amp=3.0, reset=1.0, learn_first=True, hold_learn=False)
NO = dict(learn_first=False)
BATCHES = {
    '2': [('k120', dict(NO)), ('k60', dict(NO, k=60)), ('k40', dict(NO, k=40)), ('k80', dict(NO, k=80)),
          ('k60 hold_learn', dict(NO, k=60, hold_learn=True)), ('k60 set point 0.02', dict(NO, k=60, beta=0.02)),
          ('k60 fatigue 0.1/300', dict(NO, k=60, gamma=0.1, tau_a=300.0)),
          ('k60 fatigue 0.2/300', dict(NO, k=60, gamma=0.2, tau_a=300.0)), ('k60 amp 6', dict(NO, k=60, amp=6.0)),
          ('k60 reset 0.7', dict(NO, k=60, reset=0.7)), ('k40 idf 0.5', dict(NO, k=40, idf=0.5)),
          ('k120 set point 0.05', dict(NO, beta=0.05)), ('k120 fatigue 0.1/300', dict(NO, gamma=0.1, tau_a=300.0)),
          ('k80 hold_learn fatigue 0.1/300', dict(NO, k=80, hold_learn=True, gamma=0.1, tau_a=300.0))],
    '1': [('M6 replica', {}), ('no learning at load onset', dict(learn_first=False)), ('k 60', dict(k=60)),
          ('k 60 + no onset learning', dict(k=60, learn_first=False)), ('slow set point b0.05', dict(beta=0.05)),
          ('slow set point b0.2', dict(beta=0.2)), ('fatigue g0.3 tau200', dict(gamma=0.3)),
          ('c_rec 0.5', dict(c_rec=0.5)), ('c_rec 2.0', dict(c_rec=2.0)), ('idf 0.5', dict(idf=0.5)),
          ('amp 6', dict(amp=6.0)), ('partial reset 0.5', dict(reset=0.5))],
}


def main():
    b = sys.argv[1] if len(sys.argv) > 1 else '1'
    if b in CBATCHES:
        for name, over in CBATCHES[b]:
            P = dict(CB); P.update(over)
            print(row_cont(name, run_cont(P)), flush=True)
        return
    settings = BATCHES.get(b)
    if settings is None:                                    # ad-hoc: key=val,key=val;...
        settings = []
        for spec in b.split(';'):
            kv = dict(x.split('=') for x in spec.split(',') if x)
            P = {k: (v == 'True' if v in ('True', 'False') else float(v) if '.' in v else int(v)) for k, v in kv.items()}
            settings.append((spec, P))
    for name, over in settings:
        P = dict(BASE); P.update(over)
        print(row(name, run(P)), flush=True)


# ===================================================================================== continuous update (no reset)
def ucode(ch, seed=0, m=8):
    return np.random.default_rng([seed, 1, ord(ch)]).choice(N, m, replace=False)


def bcode(a, b, seed=0, m=8):
    return np.random.default_rng([seed, 2, ord(a), ord(b)]).choice(N, m, replace=False)


def hear_line(m, c, P, rng, learn=False):
    """the line flows into the ongoing thought, char by char (3 ticks per char); input = codes of the chars heard so
    far (the line layer), gain = amp * ((1 - nov) + nov * novelty), novelty = share of the input not in the thought."""
    g = np.zeros(N, bool); prev = None
    for ch in LINES[c]:
        g[ucode(ch)] = True
        if prev is not None:
            g[bcode(prev, ch)] = True
        prev = ch
        w = np.ones(N, F32)
        if m.idf > 0:
            w = np.clip((float(m.f[g].mean()) / np.maximum(m.f, 1e-4)) ** m.idf, 0.1, 3.0).astype(F32)
        for k in range(3):
            novelty = 1.0 - (g & m.R.A.s).sum() / max(g.sum(), 1)
            gain = P['amp'] * ((1.0 - P['nov']) + P['nov'] * novelty)
            m.step((gain * g * w).astype(F32), learn=learn)
    if m.idf > 0:
        m.f += (g.astype(F32) - m.f) / m.tau_f
    if m.k_adapt > 0:
        m.R.A.k = int(max(8, round(m.k_adapt * float(g.sum()))))


def run_cont(P, seed=0, n_train=1500):
    rng = np.random.default_rng(seed)
    order = rng.permutation(L); rank = np.empty(L, int); rank[order] = np.arange(L)
    pz = 1.0 / (np.arange(L) + 1.0); pz /= pz.sum(); prob = pz[rank]
    m = LoopPFC(**P)
    t0 = time.time()
    for i in range(n_train):
        c = int(rng.choice(L, p=prob))
        hear_line(m, c, P, rng, learn=P.get('learn_in', False))
        m.hold(int(rng.integers(5, 26)), rng, learn=P.get('hold_learn', True), record=set())
    # takeover and context: line A, gap 20, line B; decode B (A among the candidates) and A (B excluded)
    tk = {3: [], 6: [], 10: []}; ctx = {3: [], 6: [], 10: []}
    for tr in range(30):
        cand = rng.choice(L, 21, replace=False); A, B = cand[0], cand[1]
        hear_line(m, A, P, rng); m.hold(20, rng)
        hear_line(m, B, P, rng)
        traj = m.hold(10, rng)
        for T in (3, 6, 10):
            s = traj[T - 1]
            tk[T].append(int(np.argmax([score(s, c) for c in cand[:20]]) == 1))
            others = [A] + list(cand[2:21])
            ctx[T].append(int(np.argmax([score(s, c) for c in others]) == 0))
    # hold after a line: decode 30/60/120, collapse, period-2; long silence: 500, favourites, drift
    dec = {30: [], 60: [], 120: []}; states = []; p2 = []; used = []
    for tr in range(30):
        cand = rng.choice(L, 20, replace=False)
        hear_line(m, cand[0], P, rng)
        traj = m.hold(120, rng)
        for T in dec:
            dec[T].append(int(np.argmax([score(traj[T - 1], c) for c in cand]) == 0))
        states.append(traj[59]); used.append(cand[0])
        p2.append(np.mean([jac(traj[t], traj[t + 2]) - jac(traj[t], traj[t + 1]) for t in range(10, 117)]))
    def pairJ(S):
        return float(np.mean([jac(S[i], S[j]) for i in range(len(S)) for j in range(i + 1, len(S))]))
    d500, drift, fav_top, fav_pct = [], [], [], []
    for tr in range(12):
        cand = rng.choice(L, 20, replace=False)
        hear_line(m, cand[0], P, rng)
        s60, s500 = m.hold(500, rng, record={60, 500})
        d500.append(int(np.argmax([score(s500, c) for c in cand]) == 0)); drift.append(jac(s60, s500))
        near = int(np.argmax([score(s500, c) for c in range(L)]))
        fav_top.append(rank[near] < L // 10); fav_pct.append(rank[near] / (L - 1))
    return dict(tk={T: float(np.mean(v)) for T, v in tk.items()}, ctx={T: float(np.mean(v)) for T, v in ctx.items()},
                dec={T: float(np.mean(v)) for T, v in dec.items()}, gen60=pairJ(states),
                gin=pairJ([CODES[c] for c in used]), p2=float(np.mean(p2)), d500=float(np.mean(d500)),
                drift=float(np.mean(drift)), fav_top=float(np.mean(fav_top)), fav_pct=float(np.mean(fav_pct)),
                secs=time.time() - t0)


def row_cont(name, r):
    return (f"{name:36s} | takeover B@3/6/10 {r['tk'][3]:.2f} {r['tk'][6]:.2f} {r['tk'][10]:.2f} | context A@3/6/10 "
            f"{r['ctx'][3]:.2f} {r['ctx'][6]:.2f} {r['ctx'][10]:.2f} | hold dec30/60/120 {r['dec'][30]:.2f} "
            f"{r['dec'][60]:.2f} {r['dec'][120]:.2f} | collapse J60 {r['gen60']:.2f} (in {r['gin']:.2f}) | dec500 "
            f"{r['d500']:.2f} J(60,500) {r['drift']:.2f} fav top10% {r['fav_top']:.2f} pct {r['fav_pct']:.2f} | "
            f"p2 {r['p2']:+.3f} | {r['secs']:.0f}s")


CB = dict(k=60, lam=0.9, c_rec=1.0, beta=0.0, gamma=0.0, tau_a=100.0, idf=0.0, amp=2.0, nov=0.0, learn_in=False,
          hold_learn=True)
CBATCHES = {
    'c1': [('k60 amp2', {}), ('k60 amp1', dict(amp=1.0)), ('k60 amp3', dict(amp=3.0)),
           ('k60 amp2 novelty1', dict(nov=1.0)), ('k60 amp3 novelty1', dict(amp=3.0, nov=1.0)),
           ('k60 amp2 fatigue 0.2/100', dict(gamma=0.2)), ('k60 amp2 fatigue 0.4/100', dict(gamma=0.4)),
           ('k60 amp2 nov1 fatigue 0.2/100', dict(nov=1.0, gamma=0.2)), ('k40 amp2 idf .5', dict(k=40, idf=0.5)),
           ('k40 amp2 idf .5 nov1 fat .2', dict(k=40, idf=0.5, nov=1.0, gamma=0.2)), ('k80 amp2', dict(k=80)),
           ('k60 amp2 learn while hearing', dict(learn_in=True)), ('k60 amp2 lam .8', dict(lam=0.8))],
    'c2': [('k60 amp4', dict(amp=4.0)), ('k60 amp6', dict(amp=6.0)), ('k60 amp3 lam.8', dict(amp=3.0, lam=0.8)),
           ('k60 amp4 lam.8', dict(amp=4.0, lam=0.8)), ('k60 amp6 lam.8', dict(amp=6.0, lam=0.8)),
           ('k60 amp4 fatigue .2/50', dict(amp=4.0, gamma=0.2, tau_a=50.0)),
           ('k60 amp4 lam.8 fatigue .1/100', dict(amp=4.0, lam=0.8, gamma=0.1)),
           ('k40 idf.5 amp4', dict(k=40, idf=0.5, amp=4.0)), ('k40 idf.5 amp4 lam.8', dict(k=40, idf=0.5, amp=4.0, lam=0.8)),
           ('k60 amp4 learn in, no hold learn', dict(amp=4.0, learn_in=True, hold_learn=False)),
           ('k60 amp4 learn in + hold', dict(amp=4.0, learn_in=True)), ('k80 amp4 lam.8', dict(k=80, amp=4.0, lam=0.8))],
}

if __name__ == '__main__':
    main()
