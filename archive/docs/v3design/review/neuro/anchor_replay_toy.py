"""Neuro review toy 2 (local, < 100 MB, ~1-3 min). Extends the synthesis drift_toy (same world, learner and ST
'latest x age' context; logic copied, not imported) with two things ARCH_v3_core leaves unspecified:

(A) Awake replay while mu is low (>= 12 silent ticks, n_rep = 8 per quiet period, uniform over a 10 k store of heard
    events, normal learning rate, M3 / B2). Two readings of "its context drives MIX, while mu gates the live
    pathways down":
      stored : the replayed episode's own stored anchored features are the context (LATEST's live output is gated off)
      live   : every MIX unit keeps its anchor branch on the LIVE LATEST x TIME, so the replayed successor is taught
               in the live context (held question x current age)
(C) Independent per-event jitter of the format wait (10) and the give-up delay (45), x[0.5, 2] and x[0.75, 1.33].
(B) Distractor onsets: an eye / other event (6 types) at rate r per tick loads LATEST (M1: "at the sending pathway's
    next event onset, a persistent area loads the new pattern"), so the held question moves to the older slot.
Schedule: format 600 -> untutored 3000, p_sil 0.3 (drift_toy's main case). 5 seeds. Accuracy = frozen probe as in
drift_toy, with and without distractors in the probe window.
Run: python3 -I anchor_replay_toy.py
"""
import numpy as np, itertools, json

NF, NX, ND = 4, 8, 6
ANS = [4, 2, 3, 3]
ETA, THETA = 0.35, 0.3


def bins(tau):
    return [k for k in range(0, 9) if 2 ** k <= tau < 2 ** (k + 2)]


class Agent:
    def __init__(self, toks):
        self.ix = {t: i for i, t in enumerate(toks)}; self.toks = toks; self.V = len(toks)
        self.W = {}; self.L = None; self.O = None; self.store = []

    def feats(self, now):
        if self.L is None:
            return []
        src, g, toff = self.L; g = sorted(g)
        og = sorted(self.O[1]) if self.O is not None else []
        osrc = self.O[0] if self.O is not None else None
        B = bins(max(1, now - toff)); F = []
        F += [('Lt', src, a, k) for a in g for k in B]
        F += [('LLt', src, a, b, k) for a, b in itertools.combinations(g, 2) for k in B]
        F += [('LOt', src, a, k, osrc, o) for a in g for k in B for o in og]
        return F

    def pred(self, F):
        d = np.zeros(self.V)
        for f in F:
            w = self.W.get(f)
            if w is not None:
                d += w
        return np.clip(d, 0, 1)

    def learn(self, F, items):
        if not F:
            return
        p = self.pred(F); t = np.zeros(self.V)
        for it in items:
            t[self.ix[it]] = 1
        dw = ETA * (t - p) / len(F)
        for f in F:
            w = self.W.get(f)
            if w is None:
                w = self.W[f] = np.zeros(self.V)
            w += dw

    def hear(self, items, now, dur, learn=True):
        if learn:
            F = self.feats(now); self.learn(F, items)
            if F:
                self.store.append((F, list(items))); self.store = self.store[-10000:]
        self.O, self.L = self.L, ('o', list(items), now + dur)

    def say(self, items, now, dur):
        self.O, self.L = self.L, ('s', list(items), now + dur)

    def distract(self, d, now):
        self.O, self.L = self.L, ('e', [f'e{d}'], now + 1)


def run(seed, replay='none', r_dis=0.0, p_sil=0.3, n_fmt=600, n_free=3000, n_rep=8, tau_f=10, jit=None):
    rng = np.random.default_rng(seed)
    toks = [f'f{i}' for i in range(NF)] + [f'x{i}' for i in range(NX)] + ['ok'] + [f'e{d}' for d in range(ND)]
    at = [[f'a{f}_{k}' for k in range(ANS[f])] for f in range(NF)]
    toks += [t for r in at for t in r]
    table = {(f, x): at[f][rng.integers(ANS[f])] for f in range(NF) for x in range(NX)}
    ag = Agent(toks); now = 0; out = {}

    def dis_times(t0, t1):
        n = rng.poisson(r_dis * (t1 - t0)) if r_dis > 0 else 0
        return sorted((int(t0 + rng.integers(1, max(2, t1 - t0))), int(rng.integers(ND))) for _ in range(n))

    def decide(qend, dlist, explore):
        """scan tau 1..44; apply distractors as they occur; return tau, token"""
        di = 0
        for tau in range(1, 45):
            while di < len(dlist) and dlist[di][0] <= qend + tau:
                ag.distract(dlist[di][1], dlist[di][0]); di += 1
            p = ag.pred(ag.feats(qend + tau))
            if p.max() >= THETA:
                j = int(np.argmax(p + 1e-9 * rng.random(ag.V)))
                if rng.random() < explore:
                    j = int(rng.integers(ag.V))
                return tau, ag.toks[j], di
        return None, None, di

    def quiet(t0, t1):
        """replay during a quiet period [t0, t1) once mu < 0.05 (12 silent ticks)"""
        if replay == 'none' or t1 - t0 < 13 or not ag.store:
            return
        for j in range(n_rep):
            tr = t0 + 12 + int(j * (t1 - t0 - 12) / n_rep)
            F_s, items = ag.store[int(rng.integers(len(ag.store)))]
            F = F_s if replay == 'stored' else ag.feats(tr)
            ag.learn(F, items)

    def frozen(with_dis):
        st = (ag.L, ag.O); acc = 0; n = 0
        for f in range(NF):
            for x in range(NX):
                for rep in range(3 if with_dis else 1):
                    ag.L, ag.O = st
                    ag.hear([f'f{f}', f'x{x}'], 10 ** 7, 6, learn=False)
                    dl = dis_times(10 ** 7 + 6, 10 ** 7 + 50) if with_dis else []
                    tau, tok, _ = decide(10 ** 7 + 6, dl, 0.0)
                    acc += (tok == table[(f, x)]); n += 1
        ag.L, ag.O = st
        return round(acc / n, 3)

    for phase, n in (('format', n_fmt), ('free', n_free)):
        for i in range(1, n + 1):
            if phase == 'free' and (i - 1) in (300, 900, 2100):
                out[f'free{i - 1}'] = frozen(False)
            f, x = int(rng.integers(NF)), int(rng.integers(NX)); Q = [f'f{f}', f'x{x}']; a = table[(f, x)]
            now += 15
            ag.hear(Q, now, 6); qend = now + 6
            dl = dis_times(qend, qend + 60)
            tau, tok, di = decide(qend, dl, 0.05 if phase != 'free' else 0.0)
            silent = tau is None or rng.random() < p_sil
            tf = int(round(tau_f * rng.uniform(*jit))) if jit else tau_f      # independent per-event jitter
            gu = int(round(45 * rng.uniform(*jit))) if jit else 45
            if phase == 'format' and (silent or tau >= tf):
                # apply distractors up to the format answer, then learn the answer in the current context
                for t_d, d in dl[di:]:
                    if t_d <= qend + tf:
                        ag.distract(d, t_d)
                ag.hear([a], qend + tf, 3); now = qend + tf + 3
                quiet(now, now + 15)
                continue
            if silent:
                for t_d, d in dl[di:]:
                    if t_d <= qend + 45:
                        ag.distract(d, t_d)
                if phase == 'format':
                    ag.hear([a], qend + 45, 3); now = qend + 48
                else:
                    quiet(qend, qend + gu)          # wait for the give-up: replay once mu is low
                    now = qend + gu - 15
                continue
            ag.say([tok], qend + tau, 3); stop = qend + tau + 3
            if phase == 'format':
                t_r = stop + 6
                ag.hear(['ok', a] if tok == a else [a], t_r, 6); now = t_r + 6
                quiet(now, now + 15)
            else:
                t_next = max(qend + gu, stop + 13)
                quiet(stop, t_next)
                now = t_next - 15
        out[f'{phase}{n}'] = frozen(False)
        out[f'{phase}{n}_dis'] = frozen(True)
    return out


if __name__ == '__main__':
    res = {}
    for name, kw in (('ST, no replay, no distractors', dict()),
                     ('ST, replay=stored', dict(replay='stored')),
                     ('ST, replay=live (anchor stays on live LATEST)', dict(replay='live')),
                     ('ST, distractors r=0.005/tick', dict(r_dis=0.005)),
                     ('ST, distractors r=0.02/tick (crowd-like)', dict(r_dis=0.02)),
                     ('ST, replay=stored + distractors r=0.02', dict(replay='stored', r_dis=0.02)),
                     ('ST, independent per-event jitter x[0.5,2] on format wait and give-up', dict(jit=(0.5, 2.0))),
                     ('ST, independent per-event jitter x[0.75,1.33]', dict(jit=(0.75, 1.33)))):
        rows = [run(s, **kw) for s in range(5)]
        agg = {k: (round(float(np.mean([r[k] for r in rows])), 3), round(float(np.std([r[k] for r in rows])), 3)) for k in rows[0]}
        res[name] = agg
        print('==', name)
        print('   ' + '  '.join(f'{k}: {m:.2f}±{s:.2f}' for k, (m, s) in agg.items()), flush=True)
    json.dump(res, open('anchor_replay_toy.json', 'w'), indent=1)
