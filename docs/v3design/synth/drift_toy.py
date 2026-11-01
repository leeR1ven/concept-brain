"""Synthesis check (local, seconds, <100 MB): does a context anchored on (latest event x its age) keep answers stable
in long untutored use when the agent is sometimes SILENT, i.e. when the next question ("give-up") arrives at the same
delay as a told answer?  This is the case the neuro/develop toys did not cover (their agent always answered) and the
v2.2 collapse after 2,100-3,000 untutored dialogues suggests (agent silent 0.20-0.34; partner lines learned).

Symbolic world (as neuro/role_toy.py): question Q = [frame f, filler x] by the partner; answer = one token ans(f, x).
Timing is bench-like (PROVED from world.py by the v2.2 report): a recast / a told answer after my silence starts at
max(Q_end + 45, my_stop + 6); the next probe at max(Q_end + 45, my_stop + 13).  New world option (W1, formats):
the caregiver answers herself at tau_f (= 10) unless I started before.

Learner: one signed delta rule (bench R2): p = clip(sum w, 0, 1); dw = eta (t - p)/|F|, eta 0.35, over the context
features F at the heard line's onset.  Own lines are never targets (efference-predicted) but enter LATEST as source s.
Speaking: scan tau = 1..44 after Q's end; answer the argmax at the first tau whose max p >= theta (accumulate to bound).

Context variants (the only difference):
  V2   ARCH_v2: context = last OTHER line (own speech excluded), no time.
  S    self in the stream; features anchored on LATEST (singles, LxL, LxOLDER), source-coded; no time.
  ST0  S + unanchored time features (time-bin singles and LATEST x bin); LATEST-only features still present.
  ST   anchored on (LATEST item x age bin): every feature carries the latest event's age (soft log bins, 2 octaves).
Run: python3 -I drift_toy.py [out.json]
"""
import numpy as np, itertools, json, sys

NF, NX = 4, 8
ANS = [4, 2, 3, 3]
ETA, THETA = 0.35, 0.3


def bins(tau):
    """soft log time bins: bin k covers [2^k, 2^(k+2)); every tau >= 1 is in 1 or 2 bins"""
    return [k for k in range(0, 9) if 2 ** k <= tau < 2 ** (k + 2)]


class Agent:
    def __init__(self, toks, variant):
        self.ix = {t: i for i, t in enumerate(toks)}; self.toks = toks; self.V = len(toks); self.v = variant
        self.W = {}
        self.L = None; self.O = None          # (src, items, t_off) latest / older event
        self.last_other = None

    def feats(self, now):
        v = self.v
        if v == 'V2':
            g = sorted(self.last_other or [])
            return [('s', a) for a in g] + [('p', a, b) for a, b in itertools.combinations(g, 2)]
        if self.L is None:
            return []
        src, g, toff = self.L; g = sorted(g)
        og = sorted(self.O[1]) if self.O is not None else []
        osrc = self.O[0] if self.O is not None else None
        F = []
        if v in ('S', 'ST0'):
            F += [('L', src, a) for a in g] + [('LL', src, a, b) for a, b in itertools.combinations(g, 2)]
            F += [('LO', src, a, osrc, o) for a in g for o in og]
        if v in ('ST0', 'ST'):
            B = bins(max(1, now - toff))
            F += [('Lt', src, a, k) for a in g for k in B]
            if v == 'ST':
                F += [('LLt', src, a, b, k) for a, b in itertools.combinations(g, 2) for k in B]
                F += [('LOt', src, a, k, osrc, o) for a in g for k in B for o in og]
            else:
                F += [('T', k) for k in B]
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
            self.learn(self.feats(now), items)
        self.O, self.L = self.L, ('o', list(items), now + dur)
        self.last_other = list(items)

    def say(self, items, now, dur):
        self.O, self.L = self.L, ('s', list(items), now + dur)

    def decide(self, now0, rng=None, explore=0.0):
        """scan tau = 1..44 after the heard line's end; return (tau, token) of the first confident prediction"""
        for tau in range(1, 45):
            p = self.pred(self.feats(now0 + tau))
            if p.max() >= THETA:
                j = int(np.argmax(p + 1e-9 * (rng.random(self.V) if rng is not None else 0)))
                if rng is not None and rng.random() < explore:
                    j = int(rng.integers(self.V))
                return tau, self.toks[j]
        return None, None


def run(variant, seed, schedule, p_sil=0.0, tau_f=10, jitter=0.0):
    rng = np.random.default_rng(seed)
    toks = [f'f{i}' for i in range(NF)] + [f'x{i}' for i in range(NX)] + ['ok']
    at = [[f'a{f}_{k}' for k in range(ANS[f])] for f in range(NF)]
    toks += [t for r in at for t in r]
    table = {(f, x): at[f][rng.integers(ANS[f])] for f in range(NF) for x in range(NX)}
    ag = Agent(toks, variant)
    now = 0
    out = {}
    jit = lambda v: int(round(v * (1 + jitter * (2 * rng.random() - 1)))) if jitter else v

    def frozen():
        st = (ag.L, ag.O, ag.last_other); acc = 0; lat = []
        for f in range(NF):
            for x in range(NX):
                ag.L, ag.O, ag.last_other = st
                ag.hear([f'f{f}', f'x{x}'], 10 ** 7, 6, learn=False)
                tau, tok = ag.decide(10 ** 7 + 6)
                acc += (tok == table[(f, x)]); lat += [tau] if tau else []
        ag.L, ag.O, ag.last_other = st
        return round(acc / (NF * NX), 3), (round(float(np.mean(lat)), 1) if lat else None)

    for phase, n, *chk in schedule:
        chk = set(chk[0]) if chk else {n}
        sil = 0
        for i in range(1, n + 1):
            def one():
                nonlocal now, sil
                f, x = int(rng.integers(NF)), int(rng.integers(NX)); Q = [f'f{f}', f'x{x}']; a = table[(f, x)]
                now += jit(15)
                ag.hear(Q, now, 6); qend = now + 6
                tau, tok = ag.decide(qend, rng, explore=0.05 if phase != 'free' else 0.0)
                silent = tau is None or rng.random() < p_sil
                if phase == 'model':                      # other2 answers 3 ticks after Q (bench tutor='model')
                    ag.hear([a], qend + 3, 3); now = qend + 6; return
                tf = jit(tau_f)
                if phase == 'format' and (silent or tau >= tf):
                    ag.hear([a], qend + tf, 3); now = qend + tf + 3; return
                if silent:
                    sil += 1
                    if phase in ('recast', 'format'):     # told answer after my silence (format handled above)
                        ag.hear([a], qend + 45, 3); now = qend + 48
                    else:                                 # untutored: the next probe comes at Q_end + 45 (give-up)
                        now = qend + 45 - 15              # the loop adds 15 before the next Q -> onset at qend + 45
                    return
                ag.say([tok], qend + tau, 3); stop = qend + tau + 3
                if phase in ('recast', 'format'):
                    t_r = max(qend + 45, stop + 6) if phase == 'recast' else stop + 6
                    ag.hear(['ok', a] if tok == a else [a], t_r, 6); now = t_r + 6
                else:
                    now = max(qend + 45, stop + 13) - 15
            one()
            if i in chk and i != n:
                out[f'{phase}{i}'] = frozen()
        out[f'{phase}{n}'] = frozen(); out[f'{phase}{n}_sil'] = round(sil / n, 3)
    return out


if __name__ == '__main__':
    U = ('free', 3000, [300, 900, 2100, 3000])
    sched = {
        'format600 -> untutored3000, p_sil 0': ([('format', 600), U], 0.0),
        'format600 -> untutored3000, p_sil 0.3': ([('format', 600), U], 0.3),
        'model600 -> untutored3000, p_sil 0.3': ([('model', 600), U], 0.3),
        'recast1500 -> untutored3000, p_sil 0.3 (today)': ([('recast', 1500), U], 0.3),
        'format600 -> untutored3000, p_sil 0.3, timing jitter x[0.5,1.5]': ([('format', 600), U], 0.3, 0.5),
    }
    res = {}
    for name, spec in sched.items():
        sch, ps = spec[0], spec[1]; jit = spec[2] if len(spec) > 2 else 0.0
        res[name] = {}
        print('==', name)
        for v in ('V2', 'S', 'ST0', 'ST'):
            rows = [run(v, s, sch, p_sil=ps, jitter=jit) for s in range(5)]
            keys = [k for k in rows[0] if not k.endswith('_sil')]
            agg = {}
            for k in keys:
                acc = [r[k][0] for r in rows]; lat = [r[k][1] for r in rows if r[k][1] is not None]
                agg[k] = (round(float(np.mean(acc)), 3), round(float(np.std(acc)), 3),
                          round(float(np.mean(lat)), 1) if lat else None)
            sil = {k: round(float(np.mean([r[k] for r in rows])), 2) for k in rows[0] if k.endswith('_sil')}
            res[name][v] = {'acc': agg, 'silence': sil}
            print(f'  {v:4s}', '  '.join(f'{k}: {a:.2f}±{s:.2f}' + (f' (lat {l})' if l else '')
                                         for k, (a, s, l) in agg.items()), ' silence', sil)
    json.dump(res, open(sys.argv[1] if len(sys.argv) > 1 else 'drift_toy.json', 'w'), indent=1)
