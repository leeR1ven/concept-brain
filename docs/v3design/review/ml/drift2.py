"""ML reviewer re-check of synth/drift_toy.py (ARCH_v3_core §0.2, §6.5).

Same symbolic world, learner and ST/S contexts as drift_toy (code re-implemented; crisp setting reproduces it), plus:
  * jit_all  : per-event log-uniform jitter x[lo,hi] on EVERY world delay (format self-answer tau_f, give-up delay 45,
               stop+6 / stop+13 gaps, the 15-tick lead-in).  drift_toy jittered only tau_f and the lead-in.
  * graded   : time cells with log-Gaussian tuning (sigma in octaves) instead of crisp 2-octave boxes.
  * meta     : the designer's metaplasticity (2026-11-01): per-synapse rate eta / (1 + |w| / w0) * use-factor.
  * horizon  : untutored to 10,000 dialogues (the L6 pass bar).
Run: python3 -I drift2.py  (one process, < 100 MB)
"""
import numpy as np, itertools, json, sys, math

NF, NX = 4, 8
ANS = [4, 2, 3, 3]
ETA, THETA = 0.35, 0.3


class Agent:
    def __init__(self, toks, variant, graded=0.0, meta=None):
        self.ix = {t: i for i, t in enumerate(toks)}; self.toks = toks; self.V = len(toks); self.v = variant
        self.W = {}; self.U = {}; self.L = None; self.O = None
        self.graded = graded; self.meta = meta

    def tbins(self, tau):
        tau = max(1, tau)
        if not self.graded:
            return [(k, 1.0) for k in range(0, 9) if 2 ** k <= tau < 2 ** (k + 2)]
        lt = math.log2(tau); s = self.graded
        out = []
        for k in range(0, 9):
            g = math.exp(-(lt - (k + 1)) ** 2 / (2 * s * s))   # centre 2^(k+1) = geometric centre of [2^k, 2^(k+2))
            if g >= 0.02:
                out.append((k, g))
        return out

    def feats(self, now):
        if self.L is None:
            return {}
        src, g, toff = self.L; g = sorted(g)
        og = sorted(self.O[1]) if self.O is not None else []
        osrc = self.O[0] if self.O is not None else None
        F = {}
        if self.v == 'S':
            for a in g: F[('L', src, a)] = 1.0
            for a, b in itertools.combinations(g, 2): F[('LL', src, a, b)] = 1.0
            for a in g:
                for o in og: F[('LO', src, a, osrc, o)] = 1.0
            return F
        B = self.tbins(now - toff)
        for k, x in B:
            for a in g: F[('Lt', src, a, k)] = x
            for a, b in itertools.combinations(g, 2): F[('LLt', src, a, b, k)] = x
            for a in g:
                for o in og: F[('LOt', src, a, k, osrc, o)] = x
        return F

    def pred(self, F):
        d = np.zeros(self.V)
        for f, x in F.items():
            w = self.W.get(f)
            if w is not None:
                d += x * w
        return np.clip(d, 0, 1)

    def learn(self, F, items):
        if not F:
            return
        p = self.pred(F); t = np.zeros(self.V)
        for it in items:
            t[self.ix[it]] = 1
        err = (t - p); sx = sum(F.values())
        for f, x in F.items():
            w = self.W.get(f)
            if w is None:
                w = self.W[f] = np.zeros(self.V); self.U[f] = np.zeros(self.V)
            rate = ETA * x / sx
            if self.meta is not None:          # designer's idea: strong, used weights resist change both ways
                w0, ucap = self.meta
                rate = rate / (1.0 + np.abs(w) / w0)
                if ucap:                       # plus use-dependence (M2's 1/n_use with a floor)
                    rate = rate * np.maximum(1.0 / (1.0 + self.U[f] / ucap), 0.1)
                self.U[f] += x * (t > 0)       # use = times this synapse was active with its target present
            w += rate * err

    def hear(self, items, now, dur, learn=True):
        if learn:
            self.learn(self.feats(now), items)
        self.O, self.L = self.L, ('o', list(items), now + dur)

    def say(self, items, now, dur):
        self.O, self.L = self.L, ('s', list(items), now + dur)

    def decide(self, now0, rng=None, explore=0.0):
        for tau in range(1, 45):
            p = self.pred(self.feats(now0 + tau))
            if p.max() >= THETA:
                j = int(np.argmax(p + 1e-9 * (rng.random(self.V) if rng is not None else 0)))
                if rng is not None and rng.random() < explore:
                    j = int(rng.integers(self.V))
                return tau, self.toks[j]
        return None, None


def run(variant, seed, schedule, p_sil=0.0, tau_f=10, jit_all=None, graded=0.0, meta=None):
    rng = np.random.default_rng(seed)
    toks = [f'f{i}' for i in range(NF)] + [f'x{i}' for i in range(NX)] + ['ok']
    at = [[f'a{f}_{k}' for k in range(ANS[f])] for f in range(NF)]
    toks += [t for r in at for t in r]
    table = {(f, x): at[f][rng.integers(ANS[f])] for f in range(NF) for x in range(NX)}
    ag = Agent(toks, variant, graded, meta)
    now = 0; out = {}

    def J(v):          # per-event log-uniform jitter on a world delay
        if not jit_all:
            return v
        lo, hi = jit_all
        return max(1, int(round(v * math.exp(rng.uniform(math.log(lo), math.log(hi))))))

    def frozen():
        st = (ag.L, ag.O); acc = 0; lat = []
        for f in range(NF):
            for x in range(NX):
                ag.L, ag.O = st
                ag.hear([f'f{f}', f'x{x}'], 10 ** 7, 6, learn=False)
                tau, tok = ag.decide(10 ** 7 + 6)
                acc += (tok == table[(f, x)]); lat += [tau] if tau else []
        ag.L, ag.O = st
        return round(acc / (NF * NX), 3), (round(float(np.mean(lat)), 1) if lat else None)

    for phase, n, chk in schedule:
        sil = 0
        for i in range(1, n + 1):
            f, x = int(rng.integers(NF)), int(rng.integers(NX)); Q = [f'f{f}', f'x{x}']; a = table[(f, x)]
            now += J(15)
            ag.hear(Q, now, 6); qend = now + 6
            tau, tok = ag.decide(qend, rng, explore=0.05 if phase != 'free' else 0.0)
            silent = tau is None or rng.random() < p_sil
            tf = J(tau_f)
            if phase == 'format' and (silent or tau >= tf):
                ag.hear([a], qend + tf, 3); now = qend + tf + 3
            elif silent:
                sil += 1
                if phase == 'format':
                    ag.hear([a], qend + J(45), 3); now = qend + 48
                else:                             # untutored give-up: next probe after a jittered delay
                    now = qend + J(45) - 15        # (the loop adds the lead-in; jittered too)
            else:
                ag.say([tok], qend + tau, 3); stop = qend + tau + 3
                if phase == 'format':
                    t_r = stop + J(6)
                    ag.hear(['ok', a] if tok == a else [a], t_r, 6); now = t_r + 6
                else:
                    now = max(qend + J(45), stop + J(13)) - 15
            if i in chk:
                out[f'{phase}{i}'] = frozen()
        out[f'{phase}_sil'] = round(sil / n, 3)
    return out


if __name__ == '__main__':
    NFREE = int(sys.argv[1]) if len(sys.argv) > 1 else 10000
    chk_free = [c for c in (300, 1000, 3000, 10000) if c <= NFREE]
    sched = [('format', 600, [600]), ('free', NFREE, chk_free)]
    conds = [
        ('A  drift_toy setting (crisp bins, no world jitter)', dict()),
        ('B  per-event jitter x[0.75,1.33] on all delays (development range)', dict(jit_all=(0.75, 1.33))),
        ('C  per-event jitter x[0.5,2] on all delays (sealed range)', dict(jit_all=(0.5, 2.0))),
        ('D  graded log-Gaussian time cells, sigma 0.85 oct, no jitter', dict(graded=0.85)),
        ('E  graded sigma 0.85 + jitter x[0.5,2]', dict(graded=0.85, jit_all=(0.5, 2.0))),
        ('F  C + metaplasticity |w| (w0 0.3)', dict(jit_all=(0.5, 2.0), meta=(0.3, 0))),
        ('G  C + metaplasticity |w| + use (w0 0.3, ucap 5)', dict(jit_all=(0.5, 2.0), meta=(0.3, 5))),
    ]
    res = {}
    for name, kw in conds:
        res[name] = {}
        print('==', name, flush=True)
        for v in (('S', 'ST') if name.startswith('A') or name.startswith('C') else ('ST',)):
            rows = [run(v, s, sched, p_sil=0.3, **kw) for s in range(5)]
            keys = [k for k in rows[0] if not k.endswith('_sil')]
            agg = {}
            for k in keys:
                acc = [r[k][0] for r in rows]; lat = [r[k][1] for r in rows if r[k][1] is not None]
                agg[k] = (round(float(np.mean(acc)), 3), round(float(np.std(acc)), 3),
                          round(float(np.mean(lat)), 1) if lat else None, [r[k][0] for r in rows])
            res[name][v] = agg
            print(f'  {v:3s}', '  '.join(f'{k}: {a:.2f}±{s:.2f}' + (f' (lat {l})' if l else '')
                                         for k, (a, s, l, _) in agg.items()), flush=True)
    json.dump(res, open('drift2.json', 'w'), indent=1)
