"""Skeptic toy (local, seconds): does ARCH_v3_core's anchored context (latest x age, 'ST' of synth/drift_toy.py) stay
stable when O1 is applied the way the core states it -- 'every prediction synapse predicts its target area's next
activity' (tick level, so 'predicted but absent -> weaken' also fires on silent ticks) -- instead of drift_toy's
event-level learning (one update at each heard onset, no negatives on silent ticks)?

Also replaces drift_toy's 'first tau with p >= theta' decision by the core's M4 release rule
I += (G + u) * e  (e = margin best - second, bound 1, reset at event onsets), with G fixed (optimistic: Go already learned),
and an optional eta schedule eta * max(1/n_use, floor) per feature (M2 as written), and CD = 4 ticks.

World timing as drift_toy / v2.2 (PROVED there from world.py): format self-answer at tau_f = 10 unless I started before;
told answer after my silence at Q_end + 45; untutored next probe at max(Q_end + 45, stop + 13); praise/recast at stop + 6.
Run: python3 -I tick_toy.py
"""
import numpy as np, itertools, sys

NF, NX = 4, 8
ANS = [4, 2, 3, 3]
ETA = 0.35


def bins(tau):
    return [k for k in range(0, 9) if 2 ** k <= tau < 2 ** (k + 2)]


class Agent:
    def __init__(self, toks, eta_sched=None):
        self.ix = {t: i for i, t in enumerate(toks)}; self.toks = toks; self.V = len(toks)
        self.W = {}; self.n = {}
        self.L = None; self.O = None
        self.eta_sched = eta_sched   # None: constant eta; float f: eta * max(1/n_use, f)

    def feats(self, now):          # 'ST' of drift_toy: every feature carries the latest event's age
        if self.L is None:
            return []
        src, g, toff = self.L; g = sorted(g)
        og = sorted(self.O[1]) if self.O is not None else []
        osrc = self.O[0] if self.O is not None else None
        B = bins(max(1, now - toff))
        F = [('Lt', src, a, k) for a in g for k in B]
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
        err = (t - p) / len(F)
        for f in F:
            w = self.W.get(f)
            if w is None:
                w = self.W[f] = np.zeros(self.V); self.n[f] = 0
            self.n[f] += 1
            eta = ETA if self.eta_sched is None else ETA * max(1.0 / self.n[f], self.eta_sched)
            w += eta * err

    def event(self, src, items, t_on, dur):
        self.O, self.L = self.L, (src, list(items), t_on + dur)


def run(seed, schedule, mode='tick', decide='acc', p_sil=0.3, tau_f=10, G=0.8, u=0.15, theta=0.3, CD=4,
        eta_sched=None):
    rng = np.random.default_rng(seed)
    toks = [f'f{i}' for i in range(NF)] + [f'x{i}' for i in range(NX)] + ['ok']
    at = [[f'a{f}_{k}' for k in range(ANS[f])] for f in range(NF)]
    toks += [t for r in at for t in r]
    table = {(f, x): at[f][rng.integers(ANS[f])] for f in range(NF) for x in range(NX)}
    ag = Agent(toks, eta_sched)
    st = dict(now=0, last_learn=0)
    out = {}

    def gap_negatives(t_from, t_to):
        """tick mode: every tick strictly between t_from and t_to with no heard onset -> 'predicted but absent'"""
        if mode != 'tick':
            return
        for t in range(t_from + 1, t_to):
            ag.learn(ag.feats(t), [])

    def hear(items, t_on, dur, learn=True):
        if learn:
            gap_negatives(st['last_learn'], t_on)
            ag.learn(ag.feats(t_on), items)
        ag.event('o', items, t_on, dur); st['last_learn'] = t_on + dur

    def say(items, t_on, dur):
        if mode == 'tick':
            gap_negatives(st['last_learn'], t_on)
        ag.event('s', items, t_on, dur); st['last_learn'] = t_on + dur

    def choose(qend, cap, learn=True):
        """returns (onset tau, token) or (None, None); no learning inside (negatives are applied in bulk later,
        which is exact because the context does not change until the next event)"""
        I = 0.0
        for tau in range(1, cap):
            p = ag.pred(ag.feats(qend + tau))
            o = np.sort(p)
            if decide == 'thr':
                if o[-1] >= theta:
                    return tau, toks[int(np.argmax(p))]
            else:
                e = o[-1] - o[-2]
                I += (G + u) * e
                if I >= 1.0:
                    if tau + CD < cap:
                        return tau + CD, toks[int(np.argmax(p))]
                    return None, None
        return None, None

    def frozen():
        stt = (ag.L, ag.O); acc = 0
        for f in range(NF):
            for x in range(NX):
                ag.L, ag.O = stt
                ag.event('o', [f'f{f}', f'x{x}'], 10 ** 7, 6)
                tau, tok = choose(10 ** 7 + 6, 45)
                acc += (tok == table[(f, x)])
        ag.L, ag.O = stt
        return round(acc / (NF * NX), 3)

    for phase, n, *chk in schedule:
        chk = set(chk[0]) if chk else {n}
        sil = 0; right = 0; partner_tok = 0
        for i in range(1, n + 1):
          if i - 1 in chk and i - 1 not in (0, n):
              out[f'{phase}{i - 1}'] = frozen()
          for _once in (0,):
            f, x = int(rng.integers(NF)), int(rng.integers(NX)); Q = [f'f{f}', f'x{x}']; a = table[(f, x)]
            st['now'] += 15
            hear(Q, st['now'], 6); qend = st['now'] + 6
            if phase == 'model':
                hear([a], qend + 3, 3); st['now'] = qend + 6; continue
            cap = tau_f if phase == 'format' else 45
            tau, tok = choose(qend, cap)
            if tau is None or rng.random() < p_sil:
                tau = None
            if tau is None:
                sil += 1
                if phase == 'format':
                    hear([a], qend + tau_f, 3); st['now'] = qend + tau_f + 3
                elif phase == 'recast':
                    hear([a], qend + 45, 3); st['now'] = qend + 48
                else:
                    gap_negatives(st['last_learn'], qend + 45); st['last_learn'] = qend + 45 - 1
                    st['now'] = qend + 45 - 15
                continue
            say([tok], qend + tau, 3); stop = qend + tau + 3
            right += tok == a; partner_tok += tok[0] in 'fx'
            if phase in ('recast', 'format'):
                t_r = max(qend + 45, stop + 6) if phase == 'recast' else stop + 6
                hear(['ok', a] if tok == a else [a], t_r, 6); st['now'] = t_r + 6
            else:
                nxt = max(qend + 45, stop + 13)
                gap_negatives(st['last_learn'], nxt); st['last_learn'] = nxt - 1
                st['now'] = nxt - 15
        out[f'{phase}{n}'] = frozen()
        out[f'{phase}{n}_live'] = (round(right / n, 2), round(sil / n, 2), round(partner_tok / n, 2))
    return out


if __name__ == '__main__':
    U = ('free', 3000, [300, 900, 2100])
    sched = [('format', 600), U]
    arms = [
        ('event-level, threshold (= drift_toy ST)', dict(mode='event', decide='thr')),
        ('event-level, M4 accumulator', dict(mode='event', decide='acc')),
        ('tick-level O1, threshold', dict(mode='tick', decide='thr')),
        ('tick-level O1, M4 accumulator', dict(mode='tick', decide='acc')),
        ('event-level, acc, eta*max(1/n_use, 0.1)', dict(mode='event', decide='acc', eta_sched=0.1)),
        ('tick-level, acc, eta*max(1/n_use, 0.1)', dict(mode='tick', decide='acc', eta_sched=0.1)),
    ]
    for name, kw in arms:
        rows = [run(s, sched, **kw) for s in range(5)]
        keys = [k for k in rows[0] if not k.endswith('_live')]
        msg = '  '.join(f'{k}: {np.mean([r[k] for r in rows]):.2f}' for k in keys)
        live = {k: tuple(np.round(np.mean([r[k] for r in rows], axis=0), 2)) for k in rows[0] if k.endswith('_live')}
        print(f'== {name}\n   frozen acc  {msg}\n   live (right, silent, partner-token) {live}', flush=True)
