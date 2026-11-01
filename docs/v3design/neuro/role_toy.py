"""Toy: does role-aware listening need hand rules, or does it fall out of
(1) the self in the input stream (efference-tagged source) and (2) a context that keeps the
latest line apart from the older one (fast / slow population)?

Symbolic dialogue world (no acoustics). Lines are bags of tokens.
  question  Q = [frame f, filler x]        (said by the partner)
  answer    a = [ans(f, x)]                (by a model speaker, by the partner as a recast/correction)
Phases: 'model' (Q, other speaker answers), 'recast' (Q, agent replies, partner says the correct answer,
reward 1 if the agent was right), 'free' (Q, agent replies, partner just asks the next question).

Learner: one signed delta rule (the bench's R2 form): p_j = clip(sum_i w_ij, 0, 1),
dw_ij = eta * m * (t_j - p_j) / |A| over the active features A of the context at the line's onset.
Speaking = say the top token predicted in the context 'it is my turn after the partner's line'.

Context variants (the only thing that differs):
  A  v2      : context = last line of ANOTHER speaker (own speech never enters)       [ARCH_v2]
  B  bag     : context = bag(last other line + my own reply after it), no source tag  [ct_own-like]
  C  slot    : context = group L (latest line, items tagged with its source self/other)
               + group O (the line before it); features = singles + pairs within L + pairs L x O
Production route (only C*): my own utterance leaves an eligibility trace on its context C_spk;
the next line from another speaker is a target for C_spk with weight m:
  C0  m = 0                       (no route)
  C1  m = 1                       (commit always: RAL without any cancel rule)
  C2  m = reward                  (DA burst only)
  C3  m = max(reward, answerness) (answerness = the line's tokens are predicted by the agent's own
                                   answer readout for C_spk at >= 0.05: a learned 'is this an answer to
                                   what I was answering' test; ASSUMED form)
  C4  m = max(reward, family)      (family = the line's tokens have an outstar expectation >= 0.1 from a
                                   single feature of C_spk, learned from rewarded / overheard answers)
  R   (RAL-like, context A)        a heard line's learning waits; cancelled if I speak next, unless rewarded
R, C2, C3, C4 also have a critic + actor with signed RPE (bursts and dips), as ARCH_v2.1.
Measures: frozen accuracy on all (f, x) after each phase; role confusion = share of replies that are a
frame token (a question word). Run: python3 -I role_toy.py
"""
import numpy as np, itertools, json, sys

NF, NX = 4, 8
ANS_SIZES = [4, 2, 3, 3]          # colours, tastes, counts, places


def make_world(seed):
    rng = np.random.default_rng(seed)
    toks = [f'f{i}' for i in range(NF)] + [f'x{i}' for i in range(NX)]
    ans_tok = [[f'a{f}_{k}' for k in range(ANS_SIZES[f])] for f in range(NF)]
    toks += [t for row in ans_tok for t in row]
    table = {(f, x): ans_tok[f][rng.integers(ANS_SIZES[f])] for f in range(NF) for x in range(NX)}
    return toks, table, rng


class Learner:
    def __init__(self, toks, variant, eta=0.35):
        self.toks = toks; self.ix = {t: i for i, t in enumerate(toks)}
        self.V = len(toks); self.variant = variant; self.eta = eta
        self.W = {}                       # feature -> np.array(V)
        self.L = None; self.O = None      # (source, items) of the latest / previous line
        self.last_other = None            # items of the last other-speaker line (variant A)
        self.own_after = None             # my reply after last_other (variant B)
        self.trace = None                 # production eligibility: (features, own reply)
        self.pending = None               # variant R (RAL-like): a heard line's learning waits for its role
        self.rpe = variant in ('R', 'C2', 'C3', 'C4')   # critic + actor with signed RPE (DA bursts and dips), as v2.1
        self.Vw = {}
        self.Out = {}; self.Outn = {}     # C4: outstar from each single feature of the open context to the
                                          # tokens heard in a known answer position (rewarded or model reply)

    # --- context features ---
    def feats(self):
        v = self.variant[0]
        F = []
        if v in ('A', 'R'):
            g = sorted(self.last_other or [])
            F = [('s', a) for a in g] + [('p', a, b) for a, b in itertools.combinations(g, 2)]
        elif v == 'B':
            g = sorted(set((self.last_other or []) + (self.own_after or [])))
            F = [('s', a) for a in g] + [('p', a, b) for a, b in itertools.combinations(g, 2)]
        else:
            if self.L is not None:
                src, g = self.L; g = sorted(g)
                F += [('L', src, a) for a in g] + [('LL', src, a, b) for a, b in itertools.combinations(g, 2)]
                if self.O is not None:
                    osrc, og = self.O
                    F += [('LO', src, a, osrc, b) for a in g for b in sorted(og)]
        return F

    def drive(self, F):
        d = np.zeros(self.V)
        for f in F:
            w = self.W.get(f)
            if w is not None: d += w
        return d

    def learn(self, F, items, m=1.0):
        if not F or m <= 0: return
        p = np.clip(self.drive(F), 0, 1)
        t = np.zeros(self.V)
        for it in items: t[self.ix[it]] = 1
        dw = self.eta * m * (t - p) / len(F)
        for f in F:
            w = self.W.get(f)
            if w is None: w = self.W[f] = np.zeros(self.V)
            w += dw

    # --- events ---
    def _resolve(self, spoke):
        if self.pending is not None:
            F, items, r = self.pending
            if r > 0 or not spoke:                     # commit: rewarded, or nobody answered it
                self.learn(F, items)
            self.pending = None                        # cancel: I answered it -> it was a question

    def _outstar(self, F, items):
        t = np.zeros(self.V)
        for it in items: t[self.ix[it]] = 1
        for f in F:
            if f[0] != 'L': continue
            o = self.Out.get(f)
            if o is None: o = self.Out[f] = np.zeros(self.V); self.Outn[f] = 0
            self.Outn[f] += 1
            o += (t - o) * max(1.0 / self.Outn[f], 0.02)

    def _rpe(self, Fs, own, r):
        V = float(np.clip(sum(self.Vw.get(f, 0.0) for f in Fs), 0, 1))
        d = r - V
        for f in Fs:
            self.Vw[f] = self.Vw.get(f, 0.0) + 0.2 * d / len(Fs)
            w = self.W.get(f)
            if w is None: w = self.W[f] = np.zeros(self.V)
            w[self.ix[own[0]]] += 0.3 * d / len(Fs)     # actor step on my own choice (burst > 0, dip < 0)

    def hear(self, items, reward=0.0, learn=True):
        F = self.feats()
        if learn and self.rpe and self.trace is not None:
            self._rpe(self.trace[0], self.trace[1], reward)   # the reaction to my utterance
        if self.variant == 'R':
            self._resolve(spoke=False)
            if learn: self.pending = (F, list(items), reward)
            learn = False
        if learn:
            self.learn(F, items)                       # every heard other line: predicted from its onset context
            if self.variant == 'C4' and self.trace is None and self.L is not None and self.L[0] == 'o':
                self._outstar(F, items)                # another speaker answered the line: a known answer position
            if self.trace is not None and self.variant in ('C1', 'C2', 'C3', 'C4'):
                Fs, own = self.trace
                if self.variant == 'C1': m = 1.0
                elif self.variant == 'C2': m = reward
                elif self.variant == 'C3':
                    p = np.clip(self.drive(Fs), 0, 1)
                    ansness = 1.0 if min(p[self.ix[it]] for it in items) >= 0.05 else 0.0
                    m = max(reward, ansness)
                else:
                    sing = [f for f in Fs if f[0] == 'L']
                    e = max((min(self.Out[f][self.ix[it]] for it in items) for f in sing if f in self.Out),
                            default=0.0)
                    m = max(reward, 1.0 if e >= 0.1 else 0.0)
                    if reward > 0: self._outstar(Fs, items)
                self.learn(Fs, items, m)
        self.trace = None
        self.O, self.L = self.L, ('o', list(items))
        self.last_other = list(items); self.own_after = None

    def speak(self, rng, explore=0.1):
        if self.variant == 'R': self._resolve(spoke=True)
        F = self.feats()
        d = self.drive(F)
        if rng.random() < explore or d.max() < 0.05:
            j = int(rng.integers(self.V))
        else:
            j = int(np.argmax(d + 1e-6 * rng.random(self.V)))
        reply = [self.toks[j]]
        self.trace = (F, reply)
        self.O, self.L = self.L, ('s', reply)
        self.own_after = reply
        return reply

    def answer_frozen(self, items):
        """frozen probe: hear Q without learning, answer greedily, restore state"""
        st = (self.L, self.O, self.last_other, self.own_after, self.trace)
        self.O, self.L = self.L, ('o', list(items)); self.last_other = list(items); self.own_after = None
        d = self.drive(self.feats())
        r = self.toks[int(np.argmax(d))] if d.max() > 0 else None
        self.L, self.O, self.last_other, self.own_after, self.trace = st
        return r


def run(variant, seed, schedule, takeup=0.0):
    toks, table, rng = make_world(seed)
    ag = Learner(toks, variant)
    out = {}
    def q():
        f, x = int(rng.integers(NF)), int(rng.integers(NX))
        return f, x, [f'f{f}', f'x{x}']
    for phase, n in schedule:
        conf = 0; nrep = 0
        for _ in range(n):
            f, x, Q = q(); a = table[(f, x)]
            ag.hear(Q)
            if phase == 'model':
                ag.hear([a])
            else:
                r = ag.speak(rng); nrep += 1
                conf += r[0].startswith('f')
                if phase == 'recast':
                    ag.hear([a], reward=float(r[0] == a))
                    if rng.random() < takeup:
                        ag.speak(rng)                          # the child talks right after the reaction
                # 'free': nothing; the next question comes
        acc = np.mean([ag.answer_frozen([f'f{f}', f'x{x}']) == table[(f, x)]
                       for f in range(NF) for x in range(NX)])
        out[f'{phase}{n}'] = {'acc': round(float(acc), 3),
                              'role_conf': round(conf / max(nrep, 1), 3) if nrep else None}
    return out


if __name__ == '__main__':
    variants = ['A', 'B', 'R', 'C0', 'C1', 'C2', 'C3', 'C4']
    schedules = {
        'model->free': [('model', 600), ('free', 300), ('free', 600)],
        'recast->free': [('recast', 1500), ('free', 300), ('free', 600)],
        'model->recast->free': [('model', 300), ('recast', 600), ('free', 300), ('free', 600)],
        'recast+takeup0.85->free': [('recast', 1500), ('free', 300), ('free', 600)],
        'recast+takeup1.0->free': [('recast', 1500), ('free', 300), ('free', 600)],
    }
    res = {}
    for sname, sch in schedules.items():
        res[sname] = {}
        for v in variants:
            rows = [run(v, s, sch, takeup=(float(sname.split('takeup')[1].split('-')[0]) if 'takeup' in sname else 0.0)) for s in range(5)]
            keys = rows[0].keys()
            res[sname][v] = {k: {'acc': round(float(np.mean([r[k]['acc'] for r in rows])), 3),
                                 'acc_sd': round(float(np.std([r[k]['acc'] for r in rows])), 3),
                                 'role_conf': (round(float(np.mean([r[k]['role_conf'] for r in rows])), 3)
                                               if rows[0][k]['role_conf'] is not None else None)}
                             for k in keys}
    for sname in res:
        print('==', sname)
        for v in variants:
            print(f'  {v:3s}', '  '.join(f"{k}: acc {d['acc']:.2f}±{d['acc_sd']:.2f}"
                                         + (f" conf {d['role_conf']:.2f}" if d['role_conf'] is not None else '')
                                         for k, d in res[sname][v].items()))
    json.dump(res, open(sys.argv[1] if len(sys.argv) > 1 else 'role_toy.json', 'w'), indent=1)
