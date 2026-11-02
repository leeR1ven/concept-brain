"""The designer's metaplasticity idea (2026-11-01): 'frequently used, strong weights become harder to raise and lower'.
Toy: one question context c (|F| active features) with an NLMS readout over V answer tokens (bench R2 form).
  phase T  (tutored): n_T = 20 presentations with target A.
  then either
  R1 'untutored wrong targets' (the v2.2 failure kind: an unanswered question's successor is learned as its answer):
       each presentation of c carries a random wrong target with prob q = 0.3, otherwise no learning event.
       Survival = presentations until c no longer answers A (argmax != A or p_A < theta).  x32 = dialogues in a
       32-question world.
  R2 'correction': every presentation carries the new right answer A*.  Latency = presentations until c answers A*.
  R3 'correction, then untutored': corrected (R2 until it answers A*, +5), then R1: does the OLD habit return?
Rules: plain NLMS; |w|-metaplastic (designer); use-metaplastic (M2's 1/n_use + floor = tune's maturation);
       both; 2-variable Benna-Fusi synapse (visible u1 pulled to hidden u2); salience-gated consolidation
       (fast/slow weight, slow absorbs fast only on confirmed predictions).
Run: python3 -I meta_toy.py
"""
import numpy as np, json

V, F, ETA, THETA = 10, 10, 0.35, 0.3


class Syn:
    def __init__(self, rule, prm):
        self.rule, self.prm = rule, prm
        self.w = np.zeros((F, V)); self.u2 = np.zeros((F, V)); self.n = np.zeros((F, V)); self.ws = np.zeros((F, V))

    def W(self):
        if self.rule == 'salience':
            return self.w + self.ws
        return self.w

    def p(self):
        return np.clip(self.W().sum(0), 0, 1)

    def step(self, target):
        p = self.p(); t = np.zeros(V)
        if target is None:                             # time passes, no learning event (coupling / decay only)
            err = np.zeros((F, V))
        else:
            t[target] = 1; err = (t - p)[None, :] * np.ones((F, 1))
        r, q = self.rule, self.prm
        rate = ETA / F
        if r in ('absw', 'both'):
            rate = rate / (1 + np.abs(self.W()) / q['w0'])
        if r in ('use', 'both'):
            rate = rate * np.maximum(1.0 / (1.0 + self.n / q['ucap']), q['floor'])
        dw = rate * err
        if r == 'bf':                                  # Benna & Fusi 2016, two variables
            self.w += dw + q['g'] * (self.u2 - self.w)
            self.u2 += q['g'] / q['C2'] * (self.w - self.u2)
        elif r == 'salience':                          # fast weight learns; slow absorbs it when the guess was right
            self.w += dw
            if target is not None and int(np.argmax(p)) == target and p.max() >= THETA:
                self.ws += q['k'] * self.w; self.w *= (1 - q['k'])
            self.w *= (1 - q['decay'])                 # the unconsolidated part fades
        else:
            self.w += dw
        self.n += (t[None, :] > 0)                     # use = this synapse was active with its target present

    def answers(self, a):
        p = self.p(); return int(np.argmax(p)) == a and p.max() >= THETA


def trial(rule, prm, seed, nT=20, q=0.3, horizon=4000):
    rng = np.random.default_rng(seed); A, A2 = 0, 1
    s = Syn(rule, prm)
    for _ in range(nT):
        s.step(A)
    import copy
    base = copy.deepcopy(s)
    # R1
    surv = horizon
    for i in range(1, horizon + 1):
        s.step(int(rng.integers(2, V)) if rng.random() < q else None)
        if not s.answers(A):
            surv = i; break
    # R2
    s = copy.deepcopy(base); lat = None
    for i in range(1, 2001):
        s.step(A2)
        if s.answers(A2):
            lat = i; break
    # R3: after correction (+5 more), untutored wrong targets: does A (old habit) come back before A2 is lost?
    for _ in range(5):
        s.step(A2)
    back = 0; surv2 = horizon
    for i in range(1, horizon + 1):
        s.step(int(rng.integers(2, V)) if rng.random() < q else None)
        pp = s.p()
        if int(np.argmax(pp)) == A and pp.max() >= THETA:
            back = 1
        if not s.answers(A2):
            surv2 = i; break
    return surv, lat, back, surv2


RULES = [
    ('plain NLMS (today R2)', 'plain', {}),
    ('|w| meta w0=0.05', 'absw', {'w0': 0.05}),
    ('|w| meta w0=0.01', 'absw', {'w0': 0.01}),
    ('use meta ucap=5 floor 0.1', 'use', {'ucap': 5, 'floor': 0.1}),
    ('use meta ucap=5 floor 0.02', 'use', {'ucap': 5, 'floor': 0.02}),
    ('|w|+use (designer form) w0 .05 ucap 5 floor .1', 'both', {'w0': 0.05, 'ucap': 5, 'floor': 0.1}),
    ('Benna-Fusi 2-var g .2 C2 20', 'bf', {'g': 0.2, 'C2': 20}),
    ('Benna-Fusi 2-var g .5 C2 50', 'bf', {'g': 0.5, 'C2': 50}),
    ('confirmed-consolidation k .2 decay .05', 'salience', {'k': 0.2, 'decay': 0.05}),
]

if __name__ == '__main__':
    res = {}
    print(f'{"rule":48s} {"R1 survival":>12s} {"R2 latency":>11s} {"R3 old back":>11s} {"R3 survival":>11s}')
    for name, r, prm in RULES:
        rows = [trial(r, prm, s) for s in range(30)]
        sv = np.median([x[0] for x in rows]); lt = np.median([x[1] if x[1] else 2001 for x in rows])
        bk = np.mean([x[2] for x in rows]); sv2 = np.median([x[3] for x in rows])
        res[name] = dict(R1_survival=float(sv), R2_latency=float(lt), R3_oldback=float(bk), R3_survival=float(sv2))
        print(f'{name:48s} {sv:12.0f} {lt:11.0f} {bk:11.2f} {sv2:11.0f}', flush=True)
    print('(survival/latency in presentations of the context; x32 = dialogues in a 32-question world)')
    json.dump(res, open('meta_toy.json', 'w'), indent=1)
