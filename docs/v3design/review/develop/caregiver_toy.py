"""Developmental review toy (local, < 100 MB, ~1 min): how realistic must the W1 'format' caregiver be for the anchored
(latest x age) context of ARCH_v3_core to work?

Re-uses the Agent class of the synthesis' own toy (docs/v3design/synth/drift_toy.py, imported read-only, unchanged), and
replaces only the WORLD (the caregiver), with behaviours real caregivers show after a child's silence:
  g_quick : 'moves on' -- she asks the next question after a short pause (U[8,16] ticks) instead of answering herself
            (series of questions in picture-book talk; response pursuit by moving on).  Also used for give-ups in the
            untutored phase (she moves on quickly with the same probability).
  q_reask : 're-asks' -- at tau_f she repeats the question (pursuit / prompt), then answers herself tau_f later.
  gu_jit  : per-event jitter (+-) of the 45-tick give-up delay (the synthesis toy jittered only tau_f and the 15-tick gap).
  scale   : one timing factor for the whole untutored test world (sealed range x[0.5, 2]); development is at x1.
Output: frozen accuracy on all 32 (frame, filler) pairs, and the share of answers that are QUESTION tokens (echo).
Run: python3 -I caregiver_toy.py
"""
import sys, json
import numpy as np

sys.path.insert(0, 'docs/v3design/synth')
from drift_toy import Agent, NF, NX, ANS          # noqa: E402  (read-only import; that file is not modified)


def run(variant, seed, n_format=600, n_free=3000, p_sil=0.3, tau_f=10, g_quick=0.0, q_reask=0.0, gu_jit=0.0,
        scale=1.0, checks=(300, 900, 3000), p_tut=0.0):
    rng = np.random.default_rng(seed)
    toks = [f'f{i}' for i in range(NF)] + [f'x{i}' for i in range(NX)] + ['ok']
    at = [[f'a{f}_{k}' for k in range(ANS[f])] for f in range(NF)]
    toks += [t for r in at for t in r]
    qtok = set(toks[:NF + NX])
    table = {(f, x): at[f][rng.integers(ANS[f])] for f in range(NF) for x in range(NX)}
    ag = Agent(toks, variant)
    now = 0
    out = {}

    def frozen():
        st = (ag.L, ag.O, ag.last_other); acc = echo = sil = 0
        for f in range(NF):
            for x in range(NX):
                ag.L, ag.O, ag.last_other = st
                ag.hear([f'f{f}', f'x{x}'], 10 ** 7, 6, learn=False)
                tau, tok = ag.decide(10 ** 7 + 6)
                acc += tok == table[(f, x)]; echo += tok in qtok; sil += tok is None
        ag.L, ag.O, ag.last_other = st
        n = NF * NX
        return round(acc / n, 3), round(echo / n, 3), round(sil / n, 3)

    def answer_after(Q, a, qend, sc):
        """child's turn after a question heard at qend; tutored format world. returns the new 'now' (before +15)."""
        tau, tok = ag.decide(qend, rng, explore=0.05)
        silent = tau is None or rng.random() < p_sil
        tf = int(round(tau_f * sc))
        if silent or tau >= tf:
            r = rng.random()
            if r < g_quick:                                   # moves on: next question soon, no answer given
                return qend + int(rng.integers(8, 17) * sc) - int(15 * sc)
            if r < g_quick + q_reask:                         # re-asks, then answers herself
                ag.hear(Q, qend + tf, 6); qe2 = qend + tf + 6
                tau2, tok2 = ag.decide(qe2, rng, explore=0.05)
                if tau2 is not None and rng.random() >= p_sil and tau2 < tf:
                    ag.say([tok2], qe2 + tau2, 3); stop = qe2 + tau2 + 3
                    t_r = stop + 6; ag.hear(['ok', a] if tok2 == a else [a], t_r, 6); return t_r + 6
                ag.hear([a], qe2 + tf, 3); return qe2 + tf + 3
            ag.hear([a], qend + tf, 3); return qend + tf + 3     # answers herself (the W1 format)
        ag.say([tok], qend + tau, 3); stop = qend + tau + 3
        t_r = stop + 6                                         # polite contingent reaction (W4)
        ag.hear(['ok', a] if tok == a else [a], t_r, 6); return t_r + 6

    for i in range(1, n_format + 1):                          # development world, timing x1
        f, x = int(rng.integers(NF)), int(rng.integers(NX)); Q = [f'f{f}', f'x{x}']
        now += 15; ag.hear(Q, now, 6)
        now = answer_after(Q, table[(f, x)], now + 6, 1.0)
    out['format'] = frozen()
    sc = scale
    for i in range(1, n_free + 1):                            # untutored use (nobody reacts), timing x scale
        f, x = int(rng.integers(NF)), int(rng.integers(NX)); Q = [f'f{f}', f'x{x}']
        now += int(15 * sc); ag.hear(Q, now, 6); qend = now + 6
        if p_tut and rng.random() < p_tut:                   # life goes on: the same caregiver still tutors sometimes
            now = answer_after(Q, table[(f, x)], qend, 1.0)
            if i in checks:
                out[f'free{i}'] = frozen()
            continue
        tau, tok = ag.decide(qend, rng, explore=0.0)
        if tau is None or rng.random() < p_sil:              # silent -> she gives up and asks the next question
            if rng.random() < g_quick:
                d = int(rng.integers(8, 17) * sc)
            else:
                d = int(round(45 * sc * (1 + gu_jit * (2 * rng.random() - 1))))
            now = qend + d - int(15 * sc)
        else:
            ag.say([tok], qend + tau, 3); stop = qend + tau + 3
            now = max(qend + int(45 * sc), stop + int(13 * sc)) - int(15 * sc)
        if i in checks:
            out[f'free{i}'] = frozen()
    return out


def main():
    arms = [
        ('W1 as specified (toy default)', {}),
        ('moves on quickly 10 %', dict(g_quick=0.10)),
        ('moves on quickly 25 %', dict(g_quick=0.25)),
        ('re-asks 30 %', dict(q_reask=0.30)),
        ('re-asks 60 %', dict(q_reask=0.60)),
        ('give-up delay jitter +-50 %', dict(gu_jit=0.5)),
        ('untutored world timing x0.5', dict(scale=0.5)),
        ('untutored world timing x2', dict(scale=2.0)),
        ('realistic mix: move-on 15 %, re-ask 30 %, jitter 50 %', dict(g_quick=0.15, q_reask=0.30, gu_jit=0.5)),
    ]
    res = {}
    for name, kw in arms:
        res[name] = {}
        for v in ('S', 'ST'):
            rows = [run(v, s, **kw) for s in range(5)]
            agg = {k: tuple(round(float(np.mean([r[k][j] for r in rows])), 2) for j in range(3)) for k in rows[0]}
            res[name][v] = agg
            print(f'{name:55s} {v:3s} ' + '  '.join(f'{k}: acc {a:.2f} echo {e:.2f} sil {s:.2f}'
                                                   for k, (a, e, s) in agg.items()), flush=True)
    json.dump(res, open('caregiver_toy.json', 'w'), indent=1)


def main2():
    arms = [
        ('realistic mix, care continues 50 %', dict(g_quick=0.15, q_reask=0.30, gu_jit=0.5, p_tut=0.5)),
        ('W1 as specified, care continues 50 %', dict(p_tut=0.5)),
        ('moves on 10 %, care continues 50 %', dict(g_quick=0.10, p_tut=0.5)),
    ]
    res = {}
    for name, kw in arms:
        res[name] = {}
        for v in ('S', 'ST'):
            rows = [run(v, s, **kw) for s in range(5)]
            agg = {k: tuple(round(float(np.mean([r[k][j] for r in rows])), 2) for j in range(3)) for k in rows[0]}
            res[name][v] = agg
            print(f'{name:55s} {v:3s} ' + '  '.join(f'{k}: acc {a:.2f} echo {e:.2f} sil {s:.2f}'
                                                   for k, (a, e, s) in agg.items()), flush=True)
    json.dump(res, open('caregiver_toy2.json', 'w'), indent=1)


if __name__ == '__main__':
    (main2 if len(sys.argv) > 1 and sys.argv[1] == '2' else main)()
