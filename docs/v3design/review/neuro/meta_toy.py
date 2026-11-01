"""Neuro review toy 4 (local, < 100 MB, < 3 min): the designer's metaplasticity idea of 2026-11-01
("strong, often-used weights become harder to raise AND to lower, so a learned ability is kept for life") against
brain-derived alternatives, on GEN's learning problem.

Learner: GEN-like signed delta rule p = clip(sum_f w[f], 0, 1), dw = eta_eff (t - p)/|F| (bench R2), binary features.
Items: 64 questions, each = 2 unique (anchored) features + 1 of 4 shared frame features; 16 answer tokens.
Childhood: 32 OLD questions taught 8 times each, interleaved (salient: format + praise).
Adult life, 20,000 events; each event is one of
  - format item (p 0.05): one of 16 NEW-F questions taught with its answer, salient (praise follows);
  - overheard item (p 0.05): one of 16 NEW-O questions taught with its answer, NOT salient (no reward to me, W2);
  - drift (p_d): an OLD question gets a WRONG target in its own context, not salient:
        systematic: always the same wrong token per question (a structural error such as v2.2's give-up route);
        noisy: a random wrong token each time (mishearing, misattribution);
  - correction: at event 10,000 four OLD questions change their answer; each is re-taught (salient: 'buShi X, shi Y')
    once per 250 events until 12,000 (8 times) and never again;
  - otherwise nothing is learned (the agent answers itself; own output is not a target).
Rules (eta 0.35 base):
  R0 plain        eta
  R1 M2           eta * max(1/n_use, 0.1), n_use per synapse row (ARCH_v3 M2 as written)
  R2 designer     R1 / (1 + 20 |w|): strong weights harder to move both ways (applied to the signed weight)
  R3 cascade      Benna & Fusi 2016 chain u1 (expressed) - u2 - u3, C = 1, 4, 16; g12 0.002, g23 0.0002 per event
                  (untuned; learning on u1 with the R1 rate)
  R4 tag-capture  w = w_fast + w_late. Learning at the plain rate eta acts on w_fast, which decays with tau 2,000
                  events (early LTP; 2,000 dialogues ~ 1.5e5 ticks ~ 3 h at 67 ms per tick).
                  On a salient event the synapses just changed are captured: w_late += 0.5 w_fast, w_fast *= 0.5
                  (synaptic tagging and capture, Frey & Morris 1997; Redondo & Morris 2011; DA / novelty as the PRP
                  trigger, Lisman, Grace & Duzel 2011). w_late never decays and only changes by capture.
  R5 R4 + capture also when |w_fast| > 0.25 (strong / repeated induction makes its own PRPs; Frey & Morris 1997)
Measures (5 seeds): old items right at 10k / 20k; corrected items right at 12k and at 20k (8k events after the last
re-teaching); NEW-F and NEW-O right at 20k.
Run: python3 -I meta_toy.py
"""
import numpy as np, json

NQ, NOLD, NTOK, NFR = 64, 32, 16, 4
ETA = 0.35


def run(rule, drift, p_d, seed, T=20000, tau_fast=2000.0):
    rng = np.random.default_rng(seed)
    nfeat = 2 * NQ + NFR
    feats = [np.array([2 * q, 2 * q + 1, 2 * NQ + q % NFR]) for q in range(NQ)]
    ans = rng.integers(NTOK, size=NQ)
    wrong = (ans + 1 + rng.integers(NTOK - 1, size=NQ)) % NTOK
    U = [np.zeros((nfeat, NTOK)) for _ in range(3)]          # u1 / w_fast, u2 / w_late, u3
    nuse = np.zeros(nfeat)
    C = np.array([1.0, 4.0, 16.0]); g12, g23 = 0.002, 0.0002
    NEWF = list(range(NOLD, NOLD + 16)); NEWO = list(range(NOLD + 16, NQ))

    def expressed():
        return U[0] + U[1] if rule in ('R4', 'R5') else U[0]

    def pred(q, E=None):
        E = expressed() if E is None else E
        return np.clip(E[feats[q]].sum(0), 0, 1)

    def learn(q, tok, salient):
        F = feats[q]; p = pred(q); t = np.zeros(NTOK); t[tok] = 1
        err = (t - p) / len(F)
        for f in F:
            nuse[f] += 1
            eta = ETA if rule in ('R0', 'R4', 'R5') else ETA * max(1.0 / nuse[f], 0.1)
            dw = eta * err
            if rule == 'R2':
                dw = dw / (1 + 20.0 * np.abs(U[0][f]))
            U[0][f] = U[0][f] + dw
            if rule in ('R4', 'R5') and (salient or (rule == 'R5' and np.abs(U[0][f]).max() > 0.25)):
                U[1][f] += 0.5 * U[0][f]; U[0][f] *= 0.5

    def tick():
        if rule == 'R3':
            u1, u2, u3 = U
            f12 = g12 * (u1 - u2); f23 = g23 * (u2 - u3)
            U[0] = u1 - f12 / C[0]; U[1] = u2 + (f12 - f23) / C[1]; U[2] = u3 + f23 / C[2]
        elif rule in ('R4', 'R5'):
            U[0] *= np.exp(-1 / tau_fast)

    def right(qs, table):
        E = expressed()
        return float(np.mean([(np.argmax(pred(q, E)) == table[q]) and pred(q, E).max() > 0.3 for q in qs]))

    for rep in range(8):
        for q in rng.permutation(NOLD):
            learn(q, ans[q], True); tick()
    out = {'child': right(range(NOLD), ans)}
    corr = rng.choice(NOLD, 4, replace=False); unchanged = [q for q in range(NOLD) if q not in corr]
    new_ans = ans.copy()
    for ev in range(1, T + 1):
        r = rng.random()
        if r < 0.05:
            q = NEWF[int(rng.integers(16))]; learn(q, ans[q], True)
        elif r < 0.10:
            q = NEWO[int(rng.integers(16))]; learn(q, ans[q], False)
        elif r < 0.10 + p_d:
            q = int(rng.integers(NOLD))
            tok = wrong[q] if drift == 'systematic' else int((new_ans[q] + 1 + rng.integers(NTOK - 1)) % NTOK)
            learn(q, tok, False)
        if ev == 10000:
            out['old10k'] = right(unchanged, ans)
            for q in corr:
                new_ans[q] = (ans[q] + 3) % NTOK
        if 10000 < ev <= 12000 and ev % 250 == 0:
            for q in corr:
                learn(q, new_ans[q], True)
        if ev == 12000:
            out['corr12k'] = right(corr, new_ans)
        tick()
    out['old20k'] = right(unchanged, ans)
    out['corr20k'] = right(corr, new_ans)
    out['newF20k'] = right(NEWF, ans)
    out['newO20k'] = right(NEWO, ans)
    return out


if __name__ == '__main__':
    res = {}
    for drift, p_d in (('none', 0.0), ('noisy', 0.02), ('systematic', 0.02)):
        print(f'== drift {drift}, p_d {p_d} (about {int(20000 * p_d / NOLD)} wrong teachings per old item over life)')
        for rule in ('R0', 'R1', 'R2', 'R3', 'R4', 'R5'):
            rows = [run(rule, drift, p_d, s) for s in range(5)]
            agg = {k: round(float(np.mean([r[k] for r in rows])), 2) for k in rows[0]}
            res[f'{drift}/{rule}'] = agg
            print(f'   {rule}', '  '.join(f'{k} {v}' for k, v in agg.items()), flush=True)
    print('== tag-capture with a faster early component, tau_fast 300 events')
    for drift, p_d in (('none', 0.0), ('noisy', 0.02), ('systematic', 0.02)):
        for rule in ('R4', 'R5'):
            rows = [run(rule, drift, p_d, s, tau_fast=300.0) for s in range(5)]
            agg = {k: round(float(np.mean([r[k] for r in rows])), 2) for k in rows[0]}
            res[f'{drift}/{rule}/tau300'] = agg
            print(f'   {drift:10s} {rule}', '  '.join(f'{k} {v}' for k, v in agg.items()), flush=True)
    json.dump(res, open('meta_toy.json', 'w'), indent=1)
