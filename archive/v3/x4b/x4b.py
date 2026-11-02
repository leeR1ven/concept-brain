"""X4b: the designer's own merge rule (IDEAS #19) in the X4 harness, plus the 'offspring keeps living' head-start test.

Reuses x4_merge.py unchanged (copied here): the same lineage, agents, rule, lessons, raising, readout and metrics.

Operators. They act on the learned deltas D = w - birth, in the reference sibling's index space. With a shared birth
wiring, picking or averaging deltas equals picking or averaging the total weights, because every parent's total is
birth + D_k and the rule keeps birth + D_k >= 0.
  mean, sign            : X4's reference rows (plain mean; TIES sign election, trim 10 %)
  cross                 : per connection, with probability p_mean = 0.2 the parents' mean, otherwise one parent drawn
                          uniformly. Two parents: 40 / 40 / 20 (IDEAS #19). Pool of n: 0.8/n each + 0.2 mean.
  pick                  : p_mean = 0 (pure per-connection uniform crossover); sensitivity row
  cross+noise(s)        : cross, then every PFC -> AUD connection of the child (birth + merged) is multiplied by
                          (1 + s * N(0,1)) and clipped at 0
  cross+noise_all(s)    : as above, plus the same perturbation on the child's AUD -> PFC and PFC -> PFC birth links
Offspring (IDEAS #19, 2026-11-01 addendum): the merged (and perturbed) PFC -> AUD weights become the child's BIRTH
  links. They do not decay (ARCHITECTURE.md 2.8). Its learned part starts at 0. Variant 'cross/learned': the inherited
  part stays a learned part and keeps decaying.
  'cross+noise xL': the merged learned part (cross+noise(0.1) minus birth) is multiplied by L before it becomes the
  child's birth links, i.e. one strong pass of the existing global decay at birth ("an overall rough memory").
Life after birth: in each round the agent hears every non-conflicting lesson of the parents (shared + every disjoint
  lesson) once, plus len/2 filler lines,
  shuffled. All agents of a scenario-seed get the same order. After each round the tests run without learning.
  Compared: the merged children, a fresh unmerged birth (same structure as the reference sibling), and parents 0 and 1
  hearing the same curriculum (their new lessons are the other parents' lessons).
Run: python3 -I x4b.py [R] [n_seed] [scenario,...]
"""
import copy
import json
import os
import resource
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(1, os.path.join(HERE, '..', 'x345'))   # repo: the unchanged X4 harness (x4_merge.py)
import x4_merge as X                                      # noqa: E402
from x4_merge import N_AX, V, Agent, Lineage, answer, ties, make_lessons, raise_sib, BASE   # noqa: E402


def rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def pats(ag):
    if getattr(ag, '_M', None) is None:
        M = np.zeros((V, N_AX), np.float32)
        for c in range(V):
            p = ag.pattern(c)
            M[c, p] = 1.0 / max(1, len(p))
        ag._M = M
    return ag._M


def scorer(ag, B=None, D=None):
    """batch form of Agent.scores (twolayer): the same formula, with W and the strengths computed once"""
    B = ag.B_pa if B is None else B
    D = ag.D_pa if D is None else D
    W = np.maximum(B + D, 0.0)
    so, si = W.sum(1) + 1e-6, W.sum(0) + 1e-6
    Wn = (W / np.sqrt(so)[:, None]) / np.sqrt(si)[None, :]
    M = pats(ag)

    def f(q):
        x = ag.code(q)[-1]
        return M @ Wn[x].sum(0)
    return f


def to_ref(ref, ag):
    """sibling's learned PFC -> AUD part in the reference's AUD index space (stable IDs by ID; X4's non-aligned map)"""
    if ag is ref:
        return ag.D_pa.copy()
    am = X.aud_map(ref, ag, False)
    D = np.zeros_like(ag.D_pa)
    ok = am >= 0
    D[:, am[ok]] = ag.D_pa[:, ok]
    return D


def cross(Ds, rng, p_mean=0.2):
    D = np.stack(Ds)
    n = len(Ds)
    u = rng.random(D.shape[1:], dtype=np.float32)
    k = np.clip(((u - p_mean) / (1.0 - p_mean) * n).astype(np.int8), 0, n - 1)
    out = np.take_along_axis(D, k[None].astype(np.int64), 0)[0]
    return np.where(u < p_mean, D.mean(0), out).astype(np.float32)


def perturb(W, s, rng):
    return np.maximum(W * (1.0 + s * rng.standard_normal(W.shape, dtype=np.float32)), 0.0).astype(np.float32)


def child_agent(lin, cfg, ref_seed, ref, W_pa, inherit='birth', s_all=0.0, rng=None):
    ch = Agent(lin, cfg, ref_seed)                           # same birth structure as the reference sibling
    assert np.array_equal(ch.dead, ref.dead) and np.array_equal(ch.B_ap, ref.B_ap)
    if inherit == 'birth':
        ch.B_pa = W_pa.astype(np.float32)
    else:
        ch.D_pa = (W_pa - ch.B_pa).astype(np.float32)
    if s_all > 0:
        ch.B_ap = perturb(ch.B_ap, s_all, rng)
        ch.mask_ap = ch.B_ap > 0
        L2 = copy.copy(lin)
        L2.B_pp = perturb(lin.B_pp, s_all, rng)
        ch.L = L2
    ch.D_ap = np.zeros((1, 1), np.float32)                   # unused (fixed near-input links)
    ch.thr.theta = ref.thr.theta
    ch.rng = np.random.default_rng(ref_seed + 555)
    return ch


def live(ag, cur, R, seed, tests, theta, sh):
    rng = np.random.default_rng(seed)
    hist = {k: [] for k in tests}

    def test():
        f = scorer(ag)
        for k, items in tests.items():
            row = []
            for q, a in items:
                got = answer(f(q), theta, sh)
                row.append((got is None) if isinstance(a, list) else (got == a))
            hist[k].append(row)
    test()
    nf = len(cur) // 2
    for r in range(R):
        items = [list(q) + [a] for q, a in cur] + [list(rng.choice(V, 3, replace=False)) for _ in range(nf)]
        for i in rng.permutation(len(items)):
            ag.hear([int(v) for v in items[i]])
        test()
    out = {}
    for k, v in hist.items():
        h = np.array(v, bool)                                # (R + 1, n_items)
        first = np.where(h.any(0), h.argmax(0), R + 1)
        suff = np.flip(np.cumprod(np.flip(h, 0), 0), 0).astype(bool)
        stable = np.where(suff.any(0), suff.argmax(0), R + 1)
        out[k] = dict(curve=[round(float(x), 3) for x in h.mean(1)], first=first.tolist(), stable=stable.tolist())
    return out


def run_scenario(name, cfg, n_sib, seed, R, life=True):
    t0 = time.time()
    sh = cfg['sharp']
    lin = Lineage(seed_lin=1 + seed)
    rng = np.random.default_rng(50 + seed)
    S, own, C = make_lessons(rng, n_sib=n_sib)
    seeds = [100 * seed + 11 * (k + 1) for k in range(n_sib)]
    sibs = [raise_sib(lin, cfg, seeds[k], S + own[k] + [(q, a[k]) for q, a in C], cfg['n_rep'], cfg['n_fill'])
            for k in range(n_sib)]
    best = [ag.scores(q).max() for k, ag in enumerate(sibs) for q, a in S + own[k]]
    theta = 0.3 * float(np.median(best))
    res = dict(scenario=name, seed=seed, theta=theta, parents={}, day0={}, life={})
    for k, ag in enumerate(sibs):
        sets = {'shared': S, 'own': own[k], 'conflict_own': [(q, a[k]) for q, a in C]}
        res['parents'][k] = X.evaluate(scorer(ag), sets, theta, sh)
        f = scorer(ag)
        mg = []
        for q, a in own[k]:                                  # the owning parent's margin: right drive / best other
            sc = f(q)
            mg.append(round(float(sc[a] / max(1e-9, np.max(np.delete(sc, a)))), 3))
        res.setdefault('margin', {})[k] = mg
    sets = {'shared': S, 'conflict': C}
    for k in range(n_sib):
        sets[f'only{k}'] = own[k]
    ref = sibs[0]
    Ds = [to_ref(ref, ag) for ag in sibs]
    B = ref.B_pa
    mrng = np.random.default_rng(900 + seed)
    W = {'mean': np.maximum(B + np.mean(np.stack(Ds), 0), 0.0).astype(np.float32),
         'sign': np.maximum(B + ties(Ds, cfg['trim']), 0.0).astype(np.float32),
         'cross': np.maximum(B + cross(Ds, mrng, 0.2), 0.0),
         'pick': np.maximum(B + cross(Ds, mrng, 0.0), 0.0)}
    del Ds
    # day 0: every operator, child born with the merged links as birth links
    plan = [(op, W[op], 0.0) for op in ('mean', 'sign', 'cross', 'pick')]
    for s in (0.03, 0.1, 0.3):
        Wn = perturb(W['cross'], s, mrng)
        if s == 0.1:
            W['cross+noise'] = Wn
        plan.append((f'cross+noise({s})', Wn, 0.0))
        plan.append((f'cross+noise_all({s})', Wn, s))
    for op, Wop, s_all in plan:
        ch = child_agent(lin, cfg, seeds[0], ref, Wop, s_all=s_all, rng=mrng)
        res['day0'][op] = X.evaluate(scorer(ch), sets, theta, sh)
        if s_all > 0:                                        # how much the extra birth-link noise moves the codes
            jac = []
            for q, a in S + own[0]:
                xa, xb = ref.code(list(q))[-1], ch.code(list(q))[-1]
                jac.append(len(np.intersect1d(xa, xb)) / max(1, len(np.union1d(xa, xb))))
            res['day0'][op]['code_jaccard_vs_ref'] = round(float(np.mean(jac)), 3)
        del ch
    del plan
    print(json.dumps(dict(scenario=name, seed=seed, stage='day0', secs=round(time.time() - t0, 1),
                          rss=round(rss_mb()), day0={k: v for k, v in res['day0'].items()})), flush=True)
    if life:
        cur = S + sum(own, [])                                # every non-conflicting lesson of the parents
        tests = {f'only{k}': own[k] for k in range(n_sib)}
        tests.update(shared=S, conflict=C)
        lseed = 7000 + seed
        for lam in (0.3, 0.1):                               # born after one strong pass of the global decay
            W[f'cross+noise x{lam}'] = np.maximum(B + lam * (W['cross+noise'] - B), 0.0).astype(np.float32)
        agents = [(op, 'birth', 0.0) for op in ('mean', 'sign', 'cross', 'cross+noise')]
        agents += [('cross+noise_all', 'birth', 0.1), ('cross+noise x0.3', 'birth', 0.0), ('cross+noise x0.1', 'birth', 0.0)]
        if name == 'fixed_in':
            agents.append(('cross', 'learned', 0.0))
        lrng = np.random.default_rng(950 + seed)
        for op, inh, s_all in agents:
            ch = child_agent(lin, cfg, seeds[0], ref, W[op if s_all == 0 else 'cross+noise'], inherit=inh,
                             s_all=s_all, rng=lrng)
            key = op if inh == 'birth' else op + '/learned'
            res['life'][key] = live(ch, cur, R, lseed, tests, theta, sh)
            del ch
        fr = child_agent(lin, cfg, seeds[0], ref, ref.B_pa.copy())     # fresh birth: lineage birth links only
        res['life']['fresh'] = live(fr, cur, R, lseed, tests, theta, sh)
        del fr
        W.clear()
        for k in range(min(2, n_sib)):                        # the parent continues its own life on the same curriculum
            res['life'][f'parent{k}'] = live(sibs[k], cur, R, lseed, tests, theta, sh)
    res['secs'] = round(time.time() - t0, 1)
    res['rss_mb'] = round(rss_mb())
    return res


SCEN = {'fixed_in': (dict(trim=0.1), 2),
        'fixed_in+stable_ids': (dict(trim=0.1, stable=True), 2),
        'pool4': (dict(trim=0.1), 4)}

if __name__ == '__main__':
    R = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    n_seed = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    names = sys.argv[3].split(',') if len(sys.argv) > 3 else list(SCEN)
    tag = sys.argv[4] if len(sys.argv) > 4 else 'x4b'
    out = []
    for seed in range(n_seed):
        for name in names:
            ov, nsib = SCEN[name]
            r = run_scenario(name, dict(BASE, **ov), nsib, seed, R)
            out.append(r)
            print(json.dumps(dict(scenario=name, seed=seed, stage='done', secs=r['secs'], rss=r['rss_mb'])), flush=True)
            json.dump(out, open(os.path.join(HERE, tag + '.json'), 'w'))
