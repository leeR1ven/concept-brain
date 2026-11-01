"""X4b follow-up: sign-elected merge with the inherited part shrunk (x0.3, x0.1) at birth, plus the fresh birth again
(it must reproduce x4b.json exactly, since every seed is the same). Same scenarios, seeds and life as x4b.py.
Run: python3 -I x4b_extra.py [R] [n_seed]"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(1, os.path.join(HERE, '..', 'x345'))   # repo: the unchanged X4 harness (x4_merge.py)
import x4b as B                                            # noqa: E402
import x4_merge as X                                       # noqa: E402


def run(name, cfg, n_sib, seed, R):
    t0 = time.time()
    sh = cfg['sharp']
    lin = X.Lineage(seed_lin=1 + seed)
    rng = np.random.default_rng(50 + seed)
    S, own, C = X.make_lessons(rng, n_sib=n_sib)
    seeds = [100 * seed + 11 * (k + 1) for k in range(n_sib)]
    sibs = [X.raise_sib(lin, cfg, seeds[k], S + own[k] + [(q, a[k]) for q, a in C], cfg['n_rep'], cfg['n_fill'])
            for k in range(n_sib)]
    best = [ag.scores(q).max() for k, ag in enumerate(sibs) for q, a in S + own[k]]
    theta = 0.3 * float(np.median(best))
    ref = sibs[0]
    Ds = [B.to_ref(ref, ag) for ag in sibs]
    Bp = ref.B_pa
    Dsign = X.ties(Ds, cfg['trim'])
    del Ds
    mrng = np.random.default_rng(1900 + seed)
    Wsn = B.perturb(np.maximum(Bp + Dsign, 0.0).astype(np.float32), 0.1, mrng)     # sign + the same slight noise
    sets = {'shared': S, 'conflict': C}
    for k in range(n_sib):
        sets[f'only{k}'] = own[k]
    cur = S + sum(own, [])
    tests = {f'only{k}': own[k] for k in range(n_sib)}
    tests.update(shared=S, conflict=C)
    res = dict(scenario=name, seed=seed, theta=theta, day0={}, life={})
    for lam in (1.0, 0.3, 0.1):
        W = np.maximum(Bp + lam * (Wsn - Bp), 0.0).astype(np.float32)
        key = 'sign+noise' if lam == 1.0 else f'sign+noise x{lam}'
        ch = B.child_agent(lin, cfg, seeds[0], ref, W)
        res['day0'][key] = X.evaluate(B.scorer(ch), sets, theta, sh)
        res['life'][key] = B.live(ch, cur, R, 7000 + seed, tests, theta, sh)
        del ch, W
    fr = B.child_agent(lin, cfg, seeds[0], ref, ref.B_pa.copy())
    res['life']['fresh'] = B.live(fr, cur, R, 7000 + seed, tests, theta, sh)
    res['secs'] = round(time.time() - t0, 1)
    res['rss_mb'] = round(B.rss_mb())
    return res


if __name__ == '__main__':
    R = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    n_seed = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    out = []
    for seed in range(n_seed):
        for name, (ov, nsib) in B.SCEN.items():
            r = run(name, dict(X.BASE, **ov), nsib, seed, R)
            out.append(r)
            print(json.dumps(dict(scenario=name, seed=seed, secs=r['secs'], rss=r['rss_mb'])), flush=True)
            json.dump(out, open(os.path.join(HERE, 'x4b_extra.json'), 'w'))
