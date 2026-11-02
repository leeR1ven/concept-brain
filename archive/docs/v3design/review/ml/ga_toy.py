"""Sample cost of the §2.4c lineage loop for the 30-gene genome (P = 12, keep K = 4, G = 10, 40/40/20 + mutation
p 0.1 x exp N(0, 0.1), 6 pair-children + 6 pool-children).  Genes in log space z = ln(g / default).
True fitness: 6 effective genes, quadratic, optimum z* ~ N(0, 0.3) (defaults within ~30 % of the optimum);
24 fake genes.  Curvature a so that a 20 % miss on one gene costs c_20 noise SDs (c_20 = 0.25 / 0.5 / 1).
Observed score = true + N(0, 1) (one life, fresh world seed per generation).
Reported: true-fitness gain of the final best-scored individual over the defaults, in noise-SD units, and the
fraction of runs that end worse than the defaults.  200 runs per setting.
Run: python3 -I ga_toy.py
"""
import numpy as np, json

NG, NE = 30, 6


def ga(rng, a, zs, P=12, K=4, G=10, pool=True):
    f = lambda Z: -(a * (Z[:, :NE] - zs[None]) ** 2).sum(1)
    Z = rng.uniform(np.log(0.8), np.log(1.2), (P, NG))
    for g in range(G):
        obs = f(Z) + rng.standard_normal(P)
        top = Z[np.argsort(-obs)[:K]]
        kids = []
        for i in range(P):
            if i < P // 2 or not pool:
                pa, pb = top[rng.choice(K, 2, replace=False)]
                u = rng.random(NG)
                c = np.where(u < 0.4, pa, np.where(u < 0.8, pb, 0.5 * (pa + pb)))
            else:
                c = top.mean(0)
            m = rng.random(NG) < 0.1
            c = c + m * rng.normal(0, 0.1, NG)
            kids.append(c)
        Z = np.array(kids)
    obs = f(Z) + rng.standard_normal(P)
    best = Z[np.argmax(obs)]
    fake_drift = float(np.abs(best[NE:]).mean())
    return f(best[None])[0], fake_drift, float(f(Z).mean())


if __name__ == '__main__':
    res = {}
    for c20 in (0.25, 0.5, 1.0):
        a = c20 / np.log(1.2) ** 2
        gains, worse, drift, popm = [], [], [], []
        for run in range(200):
            rng = np.random.default_rng(run)
            zs = rng.normal(0, 0.3, NE)
            f0 = -(a * zs ** 2).sum()                 # the defaults
            fopt = 0.0
            fb, d, fm = ga(rng, a, zs)
            gains.append((fb - f0)); worse.append(fb < f0); drift.append(d); popm.append(fm - f0)
        res[c20] = dict(default_regret=float(np.mean([0])),
                        gain_mean=round(float(np.mean(gains)), 2), gain_sd=round(float(np.std(gains)), 2),
                        p_worse=round(float(np.mean(worse)), 2), pop_gain=round(float(np.mean(popm)), 2),
                        fake_gene_abs_logdrift=round(float(np.mean(drift)), 3))
        # regret of the defaults for reference (how much there is to gain at all)
        reg = np.mean([(a * np.random.default_rng(r).normal(0, 0.3, NE) ** 2).sum() for r in range(200)])
        res[c20]['default_regret'] = round(float(reg), 2)
        print(f'c20={c20}: defaults are {reg:.2f} SD below the optimum; GA (120 lives) gains {np.mean(gains):.2f} '
              f'± {np.std(gains):.2f} SD (best individual), population mean gain {np.mean(popm):.2f}; '
              f'ends worse than defaults in {np.mean(worse):.0%} of runs; fake genes drift |ln| {np.mean(drift):.3f}',
              flush=True)
    # power of the gene audit (born-wired 36.9): +-20 % change, detect an effect of d SD at alpha .05, power .8
    for d in (0.25, 0.5, 1.0):
        n = 2 * (1.96 + 0.84) ** 2 / d ** 2
        print(f'gene audit, unpaired: effect {d} SD needs {n:.0f} lives per arm -> 30 genes x 3 arms = '
              f'{30 * 3 * n:,.0f} lives ~ {30 * 3 * n * 1630 / 3600:,.0f} CPU-h')
    json.dump(res, open('ga_toy.json', 'w'), indent=1)
