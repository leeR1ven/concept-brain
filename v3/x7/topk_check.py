"""Explain research 05's 0.24: same harness, but per-stimulus top-k (competition) instead of the per-neuron threshold."""
import numpy as np
import sens
for seed in (0, 1, 2):
    S = sens.Sounds(seed)
    sim = np.stack([S.window([(1, S.syl(c))]) for c in range(sens.NSYL)])
    simv = np.stack([S.window([(1, S.syl(c, 'self'))]) for c in range(sens.NSYL)])
    pa, pb = np.triu_indices(sens.NSYL, 1)
    for arm in ('dom1', 'eq7'):
        layers = sens.build(arm, seed)
        out = []
        for X in (sim, simv):
            h = sens.encode(X); hs = [h]
            for L, tg in zip(layers, sens.TARGET):
                r = sens.ratio(h, L) * L['base'][None]            # raw score
                r = r + 1e-6 * np.random.default_rng(1).random(r.shape[1])[None]   # fixed tie-break
                k = int(tg * r.shape[1])
                kth = np.partition(r, -k, axis=1)[:, -k][:, None]
                h = r >= kth; hs.append(h)
            out.append(hs)
        jd = [round(float(sens.jacc(out[0][l][pa], out[0][l][pb]).mean()), 3) for l in range(5)]
        js = [round(float(sens.jacc(out[0][l], out[1][l]).mean()), 3) for l in range(5)]
        print(seed, arm, 'top-k  J_diff', jd, ' J_sim(voice)', js)
