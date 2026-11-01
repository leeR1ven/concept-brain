"""Research-05 setting: ear window = exactly one syllable (8 bands x 6 slices); different syllables' overlap per layer."""
import numpy as np
import sens
sens.TW = 6
for seed in (0, 1, 2):
    S = sens.Sounds(seed)
    sim = np.stack([S.window([(0, S.syl(c))]) for c in range(sens.NSYL)])
    simv = np.stack([S.window([(0, S.syl(c, 'self'))]) for c in range(sens.NSYL)])
    pa, pb = np.triu_indices(sens.NSYL, 1)
    for arm in ('dom1', 'eq4', 'eq7'):
        layers = sens.build(arm, seed)
        th = sens.calibrate(sens.encode(sim), layers)
        h0 = sens.forward(sens.encode(sim), layers, th); h1 = sens.forward(sens.encode(simv), layers, th)
        sil = sens.forward(sens.encode(np.zeros((1, 8, 6), int)), layers, th)
        jd = [round(float(sens.jacc(h0[l][pa], h0[l][pb]).mean()), 3) for l in range(5)]
        js = [round(float(sens.jacc(h0[l], h1[l]).mean()), 3) for l in range(5)]
        cnt = [(int(sil[l].sum()), int(h0[l].sum(1).mean())) for l in range(5)]
        print(seed, arm, 'thr J_diff', jd, 'J_sim', js, '(silence, syl) counts', cnt)
