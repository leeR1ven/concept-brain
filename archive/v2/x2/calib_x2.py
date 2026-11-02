# -*- coding: utf-8 -*-
"""x2: calibrate the stability rule on a trained agent saved with stab_track=1 (use traces kept, steps unchanged).
Prints the distribution of u, |w| and u*|w| over GEN synapses that took part in learning, and the step factor
f = max(1/(1 + (beta/w_ref) u |w|), f_min) for candidate strengths.  usage: python3 calib_x2.py ck/x_tr.pkl.gz"""
import sys, os
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import drift_x2

o = drift_x2.load_pickle(sys.argv[1])
core = o['agent'].core
for name, R in (('GEN', core.gen), ('gate', core.gate), ('gateR', core.gateR)):
    if R is None or R.U is None:
        continue
    n = core.mix.n
    W, U = R.W[:n], R.U[:n]
    m = U > 0
    aw, u = np.abs(W[m]), U[m]
    uw = u * aw
    q = lambda x: ' '.join(f"p{p}={np.percentile(x, p):.4g}" for p in (10, 50, 90, 99))
    print(f"{name}: synapses used {m.sum()} of {m.size} ({m.mean():.3f}); rows used {(m.any(1)).sum()}")
    print(f"  u   {q(u)}  max {u.max():.0f}")
    print(f"  |w| {q(aw)}  max {aw.max():.4g}")
    print(f"  u|w| {q(uw)}")
    strong = aw >= np.percentile(aw, 90)
    print(f"  strong (|w| >= p90 {np.percentile(aw, 90):.4g}): u {q(u[strong])} | u|w| {q(uw[strong])}")
    for k in (0.1, 0.3, 1, 3, 10, 30, 100):     # k = beta / w_ref
        f = np.maximum(1 / (1 + k * uw), 0.05)
        print(f"  beta/w_ref={k:>6}: f all {q(f)} | strong mean {f[strong].mean():.3f} | at floor {np.mean(f <= 0.0500001):.3f}")
mx = core.mix
print('layer: ev', mx.ev, 'pruned', mx.stats['pruned'], 'regrown', mx.stats['regrown'], 'fires p50/p90',
      np.percentile(mx.fires, 50), np.percentile(mx.fires, 90), 'ever', mx.ever.mean())
print('learn events', core.stats['learn_events'], 'awake recalls', core.stats['awake_recalls'], 'hip', core.hip.n, core.hip.clock)
