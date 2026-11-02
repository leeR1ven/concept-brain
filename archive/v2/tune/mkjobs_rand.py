# random configurations around a base: each draws k parameters from their ranges (Hyperband-style first rung)
# usage: python3 mkjobs_rand.py out.txt N k rngseed prefix "rung args" "base cfg args" [param1,param2,... to restrict]
import sys, numpy as np
SPACE = {
 'eta': ('log', 0.12, 0.8), 'eta_g': ('log', 0.2, 1.2), 'eta_critic': ('log', 0.05, 0.6), 'eta_actor': ('lin', 0.0, 0.6),
 'eta_prime': ('log', 0.005, 0.08), 'eta_attd': ('log', 0.005, 0.08), 'z_rate': ('log', 0.03, 0.3),
 'theta_z': ('lin', 0.7, 0.8), 'floor': ('lin', 0.15, 0.5),
 'match_thr': ('lin', 0.45, 0.8), 'match_n': ('int', 3, 12), 'rec_thr': ('lin', 0.65, 0.95), 'rf_thr': ('lin', 0.45, 0.8),
 'conf_min': ('int', 3, 10), 'rate_max': ('lin', 0.25, 0.6), 'kappa': ('log', 0.7, 4.0),
 'match_min': ('lin', 0.35, 0.65), 'match_gap': ('lin', 0.15, 0.45), 'att_thr': ('lin', 0.3, 0.8),
 'hip_theta': ('lin', 0.1, 0.4), 'gain': ('lin', 0.6, 1.5),
 'ctx_lam': ('lin', 0.7, 0.97), 'ctx_beta': ('lin', 0.0, 4.0), 'ctx_tau': ('log', 50, 800), 'w_ctx': ('lin', 0.0, 1.0),
 'hip_gamma': ('cat', [0.999, 0.9995, 0.9998, 0.9999]), 'hip_novelty': ('cat', [1.0, 1.0, 0.9, 0.8]),
 'recruit_p': ('cat', [1.0, 1.0, 0.8, 0.6]),
 'mature': ('cat', [0.0, 20.0, 50.0, 100.0, 200.0]), 'gen_mmin': ('cat', [0.02, 0.05, 0.1, 0.2]),
 'mat_norm': ('cat', [0.0, 0.5, 1.0, 1.0]),
 'wd_gen': ('cat', [0.0, 0.0, 0.1, 0.3, 1.0]),
 'recall_n': ('cat', [0, 4, 8, 16]), 'eta_recall': ('log', 0.05, 0.4), 'rec_pow': ('cat', [0.0, 0.5, 1.0]),
 'rec_boost': ('cat', [0.0, 0.0, 0.05, 0.2]),
 'split_use': ('cat', [0, 0, 100, 300, 1000]), 'split_err': ('lin', 0.3, 0.7), 'split_n': ('cat', [3, 5, 10, 30]),
}
out, N, k, rs, prefix, rung, base = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5], sys.argv[6], sys.argv[7]
allow = sys.argv[8].split(',') if len(sys.argv) > 8 and sys.argv[8] else list(SPACE)
rng = np.random.default_rng(rs)
def draw(p):
    s = SPACE[p]
    if s[0] == 'log':
        return float(np.exp(rng.uniform(np.log(s[1]), np.log(s[2]))))
    if s[0] == 'lin':
        return float(rng.uniform(s[1], s[2]))
    if s[0] == 'int':
        return int(rng.integers(s[1], s[2] + 1))
    return s[1][int(rng.integers(len(s[1])))]
bd = dict(a.split('=', 1) for a in base.split())
lines = []
for i in range(N):
    d = dict(bd)
    for p in rng.choice(allow, size=min(k, len(allow)), replace=False):
        v = draw(p)
        d[p] = (f"{v:.4g}" if isinstance(v, float) else str(v))
    if 'wd_gen' in d and 'wd_gate' not in bd:
        d['wd_gate'] = d['wd_gen']                        # one global decay over both paths
    if d.get('mature', '0') in ('0', '0.0'):
        for q in ('gen_mmin', 'mat_norm'):
            d.pop(q, None)
    if d.get('recall_n', '0') == '0':
        for q in ('eta_recall', 'rec_pow', 'rec_boost'):
            d.pop(q, None)
    if d.get('split_use', '0') == '0':
        for q in ('split_err', 'split_n'):
            d.pop(q, None)
    args = " ".join(f"{a}={b}" for a, b in sorted(d.items()))
    lines.append(f"tag={prefix}{i:02d} seed=0 {rung} {args}")
open(out, 'w').write("\n".join(lines) + "\n")
print(len(lines), "jobs ->", out)
