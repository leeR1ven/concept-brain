# rung 1 (A300 + B500, frozen B probes, B drift): one-at-a-time scan around the develop configuration
base = "R=1 nS=500 evaln=80 drift=150"
scan = [
 ("eta0.15", "eta=0.15"), ("eta0.25", "eta=0.25"), ("eta0.5", "eta=0.5"), ("eta0.7", "eta=0.7"),
 ("etag0.25", "eta_g=0.25"), ("etag1.0", "eta_g=1.0"),
 ("etac0.05", "eta_critic=0.05"), ("etac0.5", "eta_critic=0.5"),
 ("etaa0", "eta_actor=0"), ("etaa0.6", "eta_actor=0.6"),
 ("etap0.005", "eta_prime=0.005"), ("etap0.06", "eta_prime=0.06"),
 ("etad0.005", "eta_attd=0.005"), ("etad0.06", "eta_attd=0.06"),
 ("zr0.03", "z_rate=0.03"), ("zr0.3", "z_rate=0.3"),
 ("thz0.7", "theta_z=0.7"), ("thz0.8", "theta_z=0.8"),
 ("floor0.2", "floor=0.2"), ("floor0.4", "floor=0.4"), ("floor0.5", "floor=0.5"),
 ("mthr0.45", "match_thr=0.45"), ("mthr0.75", "match_thr=0.75"),
 ("mn3", "match_n=3"), ("mn10", "match_n=10"),
 ("rthr0.7", "rec_thr=0.7"), ("rthr0.95", "rec_thr=0.95"),
 ("rf0.45", "rf_thr=0.45"), ("rf0.75", "rf_thr=0.75"),
 ("cmin3", "conf_min=3"), ("cmin10", "conf_min=10"),
 ("rmax0.25", "rate_max=0.25"), ("rmax0.6", "rate_max=0.6"),
 ("kap1", "kappa=1.0"), ("kap4", "kappa=4.0"),
 ("mmin0.35", "match_min=0.35"), ("mmin0.65", "match_min=0.65"),
 ("mgap0.15", "match_gap=0.15"), ("mgap0.45", "match_gap=0.45"),
 ("athr0.3", "att_thr=0.3"), ("athr0.8", "att_thr=0.8"),
 ("clam0.75", "ctx_lam=0.75"), ("clam0.97", "ctx_lam=0.97"),
 ("cbeta0", "ctx_beta=0"), ("cbeta4", "ctx_beta=4"),
 ("ctau50", "ctx_tau=50"), ("ctau800", "ctx_tau=800"),
 ("wctx0", "w_ctx=0"), ("wctx0.5", "w_ctx=0.5"),
 ("hg0.999", "hip_gamma=0.999"), ("hg0.9999", "hip_gamma=0.9999"),
 ("hnov0.8", "hip_novelty=0.8"), ("hnov0.6", "hip_novelty=0.6"),
 ("rp0.5", "recruit_p=0.5"),
 ("mat50", "mature=50 gen_mmin=0.05"), ("mat200", "mature=200 gen_mmin=0.05"),
 ("mat50nu1", "mature=50 gen_mmin=0.05 mat_norm=1"), ("mat200nu1", "mature=200 gen_mmin=0.05 mat_norm=1"),
 ("wd0.3", "wd_gen=0.3 wd_gate=0.3"), ("wd2", "wd_gen=2 wd_gate=2"),
 ("rec4", "recall_n=4 eta_recall=0.2 rec_pow=1"), ("rec4p0", "recall_n=4 eta_recall=0.2 rec_pow=0"),
 ("rec16", "recall_n=16 eta_recall=0.1 rec_pow=0.5"),
 ("split", "split_use=200 split_err=0.5 split_n=5 mature=50 gen_mmin=0.05 mat_norm=1"),
 ("prune", "prune_age=30000 prune_every=5000"),
]
lines = [f"tag=r1_base seed={s} {base}" for s in (0, 1, 2)]
lines += [f"tag=r1_{t} seed=0 {base} {a}" for t, a in scan]
import sys
open(sys.argv[1], "w").write("\n".join(lines) + "\n")
print(len(lines), "jobs")
