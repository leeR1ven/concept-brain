# rung 2 (A300 + 3 x 500, frozen B/C/D probes after each stage, C drift 300): stability mechanisms + curriculum
import sys
base = "R=3 nS=500 evaln=80 drift=300"
MAT = "mature=50 gen_mmin=0.05 mat_norm=1"
BIG = "hip_cap=10000 hip_ndg=6000 hip_kdg=120"
REC = "recall_n=8 eta_recall=0.2 rec_pow=0"
scan = [
 ("inter", "cur=inter"),
 ("mixA", "cur=mix b2=1 c2=3 b3=1 c3=1 d3=4"),
 ("mixB", "cur=mix b2=1 c2=1 b3=1 c3=1 d3=2"),
 ("mat50", "mature=50 gen_mmin=0.05"),
 ("mat50nu1", MAT),
 ("mat200nu1", "mature=200 gen_mmin=0.05 mat_norm=1"),
 ("mat50nu1g", MAT + " mature_g=50 gate_mmin=0.05"),
 ("split100", MAT + " split_use=100 split_err=0.5 split_n=5"),
 ("split300", MAT + " split_use=300 split_err=0.5 split_n=5"),
 ("split1000", MAT + " split_use=1000 split_err=0.5 split_n=5"),
 ("split300nomat", "split_use=300 split_err=0.5 split_n=5"),
 ("rec8", REC),
 ("rec8p1", "recall_n=8 eta_recall=0.2 rec_pow=1"),
 ("rec8big", REC + " " + BIG),
 ("rec16big", "recall_n=16 eta_recall=0.2 rec_pow=0 " + BIG),
 ("rec8bigboost", REC + " rec_boost=0.1 " + BIG),
 ("rec8bige35", "recall_n=8 eta_recall=0.35 rec_pow=0 " + BIG),
 ("big", BIG),
 ("hg0.9999", "hip_gamma=0.9999"),
 ("hnov0.8", "hip_novelty=0.8"),
 ("wd0.3", "wd_gen=0.3 wd_gate=0.3"),
 ("wd1mat", MAT + " wd_gen=1 wd_gate=1"),
 ("combo1", REC + " " + BIG + " " + MAT + " split_use=300 split_err=0.5 split_n=5"),
 ("combo2", "cur=inter " + REC + " " + BIG + " " + MAT + " split_use=300 split_err=0.5 split_n=5"),
 ("combo3", REC + " " + BIG + " " + MAT),
 ("combo4", "cur=inter " + REC + " " + BIG),
]
lines = [f"tag=r2_base seed={s} {base}" for s in (0, 1)]
lines += [f"tag=r2_{t} seed=0 {base} {a}" for t, a in scan]
open(sys.argv[1], "w").write("\n".join(lines) + "\n")
print(len(lines), "jobs")
