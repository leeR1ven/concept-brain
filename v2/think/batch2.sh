#!/bin/bash
# short ablations of the turn-taking parts (A100 + B400, seed 0): which mechanism keeps 0 overlaps
export OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
cd "$(cd "$(dirname "$0")" && pwd)"
run() { nice python3 run_think.py seed=0 nA=100 nB=400 stages=AB floor=0.3 policy=free tag=abl_$1 "${@:2}" > out/abl_$1.stdout 2>&1; }
run full
run no_learned_brake eta_oth=0
run no_prior tau_w=1000
run no_prior_no_learn tau_w=1000 eta_oth=0
run reflex_only tau_w=1000 eta_oth=0 brake_now=1
run reflex_full brake_now=1
run no_vgo eta_go=0
run no_urgency u_reply=0
echo ABL_DONE >> out/batch2.progress
