#!/bin/bash
# sequential runs (one heavy process at a time).  FINAL = the main think configuration.
export OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
cd "$(cd "$(dirname "$0")" && pwd)"
FINAL="floor=0.3 policy=free ior=0 think_k=3 urg_gate=0 brake_now=1"
p() { echo "$1 $(date +%T)" >> out/batch1.progress; }
p start
nice python3 run_think.py seed=0 nA=300 nB=2000 nC=2000 nD=2000 stages=ABCD tag=final_s0 $FINAL save=1 > out/final_s0.stdout 2>&1; p final_s0
nice python3 eval_frozen.py out/think_final_s0_agent.pkl stages=BCD n=150 nD=200 seed=0 pols=budget:0,budget:1,budget:2,budget:3,free,v1 > out/frozen_final_s0.out 2>&1; p frozen_final
nice python3 eval_frozen.py out/think_free_s0_agent.pkl stages=BCD n=150 nD=200 seed=0 pols=budget:0,budget:1,budget:2,budget:3,budget:4,budget:8,free,v1 > out/frozen_ior_s0.out 2>&1; p frozen_ior
nice python3 run_life.py agent=dev seed=0 tutor=recast pre=30000 ticks=500000 tag=dev_s0 > out/life_dev_s0.stdout 2>&1; p life_dev
nice python3 run_life.py agent=think seed=0 tutor=recast pre=30000 ticks=500000 tag=think_s0 ior=0 think_k=3 urg_gate=0 > out/life_think_s0.stdout 2>&1; p life_think
nice python3 run_bench.py seed=1 nA=300 nB=2000 nC=2000 nD=2000 stages=ABCD tag=base_s1 floor=0.3 > out/base_s1.stdout 2>&1; p base_s1
nice python3 run_think.py seed=1 nA=300 nB=2000 nC=2000 nD=2000 stages=ABCD tag=final_s1 $FINAL > out/final_s1.stdout 2>&1; p final_s1
run() { nice python3 run_think.py seed=0 nA=100 nB=400 stages=AB tag=abl_$1 floor=0.3 policy=free ior=0 think_k=3 urg_gate=0 "${@:2}" > out/abl_$1.stdout 2>&1; }
run full brake_now=1
run no_reflex brake_now=0
run no_learned_brake brake_now=0 eta_oth=0
run reflex_only brake_now=1 eta_oth=0 tau_w=1000
run no_prior brake_now=0 tau_w=1000
run no_vgo brake_now=1 eta_go=0
run no_urgency brake_now=1 u_reply=0
p ablations
p ALL_DONE
