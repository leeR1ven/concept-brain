#!/bin/bash
# FINAL2 = the main think configuration (hard transition window for replies; urgency 0.15 unless the line reacts
# to my own utterance; reflex).  One heavy process at a time.
export OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
cd "$(cd "$(dirname "$0")" && pwd)"
F2="floor=0.3 policy=free ior=0 think_k=3 urg_gate=0 brake_now=1 late_release=0 u_reply=0.15 u_R1=0"
p() { echo "$1 $(date +%T)" >> out/batch3.progress; }
p start
nice python3 run_think.py seed=0 nA=300 nB=2000 nC=2000 nD=2000 stages=ABCD tag=final2_s0 $F2 save=1 > out/final2_s0.stdout 2>&1; p final2_s0
nice python3 run_life.py agent=think seed=0 tutor=recast pre=30000 ticks=500000 tag=think2_s0 ior=0 think_k=3 urg_gate=0 late_release=0 u_reply=0.15 u_R1=0 > out/life_think2_s0.stdout 2>&1; p life_think2
nice python3 run_life.py agent=dev seed=0 tutor=recast pre=30000 ticks=500000 tag=dev_s0 > out/life_dev_s0.stdout 2>&1; p life_dev
nice python3 run_bench.py seed=1 nA=300 nB=2000 nC=2000 nD=2000 stages=ABCD tag=base_s1 floor=0.3 > out/base_s1.stdout 2>&1; p base_s1
nice python3 run_think.py seed=1 nA=300 nB=2000 nC=2000 nD=2000 stages=ABCD tag=final2_s1 $F2 > out/final2_s1.stdout 2>&1; p final2_s1
nice python3 eval_frozen.py out/think_free_s0_agent.pkl stages=BCD n=150 nD=200 seed=0 pols=budget:3,budget:4,budget:6,budget:8,free > out/frozen_ior_s0.out 2>&1; p frozen_ior
run() { nice python3 run_think.py seed=0 nA=100 nB=400 stages=AB tag=abl_$1 floor=0.3 policy=free ior=0 think_k=3 urg_gate=0 "${@:2}" > out/abl_$1.stdout 2>&1; }
run full2 brake_now=1 late_release=0 u_reply=0.15 u_R1=0
run no_window brake_now=1 late_release=1 u_reply=0.15 u_R1=0
run no_reflex_no_window brake_now=0 late_release=1 u_reply=0.15 u_R1=0
run no_learned_brake brake_now=1 late_release=0 u_reply=0.15 u_R1=0 eta_oth=0
run no_vgo brake_now=1 late_release=0 u_reply=0.15 u_R1=0 eta_go=0
run no_urgency brake_now=1 late_release=0 u_reply=0 u_R1=0
run no_R_gate brake_now=1 late_release=0 u_reply=0.15 u_R1=1
p ablations
p ALL_DONE
