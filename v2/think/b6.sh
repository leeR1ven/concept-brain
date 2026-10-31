#!/bin/bash
export OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
cd "$(cd "$(dirname "$0")" && pwd)"
until grep -q ALL_DONE out/b5.progress; do sleep 5; done
L4="ior=0 think_k=3 urg_gate=0 late_release=0 u_reply=0.15 u_R1=1 own_tail=1"
nice python3 run_life.py agent=think seed=0 tutor=recast pre=30000 ticks=500000 tag=think5_s0 $L4 > out/life_think5_s0.stdout 2>&1
echo life_think5 $(date +%T) >> out/b6.progress
nice python3 run_life.py agent=dev seed=0 tutor=recast pre=30000 ticks=500000 tag=dev5_s0 > out/life_dev5_s0.stdout 2>&1
echo life_dev5 $(date +%T) >> out/b6.progress
echo ALL_DONE >> out/b6.progress
