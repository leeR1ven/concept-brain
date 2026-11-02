#!/bin/bash
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1
H="h_gate=1 h_conf=1 h_relock=1 h_vbound=1 h_nocreate=1 h_level=1 h_denoise=1"
for cond in "2 0.0 0.0" "2 0.0 1.0" "16 0.004 0.0" "16 0.004 1.0" "16 0.02 0.0" "16 0.02 1.0" "64 0.02 0.0"; do
  set -- $cond
  [ -f results/w2/w2_v21_N$1_p$2_s$3.json ] || nice python3 -B w2_agent.py $1 2 $2 $3 30000 0 tag=w2_v21_N$1_p$2_s$3 > results/w2/log_v21_N$1_p$2_s$3.txt 2>&1
  nice python3 -B w2_agent.py $1 2 $2 $3 30000 0 tag=w2_H_N$1_p$2_s$3 $H > results/w2/log_H_N$1_p$2_s$3.txt 2>&1
done
echo W2DONE > results/w2/grid.done
