#!/bin/bash
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1
for f in rec/w2_N2_p0.0_s1.0_seed0.pkl rec/w2_N16_p0.02_s0.0_seed0.pkl rec/w2_N16_p0.004_s1.0_seed0.pkl rec/w2_N16_p0.02_s1.0_seed0.pkl rec/w2_N64_p0.02_s0.0_seed0.pkl; do
  nice python3 -B ear_eval.py $f gate "$@"
done
echo EVALDONE
