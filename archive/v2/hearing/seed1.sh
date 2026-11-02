#!/bin/bash
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1
for c in "2 2 0.0 0.0" "2 2 0.0 1.0" "16 4 0.004 0.0" "16 4 0.004 1.0" "16 4 0.02 1.0"; do set -- $c; nice python3 -B record.py $1 $2 $3 $4 1 1; done
for f in rec/w2_N2_p0.0_s0.0_seed1.pkl rec/w2_N2_p0.0_s1.0_seed1.pkl rec/w2_N16_p0.004_s0.0_seed1.pkl rec/w2_N16_p0.004_s1.0_seed1.pkl rec/w2_N16_p0.02_s1.0_seed1.pkl; do
  nice python3 -B ear_eval.py $f v21,gate g_backtrack=0 conf=1 relock=1 vbound=1 nocreate=1 level=1 denoise=1 tag=_final
done
echo S1DONE > results/seed1.done
