#!/bin/bash
# ear-alone recordings: 1 day per condition, seed 0; listeners K=2 (N=2) or 4
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1
for s in 0.0 0.5 1.0; do
  nice python3 -B record.py 2 2 0.0 $s 1 0
  nice python3 -B record.py 16 4 0.02 $s 1 0
  nice python3 -B record.py 64 4 0.02 $s 1 0
done
nice python3 -B record.py 2 2 0.0 2.0 1 0
nice python3 -B record.py 16 4 0.004 0.0 1 0
nice python3 -B record.py 16 4 0.004 1.0 1 0
echo RECDONE
