#!/bin/bash
# BIGGPU: the only GPU job; up to 36 GB
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 X10_GPU_FRAC=0.88
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv
for s in 8 12 16; do
  echo "=== timing s=$s"; python3 timing.py --s $s --ticks 1000 --prof_ticks 100 --out timing_s$s.json; echo "exit $?"
done
