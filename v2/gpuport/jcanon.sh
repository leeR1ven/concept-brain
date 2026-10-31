#!/bin/bash
cd "$(dirname "$0")"
mkdir -p out/jruns
for s in 0 1 2 3 4; do
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 run_j_canon.py tag=np_canon seed=$s w_eye_layout=retino rebind=1 ct=1 init_min_tau=10 hip_rho=2 ctx=0 out=out/jruns/np_canon_s$s.json > out/jruns/np_canon_s$s.log 2>&1
done
