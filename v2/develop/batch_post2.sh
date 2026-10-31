#!/bin/bash
export OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
cd "$(dirname "$0")"
until [ -f out/batch_seed1.done ]; do sleep 10; done
nice python3 diag_gate.py out/st_base_s0_agent.pkl > out/diag_gate.txt 2>&1
nice python3 drift_eta.py out/st_base_s0_agent.pkl > out/drift_eta.txt 2>&1
M1_ONLY="hippo, no sleep" nice python3 test_modules.py M1 > out/m1_nosleep.txt 2>&1
echo DONE > out/batch_post2.done
