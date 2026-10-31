#!/bin/bash
export OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
cd "$(dirname "$0")"
nice python3 diag_gate.py out/st_base_s0_agent.pkl > out/diag_gate.txt 2>&1
nice python3 drift_eta.py out/st_base_s0_agent.pkl > out/drift_eta.txt 2>&1
M1_ONLY="hippo, no sleep,hippo+sleep,hippo+priority sleep" nice python3 test_modules.py M1 > out/m1_sleep.txt 2>&1
nice python3 run_bench.py seed=0 nA=150 nB=700 stages=AB tag=rot_reset floor=0.3 own_rotate=1 dialogue_reset=1 > /dev/null 2>&1
nice python3 run_bench.py seed=0 nA=150 nB=700 stages=AB tag=norot_reset floor=0.3 own_rotate=0 dialogue_reset=1 > /dev/null 2>&1
echo POST_DONE > out/batch_post.done
