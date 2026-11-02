#!/bin/bash
# symbolic regression (learning reviewer's gen_probe on the v21 core): variants x seeds, one process at a time
for s in 0 1 2; do
  python3 gen_probe_v21.py variant=base n=900 seed=$s out=gp_dev2.jsonl > /dev/null
  python3 gen_probe_v21.py variant=base n=900 seed=$s out=gp_dev2.jsonl ral=1 ral_cont=none > /dev/null
  python3 gen_probe_v21.py variant=base n=900 seed=$s out=gp_dev2.jsonl ral=1 > /dev/null
  python3 gen_probe_v21.py variant=base n=900 seed=$s out=gp_dev2.jsonl ral=1 ral_hip_own=1 > /dev/null
  python3 gen_probe_v21.py variant=base n=900 seed=$s out=gp_dev2.jsonl ral=1 ral_cont=rw > /dev/null
done
for s in 0 1 2; do
  python3 gen_probe_v21.py variant=base n=900 seed=$s tutor=mixed out=gp_dev2.jsonl > /dev/null
  python3 gen_probe_v21.py variant=Xcut n=900 seed=$s tutor=mixed out=gp_dev2.jsonl > /dev/null
  python3 gen_probe_v21.py variant=base n=900 seed=$s tutor=mixed out=gp_dev2.jsonl ral=1 > /dev/null
done
echo GPDONE
