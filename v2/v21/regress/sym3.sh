#!/bin/bash
# symbolic regression on the v21 core (onsets timed): 3 seeds x variants, recast and mixed tutors -> gp_v21.jsonl
cd "$(dirname "$0")"
for s in 0 1 2; do
  for kv in "" "ral=1" "ral=1 ral_other=1" "ral=1 ral_cont=none" "ral=1 ral_hip_own=1" "ral=1 ral_cont=rw"; do
    python3 gen_probe_v21.py variant=base n=900 seed=$s out=gp_v21.jsonl $kv > /dev/null
  done
  for kv in "" "ral=1" "ral=1 ral_other=1"; do
    python3 gen_probe_v21.py variant=base n=900 seed=$s tutor=mixed out=gp_v21.jsonl $kv > /dev/null
  done
  python3 gen_probe_v21.py variant=Xcut n=900 seed=$s tutor=mixed out=gp_v21.jsonl > /dev/null
done
echo SYMDONE
