#!/bin/bash
# v2.2 symbolic regression, one process at a time: canonical tie rule (MIX) and c_flat, recast and mixed tutors
CF="rm=1 rm_train=1 rm_prune=1 rm_age=20000 rm_regrow=1 rm_alloc=flat"
for s in 0 1 2; do
  python3 gen_probe_v22.py variant=base n=900 seed=$s ral=1 ral_other=1 > /dev/null
  python3 gen_probe_v22.py variant=base n=900 seed=$s ral=1 ral_other=1 $CF > /dev/null
  python3 gen_probe_v22.py variant=base n=900 seed=$s tutor=mixed ral=1 ral_other=1 > /dev/null
  python3 gen_probe_v22.py variant=base n=900 seed=$s tutor=mixed ral=1 ral_other=1 $CF > /dev/null
  OUTJ=gen_probe2_v22_out.jsonl python3 gen_probe2_v22.py variant=base n=900 seed=$s heldstream=0 ral=1 ral_other=1 > /dev/null
  OUTJ=gen_probe2_v22_out.jsonl python3 gen_probe2_v22.py variant=base n=900 seed=$s heldstream=0 ral=1 ral_other=1 $CF > /dev/null
  echo "seed $s done $(date +%T)"
done
echo SYMDONE
