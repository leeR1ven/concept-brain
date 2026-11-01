#!/bin/bash
SQ="rm=1 rm_train=1 rm_prune=1 rm_age=20000 rm_regrow=1 rm_alloc=pow:0.5"
for s in 0 1 2; do
  python3 gen_probe_v22.py variant=base n=900 seed=$s ral=1 ral_other=1 $SQ > /dev/null
  python3 gen_probe_v22.py variant=base n=900 seed=$s tutor=mixed ral=1 ral_other=1 $SQ > /dev/null
  OUTJ=gen_probe2_v22_out.jsonl python3 gen_probe2_v22.py variant=base n=900 seed=$s heldstream=0 ral=1 ral_other=1 $SQ > /dev/null 2>&1
  echo "sq seed $s done $(date +%T)"
done
echo SQDONE
