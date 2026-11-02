#!/bin/bash
export OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
cd "$(dirname "$0")"
until [ -f out/batch_post.done ]; do sleep 10; done
nice python3 sizetune.py tag=c_inter seed=1 cur=inter > out/st_c_s1.stdout 2>&1
nice python3 sizetune.py tag=base seed=1 cur=seq > out/st_base_s1.stdout 2>&1
echo DONE > out/batch_seed1.done
