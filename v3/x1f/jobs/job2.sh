#!/bin/bash
# runs every line of jobs.txt as a shell command (each a python3 -I ... call), 4 at a time
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
mkdir -p out6 out7 logs
i=0
while read -r line; do
  [ -z "$line" ] && continue
  ( echo "START $i $(date +%T) $line"; bash -c "$line" > logs/job_$i.log 2>&1; echo "END $i rc=$? $(date +%T)"; tail -3 logs/job_$i.log | cut -c1-300 ) &
  i=$((i+1))
  while [ $(jobs -rp | wc -l) -ge 4 ]; do sleep 3; done
done < jobs.txt
wait
echo ALLDONE $(date +%T)
