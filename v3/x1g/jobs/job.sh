#!/bin/bash
# runs every line of jobs.txt ("<name> <seed> [key=value ...]") as one run8.py life, NPAR at a time
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
NPAR=${NPAR:-4}
mkdir -p out8 logs
nproc; python3 -c "import numpy; print('numpy', numpy.__version__)"
i=0
while read -r name seed extra; do
  [ -z "$name" ] && continue
  ( echo "START $i $(date +%T) $name $seed $extra"; python3 -I run8.py $name $seed $extra > logs/${name}_s${seed}.log 2>&1; echo "END $i $name $seed rc=$? $(date +%T)"; grep -h " O[0-9]* \|wrote\|Error\|Traceback" logs/${name}_s${seed}.log | tail -3 | cut -c1-300 ) &
  i=$((i+1))
  while [ $(jobs -rp | wc -l) -ge $NPAR ]; do sleep 3; done
done < jobs.txt
wait
echo ALLDONE $(date +%T)
