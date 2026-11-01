#!/bin/bash
# runs every line of jobs.txt ("<arm> <seed> [key=value ...]") as one run6.py life, 4 at a time
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
mkdir -p out6 logs
nproc; python3 -c "import numpy; print('numpy', numpy.__version__)"
i=0
while read -r arm seed extra; do
  [ -z "$arm" ] && continue
  ( echo "START $i $(date +%T) $arm $seed $extra"; python3 -I run6.py $arm $seed $extra > logs/${arm}_s${seed}_$i.log 2>&1; echo "END $i $arm $seed rc=$? $(date +%T)"; grep -h "O10000\|O30000\|wrote" logs/${arm}_s${seed}_$i.log | tail -2 | cut -c1-400 ) &
  i=$((i+1))
  while [ $(jobs -rp | wc -l) -ge 4 ]; do sleep 3; done
done < jobs.txt
wait
echo ALLDONE $(date +%T)
