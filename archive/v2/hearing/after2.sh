#!/bin/bash
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1
until [ -f results/after.done ]; do sleep 20; done
./eval_grid.sh v21 tag=_v21pure > results/eval_v21pure.log 2>&1
echo DONE2 > results/after2.done
