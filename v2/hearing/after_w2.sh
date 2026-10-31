#!/bin/bash
# runs after the world2 grid: the final ear-alone grid, then the 10-day ear-alone life with a noise schedule
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1
until [ -f results/w2/grid.done ]; do sleep 20; done
./eval_grid.sh v21,gate g_backtrack=0 conf=1 relock=1 vbound=1 nocreate=1 level=1 denoise=1 tag=_final > results/eval_final.log 2>&1
nice python3 -B record.py 16 2 0.004 0.0 10 0 0:0.0,3:1.0,6:0.5,8:1.0 > results/life_rec.log 2>&1
nice python3 -B life_eval.py rec/w2_N16_p0.004_s0.0_seed0_sched.pkl gate g_backtrack=0 conf=1 relock=1 vbound=1 nocreate=1 level=1 denoise=1 tag=_final > results/life_final.log 2>&1
nice python3 -B life_eval.py rec/w2_N16_p0.004_s0.0_seed0_sched.pkl v21 tag=_v21 > results/life_v21.log 2>&1
echo AFTERDONE > results/after.done
