#!/bin/bash
cd "$(dirname "$0")"
./eval_grid.sh v21,gate tag=_g > results/eval_g.log 2>&1
./eval_grid.sh gate relock=1 tag=_rl > results/eval_rl.log 2>&1
./eval_grid.sh gate relock=1 vbound=1 tag=_rlvb > results/eval_rlvb.log 2>&1
./eval_grid.sh gate relock=1 vbound=1 nocreate=1 tag=_rlvbnc > results/eval_rlvbnc.log 2>&1
echo MULTIDONE > results/eval_multi.done
