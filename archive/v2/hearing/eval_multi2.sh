#!/bin/bash
cd "$(dirname "$0")"
./eval_grid.sh gate g_backtrack=0 conf=1 relock=1 vbound=1 nocreate=1 tag=_c > results/eval_c.log 2>&1
./eval_grid.sh gate g_backtrack=1 conf=1 relock=1 vbound=1 nocreate=1 tag=_cb > results/eval_cb.log 2>&1
./eval_grid.sh gate g_backtrack=0 conf=1 tag=_gc > results/eval_gc.log 2>&1
echo MULTIDONE > results/eval_multi2.done
