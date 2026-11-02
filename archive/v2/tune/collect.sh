#!/bin/bash
# copy finished server results into out/runs (job label prefix avoids clashes); show job states
Q=${GPU_QUEUE:?set GPU_QUEUE to the job-runner queue dir}
T="$(cd "$(dirname "$0")" && pwd)"
mkdir -p $T/out/runs
for d in $Q/tune_*; do
  n=$(basename $d); [ "$n" = "tune_t1" ] && continue
  st=running; [ -f $d/DONE ] && st="done $(cat $d/DONE)"
  if [ -d $d/result/out/runs ]; then
    for f in $d/result/out/runs/*.json; do [ -f "$f" ] && [ ! -e $T/out/runs/$(basename $f) ] && cp "$f" $T/out/runs/; done
  fi
  if [ -d $d/result/out/life ]; then
    mkdir -p $T/out/life
    for f in $d/result/out/life/*.json; do [ -f "$f" ] && cp "$f" $T/out/life/; done
  fi
  echo "$n: $st"
done
