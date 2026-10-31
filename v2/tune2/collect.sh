#!/bin/bash
# copy finished server results into out/runs, out/life, out/joblogs; show job states
Q=${GPU_QUEUE:?set to the runner queue folder}
T=$(dirname "$0")
mkdir -p $T/out/runs $T/out/life $T/out/joblogs
for d in $Q/tune2_*; do
  n=$(basename $d)
  st=running; [ -f $d/DONE ] && st="done $(cat $d/DONE)"
  if [ -d $d/result/out/runs ]; then
    for f in $d/result/out/runs/*.json; do [ -f "$f" ] && [ ! -e $T/out/runs/$(basename $f) ] && cp "$f" $T/out/runs/; done
  fi
  if [ -d $d/result/out/life ]; then
    for f in $d/result/out/life/*; do [ -f "$f" ] && [ ! -e $T/out/life/$(basename $f) ] && cp "$f" $T/out/life/; done
  fi
  [ -f $d/result/run.log ] && cp $d/result/run.log $T/out/joblogs/$n.log
  echo "$n: $st"
done
