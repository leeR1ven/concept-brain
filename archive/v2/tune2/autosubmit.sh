#!/bin/bash
# keep at most 2 tune2 jobs queued/running: submit the labels in jobs/QUEUE (one "label jobfile procs timeout est" per line)
Q=${GPU_QUEUE:?set to the runner queue folder}
T=$(dirname "$0")
cd $T
while true; do
  [ -s jobs/QUEUE ] || { echo "queue empty $(date +%T)"; exit 0; }
  active=0
  for d in $Q/tune2_*; do [ -d "$d" ] && [ ! -f "$d/DONE" ] && active=$((active+1)); done
  if [ $active -lt 2 ]; then
    line=$(head -1 jobs/QUEUE); sed -i '1d' jobs/QUEUE
    ./submit.sh $line && echo "$(date +%T) submitted $line"
    ./collect.sh > /dev/null
    sleep 20
  else
    sleep 30
  fi
done
