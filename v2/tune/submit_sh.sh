#!/bin/bash
# usage: submit_sh.sh <label> <job.sh> [timeout_min=38]   (custom job script; copies the code + bench)
set -e
L=$1; JS=$2; TM=${3:-38}
Q=${GPU_QUEUE:?set GPU_QUEUE to the job-runner queue dir}/tune_$L
T="$(cd "$(dirname "$0")" && pwd)"
[ -e $Q ] && { echo "exists $Q"; exit 1; }
mkdir -p $Q/out/runs $Q/out/life
cp $T/{core.py,rules.py,agent.py,tune.py,batch.py,life_run.py} $Q/
cp -r $T/bench_copy $Q/; rm -rf $Q/bench_copy/__pycache__
cp $JS $Q/job.sh
echo $TM > $Q/TIMEOUT
touch $Q/SUBMIT
echo "submitted $Q"
