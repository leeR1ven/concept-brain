#!/bin/bash
# usage: submit.sh <label> <jobs.txt> [procs=4] [timeout_min=38]
set -e
L=$1; JF=$2; P=${3:-4}; TM=${4:-38}
Q=${GPU_QUEUE:?set GPU_QUEUE to the job-runner queue dir}/tune_$L
T="$(cd "$(dirname "$0")" && pwd)"
[ -e $Q ] && { echo "exists $Q"; exit 1; }
mkdir -p $Q/out/runs
cp $T/{core.py,rules.py,agent.py,tune.py,batch.py} $Q/
cp -r $T/bench_copy $Q/; rm -rf $Q/bench_copy/__pycache__
cp $JF $Q/jobs.txt
DL=$(( TM * 60 - 150 ))
cat > $Q/job.sh <<EOS
#!/bin/bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
python3 batch.py jobs.txt $P $DL
EOS
echo $TM > $Q/TIMEOUT
touch $Q/SUBMIT
echo "submitted $Q ($(grep -vc '^#' $JF) jobs, $P procs, ${TM} min)"
