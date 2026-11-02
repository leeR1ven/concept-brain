#!/bin/bash
# usage: submit.sh <label> <jobs.txt> [procs=4] [timeout_min=40] [est_secs=1000]
set -e
L=$1; JF=$2; P=${3:-4}; TM=${4:-40}; EST=${5:-1000}
Q=${GPU_QUEUE_A800:?set to the runner queue_a800 folder}/tune2_$L
T=$(dirname "$0")
[ -e $Q ] && { echo "exists $Q"; exit 1; }
mkdir -p $Q/out/runs $Q/out/life
cp $T/{core.py,rules.py,agent.py,core_think.py,agent_think.py,body_think.py,core_synth.py,agent_synth.py,run_j.py,core_t2.py,agent_t2.py,run_t2.py,life_t2.py,batch.py} $Q/
cp -r $T/bench_copy $Q/; rm -rf $Q/bench_copy/__pycache__
cp $JF $Q/jobs.txt
DL=$(( TM * 60 - 120 ))
cat > $Q/job.sh <<EOS
#!/bin/bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
nproc; python3 -c "import numpy; print('numpy', numpy.__version__)"
python3 batch.py jobs.txt $P $DL $EST
EOS
echo $TM > $Q/TIMEOUT
touch $Q/SUBMIT
echo "submitted $Q ($(grep -vc '^#' $JF) runs, $P procs, ${TM} min, est ${EST}s)"
