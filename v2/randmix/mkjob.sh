#!/bin/bash
# usage: bash mkjob.sh <label> <P> <timeout_min> [queue|queue_a800] < jobs.txt -> arch/v2/gpu/<queue>/randmix_<label>/
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
QD=${4:-queue}
Q=$HERE/../gpu/$QD/randmix_$1
[ -e "$Q" ] && { echo "exists: $Q"; exit 1; }
mkdir -p "$Q"
cp $HERE/{randlayer.py,rcore.py,run_rm.py,core.py,core_think.py,core_synth.py,rules.py,agent.py,agent_think.py,agent_synth.py,body_think.py,batch.py,gen_probe.py,gen_probe2.py,post.py,full.py,randtorch.py,test_torch.py,gpu_bench.py} "$Q"/
mkdir -p "$Q/bench_copy" && cp $HERE/bench_copy/*.py $HERE/bench_copy/*.json $HERE/bench_copy/READY "$Q/bench_copy/"
cat > "$Q/jobs.txt"
cat > "$Q/job.sh" << EOS
#!/bin/bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
nproc; python3 -c "import numpy; print('numpy', numpy.__version__)"
python3 batch.py jobs.txt $2
EOS
echo $3 > "$Q/TIMEOUT"
du -sh "$Q"
touch "$Q/SUBMIT"
echo submitted $Q
