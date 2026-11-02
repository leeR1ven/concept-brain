#!/bin/bash
# usage: bash mkjob.sh <label> <P> <timeout_min> <queue_a800|queue_a800gpu> < jobs.txt -> arch/v2/gpu/<queue>/meaning_<label>/
# (Coordinator job tooling, kept as a record: it expects the flat scratch layout with the v2.1 / randmix / ARCH_v2 files beside it.)
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
QD=$4
Q=$HERE/../gpu/$QD/meaning_$1
[ -e "$Q" ] && { echo "exists: $Q"; exit 1; }
mkdir -p "$Q"
cp $HERE/*.py "$Q"/
rm -f "$Q"/gen_probe2_v21_orig.py
mkdir -p "$Q/bench_copy" "$Q/out/runs" "$Q/out/post" "$Q/out/gp1" "$Q/out/logs" && cp $HERE/bench_copy/*.py $HERE/bench_copy/*.json $HERE/bench_copy/READY "$Q/bench_copy/"
cat > "$Q/jobs.txt"
cat > "$Q/job.sh" << EOS
#!/bin/bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
nproc; python3 -c "import numpy; print('numpy', numpy.__version__)"; python3 --version
python3 batch.py jobs.txt $2
EOS
echo $3 > "$Q/TIMEOUT"
du -sh "$Q"
touch "$Q/SUBMIT"
echo submitted $Q
