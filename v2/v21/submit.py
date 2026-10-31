# -*- coding: utf-8 -*-
"""Make a runner job: copy the v21 code + bench copy into gpu/<queue>/v21_<label>/, write jobs.txt (one command per
line, run P at a time) and job.sh, then touch SUBMIT.  usage: python3 submit.py <label> <queue|queue_a800> <P> <minutes> <jobsfile>"""
import os, shutil, sys
HERE = os.path.dirname(os.path.abspath(__file__))
label, queue, P, mins, jf = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
d = os.path.join(HERE, '..', 'gpu', queue, f'v21_{label}')
assert not os.path.exists(d), d
os.makedirs(d)
for f in os.listdir(HERE):
    if f.endswith('.py'):
        shutil.copy(os.path.join(HERE, f), d)
shutil.copytree(os.path.join(HERE, 'bench_copy'), os.path.join(d, 'bench_copy'),
                ignore=shutil.ignore_patterns('__pycache__', 'cache'))
if os.path.isdir(os.path.join(HERE, 'regress')):
    shutil.copytree(os.path.join(HERE, 'regress'), os.path.join(d, 'regress'),
                    ignore=shutil.ignore_patterns('__pycache__', '*.json', '*.jsonl', '*.out', '*.log'))
lines = [l.strip() for l in open(jf) if l.strip() and not l.startswith('#')]
open(os.path.join(d, 'jobs.txt'), 'w').write('\n'.join(lines) + '\n')
open(os.path.join(d, 'job.sh'), 'w').write(f"""#!/bin/bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
mkdir -p out/runs out/logs out/life
nproc; python3 -c "import numpy; print('numpy', numpy.__version__)"
i=0
while read -r line; do
  ( echo "START $i $(date +%T) $line"; eval "python3 $line" > out/logs/run$i.log 2>&1; echo "END $i rc=$? $(date +%T)"; tail -3 out/logs/run$i.log ) &
  i=$((i+1))
  while [ $(jobs -rp | wc -l) -ge {P} ]; do sleep 5; done
done < jobs.txt
wait
echo ALLDONE $(date +%T)
""")
open(os.path.join(d, 'TIMEOUT'), 'w').write(str(mins))
open(os.path.join(d, 'SUBMIT'), 'w').write('')
print('submitted', d)
