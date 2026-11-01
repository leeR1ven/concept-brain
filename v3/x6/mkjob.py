"""make a queue_a800 job folder: python3 mkjob.py <job name> <specs.json> [<specs2.json> ...]; 4 processes"""
import os, shutil, sys, json
HERE = os.path.dirname(os.path.abspath(__file__))
Q = os.path.join(HERE, '..', 'gpu', 'queue_a800')
name = sys.argv[1]; d = os.path.join(Q, name)
assert not os.path.exists(d), d
os.makedirs(os.path.join(d, 'specs'))
for f in ('x3_hippo.py', 'x6_hippo.py', 'x6_run.py', 'stream.json'):
    shutil.copy(os.path.join(HERE, f), d)
lines = []
for sp in sys.argv[2:]:
    shutil.copy(sp, os.path.join(d, 'specs'))
    lines.append(f'x6_run.py specs/{os.path.basename(sp)}')
open(os.path.join(d, 'jobs.txt'), 'w').write('\n'.join(lines) + '\n')
open(os.path.join(d, 'job.sh'), 'w').write('''#!/bin/bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
mkdir -p out/logs
nproc; python3 -c "import numpy; print('numpy', numpy.__version__)"
i=0
while read -r line; do
  ( echo "START $i $(date +%T) $line"; python3 -I $line > out/logs/run$i.log 2>&1; echo "END $i rc=$? $(date +%T)"; tail -2 out/logs/run$i.log ) &
  i=$((i+1))
  while [ $(jobs -rp | wc -l) -ge 4 ]; do sleep 5; done
done < jobs.txt
wait
rm -f stream.json
echo ALLDONE $(date +%T)
''')
open(os.path.join(d, 'TIMEOUT'), 'w').write(sys.argv[0] and os.environ.get('X6_TIMEOUT', '90'))
print('made', d, lines)
