"""make a queue_a800 job folder: python3 x6b_mkjob.py <job name> <specs.json> [...]; 4 processes"""
import os, shutil, sys
HERE = os.path.dirname(os.path.abspath(__file__))
Q = os.path.join(HERE, '..', 'gpu', 'queue_a800')
name = sys.argv[1]; d = os.path.join(Q, name)
assert name.startswith('x6b_') and not os.path.exists(d), d
os.makedirs(os.path.join(d, 'specs'))
for f in ('x3_hippo.py', 'x6_hippo.py', 'x6b_hippo.py', 'x6b_run.py', 'stream.json'):   # repo: from ., ../x6, ../x345
    src = next(p for p in (os.path.join(HERE, x, f) for x in ('.', os.path.join('..', 'x6'), os.path.join('..', 'x345')))
               if os.path.exists(p))
    shutil.copy(src, d)
lines = []
for sp in sys.argv[2:]:
    shutil.copy(sp, os.path.join(d, 'specs'))
    lines.append(f'x6b_run.py specs/{os.path.basename(sp)}')
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
open(os.path.join(d, 'TIMEOUT'), 'w').write(os.environ.get('X6B_TIMEOUT', '90'))
print('made', d, lines)
