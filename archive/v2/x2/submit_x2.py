# -*- coding: utf-8 -*-
"""(x2 copy of v22/submit.py) Make a runner job in gpu/queue_a800/x2_<label>/: copy the x2 code + bench copy (+ optional
checkpoint pickles into ck/), write jobs.txt (one command per line, run P at a time) and job.sh, then touch SUBMIT.
Checkpoints written by the job larger than 45 MB are split into 40 MB parts (the runner copies back files < 50 MB).
usage: python3 submit_x2.py <label> <P> <minutes> <jobsfile> [ck files to upload ...]"""
import os, shutil, sys
HERE = os.path.dirname(os.path.abspath(__file__))
label, P, mins, jf = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
cks = sys.argv[5:]
d = os.path.join(HERE, '..', 'gpu', 'queue_a800', f'x2_{label}')
assert not os.path.exists(d), d
os.makedirs(d)
for f in os.listdir(HERE):
    if f.endswith('.py'):
        shutil.copy(os.path.join(HERE, f), d)
shutil.copytree(os.path.join(HERE, 'bench_copy'), os.path.join(d, 'bench_copy'),
                ignore=shutil.ignore_patterns('__pycache__', 'cache'))
os.makedirs(os.path.join(d, 'ck'))
for c in cks:
    shutil.copy(c, os.path.join(d, 'ck', os.path.basename(c)))
lines = [l.strip() for l in open(jf) if l.strip() and not l.startswith('#')]
open(os.path.join(d, 'jobs.txt'), 'w').write('\n'.join(lines) + '\n')
open(os.path.join(d, 'job.sh'), 'w').write(f"""#!/bin/bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
mkdir -p out/runs out/logs out/life ck
nproc; python3 -c "import numpy; print('numpy', numpy.__version__)"
for f in ck/*.part00; do [ -e "$f" ] && cat "${{f%.part00}}".part* > "${{f%.part00}}" && rm "${{f%.part00}}".part*; done
ls -la ck
i=0
while read -r line; do
  ( echo "START $i $(date +%T) $line"; eval "python3 $line" > out/logs/run$i.log 2>&1; echo "END $i rc=$? $(date +%T)"; tail -3 out/logs/run$i.log ) &
  i=$((i+1))
  while [ $(jobs -rp | wc -l) -ge {P} ]; do sleep 5; done
done < jobs.txt
wait
for f in ck/*.pkl.gz; do [ -e "$f" ] || continue; s=$(stat -c%s "$f"); if [ $s -gt 45000000 ]; then split -b 40m -d "$f" "$f".part && rm "$f"; fi; done
for f in {' '.join('ck/' + os.path.basename(c) for c in cks)}; do rm -f "$f"; done
ls -la ck
echo ALLDONE $(date +%T)
""")
open(os.path.join(d, 'TIMEOUT'), 'w').write(str(mins))
open(os.path.join(d, 'SUBMIT'), 'w').write('')
print('submitted', d)
