# -*- coding: utf-8 -*-
"""Make an A800 runner job: copy the hearing code (+ v21copy with its bench copy; + world2 if asked) into
gpu/queue_a800/hearing_<label>/, write jobs.txt (one command per line, P at a time) and job.sh, then touch SUBMIT.
usage: python3 submit_h.py <label> <P> <minutes> <jobsfile> [w2]"""
# (Coordinator job tooling, kept as a record: it expects the scratch layout with v21copy/ beside it.)
import os, shutil, sys
HERE = os.path.dirname(os.path.abspath(__file__))
label, P, mins, jf = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
with_w2 = len(sys.argv) > 5 and sys.argv[5] == 'w2'
d = os.path.join(HERE, '..', 'gpu', 'queue_a800', f'hearing_{label}')
assert not os.path.exists(d), d
os.makedirs(d)
for f in ('hear.py', 'agent_h.py', 'run_h_canon.py', 'record.py', 'ear_eval.py', 'w2_agent.py'):
    shutil.copy(os.path.join(HERE, f), d)
ign = shutil.ignore_patterns('__pycache__', 'cache', 'out', '*.pkl')
shutil.copytree(os.path.join(HERE, 'v21copy'), os.path.join(d, 'v21copy'), ignore=ign)
if with_w2:                                          # w2_agent.py / record.py import ../bench/world2
    os.makedirs(os.path.join(d, 'bench'))
    shutil.copytree(os.path.join(HERE, '..', 'bench', 'world2'), os.path.join(d, 'bench', 'world2'),
                    ignore=shutil.ignore_patterns('__pycache__', 'results', 'results_fovea', 'checks'))
    for f in ('physics.py', 'metrics.py', 'corpus_chars.json'):
        shutil.copy(os.path.join(HERE, '..', 'bench', f), os.path.join(d, 'bench', f))
lines = [l.strip() for l in open(jf) if l.strip() and not l.startswith('#')]
open(os.path.join(d, 'jobs.txt'), 'w').write('\n'.join(lines) + '\n')
open(os.path.join(d, 'job.sh'), 'w').write(f"""#!/bin/bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
mkdir -p out/runs out/logs out/w2
nproc; python3 --version; python3 -c "import numpy; print('numpy', numpy.__version__)"
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
