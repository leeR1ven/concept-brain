"""build the A800 job folders for X8 (not submitted here: SUBMIT is touched separately, 2 jobs at a time)"""
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
Q = os.path.join(HERE, '..', 'gpu', 'queue_a800')

SWEEP = ['A0', 'A3', 'A3R', 'A4', 'A34', 'A34U', 'A12', 'A12U']
W0 = ['A3w', 'A4H', 'A34H']
runs = []
for seed in range(5):
    for arm in SWEEP:
        for w in (0.0, 0.25, 0.5, 1.0):
            runs.append((f'{arm}_w{w}_s{seed}', f'{arm} {w} {seed}', ''))
    for arm in W0:
        runs.append((f'{arm}_w0.0_s{seed}', f'{arm} 0.0 {seed}', ''))
sens = []
for seed in range(3):
    for arm in ('A34', 'A34U'):
        for w in (0.0, 1.0):
            for tag, ov in (('gh0.95', 'gamma_h=0.95'), ('gh0.99', 'gamma_h=0.99'), ('rh0.5', 'r_hip=0.5'),
                            ('rh2', 'r_hip=2.0')):
                sens.append((f'{arm}_{tag}_w{w}_s{seed}', f'{arm} {w} {seed}', ov + ' life=false'))

JOB_SH = """#!/bin/bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
mkdir -p out/runs out/logs
nproc; python3 -c "import numpy; print('numpy', numpy.__version__)"
i=0
while read -r name arm w seed extra; do
  [ -z "$name" ] && continue
  ( echo "START $i $(date +%T) $name"; python3 -I x8.py $arm $w $seed out/runs/$name.json $extra > out/logs/$name.log 2>&1; echo "END $i $name rc=$? $(date +%T)" ) &
  i=$((i+1))
  while [ $(jobs -rp | wc -l) -ge 4 ]; do sleep 3; done
done < jobs.txt
wait
echo ALLDONE $(date +%T)
"""


def build(name, items):
    d = os.path.join(Q, name)
    assert not os.path.exists(d), d
    os.makedirs(d)
    for f in ('x8.py', 'x4_merge.py'):
        shutil.copy(os.path.join(HERE, f), d)
    with open(os.path.join(d, 'job.sh'), 'w') as fh:
        fh.write(JOB_SH)
    with open(os.path.join(d, 'jobs.txt'), 'w') as fh:
        for n, a, e in items:
            a1, a2, a3 = a.split()
            fh.write(f'{n} {a1} {a2} {a3} {e}\n'.replace('  ', ' '))
    with open(os.path.join(d, 'TIMEOUT'), 'w') as fh:
        fh.write('90')
    print(name, len(items))


def build_c(name='x8_c1'):
    """added after the first results (PLAN 9.8): corrections of freshly learned answers, no stretch"""
    items = []
    for seed in range(5):
        for arm in ['A0', 'A3w', 'A3', 'A3R', 'A4H', 'A4', 'A34H', 'A34', 'A34U', 'A12', 'A12U']:
            for w in (0.0, 0.25, 1.0):
                items.append((f'C-{arm}_w{w}_s{seed}', f'{arm} {w} {seed}',
                              'n_ask=0 n_fill_s=0 life=false amnesia=false do_new=false'))
    build(name, items)


if __name__ == '__main__' and len(sys.argv) > 1 and sys.argv[1] == 'c':
    build_c()
elif __name__ == '__main__':
    n_main = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    tag = sys.argv[2] if len(sys.argv) > 2 else 'm'
    parts = [runs[k::n_main] for k in range(n_main)]
    for k, p in enumerate(parts):
        build(f'x8_{tag}{k + 1}', p)
    build(f'x8_{tag}sens', sens)

