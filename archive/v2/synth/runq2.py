# -*- coding: utf-8 -*-
"""persistent job queue: python3 runq2.py jobs.txt P -- re-reads jobs.txt (append lines any time: '<name> <command>'),
starts each new name once, keeps the number of heavy processes (run_j.py / life_synth.py / ablate_synth.py, any queue)
<= P.  Stops when the file contains a line 'END'.  Log: out/queue.log."""
import subprocess, sys, time, os
HERE = os.path.dirname(os.path.abspath(__file__))
P = int(sys.argv[2]) if len(sys.argv) > 2 else 3
env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
started, running = set(), []
def log(s):
    with open(os.path.join(HERE, 'out', 'queue.log'), 'a') as f:
        f.write(time.strftime('%H:%M:%S') + ' ' + s + '\n')
def heavy():
    r = subprocess.run("pgrep -fc '^python3 -B (run_j|life_synth|ablate_synth|world2_smoke)'", shell=True, capture_output=True, text=True)
    try:
        return int(r.stdout.strip() or 0)
    except ValueError:
        return 0
while True:
    lines = [l.strip() for l in open(sys.argv[1]) if l.strip() and not l.startswith('#')]
    stop = 'END' in lines
    jobs = [l for l in lines if l != 'END' and l.split(' ', 1)[0] not in started]
    running = [(n, p) for n, p in running if p.poll() is None or log(f'{n} end rc={p.returncode}')]
    while jobs and heavy() < P:
        name, cmd = jobs.pop(0).split(' ', 1)
        started.add(name)
        out = open(os.path.join(HERE, 'out', name + '.log'), 'w')
        p = subprocess.Popen('nice ' + cmd, shell=True, cwd=HERE, stdout=out, stderr=subprocess.STDOUT, env=env)
        running.append((name, p)); log(f'{name} start: {cmd}')
        time.sleep(3)
    if stop and not jobs and not running:
        break
    time.sleep(10)
log('queue2 done')
