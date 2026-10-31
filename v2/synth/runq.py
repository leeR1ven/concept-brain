# -*- coding: utf-8 -*-
"""tiny job queue: python3 runq.py jobs.txt P  -- runs the commands (one per line, '#' comments) with at most P at once;
appends 'name start/end' lines to out/queue.log.  Each command line: <name> <command...> (stdout -> out/<name>.log)"""
import subprocess, sys, time, os
HERE = os.path.dirname(os.path.abspath(__file__))
jobs = [l.strip() for l in open(sys.argv[1]) if l.strip() and not l.startswith('#')]
P = int(sys.argv[2]) if len(sys.argv) > 2 else 3
env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
running = []
def log(s):
    with open(os.path.join(HERE, 'out', 'queue.log'), 'a') as f:
        f.write(time.strftime('%H:%M:%S') + ' ' + s + '\n')
while jobs or running:
    running = [(n, p) for n, p in running if p.poll() is None or log(f'{n} end rc={p.returncode}')]
    while jobs and len(running) < P:
        name, cmd = jobs.pop(0).split(' ', 1)
        out = open(os.path.join(HERE, 'out', name + '.log'), 'w')
        p = subprocess.Popen('nice ' + cmd, shell=True, cwd=HERE, stdout=out, stderr=subprocess.STDOUT, env=env)
        running.append((name, p)); log(f'{name} start: {cmd}')
    time.sleep(5)
log('queue done ' + sys.argv[1])
