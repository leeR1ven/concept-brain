# -*- coding: utf-8 -*-
"""run the command lines of jobs.txt with P worker processes (server job); usage: python3 batch.py jobs.txt P"""
import os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
HERE = os.path.dirname(os.path.abspath(__file__))
jobs = [l.strip() for l in open(sys.argv[1]) if l.strip() and not l.startswith('#')]
P = int(sys.argv[2])
env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
t0 = time.time()
os.makedirs(os.path.join(HERE, "out", "logs"), exist_ok=True)


def run(j):
    i, line = j
    parts = line.split()
    with open(os.path.join(HERE, "out", "logs", f"job{i}.log"), "w") as f:
        p = subprocess.run([sys.executable, os.path.join(HERE, parts[0])] + parts[1:], cwd=HERE, env=env,
                           stdout=f, stderr=subprocess.STDOUT, text=True)
    print(f"=== job {i} rc={p.returncode} t={time.time() - t0:.0f}s: {line}", flush=True)


with ThreadPoolExecutor(P) as ex:
    list(ex.map(run, list(enumerate(jobs))))
print(f"BATCH DONE {time.time() - t0:.0f}s", flush=True)
