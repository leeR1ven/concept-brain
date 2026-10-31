# -*- coding: utf-8 -*-
"""Run a list of tune.py configurations with P worker processes (each a subprocess; BLAS threads 1).
usage: python3 batch.py jobs.txt P [deadline_secs]   (one line of tune.py arguments per job; '#' comments)
A job is not started when less than its expected time is left before the deadline (so the batch never times out)."""
import os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
jobs = [l.strip() for l in open(sys.argv[1]) if l.strip() and not l.startswith('#')]
P = int(sys.argv[2])
deadline = time.time() + float(sys.argv[3]) if len(sys.argv) > 3 else None
env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
t0 = time.time()
durs = []


def run(j):
    i, line = j
    if deadline is not None:
        est = (sum(durs) / len(durs)) if durs else 0
        if time.time() + est * 1.1 > deadline:
            print(f"SKIP {i} (deadline) {line[:80]}", flush=True)
            return
    ts = time.time()
    p = subprocess.run([sys.executable, os.path.join(HERE, "tune.py")] + line.split(), cwd=HERE, env=env,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    durs.append(time.time() - ts)
    tail = "\n".join(p.stdout.strip().splitlines()[-6:])
    print(f"=== job {i} rc={p.returncode} {time.time() - ts:.0f}s (t={time.time() - t0:.0f}s): {line[:160]}\n{tail}", flush=True)


with ThreadPoolExecutor(P) as ex:
    list(ex.map(run, list(enumerate(jobs))))
print(f"BATCH DONE {time.time() - t0:.0f}s", flush=True)
