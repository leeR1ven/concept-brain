# -*- coding: utf-8 -*-
"""Run a list of driver command lines with P worker processes (BLAS threads 1).
usage: python3 batch.py jobs.txt P deadline_secs est_secs
jobs.txt: one line per run: '<script.py> key=val ...' ('#' comments).  A run is not started when less than its
expected time (running mean of finished runs, else est_secs) x 1.05 is left before the deadline."""
import os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
jobs = [l.strip() for l in open(sys.argv[1]) if l.strip() and not l.startswith('#')]
P = int(sys.argv[2])
deadline = time.time() + float(sys.argv[3])
est0 = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0
env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
t0 = time.time()
durs = {}


def run(j):
    i, line = j
    parts = line.split()
    kind = parts[0]
    d = durs.get(kind, [])
    est = (sum(d) / len(d)) if d else est0
    if time.time() + est * 1.05 > deadline:
        print(f"SKIP {i} (deadline) {line}", flush=True)
        return
    ts = time.time()
    os.makedirs(os.path.join(HERE, "out", "logs"), exist_ok=True)
    lp = os.path.join(HERE, "out", "logs", f"run{i}.log")
    print(f"--- start run {i} (t={ts - t0:.0f}s): {line}", flush=True)
    with open(lp, "w") as f:                               # streamed: progress survives a timeout
        p = subprocess.run([sys.executable, "-u", os.path.join(HERE, kind)] + parts[1:], cwd=HERE, env=env,
                           stdout=f, stderr=subprocess.STDOUT, text=True)
    durs.setdefault(kind, []).append(time.time() - ts)
    tail = "\n".join(open(lp).read().strip().splitlines()[-3:])
    print(f"=== run {i} rc={p.returncode} {time.time() - ts:.0f}s (t={time.time() - t0:.0f}s): {line}\n{tail}", flush=True)


with ThreadPoolExecutor(P) as ex:
    list(ex.map(run, list(enumerate(jobs))))
print(f"BATCH DONE {time.time() - t0:.0f}s", flush=True)
