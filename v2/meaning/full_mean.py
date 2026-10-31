# -*- coding: utf-8 -*-
"""one full measurement (server job): run_mean.py save=1 (staged J, frozen protocol) then post_mean.py on the pickled
agent (parts: h held-out per task with references, f fault probe, l meaning lesions), then the pickle is deleted.
usage: python3 full_mean.py tag=<t> seed=<s> [post=hfl] <run_mean keys>"""
import os, subprocess, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
args = sys.argv[1:]
kv = dict(a.split('=', 1) for a in args if '=' in a)
parts = kv.pop('post', 'hfl')
args = [a for a in args if not a.startswith('post=')] + ['save=1']
tag, seed = kv['tag'], kv['seed']
t0 = time.time()
r = subprocess.run([sys.executable, os.path.join(HERE, 'run_mean.py')] + args, cwd=HERE)
print(f"run_mean rc={r.returncode} {time.time() - t0:.0f}s", flush=True)
pk = os.path.join(HERE, 'out', f'agent_{tag}_s{seed}.pkl')
if r.returncode == 0 and parts and os.path.exists(pk):
    r2 = subprocess.run([sys.executable, os.path.join(HERE, 'post_mean.py'), f'tag={tag}', f'seed={seed}', f'parts={parts}'],
                        cwd=HERE)
    print(f"post rc={r2.returncode} {time.time() - t0:.0f}s", flush=True)
if os.path.exists(pk):
    os.remove(pk)
