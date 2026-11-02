# -*- coding: utf-8 -*-
"""one full measurement on a server job: run_rm.py (staged J, pickles the trained agent) then post.py on that agent,
then the pickle is deleted (too large to copy back).  usage: python3 full.py tag=<t> seed=<s> [post=hfd] <run_rm keys>"""
import os, subprocess, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
args = sys.argv[1:]
kv = dict(a.split('=', 1) for a in args if '=' in a)
parts = kv.pop('post', 'hfd')
args = [a for a in args if not a.startswith('post=')]
tag, seed = kv['tag'], kv['seed']
t0 = time.time()
r = subprocess.run([sys.executable, os.path.join(HERE, 'run_rm.py')] + args, cwd=HERE)
print(f"run_rm rc={r.returncode} {time.time() - t0:.0f}s", flush=True)
pk = os.path.join(HERE, 'out', 'agents', f'{tag}_s{seed}.pkl')
if r.returncode == 0 and parts and os.path.exists(pk):
    r2 = subprocess.run([sys.executable, os.path.join(HERE, 'post.py'), f'tag={tag}', f'seed={seed}', f'parts={parts}'],
                        cwd=HERE)
    print(f"post rc={r2.returncode} {time.time() - t0:.0f}s", flush=True)
if os.path.exists(pk):
    os.remove(pk)
