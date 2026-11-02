"""screen the existing motor knobs (g_ret_mot, c_mb, theta_m) for a quiet wait on a trained brain (learning off):
does the child stay silent after a question unless the inner hearing holds a whole syllable (forced diagnostic)?
  python3 -I motor_screen.py <arm> <seed> <n_teach> [key=value ...]
"""
import sys, os, time, json, itertools
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
for _x in ('x1d', 'x1c'):   # repo: core4/phase1/run2/regimes from ../x1d, core3/world/run from ../x1c
    _p = __import__('os').path.join(__import__('os').path.dirname(__import__('os').path.abspath(__file__)), '..', _x)
    if _p not in __import__('sys').path:
        __import__('sys').path.append(_p)
import run3, phase1
from run3 import Physics, World, Brain, birth_rates, run_until, probe2, release_stats, summarize_probe, parse, ARMS

arm, seed, nt = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
kv = {k: parse(v) for k, v in (a.split('=', 1) for a in sys.argv[4:])}
bk = dict(ARMS[arm]['b'], **kv)
ph = Physics()
world = World(ph, seed)
nol = dict(bk); keep = {k: bk.get(k) for k in phase1.NOLEARN}; nol.update(phase1.NOLEARN)
brain = Brain(ph, seed, rates=birth_rates(ph, world), **nol)
run_until(brain, World(ph, seed + 50), 200)
from core5 import DEF
for k in phase1.NOLEARN:
    brain.c[k] = keep[k] if keep[k] is not None else DEF[k]
t0 = time.time()
run_until(brain, world, nt)
print('trained', nt, f'{time.time() - t0:.0f}s', 'knobs at training', {k: brain.c[k] for k in ('g_ret_mot', 'c_mb', 'theta_m')}, flush=True)
allp = [(f, x) for f in range(4) for x in range(8)]
base = {k: brain.c[k] for k in ('g_ret_mot', 'c_mb', 'theta_m')}
GR = json.loads(os.environ.get("MGRID", "[[1.0, 0.5, 0.3, 0.15], [1.2, 0.4, 0.0], [0.22, 0.35, 0.5]]"))
for g, cm, th in itertools.product(*GR):
    brain.c.update(g_ret_mot=g, c_mb=cm, theta_m=th)
    pr = probe2(brain, ph, world, allp, seed)
    s = summarize_probe(pr)['seen']
    prf = probe2(brain, ph, world, allp, seed, force=(6, 1.0))
    sf = summarize_probe(prf)['seen']
    rel = release_stats(prf)
    print(f'g_ret_mot {g:<5} c_mb {cm:<4} theta_m {th:<5} | probe silent {s["silent"]:.2f} garble {s["garble"]:.2f} talk {s["talk"]:.2f} first {s["first"]:.2f}'
          f' | FORCED lag6 first {sf["first"]:.2f} acc {sf["acc"]:.2f} garble {sf["garble"]:.2f} said_when_pred {rel["said_when_pred"]:.2f}', flush=True)
brain.c.update(base)
