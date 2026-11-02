"""(a) speech basics screen, learning off: existing motor knobs only (c_mb, gamma_m, theta_m, g_ret_mot).
Wanted: quiet while waiting / overhearing; a forced whole-answer inner hearing (the sound -> action map) is said exactly.
  python3 -I screen_motor.py <seed> <n_overheard_before> [key=value ...]
"""
import sys, os, itertools, json, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common6 import make_brain, run_until, probe6, summ, BASE
from world6 import OverhearWorld, overhear_mech
from run import parse
from wire import wire_speech

seed, n_ov = int(sys.argv[1]), int(sys.argv[2])
kv = {k: parse(v) for k, v in (a.split('=', 1) for a in sys.argv[3:])}
grid = json.loads(kv.pop('grid', '{"c_mb": [0.0], "gamma_m": [0.1], "theta_m": [0.15, 0.22], "g_ret_mot": [1.0], "g_dir": [0.5, 0.25]}'))
lam = kv.pop('lam', 0.3)
t0 = time.time()
standin = kv.pop('standin', False)
ph, brain = make_brain(seed, dict(BASE, **kv))
if not standin:
    print(wire_speech(brain, ph, lam))
ow = OverhearWorld(ph, seed)
if n_ov:
    run_until(brain, ow, n_ov)
print('overheard', n_ov, 'mech', overhear_mech(ow, 0, n_ov), f'{time.time() - t0:.0f}s', flush=True)
allp = [(f, x) for f in range(4) for x in range(8)]
seen = [p for p in allp if p not in ow.held]

snap = brain.snapshot()
for vals in itertools.product(*grid.values()):
    kn = dict(zip(grid.keys(), vals))
    brain.c.update(kn)
    smx = summ(probe6(brain, ph, ow.props, ow.held, seen, seed, force=1, force_mix=True))
    s = summ(probe6(brain, ph, ow.props, ow.held, seen, seed))
    sf = summ(probe6(brain, ph, ow.props, ow.held, seen, seed, force=1))
    w2 = OverhearWorld(ph, seed + 11)
    brain.restore(snap)
    run_until(brain, w2, 40, learn=False)
    om = overhear_mech(w2, 0, 40)
    brain.restore(snap)
    print(' '.join(f'{k} {v:<5}' for k, v in kn.items()), f'| MIX said {1 - smx["silent"]:.2f} garble {smx["garble"]:.2f} | probe silent {s["silent"]:.2f} garble {s["garble"]:.2f} '
          f'talk {s["talk"]:.2f} acc {s["acc"]:.2f} | FORCED answer acc {sf["acc"]:.2f} first {sf["first"]:.2f} garble {sf["garble"]:.2f} '
          f'silent {sf["silent"]:.2f} | overhear utts/dlg {om["child_utts_per_dlg"]:.3f} garbled {om["garbled_per_dlg"]:.3f}', flush=True)
print('done', f'{time.time() - t0:.0f}s')
