"""11-02 tests on the stage-(a) candidate (C by default): fast PFC (main), band overlap, excitability k.
usage: python3 run_fast.py --mode fast|band|k --seed 0 --out out_fast [--quick]"""
import argparse
import json
import os

from brain import Brain
from common import config, log, save_json
from life import Life
import probes
import probes2
from wiring import measure_codes, select_codes

ap = argparse.ArgumentParser()
ap.add_argument('--mode', default='fast')
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--out', default='out_fast')
ap.add_argument('--base', default='{"inhib": "P1", "loop": "recip", "g_loop": 1.0, "w44": 0.0, "g44": 0.1, "g_hh": 0.1, '
                                  '"g_h1": 0.1}')   # candidate B (tentative base, coordinator 11-02)
ap.add_argument('--quick', action='store_true')
ap.add_argument('--arms', default=None, help='comma list of arm names to run')
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
base = json.loads(a.base)
q = a.quick

if a.mode in ('fast', 'traj'):
    ARMS = {'F0': dict(eta=0.02, eta_m=0.01, gamma_pfc=0.99995), 'F5': dict(eta=0.1, eta_m=0.05, gamma_pfc=0.999),
            'F20': dict(eta=0.4, eta_m=0.2, gamma_pfc=0.99)}       # x1 / x5 / x20 of B's PFC rate
elif a.mode == 'band':                    # birth-only overlap trace (coordinator 11-02)
    r = (0.20, 0.12, 0.08, 0.05)
    ARMS = {'medium (A1 <= 0.9)': {}}
    for h in (0.6, 0.3):
        ARMS[f'A1 <= {h}'] = dict(band_sound=(0.002, h))
        ARMS[f'A1+A2 <= {h}'] = dict(band_depth_sound=2, band_sound=[(0.002, h), (0.9 * r[1], h)])
    ARMS['sound rates x0.5'] = dict(sound_rates=tuple(x * 0.5 for x in r))
elif a.mode == 'collapse':
    ARMS = {}
    for m in (8, 32):
        ARMS[f'noMC m{m}'] = dict(mc=False, force_per_muscle=m, b_block=1.2, g_ret_force=1.6)
        ARMS[f'MC pools m{m}'] = dict(mc=True, force_per_muscle=m, b_block=1.2, g_ret_force=1.6)
else:
    ARMS = {'k1': dict(ctx_k=1.0, calib_online_ticks=6000), 'k2': dict(ctx_k=2.0, calib_online_ticks=6000)}

if a.arms:
    ARMS = {k: v for k, v in ARMS.items() if k in a.arms.split(',')}
for arm, extra in ARMS.items():
    c = config(1, birth_dialogues=30 if q else 200, **{**base, **extra})
    life = Life(c, a.seed, wire=False)
    life.codes = select_codes(measure_codes(life.brain, life.coch))
    res = dict(mode=a.mode, arm=arm, config={**base, **extra}, seed=a.seed, n1=life.brain.n1, calib=life.brain.calib,
               chain_report=life.brain.chain_report)
    if a.mode == 'collapse':
        life.wire_mc()
        res['registry'] = life.registry
        res['collapse'] = probes2.collapse_test(life, a.seed)
        log(arm, {k: {kk: round(vv, 2) if isinstance(vv, float) else vv for kk, vv in v.items()}
                  for k, v in res['collapse'].items()})
        save_json(os.path.join(a.out, f'collapse_{arm.replace(" ", "_")}_seed{a.seed}.json'), res)
        continue
    if a.mode == 'traj':        # does the hold grow with training? a3 probe at 0 / 100 / 300 / 1,000 dialogues
        lags = (0, 6, 12) if q else (0, 6, 12, 24, 48)
        traj = {}
        done = 0
        for cp in ((0, 5, 10) if q else (0, 100, 300, 1000)):
            life.brain.learning = True
            life.run_dialogues(cp - done)
            done = cp
            r = dict(off=probes2.codes_ext(life, lags, learn=False, n_ctx=2 if q else 4),
                     on=probes2.codes_ext(life, lags, learn=True, n_ctx=2 if q else 4))
            traj[cp] = r
            log(arm, 'dialogues', cp, 'lag12 off', round(r['off']['L4_q32'][2], 3), 'filler12 off',
                round(r['off']['L4_filler'][2], 3), 'on', round(r['on']['L4_q32'][2], 3) if 'on' in r else None,
                'motion', [round(z, 2) for z in r['off']['motion_J']], 'remembered', r['off']['remembered_until'])
            res['trajectory'] = traj
            save_json(os.path.join(a.out, f'traj_{arm}_seed{a.seed}.json'), res)
        continue
    act = probes.activity(life, 5 if q else (30 if a.mode == 'band' else 50))
    res['a1'], res['a1_pass'] = act, probes.a1_pass(act)
    lags = (0, 6, 12) if q else (0, 6, 12, 24, 48)
    if a.mode == 'fast':
        res['codes_learn_off'] = probes2.codes_ext(life, lags, learn=False, n_ctx=2 if q else 4)
        res['codes_learn_on'] = probes2.codes_ext(life, lags, learn=True, n_ctx=2 if q else 4)
        res['one_shot'] = probes2.one_shot(life, Ds=(50, 100) if q else (200, 500, 1000, 2000))
        res['long_term'] = probes2.long_term(life, reps=4 if q else 20, dialogues=10 if q else 300)
        log(arm, 'off', res['codes_learn_off']['L4_q32'], 'on', res['codes_learn_on']['L4_q32'],
            'remembered off/on', res['codes_learn_off']['remembered_until'], res['codes_learn_on']['remembered_until'])
        log(arm, 'one-shot', res['one_shot'])
        log(arm, 'long-term', {k: v for k, v in res['long_term'].items() if k != 'stability'},
            'a1L', res['long_term']['stability']['pass_'])
    else:                                                  # band: overlap trace, a1 with the floor, a2 numbers
        res['overlap'] = probes2.overlap_trace(life)
        res['codes'] = probes.newborn_codes(life, n_ctx=2, n_noise=1, lags=12)
        ov = res['overlap']
        log(arm, 'a1', res['a1_pass'], 'L1 sil/sp', round(act['L1']['silence']), round(act['L1']['speech']),
            'J_diff pooled', {k: ov[k]['pooled']['J_diff'] for k in probes2.LAT},
            'evoked', {k: ov[k]['evoked']['J_diff'] for k in probes2.LAT}, 'inner', ov['L4_inner'])
    save_json(os.path.join(a.out, f'{a.mode}_{arm.replace(" ", "_").replace("<=", "le")}_seed{a.seed}.json'), res)
