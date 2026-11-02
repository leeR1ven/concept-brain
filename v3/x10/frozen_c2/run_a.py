"""Stage (a) [PLAN 6a, coordinator 11-02].
newborn mode: for every grid configuration, a new brain on the same birth stream, online birth calibration with
that configuration's loop and lateral on, then a1 (activity, learning off) and a2/a3 (question codes).
a1L mode: learning-on stability over 300 overheard dialogues for given configurations, with the §四.10 remedy
ladder (parameters -> inhibition strength -> per-neuron long-run threshold) tried in order until it passes.
usage: python3 run_a.py --mode newborn --grid loop --part 0 --nparts 2 --out out_a
       python3 run_a.py --mode a1L --configs cands.json --out out_a"""
import argparse
import itertools
import json
import os

from brain import Brain
from common import config, log, save_json
from life import Life
import probes

ap = argparse.ArgumentParser()
ap.add_argument('--mode', default='newborn')
ap.add_argument('--s', type=int, default=1)
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--out', default='out_a')
ap.add_argument('--grid', default='loop', help='loop | size | one')
ap.add_argument('--base', default='{}', help='json overrides for every configuration (e.g. the chosen loop)')
ap.add_argument('--part', type=int, default=0)
ap.add_argument('--nparts', type=int, default=1)
ap.add_argument('--configs', default=None, help='json list of overrides (a1L mode)')
ap.add_argument('--quick', action='store_true')
ap.add_argument('--a1_dialogues', type=int, default=50)
ap.add_argument('--l_dialogues', type=int, default=300)
ap.add_argument('--seeds', default=None, help='comma list (choose mode)')
ap.add_argument('--context', action='store_true', help='context test at birth and at the end of a1L (choose mode)')
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
base_over = json.loads(a.base)
RATES = {'default': (0.20, 0.12, 0.08, 0.05), 'sparse': (0.10, 0.06, 0.04, 0.025), 'dense': (0.30, 0.20, 0.12, 0.08)}
if a.grid == 'loop':
    grid = [dict(inhib=i, loop=lp, g_loop=g, w44=w) for i, lp, g, w in
            itertools.product(['P0', 'P1'], ['recip', 'random'], [0.25, 0.5, 1.0, 2.0], [0.0, 0.5, 1.0])]
elif a.grid == 'size':
    grid = [dict(pfc_size=z, sound_rates=RATES[r]) for z, r in
            itertools.product(['small', 'medium', 'large'], ['default', 'sparse', 'dense'])]
    grid += [dict(band_depth_sound=4, band=(0.002, 0.90)), dict(pfc_size='medium', band=(0.002, 0.99))]
else:
    grid = [dict()]
grid = grid[a.part::a.nparts]


def name_of(over):
    n = '_'.join(f'{k}{v}' for k, v in sorted(over.items())) or 'default'
    return f's{a.s}_seed{a.seed}_' + n.replace(' ', '').replace('(', '').replace(')', '').replace(',', '-')


def build(life, over):
    """a new brain for this configuration on the life's birth stream, calibrated online as it runs"""
    c = config(a.s, birth_dialogues=life.c['birth_dialogues'], **over)
    life.c = c
    life.brain = Brain(c, a.seed, life.coch, life.birth_A0, life.birth_speech)
    traj = life.brain.calibrate_online(life.birth_A0[-c['calib_online_ticks']:])
    life.brain.learning = False
    return c, traj


bd = 30 if a.quick else 200

if a.mode == 'choose':   # candidates x seeds: newborn checks + learning-on stability (extended a1L when >= 1000)
    for seed in [int(x) for x in a.seeds.split(',')]:
        a.seed = seed
        life = Life(config(a.s, birth_dialogues=bd, **base_over), seed, wire=False)
        for over in json.loads(open(a.configs).read()):
            o = {**base_over, **over}
            name = name_of(o)
            c, traj = build(life, o)
            ctx = {}
            if a.context:
                ctx['birth'] = probes.context_test(life, reps=2 if a.quick else 4)
            act = probes.activity(life, 5 if a.quick else a.a1_dialogues)
            cod = probes.newborn_codes(life, n_ctx=2 if a.quick else 4, n_noise=1, lags=12)
            st = probes.stability(life, 30 if a.quick else a.l_dialogues) if a.l_dialogues > 0 else None
            if a.context and st is not None:
                ctx['after_a1L'] = probes.context_test(life, reps=2 if a.quick else 4)
            res = dict(config=o, seed=seed, n1=life.brain.n1, calib=life.brain.calib, registry=life.registry,
                       a1=act, a1_pass=probes.a1_pass(act), codes=cod, stability=st, context=ctx)
            save_json(os.path.join(a.out, 'choose_' + name + '.json'), res)
            log('choose', name, 'a1', res['a1_pass'], 'a2', cod['a2_pass'], 'a3', cod['a3_pass'],
                'lag12', round(cod['L4_decode'][12], 3), 'Hs J margin', round(cod['Hs_J_margin'], 3),
                'a1L', st and st['pass_'], 'ext', st and st.get('pass_ext'),
                'trend', st and {k: round(v['ratio'], 2) for k, v in st.get('trend', {}).items()})
            for when, cx in ctx.items():
                for arm, v in cx.items():
                    log('context', when, arm, 'L4 q32', [round(z, 2) for z in v['L4_q32']], 'Hs q32',
                        [round(z, 2) for z in v.get('Hs_q32', [])], 'remembered until', v.get('L4_remembered_until'))
    raise SystemExit(0)

life = Life(config(a.s, birth_dialogues=bd, **base_over), a.seed, wire=False)

if a.mode == 'newborn':
    for over in grid:
        over = {**base_over, **over}
        name = name_of(over)
        c, traj = build(life, over)
        act = probes.activity(life, 5 if a.quick else a.a1_dialogues)
        res = dict(config=over, calib=life.brain.calib, calib_traj=traj[-5:], chain_report=life.brain.chain_report,
                   n1=life.brain.n1, a1=act, a1_pass=probes.a1_pass(act))
        res.update(probes.newborn_codes(life, n_ctx=2 if a.quick else 4, n_noise=1, lags=12))
        save_json(os.path.join(a.out, name + '.json'), res)
        log(name, 'a1', res['a1_pass'], 'a2', res['a2_pass'], 'a3', res['a3_pass'],
            'lag0', round(res['L4_decode'][0], 3), 'lag12', round(res['L4_decode'][12], 3),
            'L1 sil/sp', round(act['L1']['silence']), round(act['L1']['speech']), 'L4', round(act['L4']['speech']))

elif a.mode == 'sens':   # one-at-a-time sensitivity around the base configuration (designer 11-02)
    b0 = config(a.s, **base_over)
    r0 = b0['sound_rates']
    VAR = [('base', {}),
           ('rates x0.5', dict(sound_rates=tuple(x * 0.5 for x in r0))),
           ('rates x2', dict(sound_rates=tuple(min(0.6, x * 2) for x in r0))),
           ('size small', dict(pfc_size='small')), ('size large', dict(pfc_size='large')),
           ('band every depth', dict(band_depth_sound=4, band_rel=0.0)),
           ('g_loop x0.5', dict(g_loop=b0['g_loop'] * 0.5)), ('g_loop x2', dict(g_loop=b0['g_loop'] * 2)),
           ('w44 0', dict(w44=0.0)), ('w44 x2', dict(w44=max(0.5, b0['w44'] * 2))),
           ('g31 0.3', dict(g31=0.3)), ('g44 0.3', dict(g44=0.3)),
           ('eta x0.25', dict(eta=b0['eta'] / 4, eta_m=b0['eta_m'] / 4)), ('w_ref x4', dict(w_ref=b0['w_ref'] * 4)),
           ('decay faster', dict(gamma_pfc=0.9995, gamma_motor=0.999)),
           ('g_pm +0.5', dict(pm_extra=0.5)), ('g_ih x2', dict(g_ih=b0['g_ih'] * 2)),
           ('g_hh, g_h1 0.3', dict(g_hh=0.3, g_h1=0.3)), ('kh 25', dict(kh=25)), ('kh 100', dict(kh=100))]
    VAR = VAR[a.part::a.nparts]
    for label, extra in VAR:
        o = {**base_over, **extra}
        name = name_of(o)
        c, traj = build(life, o)
        act = probes.activity(life, 5 if a.quick else 30)
        cod = probes.newborn_codes(life, n_ctx=2 if a.quick else 4, n_noise=1, lags=12)
        st = probes.stability(life, 30 if a.quick else a.l_dialogues)
        save_json(os.path.join(a.out, 'sens_' + name + '.json'), dict(label=label, config=o, calib=life.brain.calib,
                                                                       n1=life.brain.n1, birth_budget=act,
                                                                       a1_pass=probes.a1_pass(act), codes=cod,
                                                                       stability=st))
        w = st['windows'][-1] if st['windows'] else {}
        log('sens', label, 'n1', life.brain.n1, 'L4 frac birth', round(act['L4']['speech'] / life.brain.pfc.n, 3),
            'a1', probes.a1_pass(act), 'a2', cod['a2_pass'], 'a3', cod['a3_pass'], 'lag12',
            round(cod['L4_decode'][12], 3), 'Hs lag0', round(cod['Hs_decode'][0], 3) if 'Hs_decode' in cod else None,
            'a1L', st['pass_'], 'last window', {k: round(v, 2) for k, v in w.items() if isinstance(v, float)})

else:  # a1L: learning-on stability with the remedy ladder (§四.10 order); every rung is registered
    LADDER = [
        ('R0 as chosen', {}),
        ('R1 learned gains 0.3', dict(g44=0.3, g_hh=0.3, g_h1=0.3)),
        ('R2 learned gains 0.1', dict(g44=0.1, g_hh=0.1, g_h1=0.1)),
        ('R3 gains 0.1 + slower rates', dict(g44=0.1, g_hh=0.1, g_h1=0.1, eta=0.005, eta_m=0.0025, eta_h=0.1,
                                             eta_h_m=0.05)),
        ('R4 + stronger inhibition (step 2)', dict(g44=0.1, g_hh=0.1, g_h1=0.1, eta=0.005, eta_m=0.0025,
                                                   eta_h=0.1, eta_h_m=0.05, g_ih=2.0, pm_extra=0.5)),
    ]
    for over in json.loads(open(a.configs).read()):
        for rung, extra in LADDER:
            o = {**base_over, **over, **extra}
            name = name_of(o)
            c, traj = build(life, o)
            act = probes.activity(life, 5 if a.quick else 30)
            st = probes.stability(life, 30 if a.quick else a.l_dialogues)
            save_json(os.path.join(a.out, 'a1L_' + name + '.json'), dict(config=o, rung=rung, calib=life.brain.calib,
                                                                         birth_budget=act, stability=st))
            w = st['windows'][-1] if st['windows'] else {}
            log('a1L', name, rung, 'pass', st['pass_'], 'last window', {k: round(v, 2) for k, v in w.items()
                                                                         if isinstance(v, float)})
            if st['pass_']:
                break
