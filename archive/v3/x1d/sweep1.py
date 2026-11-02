"""phase-1 sweep driver: one process, configs run one after another.
  python3 -I sweep1.py <listname> [seed] [birth_ex] [n_ctx]
Config lists are in CONFIGS below; results go to out1/<name>_s<seed>.json and a summary line to out1/SWEEP_<list>.txt
"""
import sys, os, json, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'x1c'))   # repo: core3.py, world.py, run.py from ../x1c
from phase1 import measure

K0 = dict(kappa=0.0)
CONFIGS = {
    'v1': [
        # ---- global quantile threshold (shortcut reference)
        ('G_ff', dict(act='global', g_loop=0.0, w33=0.0)),
        ('G_lat0.4', dict(act='global', g_loop=0.0, w33=0.4)),
        ('G_lat1', dict(act='global', g_loop=0.0, w33=1.0)),
        ('G_loop2', dict(act='global', g_loop=2.0, w33=0.1, pred_rel=False)),
        ('G_loop4', dict(act='global', g_loop=4.0, w33=0.1, pred_rel=False)),
        ('G_loop8', dict(act='global', g_loop=8.0, w33=0.4, pred_rel=False)),
        ('G_loop8rel', dict(act='global', g_loop=8.0, w33=0.1, pred_rel=True, f1=0.2)),
        ('G_k80', dict(act='global', g_loop=4.0, w33=0.1, pred_rel=False, k2=60, k3=80, k_in=20)),
        ('G_fan', dict(act='global', g_loop=4.0, w33=0.1, pred_rel=False, fan12=60, fan23=80, fan31=60)),
        # ---- local feedback inhibition (fallback)
        ('A_g2', dict(act='local', g_loop=2.0, w33=0.1, pred_rel=False)),
        ('A_g4', dict(act='local', g_loop=4.0, w33=0.4, pred_rel=False)),
        ('A_g4_gl3', dict(act='local', g_loop=4.0, w33=0.4, pred_rel=False, g_loc=0.3)),
        ('A_g4_gl3rel', dict(act='local', g_loop=4.0, w33=0.4, pred_rel=True, f1=0.2, g_loc=0.3)),
        ('A_lat1_gl3', dict(act='local', g_loop=0.0, w33=1.0, g_loc=0.3)),
        # ---- no PFC inhibition: thresholds a fraction of each neuron's own in-weight sum (main arm)
        ('N_g4', dict(act='frac', g_loop=4.0, w33=0.1, f1=0.35, f2=0.15, f3=0.2)),
        ('N_g4_f1lo', dict(act='frac', g_loop=4.0, w33=0.1, f1=0.15, f2=0.15, f3=0.2)),
        ('N_lat1', dict(act='frac', g_loop=4.0, w33=1.0, f1=0.2, f2=0.15, f3=0.2)),
        ('N_lat1_f3', dict(act='frac', g_loop=4.0, w33=1.0, f1=0.2, f2=0.15, f3=0.15)),
        ('N_own', dict(act='own', g_loop=4.0, w33=0.4)),
    ],
    'v2': [
        # ---- no PFC inhibition (frac): scan towards the boundary between dying and saturating
        ('N_C', dict(act='frac', g_loop=0.0, w33=0.1, f2=0.22, f3=0.3)),                       # X1c arm C
        ('N_f1.6', dict(act='frac', g_loop=4.0, w33=0.1, f1=0.6, f2=0.22, f3=0.3)),
        ('N_f1.75', dict(act='frac', g_loop=4.0, w33=0.1, f1=0.75, f2=0.22, f3=0.3)),
        ('N_f1.6_f3.25', dict(act='frac', g_loop=4.0, w33=0.1, f1=0.6, f2=0.22, f3=0.25)),
        ('N_lat.5_f3.3', dict(act='frac', g_loop=0.0, w33=0.5, f2=0.22, f3=0.3)),
        ('N_lat.5_f3.35', dict(act='frac', g_loop=0.0, w33=0.5, f2=0.22, f3=0.35)),
        # ---- global: low fan-in (sensitivity ~ 1)
        ('G_lowK', dict(act='global', g_loop=4.0, w33=0.4, pred_rel=False, fan12=8, fan23=8, fan31=6, fan33=4)),
        ('G_lowK2', dict(act='global', g_loop=4.0, w33=0.4, pred_rel=False, fan12=6, fan23=6, fan31=4, fan33=3)),
        ('G_lowK_lat', dict(act='global', g_loop=0.0, w33=1.0, fan12=8, fan23=8, fan33=3)),
        ('G_lowK_mute', dict(act='global', g_loop=4.0, w33=0.4, pred_rel=False, fan12=8, fan23=8, fan31=6, fan33=4, mouth=False)),
        ('G_loop4_mute', dict(act='global', g_loop=4.0, w33=0.1, pred_rel=False, mouth=False)),
        # ---- local: lower inhibition gain, low fan-in
        ('A_lowK', dict(act='local', g_loop=4.0, w33=0.4, pred_rel=False, fan12=8, fan23=8, fan31=6, fan33=4, g_loc=0.3)),
    ],
    'v3': [
        # ---- motor default: blocking wins (hand-wired speech basics allowed, designer 11-01); babble at birth
        ('G_lowK_m1', dict(act='global', g_loop=4.0, w33=0.4, pred_rel=False, fan12=8, fan23=8, fan31=6, fan33=4, g_ret_mot=0.3, c_mb=0.4)),
        ('G_lowK_m2', dict(act='global', g_loop=4.0, w33=0.4, pred_rel=False, fan12=8, fan23=8, fan31=6, fan33=4, g_ret_mot=0.3, c_mb=0.4, theta_m=0.4)),
        # ---- traverse the ordered-generic <-> chaotic-specific transition (mute = own voice not heard, reference)
        ('G_K12_mute', dict(act='global', g_loop=4.0, w33=0.4, pred_rel=False, fan12=12, fan23=12, fan31=10, fan33=6, mouth=False)),
        ('G_K16_mute', dict(act='global', g_loop=4.0, w33=0.2, pred_rel=False, fan12=16, fan23=16, fan31=15, fan33=8, mouth=False)),
        ('G_K8g2_mute', dict(act='global', g_loop=2.0, w33=0.2, pred_rel=False, fan12=8, fan23=8, fan31=6, fan33=4, mouth=False)),
        ('G_K8lat_mute', dict(act='global', g_loop=4.0, w33=1.0, pred_rel=False, fan12=8, fan23=8, fan31=6, fan33=4, mouth=False)),
        ('G_K8k20_mute', dict(act='global', g_loop=4.0, w33=0.4, pred_rel=False, fan12=8, fan23=8, fan31=6, fan33=4, k3=20, k2=15, k_in=6, mouth=False)),
        ('G_K24lat_mute', dict(act='global', g_loop=4.0, w33=0.4, pred_rel=False, mouth=False)),
        # ---- no inhibition, fine scan of the boundary
        ('N_f1.65', dict(act='frac', g_loop=4.0, w33=0.1, f1=0.65, f2=0.22, f3=0.3)),
        ('N_f1.7', dict(act='frac', g_loop=4.0, w33=0.1, f1=0.7, f2=0.22, f3=0.3)),
    ],
    'v4': [
        # ---- reciprocal birth loop (designer msg 84: G returns to A), three activity controls
        ('G_rec4', dict(act='global', birth31='recip', g_loop=4.0, w33=0.1, pred_rel=False)),
        ('G_rec8', dict(act='global', birth31='recip', g_loop=8.0, w33=0.1, pred_rel=False)),
        ('G_rec4_k20', dict(act='global', birth31='recip', g_loop=4.0, w33=0.1, pred_rel=False, k_in=20)),
        ('G_rec4_lat', dict(act='global', birth31='recip', g_loop=4.0, w33=0.4, pred_rel=False)),
        ('A_rec4', dict(act='local', birth31='recip', g_loop=4.0, w33=0.1, pred_rel=False, g_loc=0.3)),
        ('N_rec_f1.5', dict(act='frac', birth31='recip', g_loop=4.0, w33=0.1, f1=0.5, f2=0.22, f3=0.3)),
        ('N_rec_f1.3', dict(act='frac', birth31='recip', g_loop=4.0, w33=0.1, f1=0.3, f2=0.22, f3=0.3)),
        # ---- stronger input reset: weak internal L1 -> L2 and weak lateral during the question
        ('G_K12_gi.2', dict(act='global', g_loop=4.0, w33=0.1, pred_rel=False, fan12=12, fan23=12, fan31=10, fan33=6, g_int=0.2)),
        ('G_K12_gi.2_mute', dict(act='global', g_loop=4.0, w33=0.1, pred_rel=False, fan12=12, fan23=12, fan31=10, fan33=6, g_int=0.2, mouth=False)),
        ('G_rec4_gi.2', dict(act='global', birth31='recip', g_loop=4.0, w33=0.1, pred_rel=False, g_int=0.2)),
    ],
    'v5': [
        # ---- coordinator idea 1: slow bounded divisive gain per L2/L3 unit (designer's "self-adjusting threshold by
        #      long-run activity", divisive form)
        ('G_rec4_lat_dg', dict(act='global', birth31='recip', g_loop=4.0, w33=0.4, pred_rel=False, dgain=True)),
        ('G_loop4_dg', dict(act='global', g_loop=4.0, w33=0.1, pred_rel=False, dgain=True)),
        ('G_lowK_dg', dict(act='global', g_loop=4.0, w33=0.4, pred_rel=False, fan12=8, fan23=8, fan31=6, fan33=4, dgain=True)),
        ('G_rec4_lat_dg4', dict(act='global', birth31='recip', g_loop=4.0, w33=0.4, pred_rel=False, dgain=True, dg_lo=0.25, dg_hi=4.0)),
        # ---- local inhibition applied within the same tick (reader's code finding)
        ('A_same', dict(act='local', birth31='recip', g_loop=4.0, w33=0.1, pred_rel=False, loc_same=True)),
        ('A_same_dg', dict(act='local', birth31='recip', g_loop=4.0, w33=0.1, pred_rel=False, loc_same=True, dgain=True)),
        # ---- no inhibition + X3 patch-mean subtraction of loop / prediction drive (fallback)
        ('N_pm1', dict(act='frac', birth31='recip', g_loop=4.0, w33=0.1, f1=0.5, f2=0.22, f3=0.3, pm=1.0)),
        ('N_pm1_lat', dict(act='frac', birth31='recip', g_loop=4.0, w33=0.5, f1=0.5, f2=0.22, f3=0.25, pm=1.0)),
    ],
    "conf": [
        ("G_rec4_lat", dict(act="global", birth31="recip", g_loop=4.0, w33=0.4, pred_rel=False)),
        ("G_loop8", dict(act="global", g_loop=8.0, w33=0.4, pred_rel=False)),
        ("G_lowK_mute", dict(act="global", g_loop=4.0, w33=0.4, pred_rel=False, fan12=8, fan23=8, fan31=6, fan33=4, mouth=False)),
        ("N_pm1", dict(act="frac", birth31="recip", g_loop=4.0, w33=0.1, f1=0.5, f2=0.22, f3=0.3, pm=1.0)),
        ("A_g4", dict(act="local", g_loop=4.0, w33=0.4, pred_rel=False)),
    ],
    "x4": [
        ("G_rec4_lat_x4", dict(act="global", birth31="recip", g_loop=4.0, w33=0.4, pred_rel=False, n2=1600, n3=1600, k2=120, k3=160, k_in=10)),
    ],
}


def summary(name, r):
    s, c = np.array(r['same']), np.array(r['cross'])
    L = slice(10, 13)
    within = np.array(r['within_J1'][1:13])
    d = dict(name=name, same0=s[0], cross0=c[0], same12=round(float(s[L].mean()), 3), cross12=round(float(c[L].mean()), 3),
             gap12=round(float((s - c)[L].mean()), 3), dq12=round(float(np.mean(r['dec_q'][10:13])), 3),
             dfill12=round(float(np.mean(r['dec_fill_given_frame'][10:13])), 3),
             dfr12=round(float(np.mean(r['dec_frame'][10:13])), 3), noise12=round(float(np.mean(r['noise_same'][10:13])), 3),
             dmg12=r['damage']['0.1'][12], within=round(float(within.mean()), 3),
             n3=round(float(np.mean(r['n3_lag'])), 1), n3max=max(r['n3_lag_max']), pempty=round(float(np.mean(r['p_empty_lag'])), 3),
             mouth=round(float(np.mean(r['mouth_lag'])), 2), long=r['motion_long_silence'], t=r['time_s'])
    return d


if __name__ == '__main__':
    lst = sys.argv[1]
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    bex = int(sys.argv[3]) if len(sys.argv) > 3 else 200
    nctx = int(sys.argv[4]) if len(sys.argv) > 4 else 4
    only = sys.argv[5].split(',') if len(sys.argv) > 5 else None
    os.makedirs(os.path.join(HERE, 'out1'), exist_ok=True)
    fs = open(os.path.join(HERE, 'out1', f'SWEEP_{lst}_s{seed}.txt'), 'a')
    for name, cfg in CONFIGS[lst]:
        if only and name not in only:
            continue
        cfg = dict(K0, **cfg)
        mouth = cfg.pop('mouth', True)
        r = measure(cfg, seed=seed, n_ctx=nctx, birth_ex=bex, mouth=mouth)
        r['birth_ex'] = bex
        json.dump(r, open(os.path.join(HERE, 'out1', f'{name}_s{seed}.json'), 'w'), indent=1, default=str)
        sm = summary(name, r)
        print('SUMMARY', json.dumps(sm), flush=True)
        fs.write(json.dumps(sm) + '\n')
        fs.flush()
