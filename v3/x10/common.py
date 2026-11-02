"""X10 shared config and helpers. Every entry below is a register item of PLAN.md (ids in brackets)."""
import json
import os
import time

import numpy as np
import torch

DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if DEV.type == 'cuda':                                     # runner etiquette: 18 GB per job (36 GB with BIGGPU)
    torch.cuda.set_per_process_memory_fraction(float(os.environ.get('X10_GPU_FRAC', '0.44')))


def config(s=1, **over):
    c = dict(
        s=s,
        # world / physics [W1-W4]
        own_level=0.6, voice_level=0.9, noise_sd=0.04, e_max=1.5, th_lo=0.12, th_hi=1.0, levels_per_s=3,
        lineage_seed=4321,
        # sound area [S1-S6]
        n_layers=4, units_per_pos=2, sound_fan=6, sound_dy=(1, 2, 4, 8), sound_dz=(2, 3, 4, 6),
        sound_rates=(0.20, 0.12, 0.08, 0.05),
        # chain band per area (designer 11-02: which units count as always-on / never-on / feature units are tunable
        # parameters per area, chosen by the experiment, not learned): (low, high) birth firing-rate cut.
        # band_sound may also be a list of 4 (low, high) pairs, one per layer. Motor: the newborn never moves, so
        # force neurons have birth rate 0; the default low cut 0.0 keeps them (tonic blocking, rate 1.0, is cut).
        band=(0.002, 0.90), band_sound=None, band_motor=(0.0, 0.90),
        # designer 11-02: feature status is a tunable parameter per area, and a large PFC is a hypothesis to test.
        # Sound layers shallower than band_depth_sound use `band`; deeper layers keep a unit iff its birth rate is
        # >= band_rel x that layer's own target rate (and <= band[1]); band_rel = 0 chains every deeper unit.
        # Presets (pfc_size): small / medium (default) / large, see PFC_SIZE below.
        band_depth_sound=1, band_rel=0.9, pfc_size='medium',
        # chains [C1]: return drive in units of the target unit's own threshold (>1: the return alone lights S)
        g_ret=1.05,
        # PFC [P1-P12]
        ff_fan=24, r_ff=0.10, r_learn=0.15, patch=33, inhib='P1', g_pm=1.0, pm_extra=0.0,
        loop='recip', loop_fan=30, g_loop=1.0, lat_fan=20, w44=0.5, theta1=1.0, g31=1.0, g44=1.0,
        post='chain', rho=1, ctx_k=0.0,
        # learning [R1-R4]
        eta=0.02, eta_m=0.01, w_ref=1.0, floor=0.01, fall_win=2, decay_every=200, prune=1e-3,
        gamma_pfc=0.99995, gamma_motor=0.9999, gamma_h_base=0.01,
        # hippocampus [H1-H5]
        hip=True, nh_per_s=5000, kh=50, rec_fan=24, chh_per_s=500, ch1_per_s=200, eta_h=0.3, eta_h_m=0.15,
        theta_h1=1.0, g_h1=1.0, g_hh=1.0, g_ih=1.0,
        # motor [M1-M6]
        theta_m=0.3, b_block=0.8, c_stop=2.0, ladder_step=0.25, n_brake_per_s=8.0, theta_stop=1.0,
        g_ret_force=1.3, force_per_muscle=0, mc=False, mc_nI=16, mc_k=4, mc_e=3.0, mc_i=4.0, mc_src='force',           # 0: m = s (arm M sets m explicitly)
        # gen-0 speech wiring G1P (designer 11-02: inside the PFC, L4 syllable units -> L1 force copies)
        g1p_rel=1.5, g1p_own_excl=False, g1p_own_cut=0.2, g1p_p=0.5, g1p_margin=0.3, g1p_src='heard', g1p_evoked=False, g1p_off=0.2,
        # concept area (arm C+ only)
        concept=False,
        # birth period (dialogues, learning off)
        birth_dialogues=200, calib_ticks=3000, calib_online_ticks=4000,
        # implementation only: tables up to this many entries are scanned whole each tick (no host sync)
        dense_limit=0,
        # P1 (11-02, ARCH 2.3): upper chain-band cut on A1 and A2 (near-input often-on units carry no features)
        band_cut_a12=None,
        # gen-0 hand-wired links (G1P, MC) stored at the strength of a well-used link, in units of w_ref (0: as computed)
        gen0_strength=0.0,
        # fixes 11-02 (arms; defaults reproduce the old build): [A] PFC L4 count = |L3(t-1)| (k most-driven after patch
        # inhibition, positive drive; L2, L3 keep their birth thresholds); [B] Hs = kh most-driven units per tick
        pfc_count='theta', hs_count='theta',
        # reward tool [R6] (ARCH 2.8; form and parameters from X8): changes of the last r_win ticks, fading;
        # pleased adds them again x r_pleased; upset adds them again x -(1 + r_upset) (net reversal) and links the
        # speech state (L4 now) to the stop copies by r_stop x the rule's step
        reward=False, r_win=24, r_pleased=2.0, r_upset=1.0, r_stop=2.0, r_reverse='all',
        # touch area incl. muscle force [T1-T5] (designer 11-02, ARCH 2.2): input = each muscle's actual force as
        # thermometer pairs on touch_levels levels (0: = force neurons per muscle, so it scales with s); local layers
        # like the sound area (touch_upp units per position is the size proportion); chain band with the P1 upper cut
        touch=False, touch_levels=0, touch_layers=2, touch_upp=2, touch_fan=6, touch_dy=(1, 2), touch_dz=(2, 4),
        touch_rates=(0.20, 0.12), band_touch=(0.002, 0.6),
    )
    if 'pfc_size' in over:
        c.update(PFC_SIZE[over['pfc_size']])
    for k, v in over.items():
        if k not in c:
            raise KeyError(f'unknown config key {k}')
        c[k] = v
    if c['band_cut_a12'] is not None:
        h = c['band_cut_a12']
        c['band_depth_sound'] = 2
        c['band_sound'] = [(c['band'][0], h), (c['band_rel'] * c['sound_rates'][1], h)]
    return c


# PFC-size presets from the chain-band knobs (registered; sizes at s = 1 measured by sizing/band_sizes.py)
PFC_SIZE = {
    'small': dict(band_depth_sound=0, band_rel=0.95),     # ~1.8k sound copies: units near their layer's target rate
    'medium': dict(band_depth_sound=1, band_rel=0.9),     # ~4.1k: A1 by the band, deeper units >= 0.9 r_l
    'large': dict(band_depth_sound=1, band_rel=0.0),      # ~8.2k: A1 by the band, every deeper unit
}


def gammas(c):
    """[R3] one global decay per 200 ticks; hippocampus fastest, PFC (deepest) slowest."""
    return dict(hip=1.0 - c['gamma_h_base'] / c['s'], motor=c['gamma_motor'], pfc=c['gamma_pfc'])


class Timer:
    def __init__(self):
        self.t = {}

    def add(self, k, dt):
        self.t[k] = self.t.get(k, 0.0) + dt


class Prof:
    """per-section wall time with device synchronisation (profiling only)"""

    def __init__(self):
        self.t, self.last = {}, None

    def mark(self):
        sync()
        self.last = time.time()

    def lap(self, k):
        sync()
        now = time.time()
        self.t[k] = self.t.get(k, 0.0) + now - self.last
        self.last = now

    def __bool__(self):
        return True


def sync():
    if DEV.type == 'cuda':
        torch.cuda.synchronize()


def save_json(path, obj):
    def conv(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, torch.Tensor):
            return o.detach().cpu().tolist()
        raise TypeError(type(o))
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(obj, f, default=conv, indent=1)
    os.replace(tmp, path)


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)
