"""All parameters in one place. s scales the whole brain (ARCH 五: the whole brain grows together)."""


def config(s=1, **over):
    c = dict(
        s=s,
        # world physics (the world, not the brain): cochlea levels and thresholds, own voice at 0.6 through the bone
        levels_per_s=3, th_lo=0.12, th_hi=1.0, own_level=0.6, voice_level=0.9, noise_sd=0.04, e_max=1.5,
        lineage_seed=4321, birth_dialogues=200,
        # local nets, the same for every area (ARCH 二.4): K inputs per unit, units per grid position, windows by depth
        K=6, upp=2, dy=(1, 2, 4, 8), dz=(2, 3, 4, 6),
        sound_rates=(0.20, 0.12, 0.08, 0.05), touch_rates=(0.20, 0.12),     # fewer lit deeper (ARCH 二.5)
        band=(0.002, 0.6),          # birth firing rate of a feature unit; outside it: no chain (ARCH 三 链式神经元)
        g_ret=1.05,                 # a return alone lights its unit (x the unit's own threshold)
        # PFC (ARCH 三 前额叶)
        g_loop=1.0, loop_fan=30, theta1=1.0, r_learn=0.15, g44=0.1, g31=1.0,
        tau_usual=1000,             # ticks over which a copy's own usual internal drive is averaged (ARCH 二.4)
        context=0.1,                # excitability: copies lit from inside <= context x the external input count (0.3 runs away)
        # hippocampus (ARCH 三 海马体): own population, fixed count kh; recall lights the context share most driven
        nh=20000, kh=50, rec_fan=24, n_hh=500, n_h1=2000, g_hh=0.01,
        # motor (ARCH 三 运动区): m force neurons per muscle (x s), blockers, stop neurons, brake, collapse pools
        m=8, theta_m=0.3, b_block=1.2, c_stop=2.0, n_brake=8, theta_stop=1.0, g_ret_force=0.95,
        mc_n=16, mc_e=3.0, mc_i=4.0,
        # the one rule and the one decay (ARCH 四); decay per decay_every ticks: hippocampus fastest, PFC slowest
        eta=0.02, eta_h=0.3, w_ref=1.0, pow=1.0, floor=0.01, fall_win=2, decay_every=200, prune=1e-3,
        gamma_hip=0.99, gamma_motor=0.9999, gamma_pfc=0.99995,
        # generation 0 (ARCH 五): hand-wired speech link and collapse pools, stored at a well-used strength
        gen0_strength=20.0, g1p_rel=1.5, g1p_p=0.3, g1p_k=30,
        # reward tool (ARCH 四)
        r_win=24, r_pleased=2.0, r_upset=1.0, r_stop=2.0,
    )
    for k, v in over.items():
        if k not in c:
            raise KeyError(f'unknown parameter {k}')
        c[k] = v
    return c
