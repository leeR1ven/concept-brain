"""All parameters in one place. s scales the whole brain (ARCH 五: the whole brain grows together)."""


def config(s=1, **over):
    c = dict(
        s=s,
        # world physics (the world, not the brain): cochlea levels and thresholds, own voice at 0.6 through the bone
        levels_per_s=3, th_lo=0.12, th_hi=1.0, own_level=0.6, voice_level=0.9, noise_sd=0.04, e_max=1.5,
        lineage_seed=4321, birth_dialogues=200,
        syl_ticks=3, gap_turn=(4, 10), gap_dlg=(12, 25),   # ticks per syllable (1: the whole syllable at once, three
        # bands in the cochlea); pauses between turns and dialogues (shorter with 1 tick per syllable)
        syl_hold=1,                 # syl_ticks 1: ticks the whole syllable is held (2: the concept area, a tick behind
        # the sound area, and the sound area reach the PFC on the same syllable at its last tick; 11-10)
        # local nets, the same for every area (ARCH 二.4): K inputs per unit, units per grid position, windows by depth
        K=6, upp=2, dy=(1, 2, 4, 8), dz=(2, 3, 4, 6),
        sound_rates=(0.06, 0.036, 0.024, 0.015), touch_rates=(0.06, 0.036),  # fewer lit deeper (二.5); PFC under 5 % (designer 11-03)
        band=(0.002, 0.6),          # birth firing rate of a feature unit; outside it: no chain (ARCH 三 链式神经元)
        g_ret=1.05,                 # a return alone lights its unit (x the unit's own threshold)
        # PFC (ARCH 三 前额叶)
        g_loop=1.0, loop_fan=30, theta1=1.0, r_learn=0.15, g44=0.1, g31=1.0, pfc_mid_rate=0.3,
        pfc_depth=4,                # PFC layers: L1 (copies), middle layers, the last; they only put time between in and out
        l4_usual=False,             # the last layer lights max(this tick's input count, the usual one) (quiet moments)
        check_step=0.3, check_margin=2.0,    # last layer -> first layer, each copy's own check (ARCH 四, 11-07)
        predict_from=('L4',),       # the PFC layers that learn what comes next in L1 (11-10: 'L1','L2','L3','L4')
        pfc_from_concept=False,     # True: the PFC (and hippocampus) get the senses only through the concept area's
        # copies; force and stop copies stay direct (designer 11-10: the PFC sits behind association areas)
        predict_from_concept=False, # True: in the first layer only the concept copies learn what comes next (11-10:
        # with all copies the PFC stays at word-frequency level; with the concept copies near previous-syllable counts)
        pfc_graded=False,           # True (with a concept area): what comes next is learned by the graded table among
        # the copies the PFC hears (net.GradedTable: change = step x (came - push), excitatory links and inhibitory
        # partner units, no link changes sign; designer 11-10 "可以，你们测试决定"). graded_step: its step
        graded_step=0.15,
        graded_gain=3.0,            # the graded table's push (a share 0..1) x this = its push in the copies' units, where
        # theta1 = 1 lights a thought: 3 = a copy that always follows is pushed to theta + margin; a code's copies each
        # come in only part of its syllable's hearings, so their shares are small and a larger gain may be needed
        concept_rule='',            # 'b': the concept area's learned links (every sound feature -> every unit, E / I >= 0)
        # change by the graded check (grow.ConceptArea.step_b; bench 11-10: with something fixed lighting a syllable's
        # units half of the time, the sound alone comes to light the same units: identity 0.82 -> 0.90, A->B 0.47 -> 0.67)
        concept_step=0.15,
        concept_ret_teach=False,    # True: units lit only by a thought's return count as "lit" in the concept check (their
        # echo teaches the sound links; build 9 first runs: ceiling 0.78 -> 0.30-0.44, so off; a return still lights its unit)
        sight_bits=0,               # > 0: a second sense for the concept area - the world shows a fixed pattern of this
        # many bits with a share p_see of the syllables heard (its written sign); probes are by sound alone
        p_see=0.5,
        sight_area=False,           # True: what is seen goes through a visual area made like the others (sight_bits pixels,
        sight_rates=(0.25, 0.12),   # one thermometer pair each; its feature copies reach the concept area by chains:
        # concept_from must name ('sight', 0)); False: the sight bits are wired straight to concept units (stand-in)
        sight_to_code=True,         # False (test switch): seen units take part in the check but are not part of the code sent on
        sight_ref='usual',          # what a seen sign is compared with: each unit's usual drive while something is seen
        # (ARCH 二.4); 'rest' = the drive when nothing is seen (build 9 first runs: generic units, sets overlap 13 %)
        concept_ret_adds=False,     # True: a return lights its concept unit beside the heard ones (does not replace one)
        hip_from_heard=True,        # with pfc_from_concept: the hippocampus's birth fan-in is drawn from the copies it hears
        hip_fan=150,                # (concept / force / stop), hip_fan per unit (11-11)
        hip_recall=True,            # False: the hippocampus's recall lights nothing (test switch: the PFC alone)
        chain_brake_stop=True,      # False: stop neurons lit only by the innate brake send no chain signal to the PFC
        tau_usual=1000,            # ticks over which a copy's own usual internal drive is averaged (ARCH 二.4)
        context=0.1,                # excitability: copies lit from inside <= context x the usual external input count
        # hippocampus (ARCH 三 海马体): own population, fixed count kh; recall lights the context share most driven
        nh=20000, kh=50, rec_fan=24, n_hh=500, n_h1=2000, g_hh=0.01,
        # the shared concept area (ARCH 三 抽象概念区): nc x s units, kc x s on per tick, n_cl candidate links per feature
        nc=2000, kc=20, n_cl=64, g_c=1.0,
        eta_concept=0.0,            # feature -> concept learning; 0 = birth links only (11-10: learning by co-activity
        # merged the syllables' codes, 0.05 -> 0.6 cosine, and Hebbian association fell from n-gram level to chance)
        concept_from=None,          # what the concept area reads. None: every sensory feature. A list: a number is that
        # layer of the sound area, a pair is (area, layer), e.g. (0,) or (0, ('touch', 0)). (Until 11-10 a number also
        # took that layer of the touch area: a life made before keeps what it was made with.) (0,): the first, 11-10: the
        # deeper fixed layers lag a tick each and keep less of what was heard: syllable told apart 46 % vs 24 %)
        concept_sync=False,         # True: the concept copies reach the PFC on the same tick as the features the area
        # read (it is part of the sensing, like layer 1 reading the ear on the same tick); False: a tick later. With one
        # tick per syllable, a tick later means the PFC links each syllable's concept to the next-but-one (11-10)
        # motor (ARCH 三 运动区): m force neurons per muscle (x s), blockers, stop neurons, brake, collapse pools
        speak=False,                # True (big world, syl_ticks 1, with a concept area): the child can speak (core/speak.py):
        # 72 motor channels, the generation-0 speech link from thought concept copies, motor pools, the mouth
        speak_n=600, speak_unit_max=0.05, speak_rel=1.5, speak_share=0.5,
        mouth_tol=1.0,              # < 1: a posture near a syllable's (Jaccard >= mouth_tol) says that syllable
        mirror=False, mirror_min=0.5,   # True: a syllable heard silently lights its intent copies for the graded check n_mus=24, mus_per_syl=4,
        reward=False,               # True (with speak): the adults are pleased when the child says one of the next r_ahead
        r_ahead=3, r_match=1.0, r_near=0.5,   # syllables they are about to say (r_match), or it in another tone (r_near)
        m=8, theta_m=0.3, b_block=1.2, c_stop=2.0, n_brake=8, theta_stop=1.0, g_ret_force=0.95,
        mc_n=16, mc_e=3.0, mc_i=4.0,
        # the one rule and the one decay (ARCH 四); decay per decay_every ticks: hippocampus fastest, PFC slowest
        eta=0.02, eta_h=0.3, w_ref=1.0, pow=1.0, floor=0.01, fall_win=2, decay_every=200, prune=1e-3,
        gamma_hip=0.99, gamma_motor=0.9999, gamma_pfc=0.99995, gamma_concept=None,
        # generation 0 (ARCH 五): hand-wired speech link and collapse pools, stored at a well-used strength
        gen0_strength=20.0, g1p_rel=1.5, g1p_p=0.3, g1p_k=30,
        # reward tool (ARCH 四)
        r_win=24, r_pleased=2.0, r_upset=1.0, r_stop=2.0,
        # ARCH 四: the reward changes only the last layer -> action (force / stop) copy links, 别的连接不动 ('all': every
        # learned table, the 11-11 behaviour; Fable a35 4b: a gamed reward then also amplified concept / graded learning)
        reward_scope='action',
    )
    for k, v in over.items():
        if k not in c:
            raise KeyError(f'unknown parameter {k}')
        c[k] = v
    return c
