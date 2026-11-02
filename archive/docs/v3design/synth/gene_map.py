"""Map every one of today's 265 named knobs (knob_list.out) onto the v3 genome (30 genes), fixed species constants,
sizes (x s), the lineage seed, or deletion.  PROVED by count.  Run: python3 -I gene_map.py"""
import re, collections

GENES = {
    'G1': 'a_mix  conjunction-area active fraction', 'G2': 'k_th   items held in the slow persistent area',
    'G3': 'lam    held-content factor when a new event loads', 'G4': 'th_acc category acceptance threshold',
    'G5': 'tau_set adaptive set-point time constant', 'G6': 'k_on   present-margin above the set-point (ear, eye, brake)',
    'G7': 'tau_fat fatigue time constant', 'G8': 'w_td   top-down (restoration) weight cap',
    'G9': 'T0     shortest time-cell bin', 'G10': 'eta_p  prediction-form rate',
    'G11': 'eta_v  value-form rate (actor, gates)', 'G12': 'eta_r  representation-form rate floor (sensory)',
    'G13': 'r_eta  plasticity ratio per hierarchy level', 'G14': 'tau_e  eligibility time constant',
    'G15': 'T_pr   prune age in the unit\'s own firings', 'G16': 'a_dg   DG active fraction',
    'G17': 'h_hip  hippocampal half-life / capacity', 'G18': 'rho_rec recency exponent of recall',
    'G19': 'th_rec recall overlap threshold', 'G20': 'n_rep  reactivations per quiet period',
    'G21': 'u      urgency per tick', 'G22': 'eps    exploration rate', 'G23': 'tau_ref refractoriness after release',
    'G24': 'gamma  TD discount per tick', 'G25': 'tau_mu ACh-like mode time constant',
    'G26': 'r_bab  spontaneous babble rate', 'G27': 'w_food value: hunger relief',
    'G28': 'w_resp value: being responded to', 'G29': 'w_plea value: other pleased',
    'G30': 'w_upset value: other upset or hurt (>= 1.5 w_plea)'}
NEW = {'G8', 'G9', 'G22', 'G23', 'G24', 'G27', 'G28', 'G29', 'G30'}     # no knob today (world reward numbers stand in for 27-30)

M = {}
def put(cat, names):
    for n in names.split():
        M[n] = cat
# ---- Cfg (85)
put('size', 'V n_eye hip_cap hip_ndg'); put('seed', 'seed')
put('G10', 'eta eta_critic eta_recall'); put('G11', 'eta_g eta_actor'); put('G13', 'eta_prime eta_attd')
put('G17', 'hip_gamma'); put('G16', 'hip_kdg'); put('G19', 'hip_theta'); put('G20', 'recall_n'); put('G15', 'prune_age')
put('species', 'gain max_len w_scene')
put('del', 'eta_sleep mature floor use_gen copy_prompt copy_next copy_comp hippo ctx att_learn match kappa focus '
           'reward_actor sleep_replay learn_listen retino branch_norm rf_focus rf_thr match_thr rec_thr vis_rec '
           'self_ground rate_max conf_min match_n attd_n prime_internal mix_cap ctx_n ctx_k gen_mmin sleep_mode '
           'sleep_passes sleep_k hip_bigram own_rotate dialogue_reset role_units gate_noid gate_freq hip_novelty '
           'contingency w_ctx ctx_lam ctx_homeo mat_norm mature_g gate_mmin wd_gen wd_gate wd_every rec_pow rec_boost '
           'split_use split_err split_n recruit_p prune_every ctx_beta ctx_tau match_min match_gap att_thr')
# ---- TCfg (5), SCfg (17)
put('del', 'think_k ior adopt_margin nop_keys scene_recall')
put('G3', 'ct_lam'); put('G2', 'ct_k'); put('G13', 'ct_assoc_eta'); put('G7', 'ct_fat_tau'); put('G18', 'hip_rho')
put('species', 'ct_theta ct_gain ct_fat')
put('del', 'ct ct_tau ct_own ct_surp ct_assoc seg rel name_ctx name_vr_only')
# ---- RCfg (26)
put('del', 'ral ral_W ral_cont ral_theta ral_T ral_strict ral_hip_own ral_replay_own ral_cut ral_init ral_other '
           'ral_W_other ral_q ral_q_eta ral_q_thr ral_q_prior say_rec say_loc say_gain say_loc_thr say_noprompt '
           'scene_held val_act rec_fix')
put('G11', 'eta_act'); put('G14', 'act_T')
# ---- DevAgent (7) (seed, V already mapped)
put('G4', 'theta_z'); put('G12', 'z_rate'); put('G25', 'quiet_ticks'); put('del', 'learn_cats life_eye')
# ---- ThinkAgent (24)
put('species', 'theta_I'); put('G21', 'u_reply'); put('G10', 'eta_oth'); put('G11', 'eta_go')
put('del', 'policy g0_reply g0_scene u_scene beta g_max nogo nogo_late L learn_turn tau_w urg_gate close_on_scene '
           'oth_kind brake_now eye_feats no_react late_release u_R1 own_tail')
# ---- SynthAgent (6) (quiet_ticks already mapped)
put('del', 'wander rebind wander_after turn_feats init_min_tau')
# ---- AgentV21 (5)
put('G14', 'react_T'); put('del', 'voice val_veto dip soc_feats')
# ---- AgentH (16)
put('del', 'h_gate h_back h_relock h_vbound h_nocreate h_conf h_level h_denoise h_kback h_trim h_stuck h_lock h_lvth')
put('G6', 'h_kon'); put('species', 'h_koff'); put('G5', 'h_tau')
# ---- MeanCodes (20) (V already mapped)
put('size', 'D'); put('species', 'km'); put('G13', 'eta_min'); put('G5', 'tau_b')
put('del', 'kx h U m_min n_min mode alpha kcl eta_c tau_cl eps bfloor init max_swap kcls')
# ---- RandMix (20)  (alpha / eta / theta / age clash with other layers' names: handled by layer below)
# ---- Body (16)
# ---- SoundGate (8), AgentV22 (10)

LAYER_OVERRIDE = {   # names whose meaning depends on the layer
    ('RandMix', 'scale'): 'size', ('RandMix', 'fan'): 'species', ('RandMix', 'alpha'): 'G1', ('RandMix', 'n_c'): 'size',
    ('RandMix', 'k_c'): 'species', ('RandMix', 'k_s'): 'species', ('RandMix', 'n_f'): 'size', ('RandMix', 'code'): 'del',
    ('RandMix', 'r_f'): 'species', ('RandMix', 'train'): 'del', ('RandMix', 'eta'): 'G13', ('RandMix', 'every'): 'del',
    ('RandMix', 'prune'): 'del', ('RandMix', 'age'): 'G15', ('RandMix', 'regrow'): 'del', ('RandMix', 'birth'): 'species',
    ('RandMix', 'min_units'): 'del', ('RandMix', 'theta'): 'species', ('RandMix', 'cache_cap'): 'del',
    ('RandMix', 'alloc'): 'del',
    ('Body', 'n_prog'): 'size', ('Body', 'cue_w'): 'del', ('Body', 'end_push'): 'species', ('Body', 'theta'): 'species',
    ('Body', 'rho'): 'species', ('Body', 'end_rule'): 'species', ('Body', 'Ub'): 'species', ('Body', 'rep'): 'species',
    ('Body', 'breath'): 'species', ('Body', 'f_w'): 'species', ('Body', 'f_thr'): 'G6', ('Body', 'cd_len'): 'species',
    ('Body', 'recruit'): 'del', ('Body', 'babble_rate'): 'G26', ('Body', 'keep'): 'del', ('Body', 'keep_window'): 'del',
    ('SoundGate', 'k_on'): 'G6', ('SoundGate', 'k_off'): 'species', ('SoundGate', 'tau'): 'G5',
    ('SoundGate', 'stuck'): 'del', ('SoundGate', 'backtrack'): 'del', ('SoundGate', 'nu_min'): 'del',
    ('SoundGate', 'k_trim'): 'del', ('SoundGate', 'k_back'): 'del',
}

layers = collections.OrderedDict(); cur = None
for line in open('knob_list.out', encoding='utf-8'):
    m = re.match(r'== (\w+) \((\d+)\)', line)
    if m:
        cur = m.group(1); layers[cur] = []; continue
    if cur and line.startswith('   '):
        layers[cur] = re.findall(r'(\w+)=', line)
tot = collections.Counter(); per_gene = collections.defaultdict(list); unmapped = []
for L, names in layers.items():
    for n in names:
        c = LAYER_OVERRIDE.get((L, n)) or ('del' if L == 'AgentV22' else M.get(n))
        if c is None:
            unmapped.append((L, n)); continue
        tot['gene' if c.startswith('G') else c] += 1
        if c.startswith('G'):
            per_gene[c].append(f'{L}.{n}')
n_all = sum(len(v) for v in layers.values())
print('knobs', n_all, dict(tot), 'unmapped', unmapped)
for g in sorted(GENES, key=lambda s: int(s[1:])):
    src = ', '.join(per_gene.get(g, [])) or ('(new)' if g in NEW else '(none)')
    print(f'{g:4s} {GENES[g]:55s} <- {src}')
print('genes fed by today\'s knobs:', len(per_gene), ' new genes:', len(NEW), ' total genes:', len(GENES))
