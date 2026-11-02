"""Skeptic re-count of ARCH_v3_core's '265 knobs -> 189 deleted' (synth/gene_map.py), split by what a deletion is:
a switch (bool / str / on-off int), a numeric constant of a mechanism that is gone, or a numeric constant of a mechanism
the core keeps in another form ('absorbed': its constant reappears, undeclared, inside M1-M4 / P1-P5).
Inputs: synth/knob_list.out (defaults) and synth/gene_map.py's 'del' lists.  Classification by hand below (ASSUMED
judgement, one line per group).  Run: python3 -I knob_audit.py"""
import re, collections
SYN = 'docs/v3design/synth/'
src = open(SYN + 'gene_map.py').read()
dels = set()
for m in re.finditer(r"put\('del', '([^']*)'(?:\s*\n\s*'([^']*)')*", src):
    pass
for blk in re.findall(r"put\('del',((?:\s*'[^']*')+)\)", src):
    for s in re.findall(r"'([^']*)'", blk):
        dels.update(s.split())
lay = collections.OrderedDict(); cur = None
for line in open(SYN + 'knob_list.out'):
    m = re.match(r'== (\w+) \((\d+)\)', line)
    if m: cur = m.group(1); lay[cur] = {}; continue
    if cur and line.startswith('   '):
        for k, v in re.findall(r"(\w+)=('[^']*'|<class '\w+'>|[^,\s]+)", line):
            lay[cur][k] = v
override_del = {('RandMix', k) for k in 'code train every prune regrow min_units cache_cap alloc'.split()} | \
    {('Body', k) for k in 'cue_w recruit keep keep_window'.split()} | \
    {('SoundGate', k) for k in 'stuck backtrack nu_min k_trim k_back'.split()}
SWITCH_NAMES = set('''use_gen copy_prompt copy_next copy_comp hippo ctx att_learn match focus reward_actor sleep_replay
 learn_listen retino branch_norm rf_focus vis_rec self_ground prime_internal sleep_mode hip_bigram own_rotate
 dialogue_reset role_units gate_noid gate_freq ctx_homeo ior nop_keys scene_recall ct rel name_vr_only ral ral_cont
 ral_strict ral_hip_own ral_cut ral_other ral_q say_rec say_loc say_noprompt scene_held val_act rec_fix learn_cats
 life_eye policy learn_turn urg_gate close_on_scene oth_kind brake_now eye_feats no_react late_release own_tail
 rebind turn_feats voice dip soc_feats h_gate h_back h_relock h_vbound h_nocreate h_conf h_level h_denoise mode init
 code train prune regrow alloc recruit keep backtrack veto_R1 dip_scope gain_reply seg'''.split())
DEAD = set('''eta_sleep sleep_passes sleep_k mature mat_norm mature_g ctx_n ctx_k w_ctx ctx_lam ctx_beta ctx_tau wd_gen
 wd_gate wd_every split_use split_err split_n gen_mmin gate_mmin rec_boost ct_tau ct_own ct_surp name_ctx ral_q_eta
 ral_q_thr ral_q_prior veto_rel nu_min'''.split())           # mechanism off in the final v2.2 config (report.md §6)
GONE = set('''ral_W ral_theta ral_T ral_replay_own ral_init ral_W_other say_gain say_loc_thr veto_eps veto_W veto_gate
 veto_fast dip_g dip_pthr val_veto g0_reply g0_scene u_scene beta g_max nogo nogo_late L tau_w u_R1 init_min_tau
 wander_after h_kback h_trim h_stuck h_lock h_lvth stuck k_trim k_back kx h U m_min n_min alpha kcl eta_c tau_cl eps
 bfloor max_swap kcls think_k adopt_margin wander cue_w keep_window'''.split())   # mechanism deleted with no successor
ABSORBED = set('''floor kappa rf_thr match_thr rec_thr rate_max conf_min match_n attd_n match_min match_gap att_thr
 contingency hip_novelty rec_pow recruit_p ct_assoc'''.split())   # mechanism kept in another form; its constants recur
IMPL = set('mix_cap prune_every every min_units cache_cap'.split())   # implementation caps / batching
c = collections.Counter(); unk = []
for L, kv in lay.items():
    for k, v in kv.items():
        isdel = (L, k) in override_del or (k in dels and not (L == 'RandMix' and k in ('alpha', 'eta', 'theta', 'age')))
        if L == 'AgentV22': isdel = True
        if not isdel: continue
        cat = 'switch' if k in SWITCH_NAMES else 'dead numeric' if k in DEAD else 'gone numeric' if k in GONE \
            else 'absorbed numeric' if k in ABSORBED else 'implementation cap' if k in IMPL else None
        if cat is None: unk.append(f'{L}.{k}'); cat = '?'
        c[cat] += 1
print('deleted knobs by kind:', dict(c), ' total', sum(c.values()))
print('unclassified:', unk)
