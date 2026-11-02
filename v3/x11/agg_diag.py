"""Compact table of diag_scale.py outputs. usage: python3 agg_diag.py diag_seed0.json [diag_seed1.json]"""
import json
import sys


def f(x, n=3):
    return '-' if x is None else f'{x:.{n}f}'


for path in sys.argv[1:]:
    D = json.load(open(path))
    print('=====', path)
    for state in ('newborn', 'replica_1000', 'checkpoint'):
        if state not in D:
            continue
        r = D[state]
        print(f"--- {state} at {r['at_dialogues']} dialogues, occurrences {r['n_occ']}")
        print('  (a) x_int (t0+1..t0+3) ranked against:   top1  top5  mrr  | unigram ref top1 top5')
        for k, v in r['a_echo_x_int'].items():
            u = v['unigram_ref']
            print(f"      {k:16s} {f(v['top1'])} {f(v['top5'])} {f(v['mrr'])} | {f(u['top1'])} {f(u['top5'])}")
        b = r['b_sources']
        ll = b['learned_lit (thresholded)']
        print(f"  (b) learned L41 lit (thresholded): target {f(ll['target']['top1'])} {f(ll['target']['top5'])}, "
              f"previous {f(ll['previous']['top1'])} {f(ll['previous']['top5'])}, lit/window {f(ll['lit_per_window'], 1)}")
        ld = b['learned_drive (graded)']
        print(f"      learned L41 drive (graded): target mean-per-code {f(ld['target_mean']['top1'])} {f(ld['target_mean']['top5'])}"
              f" | sum {f(ld['target_sum']['top1'])} {f(ld['target_sum']['top5'])} | cos {f(ld['target_cos']['top1'])} "
              f"{f(ld['target_cos']['top5'])} | previous mean {f(ld['previous_mean']['top1'])} {f(ld['previous_mean']['top5'])}"
              f" | drive/copy/tick {f(ld['drive_per_copy_tick'], 4)}")
        lp = b['loop_drive (graded)']
        print(f"      birth loop drive (graded): target {f(lp['target_mean']['top1'])} {f(lp['target_mean']['top5'])} | previous "
              f"{f(lp['previous_mean']['top1'])} {f(lp['previous_mean']['top5'])} | drive/copy/tick {f(lp['drive_per_copy_tick'], 4)}")
        for k, m in r['b_magnitude'].items():
            print(f"      max drive / theta [{k}]: median {f(m['max_over_theta_median'])} p90 {f(m['max_over_theta_p90'])}, "
                  f"units above theta per tick {f(m['units_above_theta_mean'], 2)}")
        h = r['c_hip']
        print(f"  (c) Hs->H1 synapses w>0 {h['H1t_synapses_w_pos']} of {h['H1t_synapses']} (w max {f(h['H1t_w_max'])}); "
              f"Hs<->Hs w>0 {h['HH_synapses_w_pos']} of {h['HH_synapses']}; g_h1 {h['g_h1']} theta_h1 {h['theta_h1']}")
        t = r['learned_tables']
        print(f"      L4->L1 w>0 {t['L41_w_pos']} of {t['L41_synapses']} (w max {f(t['L41_w_max'])}); L4<->L4 w>0 {t['L44_w_pos']}")
        d = r['d_L4']
        print(f"  (d) L4 at t0+3: on {f(d['frac_on'])}, Jaccard same syllable {f(d['J_same_syllable'])}, different {f(d['J_different'])}")
