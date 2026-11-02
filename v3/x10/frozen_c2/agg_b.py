"""Stage (b) tables. python3 agg_b.py <folders with run_b JSON>"""
import glob
import json
import sys

rows = []
for d in sys.argv[1:]:
    for f in sorted(glob.glob(d + '/*.json')):
        rows.append((f, json.load(open(f))))
print('| arm | setting | b1 said | b2 said | b1/b2 garble | b3 own loop | b4 echo/line | b5 mix garble | b6 brake | '
      'b7 utt/dlg | b7 b1 after | inner: clear/mumble/garble/other/silent | adult x held: clear/mumble/garble/other/silent |')
print('|' + '---|' * 13)
for f, r in rows:
    cfg = r['config']
    st = ' '.join(f'{k}={v}' for k, v in cfg.items() if k in ('g1p_src', 'mc', 'mc_src', 'g_ret_force') or k.startswith('band_cut'))
    sl = r.get('speech_links', {})
    fmt = lambda d: '/'.join(f"{d.get(k, 0):.2f}" for k in ('clear', 'mumble', 'garble', 'other', 'silent')) if d else '-'
    print(f"| {r['arm']} | {st} | {r['b1_forced_syllable_said']:.2f} | {r['b2_forced_answer_said']:.2f} | "
          f"{r['b1b2_garble']:.2f} | {r['b3_own_voice_loop']:.2f} | {r['b4_echo_per_line']:.2f} | "
          f"{r['b5_mixture_garble']:.2f} | {r['b6_brake_stops']:.2f} | {r['b7_utterances_per_dialogue']:.3f} | "
          f"{r['b7_b1_after_overhearing']:.2f} | {fmt(sl.get('inner'))} | {fmt(sl.get('adult_hold'))} |")
print()
print('| arm | setting | link before (top-4 exact, margin, true/other w) | link after | scaffold: forced syllables, decoded right, '
      'a1L windows ok | lag 12 after (newborn rule) | G1P units per syllable (min / median / flagged < 10) |')
print('|' + '---|' * 7)
for f, r in rows:
    cfg = r['config']
    st = ' '.join(f'{k}={v}' for k, v in cfg.items() if k in ('g1p_src', 'mc', 'mc_src', 'g_ret_force') or k.startswith('band_cut'))
    lk = lambda d: f"{d['top4_exact']:.2f}, {d['margin_median']:.2f}, {d['true_other_weight_ratio']:.2f}" if d else '-'
    sc = r.get('scaffold')
    scs = f"{sc['forced_syllables']}, {sc['forced_ticks_decoded_right']:.2f}, {sc['windows_ok']:.2f}" if sc else '-'
    ca = r.get('codes_after')
    lag = f"{ca['L4_decode']['12']:.2f} (a3 {'P' if ca['a3_pass'] else '-'})" if ca else '-'
    g = r.get('registry', {}).get('G1P')
    gs = (f"{min(g['per_syllable'])} / {sorted(g['per_syllable'])[len(g['per_syllable']) // 2]} / "
          f"{len(g['flagged_below_10'])}") if g else '-'
    print(f"| {r['arm']} | {st} | {lk(r.get('links_before'))} | {lk(r.get('links'))} | {scs} | {lag} | {gs} |")
