"""Long overhearing life tables. python3 agg_c2.py <c2 JSON files>"""
import json
import sys

for f in sys.argv[1:]:
    r = json.load(open(f))
    print(f"\n### {r.get('tag')} (seed 0)")
    rate = r.get('rate', [])
    if rate:
        print('dialogues per hour (last segments):', [round(x['per_hour']) for x in rate[-4:]])
    print('\nspontaneous speech per 1,000 dialogues')
    print('| dialogues | utts/dlg | clear/dlg | single clear syl/dlg | garbled/dlg | during adult speech/dlg | in Q gap/dlg | ticks | top clear syllables | examples |')
    print('|---|---|---|---|---|---|---|---|---|---|')
    for s in r.get('speech', []):
        print(f"| {s['dialogues']} | {s['child_utts_per_dlg']} | {s['clear_utts_per_dlg']} | {s['clear_single_syllable_per_dlg']} | "
              f"{s['garbled_per_dlg']} | {s['during_adult_speech_per_dlg']} | {s['in_q_gap_per_dlg']} | {s['mean_ticks']} | "
              f"{', '.join(f'{a}×{n}' for a, n in s['top_clear_syllables'][:5])} | {' / '.join(s['examples'][-3:])} |")
    print('\nactivity windows (every 100 dialogues; last 5)')
    print('| dialogue | L2/L1 | L3/L2 | L4/L3 | L4 | L4 max share | Hs (min) | A4 | force | stop | T1 |')
    print('|---|---|---|---|---|---|---|---|---|---|---|')
    for wv in r.get('windows', [])[-5:]:
        print(f"| {wv['dialogue']} | {wv['L2_L1']:.2f} | {wv['L3_L2']:.2f} | {wv['L4_L3']:.2f} | {wv['L4']:.0f} | {wv['L4_frac_max']:.2f} | "
              f"{wv['Hs']:.0f} ({wv['Hs_min']:.0f}) | {wv['A4']:.0f} | {wv['force']:.1f} | {wv['stop']:.1f} | {wv.get('T1', '-')} |")
    print('\ncheckpoints (frozen copies)')
    print('| dialogues | QA seen exact / held exact / frame baseline | silent / garble | Q→A links: drive ratio, ranked first (seen / held) | next syllable first (chance 0.03) / mean rank; answer first | say when thinking: said / any x / garble / stop | one-shot (D=200): PFC / hip / either; control |')
    print('|---|---|---|---|---|---|---|')
    for cp, c in sorted(r.get('checkpoints', {}).items(), key=lambda kv: int(kv[0])):
        qa, ql, ns, sw, os_ = c['qa'], c['qa_links'], c['next_syllable'], c['say_when_thinking'], c['one_shot']
        o = next((v for k, v in os_.items() if k.startswith('heard_once')), {})
        ctl = next((v for k, v in os_.items() if k.startswith('control')), {})
        print(f"| {cp} | {qa['seen']['exact']:.2f} / {qa['held']['exact']:.2f} / {qa['frame_baseline']:.2f} | "
              f"{qa['seen']['silent']:.2f} / {qa['seen']['garble']:.2f} | {ql['seen']['drive_ratio_answer_vs_others']:.2f}, "
              f"{ql['seen']['answer_ranked_first']:.2f} / {ql['held']['drive_ratio_answer_vs_others']:.2f}, {ql['held']['answer_ranked_first']:.2f} | "
              f"{ns['next_in_question_first']:.2f} / {ns['next_in_question_mean_rank']:.1f}; {ns['answer_after_question_first']:.2f} | "
              f"{sw['said']:.2f} / {sw['any_x']:.2f} / {sw['garble']:.2f} / {sw['stop']:.2f} | "
              f"{o.get('pfc_route_hit', 0):.2f} / {o.get('hip_route_hit', 0):.2f} / {o.get('either', 0):.2f}; {ctl.get('either', 0):.2f} |")
