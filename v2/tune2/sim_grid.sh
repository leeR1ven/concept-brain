#!/bin/bash
# summarise a wander_sim grid: one line per setting
python3 wander_sim.py "$@" 2>&1 | python3 -c "
import sys, json
for l in sys.stdin:
    if l.startswith('#'): print(l.strip()); continue
    d=json.loads(l); print(f\"as {d['assoc']} fat {d['fat']} tau {d['fat_tau']} k {d['k']} hub {d['hub']} c0 {d['c0']} | top20 {d['held_top20']} exc {d['excess_top20']:+.3f} KL {d['KL']} eff {d['eff_topics']} top1 {d['top1_dom']} | top5 {d['top5_held']} lift {d['top5_lift']} c5 {d['top5_content']} | cont {d['content_held']} ({d['content_excess']:+.3f}) | long {d['long_n']} endC {d['long_end_content']} endT5 {d['long_end_top5']} endDist {d['long_end_distinct']} | END top5 {''.join(d['end_top5'])} {d['end_top5_share']} (heard {d['end_top5_heard']}) H {d['end_H']} c5 {'.'.join(d['end_top5_content'])} {d['end_c5_share']} (heard {d['end_c5_heard']})\")
"
