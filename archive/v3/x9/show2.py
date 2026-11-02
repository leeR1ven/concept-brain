import json, glob, sys, os
for p in sorted(glob.glob(sys.argv[1] + '/*.json')):
    r = json.load(open(p)); n = os.path.basename(p)[:-5]
    if 'res' not in r: continue
    m = r['res'][-1]; d = r['drift']
    g = lambda k: m.get(k, float('nan'))
    print(f"{n:16s} L2c1 W {g('L2_c1_within'):.3f} B {g('L2_c1_between'):.3f} held {g('L2_c1_held_dec'):.2f} mg {g('L2_c1_held_margin'):+.3f} seen {g('L2_c1_seen_dec'):.2f} | L2w W {g('L2_within'):.3f} B {g('L2_between'):.3f} held {g('L2_held_dec'):.2f} | all xm {g('xm_match'):.3f}/{g('xm_mis'):.3f} t1 {g('xm_top1'):.2f} L2xm t1 {g('L2_xm_top1'):.2f} | col L2 {g('L2_collapse'):.3f} | drift all {d.get('drift_words',-1):.3f} L2c1 {d.get('L2_drift_first',-1):.3f} L2w {d.get('L2_drift_words',-1):.3f} | on2 {r.get('n_on2',{}).get('mean',0):.0f}")
