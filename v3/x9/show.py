import json, glob, sys, os
for p in sorted(glob.glob(sys.argv[1] + '/*.json')):
    r = json.load(open(p))
    n = os.path.basename(p)[:-5]
    for m in r['res']:
        if m['frac'] < 0.5 and len(sys.argv) < 3: continue
        print(f"{n:14s} f{m['frac']:.2f} W {m['within']:.3f} B {m['between']:.3f} held {m['held_dec']:.2f} mg {m['held_margin']:+.3f} seen {m['seen_dec']:.2f} | c1 W {m['c1_within']:.3f} B {m['c1_between']:.3f} held {m['c1_held_dec']:.2f} seen {m['c1_seen_dec']:.2f} | xm {m['xm_match']:.3f}/{m['xm_mis']:.3f} t1 {m['xm_top1']:.2f} | col {m['collapse']:.3f} sz {m['size_word']:.0f}")
    print(f"{n:14s} drift {r['drift'].get('drift_words', float('nan')):.3f} chars {r['drift'].get('drift_chars', float('nan')):.3f}  secs {r['secs']} n_on {r['n_on']['mean']:.0f} ceil {r['n_on']['ceil_frac']:.3f}")
