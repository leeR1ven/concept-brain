# -*- coding: utf-8 -*-
"""Which perceivable cue marks a NEW source that starts while another is sounding (overlapped onset)?
For each listener: ticks where a source's first fresh pair arrives while another source is audible (overlapped onset,
by attenuation class of the new source and whether it is louder than every ongoing source) vs ordinary ticks inside
sound with no new source.  Cues: e2 ratio to the max of the previous 3 pairs; rise of the max cell level over the
previous 6 ticks; number of 'strong' cells (>= 4) per slice in the fresh pair.  usage: python3 onset_cues.py rec.pkl"""
import sys, pickle, collections
import numpy as np
R = pickle.load(open(sys.argv[1], 'rb'))
rows = collections.defaultdict(list)
for L in R['fresh']:
    fr = R['fresh'][L].astype(np.int64); gt = R['gt'][L]
    e2 = fr.reshape(len(fr), -1).sum(1); mx = fr.reshape(len(fr), -1).max(1)
    strong = (fr >= 4).sum(1).max(1)          # max over the 2 slices of the number of cells >= 4
    seen = set()
    for t in range(6, len(fr)):
        srcs = gt[t]
        if not srcs:
            continue
        new = [s for s in srcs if (s[0], s[1], s[2]) not in seen]
        for s in srcs:
            seen.add((s[0], s[1], s[2]))
        old = [s for s in srcs if s not in new]
        prev_sound = any(gt[t - k] for k in (1, 2, 3))
        r = e2[t] / max(1, e2[t - 3:t].max())
        rise = mx[t] - mx[t - 6:t].max()
        if new and prev_sound:
            a = min(s[5] for s in new); ao = min((s[5] for s in old), default=9)
            prev_att = min((s[5] for k in (1, 2, 3) for s in gt[t - k]), default=9)
            key = f"new a{min(a,3)} {'louder' if a < prev_att else 'not-louder'}"
            rows[key].append((r, rise, strong[t]))
        elif not new and prev_sound:
            rows['continuation'].append((r, rise, strong[t]))
for k in sorted(rows):
    v = np.array(rows[k], float)
    print(f"{k:24s} n {len(v):6d} | e2 ratio p10/50/90 {np.percentile(v[:,0],10):.2f} {np.percentile(v[:,0],50):.2f} {np.percentile(v[:,0],90):.2f}"
          f" | P(ratio>1.4) {np.mean(v[:,0]>1.4):.2f} | maxcell rise>=2 {np.mean(v[:,1]>=2):.2f} >=1 {np.mean(v[:,1]>=1):.2f} | strong>2 {np.mean(v[:,2]>2):.2f}")
