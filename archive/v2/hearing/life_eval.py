# -*- coding: utf-8 -*-
"""One unbroken life of the ear module (frozen PFC) with a changing noise level: per-day line recall / precision,
near coverage, categories (cumulative), junk and duplicates, floor mu, quiet moments.  The listener's state is never
reset.  usage: python3 life_eval.py rec_sched.pkl variant [key=val ...]"""
import os, sys, json, pickle, collections
import numpy as np
from ear_eval import Listener, true_lines, label_at, Ear0, feat0
fn, var = sys.argv[1], sys.argv[2]; cfg = dict(a.split('=', 1) for a in sys.argv[3:])
R = pickle.load(open(fn, 'rb')); seed = R['seed']; D = 30000
F0 = feat0(Ear0(seed=seed), R['warm'])
out = dict(rec=os.path.basename(fn), var=var, cfg=cfg, sched=R.get('sched'), days=[])
per_day = collections.defaultdict(lambda: collections.defaultdict(list))
for Lid in sorted(R['fresh']):
    ear = Ear0(seed=seed); ear.set_feat0(F0)
    L = Listener(ear, F0, var, R['warm'], cfg)
    fr = R['fresh'][Lid].astype(np.int64); gt = R['gt'][Lid]
    win = np.zeros((8, 6), np.int64); ends = []
    for t in range(len(fr)):
        win = np.concatenate([win[:, 2:], fr[t]], 1); L.tick(t, win, (t % D) < 24000)
        if (t + 1) % D == 0:
            ends.append(dict(cats=L.Z.n, mu=None if L.gate is None else round(L.gate.mu, 2), qm=L.qm))
    TL = true_lines(gt)
    det = set(te for te, _ in L.onsets)
    by_cat = collections.defaultdict(collections.Counter); cov = collections.Counter()
    for t, c, li in L.samples:
        lab, key = label_at(gt, t)
        if c is not None:
            by_cat[c][lab] += 1
            if key is not None:
                cov[key] += 1
    junkl = ('mis', 'overlap', 'none')
    maj = {c: cnt.most_common(1)[0][0] for c, cnt in by_cat.items()}
    first = {}
    for t, c, li in L.samples:
        if c is not None and c not in first:
            first[c] = t
    prev_qm = 0
    for d in range(len(ends)):
        lo, hi = d * D, (d + 1) * D
        tl = [(k, v) for k, v in TL.items() if lo <= v['t_first'] < hi]
        near = [(k, v) for k, v in tl if v['att0'] == 0]
        on = [te for te, _ in L.onsets if lo <= te < hi]
        starts = set(v['t_first'] for v in TL.values())
        newc = [c for c, t in first.items() if lo <= t < hi]
        pd = per_day[d]
        pd['recall'].append(np.mean([v['t_first'] in det for k, v in tl]) if tl else np.nan)
        pd['near_recall'].append(np.mean([v['t_first'] in det for k, v in near]) if near else np.nan)
        pd['near_cover'].append(np.mean([min(1, cov[k] / max(1, v['n'] // 3)) for k, v in near]) if near else np.nan)
        pd['precision'].append(np.mean([te in starts for te in on]) if on else np.nan)
        pd['cats_end'].append(ends[d]['cats'])
        pd['new_junk'].append(sum(1 for c in newc if maj.get(c) in junkl))
        pd['mu_end'].append(ends[d]['mu'] if ends[d]['mu'] is not None else 0.0)
        pd['quiet'].append(ends[d]['qm'] - prev_qm); prev_qm = ends[d]['qm']
for d in sorted(per_day):
    row = {k: round(float(np.nanmean(v)), 3) for k, v in per_day[d].items()}
    row['day'] = d; row['noise'] = None
    sch = R.get('sched') or {}
    cur = [s for dd, s in sorted(sch.items()) if dd <= d]
    row['noise'] = cur[-1] if cur else R['noise']
    out['days'].append(row)
    print(row, flush=True)
os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'life'), exist_ok=True)
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'life', os.path.basename(fn).replace('.pkl', f'_{var}{cfg.get("tag", "")}.json')), 'w') as f:
    json.dump(out, f, default=str)
