# -*- coding: utf-8 -*-
"""Ear module alone (frozen PFC) on a world2 recording (record.py): the v2.1 front end vs the open-world variants.

The listener's front end is replayed tick by tick exactly as AgentV21 runs it for another speaker's sound (no own
voice: listeners are silent): E -> line onset (E > 0 after E == 0) -> syllable clock (sample at clk % 3 == 2) ->
FEAT0 -> Z categories (DevAgent._detect: theta 0.75, rate 0.1, cap 256, a new category only if > 40 FEAT0 bits) ->
line end (E == 0); quiet moments (12 ticks E == 0, not in a line), long silences (200), brake input F (E >= 10).
Variants feed the same logic with the gated window of hear.SoundGate (and later keys).
Scoring uses the recording's ground truth only:
  true line   = one source's utterance (caregiver say / agent bout) audible to the listener; onset = first tick its
                fresh slices reach the ear; class by attenuation at onset: near a=0, mid a=1-2, far a>=3; 'clean'
                start = no other source audible in the 3 ticks before
  line recall = a detected line whose (back-tracked) onset is exactly the true onset (aligned syllable clock)
  precision   = detected lines whose onset is exactly some true onset
  sample label= the char of the dominant source if its window is aligned (whole char), else 'mis'; 'overlap' if two
                sources within 1 level; 'none' if no source (noise)
  purity      = share of char-labelled samples that fall in their category's majority char
usage: python3 ear_eval.py rec.pkl variant[,variant...] [key=val ...]   (variants: v21, gate)"""
import os, sys, json, time, pickle, collections
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'v21')); sys.path.insert(1, os.path.join(HERE, '..', 'synth'))
sys.path.insert(0, os.path.join(HERE, '..', 'synth', 'bench_copy'))
from frontend import Ear0, feat0                       # noqa: E402
from rules import Templates                            # noqa: E402
from hear import SoundGate, cell_gate, level_shifts    # noqa: E402


def true_lines(gt):
    """per listener: dict key -> dict(t_first, t_last, att0, clean)"""
    lines = {}
    prev_any = collections.deque([False, False, False], maxlen=3)
    for t, srcs in enumerate(gt):
        for (kind, sid, t0, ch, ph, att) in srcs:
            k = (kind, sid, t0)
            L = lines.get(k)
            if L is None:
                lines[k] = dict(t_first=t, t_last=t, att0=att, clean=not any(prev_any), n=1)
            else:
                L['t_last'] = t; L['n'] += 1
        prev_any.append(len(srcs) > 0)
    return lines


def label_at(gt, t):
    srcs = gt[t]
    if not srcs:
        return 'none', None
    srcs = sorted(srcs, key=lambda s: s[5])
    d = srcs[0]
    if len(srcs) > 1 and srcs[1][5] - d[5] <= 1:
        return 'overlap', None
    kind, sid, t0, ch, ph, att = d
    if ph != 2 or t < 2:
        return 'mis', None
    for dt, want in ((1, 1), (2, 0)):                  # the two previous fresh pairs: same line, same char, phases 1, 0
        ok = any(s[0] == kind and s[1] == sid and s[2] == t0 and s[3] == ch and s[4] == want for s in gt[t - dt])
        if not ok:
            return 'mis', None
    return ch, (kind, sid, t0)


class Listener:
    def __init__(self, ear, F0, variant, warm, cfg):
        self.ear, self.F0 = ear, F0
        self.Z = Templates(len(F0), cap=256, theta=0.75, rate=0.1)
        self.variant = variant
        self.gate = None
        if variant != 'v21':
            g = {k[2:]: float(v) for k, v in cfg.items() if k.startswith('g_')}
            self.gate = SoundGate(**{k: (int(v) if k in ('stuck', 'backtrack') else v) for k, v in g.items()})
            self.gate.calibrate(warm)
        self.level = int(cfg.get('level', 0)) if variant != 'v21' else 0; self.th_lv = float(cfg.get('th_lv', 0.85)); self.n_level = 0
        self.denoise = int(cfg.get('denoise', 0))
        self.in_line, self.clk, self.E_prev, self.quiet = False, 0, 0.0, 0
        self.onsets, self.ends, self.samples = [], [], []        # (t_eff, t), t, (t, cat, line index)
        self.relock = int(cfg.get('relock', 0)) if variant != 'v21' else 0
        self.th_lock = float(cfg.get('th_lock', 0.85))
        self.vbound = int(cfg.get('vbound', 0))
        self.mixed_nocreate = int(cfg.get('nocreate', 0))
        self.conf_create = int(cfg.get('conf', 0))
        self.conf_learn = int(cfg.get('conf_learn', 0))
        self.blocked = 0
        self.hist = []                                          # (t, j, m, f) of the last 2 in-line ticks
        self.mixed = False
        self.n_relock = 0
        self.qm, self.long_sil, self.F_on, self.day = 0, 0, 0, 0
        self.new_cats = 0

    def detect(self, f, jm=None, create=True, win=None):
        Z = self.Z
        j, m = Z.best(f) if jm is None else jm
        if j >= 0 and m >= 0.75:
            if not (self.conf_learn and self.gate is not None and not self.gate.conf):
                Z.learn(j, f.astype(np.float32))
            return j
        if self.level and win is not None and Z.n:          # a quieter (farther) instance of a known syllable?
            best = (-1, 0.0)
            for k, w2 in level_shifts(win, cell_gate(self.gate)):
                jj, mm = Z.best(self.ear.respond(self.ear.encode(w2))[self.F0])
                if mm > best[1]:
                    best = (jj, mm)
            if best[1] >= self.th_lv:
                self.n_level += 1
                return best[0]
        if create and Z.n < 256 and f.sum() > 40:
            self.new_cats += 1
            return Z.new(f.astype(np.float32))
        return None

    def tick(self, t, win, daytime):
        shift = 0
        if self.gate is None:
            E = float(win.sum())
        else:
            w, E, shift = self.gate.gate(win)
        onset = False
        if E > 0 and self.E_prev == 0 and not self.in_line:
            self.in_line = True; self.clk = 0; onset = True
            self.onsets.append((t - shift, t))
            self.hist = []; self.mixed = False
        elif self.in_line and E > 0:
            self.clk += 1
        if self.in_line and E > 0 and (self.relock or self.clk % 3 == 2):
            if self.denoise and self.gate is not None:
                g = cell_gate(self.gate)
                if g > 0:
                    win = np.where(win > g, win, 0)
            f = self.ear.respond(self.ear.encode(win))[self.F0]
            jm = self.Z.best(f) if self.relock else None
            if self.clk % 3 == 2:
                if self.relock and jm[1] < 0.75 and self.hist:
                    tb, jb, mb, fb = max(self.hist, key=lambda h: h[2])
                    if mb >= self.th_lock:                     # syllables are recognised at another phase: relock
                        d = t - tb
                        self.n_relock += 1; self.mixed = True
                        if self.vbound:                        # a new rhythm = a new speaker: the line ends here
                            self.ends.append(t)
                            self.onsets.append((tb - 2, t))
                        self.samples.append((tb, self.detect(fb, (jb, mb)), len(self.onsets) - 1))
                        self.clk = 0 if d == 1 else 1
                        jm = None
                if jm is not None or not self.relock:
                    create = not (self.mixed and self.mixed_nocreate)
                    if self.conf_create and self.gate is not None and not self.gate.conf:
                        create = False; self.blocked += 1
                    self.samples.append((t, self.detect(f, jm, create, win), len(self.onsets) - 1))
            if self.relock:
                self.hist = (self.hist + [(t, jm[0], jm[1], f)])[-2:] if jm is not None else []
        if self.in_line and E == 0:
            self.in_line = False
            self.ends.append(t)
        if onset and shift:
            self.clk = shift                                  # post-step phase correction (agent_h does the same)
        if E == 0 and not self.in_line:
            self.quiet += 1
        else:
            self.quiet = 0
        if daytime:
            self.day += 1
            self.qm += int(self.quiet == 12)
            self.long_sil += int(self.quiet == 200)
            self.F_on += int(E >= 10)
        self.E_prev = E


def score(L, gt, day_frac):
    TL = true_lines(gt)
    starts = {}
    for k, v in TL.items():
        starts.setdefault(v['t_first'], []).append(k)
    det_on = [te for te, _ in L.onsets]
    det_set = set(det_on)
    cls = lambda a: 'near' if a == 0 else ('mid' if a <= 2 else 'far')
    rec = collections.defaultdict(lambda: [0, 0])
    for k, v in TL.items():
        c = cls(v['att0'])
        for key in (c, c + ('_clean' if v['clean'] else '_overlapped'), 'all'):
            rec[key][0] += 1
            rec[key][1] += int(v['t_first'] in det_set)
    prec_hit = sum(1 for te in det_on if te in starts)
    noise_only = sum(1 for te, t in L.onsets if not gt[t])
    labk = [(label_at(gt, t), c, li) for t, c, li in L.samples]
    labs = [(lk[0], c) for lk, c, li in labk]
    cov = collections.Counter(); keys_by_line = collections.defaultdict(set)
    for (l, key), c, li in labk:
        if key is not None and c is not None:
            cov[key] += 1
            keys_by_line[li].add(key)
    covr = collections.defaultdict(list)
    for k, v in TL.items():
        nch = max(1, v['n'] // 3)
        c = cls(v['att0'])
        covr[c + ('_clean' if v['clean'] else '_overlapped')].append(min(1.0, cov[k] / nch))
        covr[c].append(min(1.0, cov[k] / nch))
    n_s = len(labs)
    lab_cnt = collections.Counter(l for l, _ in labs)
    by_cat = collections.defaultdict(collections.Counter)
    for l, c in labs:
        if c is not None:
            by_cat[c][l] += 1
    junk = ('mis', 'overlap', 'none')
    char_assigned = [(l, c) for l, c in labs if c is not None and l not in junk]
    maj_char = {c: max(((n, l) for l, n in cnt.items() if l not in junk), default=(0, None))[1] for c, cnt in by_cat.items()}
    pur = sum(1 for l, c in char_assigned if maj_char.get(c) == l) / max(1, len(char_assigned))
    junk_cats = sum(1 for c, cnt in by_cat.items() if cnt.most_common(1)[0][0] in junk)
    maj = {c: cnt.most_common(1)[0][0] for c, cnt in by_cat.items()}
    dup_cats = sum(v - 1 for k, v in collections.Counter(l for l in maj.values() if l not in junk).items() if v > 1)
    days = max(L.day / 24000.0, 1e-9)
    return dict(
        true_lines=len(TL), detected=len(det_on),
        recall={k: (v[1], v[0], round(v[1] / v[0], 3) if v[0] else None) for k, v in sorted(rec.items())},
        precision=round(prec_hit / max(1, len(det_on)), 3), noise_only_lines=noise_only,
        backtracked=sum(1 for te, t in L.onsets if te != t),
        samples=n_s, sample_labels={k: round(v / max(1, n_s), 3) for k, v in lab_cnt.most_common(6)},
        clean_sample_frac=round(sum(v for k, v in lab_cnt.items() if k not in junk) / max(1, n_s), 3),
        undetected=round(sum(1 for _, c in labs if c is None) / max(1, n_s), 3),
        cats=L.Z.n, junk_cats=junk_cats, dup_cats=dup_cats, purity=round(pur, 3),
        coverage={k: round(float(np.mean(v)), 3) for k, v in sorted(covr.items())},
        mixed_lines=round(sum(1 for v in keys_by_line.values() if len(v) > 1) / max(1, len(keys_by_line)), 3),
        relocks=L.n_relock, blocked=L.blocked, level_matches=L.n_level, distinct_chars=len(set(l for l, c in labs if l not in junk)),
        quiet_moments_per_day=round(L.qm / days, 1), long_silences_per_day=round(L.long_sil / days, 2),
        F_on_frac=round(L.F_on / max(1, L.day), 3),
        gate=(None if L.gate is None else dict(mu=round(L.gate.mu, 2), nu=round(L.gate.nu, 2), **L.gate.stats)))


def main():
    fn = sys.argv[1]
    variants = sys.argv[2].split(',')
    cfg = dict(a.split('=', 1) for a in sys.argv[3:] if '=' in a)
    t0 = time.time()
    R = pickle.load(open(fn, 'rb'))
    seed = R['seed']
    F0 = feat0(Ear0(seed=seed), R['warm'])
    dayT = R['day_ticks'] + R['night_ticks']
    out = dict(rec=os.path.basename(fn), N=R['N'], K=R['K'], p=R['p'], noise=R['noise'], ticks=R['ticks'], cfg=cfg,
               variants={})
    for var in variants:
        per = []
        for Lid in sorted(R['fresh']):
            fr = R['fresh'][Lid].astype(np.int64)
            gt = R['gt'][Lid]
            ear = Ear0(seed=seed); ear.set_feat0(F0)
            L = Listener(ear, F0, var, R['warm'], cfg)
            T = len(fr)
            win = np.zeros((8, 6), np.int64)
            for t in range(T):
                win = np.concatenate([win[:, 2:], fr[t]], axis=1)
                L.tick(t, win, (t % dayT) < R['day_ticks'])
            per.append(score(L, gt, None))
        agg = {}
        for k in ('true_lines', 'detected', 'precision', 'noise_only_lines', 'backtracked', 'samples', 'clean_sample_frac',
                  'undetected', 'cats', 'junk_cats', 'purity', 'quiet_moments_per_day', 'long_silences_per_day',
                  'F_on_frac', 'mixed_lines', 'relocks', 'blocked', 'distinct_chars', 'dup_cats', 'level_matches'):
            agg[k] = round(float(np.mean([p[k] for p in per])), 3)
        ck = set().union(*[p['coverage'].keys() for p in per])
        agg['coverage'] = {k: round(float(np.mean([p['coverage'][k] for p in per if k in p['coverage']])), 3) for k in sorted(ck)}
        rk = set().union(*[p['recall'].keys() for p in per])
        agg['recall'] = {k: round(sum(p['recall'].get(k, (0, 0, 0))[0] for p in per) /
                                  max(1, sum(p['recall'].get(k, (0, 0, 0))[1] for p in per)), 3) for k in sorted(rk)}
        agg['recall_n'] = {k: sum(p['recall'].get(k, (0, 0, 0))[1] for p in per) for k in sorted(rk)}
        out['variants'][var] = dict(mean=agg, per_listener=per)
        print(f"{os.path.basename(fn)} {var}: lines {agg['detected']:.0f}/{agg['true_lines']:.0f} prec {agg['precision']} "
              f"recall near {agg['recall'].get('near')} mid {agg['recall'].get('mid')} far {agg['recall'].get('far')} | "
              f"cats {agg['cats']} junk {agg['junk_cats']} dup {agg['dup_cats']} purity {agg['purity']} clean samples {agg['clean_sample_frac']} | "
              f"quiet/day {agg['quiet_moments_per_day']} long {agg['long_silences_per_day']} F {agg['F_on_frac']} | "
              f"cover near {agg['coverage'].get('near')} (ovl {agg['coverage'].get('near_overlapped')}) mid {agg['coverage'].get('mid')} "
              f"| mixed {agg['mixed_lines']} relocks {agg['relocks']} lv {agg['level_matches']} chars {agg['distinct_chars']}", flush=True)
    out['secs'] = round(time.time() - t0, 1)
    tag = cfg.get('tag', '')
    on = os.path.join(HERE, 'results', 'ear', os.path.basename(fn).replace('.pkl', f'{tag}.json'))
    os.makedirs(os.path.dirname(on), exist_ok=True)
    with open(on, 'a') as f:
        f.write(json.dumps(out, ensure_ascii=False, default=str) + '\n')


if __name__ == '__main__':
    main()
