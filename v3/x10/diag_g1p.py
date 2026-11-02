"""Why is G1P silent? Separates (i) L4 syllable specificity, (ii) selection timing, (iii) return gain vs blockers.
usage: python3 diag_g1p.py --set key=value ..."""
import argparse

import numpy as np
import torch

from common import DEV, config, log, save_json

from life import Life
import probes
from world_adapter import ADULTS, NSYL, SYL, C1

ap = argparse.ArgumentParser()
ap.add_argument('--set', nargs='*', default=[])
ap.add_argument('--out', default='diag_g1p.json')
a = ap.parse_args()


def val(v):
    try:
        return int(v)
    except ValueError:
        try:
            return float(v)
        except ValueError:
            return v


over = {k: val(v) for k, v in (x.split('=') for x in a.set)}
life = Life(config(1, **over), 0)
b, ph, mot = life.brain, life.ph, life.brain.motor
seg = [sg for sg in b.chains.segs if sg[0] == 'force'][0]
res = dict(config=over, registry=life.registry['G1P'])

# (iii) force x's force copies internally for 3 ticks (bypassing L4 and G1P): does the mouth say x?
out3, inh = {}, []
with probes.Frozen(life) as snap:
    cnt = {}
    for x in range(NSYL):
        b.restore(snap)
        m = torch.zeros(b.n1, dtype=torch.bool, device=DEV)
        own = torch.as_tensor(ph.need_mat[x], device=DEV)[mot.muscle_of]
        m[seg[2]:seg[3]] = own
        life.probe_env.reset_track()
        probes.play(life, [None] * 16, force={6 + k: m for k in range(3)},
                    rec=lambda lf, t: inh.append((float(lf.brain.motor.last['inh']), int(lf.brain.motor.stop.on.sum()),
                                                  int(lf.brain.motor.force.on.sum()))) if 7 <= t <= 9 else None)
        k = probes.classify(list(life.probe_env.syl), list(life.probe_env.events), x)
        cnt[k] = cnt.get(k, 0) + 1
    out3 = {k: v / NSYL for k, v in cnt.items()}
I = np.array(inh)
res['iii_force_copies_forced'] = dict(outcome=out3, inh_mean=float(I[:, 0].mean()), stops_on_mean=float(I[:, 1].mean()),
                                      force_on_mean=float(I[:, 2].mean()),
                                      ret_drive=float(mot.ret_drive[0]), thr=float(mot.thr_f[0]))
log('(iii)', res['iii_force_copies_forced'])

# (i), (ii): L4 codes of each syllable by window after the onset; specificity counts
wins = {}
with probes.Frozen(life) as snap:
    H = {}
    for x in range(NSYL):
        for vi, v in enumerate(ADULTS[:2]):
            b.restore(snap)
            rec = []
            probes.play(life, [None] * 4 + [(x, v, 1.0)] * 3 + [None] * 14, mute=True,
                        rec=lambda lf, t: rec.append(lf.brain.pfc.L4.on.clone()))
            H[(x, vi)] = torch.stack(rec).float()
    base = torch.stack([H[(x, 0)][0:4].mean(0) for x in range(NSYL)]).mean(0)   # before the onset (held state)
    for lo, hi in ((4, 7), (7, 10), (10, 13), (13, 16)):
        M = torch.stack([torch.stack([H[(x, vi)][lo:hi].mean(0) for vi in range(2)]).mean(0) for x in range(NSYL)])
        per = []
        for x in range(NSYL):
            other = torch.cat([M[:x], M[x + 1:]]).max(0).values
            per.append([int(((M[x] >= 0.5) & (M[x] - other >= mg)).sum()) for mg in (0.1, 0.2, 0.3)])
        per = np.array(per)
        wins[f'{lo - 4}-{hi - 5} after onset'] = dict(median_selective_m01_m02_m03=np.median(per, 0).tolist(),
                                                     zero_m03=int((per[:, 2] == 0).sum()),
                                                     L4_active_frac=float(M.mean()))
    # overlap: same syllable two voices vs different syllables (ticks 4-9)
    codes = {k: (v[4:10].mean(0) >= 0.5) for k, v in H.items()}
    jac = lambda p, q: float((p & q).sum()) / max(1, float((p | q).sum()))
    same = np.mean([jac(codes[(x, 0)], codes[(x, 1)]) for x in range(NSYL)])
    diff = np.mean([jac(codes[(x, 0)], codes[(y, 0)]) for x in range(NSYL) for y in range(NSYL) if x < y])
    held = np.mean([jac(codes[(x, 0)], base >= 0.5) for x in range(NSYL)])
res['i_ii_L4'] = dict(windows=wins, J_same_syllable_two_voices=same, J_different_syllables=diff,
                      J_syllable_vs_pre_onset_state=held, L4_frac_pre_onset=float(base.mean()))
log('(i)/(ii)', res['i_ii_L4'])

# inner hearing: how much of x's selected L4 units fire, and the G1P drive into x's own force copies (vs theta_1)
sel = life.l4_sel
fr, dr = [], []
with probes.Frozen(life) as snap:
    for x in range(NSYL):
        if not bool(sel[x].any()):
            continue
        b.restore(snap)
        rec = []
        own = torch.as_tensor(ph.need_mat[x], device=DEV)[mot.muscle_of]
        probes.play(life, [None] * 18, force={6 + k: probes.code_mask(life, [x]) for k in range(3)},
                    rec=lambda lf, t: rec.append((lf.brain.pfc.L4.on[sel[x]].float().mean().item(),
                                                  lf.brain.pfc.d1[seg[2]:seg[3]][own].mean().item())))
        fr.append(max(r[0] for r in rec[8:16]))
        dr.append(max(r[1] for r in rec[8:16]))
res['inner_hearing'] = dict(max_share_of_selected_L4_units_on=float(np.median(fr)), max_G1P_drive_own_copies=float(np.median(dr)),
                            theta1=life.c['theta1'])
log('inner', res['inner_hearing'])
save_json(a.out, res)
