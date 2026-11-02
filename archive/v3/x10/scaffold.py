"""Learned-speech arm (designer 11-02, ARCH 2.7): "一开始强制模型按规律说话，模型自动就会听到自己的声音，然后把声音和动作联系起来".
Generation-0 scaffold 'forced regular speaking': a program (the teacher moving the mouth) makes syllable x's force
neurons fire on the forced ticks, regardless of the blockers. It is outside the brain and logged here. No G1, no
G1P. Learning is on everywhere as normal. The child hears its own voice through the world physics, and the one rule
links the held sound code (PFC L4) to the force copies in PFC L1 through the learned L4 -> L1 table (L41).

Registered patterns (30 forced syllables per syllable in every arm; syllable 3 ticks, gap 3 ticks, pause 12 ticks;
syllables in random order):
  F1 repetition : x x x                     (10 patterns per syllable)
  F2 single     : x, pause                  (30 patterns per syllable; control, expected to fail like X7)
  F3 adult first: adult says x, then x x    (15 patterns per syllable; adult voice C1 or C2 alternating)
  F4 timed to the thought (11-02): adult says x, 3-tick gap, ONE forced x (30 patterns per syllable). The forced
     onset falls 6 ticks after the adult's onset, where L4 is most specific to x (x10_dlag: x ranked first 1.00 at
     tick 6, 0.06 by tick 10), so the force copies rise while the PFC holds x; a 12-tick pause follows."""
import numpy as np
import torch

from common import DEV
from world_adapter import C1, C2, NSYL

SYL_T, GAP, PAUSE = 3, 3, 12


def script(arm, rng, per_syl=30):
    reps = {'F1': 3, 'F2': 1, 'F3': 2, 'F4': 1}[arm]
    order = [x for x in range(NSYL) for _ in range(per_syl // reps)]
    rng.shuffle(order)
    ticks = []
    for i, x in enumerate(order):
        if arm in ('F3', 'F4'):
            v = C1 if i % 2 == 0 else C2
            ticks += [((x, v, 1.0), None)] * SYL_T + [(None, None)] * GAP
        for r in range(reps):
            ticks += [(None, x)] * SYL_T
            if r < reps - 1:
                ticks += [(None, None)] * GAP
        ticks += [(None, None)] * PAUSE
    return ticks


def run(life, arm, per_syl=30, window=400, on_tick=None):
    b, env, ph = life.brain, life.probe_env, life.ph
    mot = b.motor
    masks = [torch.as_tensor(ph.need_mat[x], device=DEV)[mot.muscle_of] for x in range(NSYL)]
    ticks = script(arm, np.random.default_rng(life.seed + 9700), per_syl)
    was = b.learning
    b.learning = True
    m = np.zeros(24, bool)
    rows, forced = [], []
    forced_ticks = said_right = 0
    env.reset_track()
    prev_x = None
    for adult, x in ticks:
        b.scaffold = masks[x] if x is not None else None
        if on_tick is not None:                       # DIAGNOSTIC hook: (brain, forced syllable, forced onset?)
            on_tick(b, x, x is not None and prev_x is None)
        prev_x = x
        a0 = env.step(m, adult=adult)
        m = b.step(torch.as_tensor(a0, device=DEV), 0.0).cpu().numpy()
        if x is not None:
            forced_ticks += 1
            dec = ph.decode(m)
            said_right += int(dec is not None and dec[0] in ('s', 'm') and dec[1] == x)
        rows.append(b.counts())
        if on_tick is not None:
            on_tick(b, x, 'after')
        forced.append(x is not None)
    b.scaffold = None
    b.learning = was
    C = torch.stack(rows).float().cpu().numpy()
    names = b.count_names()
    ix = {n: i for i, n in enumerate(names)}
    wins = []
    kh = b.c['kh']
    for w0 in range(0, len(C) - window + 1, window):
        mean = C[w0:w0 + window].mean(0)
        w = dict(L2_L1=float(mean[ix['L2']] / max(mean[ix['L1']], 1)), L3_L2=float(mean[ix['L3']] / max(mean[ix['L2']], 1)),
                 L4_L3=float(mean[ix['L4']] / max(mean[ix['L3']], 1)), Hs=float(mean[ix['Hs']]) if 'Hs' in ix else None,
                 L4_frac=float(mean[ix['L4']] / b.pfc.n))
        w['ok'] = bool(all(0.5 <= w[k] <= 2.0 for k in ('L2_L1', 'L3_L2', 'L4_L3'))
                       and (w['Hs'] is None or 0.5 * kh <= w['Hs'] <= 2.0 * kh))
        wins.append(w)
    F = np.array(forced)
    budget = {}
    T = len(C)
    for lab, (lo, hi) in (('first', (0, T // 10)), ('middle', (T // 2 - T // 20, T // 2 + T // 20)), ('last', (T - T // 10, T))):
        seg = slice(lo, hi)
        budget[lab] = {n: dict(forced=float(C[seg][F[seg], i].mean()) if F[seg].any() else None,
                               unforced=float(C[seg][~F[seg], i].mean()) if (~F[seg]).any() else None)
                       for i, n in enumerate(names)}
    return dict(arm=arm, ticks=len(ticks), forced_syllables=forced_ticks // SYL_T, budget=budget,
                forced_ticks_decoded_right=said_right / max(1, forced_ticks),
                windows_ok=float(np.mean([w['ok'] for w in wins])) if wins else None, windows=wins)
