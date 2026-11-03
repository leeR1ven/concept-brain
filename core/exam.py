"""The check-up: a fixed set of measurements, run the same way on every save (and on a newborn as the baseline), so
that whether training helps is judged by trends, not by one number. The model may know a line and still not think
about it, so the first part asks whether the line is *in* the brain's state at all, as neuroscience does with
population decoding: from the activity, can a simple outside reader tell which line was heard, when the reader has
learned each line from three adult voices and is tested on the fourth (leave-one-voice-out, nearest centroid,
cosine)? The reader is a measuring instrument, not part of the model.
  A  decoding of heard lines (the most common first lines) and never-heard lines (held out), chance 1/n, from:
     input  the PFC's external copies while the line is heard
     L4     the PFC's last layer while heard / in the 12 quiet ticks after
     think  the PFC's internally lit copies after the line;  hip  the hippocampal recall after the line
     heard minus never-heard (beyond what the input itself gives) = what training put into the brain
  B  thinking range after a heard line: same line other voice vs different lines (Jaccard), and its size
  C  one-shot memory: never-heard lines, each once (voice C1); then its first three syllables, in the same voice and
     in another (D1), soon (right after all were heard) and later (after 20 dialogues of ordinary life): the rank of
     its 4th syllable among the 4th syllables of the other test lines, by the share of its sound copies lit in the
     hippocampal recall / the PFC's thoughts, counted as the gain over the same cue before hearing (1 = best,
     chance in the output)
  D  health: the share of L4 lit whatever the line, learned vs birth push into L4, learned weight sizes
usage: python -u -m core.exam <ckpt or 'newborn'> [n_lines] [key=value for a newborn]"""
import json
import sys
import time

import numpy as np
import torch

from .config import config
from .corpus_life import CorpusLife
from .net import DEV
from .world.corpus import common_order
from .world.voice import ADULTS


def first_lines(D, order, n, lo=2, hi=10):
    out = []
    for d in order:
        t = D.turns(int(d))[0]
        if lo <= len(t) <= hi:
            out.append([int(x) for x in t])
        if len(out) >= n:
            break
    return out


def record(life, ln, v):
    """one hearing, frozen: per tick input, L4, thoughts, hippocampal recall; split into heard / after windows"""
    b = life.brain
    srcs = [None] * 4 + [(x, k, v, 0.0, 0.0) for x in ln for k in range(3)] + [None] * 12
    rec = []
    life.play(srcs, rec=lambda t: rec.append((b.pfc.L1ext.on.clone(), b.pfc.L4.on.clone(), b.pfc.x_int.clone(),
                                              b.hip.h_int.clone())))
    E, L4, X, H = (torch.stack(z).float() for z in zip(*rec))
    return {'input': E[4:-12].sum(0), 'L4_heard': L4[4:-12].sum(0), 'L4_after': L4[-12:].sum(0),
            'think': X[-12:].sum(0), 'hip': H[-12:].sum(0)}


def decode(R, n):
    """leave-one-voice-out nearest centroid (cosine): share of hearings whose line is found"""
    out = {}
    for k in R[(0, ADULTS[0])]:
        V = torch.stack([torch.stack([R[(i, v)][k] for i in range(n)]) for v in ADULTS])      # [voices, lines, units]
        V = V / V.norm(dim=2, keepdim=True).clamp_min(1e-9)
        hit = 0
        for t in range(len(ADULTS)):
            C = torch.cat([V[:t], V[t + 1:]]).mean(0)
            C = C / C.norm(dim=1, keepdim=True).clamp_min(1e-9)
            hit += int((V[t] @ C.T).argmax(1).eq(torch.arange(n, device=V.device)).sum())
        out[k] = round(hit / (len(ADULTS) * n), 3)
    return out


def jac(a, b):
    a, b = a > 0, b > 0
    return float((a & b).sum()) / max(1.0, float((a | b).sum()))


def memory(life, lines, gap=20):
    """C: one hearing of each test line (learning on), the ordinary life in between, then the cue. A line's target is
    what its 4th syllable lit (the copies that rose) when it was heard; its rank is among the other lines' targets."""
    b = life.brain

    def cue(v):
        out = []
        for ln in lines:
            srcs = [None] * 4 + [(x, k, v, 0.0, 0.0) for x in ln[:3] for k in range(3)] + [None] * 9
            rec = []
            life.play(srcs, rec=lambda t: rec.append((b.hip.h_int.clone(), b.pfc.x_int.clone())))
            out.append({'hip': torch.stack([z[0] for z in rec])[13:].any(0),
                        'think': torch.stack([z[1] for z in rec])[13:].any(0)})
        return out
    lit = {'before': cue('C1')}
    was = b.learning
    b.learning = True
    codes = []
    for ln in lines:                                       # each heard once, with a pause after it
        rose = []
        for x in [None] * 4 + [(x, k, 'C1', 0.0, 0.0) for x in ln for k in range(3)] + [None] * 8:
            b.step(torch.as_tensor(life.coch.a0([] if x is None else [x]), device=DEV), 0.0)
            rose.append(b.pfc.L1ext.rose().clone())
        codes.append(torch.nonzero(torch.stack(rose[13:16]).any(0)).squeeze(1))
    b.learning = was
    lit['soon_same_voice'], lit['soon_other_voice'] = cue('C1'), cue('D1')
    b.learning = True
    n0 = life.env.n_ex
    while life.env.n_ex - n0 < gap:
        life.tick()
    b.learning = was
    lit['later_same_voice'], lit['later_other_voice'] = cue('C1'), cue('D1')
    def share(L, i, key):
        return [float(L[i][key][cd].float().mean()) if cd.numel() else 0.0 for cd in codes]
    out = {}
    for when, L in lit.items():
        if when == 'before':
            continue
        r = {}
        for key in ('hip', 'think'):
            rk = []
            for i in range(len(lines)):                       # what the hearing added, against the cue before it
                sc = [a - b0 for a, b0 in zip(share(L, i, key), share(lit['before'], i, key))]
                mine = sc[i]
                rk.append(1 + sum(v > mine for v in sc) + 0.5 * (sum(v == mine for v in sc) - 1))
            r[key] = round(float(np.mean(rk)), 2)
        out[when] = r
    return dict(**out, chance=round((len(lines) + 1) / 2, 2), candidates=len(lines))


def health(life, Rh, n):
    b, p = life.brain, life.brain.pfc
    S = torch.stack([(Rh[(i, v)]['L4_after'] > 0).float() for i in range(n) for v in ADULTS])
    F = S.mean(0)
    common = float((S * (F >= 0.5)).sum() / S.sum().clamp_min(1))
    out = {'L4_after_lit_whatever_the_line': round(common, 3)}
    for nm, L in (('L41', p.L41), ('L44', p.L44), ('HH', b.hip.HH), ('H1t', b.hip.H1t)):
        w = L.w[L.idx < L.n_post]
        out[nm] = dict(above_1=round(float((w > 1).float().mean()), 4), max=round(float(w.max()), 2) if w.numel() else 0)
    return out


def main():
    src, n = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 60
    t0 = time.time()
    if src == 'newborn':
        over = {k: json.loads(v) for k, v in (a.split('=', 1) for a in sys.argv[3:] if '=' in a)}
        life = CorpusLife(config(**{'levels_per_s': 4, 'n_brake': 20, 'birth_dialogues': 80, **over}), 0, 'data')
    else:
        life = torch.load(src, weights_only=False)['life']
    order = getattr(life, 'order', None)                    # what the child has heard most (the curriculum's order)
    heard = first_lines(life.train, common_order(life.train) if order is None else order, n)
    novel = first_lines(life.heldout, range(life.heldout.n), n, lo=4)
    out = {}
    Rh = {(i, v): record(life, ln, v) for i, ln in enumerate(heard) for v in ADULTS}
    Rn = {(i, v): record(life, ln, v) for i, ln in enumerate(novel) for v in ADULTS}
    dh, dn = decode(Rh, len(heard)), decode(Rn, len(novel))
    out['A_decode_heard'] = dh
    out['A_decode_never_heard'] = dn
    out['A_chance'] = round(1 / n, 3)
    # B thinking range after a heard line
    m = len(heard)
    for k in ('think', 'hip'):
        same = np.mean([jac(Rh[(i, 'C1')][k], Rh[(i, v)][k]) for i in range(m) for v in ADULTS[1:]])
        diff = np.mean([jac(Rh[(i, 'C1')][k], Rh[(j, v)][k]) for i in range(m) for j in range(m) if i != j
                        for v in ADULTS[1:]])
        size = np.mean([float((Rh[key][k] > 0).sum()) for key in Rh])
        out[f'B_{k}'] = dict(same_line=round(float(same), 3), different_lines=round(float(diff), 3), size=round(float(size)))
    out['D_health'] = health(life, Rh, m)
    out["C_memory"] = memory(life, novel[:40])
    out['secs'] = round(time.time() - t0)
    print('EXAM', src, json.dumps(out), flush=True)


if __name__ == '__main__':
    main()
