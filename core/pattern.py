"""Are the neurons the brain lights regular? Nothing here assumes what the brain should think of (it may go on from a
line rather than repeat it); only the lit neurons themselves are compared (Jaccard), population by population:
  input  the PFC's external copies (L1: with the thoughts; L2, L3: its fixed middle layers)
  Hs     the hippocampus's own units        hip    its recall (lit copies)
  think  the PFC's internally lit copies     L4     the PFC's last layer (L4h: while the cue is heard)
After the first three syllables of a line (the cue), the 9 quiet ticks (input, L1-L3, L4h: while it is heard):
  again    the same cue twice (the ear differs each time): same line / different lines
  voice    the cue in voice C1 and in D1: same line / different lines
  follows  Spearman, over pairs of lines, of how alike their lit neurons are with how alike their input is
           (similar input -> similar neurons: regular; 0 = no relation)
One hearing of each never-heard line (learning on). Its pattern from the 4th syllable to 9 ticks after the line:
  back     the cue before the hearing / soon after / after 20 more dialogues: does it light the pattern of its own
           hearing more than other lines' hearings? rank 0 = own first, 0.5 = chance (each hearing's mean likeness
           to all cues taken out)
  keep     the cue soon vs later: same line / different lines (does the response to the same input last?)
usage: python -u -m core.pattern <ckpt or 'newborn'> [hip_scale=f] [gap=all|no_usual|no_weights|frozen]
       [reset=L4,pfc,hip,sound,motor,all: that part of the ongoing state quieted before every cue] [key=value]"""
import json
import sys
import time

import numpy as np
import torch

from .config import config
from .corpus_life import CorpusLife
from .exam import first_lines
from .net import DEV
from .world.corpus import common_order

N = 40
POPS = ('input', 'L1', 'L2', 'L3', 'Hs', 'hip', 'think', 'L4')
HEARD = ('input', 'L1', 'L2', 'L3')               # read while the cue is heard; the others after it


def grab(b):
    p = b.pfc
    return {'input': p.L1ext.on.clone(), 'L1': p.L1.on.clone(), 'L2': p.L2.on.clone(), 'L3': p.L3.on.clone(),
            'Hs': b.hip.Hs.on.clone(), 'hip': b.hip.h_int.clone(), 'think': p.x_int.clone(), 'L4': p.L4.on.clone()}


def clear(b, what):
    """probe only: a part of the ongoing state set to quiet before a cue (what: comma-separated parts)"""
    def quiet(pop):
        pop.on, pop.prev = torch.zeros_like(pop.on), torch.zeros_like(pop.prev)
        pop.age, pop.age_prev = torch.full_like(pop.age, 99), torch.full_like(pop.age_prev, 99)
    parts = what.split(',')
    if 'L4' in parts or 'pfc' in parts or 'all' in parts:
        quiet(b.pfc.L4)
    if 'pfc' in parts or 'all' in parts:
        for pop in b.pfc.pops():
            quiet(pop)
        b.pfc.x_int = torch.zeros_like(b.pfc.x_int)
    if 'hip' in parts or 'all' in parts:
        for pop in b.hip.pops():
            quiet(pop)
        b.hip.h_int = torch.zeros_like(b.hip.h_int)
    if 'sound' in parts or 'all' in parts:
        b.sound.state = [torch.zeros_like(x) for x in b.sound.state]
        b.ret_sound = [torch.zeros_like(x) for x in b.ret_sound]
        b.ret_touch = [torch.zeros_like(x) for x in b.ret_touch]
    if 'motor' in parts or 'all' in parts:
        b.ret_force, b.ret_stop = torch.zeros_like(b.ret_force), torch.zeros_like(b.ret_stop)
        for pop in (b.motor.force, b.motor.stop, b.motor.lit):
            quiet(pop)


def jacc(A, B):
    A, B = A.float(), B.float()
    i = A @ B.T
    return (i / (A.sum(1)[:, None] + B.sum(1)[None] - i).clamp_min(1.0)).cpu().numpy()


def same_diff(J):
    n = len(J)
    off = ~np.eye(n, dtype=bool)
    return [round(float(np.diag(J).mean()), 3), round(float(J[off].mean()), 3)]


def ranks(x):
    o = np.argsort(x, kind='stable')
    r = np.empty(len(x))
    r[o] = np.arange(len(x))
    _, inv = np.unique(x, return_inverse=True)
    return (np.bincount(inv, r) / np.bincount(inv))[inv]


def spear(a, b):
    a, b = ranks(a), ranks(b)
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / max(1e-12, np.sqrt((a * a).sum() * (b * b).sum())))


def back_rank(J):
    """J[cue i, hearing j]: the rank of each cue's own hearing (0 = first, 0.5 = chance)"""
    M = J - J.mean(0, keepdims=True)
    d = np.diag(M)
    return round(float(np.mean([((M[i] > d[i]).sum() + 0.5 * ((M[i] == d[i]).sum() - 1)) / (len(M) - 1)
                                for i in range(len(M))])), 3)


def main():
    src = sys.argv[1]
    over = {k: json.loads(v) for k, v in (a.split('=', 1) for a in sys.argv[2:] if '=' in a)}
    hip_scale = over.pop('hip_scale', None)
    gap = over.pop('gap', 'all')
    reset = over.pop('reset', None)
    pre = over.pop('pre', 4)                        # quiet ticks before every cue
    t0 = time.time()
    if src == 'newborn':
        life = CorpusLife(config(**{'levels_per_s': 4, 'n_brake': 20, 'birth_dialogues': 80, **over}), 0, 'data')
    else:
        life = torch.load(src, weights_only=False)['life']
    b = life.brain
    if 'g_ret' in over and src != 'newborn':        # the return strength (ARCH 三 前额叶: 回程强度可调), set at birth
        for area in (b.sound, b.touch):
            area.ret_w = [w * over['g_ret'] / life.c['g_ret'] for w in area.ret_w]
    if 'g44' in over and src != 'newborn':          # the last layer's internal push, also set at birth
        b.pfc.L44.gain = over['g44']
    life.c.update({k: v for k, v in over.items() if k in life.c})
    if hip_scale is not None:                       # diagnostic: the hippocampus's learned past scaled by f
        for tb in b.hip.plastic():
            tb.w.mul_(float(hip_scale))
    order = getattr(life, 'order', None)
    sets = {'heard': first_lines(life.train, common_order(life.train) if order is None else order, N, lo=4),
            'novel': first_lines(life.heldout, range(life.heldout.n), N, lo=4)}

    def cue(lines, v):
        """per population [lines, units]: input while the cue is heard, the others in the 9 quiet ticks after"""
        out = {k: [] for k in POPS + ('L4h',)}
        for ln in lines:
            srcs = [None] * pre + [(x, k, v, 0.0, 0.0) for x in ln[:3] for k in range(3)] + [None] * 9
            rec = []
            if reset:
                sn = b.snapshot()
                clear(b, reset)
            life.play(srcs, rec=lambda t: rec.append(grab(b)))
            if reset:
                b.restore(sn)
            for k in POPS:
                X = torch.stack([r[k] for r in rec])[pre - 4:]           # the last 4 quiet ticks, the cue, after
                out[k].append(X[4:13].any(0) if k in HEARD else X[13:].any(0))
                if k == 'L4':
                    out['L4h'].append(X[4:13].any(0))
        return {k: torch.stack(v) for k, v in out.items()}

    res = {}
    C = {nm: (cue(L, 'C1'), cue(L, 'C1'), cue(L, 'D1')) for nm, L in sets.items()}
    for nm, (c1, c2, d1) in C.items():
        r = {}
        for k in POPS + ('L4h',):
            Jc = jacc(c1[k], c1[k])
            iu = np.triu_indices(N, 1)
            r[k] = dict(again=same_diff(jacc(c1[k], c2[k])), voice=same_diff(jacc(c1[k], d1[k])),
                        follows=None if k == 'input' else round(spear(Jc[iu], jacc(c1['input'], c1['input'])[iu]), 3),
                        size=round(float(c1[k].float().sum(1).mean()), 1))
        res[nm] = r

    novel = sets['novel']
    was = b.learning
    b.learning = True
    hear = {k: [None] * N for k in POPS}
    rng = np.random.default_rng(1)
    for i in rng.permutation(N):                    # each never-heard line once, as in life (the states go on)
        rec = []
        for x in [None] * 4 + [(x, k, 'C1', 0.0, 0.0) for x in novel[i] for k in range(3)] + [None] * 9:
            b.step(torch.as_tensor(life.coch.a0([] if x is None else [x]), device=DEV), 0.0)
            rec.append(grab(b))
        for k in POPS:
            hear[k][i] = torch.stack([r[k] for r in rec])[13:].any(0)
    hear = {k: torch.stack(v) for k, v in hear.items()}
    b.learning = was
    soon = cue(novel, 'C1'), cue(novel, 'D1')
    # what may change in the 20 dialogues: all (as in life) / no_usual (own levels held) / no_weights (links held) /
    # frozen (neither: only the ongoing activity differs)
    ws = [tb.w.clone() for tb in b.plastic()] if gap == 'no_weights' else None
    tau = life.c['tau_usual']
    if gap == 'no_usual':
        life.c['tau_usual'] = 1e12
    b.learning = gap != 'frozen'
    n0 = life.env.n_ex
    while life.env.n_ex - n0 < 20:
        life.tick()
    b.learning = was
    life.c['tau_usual'] = tau
    if ws is not None:
        for tb, w in zip(b.plastic(), ws):
            tb.w.copy_(w)
    later = cue(novel, 'C1'), cue(novel, 'D1')
    before = C['novel'][0], C['novel'][2]
    for k in POPS + ('L4h',):
        res['novel'][k]['keep'] = same_diff(jacc(soon[0][k], later[0][k]))
        if k in HEARD + ('L4h',):
            continue
        res['novel'][k]['back'] = {w: [back_rank(jacc(X[0][k], hear[k])), back_rank(jacc(X[1][k], hear[k]))]
                                   for w, X in (('before', before), ('soon', soon), ('later', later))}
    print('PATTERN', src, json.dumps(dict(hip_scale=hip_scale, gap=gap, reset=reset, pre=pre, over=over, res=res, secs=round(time.time() - t0))), flush=True)


if __name__ == '__main__':
    main()
