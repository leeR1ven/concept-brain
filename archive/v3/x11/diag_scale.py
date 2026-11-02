"""Diagnostic (frozen, learning OFF; reads a copy of a curve checkpoint and never writes it; the newborn is rebuilt
from the same seed): why is the PFC's next-syllable prediction at chance? Same held-out occurrences as the curve.
 (a) echo test: the same x_int window (t0+1..t0+3) ranked against the target syllable and the three before it
     (previous, before previous, three back), each next to the unigram reference for that syllable;
 (b) learned L4->L1 links alone: thresholded (patch inhibition, > theta_1, as x_int) and graded (raw drive summed over
     the window, scored per syllable as the mean drive over its code = dot / |code|_1, and as the raw sum); the birth
     loop alone likewise; drive magnitudes against theta_1;
 (c) hippocampus: distribution of H1's internal drive (learned Hs->H1 after its patch inhibition) against theta_h1;
     number of Hs->H1 and Hs<->Hs synapses with weight > 0;
 (d) L4 sharing: fraction on and Jaccard of L4 on t0+3 between occurrences of the same / different syllables.
usage: python3 diag_scale.py --run x11_s1 --seed 0 --out diag.json [--n_heldout 200] [--no_newborn] [--replica 1000]"""
import argparse
import os

import numpy as np
import torch

from common import DEV, log, save_json
from life11 import Life11, load_bases, x11_config
from probes11 import Frozen, heldout_plan, n_sound_copies, rank_scores
from world11 import Dialogues

HERE = os.path.dirname(os.path.abspath(__file__))


def graded(M, C, true, how):
    """rank by mean drive over each syllable's code (dot / |code|_1) or by the raw sum (dot)"""
    S = M @ C.T
    if how == 'mean':
        S = S / C.sum(1)[None].clamp_min(1e-9)
    S = S + 1e-6 * torch.rand(S.shape, device=DEV)
    tt = torch.as_tensor(true, device=DEV)
    rank = (S > S.gather(1, tt[:, None])).sum(1)
    return dict(top1=float((rank == 0).float().mean()), top5=float((rank < 5).float().mean()),
                mrr=float((1.0 / (rank.float() + 1)).mean()))


def diagnose(life, cd, test, n_heldout, uni):
    b = life.brain
    b.learning = False
    c = b.c
    nsnd = n_sound_copies(b)
    stream, occ = heldout_plan(test, n_heldout, 4242)
    pos = {s: i for i, s in enumerate(cd['sids'])}
    occ = [o for o in occ if o[1] in pos]
    n, T = len(occ), len(stream)
    o3 = np.full(T + 4, -1, np.int64)
    for j, o in enumerate(occ):
        o3[o[0] + 1:o[0] + 4] = j
    keys = ('x_int', 'learned_lit', 'learned_drive', 'loop_drive')
    P = {k: torch.zeros((n, nsnd), device=DEV) for k in keys}
    dmax = {k: [] for k in ('learned', 'loop', 'h1')}
    above = {k: [] for k in ('learned', 'loop', 'h1')}
    L4at = torch.zeros((n, b.pfc.n), dtype=torch.bool, device=DEV)
    p = b.pfc
    with Frozen(life):
        life.coch.rng = np.random.default_rng(4242)
        for t, x in enumerate(stream):
            a0 = life.coch.a0([] if x is None else [x[:5]])
            b.step(torch.as_tensor(a0, device=DEV), 0.0)
            a4 = p.L4.prev
            dl = p.L41.drive(a4)
            dp = p.loop.drive(a4)
            j = o3[t]
            if j >= 0:
                P['x_int'][j] += p.x_int[:nsnd].float()
                P['learned_lit'][j] += (p.inhib(dl, 0) > p.theta[0])[:nsnd].float()
                P['learned_drive'][j] += dl[:nsnd]
                P['loop_drive'][j] += dp[:nsnd]
                if t == occ[j][0] + 3:
                    L4at[j] = p.L4.on
            if x is not None and t % 3 == 0:
                dh = b.hip.inhib1(b.hip.H1t.drive(b.hip.Hs.prev))
                for k, d, th in (('learned', p.inhib(dl, 0), p.theta[0]), ('loop', p.inhib(dp, 0), p.theta[0]),
                                 ('h1', dh, c['theta_h1'])):
                    dmax[k].append(float(d.max()) / th)
                    above[k].append(float((d > th).sum()))
        life.coch.rng = np.random.default_rng(0)
    tot = uni.sum()
    order = np.argsort(-uni, kind='stable')
    res = dict(at_dialogues=life.world.n_ex, n_occ=n)
    # (a) echo test
    rows = {}
    for lag, idx in (('target (next)', 1), ('previous', 2), ('before previous', 3), ('three back', 7)):
        sel = np.array([o[idx] in pos for o in occ]) if idx != 1 else np.ones(n, bool)
        lab = np.array([pos[o[idx]] for o, k in zip(occ, sel) if k])
        ident = np.array([o[idx] for o, k in zip(occ, sel) if k])
        r = rank_scores(P['x_int'][torch.as_tensor(sel, device=DEV)], cd['code3'], cd['sids'], lab)[0]
        r['unigram_ref'] = dict(top1=float(np.mean(ident == order[0])), top5=float(np.mean(np.isin(ident, order[:5]))))
        rows[lag] = r
    res['a_echo_x_int'] = rows
    # (b) sources
    true = np.array([pos[o[1]] for o in occ])
    prev = np.array([pos[o[2]] for o in occ])
    res['b_sources'] = {
        'learned_lit (thresholded)': dict(target=rank_scores(P['learned_lit'], cd['code3'], cd['sids'], true)[0],
                                          previous=rank_scores(P['learned_lit'], cd['code3'], cd['sids'], prev)[0],
                                          lit_per_window=float(P['learned_lit'].sum(1).mean())),
        'learned_drive (graded)': dict(target_mean=graded(P['learned_drive'], cd['code3'], true, 'mean'),
                                       target_sum=graded(P['learned_drive'], cd['code3'], true, 'sum'),
                                       target_cos=rank_scores(P['learned_drive'], cd['code3'], cd['sids'], true)[0],
                                       previous_mean=graded(P['learned_drive'], cd['code3'], prev, 'mean'),
                                       drive_per_copy_tick=float(P['learned_drive'].mean() / 3)),
        'loop_drive (graded)': dict(target_mean=graded(P['loop_drive'], cd['code3'], true, 'mean'),
                                    previous_mean=graded(P['loop_drive'], cd['code3'], prev, 'mean'),
                                    drive_per_copy_tick=float(P['loop_drive'].mean() / 3)),
    }
    res['b_magnitude'] = {k: dict(max_over_theta_median=float(np.median(v)), max_over_theta_p90=float(np.percentile(v, 90)),
                                  units_above_theta_mean=float(np.mean(above[k])), ticks=len(v))
                          for k, v in dmax.items()}
    # (c) hippocampal synapses
    h = b.hip
    res['c_hip'] = dict(H1t_synapses_w_pos=int((h.H1t.w > 0).sum()), H1t_synapses=int((h.H1t.idx < h.n1).sum()),
                        H1t_w_max=float(h.H1t.w.max()), HH_synapses_w_pos=int((h.HH.w > 0).sum()),
                        HH_synapses=int((h.HH.idx < h.nh).sum()), g_h1=c['g_h1'], theta_h1=c['theta_h1'],
                        h1_drive=res['b_magnitude']['h1'])
    res['learned_tables'] = dict(L41_w_pos=int((p.L41.w > 0).sum()), L41_synapses=int((p.L41.idx < p.n1).sum()),
                                 L41_w_max=float(p.L41.w.max()), L44_w_pos=int((p.L44.w > 0).sum()))
    # (d) L4 sharing
    L = L4at.float()
    inter = L @ L.T
    cnt = L.sum(1)
    J = inter / (cnt[:, None] + cnt[None] - inter).clamp_min(1)
    same = torch.as_tensor(true[:, None] == true[None], device=DEV)
    off = ~torch.eye(n, dtype=torch.bool, device=DEV)
    res['d_L4'] = dict(frac_on=float(cnt.mean() / p.n), J_same_syllable=float(J[same & off].mean()),
                       J_different=float(J[~same].mean()))
    return res


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', default='x11_s1')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default='diag.json')
    ap.add_argument('--n_heldout', type=int, default=200)
    ap.add_argument('--no_newborn', action='store_true')
    ap.add_argument('--replica', type=int, default=0, help='also overhear this many dialogues from birth (learning on)')
    a = ap.parse_args()
    ck = os.path.expanduser(f'~/ckpt/{a.run}_seed{a.seed}')
    c = x11_config(1)
    bases = load_bases(HERE)
    train = Dialogues.load(os.path.join(HERE, 'data', 'train.npz'))
    birth = Dialogues.load(os.path.join(HERE, 'data', 'birth.npz'))
    test = Dialogues.load(os.path.join(HERE, 'data', 'heldout.npz'))
    uni = np.bincount(train.syl.astype(np.int64), minlength=len(bases) * 5).astype(np.float64)
    cd = torch.load(os.path.join(ck, 'codes.pt'), map_location=DEV, weights_only=False)
    out = {}
    life, _ = Life11.load(os.path.join(ck, 'life.pt'), c, a.seed, train, bases)
    log('loaded checkpoint at', life.world.n_ex, 'dialogues')
    out['checkpoint'] = diagnose(life, cd, test, a.n_heldout, uni)
    log('checkpoint', out['checkpoint'])
    save_json(a.out, out)
    del life
    torch.cuda.empty_cache()
    if not a.no_newborn:
        nb = Life11(c, a.seed, train, birth, bases)                  # the same seed's newborn, rebuilt
        out['newborn'] = diagnose(nb, cd, test, a.n_heldout, uni)
        log('newborn', out['newborn'])
        save_json(a.out, out)
        if a.replica:                     # the curve's first dialogues again (same seed, same stream, same code):
            nb.brain.learning = True      # the curve's own 1k state was overwritten by its later checkpoint
            nb.run_dialogues(a.replica)
            out[f'replica_{a.replica}'] = diagnose(nb, cd, test, a.n_heldout, uni)
            log('replica', out[f'replica_{a.replica}'])
            save_json(a.out, out)
