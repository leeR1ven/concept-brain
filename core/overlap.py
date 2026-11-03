"""Why do thoughts after different lines overlap? Three checks on one save (frozen, same lines as core.regular):
1 input: overlap of the PFC's external input copies (per area) for same line vs different lines
2 which copies: share of the input / thoughts carried by copies lit for most lines (little feature) vs few
3 connections: how the learned L4->L1 weight is spread over targets; are the always-thought copies the most fed?
usage: python -u -m core.overlap <ckpt> [n_lines]"""
import json
import sys

import numpy as np
import torch

from .net import DEV
from .regular import jac
from .world.corpus import common_order


def main():
    src, n = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 20
    life = torch.load(src, weights_only=False)['life']
    b, D, ch = life.brain, life.train, life.brain.chains
    lines = []
    for d in common_order(D):
        t = D.turns(int(d))[0]
        if 2 <= len(t) <= 10:
            lines.append([int(x) for x in t])
        if len(lines) >= n:
            break
    keys = [(i, v) for i in range(len(lines)) for v in ('C1', 'D2')]
    R = {k: {} for k in ('ext', 'ext_quiet', 'think', 'L4', 'hip')}
    for i, ln in enumerate(lines):
        for v in ('C1', 'D2'):
            srcs = [None] * 4 + [(x, k, v, 0.0, 0.0) for x in ln for k in range(3)] + [None] * 12
            rec = []
            life.play(srcs, rec=lambda t: rec.append((b.pfc.L1ext.on.clone(), b.pfc.x_int.clone(),
                                                       b.pfc.L4.on.clone(), b.hip.h_int.clone())))
            E, X, L4, H = (torch.stack(z) for z in zip(*rec))
            R['ext'][(i, v)] = E.any(0)
            R['ext_quiet'][(i, v)] = E[:4].all(0)                       # lit before the line: not from the line
            R['think'][(i, v)] = X[-12:].any(0)
            R['L4'][(i, v)] = L4[-12:].any(0)
            R['hip'][(i, v)] = H[-12:].any(0)
    nl = len(lines)

    def sd(S, mask=None):
        f = (lambda x: x) if mask is None else (lambda x: x & mask)
        same = np.mean([jac(f(S[(i, 'C1')]), f(S[(i, 'D2')])) for i in range(nl)])
        diff = np.mean([jac(f(S[(i, 'C1')]), f(S[(j, 'D2')])) for i in range(nl) for j in range(nl) if i != j])
        size = np.mean([float(f(S[k]).sum()) for k in keys])
        return dict(same=round(float(same), 3), diff=round(float(diff), 3), lit=round(float(size)))

    out = {}
    # 1 the input itself, per area
    seg = {}
    for kind, l, a, z in ch.segs:
        m = torch.zeros(ch.n, dtype=torch.bool, device=DEV)
        m[a:z] = True
        seg[f'{kind}{l}'] = m
    out['1_input_all'] = sd(R['ext'])
    out['1_input_by_area'] = {k: sd(R['ext'], m) for k, m in seg.items()}
    out['1_input_lit_before_line'] = round(float(np.mean([float(R['ext_quiet'][k].sum()) for k in keys])))
    # 2 copies lit for most lines (half or more of the 2n hearings) vs the rest
    for name in ('ext', 'think', 'L4', 'hip'):
        F = torch.stack([R[name][k] for k in keys]).float().mean(0)
        common = F >= 0.5
        S = torch.stack([R[name][k] for k in keys]).float()
        share = float((S * common).sum() / S.sum().clamp_min(1))
        out[f'2_{name}'] = dict(ever_lit=int((F > 0).sum()), lit_for_most_lines=int(common.sum()),
                                share_of_lit_that_are_those=round(share, 3),
                                without_them=sd(R[name], ~common))
        if name == 'think':
            out['2_think_by_area'] = {k: round(float((S * m).sum() / S.sum().clamp_min(1)), 3) for k, m in seg.items()}
            think_common = common
    # 3 the learned L4->L1 connections
    L = b.pfc.L41
    w = L.w.flatten()
    win = torch.zeros(L.n_post + 1, device=DEV).index_add_(0, L.idx.flatten(), w)[:-1]
    srt = win.sort(descending=True).values
    tot = float(srt.sum())
    out['3_L41'] = dict(slots=int(w.numel()), above_0_01=int((w > 0.01).sum()), above_0_1=int((w > 0.1).sum()),
                        above_1=int((w > 1).sum()), w_max=round(float(w.max()), 2),
                        top1pct_targets_share=round(float(srt[:max(1, len(srt) // 100)].sum()) / max(tot, 1e-9), 3),
                        top10pct_targets_share=round(float(srt[:max(1, len(srt) // 10)].sum()) / max(tot, 1e-9), 3),
                        in_weight_of_common_thoughts=round(float(win[think_common].mean()), 2) if think_common.any() else None,
                        in_weight_of_others=round(float(win[~think_common].mean()), 2),
                        usual_of_common_thoughts=round(float(b.pfc.usual[think_common].mean()), 2) if think_common.any() else None,
                        usual_of_others=round(float(b.pfc.usual[~think_common].mean()), 2))
    L = b.pfc.L44
    w = L.w.flatten()
    out['3_L44'] = dict(slots=int(w.numel()), above_0_1=int((w > 0.1).sum()), above_1=int((w > 1).sum()),
                        w_max=round(float(w.max()), 2))
    print('OVERLAP', json.dumps(out), flush=True)


if __name__ == '__main__':
    main()
