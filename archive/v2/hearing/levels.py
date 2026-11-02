# -*- coding: utf-8 -*-
"""Exact level facts of world2 / bench sound (PROVED by enumeration, no simulation):
  * per-cell noise pmf of max(0, clip(rint(N(0, s)), 0, 10)) and the fresh-pair (16 cells) noise energy tail;
  * fresh-pair energy e2 and aligned full-window energy E of every lexicon char in every voice, at attenuation a.
usage: python3 levels.py"""
import os, sys, json, math
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'bench') if os.path.isdir(os.path.join(HERE, 'bench')) else os.path.join(HERE, '..', 'bench'))
from world2 import World2, lex                     # noqa: E402


def cell_pmf(s, L=10):
    Phi = lambda x: 0.5 * (1 + math.erf(x / math.sqrt(2)))
    p = np.zeros(L + 1)
    for v in range(L + 1):
        lo = -np.inf if v == 0 else (v - 0.5) / s
        hi = np.inf if v == L else (v + 0.5) / s
        p[v] = (Phi(hi) if np.isfinite(hi) else 1.0) - (Phi(lo) if np.isfinite(lo) else 0.0)
    return p


def sum_pmf(p, n):
    out = np.array([1.0])
    for _ in range(n):
        out = np.convolve(out, p)
    return out


def main():
    res = {}
    for s in (0.5, 1.0, 2.0):
        p = cell_pmf(s)
        e2 = sum_pmf(p, 16)
        tail = {x: float(e2[x:].sum()) for x in (8, 10, 12, 14, 16, 18, 20, 24)}
        mean = float((np.arange(len(e2)) * e2).sum()); sd = float(np.sqrt(((np.arange(len(e2)) - mean) ** 2 * e2).sum()))
        mad = float((np.abs(np.arange(len(e2)) - mean) * e2).sum())
        Ew = sum_pmf(p, 48); mw = float((np.arange(len(Ew)) * Ew).sum())
        res[f'noise_{s}'] = dict(cell_pmf=[round(float(x), 5) for x in p[:6]], e2_mean=round(mean, 2), e2_sd=round(sd, 2),
                                 e2_mad=round(mad, 2), e2_tail={k: f'{v:.2e}' for k, v in tail.items()},
                                 window_mean=round(mw, 2), p_window_zero=f'{Ew[0]:.2e}')
    w = World2(n_agents=4, seed=0)
    v = w.voice
    chars = sorted(set(''.join(lex.all_lexicon_texts())))
    voices = ['other', 'other2', 'w2_cg2'] + [k for k in v.voice if k.startswith('w2_a')]
    for a in range(5):
        e2s, Es, nz = [], [], []
        for vk in voices:
            for c in chars:
                g = np.clip(np.clip(v.sound[c] + v.voice[vk], 0, v.levels) - a, 0, None)
                for ph in (0, 2, 4):
                    e2s.append(int(g[:, ph:ph + 2].sum()))
                Es.append(int(g.sum())); nz.append(int((g > 0).sum()))
        res[f'speech_a{a}'] = dict(e2_min=min(e2s), e2_p5=float(np.percentile(e2s, 5)), e2_mean=round(float(np.mean(e2s)), 1),
                                   E_min=min(Es), E_p5=float(np.percentile(Es, 5)), E_mean=round(float(np.mean(Es)), 1),
                                   E_max=max(Es), nonzero_cells_mean=round(float(np.mean(nz)), 1))
    res['n_lexicon_chars'] = len(chars); res['voices'] = len(voices)
    print(json.dumps(res, indent=1, ensure_ascii=False))
    json.dump(res, open(os.path.join(HERE, 'results', 'levels.json'), 'w'), indent=1, ensure_ascii=False)


if __name__ == '__main__':
    main()
