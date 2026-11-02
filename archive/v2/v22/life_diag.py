# -*- coding: utf-8 -*-
"""Step-3 diagnostics from a life log (world ground truth; the agent never reads any of this):
  * lines(L): partner lines (t0, t1, text, type, reward, who) and own utterances (t0, t1, text, source);
  * ral_by_type(L, ral_log, off): for every heard partner line resolved by role-aware listening, its true type
    (question / correct / confirm / tell_silent / social / statement / init_reply / overheard_*), and the share
    committed (learned as an answer) vs cancelled;
  * after_type(L): share of each line type after which I speak within W ticks, and what I say."""
import collections
import numpy as np


def lines(L):
    P, U = [], []
    for r in L:
        ty = r.get('type')
        if ty == 'init':
            U.append((r['t'], r['t_end'], r['text'], 'init'))
            if r.get('appropriate'):
                txt = '你好' if r['category'] == 'greeting' else '对是' + r.get('object', '')
                t0 = r['t_end'] + 2
                P.append((t0, t0 + 3 * len(txt), txt, 'init_reply', r['reward'], 'other'))
            continue
        if ty != 'ex':
            continue
        k = r['kind']
        pt = r['prompt']
        if isinstance(pt, str) and pt:
            pty = {'probe': 'question', 'recast': 'question', 'social': 'social', 'statement': 'statement',
                   'overheard': 'overheard_q'}.get(k, k)
            P.append((r['t_prompt_start'], r['t_prompt_end'], pt, pty, 0.0, r.get('prompt_by', 'other')))
        if r.get('t_first_mouth') is not None and r.get('reply'):
            U.append((r['t_first_mouth'], r['t_stop'] or r['t_first_mouth'] + 3, r['reply'], k + '_reply'))
        if k == 'recast':
            txt, t1 = r['recast'], r['t_recast_end']
            good = r['reward'] > 0
            P.append((t1 - 3 * len(txt), t1, txt, 'confirm' if good else ('correct' if r['reply'] else 'tell_silent'),
                      r['reward'], 'other'))
            if r.get('after_recast_t_first_mouth') is not None and r.get('after_recast_reply'):
                U.append((r['after_recast_t_first_mouth'], r['after_recast_t_stop'] or r['after_recast_t_first_mouth'] + 3,
                          r['after_recast_reply'], 'after_recast'))
        if k == 'overheard' and r.get('model_reply'):
            t0m = r['t_prompt_end'] + 3
            P.append((t0m, t0m + 3 * len(r['model_reply']), r['model_reply'], 'overheard_a', 0.0, 'other'))
    P.sort(); U.sort()
    return P, U


def ral_by_type(L, ral_log, off=0, tol=6):
    P, _ = lines(L)
    t0s = np.array([p[0] for p in P])
    out = collections.defaultdict(collections.Counter)
    for (t_on, t_end, what, after_own, cont, n) in ral_log:
        t = t_on - off
        j = int(np.searchsorted(t0s, t, side='right')) - 1
        ty = P[j][3] if (j >= 0 and 0 <= t - t0s[j] <= tol) else 'unmatched'
        out[ty][what] += 1
        if cont:
            out[ty]['contingent'] += 1
    return {ty: dict(n=c['commit'] + c['cancel'], commit=c['commit'], cancel=c['cancel'], contingent=c['contingent'],
                     commit_share=round(c['commit'] / max(1, c['commit'] + c['cancel']), 3)) for ty, c in out.items()}


def after_type(L, W=12):
    P, U = lines(L)
    Ut0 = np.array([u[0] for u in U]) if U else np.zeros(0)
    by = collections.defaultdict(list)
    for i, p in enumerate(P):
        t1, ty = p[1], p[3]
        nxt = P[i + 1][0] if i + 1 < len(P) else 10 ** 12
        j = int(np.searchsorted(Ut0, t1))
        sp = j < len(U) and U[j][0] - t1 <= W and U[j][0] < nxt
        by[ty].append((sp, U[j][2] if sp else None, p[2]))
    return {ty: dict(n=len(v), spoke=round(float(np.mean([x[0] for x in v])), 3),
                     repeat=round(float(np.mean([x[1] == x[2] for x in v if x[0]])), 3) if any(x[0] for x in v) else None,
                     top=collections.Counter(x[1] for x in v if x[0]).most_common(6))
            for ty, v in by.items()}
