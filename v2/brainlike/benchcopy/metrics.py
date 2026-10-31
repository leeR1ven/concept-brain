# -*- coding: utf-8 -*-
"""evaluate(log) -> dict of scores per stage and task (see README.md for every key).

Free replies (kind 'probe' and 'recast') are the scored answers: the agent answered with no model reply, cue or
recast before it.  Guided replies (kind 'force') and replies after a model (kind 'model') are scored separately.
"""
from __future__ import annotations

import collections

import numpy as np

FREE = ('probe', 'recast')


def ok(reply, acc):
    return reply in acc


def first_ok(reply, acc):
    return bool(reply) and reply[0] in {a[0] for a in acc if a}


def echo_free(reply, prompt, acc):
    """the chars that appear in the prompt are removed from the reply and from every acceptable string, then exact
    match.  An item is scorable only if EVERY acceptable string keeps >= 1 char (the answer needs information that is
    not in the prompt; e.g. 你要X吗 -> 要X is not scorable, 苹果是什么颜色 -> 红色 is scored on 红).  None = not
    scorable.  Echo scores 0 on every scorable item; the oracle scores 1."""
    ps = set(prompt)
    accr = {''.join(c for c in a if c not in ps) for a in acc}
    if not accr or '' in accr:
        return None
    return ''.join(c for c in reply if c not in ps) in accr


def parrot(reply, prompt):
    ch = [c for c in reply if c != '~']
    if not ch:
        return None
    return float(np.mean([c in prompt for c in ch]))


def _mean(x):
    x = [v for v in x if v is not None]
    return float(np.mean(x)) if x else None


def score(recs):
    """acceptable-reply metrics of a list of records (free replies)."""
    if not recs:
        return dict(n=0)
    acc = [ok(r['reply'], r['acceptable']) for r in recs]
    fst = [first_ok(r['reply'], r['acceptable']) for r in recs]
    ef = [echo_free(r['reply'], r['prompt'], r['acceptable']) for r in recs]
    out = dict(n=len(recs), acc=_mean(acc), first=_mean(fst),
               echo_free=_mean(ef), n_echo_free=sum(e is not None for e in ef),
               parrot=_mean([parrot(r['reply'], r['prompt']) for r in recs]),
               spoke=_mean([bool(r['reply']) for r in recs]),
               mean_len=_mean([len(r['reply']) for r in recs if r['reply']]))
    for name, sel in (('held', lambda r: r['held_out']), ('seen', lambda r: not r['held_out']),
                      ('familiar', lambda r: (not r['held_out']) and r['familiar'])):
        idx = [i for i, r in enumerate(recs) if sel(r)]
        out[name] = dict(n=len(idx), acc=_mean([acc[i] for i in idx]), first=_mean([fst[i] for i in idx]),
                         echo_free=_mean([ef[i] for i in idx]))
    return out


def mechanics(recs):
    """timing and articulation of the agent's mouth (all exchanges of the list)."""
    c = collections.Counter()
    lat = collections.Counter()
    runs = []
    for r in recs:
        c['exchanges'] += 1
        c['mouth_while_partner'] += r.get('mouth_while_partner', 0)
        c['exchanges_with_overlap'] += r.get('mouth_while_partner', 0) > 0
        c['mouth_during_prompt'] += r.get('mouth_during_prompt', 0)
        if r['kind'] in FREE:
            c['free'] += 1
            c[f"stop:{r.get('stop')}"] += 1
            if r.get('agent_stop'):
                c[f"agent_stop:{r['agent_stop']}"] += 1
            c['extra_turns'] += r.get('extra_turns', 0)
            if r.get('latency') is not None:
                lat[r['latency']] += 1
        runs += r.get('runs', [])                        # complete syllables only (see world.Track)
        c['garble'] += r.get('garble', 0)
    n_free = max(c['free'], 1)
    out = dict(exchanges=c['exchanges'], mouth_while_partner=c['mouth_while_partner'],
               exchanges_with_overlap=c['exchanges_with_overlap'], mouth_during_prompt=c['mouth_during_prompt'],
               syllables=len(runs), not_3_ticks=sum(x != 3 for x in runs),
               frac_not_3=(sum(x != 3 for x in runs) / len(runs)) if runs else None,
               garble=c['garble'], extra_turns=c['extra_turns'],
               spoke=sum(lat.values()) / n_free if c['free'] else None,
               latency=dict(sorted(lat.items())),
               latency_median=float(np.median(list(lat.elements()))) if lat else None,
               stop={k[5:]: v for k, v in c.items() if k.startswith('stop:')},
               agent_stop={k[11:]: v for k, v in c.items() if k.startswith('agent_stop:')})
    return out


def learning(recs):
    """learning in free use: the same items early vs late (free replies on seen items; thirds by exchange index)."""
    rs = [r for r in recs if r['kind'] in FREE and not r['held_out']]
    if len(rs) < 6:
        return dict(n=len(rs))
    n = len(rs)
    early, late = rs[:n // 3], rs[-(n // 3):]
    ie = collections.defaultdict(list); il = collections.defaultdict(list)
    for r in early:
        ie[str(r['item'])].append(ok(r['reply'], r['acceptable']))
    for r in late:
        il[str(r['item'])].append(ok(r['reply'], r['acceptable']))
    both = sorted(set(ie) & set(il))
    return dict(n=n, acc_early=_mean([ok(r['reply'], r['acceptable']) for r in early]),
                acc_late=_mean([ok(r['reply'], r['acceptable']) for r in late]),
                same_items=len(both),
                same_items_early=_mean([np.mean(ie[k]) for k in both]),
                same_items_late=_mean([np.mean(il[k]) for k in both]))


def one_shot(recs, bins=(0, 300, 1000, 3000, 10000, 10 ** 9)):
    rs = [r for r in recs if r.get('one_shot') and r['kind'] in FREE]
    out = dict(n=len(rs), acc=_mean([ok(r['reply'], r['acceptable']) for r in rs]))
    curve = {}
    for a, b in zip(bins, bins[1:]):
        sel = [r for r in rs if r.get('ticks_since_taught') is not None and a <= r['ticks_since_taught'] < b]
        if sel:
            curve[f'{a}-{b}'] = dict(n=len(sel), acc=_mean([ok(r['reply'], r['acceptable']) for r in sel]))
    out['retention'] = curve
    return out


def guided(recs):
    rs = [r for r in recs if r['kind'] == 'force']
    if not rs:
        return dict(n=0)
    return dict(n=len(rs), exact=_mean([r['reply'] == r.get('cued') for r in rs]))


def after_model(recs):
    rs = [r for r in recs if r['kind'] == 'model']
    if not rs:
        return dict(n=0)
    spoke = [r for r in rs if r.get('after_model_reply')]
    return dict(n=len(rs), spoke=len(spoke) / len(rs),
                imitation_exact=_mean([r['after_model_reply'] == r['model_reply'] for r in rs]),
                acc=_mean([ok(r['after_model_reply'], r['acceptable']) for r in rs]),
                jumped_in=_mean([bool(r['reply']) and (r['t_first_mouth'] or 0) < r.get('t_model_end', 0) for r in rs]))


def play_stats(recs):
    """stage A free play: babbled syllables, garble, distinct chars, share in the micro-world lexicon, imitations."""
    from world import LEXICON_CHARS
    rs = [r for r in recs if r['kind'] == 'play']
    if not rs:
        return dict(n=0)
    ch = [c for r in rs for c in r['reply']]
    clean = [c for c in ch if c != '~']
    return dict(n=len(rs), syllables=len(ch), per_period=len(ch) / len(rs),
                garble=(len(ch) - len(clean)) / len(ch) if ch else None, distinct=len(set(clean)),
                in_lexicon=_mean([c in LEXICON_CHARS for c in clean]),
                imitated=sum(len([x for x in r.get('imitations', []) if x[1]]) for r in rs),
                reward=float(sum(r.get('reward', 0.0) for r in rs)))


def _f1(a, b):
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    return 2 * len(sa & sb) / (len(sa) + len(sb))


def content20(recs, seed=7):
    """ARCH_v1's 20-way discrimination (run.py content_metrics): char-set F1 of the reply vs its true reply and 19
    other replies of the same set (top-1 chance 0.05, AUC 0.5); copy = share of said chars in the prompt (vs a
    shuffled prompt)."""
    rs = [r for r in recs if r['kind'] in FREE]
    n = len(rs)
    if n < 3:
        return dict(n=n)
    R = [r['acceptable'][0] for r in rs]; Q = [r['prompt'] for r in rs]; outs = [r['reply'] for r in rs]
    rg = np.random.default_rng(seed)
    kk = min(19, n - 1)
    others = [(i + 1 + rg.choice(n - 1, kk, replace=False)) % n for i in range(n)]
    shuf = [(i + 1 + rg.integers(n - 1)) % n for i in range(n)]
    top1, auc = [], []
    for i, o in enumerate(outs):
        s0 = _f1(o, R[i]); so = np.array([_f1(o, R[j]) for j in others[i]])
        auc.append(np.mean((s0 > so) + 0.5 * (s0 == so)))
        mx = max(s0, so.max()); top1.append((s0 == mx) / (1 + (so == mx).sum()))
    said = [[c for c in o if c != '~'] for o in outs]
    cp = [np.mean([c in Q[i] for c in s]) for i, s in enumerate(said) if s]
    cpx = [np.mean([c in Q[shuf[i]] for c in s]) for i, s in enumerate(said) if s]
    return dict(n=n, top1=float(np.mean(top1)), auc=float(np.mean(auc)), copy=_mean(cp), copy_shuf=_mean(cpx),
                spoke=_mean([bool(o) for o in outs]), mean_len=_mean([len(o) for o in outs if o]))


def evaluate(log, kinds=FREE):
    """scores per stage: res[stage][task] (+ 'ALL', 'SCENE', 'NOSCENE') = score(); res[stage]['mech'],
    ['guided'], ['after_model'], ['learning'], ['one_shot']; res['ALL'] over all stages."""
    res = {}
    stages = sorted({r['stage'] for r in log})
    for st in stages + ['ALL']:
        recs = log if st == 'ALL' else [r for r in log if r['stage'] == st]
        free = [r for r in recs if r['kind'] in kinds and r.get('acceptable')]
        S = {}
        for task in sorted({r['task'] for r in free}):
            S[task] = score([r for r in free if r['task'] == task])
        S['ALL'] = score(free)
        S['SCENE'] = score([r for r in free if r['needs_scene']])
        S['NOSCENE'] = score([r for r in free if not r['needs_scene']])
        S['mech'] = mechanics(recs)
        S['guided'] = guided(recs)
        S['after_model'] = after_model(recs)
        S['learning'] = learning(recs)
        S['one_shot'] = one_shot(recs)
        if any(r['kind'] == 'play' for r in recs):
            S['play'] = play_stats(recs)
        for task in ('corpus', 'corpus_fam'):
            rr = [r for r in free if r['task'] == task]
            if rr:
                S[task + '_20way'] = content20(rr)
        res[st] = S
    return res


def headline(res, stage=None):
    """one line per stage: free acceptable rate on held-out / seen / scene questions + key mechanics."""
    lines = []
    for st, S in res.items():
        if stage is not None and st != stage:
            continue
        a = S['ALL']
        if not a.get('n'):
            continue
        m = S['mech']
        f = lambda x: '-' if x is None else f'{x:.2f}'   # noqa: E731
        lines.append(f"[{st}] free n={a['n']} acc {f(a['acc'])} (held {f(a['held']['acc'])} n={a['held']['n']}, "
                     f"seen {f(a['seen']['acc'])}, scene {f(S['SCENE'].get('acc'))}) first {f(a['first'])} "
                     f"echo-free {f(a['echo_free'])} parrot {f(a['parrot'])} | overlap {m['mouth_while_partner']} "
                     f"lat {m['latency_median']} not3 {m['not_3_ticks']}/{m['syllables']} garble {m['garble']} "
                     f"extra {m['extra_turns']} stop {m['stop']}")
    return '\n'.join(lines)
