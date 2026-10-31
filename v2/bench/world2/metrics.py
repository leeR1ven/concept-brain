# -*- coding: utf-8 -*-
"""evaluate(log) -> in-life metrics of a world2 run (all agents pooled, plus per agent).  See README.md for every key.

Scored free replies: 'q' records of kind probe / recast (the agent answered before any help) and social (greetings,
goodnights).  Requests ('req') are scored by their outcome; initiatives ('init') by category.  Headline groups:
  scene      here-and-now questions in NEW situations (this prompt + this exact eye input was never followed by a
             taught answer for this agent: exact lookup cannot answer them)
  held_out   (task, kind) pairs nobody ever answers
  memory     where / novel / person / myname probes (facts seen or told earlier; always probes)
  social     greetings and goodnights
  comprehension   requests (fetch / give / come) carried out within the window
"""
from __future__ import annotations

import collections

import numpy as np

from .care import APPROPRIATE, MEMORY_TASKS
from .core import bench_module
from . import lex

_bm = bench_module('metrics')
FREE = ('probe', 'recast', 'social')
ENC_BINS = (0, 1, 2, 3, 5, 9, 10 ** 9)
DELAY_BINS = (0, 300, 1000, 3000, 10000, 30000, 100000, 10 ** 9)
WORDS = sorted(set(lex.KINDS + lex.COLOURS + lex.COUNTS + lex.ROOMS + lex.CAREGIVERS + lex.AGENT_NAMES +
                   ['大', '小', '是', '不是', '要', '你好', '早上好', '晚安']), key=len, reverse=True)


def _mean(x):
    x = [v for v in x if v is not None]
    return float(np.mean(x)) if x else None


def _acc(rs):
    return _mean([r['reply'] in r['acceptable'] for r in rs])


def _lab(a, b):
    return f'{a}' if b == a + 1 else (f'{a}+' if b > 10 ** 8 else f'{a}-{b - 1}')


def _bin(t, edges):
    return int(np.clip(np.searchsorted(edges, t, side='right') - 1, 0, len(edges) - 2))


def score(rs):
    if not rs:
        return dict(n=0)
    acc = [r['reply'] in r['acceptable'] for r in rs]
    ef = [_bm.echo_free(r['reply'], r['prompt'], r['acceptable']) for r in rs]
    out = dict(n=len(rs), acc=_mean(acc), first=_mean([bool(r.get('first_ok')) for r in rs]),
               echo_free=_mean(ef), n_echo_free=sum(e is not None for e in ef),
               parrot=_mean([_bm.parrot(r['reply'], r['prompt']) for r in rs]),
               spoke=_mean([bool(r['reply']) for r in rs]),
               mean_len=_mean([len(r['reply']) for r in rs if r['reply']]))
    for nm, sel in (('held', lambda r: r['held_out']), ('seen', lambda r: not r['held_out'])):
        idx = [i for i, r in enumerate(rs) if sel(r)]
        out[nm] = dict(n=len(idx), acc=_mean([acc[i] for i in idx]))
    return out


def _words_in(text):
    out, s = set(), text
    for w in WORDS:
        if w in s:
            out.add(w)
            s = s.replace(w, ' ')
    return out


def evaluate(log, bins=10, per_agent=True):
    q = [r for r in log if r.get('type') == 'q']
    free = [r for r in q if r['kind'] in FREE]
    reqs = [r for r in log if r.get('type') == 'req']
    inits = [r for r in log if r.get('type') == 'init']
    mons = [r for r in log if r.get('type') == 'mon']
    a2a = [r for r in log if r.get('type') == 'a2a']
    eats = [r for r in log if r.get('type') == 'eat']
    t_max = max([r.get('t', 0) for r in log] + [1])
    N = len(mons[-1]['moves']) if mons else (max([r.get('agent', 0) for r in q + inits] + [0]) + 1)
    edges = np.linspace(0, t_max, bins + 1)
    out = dict(ticks=int(t_max), days=max([r.get('day', 0) for r in log] + [0]), n_agents=N, n_free=len(free),
               n_req=len(reqs), n_init=len(inits))
    # ---------------- per task + learning curves
    by_task = collections.defaultdict(list)
    for r in free:
        by_task[r['task']].append(r)
    out['tasks'] = {t: score(rs) for t, rs in sorted(by_task.items())}
    out['tasks']['ALL'] = score(free)

    def curve(rs, key='t'):
        c = [[0, 0] for _ in range(bins)]
        for r in rs:
            b = _bin(r[key], edges)
            c[b][0] += 1
            c[b][1] += r['reply'] in r['acceptable'] if 'acceptable' in r and 'reply' in r else bool(r.get('success'))
        return [(k / n if n else None) for n, k in c]
    out['curves'] = {t: curve(rs) for t, rs in sorted(by_task.items())}
    qa = [r for r in free if r['kind'] != 'social']
    groups = {
        'scene': [r for r in qa if r.get('needs_scene') and r.get('new_situation', True) and r['task'] not in MEMORY_TASKS],
        'scene_repeat': [r for r in qa if r.get('needs_scene') and not r.get('new_situation', True)
                         and r['task'] not in MEMORY_TASKS],
        'held_out': [r for r in qa if r['held_out']],
        'memory': [r for r in qa if r['task'] in MEMORY_TASKS and r.get('new_situation', True)],
        'memory_all': [r for r in qa if r['task'] in MEMORY_TASKS],
        'social': [r for r in free if r['kind'] == 'social'],
    }
    out['groups'] = {g: dict(n=len(rs), acc=_acc(rs), curve=curve(rs)) for g, rs in groups.items()}
    # ---------------- comprehension (requests carried out)
    comp = {}
    for typ in ('fetch', 'give', 'come'):
        rs = [r for r in reqs if r['req'] == typ]
        d = dict(n=len(rs), success=_mean([r['success'] for r in rs]),
                 median_ticks=(float(np.median([r['ticks'] for r in rs if r['success']]))
                               if any(r['success'] for r in rs) else None))
        if typ == 'come':
            named = [r for r in rs if r.get('named') and r.get('others')]
            d['others_came_rate'] = (sum(r['others_came'] for r in named) / max(1, sum(r['others'] for r in named))
                                     if named else None)
            d['n_named_with_others'] = len(named)
        if typ == 'fetch':
            d['success_visible'] = _mean([r['success'] for r in rs if r.get('visible')])
            d['success_not_visible'] = _mean([r['success'] for r in rs if not r.get('visible')])
        comp[typ] = d
    comp['ALL'] = dict(n=len(reqs), success=_mean([r['success'] for r in reqs]),
                       curve=[(_mean([r['success'] for r in reqs if _bin(r['t'], edges) == b])) for b in range(bins)])
    out['comprehension'] = comp
    # ---------------- memory: retention by delay x encounters (life mode's key metric)
    ret = {}
    for task in MEMORY_TASKS:
        rs = [r for r in groups['memory_all'] if r['task'] == task and r.get('ticks_since_told') is not None]
        by_d, by_e, grid = [], [], []
        for a, b in zip(DELAY_BINS, DELAY_BINS[1:]):
            sel = [r for r in rs if a <= r['ticks_since_told'] < b]
            if sel:
                by_d.append(dict(delay=f'{a}-{b}', n=len(sel), acc=_acc(sel)))
            for c, d in zip(ENC_BINS, ENC_BINS[1:]):
                sel2 = [r for r in sel if c <= r.get('n_experienced', 0) < d]
                if sel2:
                    grid.append(dict(delay=f'{a}-{b}', encounters=_lab(c, d), n=len(sel2), acc=_acc(sel2)))
        for c, d in zip(ENC_BINS, ENC_BINS[1:]):
            sel = [r for r in rs if c <= r.get('n_experienced', 0) < d]
            if sel:
                by_e.append(dict(encounters=_lab(c, d), n=len(sel), acc=_acc(sel),
                                 median_delay=float(np.median([r['ticks_since_told'] for r in sel]))))
        ret[task] = dict(n=len(rs), acc=_acc(rs), by_delay=by_d, by_encounters=by_e, delay_x_encounters=grid)
    sc = [r for r in qa if r['task'] not in MEMORY_TASKS]
    ret['qa_by_encounters'] = [dict(encounters=_lab(c, d), n=len(sel), acc=_acc(sel)) for c, d in zip(ENC_BINS, ENC_BINS[1:])
                               for sel in [[r for r in sc if c <= r.get('n_experienced', 0) < d]] if sel]
    out['retention'] = ret
    # ---------------- initiative, vocabulary, agent-agent talk, overlaps, body
    last = mons[-1] if mons else {}
    cg_aud = np.array(last.get('cg_audible_ticks', [0] * N), float)
    quiet = np.maximum(t_max - cg_aud, 1.0)
    cats = collections.Counter(r['category'] for r in inits)
    out['initiative'] = dict(
        n=len(inits), per_1000_quiet_ticks=1000 * len(inits) / float(quiet.sum()),
        appropriate=sum(r['appropriate'] for r in inits),
        appropriate_share=_mean([r['appropriate'] for r in inits]),
        appropriate_per_1000_ticks=1000 * sum(r['appropriate'] for r in inits) / max(1.0, t_max * N),
        reward=float(sum(r['reward'] for r in inits)), categories=dict(cats),
        curve=[sum(1 for r in inits if r['appropriate'] and _bin(r['t'], edges) == b) for b in range(bins)])
    vocab_t = [dict() for _ in range(N)]                     # agent -> word -> first tick used correctly
    for r in free:
        if r['reply'] in r['acceptable']:
            for w in _words_in(r['reply']):
                vocab_t[r['agent']].setdefault(w, r['t'])
    for r in inits:
        if r['appropriate']:
            for w in _words_in(r['text']):
                vocab_t[r['agent']].setdefault(w, r['t'])
    vc = [[sum(1 for t0 in vocab_t[j].values() if t0 < edges[b + 1]) for b in range(bins)] for j in range(N)]
    out['vocabulary'] = dict(final_mean=float(np.mean([len(v) for v in vocab_t])) if N else 0.0,
                             final_max=max([len(v) for v in vocab_t] + [0]),
                             curve_mean=[float(np.mean([vc[j][b] for j in range(N)])) for b in range(bins)] if N else [],
                             words_any=sorted({w for v in vocab_t for w in v}))
    out['a2a'] = dict(turns=len(a2a), per_1000_ticks=1000 * len(a2a) / max(1, t_max),
                      gives=sum(1 for r in log if r.get('type') == 'event' and r.get('what') == 'a2a_give'))
    m = {k: int(np.sum(last.get(k, [0]))) for k in ('mouth_ticks', 'syllables', 'garble', 'bouts', 'overlap_cg',
                                                    'overlap_partner', 'overlap_ag', 'interruptions',
                                                    'interruptions_partner', 'moves', 'bumps', 'turns', 'takes',
                                                    'puts', 'gives', 'eats', 'fails', 'initiatives',
                                                    'cg_audible_ticks')}
    out['overlap'] = dict(partner_ticks=m['overlap_partner'], partner_interruptions=m['interruptions_partner'],
                          any_cg_ticks=m['overlap_cg'], any_cg_interruptions=m['interruptions'],
                          agent_ticks=m['overlap_ag'], share_of_cg_speech=m['overlap_cg'] / max(1, m['cg_audible_ticks']))
    out['body'] = dict(m, reward=float(np.sum(last.get('reward', [0]))),
                       hunger_mean=_mean(last.get('hunger', [])), eats=len(eats))
    lat = collections.Counter(r['latency'] for r in free if r.get('latency') is not None)
    out['latency'] = dict(hist=dict(sorted(lat.items())), median=float(np.median(list(lat.elements()))) if lat else None)
    # imitation of naming statements / model replies
    st = [r for r in q if r['kind'] == 'statement']
    md = [r for r in q if r['kind'] == 'model']
    out['imitation'] = dict(statements=len(st), imitated=_mean([r['acc'] for r in st]),
                            after_model=len(md), after_model_exact=_mean([r.get('after_model_reply') == r.get('model_reply')
                                                                         for r in md]))
    # ---------------- stability per time bin
    stab = []
    snaps = sorted(mons, key=lambda r: r['t'])
    prev = {}
    for b in range(bins):
        a, e = edges[b], edges[b + 1]
        fr = [r for r in free if a <= r['t'] < e]
        it = [r for r in inits if a <= r['t'] < e]
        cur = next((s for s in snaps if s['t'] >= e), snaps[-1] if snaps else {})
        d = {k: float(np.sum(cur.get(k, [0]))) - float(np.sum(prev.get(k, [0]))) for k in
             ('mouth_ticks', 'overlap_cg', 'syllables', 'garble', 'moves', 'eats')}
        prev = cur
        span = max(e - a, 1) * max(N, 1)
        stab.append(dict(t=float((a + e) / 2), free=len(fr), spoke=_mean([bool(r['reply']) for r in fr]),
                         acc=_acc(fr), init_per_1000=1000 * len(it) / span, mouth_share=d['mouth_ticks'] / span,
                         overlap_ticks=d['overlap_cg'], garble_share=d['garble'] / max(d['syllables'], 1),
                         moves_per_1000=1000 * d['moves'] / span, eats=d['eats']))
    out['stability'] = stab
    flags = []
    sp = [s['spoke'] for s in stab if s['spoke'] is not None]
    if sp and max(sp) > 0.2 and sp[-1] < 0.5 * max(sp):
        flags.append('answer rate collapsed (last bin < half of the best)')
    if stab and stab[-1]['mouth_share'] > 0.5:
        flags.append('mouth active > 50 % of the last bin (runaway)')
    out['stability_flags'] = flags
    # ---------------- per agent
    if per_agent:
        pa = []
        scene_ids = {id(r) for r in groups['scene']}
        for j in range(N):
            fj = [r for r in free if r['agent'] == j]
            rj = [r for r in reqs if r['agent'] == j]
            ij = [r for r in inits if r['agent'] == j]
            pa.append(dict(agent=j, n_free=len(fj), acc=_acc(fj),
                           scene=_acc([r for r in fj if id(r) in scene_ids]),
                           memory=_acc([r for r in fj if r['task'] in MEMORY_TASKS]),
                           comprehension=_mean([r['success'] for r in rj]), n_req=len(rj),
                           initiatives=len(ij), appropriate=sum(r['appropriate'] for r in ij),
                           vocabulary=len(vocab_t[j])))
        out['per_agent'] = pa
    return out


def headline(res):
    f = lambda x: '-' if x is None else f'{x:.2f}'   # noqa: E731
    g, c, i = res['groups'], res['comprehension'], res['initiative']
    return (f"world2 {res['n_agents']} agents, {res['ticks']} ticks ({res['days']} days): free {res['n_free']} acc "
            f"{f(res['tasks']['ALL'].get('acc'))} | scene {f(g['scene']['acc'])} (n={g['scene']['n']}) held-out "
            f"{f(g['held_out']['acc'])} memory {f(g['memory']['acc'])} (n={g['memory']['n']}) social {f(g['social']['acc'])} | "
            f"comprehension {f(c['ALL']['success'])} (fetch {f(c['fetch']['success'])} give {f(c['give']['success'])} "
            f"come {f(c['come']['success'])}, n={c['ALL']['n']}) | initiative {i['n']} (appropriate {i['appropriate']}, "
            f"reward {i['reward']:.0f}) | vocab {res['vocabulary']['final_mean']:.1f} | a2a {res['a2a']['turns']} | "
            f"talk over partner {res['overlap']['partner_ticks']} ticks / {res['overlap']['partner_interruptions']} "
            f"interruptions (any caregiver {res['overlap']['any_cg_ticks']}) | "
            f"latency {res['latency']['median']} | flags {res['stability_flags'] or 'none'}")
