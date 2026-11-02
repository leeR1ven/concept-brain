# -*- coding: utf-8 -*-
"""Life metrics added by v2.1 (from a life log; bench life.py untouched):
  initiatives per tenth of life (你好 to an empty room, nonsense, alone, greeting, naming), appropriate share (all /
  partner present), social replies, free replies by group, role confusion of free replies, and same-life trivial
  references on the same free-reply items: echo, best constant per task (oracle-chosen on these items), and per-group.
usage: python3 life_metrics.py <life_log.pkl> [<life_log.pkl> ...]"""
import collections, json, pickle, sys
import numpy as np
QW = set('什么吗哪几谁')
LIFE_FREE = ('probe', 'recast', 'social')


def tenth(rs, T):
    out = [0] * 10
    for r in rs:
        out[min(9, int(10 * r['t'] / T))] += 1
    return out


def classify(r, prompts):
    rep, pr, acc = r['reply'], r['prompt'], r.get('acceptable') or []
    if rep in acc:
        return 'ok'
    if not rep:
        return 'silent'
    if rep == pr or (len(rep) >= 2 and rep in pr):
        return 'echo'
    if rep in prompts or (len(rep) >= 2 and any(p.startswith(rep) for p in prompts)) or (set(rep) & QW):
        return 'partner'
    return 'other'


def metrics(L):
    ex = [r for r in L if r.get('type') == 'ex']
    free = [r for r in ex if r.get('kind') in LIFE_FREE and r.get('acceptable')]
    T = max(r['t'] for r in L if 't' in r)
    ini = [r for r in L if r.get('type') == 'init']
    nh = [r for r in ini if r.get('text') == '你好']
    prompts = set(r['prompt'] for r in ex if isinstance(r.get('prompt'), str))
    cat = collections.Counter(r['category'] for r in ini)
    out = dict(ticks=int(T), n_init=len(ini), categories=dict(cat),
               init_per_tenth=tenth(ini, T),
               nihao_empty_per_tenth=tenth([r for r in nh if not r['partner']], T),
               nihao_rewarded_per_tenth=tenth([r for r in nh if r['reward'] > 0], T),
               nonsense_per_tenth=tenth([r for r in ini if r['category'] == 'nonsense'], T),
               alone_per_tenth=tenth([r for r in ini if r['category'] == 'alone'], T),
               greeting_per_tenth=tenth([r for r in ini if r['category'] == 'greeting'], T),
               naming=cat.get('naming', 0),
               appropriate=float(np.mean([r['appropriate'] for r in ini])) if ini else None,
               appropriate_partner=float(np.mean([r['appropriate'] for r in ini if r['partner']]))
               if any(r['partner'] for r in ini) else None,
               appropriate_last_tenth=float(np.mean([r['appropriate'] for r in ini if r['t'] >= 0.9 * T]))
               if any(r['t'] >= 0.9 * T for r in ini) else None,
               top_init_text=collections.Counter(r.get('text') for r in ini).most_common(12))
    acc = lambda rs: round(float(np.mean([r['reply'] in r['acceptable'] for r in rs])), 4) if rs else None
    groups = {'all': free, 'scene': [r for r in free if r.get('needs_scene') and r.get('new_situation', True)],
              'held_out': [r for r in free if r.get('held_out')], 'social': [r for r in free if r['kind'] == 'social'],
              'memory': [r for r in free if r['task'] in ('fact_name', 'fact_person', 'fact_place', 'novel', 'before')]}
    refs = {}
    for g, rs in groups.items():
        if not rs:
            continue
        by = collections.defaultdict(list)
        for r in rs:
            by[r['task']].append(r)
        best = sum(collections.Counter(a for r in rr for a in set(r['acceptable'])).most_common(1)[0][1] for rr in by.values())
        echo = float(np.mean([r['prompt'] in r['acceptable'] for r in rs]))
        triv = float(np.mean([(r['prompt'] in r['acceptable']) or (collections.Counter(
            a for x in by[r['task']] for a in set(x['acceptable'])).most_common(1)[0][0] in r['acceptable']) for r in rs]))
        cls = collections.Counter(classify(r, prompts) for r in rs)
        n = len(rs)
        last = [r for r in rs if r['t_prompt_start'] >= 0.9 * T]
        refs[g] = dict(n=n, agent=acc(rs), agent_last_tenth=acc(last), echo=round(echo, 4),
                       const_best=round(best / n, 4), echo_or_const=round(triv, 4),
                       role_conf=round((cls['echo'] + cls['partner']) / n, 4),
                       cls={k: round(v / n, 3) for k, v in cls.items()})
    out['free'] = refs
    pt = collections.defaultdict(list)
    for r in free:
        pt[r['task']].append(r)
    out['per_task'] = {t: dict(n=len(rs), acc=acc(rs), held=acc([r for r in rs if r.get('held_out')]),
                               n_held=sum(1 for r in rs if r.get('held_out'))) for t, rs in sorted(pt.items())}
    return out


if __name__ == '__main__':
    for p in sys.argv[1:]:
        m = metrics(pickle.load(open(p, 'rb')))
        print('==', p)
        print(json.dumps(m, ensure_ascii=False))
