# -*- coding: utf-8 -*-
"""Analysis of a run's log (+ thoughts): latency vs difficulty, thinking steps at release, confidence calibration,
turn-taking outside free replies, learning curves.  usage: python3 analyze.py out/think_<tag> [out/bench_<base>]"""
import sys, pickle, collections, json
import numpy as np

FREE = ('probe', 'recast')


def load(prefix):
    L = pickle.load(open(prefix + '_log.pkl', 'rb'))
    try:
        TH = pickle.load(open(prefix + '_thoughts.pkl', 'rb'))
    except FileNotFoundError:
        TH = None
    return L, TH


def match_thoughts(L, TH):
    """the reply thought about each free reply's prompt (opened 0..3 ticks after the prompt's end)."""
    if TH is None:
        return {}
    by_open = collections.defaultdict(list)
    for t in TH:
        if t['kind'] == 'reply':
            by_open[t['t_open']].append(t)
    out = {}
    for r in L:
        if r['kind'] in FREE:
            for d in range(0, 4):
                c = by_open.get(r['t_prompt_end'] + d)
                if c:
                    out[r['i'], r['dlg'], r['t_prompt_end']] = c[0]
                    break
    return out


def report(prefix):
    L, TH = load(prefix)
    M = match_thoughts(L, TH)
    print(f"== {prefix}: {len(L)} records, {len(TH) if TH else 0} thoughts, matched {len(M)}")
    for st in sorted({r['stage'] for r in L}):
        fr = [r for r in L if r['stage'] == st and r['kind'] in FREE]
        if not fr:
            continue
        print(f"-- stage {st}: free replies {len(fr)}")
        rows = []
        for task in sorted({r['task'] for r in fr}):
            rs = [r for r in fr if r['task'] == task]
            for grp, sel in (('seen', lambda r: not r['held_out']), ('held', lambda r: r['held_out'])):
                g = [r for r in rs if sel(r)]
                if not g:
                    continue
                lat = [r['latency'] for r in g if r.get('latency') is not None]
                ks = [M[(r['i'], r['dlg'], r['t_prompt_end'])]['k_rel'] for r in g
                      if (r['i'], r['dlg'], r['t_prompt_end']) in M and M[(r['i'], r['dlg'], r['t_prompt_end'])]['k_rel'] is not None]
                acc = np.mean([r['reply'] in r['acceptable'] for r in g])
                rows.append((task, grp, len(g), acc, np.mean([bool(r['reply']) for r in g]),
                             np.mean(lat) if lat else None, np.percentile(lat, 90) if lat else None,
                             np.mean(ks) if ks else None))
        print("   task    grp   n     acc   spoke  lat_mean lat_p90 k@release")
        for t, g, n, a, s, lm, l9, k in rows:
            print(f"   {t:7s} {g:5s} {n:5d} {a:6.3f} {s:6.3f} {lm if lm is None else round(lm, 2)!s:>8} "
                  f"{l9 if l9 is None else round(l9, 1)!s:>7} {k if k is None else round(k, 2)!s:>8}")
        # calibration: accuracy by release confidence and by latency
        if M:
            es, ok, lats, ks = [], [], [], []
            for r in fr:
                t = M.get((r['i'], r['dlg'], r['t_prompt_end']))
                if t is None or t['e_rel'] is None or not r['reply']:
                    continue
                es.append(t['e_rel']); ok.append(r['reply'] in r['acceptable']); lats.append(r['latency']); ks.append(t['k_rel'])
            es, ok, lats = np.array(es), np.array(ok), np.array(lats)
            if len(es):
                bins = [0, 0.1, 0.3, 0.6, 1.01]
                cal = []
                for a, b in zip(bins[:-1], bins[1:]):
                    m = (es >= a) & (es < b)
                    if m.sum():
                        cal.append(f"e[{a},{b}) n={m.sum()} acc={ok[m].mean():.2f} lat={lats[m].mean():.1f}")
                print("   calibration (release confidence):", " | ".join(cal))
                lb = [(6, 7), (7, 9), (9, 13), (13, 61)]
                print("   accuracy by latency:", " | ".join(f"lat[{a},{b}) n={((lats >= a) & (lats < b)).sum()} "
                                                       f"acc={ok[(lats >= a) & (lats < b)].mean():.2f}"
                                                       for a, b in lb if ((lats >= a) & (lats < b)).sum()))
        rc = [r for r in L if r['stage'] == st and r['kind'] == 'recast']
        stm = [r for r in L if r['stage'] == st and r['kind'] == 'statement']
        ex = [r for r in L if r['stage'] == st]
        print(f"   turns: spoke after recast {np.mean([bool(r.get('after_recast_reply')) for r in rc]) if rc else None}, "
              f"after statement {np.mean([bool(r['reply']) for r in stm]) if stm else None}, overlap ticks "
              f"{sum(r.get('mouth_while_partner', 0) for r in ex)}, extra turns {sum(r.get('extra_turns', 0) for r in fr)}, "
              f"cut {sum(r.get('stop') == 'cut' for r in fr)}, max reply len {max(len(r['reply']) for r in fr)}")
    if TH:
        c = collections.Counter((t['kind'], t['outcome']) for t in TH)
        print("thought outcomes:", dict(c))
        sp = [t for t in TH if t['kind'] == 'reply' and t['spoke']]
        print("utterances per thought <= 1 by construction; reply thoughts that spoke:", len(sp))


if __name__ == '__main__':
    for p in sys.argv[1:]:
        report(p)
