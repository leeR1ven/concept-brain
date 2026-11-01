# -*- coding: utf-8 -*-
"""x2 tables: C accuracy per drift mark (per arm and seed), failure shape, diagnostics, staged scores, relearning.
usage: python3 agg_x2.py <run json files ...>   (run_x2 staged records and drift_x2 fork records)"""
import json, sys, collections
import numpy as np


def load(paths):
    out = []
    for p in paths:
        for l in open(p):
            l = l.strip()
            if l:
                out.append((p, json.loads(l)))
    return out


def marks_of(r):
    if 'score' in r:                       # run_x2 staged record
        d = (r.get('drift') or {}).get('x2') or {}
        return d.get('marks', {}), d.get('dset', '')
    d = r.get('drift') or {}
    return d.get('marks', {}), r.get('dset', '')


def main():
    rows = load(sys.argv[1:])
    print('== C accuracy after N untutored dialogues (role at the last mark: ok / partner / silent / echo / other)')
    for p, r in rows:
        ms, dset = marks_of(r)
        if not ms:
            continue
        ks = sorted(ms, key=int)
        acc = ' '.join(f"{k}:{ms[k]['acc']:.3f}" for k in ks)
        last = ms[ks[-1]]['role']
        shape = '/'.join(f"{last.get(c, 0):.2f}" for c in ('ok', 'partner', 'silent', 'echo', 'other'))
        print(f"{r['tag']:>14} s{r['seed']} {acc} | {shape} | {dset}")
    print('\n== diagnostics per mark (seg online ok/partner/silent; hip live_old/live, pool_old/pool; recall src old/new;'
          ' lesions noN / nohip; deltas: recalls, rm_pruned, blocked_nogo, blocked_R1, released, ral cancel/commit/cut)')
    for p, r in rows:
        ms, dset = marks_of(r)
        for k in sorted(ms, key=int):
            m = ms[k]
            if 'hip' not in m:
                continue
            s, h, d = m['seg'], m['hip'], m['delta']
            rs = m.get('recall_src') or {}
            les = m.get('lesion') or {}
            ls = ' '.join(f"{n}:{v['acc']:.2f}(p{v['role'].get('partner', 0):.2f},s{v['role'].get('silent', 0):.2f})"
                          for n, v in les.items())
            print(f"{r['tag']:>14} s{r['seed']} {k:>5} seg {s.get('ok', 0):.2f}/{s.get('partner', 0):.2f}/{s.get('silent', 0):.2f}"
                  f" hip {h['live_old']}/{h['live']} pool {h['pool_old']}/{h['pool']} rsrc {rs.get('old', 0)}/{rs.get('new', 0)}"
                  f" (next {rs.get('old_next', 0)}/{rs.get('new_next', 0)}) | {ls} | rec {d.get('awake_recalls')} "
                  f"prn {d.get('rm_pruned')} nogo {d.get('t_blocked_nogo')} R1 {d.get('t_blocked_R1')} rel {d.get('t_released')} "
                  f"cc {d.get('ral_cancel')}/{d.get('ral_commit')}/{d.get('ral_cut')} | top {m['role'].get('top_wrong', [])[:3]}")
    print('\n== staged (run_x2 records): J J2 F RET H DR | held-out agent vs const | per-task held-out | relearn')
    for p, r in rows:
        if 'score' not in r:
            continue
        sc = r['score']
        ho = (r.get('refs') or {}).get('held') or {}
        pt = ' '.join(f"{t}:{v['held']}" for t, v in sorted(r.get('per_task', {}).items()) if v.get('held') is not None)
        rl = r.get('relearn') or {}
        rls = ' '.join(f"{x['item']}:{x['first']}/{x['stable_from']}" for x in rl.get('items', []))
        print(f"{r['tag']:>14} s{r['seed']} J {sc['J']:.3f} J2 {sc['J2']:.3f} F {sc['F']:.3f} RET {sc['RET']:.3f} H {sc['H']:.3f} "
              f"DR {sc['DR']:.3f} | ho {ho.get('agent')} vs {ho.get('const_best')} | {pt} | viol {r['viol']} | relearn {rls} "
              f"| secs {r.get('secs')} rss {r.get('peak_rss_mb')}")
    for p, r in rows:
        if 'score' in r or 'relearn' not in r:
            continue
        rl = r['relearn']
        print(f"{r['tag']:>14} s{r['seed']} relearn " + ' '.join(f"{x['item']}:{x['first']}/{x['stable_from']}" for x in rl['items']))


if __name__ == '__main__':
    main()
