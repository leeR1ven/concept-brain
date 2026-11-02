"""Table of the scaling curve from run_scale result JSONs.
usage: python3 agg_scale.py results/curve1/x11_s1_seed0.json [results/curve1/x11_s1_seed1.json ...]"""
import json
import sys


def f(x, n=3):
    return '-' if x is None else f'{x:.{n}f}'


for path in sys.argv[1:]:
    r = json.load(open(path))
    print(f"== {r['run']}  n1 {r['n1']}  candidates {r['n_candidates']}  jobs {len(r['jobs'])}")
    for j in r['jobs']:
        print('   job', {k: (round(v, 2) if isinstance(v, float) else v) for k, v in j.items()})
    print('dlg    | PFC t1  t5  tl1  tl5 empty lit | late t1 t5 | hip t1 t5 lit | J     | uni t1 t5 | bi1 t1 t5 | tri1 t1 t5'
          ' | bi2 t1 t5 | tri2 t1 t5 | codes(interp) t1 | oracle t1')
    for cp in sorted(r['checkpoints'], key=int):
        c = r['checkpoints'][cp]
        p, la, h, o = c['pfc'], c.get('pfc_late', {}), c.get('hip') or {}, c['heard_oracle']
        ng = c.get('ngram') or {}

        def g(k):
            return ng.get(k, {})
        print(f"{int(cp):6d} | {f(p['top1'])} {f(p['top5'])} {f(p['toneless_top1'])} {f(p['toneless_top5'])} "
              f"{f(p['empty'], 2)} {f(p['mean_lit_per_window'], 0)} | {f(la.get('top1'))} {f(la.get('top5'))} | "
              f"{f(h.get('top1'))} {f(h.get('top5'))} {f(h.get('mean_lit_per_window'), 0)} | {f(c['tick_J_pfc'], 4)} | "
              f"{f(g('unigram').get('top1'))} {f(g('unigram').get('top5'))} | {f(g('bigram').get('top1'))} "
              f"{f(g('bigram').get('top5'))} | {f(g('trigram').get('top1'))} {f(g('trigram').get('top5'))} | "
              f"{f(g('bigram_lag2').get('top1'))} {f(g('bigram_lag2').get('top5'))} | {f(g('trigram_lag2').get('top1'))} "
              f"{f(g('trigram_lag2').get('top5'))} | {f(g('interp_via_codes').get('top1'))} | {f(o['top1'])}")
    print('response (first syllable of a reply turn): dlg | PFC t1 t5 | late t1 t5 | hip t1 t5 | uni | reply-uni | bi1 | bi2')
    for cp in sorted(r['checkpoints'], key=int):
        c = r['checkpoints'][cp]
        rb, rn = c.get('response', {}), (c.get('ngram') or {}).get('response', {})

        def q(d, k):
            x = d.get(k) or {}
            return f"{f(x.get('top1'))} {f(x.get('top5'))}"
        print(f"   {int(cp):6d} n {rb.get('n')} | {q(rb, 'pfc')} | {q(rb, 'pfc_late')} | {q(rb, 'hip')} | "
              f"{q(rn, 'unigram')} | {q(rn, 'unigram_reply_initial')} | {q(rn, 'bigram')} | {q(rn, 'bigram_lag2')}")
    print('activity (eval, speech ticks): dlg | L1ext L1int L1 L2 L3 L4 | H1int Hs | stop ; windows ok share since last')
    for cp in sorted(r['checkpoints'], key=int):
        c = r['checkpoints'][cp]
        k = c['counts']

        def s(n):
            return f"{k[n]['speech']:.0f}" if n in k else '-'
        a = c.get('activity', {})
        print(f"   {int(cp):6d} | {s('L1ext')} {s('L1int')} {s('L1')} {s('L2')} {s('L3')} {s('L4')} | {s('H1int')} "
              f"{s('Hs')} | {s('stop')} ; {f(a.get('share_ok'), 2)} over {len(a.get('windows_since_last', []))}")
    ws = r['windows']
    if ws:
        print('training windows: first / last:', {k: round(ws[0][k], 2) for k in ('L1', 'L1int', 'L4', 'Hs', 'max_frac')},
              {k: round(ws[-1][k], 2) for k in ('L1', 'L1int', 'L4', 'Hs', 'max_frac')},
              'ok share', round(sum(w['ok'] for w in ws) / len(ws), 3), 'n', len(ws))
