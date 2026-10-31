# -*- coding: utf-8 -*-
"""print markdown tables from the job outputs (local, tiny)."""
import json, sys, os, glob

ROOT = sys.argv[1] if len(sys.argv) > 1 else '.'


def rows(pat):
    out = []
    for fn in sorted(glob.glob(os.path.join(ROOT, pat), recursive=True)):
        for l in open(fn, encoding='utf-8'):
            l = l.strip()
            if l.startswith('{'):
                try:
                    out.append(json.loads(l))
                except Exception:
                    pass
    return out


def md(head, body):
    print('| ' + ' | '.join(head) + ' |')
    print('|' + '---|' * len(head))
    for b in body:
        print('| ' + ' | '.join(str(x) for x in b) + ' |')
    print()


dev = [r for r in rows('**/r3_dev.jsonl') if r.get('mode') == 'dev' and 'cov' in r]
if dev:
    print('### R3 dev grid (single write, 40 dev facts)')
    md(['layer', 'site', 'value', 'g/s', 'cov', 'rewrite', 'para', 'qa', 'neigh same', 'bleed'],
       [[r['layer'], r['site'], r['value'], r['g'] or r['s'], r['cov'], r['rewrite'], r['para'], r['qa'], r['neigh_same'],
         r['neigh_bleed']] for r in dev])
seq = [r for r in rows('**/r3_seq.jsonl') + rows('**/r3_ft.jsonl') if r.get('mode') in ('seq', 'ft') and 'n' in r]
if seq:
    print('### R3 sequential writes / FT')
    md(['run', 'n', 'new rewrite', 'new para', 'new qa', 'first100', 'all', 'neigh same', 'bleed', 'ppl', 'acc', 'know', 'cost'],
       [[(r.get('tag') or '') + ' ' + (f"L{r['layer']} {r.get('site', '')} {r.get('value', '')} {r.get('cov', '')}" if r['mode'] == 'seq' else f"{r['params']} lr{r['lr']}"),
         r['n'], r.get('new_rewrite'), r.get('new_para'), r.get('new_qa'), r.get('first100'), r.get('all'), r['neigh_same'],
         r['neigh_bleed'], r['ppl'], r['acc'], r['know'],
         r.get('ms_per_fact_write', r.get('ms_per_fact', ''))] for r in seq])
cos = [r for r in rows('**/r2_cos.jsonl') if r.get('mode') == 'cos' and 'method' in r]
if cos:
    print('### R2 cosine to the true gradient')
    md(['method', 'cos block (mean)', 'cos W_down (mean)', 'layers 0/6/12/18/23 (W_down)', 'norm ratio (mean)', 's/line'],
       [[r['method'], r.get('cos_block_mean', ''), r['cos_down_mean'],
         '/'.join(str(r['cos_down'][i]) for i in (0, 6, 12, 18, 23)),
         round(sum(r['norm_ratio']) / len(r['norm_ratio']), 3) if r.get('norm_ratio') else '', r['secs_per_line']] for r in cos])
on = [r for r in rows('**/r2_online.jsonl') if r.get('mode') == 'online' and 'tag' in r]
if on:
    print('### R2 online stream')
    md(['method', 'lr', 'stage', 'cloze', 'qa', 'ppl', 'acc', 'know', 'ms/update'],
       [[r['method'], r['lr'], r['tag'], r['cloze'], r['qa'], r['ppl'], r['acc'], r['know'], r.get('ms_per_update', '')] for r in on])
r1 = rows('**/r1.jsonl')
if r1:
    print('### R1')
    for r in r1:
        if r.get('kind') in ('recall', 'stream', 'heldout'):
            print(json.dumps(r, ensure_ascii=False))
