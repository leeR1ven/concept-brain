# summary of the module tests (out/gen_probe_out.jsonl, out/gen_probe2_out.jsonl) by layer variant
import json, os, collections
HERE = os.path.dirname(os.path.abspath(__file__))
def lab(r):
    rm = r.get('rm')
    if rm is None:
        v = 'MIX (dict)'
    else:
        v = 'RL ' + (('(c0) train+prune, no regrowth, birth 1.5x' if rm.get('regrow', 1) == 0 else '(c) train+prune+regrow')
                     if rm.get('train') and rm.get('prune') else '(c\') prune+regrow only' if rm.get('prune')
                     else '(b) train' if rm.get('train') else '(a) fixed')
        if rm.get('scale'): v += f" scale {rm['scale']}"
        if rm.get('alloc') == 'flat': v += ' flat'
    return v
for fn in ('gen_probe_out.jsonl', 'gen_probe2_out.jsonl'):
    G = collections.defaultdict(list)
    for l in open(os.path.join(HERE, 'out', fn)):
        r = json.loads(l)
        key = (lab(r), r['variant'], r.get('heldstream'), r.get('tutor', 'recast'))
        G[key].append((r['seed'], r['frozen']['seen'], r['frozen']['held'], r['ALL'].get('held') if isinstance(r.get('ALL'), dict) else None, r['secs']))
    print('##', fn)
    print('| layer | variant | held-out in life | tutor | seeds | frozen seen | frozen held-out (per seed) | live held-out | secs |')
    print('|---|---|---|---|---|---|---|---|---|')
    for k, v in sorted(G.items()):
        v = sorted(v)
        hs = ' / '.join(f'{x[2]:.2f}' for x in v)
        live = [x[3] for x in v if x[3] is not None]
        print(f"| {k[0]} | {k[1]} | {'asked' if k[2] else 'never'} | {k[3]} | {','.join(str(x[0]) for x in v)} | "
              f"{sum(x[1] for x in v) / len(v):.2f} | {hs} | {(sum(live) / len(live)) if live else float('nan'):.2f} | {int(sum(x[4] for x in v) / len(v))} |")
