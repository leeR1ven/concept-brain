# Multi-seed / variant summary of pipeline JSONs.
#   python3 -B seeds_table.py runs/final_s0.json runs/final_s1.json runs/final_s2.json [--vs runs/big_s0.json ...]
# Prints per task: mean ± sd over the given runs (+ per-run values), held-out, first syllable; stage ALL; learning in
# use; listening prediction vs n-gram; mechanics; motor inventory; drift; lesion effects (B/C/D stage means, as the
# report's lesion table) with mean ± sd of the drop.  Runs after --vs are listed per run next to the seed mean.
import sys, json, numpy as np

SKIP = ('ALL', 'SCENE', 'NOSCENE', 'mech', 'after_model', 'learning', 'one_shot')
BAR = {'B/slot': (0.00, 0.07, 0.06), 'B/what': (0.06, 0.06, 0.04), 'B/where': (0.32, 0.31, 0.32),
       'C/colour': (0.08, 0.16, 0.17), 'C/count': (0.10, 0.31, 0.30), 'C/offer': (0.04, 0.07, 0.48),
       'C/ritual': (0.62, 0.20, 1.00), 'C/taste': (0.37, 1.00, 1.00), 'D/before': (0.01, 0.08, 0.08),
       'D/check': (0.46, 0.48, 0.31), 'D/name': (0.00, 0.12, 0.12), 'D/oneshot': (0.00, 0.00, 0.00),
       'B/ALL': (0.12, 0.14, 0.14), 'C/ALL': (0.19, 0.30, 0.52), 'D/ALL': (0.19, 0.22, 0.15)}

args = sys.argv[1:]
vs = []
if '--vs' in args:
    i = args.index('--vs'); vs = args[i + 1:]; args = args[:i]
Rs = [json.load(open(f)) for f in args]
Vs = [json.load(open(f)) for f in vs]


def ms(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return '–'
    return f"{np.mean(xs):.2f}±{np.std(xs, ddof=1):.2f}" if len(xs) > 1 else f"{xs[0]:.2f}"


def g(R, k, f='acc'):
    st = k.split('/')[0]
    v = R.get(st, {}).get(k)
    return v.get(f) if isinstance(v, dict) else None


keys = []
for st in 'BCD':
    ks = sorted({k for R in Rs for k, v in R.get(st, {}).items() if isinstance(v, dict) and v.get('acc') is not None
                 and not k.endswith(SKIP)})
    keys += ks + [f'{st}/ALL']
head = '| task | mean±sd (seeds) | per seed | held-out | first syl | bar / prior / n-gram |' + ''.join(
    f' {v.get("tag", "var")} |' for v in Vs)
print(head); print('|' + '---|' * (6 + len(Vs)))
for k in keys:
    accs = [g(R, k) for R in Rs]
    b = BAR.get(k)
    row = f"| {k} | {ms(accs)} | {' / '.join('–' if a is None else f'{a:.2f}' for a in accs)} | {ms([g(R, k, 'held') for R in Rs])} | " \
          f"{ms([g(R, k, 'first') for R in Rs])} | {' / '.join(f'{x:.2f}' for x in b) if b else '–'} |"
    for V in Vs:
        a = g(V, k); h = g(V, k, 'held'); fs = g(V, k, 'first')
        row += f" {'–' if a is None else f'{a:.2f}'}{'' if h is None else f' (h {h:.2f})'}{'' if fs is None else f' [1st {fs:.2f}]'} |"
    print(row)

print('\nlearning in use (first -> last third of the stage, all probes):')
for st in 'BCD':
    e = [R.get(st, {}).get(f'{st}/learning', {}) or {} for R in Rs]
    print(f"  {st}: early {ms([x.get('acc_early') for x in e])} -> late {ms([x.get('acc_late') for x in e])}",
          ' | '.join(f"{V.get('tag', 'var')}: {(V.get(st, {}).get(f'{st}/learning') or {}).get('acc_early')} -> "
                     f"{(V.get(st, {}).get(f'{st}/learning') or {}).get('acc_late')}" for V in Vs))
print('\nlistening next-syllable prediction (readout vs best online n-gram):')
for st in 'BCD':
    p = [R.get(st + '_predict') or {} for R in Rs]
    print(f"  {st}: readout {ms([x.get('readout') for x in p])}  n-gram {ms([max(x.get('n1', 0), x.get('n2', 0), x.get('n3', 0)) if x else None for x in p])}",
          ' | '.join(f"{V.get('tag', 'var')}: {(V.get(st + '_predict') or {}).get('readout')}" for V in Vs))
print('\nmechanics (per run: mouth-while-partner, overlaps, latency median, frac not 3 ticks, garble, extra turns):')
for R in Rs + Vs:
    out = []
    for st in 'BCD':
        m = R.get(st, {}).get(f'{st}/mech')
        if m:
            out.append(f"{st} {m.get('mouth_while_partner')}/{m.get('exchanges_with_overlap')}/{m.get('latency_median')}/"
                       f"{m.get('frac_not_3')}/{m.get('garble')}/{m.get('extra_turns')}")
    print(f"  {R.get('tag', 's%s' % R['seed'])}: " + '; '.join(out))
print('\nmotor inventory:')
for R in Rs + Vs:
    print(f"  {R.get('tag', 's%s' % R['seed'])}: {R.get('motor')}  stats kept {R.get('stats', {}).get('kept')} babble {R.get('stats', {}).get('babble')}")
print('\ndrift A -> end (Jaccard, 2 probe lines):')
for R in Rs + Vs:
    print(f"  {R.get('tag', 's%s' % R['seed'])}: {R.get('drift_A_to_end')}")


def lmean(d, st):
    accs = [v['acc'] for k, v in d.items() if k.startswith(st + '/') and isinstance(v, dict) and v.get('acc') is not None
            and not k.endswith(SKIP)]
    return np.mean(accs) if accs else None


def lrow(R, L):
    d = R.get('lesion', {}).get(L)
    if not d:
        return None
    row = [lmean(d, st) for st in 'BCD']
    allv = [x for x in row if x is not None]
    return row + [np.mean(allv) if allv else None]


print('\nlesions (frozen copy, same probe stream; B / C / D stage means and their mean; drop = none - lesion):')
les = list(Rs[0].get('lesion', {}).keys())
print('| lesion | B | C | D | mean | drop per seed | drop mean±sd |' + ''.join(f" {V.get('tag', 'var')} mean (drop) |" for V in Vs))
print('|' + '---|' * (7 + len(Vs)))
for L in les:
    rows = [lrow(R, L) for R in Rs]
    base = [lrow(R, 'none') for R in Rs]
    drops = [b[3] - r[3] for b, r in zip(base, rows) if r and b]
    line = f"| {L} | " + ' | '.join(ms([r[i] for r in rows if r]) for i in range(4)) + \
           f" | {' / '.join(f'{x:+.2f}' for x in drops)} | {ms(drops)} |"
    for V in Vs:
        r = lrow(V, L); b = lrow(V, 'none')
        line += f" {'–' if not r else f'{r[3]:.2f}'}{'' if not (r and b) else f' ({b[3] - r[3]:+.2f})'} |"
    print(line)
print('\nsecs per run:', [round(R.get('secs', 0)) for R in Rs + Vs])
