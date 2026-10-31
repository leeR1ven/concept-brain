# lesion effects as probe-pooled accuracy (every probe counts once; tasks with 2 probes no longer swing the mean)
#   python3 -B lesion_pooled.py runs/final_s0.json runs/final_s1.json runs/final_s2.json [--vs runs/big_s0.json ...]
import sys, json, numpy as np
SKIP = ('ALL', 'SCENE', 'NOSCENE', 'mech', 'after_model', 'learning', 'one_shot')
args = sys.argv[1:]; vs = []
if '--vs' in args:
    i = args.index('--vs'); vs = args[i + 1:]; args = args[:i]
Rs = [json.load(open(f)) for f in args]; Vs = [json.load(open(f)) for f in vs]


def pooled(d, stages='BCD'):
    hit = n = 0
    for k, v in d.items():
        if isinstance(v, dict) and v.get('acc') is not None and k.count('/') == 1 and not k.endswith(SKIP) and k[0] in stages:
            hit += v['acc'] * v['n']; n += v['n']
    return hit / n if n else None


def ms(xs):
    xs = [x for x in xs if x is not None]
    return '–' if not xs else (f"{np.mean(xs):+.3f}±{np.std(xs, ddof=1):.3f}" if len(xs) > 1 else f"{xs[0]:+.3f}")


les = list(Rs[0]['lesion'].keys())
print('| lesion | pooled acc per seed | drop per seed | drop mean±sd |' + ''.join(f" {V.get('tag')} acc (drop) |" for V in Vs))
print('|' + '---|' * (4 + len(Vs)))
for L in les:
    acc = [pooled(R['lesion'][L]) if L in R['lesion'] else None for R in Rs]
    base = [pooled(R['lesion']['none']) for R in Rs]
    drops = [b - a for a, b in zip(acc, base) if a is not None]
    line = f"| {L} | {' / '.join('–' if a is None else f'{a:.3f}' for a in acc)} | {' / '.join(f'{x:+.3f}' for x in drops)} | {ms(drops)} |"
    for V in Vs:
        if L in V['lesion']:
            a = pooled(V['lesion'][L]); b = pooled(V['lesion']['none'])
            line += f" {a:.3f} ({b - a:+.3f}) |"
        else:
            line += ' – |'
    print(line)
print('probes per lesion run:', [sum(v['n'] for k, v in R['lesion']['none'].items() if isinstance(v, dict) and v.get('acc') is not None
                                     and k.count('/') == 1 and not k.endswith(SKIP)) for R in Rs + Vs])
