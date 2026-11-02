# aggregate pipeline JSONs: python3 -B agg.py res_*.json
import sys, json, numpy as np

files = sys.argv[1:]
Rs = [json.load(open(f)) for f in files]
tasks = {}
for R in Rs:
    for st in ('B', 'C', 'D'):
        for k, v in R.get(st, {}).items():
            if isinstance(v, dict) and v.get('acc') is not None and k.count('/') == 1 and not k.endswith(('ALL', 'SCENE', 'NOSCENE', 'mech', 'after_model')):
                tasks.setdefault(k, []).append(v)
print("| task | acc (mean over seeds) | seen | held-out | first char | echo-free | parrot | n per seed |")
print("|---|---|---|---|---|---|---|---|")
f = lambda xs: "–" if not [x for x in xs if x is not None] else f"{np.mean([x for x in xs if x is not None]):.2f}"
for k in sorted(tasks):
    vs = tasks[k]
    per = ', '.join('%.2f' % v['acc'] for v in vs)
    print(f"| {k} | {f([v['acc'] for v in vs])} ({per}) | "
          f"{f([v.get('seen') for v in vs])} | {f([v.get('held') for v in vs])} | {f([v.get('first') for v in vs])} | "
          f"{f([v.get('echo_free') for v in vs])} | {f([v.get('parrot') for v in vs])} | {vs[0].get('n')} |")
# mechanics
for R in Rs:
    for st in ('B', 'C', 'D'):
        m = R.get(st, {}).get(f'{st}/mech')
        if m:
            print(f"mech seed {R['seed']} {st}: {m}")
# lesions: mean task accuracy per lesion, per stage
print("\n| lesion | " + " | ".join(f"seed {R['seed']}" for R in Rs) + " |")
print("|---|" + "---|" * len(Rs))
les = list(Rs[0].get('lesion', {}).keys())
for L in les:
    row = []
    for R in Rs:
        d = R['lesion'].get(L, {})
        accs = [v['acc'] for k, v in d.items() if isinstance(v, dict) and v.get('acc') is not None and k.count('/') == 1
                and not k.endswith(('ALL', 'SCENE', 'NOSCENE', 'mech', 'after_model', 'learning', 'one_shot'))]
        row.append(f"{np.mean(accs):.3f}" if accs else "–")
    print(f"| {L} | " + " | ".join(row) + " |")
# per-task lesion table (seed 0)
R = Rs[0]
keys = sorted(k for k, v in R['lesion'].get('none', {}).items() if isinstance(v, dict) and v.get('acc') is not None
              and k.count('/') == 1 and not k.endswith(('ALL', 'SCENE', 'NOSCENE', 'mech', 'after_model')))
print("\n| lesion (seed %d) | " % R['seed'] + " | ".join(keys) + " |")
print("|---|" + "---|" * len(keys))
for L in les:
    d = R['lesion'][L]
    print(f"| {L} | " + " | ".join(f"{d[k]['acc']:.2f}" if k in d and d[k].get('acc') is not None else "–" for k in keys) + " |")
for R in Rs:
    print("drift seed", R['seed'], R.get('drift'), 'A->end', R.get('drift_A_to_end'))
    print("examples", R['seed'], R.get('B_examples', [])[-6:], R.get('C_examples', [])[-6:], R.get('D_examples', [])[-6:])
