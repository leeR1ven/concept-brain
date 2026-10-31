# writes the life reference table into README.md from results_life/*.json
import glob, json, os
R = {os.path.basename(f)[:-8]: json.load(open(f, encoding='utf-8')) for f in glob.glob('results_life/*_s0.json')}
f = lambda x: '-' if x is None else (f'{x:.2f}' if isinstance(x, float) else str(x))   # noqa: E731
lines = ['', '## Life reference numbers (seed 0, one life of ~10^6 ticks ≈ 21 h, tutor=model unless noted)', '',
         '| agent | ticks | free acc | scene (new) | held-out | memory | social | initiatives /1000 quiet | appropriate | init. reward | overlap ticks | interruptions | longest monologue | µs/tick |',
         '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
order = ['silent', 'namer', 'echo', 'lookup', 'oracle', 'refv1_model', 'refv1_force']
for nm in order:
    r = R.get(nm)
    if r is None:
        continue
    g, i = r['groups'], r['initiative']
    lines.append(f"| {nm} | {r['ticks']} | {f(r['tasks'].get('ALL', {}).get('acc'))} | {f(g['scene']['acc'])} | "
                 f"{f(g['held_out']['acc'])} | {f(g['memory']['acc'])} | {f(g['social']['acc'])} | "
                 f"{i['per_1000_quiet_ticks']:.2f} | {f(i['appropriate_share'])} | {i['reward']:.0f} | "
                 f"{r['overlap']['overlap_ticks']} | {r['overlap']['interruptions']} | {r['monologue']['longest_syllables']} | {r['us_per_tick']} |")
fo = R.get('refv1_force')
if fo:
    lines += ['', 'RefV1 (force) per task over life (acc; learning curve = 10 time bins):', '',
              '| task | n | acc | curve |', '|---|---|---|---|']
    for t, c in sorted(fo['curves'].items()):
        n = sum(x['n'] for x in c)
        acc = sum((x['acc'] or 0) * x['n'] for x in c) / max(n, 1)
        lines.append(f"| {t} | {n} | {acc:.2f} | {' '.join('-' if x['acc'] is None else f'{x['acc']:.2f}' for x in c)} |")
    lines += ['', 'RefV1 (force) retention by encounters (memory tasks): ' + '; '.join(
        f"{t}: " + ', '.join(f"{x['encounters']}x {x['acc']:.2f} (n={x['n']})" for x in v['by_encounters'])
        for t, v in fo['retention'].items() if t != 'qa_by_encounters' and v['n'])]
    lines += ['', f"RefV1 (force) stability flags: {fo['stability_flags'] or 'none'}; latency median {fo['latency']['median']}; programs {fo.get('programs')}"]
s = open('README.md').read()
if '\n## Life reference numbers' in s:
    s = s[:s.index('\n## Life reference numbers')]
open('README.md', 'w').write(s.rstrip('\n') + '\n' + '\n'.join(lines) + '\n')
print('\n'.join(lines))
