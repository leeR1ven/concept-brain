# writes the 3-seed bar table into README.md (section "Reference numbers, 3 seeds") from results_all/
import glob, json, os
import numpy as np
R = {os.path.basename(f)[:-5]: json.load(open(f, encoding='utf-8')) for f in glob.glob('results_all/*.json')}


def vals(ag, st, tu, *path):
    out = []
    for sd in (0, 1, 2):
        r = R.get(f'{ag}_{st}_{tu}_s{sd}')
        if r is None:
            continue
        x = r['res'].get(st, {})
        for p in path:
            x = x.get(p) if isinstance(x, dict) else None
        if x is not None:
            out.append(x)
    return out


def ms(v):
    return '-' if not v else (f'{np.mean(v):.2f}' + (f'±{np.std(v):.2f}' if len(v) > 1 else ''))


lines = ['', '## Reference numbers, 3 seeds (world seeds 0, 1, 2; 2,000 dialogues per stage; mean ± sd)', '',
         'Free-reply acceptable rate: ALL / held-out / scene questions. Bar = `refv1_mem` with `tutor=\'force\'`; baselines in `force`.', '',
         '| stage | bar (seeds) | echo | last | prior | lookup | ngram | oracle |', '|---|---|---|---|---|---|---|---|']
for st in 'BCD':
    cells = []
    for ag in ('refv1_mem', 'echo', 'last', 'prior', 'lookup', 'ngram', 'oracle'):
        a = vals(ag, st, 'force', 'ALL', 'acc'); h = vals(ag, st, 'force', 'ALL', 'held', 'acc'); s = vals(ag, st, 'force', 'SCENE', 'acc')
        cells.append(f'{ms(a)} / {ms(h)} / {ms(s)}' + (f' ({len(a)})' if ag == 'refv1_mem' else ''))
    lines.append(f'| {st} | ' + ' | '.join(cells) + ' |')
lines += ['', '| task | bar | prior | ngram |', '|---|---|---|---|']
for st in 'BCD':
    r0 = R[f'refv1_mem_{st}_force_s0']['res'][st]
    for t in sorted(k for k, v in r0.items() if isinstance(v, dict) and 'held' in v and k not in ('ALL', 'SCENE', 'NOSCENE')):
        lines.append(f"| {st}:{t} | {ms(vals('refv1_mem', st, 'force', t, 'acc'))} | {ms(vals('prior', st, 'force', t, 'acc'))} | "
                     f"{ms(vals('ngram', st, 'force', t, 'acc'))} |")
s = open('README.md').read()
if '\n## Reference numbers, 3 seeds' in s:
    s = s[:s.index('\n## Reference numbers, 3 seeds')]
open('README.md', 'w').write(s.rstrip('\n') + '\n' + '\n'.join(lines) + '\n')
print('\n'.join(lines))
