"""write the 5-seed job lists for the main batches from chosen configs.
  python3 mkbatch.py <out.txt> <seeds e.g. 0-4> <name>=<arm>[:key=val,key=val...] ...
each life: X1d run2 protocol (birth 200, teach 600, correction 300, untutored 1,000) unless overridden."""
import sys
out, seeds = sys.argv[1], sys.argv[2]
a, b = seeds.split('-')
lines = []
for spec in sys.argv[3:]:
    name, rest = spec.split('=', 1)
    arm, _, kv = rest.partition(':')
    extra = ' '.join(kv.split(',')) if kv else ''
    for s in range(int(a), int(b) + 1):
        lines.append(f'{arm} {s} {extra} tag=_{name}'.replace('  ', ' '))
open(out, 'w').write('\n'.join(lines) + '\n')
print(len(lines), 'lines ->', out)
