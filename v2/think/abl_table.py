# one row per short ablation run (A100 + B400, seed 0): B acc, silent, latency, overlaps, speech after recasts
import re, glob, json
rows = []
for f in sorted(glob.glob('out/abl_*.stdout')):
    s = open(f).read()
    name = f[len('out/abl_'):-len('.stdout')]
    m = re.search(r"\[B\] \[B\] free n=(\d+) acc ([\d.]+) \(held ([\d.]+).*?overlap (\d+) lat ([\d.]+).*?stop (\{[^}]*\})", s)
    t = re.search(r'"_turns": (\{[^}]*\})', s)
    tm = re.search(r'"ALL": \{"n": \d+, "spoke": ([\d.]+), "lat_med": ([\d.]+), "lat_mean": ([\d.]+)', s)
    if not m:
        rows.append((name, 'no result')); continue
    turns = json.loads(t.group(1)) if t else {}
    rows.append((name, m.group(2), m.group(3), m.group(4), tm.group(1) if tm else '-', tm.group(3) if tm else '-',
                 turns.get('after_recast_spoke'), turns.get('exchanges_with_overlap'), m.group(6)))
print('| ablation | B acc | held | overlap ticks | spoke | latency mean | spoke after recast | exchanges with overlap | stops |')
print('|---|---|---|---|---|---|---|---|---|')
for r in rows:
    print('| ' + ' | '.join(str(x) for x in r) + ' |')
