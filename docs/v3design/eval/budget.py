"""Compute budget of the minimal sealed-test version (A800 CPU, numpy, 4 processes x 5400 s per job).
Inputs (CITED v2.2 interim): a 10^6-tick life with the random layer ~1000 s on the A800 (out/life/L_cf_s5.json 1000 s);
a staged-protocol run ~2633 s (cf).  Speech-dense probe ticks cost like staged ticks: 2633 s / ~450k ticks = 5.9 ms."""
life1 = 1000.0                       # s, s=1, L=1e6 ticks
tick_speech = 2633.0 / 450e3         # s per speech-dense tick
fam, probes, ticks_per_probe = 6, 200, 90   # per family per life: 50 + 50 + 100 held-out probes at 3 checkpoints
probe1 = fam * probes * ticks_per_probe * tick_speech
base = life1 + probe1
JOB = 4 * 5400
print(f'probe cost per life {probe1:.0f} s; base life+probes {base:.0f} s')
def cost(s, L): return s * (L * life1 + probe1)       # probes do not grow with L (same number of checkpoints)
rows = []
rows.append(('S0 validation: trivial refs, knn, RNN (CPU)', 0.05 * JOB))
rows.append(('jitter test on v2.2 (staged, 3 seeds x 3 arms)', 3 * 3 * 2633))
rows.append(('opening main: v3 + v2.2, 5 seeds, lives A and B at (1,1,1)', 2 * 5 * 2 * cost(1, 1)))
rows.append(('repair control twins: 2 agents x 5 seeds', 2 * 5 * cost(1, 1)))
grid = sum(cost(s, L) for s in (0.5, 1, 2) for L in (0.5, 1, 2)) - cost(1, 1)
rows.append(('scaling 3x3 (s x L) W=1, v3, life B, 3 seeds (minus the main cell)', 3 * grid))
rows.append(('scaling W=2 at (1,1),(1,2),(2,2), v3, 3 seeds', 3 * (cost(1, 1) + cost(1, 2) + cost(2, 2))))
rows.append(('ablations: 6 core mechanisms x 3 seeds, life B, (1,1,1)', 6 * 3 * cost(1, 1)))
rows.append(('coverage: staged J2 5 seeds + life s5/s6 + world2 sigma 1, v3 and v2.2', 2 * (5 * 2633 + 2 * life1 + 1500)))
tot = 0
for name, sec in rows:
    tot += sec; print(f'{name:75s} {sec:8.0f} s  {sec / JOB:4.1f} jobs')
print(f'{"TOTAL":75s} {tot:8.0f} s  {tot / JOB:4.1f} jobs  (~{tot / JOB / 2 * 1.5:.0f} h wall at 2 job slots)')
print(f'longest single life: s=2, L=2: {cost(2, 2):.0f} s -> needs checkpoint/resume (limit 5400 s)')
