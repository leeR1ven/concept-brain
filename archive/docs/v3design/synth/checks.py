"""Arithmetic behind ARCH_v3_core's tick-by-tick walk and cost estimates (PROVED arithmetic; inputs labelled).
Run: python3 -I checks.py"""
import math

print('== (a) delta rule: p after n consistent exposures, eta 0.35 (bench R2)')
for n in (1, 2, 3, 5, 8):
    print(f'   n={n}: p = 1 - 0.65^n = {1 - 0.65 ** n:.3f}')

print('== accumulator I += G*e + u (bound 1), u = 0.15')
for G, e in ((1.0, 0.6), (0.8, 0.6), (1.0, 0.0), (0.3, 0.3)):
    step = G * e + 0.15
    print(f'   G={G}, e={e}: step {step:.2f}/tick -> release after {math.ceil(1 / step)} ticks')

print('== soft log time bins [2^k, 2^(k+2)): which bins each delay activates')
bins = lambda t: [k for k in range(9) if 2 ** k <= t < 2 ** (k + 2)]
for t in (1, 3, 5, 6, 8, 10, 12, 16, 30, 33, 45, 60):
    print(f'   tau={t:3d}: bins {bins(t)}')

print('== (b) restoration s = m + w*pi; accept if s >= 0.75; w = w_max*(1 - pi_ear)')
for wmax, pear in ((0.15, 1.0), (0.15, 0.33), (0.15, 0.0)):
    print(f'   w_max {wmax}, pi_ear {pear}: w = {wmax * (1 - pear):.3f}')
w = 0.10
for m, pi in ((0.68, 0.85), (0.66, 0.85), (0.70, 0.6), (0.62, 0.1), (0.70, 0.30)):
    s = m + w * pi
    print(f'   w=0.10 m={m} pi={pi}: s={s:.3f} -> {"accepted" if s >= 0.75 else "not accepted"}')
print(f'   clean window vs another lexicon char: 0.96 - 0.77 = {0.96 - 0.77:.2f} > w_max 0.15 -> never overridden')

print('== eligibility decay e^(-dt/tau_e), tau_e = 20 ticks')
for dt in (5, 11, 16, 25, 32, 45):
    print(f'   dt={dt}: {math.exp(-dt / 20):.2f}')

print('== learned NoGo after late-start overlaps: step = eta_v * |delta| * e, eta_v 0.2, |delta| 0.5, e 0.5')
step = 0.2 * 0.5 * 0.5
print(f'   step {step:.3f} per event -> {math.ceil(0.5 / step)} events to move Go-NoGo by 0.5')

print('== pure-neuron build, s = 1 (ASSUMED sizes; synapses = units x fan-in)')
areas = [  # name, units, fan-in per unit, active units per tick (speech tick)
    ('EAR0 (fixed local 3-D)', 1920, 4, 768), ('SYL assemblies (k=16)', 2048, 64, 16), ('VOICE', 256, 64, 8),
    ('EYE (adapting)', 1024, 1, 40), ('VR eye->word', 2048, 32, 16), ('MEAN', 512, 256, 16),
    ('LATEST (item x age x source)', 16384, 3, 200), ('THOUGHT (persistent)', 4096, 64, 192),
    ('TIME + P4 detectors', 64, 8, 6), ('MIX conjunctions', 48000, 9, 480), ('GEN out (fan-out 64 from MIX)', 2049, 1500, 16),
    ('M4 gates + critic (event-driven: 480 active inputs)', 11, 48000, 0.11), ('motor programs + forward/inverse', 2048, 128, 4),
    ('DG', 6000, 12, 120), ('CA3 (fixed fan-in 1000)', 10000, 1000, 100), ('CA1 -> cortex', 10000, 64, 100),
]
U = sum(a[1] for a in areas); S = sum(a[1] * a[2] for a in areas); lit = sum(a[3] * a[2] for a in areas)
for a in areas:
    print(f'   {a[0]:34s} units {a[1]:6d}  synapses {a[1] * a[2] / 1e6:6.2f} M')
print(f'   total units {U / 1e3:.0f} k, synapses {S / 1e6:.1f} M, lit synaptic events per speech tick ~{lit / 1e6:.2f} M')
mem = S * 8 / 1e6
print(f'   memory (4 B weight + 4 B index per synapse) ~{mem:.0f} MB at s=1, ~{4 * mem:.0f} MB at s=4')
# CPU time: speech ticks 15 % of a 1e6-tick life at ~lit events / 75 M events/s; silent ticks ~0.2 ms
for s in (1, 4):
    speech = 0.15e6 * (s * lit) / 75e6
    silent = 0.85e6 * 0.2e-3 * s
    print(f'   CPU numpy, s={s}: ~{speech + silent:,.0f} s per 1e6-tick life (speech {speech:,.0f} + silent {silent:,.0f})')
print('   born-wired calibration: 750k cells / 2M synapses at 13.3 ms per tick, unbatched consumer GPU ->'
      f' 1e6 ticks = {13.3e-3 * 1e6:,.0f} s if a v3 agent cost the same per tick')

print('== lineage loop on the A800 CPU runner (4 processes x 90 min per job)')
life = 1000 + 630           # s: 1e6-tick life (v2.2 L_cf ~1,000 s, CITED) + 6 S0 probe families (eval budget.py, CITED)
per_job = 4 * (5400 // life)
print(f'   one life + probes {life} s -> {per_job} lives per job; P=12 per generation -> 1 job/generation')
print(f'   10 generations x 2 lineages = 20 jobs ~ {10 * 90 / 60:.0f} h wall at 2 slots')
