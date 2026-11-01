"""Skeptic arithmetic (PROVED arithmetic; inputs quoted from ARCH_v3_core and the repo).  Run: python3 -I arith.py"""
import math

print('== P5b "being responded to": my efference trace at the partner\'s next onset, untutored probe timing')
print('   (next probe at max(Q_end + 45, stop + 13), world.py; tau_e = 20, G14)')
for stop in (8, 12, 16, 20, 24, 28, 32, 36, 40):
    gap = max(45 - stop, 13)
    print(f'   my stop at Q_end+{stop:2d}: next onset {gap:2d} ticks later, trace {math.exp(-gap / 20):.2f}')

print('== M2 read literally: prediction synapse with an eligibility trace (tau_e 20) and m = 1 every tick')
lam = math.exp(-1 / 20)
print(f'   effective rate per event = eta / (1 - lambda) = 0.35 / {1 - lam:.4f} = {0.35 / (1 - lam):.1f}'
      '  (> 2: the normalised delta rule overshoots and diverges)')

print('== M2 eta * max(1/n_use, floor) vs the walk\'s constant eta 0.35 (p after n consistent exposures)')
for floor in (None, 0.1, 0.02):
    p = 0.0; out = []
    for k in range(1, 11):
        eta = 0.35 if floor is None else 0.35 * max(1 / k, floor)
        p += eta * (1 - p)
        if k in (1, 3, 5, 10):
            out.append(f'n={k}: {p:.2f}')
    print(f'   floor {floor}: ' + ', '.join(out))
print('   synapse with a long use history (n_use >> 1/floor), floor 0.1: p after 5 = '
      f'{1 - (1 - 0.035) ** 5:.2f}')

print('== learned NoGo after late-start overlaps (§6.4: eta_v 0.3, |delta| 0.5, trace 0.5) to move it by 0.5')
step = 0.3 * 0.5 * 0.5
print(f'   constant eta: {math.ceil(0.5 / step)} events (doc: "about 7"; checks.py uses eta_v 0.2 -> 10)')
for floor in (0.1, 0.02):
    s, k = 0.0, 0
    while s < 0.5 and k < 100000:
        k += 1; s += step * max(1 / k, floor)
    print(f'   eta * max(1/n_use, {floor}): {k} events')

print('== compute: §7.5 s = 4 column is a 1e6-tick life, but §4 sets L proportional to s')
for lo, hi in ((2000, 6000),):
    print(f'   doc numbers at s = 4, L = 4e6: {4 * lo:,}-{4 * hi:,} s = {4 * lo / 5400:.1f}-{4 * hi / 5400:.1f} 90-min jobs per life')
print('   gpuport (v2/gpuport/report.md, A800, B = 64): 259 ms per batched tick -> a 1e6-tick generation of 64 = '
      f'{259e-3 * 1e6 / 3600:.0f} h; doc assumes 0.3-0.6 ms per batched tick ({259 / 0.6:.0f}-{259 / 0.3:.0f}x faster)')

print('== lineage loop: one generation of 12 on the A800 CPU runner (4 processes x 5,400 s per job)')
for life in (1630, 1630 + 630, 2 * 1630):
    per_proc = 5400 // life
    jobs = math.ceil(12 / (4 * per_proc)) if per_proc else float('inf')
    print(f'   life + probes {life} s: {per_proc} per process, {jobs} job(s) per generation, '
          f'{10 * jobs * 1.5:.0f} h wall for 10 generations x 2 lineages at 2 slots')

print('== drift to 10,000 untutored on v2.2-speed code (finF s0 log: staged ~2,220 s, 300->3,000 untutored 1,487 s)')
per = 1487 / 2700
print(f'   {per:.2f} s per untutored dialogue -> staged + 10,000 = {2220 + 10000 * per:,.0f} s > 5,400 s job limit')

print('== maturity: exploration and babble after M (the doc\'s own numbers)')
print('   §6.3: eps 0.05 x ~1,600 object events = 80 exploratory utterances per life; post-M share 2/3 -> ~53')
print(f'   P1d babble r_bab 0.01 per idle tick: 0.85 x 1e6 x 2/3 x 0.01 = {0.85e6 * 2 / 3 * 0.01:,.0f} post-M babble onsets')
