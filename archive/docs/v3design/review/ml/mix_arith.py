"""PROVED arithmetic: how many MIX units are eligible per event under ARCH_v3 sizes (checks.py) and the RandMix
branch rule (a branch fires iff >= 1 of its 3 lines is active; a unit is eligible iff all 3 branches fire).
Also: CA3 with a larger fan-in at s = 1 (the fix), via ca3_capacity.run."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
areas = {'LATEST': (16384, 200), 'THOUGHT': (4096, 192), 'EYE': (1024, 40), 'MEAN': (512, 16), 'P4': (64, 6),
         'CA1': (10000, 100)}
pb = {a: 1 - (1 - k / n) ** 3 for a, (n, k) in areas.items()}
print('P(branch fires), fan-in 3:', {a: round(v, 3) for a, v in pb.items()})
rand = list(areas)                       # random branches drawn from LATEST, THOUGHT, EYE, MEAN, P4, CA1
m = sum(pb[a] for a in rand) / len(rand)
m_nocA1 = sum(pb[a] for a in rand if a != 'CA1') / (len(rand) - 1)
for lat_active, label in ((200, '6-syllable line'), (32, '1-syllable event')):
    pa = 1 - (1 - lat_active / 16384) ** 3
    for mm, lab in ((m, 'recall on'), (m_nocA1, 'no recall')):
        print(f'  {label:17s} ({lab}): anchor {pa:.4f} x random {mm:.3f}^2 -> eligible MIX units = '
              f'{48000 * pa * mm * mm:.1f}  (k-WTA target 1 % = 480; v2.2 RandMix ~365 active)')
for f, lab in ((0.0122, 'LATEST'), (0.047, 'THOUGHT'), (0.039, 'EYE')):
    import math
    print(f'  fan-in for P(branch) = 0.4 on {lab}: {math.log(0.6) / math.log(1 - f):.0f} lines')
from ca3_capacity import run
Ms = [500, 1000, 2000, 4000, 8000, 16000]
for C in (2500, 5000):
    k, o = run(10000, C, 0.01, Ms, ntest=20)
    print(f'  CA3 N=10000 a=0.01 C={C} (aC={0.01*C:.0f}): F1 by M {o}', flush=True)
