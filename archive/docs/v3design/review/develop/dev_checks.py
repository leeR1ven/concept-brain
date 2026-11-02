"""Arithmetic behind the developmental review of ARCH_v3_core (PROVED arithmetic; inputs labelled CITED/ASSUMED)."""
import math
print('== values: harm (w_upset = 1.5 w_plea) vs help when the harm is perceived k ticks after the praise')
for g in (0.8, 0.95, 0.99):
    k = math.log(1.5) / -math.log(g)
    print(f'   gamma {g}: help wins once harm lags by > {k:.1f} ticks ({k * 0.075:.2f} s at 75 ms/tick)')
print('== values: praise k ticks after the decision, discounted gamma^k; eligibility exp(-k/20)')
for k in (20, 50, 100, 200, 400):
    print(f'   k={k:3d}: 0.95^k={0.95 ** k:.2e}  0.99^k={0.99 ** k:.2e}  trace={math.exp(-k / 20):.2e}')
print('== post-maturity babble hazard (P1d: r_bab 0.01 per idle tick, constant gene; babble noise below M4)')
r = 0.01
probe_share = 0.15 + 0.15 * 0.85          # CITED bench: probe_frac 0.15 of seen items + held_frac 0.15 always probes
win_probe, win_react = 30, 1              # ASSUMED idle ticks >= 5 after my stop before the next line (probe / reaction)
h = probe_share * (1 - (1 - r) ** win_probe) + (1 - probe_share) * (1 - (1 - r) ** win_react)
n_ans = 1776 * (2 / 3) * 0.8              # CITED 1,776 questions per life; 2/3 of life post-M; ASSUMED answer rate 0.8
print(f'   extra-turn hazard per answered exchange {h:.3f}; post-M answers {n_ans:.0f}; expected extra turns {h * n_ans:.0f};'
      f' P(0) = {math.exp(-h * n_ans):.1e}')
pre = 1 - (1 - r) ** 5
print(f'   babble inside the first 5 ticks of an answer window (becomes the reply): {pre:.3f} per exchange')
print('== maturity at 5,000 heard syllables')
for name, syl_day, q_day in (('bench life (per 1e6 ticks)', 15457, 1776), ('world2 4 agents', 960, 32.2), ('world2 16 agents', 560, 10.4)):
    t = 5000 / syl_day
    print(f'   {name:28s}: M after {t:5.2f} units, questions to the agent before M ~{t * q_day:4.0f}')
print(f'   child (ASSUMED 18,000 syllables/day): M after {5000 / 18000:.2f} child-days')
print('== experience per life and lineage (ASSUMED 18,000 heard syllables per child-day)')
life = 15457 / 18000
print(f'   one bench life = {life:.2f} child-days; 10 generations in sequence = {10 * life:.1f} child-days;'
      f' all 240 lives = {240 * life:.0f} child-days (mostly parallel copies)')
print(f'   child at 12 months ~{6.6e6 / 15457:.0f} bench lives; at 24 months ~{13e6 / 15457:.0f}')
cpu_lineage = 240 * 1630 / 3600
print(f'   lineage CPU (240 lives x 1,630 s, CITED synth) = {cpu_lineage:.0f} CPU-h; one 1e8-tick life at 1,000 s per 1e6 ticks'
      f' = {1e8 / 1e6 * 1000 / 3600:.0f} CPU-h')
print('== greeting walk (6.3): delta rule over the 2 bin features of tau 30 (bins 3,4); accumulate (G+u)*e, G 0, e = p')
bins = lambda t: [k for k in range(9) if 2 ** k <= t < 2 ** (k + 2)]
for n in (1, 5, 10):
    p30 = 1 - 0.65 ** n; w = {3: p30 / 2, 4: p30 / 2}
    I = 0.0; rel = None
    for tau in range(1, 80):
        p = min(1.0, sum(w.get(k, 0.0) for k in bins(tau)))
        I += 0.15 * p
        if I >= 1:
            rel = tau; break
    for gw in (30, 22, 40):
        first = rel + 4
        print(f'   visits {n:2d}: p(tau30) {p30:.2f}; release at tau {rel}, first syllable ~{first} vs partner greets at {gw}:'
              f' {"agent first" if first < gw else "partner first"}')
