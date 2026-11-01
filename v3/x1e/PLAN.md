# X1e plan: does learning the PFC top layer's internal links (L3 -> L3) carve reproducible, question-specific paths?

Written 2026-11-01 before the pilot sweep and the 5-seed batches (one timing / code check life was run first).

## Authority
ARCHITECTURE.md (11-01): sec. 2.4 (PFC: "有关的互相激活", two-layer weights, moving thinking range), sec. 2.8 (one rule,
11-01 held-neuron form, concave growth, one global decay), sec. 3 (wrong rules), sec. 4 items 1, 6, 10, 11.

## What changes (core5.py, a subclass of X1d core4.Brain; defaults reproduce core4 exactly, checked 3,000 ticks)
- L3 -> L3 learned links `L33` learn by the same onset / held rule as top -> L1:
  B rises while A is on (new or held) or just fell -> A->B grows; co-rising -> both directions grow;
  A rises while B was already on -> A->B weakens; no change at a fall. Concave growth, 1 % floor, birth links B33 exempt
  from the one global decay (dec_33 every 100 ticks).
- Two-layer weights on L3 -> L3 in the ARCH form, P33 = g33 * W / (out^a * in^(1-a)), no total re-scaling.
- Nothing else. No new module, timer, refractoriness, fatigue, self-hold (kappa = 0), event boundary.

## Arms (bases are X1d's RNpm = main, R1e = secondary)
1. L33 learning on (eta33 > 0) for both bases; a short pilot (seed 0, teach 300) chooses g33 / f3 / a33 / eta33 /
   dec_33 (minor parameters) before the 5-seed runs.
2. L33 learning x ARCH scale x4 (n2 = n3 = 1,600).
3. Shaping caregiver (X1d ShapingWorld) with the best arm.
4. Hub check: the ARCH two-layer weights on top -> L1 and L3 -> L3 (in-strength percentiles, per-syllable in-strength).
Protocol: X1d run2 (birth 200, teach 600, correction 300, untutored 1,000), 5 seeds; one longer teach (1,500) for the
best arm. Heavy batches on the A800 CPU queue (`queue_a800/x1e_*`, at most 2 jobs at a time).

## Measures (learning off after training)
- Path (phase-1 metrics on the trained brain): same vs cross Jaccard by lag; question decoding at lags 0 / 6 / 12
  (chance 1/32); damage spreading (10 %); within-path J(t, t+1) (mean over lags 1-12).
- Q -> A prediction specificity (own vs other, max over lags) and hub share.
- Accuracy: seen pairs, first syllable, held-out (secondary; needs the concept area).
- Garble, talk-over, failure to yield.

## Pass criteria (fixed now)
- **Path:** question decoding at lag 12 >= 0.30 AND within-path J(t, t+1) < 0.80.
- **Specificity:** own - other >= 0.10.
- **Seen-pair first-syllable accuracy > 0.20.**
Each is judged on the 5-seed mean at the end of teaching (teach 600; also reported at 1,500 for the long run).
If an arm fails, report where the chain breaks (path -> prediction -> release), with numbers.

## Additions (coordinator messages during the screen, written before the 5-seed batches M1-M3)
- **Quiet wait** (X1d oracle finding): the motor area's default blocking must hold during the wait. Existing knobs only:
  `g_ret_mot` 1.0 -> 0.15, `c_mb` 1.2 -> 2.0, `theta_m` 0.22 -> 0.26 (n3 = 400; chosen by a probe-time screen,
  `motor_screen.py`); at x4 `theta_m` 0.35 (the x4 PFC drives the motor copies harder). Report response rate and garble.
- **Overhear-first** (designer 11-01, ARCH sec. 2.11): before teaching, the child overhears N adult dialogues
  (Q by one adult, A by the other after 3-5 ticks, or 10-14 ticks in the `long` variant), mouth blocked by `theta_m`;
  probes, path and specificity at N = 100 / 300 / 1,000 / 3,000; then the usual teaching protocol.
- Praise (shaping) only together with the quiet wait.
- x4 protocol shortened for compute: overhear 1,000, teach 600, no correction / untutored phase.
- Pass criteria unchanged.
