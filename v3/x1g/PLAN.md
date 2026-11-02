# X1g plan: does a sparser, larger PFC let the one Hebbian rule learn question-specific Q -> A? (ARCH 4.20)

Written 2026-11-01 before the multi-seed batches. Seed-0 pilots (marked SIMULATED in the report) were used only to
check that the code runs, how fast it is, and to choose the form of the positive control (see "decisions from
pilots"). Authority: docs/ARCHITECTURE.md (11-01).

## Base (unchanged from X1f G0)
- Generation 0 = X1f G0: hand-wired speech basics A1 (sound -> muscle map) and A2 (syllable-shape blocking gate),
  quiet wait (c_mb 0, g_dir 0.25). Nothing else hand-wired in the main arms.
- RNpm: no PFC inhibitory neurons, X3 patch mean on the whole top drive (pm_ff 1), held-neuron onset rule, post side =
  next input, f2 0.18, f3 0.08.
- Training: birth (200 exchanges, rates 0), then overhearing (X1f OverhearWorld, 4 adult voices, never the held-out
  pairs), learning on. Checkpoints at 300, 1,000, 3,000 and 10,000 overheard dialogues. All probes frozen.
- After the last checkpoint: a frozen live world (60 exchanges, format and realistic caregiver) for talk-over,
  failure to yield, garble and live accuracy.

## Code (core8.py; defaults reproduce X1f core6 bit-exactly, eqtest8.py)
- `lat_fly`: the same ARCH two-layer L3 -> L3 propagation computed from the current weights (needed for n3 = 6,400).
  Exact copy of core6 when L33 does not learn (3,000+ ticks identical); with L33 learning it differs only in when
  the normalising sums are refreshed. Used in every X1g arm, including the N = 400 reference.
- Size: n2 = n3 = N. Fan-ins fixed and local (fan12 24 within r 0.2, fan23 30 within r 0.3, top -> L1 only to L1
  units within r31 0.3, birth lateral fan33 20). Patches stay ~33 units (N // 33).
- `sparse=1`: each L2 / L3 unit's threshold also follows its own long-run activity toward k / N (ARCH 4.10's next
  step "按长期活跃程度自我调整门槛"; core5's homeo3 form, same form on L2), k = 30 absolute, so the active fraction falls
  as N grows (7.5 % at 400, 1.9 % at 1,600, 0.47 % at 6,400). `kprop`: k = 0.2 N (proportional).
- Readout split `a` for top -> L1 and L3 -> L3: propagation = w / (out^a in^(1-a)); a = 0.5 is the ARCH / X1f form.
- Release `rel=1`: divisive gain + max-relative selection on the inner hearing:
  pred_j = (x_j / (1 + sum_k x_k) >= th1) and (x_j >= 0.6 max_k x_k), th1 = 0.015 (fixed now); `rel=0`: X1f's
  absolute floor 0.35.
- Every checkpoint also re-probes the same learned weights under 7 readouts (a 0.5 / 0.75 / 1.0, absolute / relative,
  th1 0.008 / 0.015 / 0.03): this isolates the readout form from training.

## Arms (3 seeds each, 0-2; 5 seeds for the N = 400 arms if the queue allows)
Main (learned path; A1 + A2 only):
| arm | N | k | L33 | a | release |
|---|---|---|---|---|---|
| M400_ref | 400 | X1f thresholds (~95 active, 24 %) | on | 0.5 | absolute (= X1f G0 in core8) |
| M400_k30_on / _off | 400 | 30 | on / off | 1.0 | relative |
| M1600_k30_on / _off | 1,600 | 30 | on / off | 1.0 | relative |
| M6400_k30_on / _off | 6,400 | 30 | on / off | 1.0 | relative |
| M1600_kp_on / _off | 1,600 | 0.2 N = 320 | on / off | 1.0 | relative |
| M1600_k30_on_a05 / _a075 | 1,600 | 30 | on | 0.5 / 0.75 | relative |

Positive control (readout isolated from path; DIAGNOSTIC, X1f G0P form): hand-wired held, question-specific paths
(B) for every seen pair, inner-hearing links (C) for the base half only; "new" = the other half, which must be learned
by the Hebbian readout. L33 off (as G0P_noL33).
| arm | N | assembly size m | generic L3 part | a / release |
|---|---|---|---|---|
| PC400_m5 | 400 | 5 (X1f) | X1f thresholds | 0.5 / absolute (= X1f G0P) |
| PC1600_m20 | 1,600 | 20, non-overlapping | ~30 active (homeostasis) | 1.0 / relative |
| PC6400_m30 | 6,400 | 30, non-overlapping | ~30 active (homeostasis) | 1.0 / relative |
The readout swaps at every checkpoint give every PC arm under every readout form.

## Measures (every checkpoint)
- learned top -> L1 drive (pre-threshold) into the answer's first-syllable copies vs the best wrong syllable, lags 1-6
  (X1f 3.4), and the fraction of questions whose answer is the top-driven syllable;
- fair specificity own_max - other1_max (X1e) with a 200-permutation null (mean, SD, 95 %), hub share;
- path decoding at lags 0 / 6 / 12 (chance 1/32): same-snapshot noise repetitions at every checkpoint, and X1e's
  context-varied phase-1 decoding at birth and at the end; within-path motion J(t, t+1); P(L3 empty);
- accuracy and first syllable on heard (seen) pairs, never-wired-C pairs (PC), held-out pairs;
- child utterances / garbled per overheard dialogue; live talk-over, failure to yield, garble per exchange;
- L2 / L3 active counts.

## Pass criteria (fixed now; judged at the best checkpoint, 3-seed mean)
1. Positive control: new-pair first syllable >= 0.5.
2. Learned path: context-varied question decoding at lag 12 >= 0.3, with within-path J(t, t+1) < 0.8 (motion kept).
3. Fair specificity (seen pairs) >= 0.10 and above the permutation null's 95 %.
4. Seen-pair first syllable > 0.2.
If a link fails: say where (code -> path -> specific prediction -> release), with numbers.

## Decisions from pilots (seed 0, before the batches)
- Overlapping random assemblies (m 5-30 drawn from a shared pool) break the hand-wired thinking already at birth
  (base pairs 0.18-0.64 instead of 1.0; decoding at lag 12 0.22-0.5). Non-overlapping assemblies work (m 20 at
  N 1,600: base pairs 1.00, decoding at lag 12 0.94), so the positive control uses non-overlapping assemblies.
- With k = 30 the L3 codes of different syllables stay distinct after learning (Jaccard 0.04 vs 0.67-0.74 with X1f
  thresholds, where L33 or the loop pulls every syllable into one generic state), but L3 is empty in about half of all
  ticks (silences).

## Addendum (2026-11-01, after the seed-0 pilots; written before the loop batches)
1. Coordinator request: with sparse homeostatic codes the pause dies (L3 empty from lag 3 on). Test whether the
   existing top -> L1 birth return loop (core4 `birth31=recip`, msg 84 "G returns to A": each L1 copy gets a birth
   link back from the 30 top units it drives most, total g_loop, exempt from decay) keeps the PFC alive. No self-hold,
   no fatigue. Arms `ML{N}_k30_{on,off}`: N = 1,600 and 6,400, k = 30, a = 1, relative release, g_loop 16.
   - Found at birth (dbg_loop.py, seed 0): with th1 = 0.015 the returned inner voice never passes the release (its share
     per line is 0.008-0.017 because a state returns to every line that drove it), so the loop cannot act. With
     th1 = 0.005 (or the absolute floor at N = 400) the loop reverberates through the pause (L3 -> inner voice -> L2 ->
     L3, about 12-75 units at N = 400). So the loop arms use th1 = 0.005; the return strength g_int stays 0.5.
   - Pass for this arm is the same as criterion 2 (decoding at lag 12 >= 0.3 with motion), also reported with 3-tick
     windows (the loop is a 3-tick cycle, so single-lag states can fall in an empty phase).
2. Post-hoc readout sensitivity (labelled as such): swaps with a stricter max-relative rho1 0.8 / 0.9 were added,
   because in the PC pilot the answer's drive was 1.5x the best wrong syllable's, so with rho1 0.6 both passed and A2
   silenced the mixture. Verdicts use the pre-registered readout; the swaps are reported separately.
3. Pilot timings on the A800: a 10,000-dialogue life takes about 3 min (N 400), 8 min (1,600), 30 min (6,400).
