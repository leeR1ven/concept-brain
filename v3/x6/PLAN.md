# X6 plan: harvest items 17, 18, 19, 21, 22 on the X3 hippocampus

Written 2026-11-01, before any run. Not to be changed after results; any later change is added at the end under
"Changes after the plan", with the reason.

**Harness.** `x3_hippo.py` (copied unchanged from `v3/x345/`) is imported by a new `x6_hippo.py`. With all new options
at their defaults, `x6_hippo.py` is the X3 base: `recode_rec`, N_H 10,000, k 64, learned fan-out C 2,000, birth
recurrence (cB 400, rho 1, gain control = minus the mean), the X3 rule (only just-changed units: falling → rising
+η_seq·g, co-rising +η_co·g, reverse −η_rev·g, g = 1/(1+w)), two-layer recall weights, quantile threshold with the
2× ceiling, one decay γ = 0.99 per 200 ticks. Stream = `stream.json` (md5 582c6de8…, same file as X3).
Seeds 0–4 (5 seeds) unless stated; 3 seeds for the costly s = 4 runs. Paired comparisons use the same seed (same
world, same probes, same birth wiring).

**One harness change, needed by item 17 only.** X3 updates the hippocampus only when its input changes. Then a
silent pause costs nothing: the last Q state links straight to A, however long the pause. For the Q–A probes the
pause is therefore *stepped*: one update every 3 ticks (one char-moment, the clock X3 uses inside lines), with the
input that is really there (the scene, plus the inner voice in arm A1). World lines keep X3's event-driven shortcut.
The unstepped X3 pause is run as a reference, not as an arm.

## Item 17: the hippocampus bridges a pause

- **Probes.** "qa" triples, each in one quiet gap with one scene: Q_i (novel 6-char line), a 12-tick silent pause
  (4 stepped moments: the line end + 3), A_i (a 2-char answer), 15 ticks, the next pair. Each life has a pool of 6
  answers (12 distinct chars). A triple uses 3 different answers. Q chars never come from the pool. Unique 6-char
  probes ('u', X3 kind) alternate with the triples, to measure side effects. Life 100 k ticks, one probe slot per
  1,500 ticks; tests every 10 k ticks.
- **Arms.**
  - A0 base: stepped pause, no inner voice, X3 learning window (only just-fallen units, i.e. 2 ticks).
  - A1 inner voice: during the 4 pause moments the AUD copies also carry a fixed random 25 % of the union of Q's char
    assemblies (no speaker-voice units). Secondary arm A1w: 10 %.
  - A2 wide window (h2 item 13), hippocampus only: no inner voice; units that fell within the last 15 ticks (up to
    5 moments) also strengthen onto rising units (+η_seq·g, flat window).
  - Reference R: X3's unstepped pause.
  - Sensitivity: A0 and A1 with a 30-tick pause (10 moments).
- **Recall (no learning).** Each pair aged 500–12,000 ticks (age from A's end) at each test time: re-hear Q through
  the copies (all 6 chars, full input, its own scene, speaker 'other' = X3 input mode), then the hippocampus runs on
  its own for (pause moments + 4) steps with X3's recall dynamics. Each step's return to AUD is decoded to a char.
- **Primary metric: right-A rate** = the first pool char decoded is a char of the right answer. Also wrong-A (another
  pool answer first; split into triple-mates and others) and none. Secondary recall mode: after Q, the pause input is
  re-presented (scene, + inner voice in A1), then 4 free steps.
- **Side effect:** unique-probe recall i100 (age < 10 k) and r50 (age < 10 k).
- **Pass:** right-A ≥ 0.8, and ≥ A0 + 0.15 (paired mean over 5 seeds, better in ≥ 4 of 5 seeds), and side-effect
  drop ≤ 0.10. If A0 itself reaches ≥ 0.8, the refinement does not beat the base → FAIL ("not needed here").

## Item 18: encoding vs recall by a strength ratio, no switch

- **No-switch dynamics.** Every update (listening, cue, recall) is
  `d = β·birth(input) + ρ·(brec(prev) − mean) + λ·rec_prop(prev)`, then the same threshold. λ > 0 always, also while
  listening. Recall keeps the same update with the input that is present in silence. X3's switch had λ = 0 while
  listening, and its recall used the learned links only (no input, no birth recurrence).
- **Sweep.** β = 1. λ on a log grid wide enough that the measured ratio R = (k-th largest chain drive in speech) /
  (k-th largest λ·rec_prop drive from a stored code) spans about 0.1–10. R is measured in every run and reported.
  The grid is placed after one small local scale check (N_H 4,000). That check is not a result.
- **Arms.** B0 = X3 switch. B(λ) = no switch, one life per λ. Each life is tested with two silent inputs (tests are
  offline, so the same life serves both): "eye" = the probe's scene stays visible (main), "dark" = no input
  (diagnostic). 5 seeds, life 100 k ticks, unique probes, tests every 5 k ticks.
- **Metric.** Fresh recall = r50 (50 % stored-code cue) at age < 10 k. Also i100 at age < 5 k, horizon, Hs fidelity.
- **Pass:** some λ with fresh r50 ≥ 0.9 in the "eye" condition, no switch. PARTIAL if ≥ 0.9 only in "dark".
  FAIL otherwise.

## Item 19: cued recall vs random kicks in quiet times

- X3 base life, 5 seeds, 200 k ticks. At t = 50, 100, 150, 200 k: 100 random kicks (X3 form: k random Hs units on)
  and 100 cued kicks. A cued kick is the inner voice of the current thought: the AUD assemblies of the first 2 chars
  of a line heard in the last 3,000 ticks (picked at random), entered through the copies with the current scene and
  no speaker-voice units. Each kick then runs 4 free steps, decoded at AUD.
- **Heard set:** within-line bigrams of every line heard before t, plus last→first char of consecutive lines < 30
  ticks apart.
- **Metrics (per kick):**
  - silent = nothing decoded;
  - chimera = at least one adjacent decoded pair never heard;
  - real = at least one pair, all heard;
  - X3-style "any heard bigram";
  - wrong pairings per 1,000 quiet ticks = (never-heard adjacent pairs inside the recall + the pair "current
    thought's last char → first recalled char" if never heard) × 1000 / 12 (a kick = 4 steps × 3 ticks; ASSUMED
    continuous replay). The current thought is the same line for both arms.
  - Cued only: right continuation = the first recalled char is the line's 3rd char.
- **Pass (cued better):** chimera rate lower by ≥ 0.20 and wrong pairings per 1,000 quiet ticks ≤ ½ of random,
  paired over 5 seeds.

## Item 21: sizing under whole-model scaling

- s ∈ {1, 2, 4}: N_H = 10 k·s. Four arms at s = 2 and 4: (a) k 64, C 2,000; (b) k 64·s, C 2,000 (X3's s sweep);
  (c) k 64, C 2,000·s; (d) k 64·s, C 2,000·s. The birth recurrence keeps the same mean input count per unit
  (cB = 400·s·64/k). Birth fan-in from the copies and the copies themselves stay fixed (the world cannot get richer
  here). Read-out fan-out C_out 1,024 fixed.
- Life 200 k ticks, unique probes only, one per 1,500 ticks, tests every 10 k ticks (60 most recent probes + 40
  older). 3 seeds per config. s = 4 runs on the A800 CPU queue.
- **Metric:** A50 = the age at which unique-probe r50 falls to 0.5 (linear interpolation over age bins 0, 1, 2, 5,
  10, 15, 20, 30, 40, 60, 80, 100, 150, 200 k); also A90, A50 of i100, Hs-level fidelity.
- **Verdict per arm** from A50(s = 4) / A50(s = 1): flat if 0.7–1.4; ∝ s if ≥ 2.8; sublinear in between.
- **Decay ∝ 1/s:** γ_s = 0.99^(1/s) per 200 ticks, on the two arms with the largest A50 at s = 4 (chosen by that
  rule), at s = 2 and 4, life 400 k (s = 1 base also rerun at 400 k). PASS if A50 grows ≥ 1.3× vs the same arm with
  γ fixed while fresh r50 (age < 5 k) stays ≥ 0.9.

## Item 22 (sensory level): one-shot fact

- **Facts.** Four lines heard once each, by 'other', in the first quiet gaps after t_f: 我的小狗叫N1, 我的小猫叫N2,
  我的鸭子叫N3, 我的小鸟叫N4. Names are reduplicated 2-char pet names from familiar chars, 4 different ones per seed
  from {豆豆, 花花, 球球, 星星, 白白, 月月, 熊熊, 果果}. None starts with 小, while the world's own names
  (小丽, 小明, 小美, 小强, each heard about 50–90 times in the stream) all do.
- **Lives.** Fresh hippocampus: t_f = 3,000. Adult: t_f = 300,000. 5 seeds each; the life ends 11 k ticks after the
  last fact.
- **Tests (no learning)** at delays of 1 k, 2 k, 5 k and 10 k ticks after each fact:
  - R1 (main): hear "我的X叫" (5 chars through the copies, speaker 'other', the scene at test time), then 2 free
    steps.
  - R1s: the same in the fact's own scene.
  - R2: hear the whole question "我的X叫什么", then 4 free steps; N1 anywhere.
- **Primary metric:** name recalled = step-1 decode is N1. Also full name (N1 N1), intrusions (another pet's new
  name, the world's old name: 小 at step 1, other).
- **Pass:** name recalled ≥ 0.8 (mean over the 4 delays) in both fresh and adult, R1. PARTIAL if it holds only for
  fresh, only in R1s, or only up to 5 k.

## Compute
Local: one small process (< 300 MB) for smoke checks only. All result runs go to `queue_a800/` as x6_* jobs, at most
2 at a time, 4 processes per job.

## Changes after the plan
All of these were made during debugging and calibration (seed 9, which is never used for a verdict), before any
result run.
1. **Cue start (all items).** X3's input mode starts a cue with one eye-only update from an empty state. That update
   skips the recurrent gain control, so it turns on about 2k units (112–128 instead of the 0–27 that are really on
   after a silence). Because of the birth recurrence, the replayed codes then take another path: overlap with the
   stored code is 0.2–0.35 by the 6th char of Q. From an exact copy of the encoding-time state it is 1.00.
   - Default here: the cue starts from the hippocampus's own state at the test time. That is usually the small
     post-line state, i.e. "the hippocampus as it is now hears the line".
   - X3's start is kept as a secondary number for the unique probes (i100x).
2. **Two-layer sums kept current when λ > 0 (item 18).** X3 recomputes S_out and S_in every 200 updates. That only
   mattered at recall, because λ = 0 while listening. With λ > 0 while listening, stale sums give a new link from a
   unit that had no links a 1/√(10⁻⁶) boost. The sums are now updated at every learning step, which is HARVEST 一.6
   (adopted), plus X3's full recompute every 200 updates. Diagnostic arm: X3's stale sums at one λ.
3. **Item 18 λ grid.** Placed from the full-size calibration run (seed 9), not from the small local check.
   Grid: λ ∈ {1.5, 2.5, 4, 6.5, 10}. Calibration: λ = 1.5 gave R ≈ 9 and fresh r50 = 0.10. At λ = 13.5 and 120
   the network locked into the first learned pattern and stopped learning (about 800 live links). Plus
   "Bstale4" (λ = 4 with X3's stale sums).
4. **Item 17 recall variants (from the seed-9 calibration, where the planned metric was about 0 in every arm, and
   the X3 unstepped reference was 0 too).**
   - Cue: re-hear Q (planned) / 50 % of Q's stored last-char code (X3's r50 form) / ideal re-hearing.
   - Ideal re-hearing replays Q from the exact encoding-time state and threshold. It isolates the storage half from
     the replay problem in change 1.
   - Continuation: F, free run (planned primary) / P, the pause input is present again, then a free run (planned
     secondary).
   - **The verdict stays on the planned primary (re-hear Q, F).** For the inner-voice arms the arm's own model
     implies P: the PFC holds Q again after hearing it again, so the inner voice is on in the recall pause too.
     That is reported as the arm's "natural" number, next to the verdict.
5. **Item 22 storage diagnostics:** I1 = ideal re-hearing of 我的X叫 (exact encoding-time state, the fact's scene);
   S50 = 50 % of the stored 叫-moment code. The verdict stays on R1.
6. A rule option "held" (ARCH §二.8 reading: a unit held on while B rises strengthens A→B) was coded and tried once
   at mid size (no effect on the stored-cue bridge). It is not used, because the brief says to keep the X3 rule.

## Follow-up diagnostics, run after the results (not pre-registered, not used for verdicts)
- d18_*: item 18 split by side. D1 = X3 listening (λ = 0) + no-switch recall with λ_rec ∈ {2.5, 6.5, 20}. D2 = λ = 4
  while listening + X3 recall.
- dr{0.5,0.25}_*: birth-recurrence strength ρ (replay reproducibility) for items 17 (A0, A1w) and 22.
- dsep_r*: ρ effect on X3's separation of similar episodes and on unique-probe recall.
- de20_22_*: item 22 with less scene weight in the hippocampal fan-in (cE 60 → 20).
- q21g_*: the pre-registered decay ∝ 1/s follow-up. Rule-chosen arms: c and d, the largest A50 at s = 4.
