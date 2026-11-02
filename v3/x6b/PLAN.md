# X6b plan: an innate listening signal, and ρ = 0 with fast weights, on the X6 / X3 hippocampus

Written 2026-11-01 after reading X6 (REPORT, tables, code) and ARCHITECTURE §四.19, and after small local smoke checks
(N_H 4,000, seed 9, 15 k ticks; listed under "What the smoke checks showed"). Written **before any full-size run**.
I will not change it after seeing results. Any later change goes under "Changes after the plan", with the reason.

Two scope messages from the coordinator came in before this plan was written. Both are folded in:
- (designer 11-01) Another scene giving another code is normal, as context-dependent memory is in humans. The
  per-source normalisation arm is **dropped**. Same-scene and other-scene numbers are reported side by side as
  measurements. The verdict on re-hearing uses the same scene.
- (designer 11-01) "The hippocampus remembers fast because its weights change fast, not because of its random
  connections." Added: ρ = 0 arms at η ×1, ×4, ×10, and ρ = 0 plus the listening signal. Separation is split by twin
  type. Same scene + same voice (pS) is reported but is not a pass criterion. Other voice (pV) and other scene (pD)
  must be separated.

## Harness
- `x3_hippo.py` and `x6_hippo.py` are copied unchanged from `x6/`. `stream.json` has the same md5 (582c6de8…).
- `x6b_hippo.py` subclasses them. With `ls = None` it is exactly X6's harness.
  - Extra tests restore the hippocampus's random-number state afterwards, so the X3 reference lives should
    reproduce X6's numbers. That is the regression check.
- Full size as X6: N_H 10,000, k 64, C 2,000, C_out 1,024, cB 400, the X3 rule, two-layer weights, decay γ 0.99 per
  200 ticks.
- Seeds 0–4 for results. Seed 9 for calibration only.

## The listening signal (LS)
- **Population.** 64 binary units, birth-wired, never learning. Each reads 300 random units of the auditory **input
  layer**: the externally heard char assembly plus the speaker's voice units.
  - A unit is on if any of its inputs is on. L = the fraction of units on.
  - One heard char gives L ≈ 0.99; fewer heard units give a smaller L; silence gives 0.
  - It is content-free and has no timer.
- **Inner voice does not reach it.** Like the brake (ARCH §二.7), it does not see the inner voice, because return
  chains do not reach the input layer.
- **Action.** s = 1 − (1 − f)·L multiplies the hippocampus's recurrent drive, i.e. both the birth recurrence and the
  learned links. f is the suppression factor: 0.1 means scaled to 10 %, Hasselmo & Schnell's 89 % suppression.
- **One update rule.** It is the same for listening, silence and recall; there is no store/recall switch:
  `d = birth(input) + s·[ρ·(brec(prev) − mean) + λ·rec_prop(prev)]`.
  - In silence s = 1, so the learned drive is at full strength (λ_full). That is recall, also inside the life's own
    silences, with learning on as always.
  - Recall tests use the same update, with the silent input: the scene stays visible, the "eye" condition.
- **Implementation.** The multiplicative gating is the presynaptic-inhibition reading of ACh (ASSUMED form). It is
  the one structural addition, allowed by the designer's 11-01 note (ARCH §二.12).

## Arms (5 seeds each; all are given the same five lives)
| id | arm | settings |
|---|---|---|
| X3 | reference: X3 with the store/recall switch | ρ 1, λ 0 while listening, X3 recall |
| NS | reference: X6 no-switch | unified, λ 2.5 always (X6's best no-switch λ), ρ 1 |
| L03, L1, L3, L5 | LS, ρ_full 1 | f = 0.03 (extension: X6 said listening needs λ ≈ 0), 0.1, 0.3, 0.5 |
| L1r, L3r, L5r | LS, ρ_full 0.5 | f = 0.1, 0.3, 0.5 |
| Z1, Z4, Z10 | ρ = 0 (no birth recurrence), X3 switch | η ×1, ×4, ×10 on all hippocampal learning rates (seq, co, rev, out) |
| ZL | ρ = 0 plus LS | f, λ_full, α and η ∈ {×1, ×10}, calibrated |

- **Decay for Z4 and Z10.** γ_k = 0.99^(ln(η_seq/w_min) / ln(k·η_seq/w_min)) per 200 ticks: 0.98740 (×4), 0.98569 (×10).
  This keeps fixed the age at which a link made by one exposure falls below w_min. The horizon A50 is reported for
  every arm, so how comparable they are can be seen. ZL uses the same γ rule for its η.
- **Calibration (seed 9 only, before the result runs).**
  - For each LS-type arm (L*, ZL) pick λ_full ∈ {3, 6, 12, 24} and the threshold smoothing α ∈ {0.3, 1.0}; for ZL
    also f ∈ {0.03, 0.1, 0.3} and η ∈ {×1, ×10}.
  - Why α is calibrated: the smoke checks showed that X3's slowly tracking threshold (α 0.3) cannot follow a drive
    scale that changes between listening and silence. α = 1 is X3's own "k-WTA stand-in" limit.
  - Calibration life: the item-18 life, shortened to 50 k ticks, eye condition only.
  - **Rule:** take the highest fresh r50 (age < 10 k). Ties within 0.02 go to the higher i100, then to the higher
    same-scene re-hearing overlap minus the other-line control.
  - The X3, NS and Z arms are not calibrated: they use X3's / X6's values (α 0.3).

## Lives (X6's probe sets and seeds; X6 spec names in brackets)
- **qa:** kinds u + qa, inner voice 10 % (iv 0.10), stepped 12-tick pause. 100 k ticks, tests every 10 k (= q17_A1w).
- **uni:** kinds u. 100 k ticks, tests every 5 k, eye and dark conditions (= q18).
- **sep:** X3's mix (u, pS, pV, rB, pD, rS). 100 k ticks, tests every 10 k (= dsep, now 5 seeds).
- **fresh / adult:** four one-shot facts 我的X叫N at t = 3 k / 300 k, tests at delays 1, 2, 5 and 10 k (= q22).

## Tests, metrics and pass criteria (fixed now)
1. **Item 18, fresh recall without a mode switch.**
   - Metric: r50 (50 % of the stored code of char 3, then 3 silent steps, eye condition), age < 10 k, uni life.
   - **Pass ≥ 0.9.** For X3 this is with its switch, so it is the reference, not a pass candidate.
2. **Re-hearing returns to the stored code (same scene).**
   - Re-hear the whole u-probe line (6 chars) from the hippocampus's state at test time, in the probe's own scene,
     with no learning.
   - Overlap per char = |re-heard ∩ stored| / max(|re-heard|, |stored|), age < 10 k, uni life.
   - Control "ovx": overlap of the same re-heard codes with a different line's stored codes. It catches a locked-in
     network, which overlaps with everything.
   - **Pass:** mean over chars 1–6 ≥ 0.7, char 6 ≥ 0.7, and ovx ≤ 0.3.
   - Also reported (not for the verdict): the 7-char fact line in the fresh life (same scene), and other-scene
     re-hearing (the scene at test time). For other scene, only lines whose scene really differs are counted.
3. **Item 22, end-to-end one-shot fact.**
   - Hear "我的X叫" in the fact's own scene, then 2 silent steps; success = step 1 decodes to the name's char (R1s).
     Mean over the four delays.
   - **Pass ≥ 0.6 in fresh and in adult.**
   - Reported beside it: R1 (scene at test time, X6's primary) and I1 (ideal re-hearing, storage only).
4. **Item 17, Q→A pause bridge, 10 % inner voice, natural recall (X6 "P").**
   - Re-hear Q, the pause input (scene + inner voice) comes back, then 4 silent steps. Success = the first answer-pool
     char is the right answer.
   - **Pass: right ≥ 0.8 and wrong ≤ 0.05**, qa life.
5. **Separation of similar episodes**, split by twin type.
   - Metric: sep_right = 50 % of A's stored char-3 code gives A's 4th char. Age < 20 k, sep life.
   - **Pass: pV (other voice) ≥ 0.9 and pD (other scene) ≥ 0.9.**
   - pS (same scene, same voice) is reported, not a criterion.
6. **Ordinary recall must not fall.**
   - Metric: i100 (re-hear the first 3 chars, own scene, then 3 silent steps) and r50, age < 10 k, uni life.
   - **Pass:** each ≥ X3's value − 0.05 (paired mean over seeds).
7. **Horizon (reported, all arms).** A50 = the age at which uni-life r50 falls to 0.5, as in X6 item 21.

**Verdict on a proposal.**
- PASS = some arm of that proposal (LS: L*; ρ = 0: Z*, ZL) passes tests 1–6.
- PARTIAL = it passes test 2 and at least one of tests 3 or 4.
- FAIL otherwise.
- Every number is reported as mean ± SD over 5 seeds, with the paired difference vs X3.

## What the smoke checks showed (small model, seed 9; not results; they shaped only the plan)
- X3 reproduces X6's main finding: same-scene re-hearing overlap 0.62 at char 1, falling to 0.28 by char 6.
- With LS and α 0.3, the threshold carried over from silence (where λ_full·learned is large) is too high at the next
  line's start. Codes shrink to 0–20 units. Hence α is calibrated.
- With λ_listen = f·λ_full ≳ 5 the network locks into one state: overlap 0.97 with everything, ovx 0.97, about 300
  live links. Hence the ovx control.
- With strong suppression (f ≤ 0.1, α 1), re-hearing is reproducible and specific: overlap 0.97–0.99, ovx 0.05.
  But in the small model r50 and the facts were 0, and the readout of the stored codes was poorer (0.3–0.8 vs 0.89).
  That must be checked at full size.
- ρ = 0 with X3's switch: overlap 0.89, r50 0.48 (X3 0.89 at that size), facts 0.

## Compute
- Local: one process at a time, at most about 150 MB, small checks only.
- Calibration and results: `queue_a800/` as `x6b_*` jobs, 4 processes each, at most 2 of my jobs at once.

## Changes after the plan
1. **(10:12, before any full-size result; the calibration jobs were queued, not run.)** A further local smoke check
   (small model, seed 9, 20 k ticks) split storage from recall with a new diagnostic, r50x3: the same r50 cue
   recalled with X3's dynamics, learned links only.
   - With f = 0.1 and λ_full 12 (λ while listening ≈ 1.2), r50x3 = 0.00, with or without learning in the silent
     updates. With λ ≈ 0 it was 0.39 (f 0.1) and 0.79 (f 1).
   - So even λ ≈ 1 while listening wrecks storage. Recall in silence needs λ_full ≳ 10 (X6 D1).
   - Hence one more calibration arm, **L001: f = 0.01, ρ 1, λ_full ∈ {6, 12, 24, 48}**, and f = 0.01 added to ZL's
     grid. If chosen, L001 is a result arm like the others, labelled as an extension beyond the brief's f values.
   - r50x3 and the readout of stored codes ("rd") are reported as diagnostics. They are not used for verdicts.
2. **(10:13, before any result run) Decay formula bug.** `eta()` in the spec generator inverted the exponent: it gave
   γ 0.99207 / 0.99302 (slower decay) instead of the planned 0.98740 / 0.98569 (faster). It is fixed. The 24 ZL η×10
   calibration lives that ran with the wrong γ were moved to `cal_wronggamma/` and recalibrated with the right γ
   (cal3, names `_e10g`). The result arms use the planned values.
3. **(10:26) Calibration outcome, by the rule.**
   - L001 λ24 α1; L03 λ24 α1; L1 λ12 α0.3; L3 λ6 α1; L5 λ3 α1; L1r λ12 α1; L3r λ3 α0.3; L5r λ3 α0.3.
   - ZL: f 0.01, λ12, α1, η×10. (λ24 had r50 0.79 vs 0.77, a tie; it went to the higher i100, 0.84 vs 0.81.)
4. **Follow-up diagnostic (not pre-registered, not used for verdicts): Z10a1** = Z10 with α = 1. ZL differs from
   Z10 in both the listening signal and α. This arm separates the two.
