# X8 plan: the designer's two correction mechanisms (ARCH §二.8) — HARVEST items 3, 4 (+5), 12

Written before any run (2026-11-01). Not to be changed after results; any change is logged in §9 with the reason.

## 1. Question
ARCH §二.8: "改正有两种机制：牢固的记忆难改；新学的东西先落在变化快的连接上（海马体就是其中之一）".
IDEAS #22 goal: a learned skill is kept for life, AND corrections stay possible.
Does (item 3) "concave on the way down too" and/or (item 4) "a fast hippocampus-like store + reward" give both?

## 2. Harness
X4 merge harness (`v3/x345/x4_merge.py`, copied unchanged into this folder and imported), scenario fixed_in, one
agent = X4's sibling 0: lineage seed 1+s, lessons `make_lessons(rng(50+s))`: 20 shared + 30 own + 15 conflict-own
= 65 lessons "q1 q2 -> a", raised exactly as X4 (`raise_sib`: 10 reps + 300 filler, shuffled, seed 100s+11).
Readout as X4: two-layer drive of the PFC state after the question onto each char's AUD assembly; an answer is
released if best >= theta and best >= 1.25 x second; theta = 0.3 x median best score after learning (per agent).
New code (`x8.py`): rule variants, the fast store, the reward tool, ask episodes, own voice, phases.

### 2.1 Rule (one rule, same for cortex and fast store)
g(w) = max(0.01, 1/(1 + w/w_ref)^pow), w = birth + learned (>= 0), w_ref 1, pow 1 (X4 values).
Growth (falling PFC -> rising AUD: +eta g; co-rising: +eta_co g) as X4 in every arm.
- base: reverse-order weakening -eta (plain); decay D <- gamma D every 200 ticks (plain).
- item 3: weakening -eta g(w); decay D <- D (1 - (1-gamma) g(w)) per period (same factor, same 1 % floor).
- reference "X4 rule": weakening concave, decay plain (this is what X4's code did; it is half of item 3).
Cortex: eta 0.3, eta_co 0.01, gamma 0.999 / 200 ticks, w_min 0.002 (X4 BASE).

### 2.2 Fast store (item 4)
Its own population H: 4,096 binary units, about 1 % on (41, quantile threshold like the PFC), fed by chain copies
of AUD (random birth fan-in 64, fixed), random birth recurrence (fan-out 128, fixed; carries order, no time cell),
its own random birth links H -> AUD (U(0, 0.02), like PFC -> AUD) plus learned H -> AUD links with the SAME rule
(variant of the arm), eta_h 1.0, eta_co_h 0.033, gamma_h 0.97 / 200 ticks (fast growth, fast decay; ARCH §二.6/8).
Recall into the same AUD: at the silent moment after a heard question (no external input), the H readout
(two-layer drive x r_hip) competes like the motor readout (same theta, same sharpness); if it wins, that char's
AUD assembly lights (inner hearing), so the PFC sees Q then A and the cortex learns Q -> A from the recall (§二.6:
recall acts through the areas, never H -> PFC directly). Recall strengthens H too (it is just another moment).
The agent's spoken answer = the same competition on (cortex drive + r_hip x H drive).
r_hip calibration (pre-registered, done once on seed 99, never on main seeds): smallest of {0.5, 1, 2, 4} for
which an adult (after phase 1 + stretch, A34 config) recalls >= 0.9 of 10 brand-new lessons right after hearing
each once. This is the defining property "one-shot episodic memory", not an outcome metric.

### 2.3 Reward (item 5, part of the item-4 arm)
Innate value only: the adult is pleased when the agent's released answer is the expected one. Then every change
made in the last 24 ticks (fading linearly with age) is amplified: added again x kappa (kappa 2, net x3), in the
cortex and the fast store. Upset sub-arm (ARCH §二.8 second half): when the agent says a wrong answer, those
changes are reversed (net x -2). The motor stop-neuron link is not modelled (no motor area here).

### 2.4 Episodes (the world)
- line "q1 q2 a" (lesson / recast / filler): heard, 3 ticks per char, then silence 15 ticks (X4 hear()).
- ask: the adult says q1 q2; the answer moment follows in silence (inner recall and/or own voice may light AUD);
  then silence. If judged (adult present): pleased -> reward; wrong or no answer -> the adult recasts "q1 q2 a"
  (ARCH §二.11: correction repeats question and answer).
- own voice (self-rehearsal): the released answer enters AUD as a fixed subset of round(w_own x 16) of its 16
  content units ("signal strength = how many neurons light", §二.1; own and other voices share content units).
  w_own = 0 means no self-rehearsal.

## 3. Phases (same lessons, same seeds, same world streams in every arm)
1. Learn: X4 raising (950 lines).
2. Stretch: 6,000 lines = 3,000 asks of random lessons (no adult reaction) + 3,000 filler lines (filler never
   starts with a lesson question), shuffled. ~144k ticks (~X2's 3,000 dialogues of 2 lines).
   - amnesia branch (fast-store arms): the fast store is wiped at the start of the stretch.
   - untouched branch (w_own 0 runs): 1,000,000 ticks of silence after phase 1 (decay only; a "life").
3. Corrections: 4 lessons per seed (fixed per seed, drawn from the 50 shared+own lessons) get new answers.
   20 rounds; each round = 4 correction episodes (judged asks: pleased if new answer, else recast) + 8 asks of
   random unchanged lessons (not judged) + 20 filler lines, shuffled. Same w_own as the stretch.
4. Adult new learning (item 20): from the end of phase 2, 30 never-heard lessons (X4's sibling-1 lessons),
   10 rounds; each round = each new lesson once as a judged ask (recast if wrong) + 15 filler. Compared with a
   fresh birth of the same arm on the same stream.

## 4. Arms
| arm | weakening/decay | fast store | reward |
|---|---|---|---|
| A0 base | plain | – | – |
| A3w (X4's own rule, reference) | concave weakening, plain decay | – | – |
| A3 item 3 | concave both | – | – |
| A3R | concave both | – | pleased |
| A4H | plain | yes | – |
| A4 item 4 | plain | yes | pleased |
| A34H | concave both | yes | – |
| A34 items 3+4 | concave both | yes | pleased |
| A34U | concave both | yes | pleased + upset |
| A12 (only if A3 and A4/A34 fail) | w = birth + F + S | – | pleased |
w_own sweep {0, 0.25, 0.5, 1.0} for A0, A3, A3R, A4, A34, A34U; w_own 0 only for A3w, A4H, A34H.
Seeds 0–4 (5). Sensitivity (A34, seeds 0–2, w_own 0 and 1): gamma_h 0.95 / 0.99, r_hip x0.5 / x2.

## 5. Metrics
- Retention = fraction of lessons right (cortex only) at the end of phase 1 that are still right (cortex only)
  after the stretch. Also absolute accuracy, and behaviour (cortex + fast store).
- Re-teachings to switch: per changed lesson, recasts received before the first judged ask answered with the new
  answer (never within 20 rounds = 21). Also: stable (new answer in every later judged ask), total recasts,
  cortex-only right on the new answers at the end (= after wiping the fast store).
- Unchanged: of the unchanged lessons right (behaviour) at the start of phase 3, the fraction still right at the
  end (behaviour and cortex only).
- Adult learning: teachings before the first right answer (never = 11) per new lesson; adult vs fresh.

## 6. Pass criteria (from the brief)
- P1 retention: cortex-only retention after the stretch (amnesia branch where a fast store exists) >= base
  (paired, mean over seeds) AND >= 0.8.
- P2 correction: >= 90 % of the changed answers (pooled over 4 x 5) switch within <= 5 re-teachings.
- P3 unchanged: unchanged behaviour retention over phase 3 >= 0.95.
- P4 self-rehearsal: P2 and P3 still hold with own voice on; reported per w_own (the w_own range that passes).
- An item PASSES if P1–P3 hold at w_own 0 and P4 holds for at least w_own 0.25; PARTIAL if some hold; else FAIL.
- Item 20 (secondary): adult with fast store needs fewer teachings than without (paired), and close to fresh.

## 7. Compute
Local: pilot/timing only, 1 process, < 300 MB. Main runs on queue_a800 (<= 2 jobs at a time, 4 procs each).

## 8. Known deviations of the harness from ARCH (accepted, not fixed)
- X4 learns PFC -> AUD only from falling and co-rising PFC units; held PFC units (on before and after an onset)
  do not learn (ARCH §二.8 refined says they should). Checked in the pilot (overlap of consecutive states);
  unchanged unless large.
- No thinking dynamics, no motor area: speech = the readout competition.

## 9. Change log (after the plan was written; all before any main run, from seed-0 pilots of 600-line stretches)
1. ask(): after the answer moment one more silent step only if something lit AUD, so an ask has the same step
   structure as a heard line. Why: two silent steps in a row lowered the live PFC threshold and changed the codes
   of what came next (diag/d1.py).
2. Tests start the probe threshold from a fixed reference (the live value at the end of phase 1), so a test does
   not depend on what was heard just before.
3. Fast-store wiring: fan-in 32 (not 64), recurrence 1.0 (not 0.5), and the store is silent without chain input
   (its recurrence alone cannot reach threshold). Why (diag/d2.py, d3.py): with the PFC's settings its codes were
   dominated by the last char; it falsely recalled an answer for 40 % of never-heard questions, and after the
   amnesia wipe its recall locked in wrong answers (store right 0.02).
4. Recall lighting: AUD lights the winner of the summed return drive (cortex + store) when the store's drive to
   that char reaches theta (instead of "the store alone wins its own competition"). Why: the store's false recalls
   on known lessons taught the cortex wrong answers (amnesia retention 0.31 vs 0.55 without a store).
5. r_hip calibration rule dropped; r_hip = 1 (both returns are two-layer normalised, so no scale factor is
   needed), sensitivity 0.5 / 2. Why: the planned one-shot criterion cannot be reached by scaling (sharpness of a
   new lesson does not depend on r_hip) and it ignored false recalls.
6. P3 control added: the same phase-3 stream with the 4 correction episodes replaced by unjudged asks of the old
   lessons. Why: in the pilot the unchanged lessons eroded from the phase-3 filler alone (0.8 after 6 rounds in
   every arm). P3 is reported as planned (end / start) AND as corrections vs control (the damage the corrections
   themselves do).
7. Item 12 (A12 = base rule + fast part + pleased; A12U = + upset) is built now and runs in the same batch to
   save a server round; its verdict counts only if items 3 and 4 fail. F decays with 0.99 / 200 ticks, S never
   decays; half of F moves into S whenever F > 0.5 after a strengthening, and all recent F moves at a pleased
   moment.
8. Added AFTER the first main results (job x8_m1): job x8_c1, corrections of freshly learned answers (phase 3
   straight after phase 1, no stretch), all arms, w_own 0 / 0.25 / 1, 5 seeds; and the correction metrics are
   also reported for the changed lessons that were still right before phase 3 ("known"). Why: the 6,000-line
   stretch erodes most lessons in every arm, so the planned step 3 mostly re-teaches answers that were already
   lost, which does not test "a strong memory is hard to change". The pre-registered criteria are unchanged and
   are still applied to the planned (post-stretch) runs; the new runs are reported next to them.
Pilot context (seed 0, not a result): 600 stretch lines cut base cortex retention to ~0.5 (filler and unanswered
asks both erode it; pure time does not); the store's recall during asks keeps lessons (0.98) but also keeps
re-storing the OLD answer during each correction episode; the upset reversal undid that and switched 4/4 in 2–4.
