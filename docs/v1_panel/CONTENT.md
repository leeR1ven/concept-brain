# CONTENT route: learned echo map + latch-replay copy + summed-drive stop (+ fast/slow first-syllable memory)

Content specialist, architecture panel, 2026-10-30. Labels: **PROVED** = algebra or exact numerical transcription of
brain/net.py; **SIMULATED** = numpy sims in this folder on the real corpus with the exact ear replica (seeds given);
**ASSUMED** = untested. Folder: `/tmp/claude-0/-root/e92d3d7d-5823-4f5c-a0ec-4fe302bff9af/scratchpad/arch/content/`.
Builds on the learnability design (T, P = T∧GO, beat ring R2→R0→R1, syllable programs Y with WTA, END, teacher cue
and END push, C11 teacher gate) and on the verifiers' findings listed in the brief. Section 8 = files.

STATUS: final. Never-heard results 3 seeds × 1k/3k/10k dialogues; memory results 2 seeds up to +10k dialogues.

---------------------------------------------------------------------------------------------------------------------
## 0. Summary

1. **Copy route (never-heard lines)** = three new unit types per program c plus one learned path:
   - **Z_c, auditory mirror of program c**: learned FEAT0→Z_c (exact rule, back term on), taught only by the model's own
     teacher-forced speech: Z_c is clamped one tick at b+4 after Y_c was held b..b+2, when the own voice of c fills
     the whole ear window (b+3) — the cleanest, context-free layer-0 code of c. Plasticity gated by own voice (V on).
     During listening Z works as a winner-take-all detector with threshold 0.75·B_Z.
   - **L_c, latch** ("c was heard in the partner's current line"): set by Z_c ∧ F, reset by the onset of the partner's
     next line, and cleared when Y_c is released (inhibition of return).
   - **E_c = L_c ∧ (GO ∨ R1)**: one-tick replay pulse of every latched char on the R2 tick before *every* boundary.
   - **learned E→Y** (exact rule): only the diagonal E_c→Y_c survives (PROVED, §2.3), so the path learns exactly
     "a heard char was repeated in the reply" — the copy gain — from ordinary dialogue pairs.
2. **Stop rule**: END wins a boundary iff its learned input ≥ x_R·(summed learned drive of all programs, copy
   included); x_R = breath reserve, ×0.9 per syllable, refilled at GO. Plus a per-reply "used" flag on each program
   (−0.5). This turns the 1.6–2.8-syllable replies of the max-rule into 4.7–6.2 syllables (corpus 6.2) with 0 loops.
3. **SIMULATED results, never-heard lines (1500 pairs, 20-way char-F1 discrimination; chance top-1 0.05, AUC 0.50),
   3 seeds** (different sound/muscle/ear/T wiring and corpus segment); mean ± sd over seeds:

   | dialogues | no copy route (learnability design) | **copy route, MAIN (γ_E 0.999)** | copy route, γ_E 0.9999 | **recommended: γ_E 0.9999 + latch set point (≤ 10 %)** | say-every-latched-char reference | ideal learned-copy rule (verifier, pairs) |
   |---|---|---|---|---|---|---|
   | 1k | top-1 0.057, AUC 0.504 | **0.136 ± 0.006, 0.587 ± 0.009** | 0.141 ± 0.006, 0.592 ± 0.007 | **0.151 ± 0.007, 0.583 ± 0.007** (parrot 0.41) | 0.197 / 0.628 | 0.21 / 0.60 |
   | 3k | 0.054 / 0.501 | **0.154 ± 0.006, 0.600 ± 0.003** | 0.168 ± 0.003, 0.610 ± 0.001 | **0.172 ± 0.007, 0.602 ± 0.004** (parrot 0.51) | 0.206 / 0.634 | 0.23 / 0.61 |
   | 10k | 0.058 / 0.507 | **0.152 ± 0.006, 0.602 ± 0.004** | 0.186 ± 0.008, 0.621 ± 0.004 | **0.202 ± 0.008, 0.616 ± 0.005** (parrot 0.59) | (latch recall 0.76–0.78) | 0.25 / 0.63 |

   ("latch set point" = latch only chars latched in ≤ 10 % of training pairs: the static form of a set point on the
   latch units; the frequent function words 你/我/了/的/是… are then rarely copied. SIMULATED as a static filter
   (eval_slowcopy_lsp_s*.out); the dynamic set-point version is ASSUMED to behave alike.)

   Copy rate (said chars that are in the heard line) 0.62–0.65 vs 0.05–0.07 for a shuffled line (human replies 0.15);
   false echoes (said, latched, not heard) 0.3–4.6 % of said chars; repeated-char and repeated-bigram loops 0.00;
   mean length 4.7–6.2 (true replies 6.2); stops by END 33–60 %, by "nothing strong enough" 40–66 %, by the
   15-syllable backstop ≤ 2.8 %. Verbatim echo is rare: reply == heard line 0.9–1.4 %, a ≥ 3-char run of the heard
   line in 5.7–6.5 % of replies (human replies 5.1 %). Stability: on fixed never-heard inputs the first syllable
   changes for 21–23 % of lines between 1k and 10k dialogues (no-copy design: 91–99 %).
4. **Honest caveats**: the route is mainly echo. It repeats about half of the heard line's distinct chars (parrot index
   0.49–0.61; human replies 0.15), though verbatim runs of ≥ 3 heard chars occur at the human rate (5.7–6.5 % vs
   5.1 %). Copy weights are frequency-weighted counts (你/我/了/的 0.5–0.7 vs topical 吃/回/爱 0.1–0.3): the one rule
   only counts copy events, and a true P(c in R | c in Q) normalisation is not locally available (§3.4). The 20-way
   metric rewards echo (parroting the whole line scores top-1 0.273), so both numbers are reported. Option with no
   metric cost: a latch set point (latch only chars latched in ≤ 10 % of pairs) lowers parroting to 0.35–0.49 (3
   seeds); with slow copy decay it is the best variant (table above).
5. **Memory (Q2, 2 seeds, 200 familiar probes × 6 exposures 100 dialogues apart, recall tested before each exposure
   and up to +10k dialogues after the last)**: the established thought→program path (P05) is a fast learner (first
   syllable 0.57–0.58 after 1 exposure, ≈ 0.8 after 3–5) but forgets (0.58–0.60 / 0.22–0.25 / 0.02–0.03 / 0.01–0.04
   at +<1k / +1–3k / +3–6k / +10k). Adding a **slow path from the same gated thought** (η/5, ~25-event budget window,
   **no decay**, weight ×3), with the copy route competing: 0.82 after 3 exposures, 0.90–0.91 after 5, and
   **0.76–0.81 / 0.54–0.58 / 0.46–0.50 / 0.39–0.40** at the same lags. Whole familiar replies: 20-way top-1 0.45–0.47
   (AUC 0.89) after 5 exposures, 0.24–0.29 (AUC 0.71–0.73) at +10k (MAIN: 0.15–0.16 / 0.59). A sparse
   hippocampus-like code for the fast path gives the same numbers; for the slow path it gives better one-shot recall.
6. **Combination (Q3)**: additive currents into the programs, one WTA per boundary. First boundary: fast + slow memory
   paths scaled ≈ 10× the copy path (otherwise echo outbids familiar recall: 0.78 → 0.25); costs −0.004 to −0.013
   top-1 on never-heard lines. Later boundaries: held-program chain + copy − 0.5·used. Stop: END iff s_END ≥
   0.9^n·Σ(learned drive). Cue margin PROVED from budgets (and measured); teacher END = clamp.
7. Decay should be counted per experience, not clock time (no idle loss); the slow path needs none (budget-bounded).

---------------------------------------------------------------------------------------------------------------------
## 1. The mechanism, unit by unit (adds to the learnability circuit; all currents from t−1 states)

Notation: b = boundary tick at which a program is released (R0); a released program Y_c is held b, b+1, b+2; the
partner's line occupies ticks 0..L−1; F = partner-talking unit (on from the 2nd tick of the line to L+2, vetoed by the
own-voice units V); GO at L+5; first boundary L+7.

| unit (per program c unless noted) | rule | role |
|---|---|---|
| Y'_c | Y'_c(t) = Y_c(t−1) | one-tick delayed copy (for the clamp timing) |
| Z_c (auditory mirror) | Z_c(t) = 1 iff [Σ_i FEAT0_i(t−1)·w_ic is the largest over Z and ≥ θ_Z = 0.75·B_Z = 1.5] OR [10·Y'_c(t−1) − 10·Y_c(t−1) ≥ 1] | detector while listening; clamped exactly at b+4 after the model itself said c (own voice fills the ear window at b+3) |
| L_c (latch) | L_c(t) = [Z_c(t−1) + F(t−1) + 2·L_c(t−1) − 3·Fon(t−1) − 3·Y_c(t−1) ≥ 2]; Fon = F ∧ ¬F2 (line onset) | holds "c was heard in the partner's current line"; own voice cannot latch (F vetoed); cleared by the next line's onset and by releasing c |
| E_c (replay) | E_c(t) = [L_c(t−1) + GO(t−1) + R1(t−1) − 2F(t−1) ≥ 2] | one-tick pulse on the R2 tick before every boundary (L+6, B+2, …) for every still-latched c |
| U_c (used) | U_c(t) = [Y_c(t−1) + U_c(t−1) − 2·GO(t−1) ≥ 1] | "already said in this reply"; −ρ = −0.5 into Y_c's score |
| x_R (breath, 1 per circuit) | ×(1 − U_b), U_b = 0.1, per released syllable; = 1 at GO | discounts the pooled drive that END must beat |
| END | wins at a boundary iff s_END ≥ x_R·Σ_y max(0, drive_y) | stop by learned "last syllable" vs everything that still wants to be said |

Learned paths (one rule: forward src(t−1) ∧ dst rising(t): w += η/(1 + w/w_ref); back src(t) ∧ dst(t−1):
w = max(w − η, 0); every 100 ticks: w ×= γ, then per-target budget over the whole column; w0 = 0):

| path | η | w_ref | γ /100 ticks | per-target budget | gate |
|---|---|---|---|---|---|
| FEAT0 → Z | 0.005 | 0.2 | 0.999 | B_Z = 2.0 | learns only while own voice is on (V) |
| E → Y (full matrix; only diagonal survives) | 0.03 | 0.2 | 0.999 | B_E = 1.5 | teacher-cued boundaries only (C11) |
| P → Y **fast** (first syllable; = learnability P→Y ×10) | 0.005 per P unit (1.0 per event) | 2.0 | 0.999 | 5.0 (binds at ~5 events) | C11 |
| P → Y **slow** (first syllable; §4) | 0.003 per P unit (0.6 per event) | 6.0 | 1 (no decay) | 15 (binds at ~25 events) | C11 |
| Y → Y (held program → next) | 0.01 | 0.2 | 0.999 | 2.0 | C11 |
| Y → END | 0.01 | 0.2 | 0.999 | none (equilibrium ≤ 1.97) | C11 |

All decays are counted per 100 *experienced* ticks (ticks with sound in the ear or a program held), not clock ticks.
Only ratios between paths matter: multiplying (η, w_ref, B) of every path and θ_rel by one factor multiplies all
weights by it and leaves every decision unchanged (PROVED from the rule's form). **Equivalent ÷10 form** (use it with
ARCH_v1's cue of 12/muscle): P fast (0.0005, 0.2, 0.5), P slow (0.0003, 0.6, 1.5), E→Y (0.003, 0.02, 0.15), Y→Y
(0.001, 0.02, 0.2), Y→END (0.001, 0.02, –), θ_rel 0.002 and used-flag penalty ρ 0.05 (every score-side constant
scales with the weights; U_b, the breath factor, is a ratio and stays 0.1) — then a program's learned input is ≤ 2.35.
§3 (never-heard) used P→Y fast at ×1 (0.0005, 0.2, 0.5) and no slow path ("MAIN"); §4.3 shows the memory scaling
("MEM") costs −0.004 to −0.013 top-1 there.

Boundary scores (training adds the cue 5·|muscles shared| and, at the END boundary, the END clamp):
- first boundary (L+7): s_y = P_fast→Y + P_slow→Y + E→Y (END cannot win: no program held, s_END = 0);
- later boundaries: s_y = Y→Y(Y_prev) + E→Y − 0.5·U_y; s_END = Y→END(Y_prev);
- END iff s_END ≥ x_R·Σ_y max(0, Y→Y + E→Y); else release argmax s_y if ≥ θ_rel = 0.02; else stop ("weak");
- backstop 15 syllables.

Timeline of one training turn (copy events marked):

| tick | partner / own voice | F | Z / L | E | Y | learning on the content paths |
|---|---|---|---|---|---|---|
| 3j+2 (j-th char's 3rd tick) | window = full char c | 1 | Z_c fires at 3j+3 (drive ≥ 1.5) → L_c on from 3j+4 | – | – | none (Z gate closed while listening) |
| L+3 | silence | 0 | latch holds | – | – | – |
| L+5 | | 0 | | | GO | – |
| L+6 | | 0 | | E_c for all latched c; P | | – |
| L+7 = B1 | | 0 | L_r1 cleared from B1+1 | | Y_r1 rises (cue) | E_c→Y_r1 +η for every latched c; P→Y_r1 +η |
| B1+1 | own voice r1 slices 0-1 | V | | | held | – |
| B1+2 | | V | | E_c for latched c ≠ r1 | held | **back term**: E_c→Y_r1 −η (c ≠ r1) ⇒ cross terms return to 0 |
| B1+3 = B2 | own r1 full in window | V | | | Y_r2 rises | E_c→Y_r2 +η; Y_r1→Y_r2 +η; Y_r2→Y_r1 −η |
| B1+4 | | V | Z_r1 clamped | | | FEAT0(B1+3)→Z_r1 +η (own voice of r1, full window) |
| B1+5 | | V | | | | back: FEAT0(B1+5)→Z_r1 −η (units shared with the next window) |
| … B_m+3 | | | | | END (teacher clamp) | Y_rm→END +η |

---------------------------------------------------------------------------------------------------------------------
## 2. Why it learns with the exact rule (incl. back term)

### 2.1 Echo map (Z) — SIMULATED + PROVED timing
- Timing (PROVED by the Stream code): own voice is heard one tick late; Y_c on b..b+2 ⇒ ear window holds c's slices
  0-1/2-3/4-5 at b+1/b+2/b+3, so FEAT0(b+3) is the full-char layer-0 code, which is a deterministic function of (c,
  voice) only (no previous char in the window). Clamping Z_c at b+4 makes it the forward source; the back term at b+5
  removes units also active in [c 4-5, next 0-3] (generic units), sharpening the template.
- The partner's c fills the window at the 3rd tick of c (k = 2); own vs partner layer-0 code Jaccard 0.89 (verifier).
- SIMULATED (zmap_diag.py, real ear replica, corpus order, 200 held-out partner lines, 1153 FEAT0 units):

  | setting | after 300 / 1k / 3k dialogues: heard char's Z is WTA winner at its clean tick | latch precision / recall at θ = 0.75·B |
  |---|---|---|
  | verifier's version (clamp b+3 = partial window, η 5e−4, 500 Z) | 0.36 / 0.50 / 0.59 | – |
  | clamp b+4 (full window), η 0.002 | 0.63 / 0.71 / 0.77 | – |
  | **clamp b+4, η 0.005, own-voice gate** | 0.81 / 0.83 / 0.87 | **0.97 / 0.73; 0.94 / 0.77; 0.99 / 0.78** (false latches 0.13 / 0.28 / 0.04 per line) |
  | same, no gate (learns from its own detections while listening) | 0.80 / 0.78 / 0.86 | 0.98/0.73; **0.79/0.75 (transient degradation)**; 0.99/0.83 |
  | η 0.01 (one event overshoots B between decay steps) | 0.86 / 0.80 / 0.88 | 0.73/0.84; 0.63/0.76; 0.89/0.81 |
- In the three full runs (fe_run.py, test pass on the 1500 never-heard lines): latch precision 0.93–0.996, recall
  0.76–0.81 at 1k, 3k and 10k dialogues, all seeds (fe_s*.log). Recall is limited by chars the model has said only rarely
  (unsaturated templates: true-drive p10 = 0.3–0.4·B).
- Gate: the own-voice gate is the corollary-discharge analogue of the C11 teacher gate; without it the map learns from
  its own detections (positive feedback) and loses precision transiently. The gate costs nothing.

### 2.2 Copy event timing — PROVED
For a latched c released at B: E_c on at B−1 (forward source), latch cleared from B+1 ⇒ E_c off at B+2 ⇒ no back term.
Net +η·(1/(1+w/w_ref)) per copy event, and nothing else ever changes E_c→Y_c (E_c fires only on R2 ticks; Y_c is on
at an R2 tick only while held, and then L_c has already been cleared).

### 2.3 Cross terms cancel exactly — PROVED (and checked numerically)
For latched c ≠ d with Y_d released at B: +η at B (E_c on at B−1, Y_d rises), then at B+2 E_c fires again (still
latched) while Y_d was on at B+1 ⇒ −η, floored at 0. Since the weight starts at 0 and the increment is exactly η when
w = 0, every off-diagonal weight returns to exactly 0 (also if a decay step falls in between: γη − η < 0 → floor 0).
The END boundary also has an R2 tick before it, so the last syllable is covered. Numerical check
(check_decide_fullbudget.py / check_decide2.py): brute-force full-matrix tick simulation of the rule vs the
decision-layer code on 400 random turns with decay steps inside replies: max off-diagonal 0.0, max deviation 7e−18
(the transient cross mass is counted in net.py's column budget if a decay step lands inside B..B+1; reproduced).
Consequence: one full E→Y matrix learns only the copy statistic; no extra wiring rule is needed.
**Caveat for ARCH_v1 (PROVED, also noted by the synthesizer):** the cancellation needs the target program to be ON at
B+1 (held). If the learned target is a one-tick selection pulse (ARCH_v1's Y), the back term never sees it, the
cross terms survive, and a full E→Y matrix would learn the bag association "heard c → say d" instead. There, wire E_c
only to its own program Y_c (part of the program's assembly, created at recruitment); the learned statistic is then
exactly the diagonal studied here (same forward events, no back events), so all results carry over.

### 2.4 Everything else
P→Y, Y→Y, Y→END are the learnability links (timing PROVED there). The decision layer used here (decide.py) is an exact
tick-level transcription (brute-force check above: P, Y→Y, END identical to 3e−9).

---------------------------------------------------------------------------------------------------------------------
## 3. Q1 — never-heard lines (SIMULATED)

Method: fe_run.py runs the real front end (ear layer 0 exact, FEAT0, T 4000/200 with hold and set point, the Z map,
the latch) tick by tick on the training timeline over 10k dialogues per seed, records per pair (Q, R, T(L+5), latched
chars), and at 1k/3k/10k dialogues runs a test pass over the 1500 never-heard pairs (pairs[−3000:][:1500]; learning
frozen). decide.py then learns the decision layer exactly (the cue always wins in training, so trajectories do not
depend on decision weights — PROVED by the cue margin, §5.1) and generates freely (decision paths frozen = C11).
Seeds: 0 = world seed 0, dialogues 0–10k; 1 = world seed 1, 10k–20k; 2 = world seed 2, 20k–30k.

### 3.1 Main table (eval_final_s*.out; MAIN = copy E3 γ 0.999 + summed-drive stop U_b 0.1 + used flag 0.5)
| seed | dialogues | no copy (learnability design) top-1 / AUC | MAIN top-1 / AUC | Δov | copy rate (shuffled) | parrot | len (p10/50/90) | stops END / weak / backstop | first syllable correct (shuffled) | γ_E 0.9999: top-1 / AUC / parrot |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 1k | 0.061 / 0.509 | 0.141 / 0.597 | +0.066 | 0.64 (0.06) | 0.53 | 5.2 (2/5/9) | 678 / 819 / 3 | 0.053 (0.041) | 0.146 / 0.600 / 0.56 |
| 1 | 1k | 0.054 / 0.504 | 0.129 / 0.581 | +0.056 | 0.62 (0.07) | 0.49 | 4.8 (2/5/8) | 870 / 630 / 0 | 0.052 (0.041) | 0.134 / 0.588 / 0.52 |
| 2 | 1k | 0.056 / 0.500 | 0.137 / 0.584 | +0.060 | 0.63 (0.07) | 0.49 | 4.7 (2/5/8) | 848 / 651 / 1 | 0.057 (0.045) | 0.143 / 0.588 / 0.52 |
| 0 | 3k | 0.049 / 0.500 | 0.152 / 0.597 | +0.063 | 0.65 (0.06) | 0.56 | 5.2 (2/5/8) | 812 / 685 / 3 | 0.061 (0.040) | 0.170 / 0.609 / 0.64 |
| 1 | 3k | 0.054 / 0.499 | 0.149 / 0.602 | +0.064 | 0.63 (0.06) | 0.59 | 5.7 (3/5/9) | 892 / 601 / 7 | 0.055 (0.039) | 0.165 / 0.610 / 0.67 |
| 2 | 3k | 0.058 / 0.503 | 0.160 / 0.601 | +0.063 | 0.65 (0.06) | 0.61 | 5.8 (3/6/10) | 715 / 784 / 1 | 0.055 (0.037) | 0.169 / 0.611 / 0.66 |
| 0 | 10k | 0.057 / 0.507 | 0.155 / 0.606 | +0.066 | 0.64 (0.06) | 0.61 | 6.2 (3/6/10) | 494 / 994 / 12 | 0.053 (0.039) | 0.195 / 0.626 / 0.72 |
| 1 | 10k | 0.054 / 0.503 | 0.155 / 0.601 | +0.069 | 0.65 (0.06) | 0.60 | 5.8 (2/5/10) | 766 / 692 / 42 | 0.057 (0.041) | 0.184 / 0.619 / 0.73 |
| 2 | 10k | 0.064 / 0.511 | 0.145 / 0.598 | +0.064 | 0.65 (0.06) | 0.59 | 5.7 (3/5/9) | 873 / 619 / 8 | 0.055 (0.041) | 0.179 / 0.619 / 0.72 |

Human replies of the same lines: copy rate 0.15 (shuffled 0.05), parrot 0.15, mean length 6.2.
Echo map / latch in these runs: precision 0.93–0.996, recall 0.76–0.81 at 1k, 3k and 10k (fe_s*.log).

### 3.2 Ablations (seed 0, 1k / 3k unless noted; seeds 1, 2 agree within ±0.01; 3-seed means where given)
| variant | top-1 | AUC | len | note |
|---|---|---|---|---|
| MAIN | 0.141 / 0.152 | 0.597 / 0.597 | 5.2 | |
| copy only at the first boundary | 0.087 / 0.089 | 0.530 / 0.540 | 3.9 | copies belong at every boundary (corpus: P(reply char k is heard) ≈ 0.13–0.15 for every k) |
| max rule instead of summed-drive stop (3 seeds, 1k/3k/10k) | 0.112 / 0.127 / 0.122 | 0.567 / 0.574 / 0.573 | 2.6–2.8 | END wins too early, latched chars are lost |
| used flag off (3 seeds, 1k/3k/10k) | 0.133 / 0.150 / 0.150 | 0.587 / 0.599 / 0.600 | 5.1–6.1 | 22 % of replies repeat a char, 7.5 % repeat a bigram (seed 0, 1k) |
| copy gain ×0.5 (3 seeds) | 0.117 / 0.132 / 0.126 | 0.574 / 0.584 / 0.581 | 4.9–6.1 | less echo (parrot 0.41–0.50) |
| copy gain ×2 | 0.146 / 0.167 | 0.602 / 0.606 | 4.9 | more echo (parrot 0.56–0.62) |
| E→Y γ 0.9999 (3 seeds, 1k/3k/10k) | 0.141 / 0.168 / 0.186 | 0.592 / 0.610 / 0.621 | 4.9–5.9 | keeps improving; parrot 0.53 / 0.66 / 0.72 |
| breath U_b 0.05 / 0.2 | 0.142 / 0.153; 0.142 / 0.151 | ≈ same | 5.4–5.7 / 4.7–4.8 | length knob |
| P→Y budget 2 instead of 0.5 | 0.130 / 0.142 | 0.596 / 0.595 | 5.4 | first syllable more often the prior (你) |
| latch set point: latch only chars latched in ≤ 10 % of pairs (3 seeds, 1k/3k/10k) | 0.142 / 0.150 / 0.160 | 0.578 / 0.590 / 0.594 | 3.8–5.4 | same top-1, parrot 0.35–0.49 instead of 0.49–0.61 (seed 0 ≤ 5 %: 0.141 / 0.149 / 0.162, parrot 0.35–0.42) |
| **E→Y γ 0.9999 + latch set point ≤ 10 % (recommended tuning; 3 seeds)** | **0.151 / 0.172 / 0.202** | **0.583 / 0.602 / 0.616** | 3.9–4.9 | best top-1 at every checkpoint; parrot 0.41 / 0.51 / 0.59 |
| recency latch: keep only the last 6 / 4 / 2 heard chars (seed 0, eval_recency_s0.out) | 0.138 / 0.154; 0.141 / 0.142; 0.112 / 0.112 | 0.594 / 0.597; 0.587 / 0.585; 0.557 / 0.553 | 4.8; 4.3; 3.6 | flat latch is right (corpus recency is weak); unlike the verifier's first-boundary-only bias, learned per-char weights at every boundary need no recency |
| Y set point β 0.05, rate 0.02 (prior subtraction; 3 seeds) | 0.149 / 0.174 / 0.176 | 0.596 / 0.612 / 0.614 | 3.9–4.8 | +0.02 top-1; shorter, more "weak" stops |

### 3.3 False echoes, parroting, loops, stability (MAIN)
- False echoes (said a latched char that was not in the heard line): 0.3–4.6 % of said chars (rises slightly with
  training as more programs exist); first-syllable false echo 0.3–3.7 %.
- First syllable is a copy in 86–93 % of replies (P→Y on never-heard lines is near-chance and copies outbid it); human
  replies start with a heard char 14.7 % of the time. §5.2 for familiar lines.
- Parroting: the model says 49–61 % of the heard line's distinct chars (human replies ≈ 15 %), as scattered single
  chars interleaved with chain chars: reply == heard line in 0.9–1.4 % of replies (short lines), a ≥ 3-char run of the
  heard line in 5.7–6.5 % (human replies 5.1 %), mean longest common run 1.35–1.43 chars (human 0.71)
  (eval_extra_s0.out). Each heard char can be said at most once per reply (latch cleared on use).
- Loops: 0.00 repeated chars / bigrams with the used flag; backstop (15) ≤ 0.8 % except seed 1 at 10k (2.8 %).
- Stability (eval_extra_s0.out; fixed never-heard inputs, weights at 1k vs later): first syllable changed for 21 / 21 /
  23 / 22 / 21 % of lines at 1.5k / 2k / 3k / 5k / 10k dialogues (no-copy design: 91–99 %), whole reply changed
  85–94 % (tails follow the still-learning Y→Y chain).

### 3.4 Normalisation (the "PMI" question) — what the rule can and cannot do
- The copy weight the rule learns is a decayed, concave **count** of copy events (2.2): frequent chars that are
  heard and repeated often (你 0.53–0.73, 我, 了, 的, 是) outweigh topical chars (吃 0.17–0.31, 回 0.12–0.24, 爱 ~0.1).
- κ_c = P(c∈R | c∈Q) would need depression on "heard but not repeated", but the rule depresses only when the target
  fired before the source; an unsaid Y_c never fires, so no local depression exists (PROVED by the rule's form).
- Corpus facts (stats_copy.out): κ is 0.18–0.31 for almost all chars (你 0.235 vs prior 0.209; 吃 0.313 vs 0.033),
  recency matters little (P(copy) 0.145/0.156/0.146 for the last three heard chars vs 0.105 at position ≥ 10), and
  copies are spread evenly over reply positions.
- On the 20-way metric the count statistic is not worse than κ: ideal "copy every char copied ≥1 time" scores top-1
  0.19/0.22/0.25 at 1k/3k/10k pairs vs 0.21/0.23/0.25 for the ideal gain rule. Normalisation makes replies less
  echo-y and more topical rather than more discriminative.
- Existing-mechanism substitutes (both tested): a **latch set point** (an IDF filter: chars latched in > 10 % of pairs
  are not latched) trades ≈ 0.01 AUC for +0.004…+0.016 top-1 and ≈ 0.1 less parroting — with slow copy decay it is
  the best variant (§0 table); a **Y set point** (prior subtraction) adds ≈ +0.02 top-1 but shortens replies.

---------------------------------------------------------------------------------------------------------------------
## 4. Q2 — memory: fast + slow first-syllable learners (SIMULATED, 2 seeds)

### 4.1 Protocol (fe_probe.py, eval_mem.py)
- Training stream = corpus dialogues (seed 0: dialogues 0–14.6k, world seed 0; seed 1: 15k–29.6k, world seed 1) with
  200 probe pairs (from dialogues 30000–30895, in neither stream) inserted 6 times each, 100 dialogues apart, starting
  uniformly in [0, 4000) (probes ≈ 21 % of the material while taught). Exact front end (ear layer 0, T, Z map, latch).
- **Prequential familiar recall**: before every exposure k = 0..5 the probe is heard in a fresh context and answered
  (lag ≈ 100 dialogues since the previous exposure, ≥ 20 ✓; k = 0 = never heard = the reference for the same lines).
- **Retention**: all probes tested at 4.6k / 5.6k / 7.6k / 14.6k dialogues, binned by the lag since their last
  exposure: +<1k, +1–3k, +3–6k, +10k.
- Learning stays on; the decision paths are C11-gated, so tests do not change weights.
- Sources: T (4000 units, 200 active = 5 %); S = a sparser held code (8000 units, 40 active = 0.5 %, same FEAT0 input,
  same one-tick self-hold; a "dentate-like" pattern-separated code). Paths P→Y, exact rule, w0 0; per-event
  increment = η·K: P05 = T, 0.1/event, budget 0.5 (binds at ~5 events), γ 0.999; Tslow = T, 0.02/event, budget 0.5
  (~25 events), **no decay**; Sfast = S, 1.0/event, budget 1.0 (1 event), γ 0.99; Smid = S like P05; Sslow = S like
  Tslow. "×w" = the path's (η, B, w_ref) multiplied by w (exactly scales its weights; PROVED from the rule's form).

### 4.2 Results — first syllable, memory paths alone (seed 0 / seed 1; 200 probes per exposure column; retention bins pool all checkpoints: n ≈ 50 / 195 / 300 / 255)
| path | k=1 | k=2 | k=3 | k=4 | k=5 | +<1k | +1–3k | +3–6k | +10k |
|---|---|---|---|---|---|---|---|---|---|
| P05 (current design) | 0.58 / 0.57 | 0.69 / 0.72 | 0.77 / 0.79 | 0.79 / 0.80 | 0.78 / 0.81 | 0.60 / 0.58 | 0.22 / 0.25 | **0.03 / 0.02** | **0.04 / 0.01** |
| Tslow | 0.26 / 0.27 | 0.47 / 0.41 | 0.72 / 0.72 | 0.82 / 0.81 | 0.89 / 0.89 | 0.73 / 0.78 | 0.58 / 0.60 | **0.53 / 0.48** | **0.41 / 0.40** |
| P05 + Tslow ×3 | 0.38 / 0.32 | 0.72 / 0.72 | 0.84 / 0.84 | 0.90 / 0.89 | 0.93 / 0.92 | 0.79 / 0.76 | 0.58 / 0.57 | 0.50 / 0.46 | 0.38 / 0.37 |
| P05 + Sslow ×3 (seed 0) | 0.57 | 0.69 | 0.78 | 0.83 | 0.87 | 0.77 | 0.58 | 0.48 | 0.39 |
| (4.6k-checkpoint evaluation only:) Sfast (one-shot, recency) | 0.47 / 0.41 | 0.46 / 0.42 | 0.43 / 0.41 | 0.42 / 0.43 | 0.42 / 0.41 | 0.08 / 0.08 | 0.00 / 0.01 | 0.00 / 0.00 | |
| Smid | 0.54 / 0.56 | 0.66 / 0.67 | 0.73 / 0.71 | 0.75 / 0.74 | 0.76 / 0.77 | 0.56 / 0.56 | 0.27 / 0.23 | 0.04 / 0.00 | |
| Sslow | 0.14 / 0.10 | 0.60 / 0.62 | 0.69 / 0.69 | 0.78 / 0.76 | 0.82 / 0.82 | 0.56 / 0.76 | 0.61 / 0.55 | 0.46 / 0.46 | |
| Smid + Tslow ×3 (hippocampus fast, cortex slow) | 0.39 / 0.33 | 0.72 / 0.70 | 0.81 / 0.83 | 0.85 / 0.88 | 0.90 / 0.90 | 0.75 / 0.78 | 0.61 / 0.53 | 0.39 / 0.44 | |
| Sfast + Tslow (equal weights; seed 0) | 0.52 | 0.56 | 0.57 | 0.55 | 0.60 | 0.33 | 0.26 | 0.18 | |
Never heard (k = 0): 0.00–0.03 for every path. Tslow2 (T, 0.04/event, γ 0.9999; seed 0, 4.6k): 0.34/0.67/0.75/0.82/
0.84, retention 0.73/0.60/0.45 ≈ Tslow.

### 4.3 Results — what the model actually says first (copy route competing) and whole replies
| first boundary | k=1 | k=2 | k=3 | k=4 | k=5 | +<1k | +1–3k | +3–6k | +10k |
|---|---|---|---|---|---|---|---|---|---|
| P05 ×1 + copy (MAIN as in §3) | 0.14 / 0.15 | 0.21 / 0.17 | 0.22 / 0.21 | 0.27 / 0.24 | 0.25 / 0.27 | 0.19 / 0.06 | 0.09 / 0.05 | 0.06 / 0.05 | 0.06 / 0.04 |
| P05 ×10 + copy | 0.47 / 0.40 | 0.70 / 0.68 | 0.78 / 0.77 | 0.79 / 0.77 | 0.79 / 0.79 | 0.62 / 0.56 | 0.19 / 0.19 | 0.05 / 0.04 | 0.05 / 0.04 |
| **MEM: P05 ×10 + Tslow ×30 + copy** | 0.34 / 0.25 | 0.69 / 0.70 | **0.82 / 0.82** | 0.89 / 0.88 | **0.91 / 0.90** | **0.81 / 0.76** | **0.58 / 0.54** | **0.50 / 0.46** | **0.40 / 0.39** |
| P05 ×10 + Sslow ×30 + copy (seed 0) | 0.54 | 0.67 | 0.78 | 0.83 | 0.85 | 0.75 | 0.59 | 0.49 | 0.41 |
| (4.6k only:) P05 ×3 + copy | 0.26 / 0.20 | 0.37 / 0.34 | 0.54 / 0.47 | 0.62 / 0.56 | 0.66 / 0.60 | 0.44 / 0.30 | 0.11 / 0.07 | 0.07 / 0.04 | |
| (4.6k only:) Smid ×10 + Tslow ×30 + copy | 0.34 / 0.26 | 0.68 / 0.69 | 0.79 / 0.81 | 0.83 / 0.86 | 0.89 / 0.88 | 0.77 / 0.78 | 0.64 / 0.49 | 0.45 / 0.46 | |

Whole free replies of the familiar probes (MAIN boundary rule, 20-way among the 200 probe replies, top-1 / AUC;
seed 0; seed 1):
| readout | k=0 (never heard) | k=1 | k=3 | k=5 | +<1k | +1–3k | +3–6k | +10k |
|---|---|---|---|---|---|---|---|---|
| MAIN (P05 ×1) | 0.12/0.59; 0.11/0.57 | 0.20/0.62; 0.20/0.64 | 0.28/0.69; 0.23/0.68 | 0.33/0.70; 0.31/0.73 | 0.25/0.65; 0.22/0.66 | 0.11/0.60; 0.16/0.63 | 0.12/0.60; 0.15/0.58 | 0.15/0.59; 0.16/0.59 |
| **MEM** | 0.10/0.58; 0.09/0.57 | 0.23/0.68; 0.17/0.66 | 0.40/0.86; 0.40/0.85 | **0.47/0.89; 0.45/0.89** | 0.33/0.84; 0.33/0.82 | 0.32/0.79; 0.31/0.79 | 0.28/0.76; 0.28/0.73 | **0.29/0.73; 0.24/0.71** |
Whole-reply top-1 saturates near 0.45 although the first syllable is right 90 % of the time: later syllables come from
the generic chain and the copy route, not from a memory of that reply (no line-specific source reaches later
boundaries; read-once lemma). Remembering whole replies would need a line-specific source at every boundary (ASSUMED
future work).

Cost on the 1500 never-heard lines (eval_pmem_nh_s*.out, 3 seeds): MEM top-1 0.127 / 0.141 / 0.148 vs MAIN 0.136 /
0.154 / 0.152 at 1k / 3k / 10k (−0.009 / −0.013 / −0.004); AUC 0.585 / 0.596 / 0.600 vs 0.587 / 0.600 / 0.602; the
first syllable is a copy in 31–40 % of replies instead of 86–93 %; length 5.3–6.8.

### 4.4 Conclusions for Q2
1. The established first-syllable path (P05: T, binding at ~5 events) is a **fast learner**: 0.58 after one exposure,
   ≈ 0.8 after 3–5, but **volatile**: 0.6 / 0.2 / 0.03 / 0.02 at +<1k / +1–3k / +3–6k / +10k dialogues. Two causes (PROVED from the
   rule): the binding budget keeps only the last ~5 events per program (a frequent first syllable is overwritten within
   tens of dialogues), and γ 0.999 erases rare ones (τ ≈ 2k pairs).
2. A **slow learner** = a second path from the same gated thought P with η/5, a budget window of ~25 events and **no
   decay** keeps 0.48–0.53 at +3–6k and 0.40–0.41 at +10k (2 seeds) and reaches 0.89 after 5 exposures, but needs
   ~3 exposures.
3. **Combination** = plain sum of currents, slow path weighted ×3 (its familiar signal must exceed the fast path's
   noise floor; equal weights let the fast path's noise win — Sfast+Tslow row). Result with the copy route
   competing: 0.82 from the 3rd exposure, 0.90–0.91 at the 5th; 0.76–0.81 / 0.54–0.58 / 0.46–0.50 / 0.39–0.40 at
   +<1k / +1–3k / +3–6k / +10k (P05 alone: 0.58–0.60 / 0.22–0.25 / 0.02–0.03 / 0.01–0.04).
   - Whether the fast learner sits on T (P05) or on a sparse hippocampus-like code S (Smid) makes no difference
     (rows agree within 0.05). So no new area is needed; if the designer prefers "hippocampus fast, cortex slow", use
     S for the fast path — same numbers.
   - The slow path on S (Sslow) keeps one-shot recall (0.51–0.57 after 1 exposure) at a small cost later (0.85–0.89 at
     k = 5): the option if one-exposure recall matters.
4. **Copy vs memory at the first boundary**: at equal scale the echo outbids familiar recall (0.78 → 0.25). The memory
   paths must be ≈ 10× the copy path (equivalently copy, chain and END ×0.1). Cost on never-heard lines −0.004 to
   −0.013 top-1 (3 seeds, 1k–10k), AUC −0.002 to −0.004.
5. **Decay per experience, not clock**: fast paths keep γ 0.999 per 100 *experienced* ticks; the slow path needs no
   decay at all (budget-bounded, PROVED; forgetting by interference only) — γ 0.9999 behaves the same (Tslow2).
   Clocking decay by experience also removes the idle-time loss of S1→GO (verifier M10) and keeps contents across
   idle periods; during continuous training it equals the clock decay used in all sims.
6. Frequent first syllables remain the weak spot (any per-program budget forgets them first). The framework's
   splitting of constantly-used neurons (several units per frequent program, each with its own budget) would extend
   their capacity (ASSUMED, not simulated).

## 5. Q3 — combination at a boundary

### 5.1 Cue margin and END in training — PROVED + SIMULATED
- Cued program scores 20 (+ learned); a program sharing 3 of 4 muscles 15 + learned. Learned input of any program is
  bounded by the budgets: P→Y 0.5 (+ slow path budget) + Y→Y 2.0 + E→Y 1.5 = 4.0 < 5 ⇒ the cue always wins (PROVED).
  Measured (margin_sp.py, seed 0, 3k): best non-cued learned score max 0.76, p99 0.69.
- END at a cued boundary: s_END ≤ 1.97 (unbudgeted equilibrium, learnability verifier) vs x_R·Σ ≥ 0.9^12·20 = 5.6
  (corpus replies ≤ 12 chars) ⇒ END never wins a cued boundary (PROVED).
- END boundary: x_R·Σ(learned drive) measured max 3.02, p99 1.71, median 0.36 ⇒ an END push of +10 wins 100 % (seed 0,
  3k). Not bounded by any budget (row sums of Y→Y and several latched copies can add up), so the teacher END signal
  should **clamp END and inhibit Y** at that boundary (PROVED then) rather than add +10.

### 5.2 First boundary: thought vs copy (SIMULATED, §4.3)
- Rule: s_y(B1) = P_fast→Y + P_slow→Y + E→Y; WTA; END cannot win B1. The memory paths are scaled ×10 relative to the
  copy path (fast: η 0.005 per T unit = 1.0 per event, budget 5.0, w_ref 2.0, γ 0.999; slow: η 0.003 per unit = 0.6
  per event, budget 15, w_ref 6.0, no decay; copy E→Y: η 0.03, cap 1.5). Equivalent and cue-friendlier: keep P at
  (0.0005, 0.5, 0.2) and (0.0003, 1.5, 0.6) and divide copy, chain and END paths (η, B, w_ref), θ_rel and the
  used-flag penalty ρ by 10 (U_b unchanged).
- Effect: familiar lines get their remembered first syllable (0.82–0.91 from the 3rd exposure); never-heard lines
  still start with a copy 35–40 % of the time, otherwise with the thought's best guess, and copies fill the later
  boundaries (never-heard top-1 −0.004 to −0.013, 3 seeds).
- Cue margin with these scales: the teacher cue must exceed the sum of all budgets of a program (fast 5 + slow 15 +
  chain 2 + copy 1.5 = 23.5 → cue ≥ 25 per muscle, or use the ÷10 form, where the sum is 2.35 and the current cues
  suffice). PROVED by the budget bound.

### 5.3 B6
- MAIN: mean length 4.7–6.2 (p10/50/90 2–3 / 5–6 / 8–10) vs 6.2 for the true replies of the same lines; stops END
  33–60 %, weak 40–66 %, backstop ≤ 0.8 % (2.8 % for seed 1 at 10k); 0 repeated chars / bigrams. Max-rule
  (learnability default): 1.6–2.8 syllables.
- MEM scaling (b6_mem.out, 10k, 3 seeds): length 6.2–6.8, backstop 0.7–3.2 %, 0 loops.
- Without the used flag: 22 % of replies repeat a char, 7.5 % a bigram (e.g. 我是的我是的).

---------------------------------------------------------------------------------------------------------------------
## 6. Q4 — targets for the first GPU milestone (derived from the 3-seed sims, with margins)

How derived: the copy route depends only on parts that are exact in my sims (ear layer 0, FEAT0, the own-voice timing
of the Stream, the decision rule) or structural (F window, latch, R2 clock, WTA). The GPU differs in T size (10k/500:
affects P→Y only), in ARCH_v1's program timing (one-tick Y pulse + Hp hold + Hc late copy: same learned statistics,
E wired diagonally, §2.3), cue 12/muscle, first boundary at L+6. Targets = 3-seed mean minus ≈ 2–3 seed-sd and ≈ 10 %
for integration differences. Never-heard = the 1500 pairs pairs[−3000:][:1500] (or any ≥ 500 lines never trained).

| check | at 1k dialogues | at 3k dialogues | sim (3 seeds) 1k / 3k |
|---|---|---|---|
| echo map: heard char's Z = WTA winner at its clean tick (200 partner lines) | ≥ 0.75 | ≥ 0.80 | 0.83 / 0.87 (seed 0) |
| latch precision / recall (vs heard chars that have a program) | ≥ 0.90 / ≥ 0.70 | same | 0.93–0.996 / 0.78–0.81 |
| never-heard 20-way top-1 (chance 0.05) | **≥ 0.12** | **≥ 0.14** | 0.136 ± 0.006 / 0.154 ± 0.006 |
| never-heard AUC (chance 0.50) | **≥ 0.57** | **≥ 0.585** | 0.587 ± 0.009 / 0.600 ± 0.003 |
| copy rate (said chars in heard line) vs shuffled line | ≥ 0.50 vs ≤ 0.10 | same | 0.62–0.65 vs 0.06–0.07 |
| same model with E→Y disabled (falsification control) | top-1 ≤ 0.07, AUC ≤ 0.52 | same | 0.054–0.064 / 0.50–0.51 |
| mean reply length; backstop share; repeated-bigram replies | 4.5–6.5; ≤ 3 %; ≤ 1 % | same | 4.7–6.2; ≤ 2.8 %; 0 |
| false echoes (said, latched, not heard) | ≤ 5 % of said chars | same | 0.3–4.6 % |
| familiar probes (§4 protocol: 100–200 pairs, 6 exposures 100 dialogues apart, tested before each exposure), first syllable with the copy route competing | ≥ 0.70 after 3 exposures, ≥ 0.80 after 5 | retention ≥ 0.65 at +<1k, ≥ 0.45 at +1–3k (later: ≥ 0.35 at +3–6k, ≥ 0.30 at +10k) | 0.82 / 0.90–0.91; 0.76–0.81 / 0.54–0.58 / 0.46–0.50 / 0.39–0.40 (2 seeds) |
| familiar probes, whole replies, 20-way among the probe replies (after 5 exposures) | top-1 ≥ 0.35, AUC ≥ 0.80 | | 0.45–0.47 / 0.89 |
| same with the slow path disabled (control: P05 only) | retention at +3–6k ≤ 0.10 | | 0.00–0.02 |
| (recommended tuning: γ_E 0.9999 + latch set point) never-heard top-1 | ≥ 0.13 | ≥ 0.15; ≥ 0.18 at 10k | 0.151 / 0.172 / 0.202 |

If a GPU run misses the map/latch checks, look first at the clamp tick (b+4 vs b+3 drops the map to the verifier's
0.36/0.50/0.59) and at F's window (the last char is detected at L, so F must still be on at L).

### 6.1 Mapping onto ARCH_v1 (arch/synth/ARCH_v1.md §1.8 SLOT)
| content-route element | ARCH_v1 form | why it is equivalent |
|---|---|---|
| Z clamp at b+4 | program's own delay line clamps Z_c 4 ticks after the program pulse (own voice full in the window at b+3) | same ear/Stream timing; ARCH_v1's Hp holds b..b+2 like my held Y |
| Z plasticity gate | ΣCD > 0 (own voice can be in the ear) | covers b+4 and b+5 of every syllable, closed while listening |
| latch set / reset | Z_c ∧ F; reset by the partner's next line onset; cleared from B+1 after Y_c's pulse | F reads the ear input window, so it is still on at L when the last char is detected |
| replay E_c | latch ∧ R2 (R2 = B−1 in ARCH_v1) | one-tick source, off at B+1 |
| E→Y | **diagonal only** (E_c → Y_c), η 0.03, cap 1.5 | ARCH_v1's Y is a one-tick pulse: cross terms would not cancel (§2.3); diagonal = same statistic |
| used flag U_c | −0.5 on Y_c until GO (−0.05 in the ÷10 form) | identical |
| stop | END iff s_END ≥ 0.9^n·Σ max(0, learned drive) | identical; teacher END = clamp (ARCH_v1: +100) |
| cue margin | cue 12/muscle: exact ≥ 48 − 0.5 vs 36 + 0.5 + 2.0 + C_slot(7) = 45.5 | E→Y ≤ 1.5 ≤ C_slot; with the memory paths use the ÷10 form (slot: slow P→Y 1.5 + copy 0.15 ≤ 7) |

---------------------------------------------------------------------------------------------------------------------
## 7. Still ASSUMED (not simulated here)

1. One integrated circuit: front end and decision layer were simulated separately (decision learning offline from
   recorded inputs; exact because the cue always wins in training, §5.1). The synthesizer's integrated toy
   (arch/synth/echo_slot.py) is the check.
2. Program recruitment by a local rule (my sims allocate Y_c/Z_c/L_c/E_c/U_c on the first cue of c, as the panel's
   toys do); unseen chars cannot be copied.
3. Z as "top-1 with absolute threshold" (a WTA group with lateral inhibition); the own-voice plasticity gate taken as
   the reply window (in the circuit: V / CD units on).
4. A single partner voice; other speakers, noise and overlapping speech untested (Z generalises from own to partner
   voice at layer-0 Jaccard 0.89; a third voice is untested).
5. Whether echo-heavy replies are acceptable as "related content" (B4) is the designer's call: by the panel's metric
   they are clearly content (and verbatim runs are human-like), but the model repeats ~4× more heard chars than humans.
   The latch set point (≤ 10 %) keeps top-1 and lowers parroting (0.35–0.49, 3 seeds).
6. Free speech with learning on (deployment): decision paths are frozen by the C11 gate, the Z map keeps learning
   from the own voice (harmless: own voice is its teacher); long free use untested.
7. Decay per experience (recommended in §4) vs the clock decay used in the sims: identical during continuous dialogue
   training, different only across idle time.

---------------------------------------------------------------------------------------------------------------------
## 8. Files (all in arch/content/)
- metric.py — corpus, 20-way metric (identical to the verifier's ceiling_verify.py on its reference rows).
- stats_copy.py → stats_copy.out — corpus copy facts and idealised copy policies.
- fe.py — front end: exact ear layer 0 (verifier's ear0.py), FEAT0, T, Z echo map, latch, training/test protocol.
- zmap_diag.py → zmap_e*.out — echo-map learning variants (clamp tick, η, own-voice gate).
- fe_run.py → fe_s{0,1,2}.pkl/.log — recorded front-end inputs, 10k dialogues per seed, test passes at 1k/3k/10k.
- decide.py — decision layer, exact rule (P→Y variants on T or S, Y→Y, Y→END, E→Y with the transient cross mass),
  free generation with the stop-rule / used-flag / set-point options; check_decide2.py, check_decide_fullbudget.py —
  brute-force full-matrix verification.
- eval_copy.py, eval_copy2.py, eval_copy3.py → eval_copy*_s*.out (sweeps); eval_final.py → eval_final_s{0,1,2}.out
  (main never-heard table); eval_extra.py → eval_extra_s0.out (stability, latch set point, echo structure);
  eval_recency.py → eval_recency_s0.out; eval_slowcopy_lsp.py → eval_slowcopy_lsp_s*.out (recommended tuning);
  eval_pgain.py → eval_pgain_s*.out and eval_pmem_nh.py → eval_pmem_nh_s*.out (never-heard cost of the memory
  scaling); margin_sp.py → margin_sp_s0_3k.out (training margins, Y set point); b6_mem.py → b6_mem.out.
- fe_probe.py → fe_probe_s{0,1}.pkl/.log (familiar-probe schedule, T and S codes); eval_mem.py → eval_mem_s*_{A,B,C,D,E}*.out.
