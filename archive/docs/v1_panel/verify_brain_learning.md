# Verification of "Listen → Latch → Launch" (arch/brain/proposal.md), LEARNING lens
(Saved by the coordinator from the verifier's final reply; the harness blocked the verifier's own write.)

Folder: this one. Method: proposal toy (arch/brain/circuit.py) imported read-only; circuit_learnpfc.py reproduces it
exactly with --pfc 0 (same channel counts 188…486 as E5; replay matches). Labels: PROVED (exact computation from net.py
or exact corpus counts), SIMULATED (circuit runs; seed 0; 20 probes; "(pass2 n)" = second test pass with different
preceding context, ±2/20 = context noise), ASSUMED. Caveat: machine shared and overloaded → single-seed circuit runs.

## Verdict: 0 FATAL, 4 SERIOUS, 6 MINOR
Holds (exact): timing law; thal→channel = +η per latched unit per GO; launch with no teacher push; every weight bounded.
Fails: size/robustness of what is learned. E5 is a "memorize, then test at once" regime (probes 44% of training) with
a hippocampus that keeps learning during the test. Realistically: familiar recall ≈ frequency level; learned stop =
recency effect; plan drifts once the PFC learns. Two simple changes restore recall (checked in the full circuit):
thal→channel budget 0.2, and hippocampus plastic at test. Stop and drift fixes not checked. Corpus-ceiling claim FALSE.

## SERIOUS
### S1. Familiar recall (B4) far below claim at budget 1.0; E5 number depends on hippocampus learning during the test
Claim: "familiar dialogues get the right first syllable 14–16/20"; "8 exposures give 14–16/20" (§0, §6.3).
1. E5 reproduced, same model, different test conditions (e5v_s0.txt):
| test condition | first syllable | overlap matched/mismatched | 20-way AUC |
|---|---|---|---|
| E5 protocol: guided pass of probes 0–9 just before; hip learning | 15/20 | .50/.10 | .79 |
| no guided pass; hip learning | 12/20 | .41/.10 | .72 |
| hip frozen (= real check()) | 9/20 | .36/.11 | .67 |
| +50 ordinary dialogues, frozen | 9/20 | .33/.17 | .64 |
| +200 ordinary dialogues, frozen | 10/20 | .22/.18 | .62 |
Replication two passes (e5fix_s0.txt): hip learning 14 (14), frozen 9 (8); +200: 12 (11), frozen 8 (7). A→B frozen 12 (10).
2. Frequency control (PROVED by counts): with hip frozen, correct probes are those whose first syllable ranks 1–4 among
E5's training first syllables (你 37, 我 26, 哪 18, 对 17 events — made frequent by the probes). Probes whose first
syllable appears only in their own 8 presentations: 1/9 (2/9 A→B). Plan mostly chooses among frequent channels.
3. Realistic share (scale_s0.txt; 150 fresh + 20 probes per round, nP 4096, hip frozen): after 2/4/6 exposures
(340/680/1020 dial.) 2/5/8 of 20 (pass2 1/5/6); AUC .48/.52/.57; +50 dial. 6 (7), AUC .59. Every hit a frequency hit
(我/你/是/不/对); outputs "我了", "你的". Plan itself fine: decoding at test 1.00 (same line Jaccard .42–.45 vs .05–.06).
4. Readout arithmetic (PROVED): I_c(P) = Σ_i a_i·|P∩P_i| over c's training plans. Binding channels (mass ≥ B, i.e.
≥ B/(η|P|) ≈ 23 GO events at B=1) score B·⟨|P∩P_i|/|P_i|⟩ (prototype score); others η·Σγ^age·|P∩P_i| (count score).
A familiar line with a rare first syllable beats binding channels only if η·k·o_s > B·o_d/|P|; with E4b (o_s 10.8,
o_d 3.9, |P| 44) k* ≈ 8.3 recent exposures within the ~2000-dialogue decay window. E5 used exactly 8. The proposal's
budget rule (B ≈ 0.3 × unbudgeted top mass ≈ 2.6–4.7) moves the wrong way.
5. Impostors / train-test mismatch (replay2.py): test plan overlaps the probe's own training plans by 9–10 of ~45 units,
an average other line ~4, the best of a few hundred others 11–12. Hip plastic at test raises overlap 10.0 → 12.7 and
recall 9 → 14 (ASSUMED mechanism: hip learns within the line, η 0.5).
FIX (checked): thal→channel budget B ≈ 0.2–0.5 instead of 1.0. Offline replay of thal→channel (reproduces the full
circuit exactly in 9/13 test points, within 2 in the rest):
| regime | B=1 | 0.5 | 0.2 | 0.1 | 0.05 |
|---|---|---|---|---|---|
| scale, 2 exposures | 2 | 1 | 6 | 8 | 3 |
| scale, 4 exposures | 5 | 5 | 13 | 10 | 4 |
| scale, 6 exposures | 7 | 12 | 13 | 11 | 3 |
| E5, hip frozen | 9 | – | 16 | – | – |
| A→B: A after A / A after B / B after B | 12/5/7 | 15/10/15 | 15/8/11 | 11/7/12 | – |
Full circuit B=0.2 (scalefix_s0.txt, realistic): 2 exposures 7/20 (8), AUC .64 (vs 2/20, .48 at B=1); 4 exposures
13/20 (12), AUC .76 (vs 5/20, .52). Keep hippocampus plastic at test (fast learner by design; check() should freeze only
slow paths): B=0.2 → 12/20 (12), AUC .74 after 2; 16/20 (15), AUC .83 after 4. Never-heard lines 0–2/20 for every B.

### S2. Learned stop (B6) is a recency effect; in realistic training replies loop to the breath limit
Claim: "40/40 learned stops, 0 by breath" (§4.3, §5).
- Why (PROVED): ear→P_rest full fan-in: one event ≈ 600 sources × 1e−4 × reserve 0.6 ≈ 0.036 vs budget 0.1; each 100
  ticks (~2 dial.) the budget keeps ≈0.58 of old mass → half-life ≈ 3 dialogues. Marker encodes "penultimate syllables of
  the last ~3 training dialogues"; in E5 the test directly followed the probes' last presentations.
- SIMULATED: same E5 model +200 ordinary dial.: 16/20 end by BREATH (12 syllables), 4/20 by marker ("我是的我是的…").
  PFC-learning run 200 dial. later: mean length 10.9, loops. A→B after B: 5.6–8.8.
- Parameter fix fails: η_rest scaled to fan-in (2.5e−6, half-life ≈78 dial.) → marker never fires at ρ=1: 20/20 BREATH at
  delay 0 and +200 (e5fix_s0.txt, scalefix_s0.txt); chains cycle ("有的小有的小", "是不是不是", "我了我了").
- Structural reason (PROVED): the timing law forces the marker to rise WITH the last syllable → learned from the ear code
  of the penultimate syllable and decided before the last is chosen; Chinese replies mark the end on the last syllable
  (了 啊 吧 呢 哈).
FIX (not checked): learn the stop from the motor channel that just ended — a stop unit rising at the boundary after the
last syllable, channel→stop learned (channel on at t0−1, off from t0 = clean source; generic ear features not sources).
Plus within-reply repeat suppression (reply-level trace) or a content-free length hazard (syllable counter → stop) so
cycles end near corpus mean length (5.9).

### S3. Plan drift once plan layers learn (proposal's [A] for B2)
SIMULATED (pfc_s0.txt; circuit_learnpfc.py --pfc 1 adds pfc→pfc, pfc→think, think→think, think→pfc learning with
budgets + set points): plan decoding at test 0.95 → 0.85 → 0.70 at 360 → 410 → 560 dial.; same-line Jaccard .12 → .09 →
.07 (vs 1.00 and .42–.45 untrained PFC). Recall at B=1: 4 (6), 5 (8), 5 (5). Test plan's overlap with probe's training
plans 4.2 → ~3.3 (impostor max ≈6). Replay at B=0.2: 14 → 11 → 8. Teacher fallback pushes 0 (B3 unaffected).
FIX (not checked): plan content from a non-drifting source — hippocampus → thalamic latch directly — or thinking layer
learns ~10× slower than thal→channel; re-test at ≥2,000 dialogues.

### S4. Never-heard content: no route in the design; the stated reason (corpus ceiling) is false
Claim: §6.5 "Δ ≤ +0.02 for any statistical learner up to 40k"; "no design can show content on never-heard lines".
Only the first syllable depends on what was heard (plan silenced at GO; later syllables from own-voice chain). Even so the
corpus allows Δov +0.06 to +0.10. Readout cannot express copying even with ideal codes (t3b_fix.py, N=1000, consistent
per-char bag codes): per-target budget → held-out first syllable 0.050 vs 0.047, Δov +0.002±0.009; + per-source budget
→ 0.093 vs 0.067, Δov +0.010±0.010. Side bug (PROVED): T3b's bag plan uses rng.choice(n, per) with per depending on line
length; numpy choice is not prefix-consistent → same char gets unrelated codes in lines of different length; T3b variant
k invalid.
FIX (not checked): make the heard line available at every reply-syllable boundary, e.g. beat-paced hippocampal replay of
it as 1-tick pulses before each boundary; the forward rule then learns "replayed c → channel c" from ordinary pairs
whenever a reply repeats a heard char (14.6% of reply chars, 3.4× chance; no shadowing needed). Readout needs a PMI-like
normalisation; per-target/per-source budgets alone are not enough.

## MINOR
- M1 (PROVED) repeated syllables never learned: same channel winning two boundaries stays on → no rising edge; own voice
  merges (Stream doesn't restart a repeated char). 13.4% of replies have an adjacent repeat; collapse() hides it. Fix: one
  silent tick between identical syllables.
- M2 (PROVED) channel recruitment is a global structural op (novelty test over all channels + free-slot allocator). Toy
  bugs: slots full → argmin(used)=0 overwrites channel 0; before first recruitment the WTA launches empty slot 0 (2
  spurious GOs at start-up). Counts: 324/1143/2548/3548 channels at 100/1k/10k/all dialogues; 2.7% of reply chars (dial.
  1k–10k) have no channel yet. Fix: no recruitment when full, or recycle least-used channel and reset its weights.
- M3 test-protocol/toy fidelity: hip learns during tests and a guided pass precedes the E5 test (+3 to +6/20); probes 44%
  of training; ρ calibrated on evaluation probes; nP 2048 not 4096; toy GO −5 brakes not −1000 vetoes; one seed; ear_D
  redirects P_rest's slots to channel 0 (double fan-in).
- M4 (PROVED) ρ coupled to training: REST cue (+64) must beat ρ·(16 + learned) → ρ > 3.7 silently stops marker learning.
- M5 (PROVED) "guided speech 1.00" cannot detect forgetting: cued channel 16 vs best 3-of-4 competitor 12 + ≤1.1 learned;
  margin 2.9 for any weights.
- M6 (PROVED) budget enforced only every 100 ticks; overshoot ≤0.16 for thal→channel. Harmless.

## Claims tried and NOT broken
- Timing law (§3) PROVED (rule_exact.py; dense and sparse branches identical): source held through t0+1 gains 0; silenced
  from t0+1 gains +η (0.465 after 100 events); transient at t0−1 +η; if it returns at t0+2 or t0+3 gain 0. Reserve scales
  only the forward term: ×1.00/0.58/0.41/0.34/0.30 for sources on 1/2/3/4/6 ticks before the rise; back term unscaled.
- thal→channel = +η per latched unit per GO, no back term (PROVED); replay of this rule alone reproduces the circuit's
  first syllables.
- "Teacher never pushes GO" (SIMULATED): 0 fallback pushes in ≈2,500 training dialogues. What is learned under the teacher
  (plan→r1, own voice k→k+1, own voice L−1→marker) is exactly what is used at test (PROVED); only mismatch = hip mode (S1).
- Bounded weights, no runaway (PROVED: every learned path has a per-target budget).
- B7 no wholesale wipe-out (ab_s0.txt): A after A 12/20, AUC .74; after 410 dial. of B 6/20, AUC .65; B 7/20. B=0.5: A
  15 → 10 (replay).
- B2 with untrained PFC: plan decoding 1.00 at 340–1020 dialogues (but see S1(5), S3).
- Corpus numbers (PROVED): always-你 0.208; P(r1∈Q) 0.148 vs 0.057; reply chars in Q 0.146 vs 0.043; decay τ ≈ 2,010
  dialogues; 13 (45 latched units) or 21 (82 units) channels bind at B=1.

## Corpus ceiling: verdict FALSE, and the right metric
PROVED (exact counts, ceiling_verify.txt; same 1500 held-out pairs as the proposal; SE of Δ ≈0.006–0.012):
| learner (trained on pairs[:N]; tested on never-heard lines) | 1k | 3k | 10k | 40k |
|---|---|---|---|---|
| proposal's "ideal" (argmax P(r1 given last char of Q) + greedy bigram), Δov | −.001 | −.001 | +.003 | +.010 |
| learned copy: say heard chars with copy gain P(c∈R given c∈Q) − P(c∈R) > 0.1, Δov | +.121 | +.124 | +.144 | +.154 |
| same, AUC / 20-way top-1 | .60/.21 | .61/.23 | .63/.25 | .63/.26 |
| first syllable copied from Q, rest Q-independent chain, Δov | +.060 | +.062 | +.088 | +.099 |
| naive-Bayes bag (no explicit copy), Δov | +.006 | +.012 | +.038 | +.083 |
| reference: echo Q back | +.114 Δov, AUC .67, top-1 .27 | | | |
| reference: always 你 | overlap .208/.208, Δ 0, AUC .50 | | | |
The proposal's learner is mode-seeking (argmax → generic continuation → Δ≈0 by construction). Picking chars whose
probability RISES given the heard line shows large content; learned-copy outputs are topical heard chars (挤, 心疼, 罪过…;
8% empty; copy_examples.py). Content on never-heard lines exists; this design lacks the route (S4).
Right metric: talk.py's overlap is precision-only and frequency-dominated (always-你 0.208). Matched-minus-mismatched Δ is
unbiased but its size depends on reply length/genericness. PRIMARY: 20-way discrimination of the expected reply from what
was said — char-set F1 vs the true reply and 19 others; report top-1 (chance 0.05) or AUC (chance 0.50). Any policy
ignoring the heard line scores exactly chance; frequency hits score nothing. Report separately for familiar probes and for
≥500 never-heard pairs; test familiar probes ≥20 dialogues after their last presentation, no teacher pass in between,
hippocampus in its normal plastic mode. SECONDARY: first-syllable accuracy split top-10 vs other first syllables; Δov with
SE; copy rate (said chars in the heard line vs shuffled line; corpus 0.146 vs 0.043); stop reason (marker vs breath) and
cycles.

## Files
rule_exact.py; ceiling_verify.py → ceiling_verify.txt; copy_examples.py; corpus_facts.py; threshold.py; t3b_fix.py →
t3b_fix_T.txt, t3b_fix_T_origbag.txt, t3b_fix_TS*.txt; e5v.py → e5v_s0.txt; circuit_learnpfc.py (--pfc 1/0; records
plans at GO); e5x.py (E5/scale/A→B schedules, two-pass tests, hip frozen/plastic, --budget, --eta_rest, recordings);
replay_thal.py, replay2.py; runs (.txt + .pkl): scale_s0, pfc_s0, ab_s0, scalefix_s0, e5fix_s0; smoke.txt.
