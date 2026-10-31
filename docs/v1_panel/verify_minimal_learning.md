# Verification of arch/minimal/proposal.md — LEARNING lens
(Saved by the coordinator from the verifier's final reply; the harness blocked the verifier's own write.)

Counts: 1 FATAL, 3 SERIOUS, 7 MINOR. Overall: content side fails as specified but is fixable.
Scripts/outputs in this folder.

## 1. Method
Recorder (rec.py): runs the proposal's brainsim.Brain unchanged (ear, mctx, hippocampus chain 3.0 / η 0.5 / γ 0.7 /
budget 2, F/F1/O/GO, motor stage). Protocol C10: listen pause 0; teacher cues first program until the model's own GO,
then each boundary, then END; 3 + 4 pause ticks. Records source states at B−1, B+1, B+2, B+3 of every release. Also
computes the learnability proposal's T alongside (4,000 units, 30 inputs from ear layer-0 feature neurons weighted by
reserve, 1-tick self-hold 3.0, top 5%, set point), not fed back.
Offline learning (learn.py), exact: in training the cue always wins (cue > Σ caps) so trajectories don't depend on
decision weights; the test's first syllable depends only on the state at e+4. Decision paths learned offline with exact
net.py arithmetic (forward η·x/(1+L/w_ref) from S(B−1); back −η from S(B+1), floor 0; every 100 ticks L·γ then per-target
budget). Unit test matches direct implementation to 6e−10.
Runs: rec_main.pkl (seed 0) and rec_main_s1.pkl (seed 1: different wiring, sounds, muscle sets), 693 pairs each: set A
20 fixed pairs × 6 rounds, each followed by one fresh corpus dialogue; set B same with another 20; final round A, then B;
check()-like probes 20 lines × 2 preceding lines under frozen+silent, learning on+silent, learning on+teacher reply.
rec_corpus.pkl: 1,200 plain dialogues (1,822 pairs) then the same probes. Reply first chars distinct within each set; A
includes 你, 我; B rarer.
Evaluation: fresh lines prequential (own choice before the cue) vs shuffled pairs and vs most frequent first char so far;
familiar recall at exposures 2–6 vs mismatched baseline; retention = set A after phase B. Side check: training GO at e+5
for 3,208/3,208 replies, 0 mouth ticks while listening.
Files: rec.py, learn.py, variants.py, decode.py, back.py, nn.py, corpus_eval.py, mrr_control.py, hipsweep.py;
decode_main/s1/corpus.out, back_main/corpus.out, var_main1/2.out, var_s1.out, nn_main/s1.out, corpus_eval.out,
hipsweep_*.out.

## 2. Issues
### FATAL F1: B4 not learnable with the specified readout (counted evidence, caps above equilibrium, hip source at GO−1)
Claims: "B2/B4 learnable but weak … above shuffled at ≥10⁴ dialogues" (§0.5, §8.1); "caps must not bind" (C8).
SIMULATED (two seeds; var_main1/2.out, var_s1.out), prequential familiar recall, seed 0 / seed 1:
| Readout / source | A exp. 2–6 (mismatched) | A at exp. 6 | A after phase B | B exp. 2–6 | B final |
|---|---|---|---|---|---|
| Proposal (hip+ear, every boundary, no binding cap) | 0.07/0.09 (0.02) | 0.15 | 0.00 | 0.00 | 0.00 |
| hip, first boundary only, counts | 0.06/0.07 (0.05) | 0.10 | 0.10 | 0.00 | 0.00 |
| hip+ear, every boundary, budget 0.02 (~2.5 events) | 0.34/0.36 | 0.45/0.35 | 0.10/0.20 | 0.46/0.41 | 0.50 |
| hip+ear, every boundary, budget 0.05 (~6 events) | 0.35/0.30 | 0.60/0.55 | 0.35/0.30 | 0.26/0.29 | 0.80/0.75 |
| hip, first boundary only, budget 0.05 | 0.36/0.37 | 0.85/0.95 | 0.80/0.70 | 0.21/0.25 | 0.90/0.95 |
| T (fixed held layer-0 code), counts | 0.88/0.87 | 1.00/0.90 | 0.95/0.95 | 0.33/0.37 | 0.85/0.90 |
| T, budget 0.01 (~5 events) | 0.94/0.93 | 0.95 | 0.85/0.80 | 0.90/0.89 | 0.95 |
| hip⇒think analogue (fixed random projection, top 5%; not the brain proposal's full latch), counts | 0.05 | 0.05 | 0.10 | 0.00 | 0.00 |
| same projection, budget 0.01 | 0.59/0.52 | 0.70/0.55 | 0.40/0.30 | 0.55 | 0.60/0.55 |
| soft-bound compression η 0.01/0.1 (seed 0) | 0.10/0.15 | — | 0.00/0.05 | 0.00/0.01 | 0.00 |
| upper bound: 1-NN over training-time states, hip / T | 0.97–0.99 / 1.00 | | | 0.95 / 1.00 | |
Mismatched baselines ≤0.05 everywhere. The information IS in the hippocampus (1-NN 0.95–0.99); the learned readout fails.
Why (PROVED with measured overlaps): w ≪ w_ref (largest ~1e−3) and caps never bind → y_c(h) ≈ η·Σ_events κ·|h∩S_e|
(κ = share the back term keeps). Familiar line ℓ heard n_ℓ times: y_c(h_ℓ) ≈ η·[N_c·I_gen + n_ℓ·(I_same − I_gen)] → c_ℓ
beats mode m only if n_ℓ > (N_m − N_c)/(I_same/I_gen − 1). Measured at e+4 under learning: hip Jaccard same line 0.26–0.31
vs different 0.16–0.18 (same at 693 and 1,822 pairs); I_same/I_gen ≈1.5 → n_ℓ > 2(N_m − N_c). hip→Y learns at every
boundary → N_m counts every appearance of the mode char anywhere in a reply; grows linearly to decay equilibrium
(~1,500 dial.) → hundreds–thousands of exposures needed; more dialogues never help (ratio set by the code). Proposal's
"≥10⁴" came from an ideal prior-correcting learner (PMI votes, corpus_stats2.py). For T the ratio ≈11: early recall but
requirement still grows (set B, trained 300 pairs later, recalled only at exposures 5–6 with counts; with budgets from
exposure 2). A binding budget removes N_c: y_c(h) = B·Σ_{i∈h} p_c(i) (learnability Lemma 5).
FIX (SIMULATED): (a) per-program budgets binding at ~5 events, END exempt (budgeted END → ratio 0.004–0.007 → every reply
hits the 12 cap); (b) GO−1 path learns only at the first boundary (S2); (c) preferably T as first-syllable source, with a
budget.
Cost (corpus_eval.out): budgets remove the prior on never-heard lines: top-1 drops from the mode's 0.08–0.09 to
0.03–0.08, outputs varied. No variant robust on never-heard lines at 1,200 dial.: best T+budget 0.033 vs 0.021 shuffled,
p = 0.037 uncorrected across 6 variants; proposal config p = 0.11 (top-1), 0.21 (MRR).

### SERIOUS
S1. Toy content tests ran with frozen weights, which do not reproduce training-time hippocampal states. t_learn_y.py calls
test(..., learning=False). Training-time state of a line overlaps its frozen-test state J 0.163/0.170 vs 0.151/0.159 for
other lines (nn_main.out, nn_s1.out); learning-on test 0.244–0.259 vs 0.152–0.160. Test could not show recall even with a
perfect readout; contradicts the proposal's own protocol (§2.5 learning on). Trained lines heard once; near-constant
outputs make "first == true" equal shuffled. Fix: prequential or learning-on (gate closed) on lines heard several times
vs mismatched pairs.
S2. First-syllable path trained at every cued boundary: next and END events ~6× first events, share the generic hip core.
First syllable on fresh lines became 了 in 246/276 (s0), 215/276 (s1); own choice matched true first char 0.004–0.011 vs
0.08 for the mode. 1,200-dial. run: first-syllable accuracy per 300 events 0.097, 0.077, 0.073, 0.067, 0.050, 0.039.
Learning hip→Y only at the first boundary restores the mode (你/我) and doubles retention under budgets (0.70–0.80 vs
0.30–0.35). Caveat: learner omits mctx→Y, whose e+4 input is the same motor rest pattern for every line (probably held 你
in the proposal's runs; cannot add line-specific evidence — PROVED).
S3. Back term erodes earlier learning ("pulses keep 96%/67%" §0.3, §5). Per-event measure reproduces (hip→first 0.98
pulse / 0.54–0.55 held; ear→next 0.70/0.41), but the rule subtracts η from every source on at B+1 that carries weight from
earlier events. Share of forward mass removed over the run (back_main.out, back_corpus.out):
| Path | First | Next | END |
|---|---|---|---|
| hip, pulse | 0.31–0.37 | 0.47–0.50 | 0.77–0.79 |
| ear, pulse | 1.45–1.61 | 0.42–0.45 | 0.61–0.62 |
| hip, held | 0.67–0.71 | 0.76–0.78 | 0.98 |
Loss grows with training: first 0.18 → 0.31 → 0.37 at 113/693/1,822 pairs. What END keeps carries no context (median END
ratio 0.1017 at true ends vs 0.0955 mid-reply) → content half of B6 cannot emerge from these sources (cause: rule timing).
FIX (timing PROVED; gain ASSUMED toward corpus ceilings 0.33 vs 0.12 for END, 0.11–0.19 next syllable): learn END and next
syllables from a one-hot HELD COPY OF THE CURRENT PROGRAM (on B…B+2, off at B+3). At B_end the copy is on at B_end−1 and
off from B_end → back term never touches it; same for next syllables except repeats (哈哈).

### MINOR
- M1 hippocampal correction right in substance, overstated in wording. check() sets b.learning=False (talk.py l.121);
  Circuit.step gates learning, homeostasis, decay on it (areas.py 273–281). At e+4: frozen J 0.96–0.98 vs 0.35–0.37;
  learning on 0.26–0.31 vs 0.16–0.18; ~14–16% line-specific (proposal ~12%). But 1-NN decoding of 20 lines stays 1.00
  with learning on; 1-NN recall from training-time states 0.95–0.99 → frozen check overstates strength, not decodability.
  No hip parameter helps (hipsweep_*.out): baseline specific share 0.14; η 0.05 → 0.09; γ 0.3 → 0.16; frozen random
  weights (η 0) → 0.32 (no longer a learning hippocampus). Ranking of GO−1 sources under learning:
  | Source | specific share | 1-NN recall | learned recall (budget) |
  |---|---|---|---|
  | T (fixed, immune to learning) | 0.86–0.87 | 1.00 | 0.93 |
  | hippocampus | 0.14–0.16 | 0.95–0.99 | 0.36 overall; 0.85–0.95 at exposure 6 with first-only learning |
  | hip⇒think analogue | 0.08–0.10 | 0.79–0.92 | 0.52–0.59 |
  | ear cortex at e+4 (same 15 neurons every line) | 0 | 0.05 | — |
  | PFC receiving | not in replica; STATE 0.15–0.60 even at line end | — | — |
- M2 program recruitment neither specified nor simulated (toy preallocates 987 programs and cues the index; real
  MotorArea.cue pushes D). Local rule: fixed D→Y detectors needing all 4 of a program's D; when ≥4 D cued and no Y fires,
  allocate a free unit, wire one-shot; delete after long disuse. With count readout late programs never win. ≤4,096 slots
  fine. Never-cued chars cannot be said. No deletion/reuse yet.
- M3 optional think→Y cannot learn in the proposal's timeline (PROVED given STATE latch hold): latched thinking state on
  at e+4 and e+6; F on only at e+7; thalamus released e+8 → each unit nets ≤η(x−1) ≤0 (STATE #7).
- M4 ear(e+4) and mctx(e+4) same for every line → their paths only add prior.
- M5 toy details: no caps, cue 100 not 50 (equivalent, caps never bind); t_back.py measures per-event set difference not
  net effect; t_gate's "trained choice survived" = survival of a content-free mode.
- M6 END-from-hip ≈21 assumes 96% retention; actual ~22% → caps bind even less. Harmless.
- M7 shuffled pairs right null for never-heard lines but: top-1 no power with near-constant output; naive MRR false
  positive (p=0.003; true target scored only once recruited → frequency bias); neighbour-target control removed it (0.269
  vs 0.270; mrr_control.py); fair version (frequent targets only) p = 0.42–0.97. Familiar lines need mismatched baseline.

## 3. Task answers
1. Back term pulse vs held: per-event figures correct; long-run retention pulses 63–69% (first), 50–53% (next), 21–23%
   (END); held 29–33%, 22–24%, 2%. END not learnable from these sources (S3).
2. Hippocampus at GO−1: freeze real; overlap effect real (0.96 → 0.26–0.31); decodability survives. Best source under
   learning: T, then hippocampus, then a projection of it; ear and mctx carry nothing at e+4.
3. Teacher gate: logically required (PROVED): a free release is the argmax of the very weights being trained → learning it
   only raises the model's own choice (pure positive feedback, no world info) → measured drift 你 → 的/了. "Learn when own
   voice matches the ear" useless (B5 makes it always true). Local error gate (learn only when the released program differs
   from the learned-input choice) is closed for every free release (PROVED), needs no teacher flag. Tested: recall same as
   C11 (0.08–0.11 counts, 0.29–0.35 budget, 0.86 with T) but retention much worse: T 0.15–0.35 vs 0.80–0.95; hip-first
   0.00–0.10 vs 0.70–0.80. Recommendation: keep C11, implemented locally ("a program learns only when its release had
   external drive"); in deployment the speaking policy freezes. Label-free learning would need imitation of the other's
   speech through an auditory→program map (ASSUMED).
4. Content: no content in proposal config (reproduced). Helps: budgets; separating first-syllable path from next-syllable
   learning; a line-specific source (strongly). Doesn't: more dialogues with count readout (PROVED), soft-bound
   compression. Nothing robust on fresh lines at 1,200 dialogues.
5. Recruitment: M2; without binding budgets late programs never win.
6. B7: no runaway (PROVED); equilibria smaller than proposal's. A then B, test A: proposal 0.00 (nothing learned); tight
   budget every boundary 0.45 → 0.10–0.20; first-only + budget 0.85–0.95 → 0.70–0.80; T 0.95 (0.80–0.85 with budget);
   error gate → 0.15–0.35.
7. Toys vs text: frozen-weight tests (S1); one exposure for trained lines; no recruitment or caps; no teacher leaks;
   held-out lines included; baseline statistic has no power.

## 4. Claims not broken
Per-event back-term survival figures; ~12% line-specific share (14–16%) and frozen J ≈0.97; check() freezes all learning;
cue always wins, caps never bind, totals bounded; C11 and the error gate stop drift in free speech; no robust content on
never-heard lines at ≤1,200 dialogues; program-level release gives exact chars; training GO at e+5 every time, mouth still
while listening.

## 5. Recommended learning-side changes (tested offline on two seeds unless marked)
1. Per-program budgets binding at ~5 events, END exempt.
2. GO−1 path learns only at the first boundary.
3. END and next syllables learn from a held one-hot copy of the current program (timing PROVED; not end-to-end tested).
4. Preferably add the fixed held source T for the first syllable.
5. Test with learning on, on familiar lines heard several times, vs mismatched pairs; report recall per exposure and after
   unrelated material.
6. Implement C11 locally.
