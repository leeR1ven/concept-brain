# REPORT: verification of "Hold by structure, learn from one-tick transients" (LEARNING lens)
(Saved by the coordinator from the verifier's final reply; the harness blocked the verifier's own write.)

Labels: PROVED = algebra or exact numpy transcription of brain/net.py on scripted spike trains; SIMULATED = numpy sims
(numbers given); ASSUMED = untested. All files in this folder.

## 0. Verdict: 1 FATAL, 1 SERIOUS, 10 MINOR
Holds: every timing link (S1→GO, P→Y, Y→Y, Y→END) forms under the exact rule in the real training order and stays; no
back-term exposure cancels any; train and test share one timeline; no runaway; B1, B3, B5, B6 independent of content
weights. Fails: content (B4) and its stability (B7). With the real ear→T thought the learned first syllable is not
reliably related to the heard line; even idealised codes give ~half the reported effect; it does not accumulate; it is
re-decided every few hundred dialogues and forgotten within ~2k. No parameter of this readout fixes it; the corpus caps
any association readout of this input near +3 to +5 points. The one large lever is the echo route (premise and map
learnability supported; latch untested).

## 1. Claims tried and NOT broken
| # | Claim | Check | Result |
|---|---|---|---|
| T1 | Dense (GPU) and sparse (CPU) Path.learn/decay are the same rule | both transcribed; 3,000 random ticks ± reserve/budget/decay (check_rule.py [0]) | max diff 0. PROVED |
| T2 | Lemma 1 formal: Δw = η·x/(1+L/w_ref)·s_i(t−1) − η·#{t'∈[t+1,t+d]: s_i(t')=1}, floor w0 | 8 edge cases (check_rule.out [L1]) | holds ± source reserve. PROVED (wording: M2) |
| T3 | S1→GO forms; S2..S8→GO stay 0 | scripted real timeline, 600 turns | 0.54/0.99/1.21/1.90/2.0 after 25/70/100/240/600; S2..S8=0; 0 back hits. 2.0 takes 269 events with decay, not 240 |
| T4 | P→Y no back-term exposure | P on L+6; Y on L+7..L+9; P off L+8..L+10 | 0 hits. PROVED |
| T5 | Train/test timelines identical | traced talk.py + toy_circuit | teacher GO L+5 = learned own GO (S1 at L+4); P L+6; ring R0 L+7 = first cue; later cues on ring R0; cue never needs to force R0. PROVED |
| T5a | F goes off at a fixed tick | code | F off exactly L+3; F reads layer 0; SenseArea.fire masks learned/return currents to layers ≥1; layer-0 split/delete 0. PROVED |
| T6 | Cue margin, guided speech exact | cued Y 20, others ≤15+2 | END ≤1.97 (M1); each char shares 3 of 4 muscles with 12.4 others on avg (max 25), all ≤17<20; need sets unique 3,970/3,970; mouth decodes every exact set (muscle_overlap.out). PROVED |
| T7 | No runaway | budgets; END equilibrium | PROVED; free use with learning on keeps 15 distinct replies after 15k self-taught replies (selfreinforce_s1.out) |
| T8 | toy_circuit tests honestly | code | probes held out; learning off; no GO/cue at test. Exception: Y allocation oracle (M5) |
| T9 | Gap not a prior/length artefact | length-matched shuffled baseline | +2.1/+2.8/+3.0 vs plain shuffle +2.8/+1.2/+2.2 (seeds 1–3). Shuffled-pair null is valid |
| T10 | Some lexical generalisation (B8) | b8.out; 11.4% of held-out prompts occur verbatim in training | unseen prompts +2.2/+2.4 ±1.0 (idealised; seen +4.4/+5.7) |
| T11 | toy_reply's missing Y→Y back term harmless | yy_back.out | full corpus identical; at 3k length 1.66 vs 1.75, later-syllable gap +1.1 vs +1.8 |
| T12 | Echo premise: other's c and own c give same layer-0 code | real ear replica, 200 chars (echo_premise.out) | FEAT0 Jaccard 0.889 (p10 0.864) vs 0.208 across chars; own-c identifies other-c 200/200 |

## 2. FATAL F1: B4 — P→Y readout of the real thought does not make the first syllable reliably related to the line
Claim: "first syllable +2 to +4 points over shuffled" (§0.5, §8.1, §9.1). Setting: η·K_T 0.1, B 2, γ 0.999, θ 0.02.
(a) Real thought code: gen_realT.py ran ear_np.World with T exactly as toy_circuit (4000/200, fan-in 30 FEAT0, hold 3.0,
set point) over the whole corpus on the training timeline; recorded T(L+5) for all 47,786 turn pairs (200 active;
random different lines overlap 0.050). content_realT.py = same learner/splits/metric. 4 seeds × 7 checkpoints
(1k/3k/5k/10k/20k/30k/45.8k pairs):
| Codes | per-seed means | pooled | checkpoints ≤0 | at 3k pairs |
|---|---|---|---|---|
| Real T | +1.06, +0.24, +0.56, +0.64 | +0.62 | 9/28 | +1.4, −0.9, −0.5, +2.3 |
| Idealised, same splits | +1.96, +0.99, +0.90, +1.83 | +1.42 | 2/28 | — |
Per-checkpoint SE ~0.9 → a single model's content indistinguishable from chance. toy_circuit probes at 2,000 dial.:
1/20 first chars in reply vs 3/20 shuffled. §8.1 numbers all from idealised codes; overstate ~2×.
(b) Idealised codes weak and seed-dependent (milestone_ideal.out, 8 seeds × 2,000 held-out lines): pairs 500/1k/2k/3k/5k
→ mean gap +0.2/+1.1/+1.1/+1.3/+1.4; runs > 2 SE 0/8, 1/8, 2/8, 3/8, 3/8. Proposal's 3k figure (+2.2) is a favourable
seed (seeds 2,3: +0.3, +0.5). Seed-1 full corpus reproduces +2.8 (vs +3.3).
(c) No accumulation: evaluated every 2,500 pairs: seed 1 +1.8 ±1.0 (min −0.2); seed 2 +1.7 ±0.9; full corpus no better
than 3k. Cause (PROVED): γ 0.999 → τ ≈ 2,000 turn pairs; B/(η·K) = 20 → a saturated target holds ~its last 20 contexts.
(d) Absolute level (idealised): first char in expected reply 10–12%; "always 你" 20.4%; exact-first-char 4.3% vs
"always 你" 9.9%. ~2/3 of the gap from outputs occurring in the heard line (implicit echo). Lines ending in the same
char get the same first syllable 92.8% vs 8.0% → T works mostly as a last-char lookup; last-char-only codes give about
the same gap (+2.3 vs +2.8) (b8.out).
(e) Cap and tuning (content_main.py, seeds 1–3): Naive Bayes on last 8 heard chars (ideal for this input) +2.3 to +4.6;
"say the last heard char" +10.2/+9.5/+9.9 (in reply 0.151 vs 0.049). Only B/(η·K) matters. B/(η·K)=40: +2.2±1.2 and
+1.5±0.7, outputs collapse to 8–12 chars. 100 → prior mode (你/我). 4–10 → rare-char noise. γ 0.9999 or 1: stable
+2.4±0.6 but rare-char outputs (照, 被, 找), absolute in-reply 4–6%. Tuning cannot rescue B4.
FIX: echo route as the content source (partly checked).
(i) premise holds (T12). (ii) map learnable with the exact rule, back term on (echo_map.py/out): mirror Z_c fires one
tick at Y_c's release; learned FEAT0→Z, η 0.0005, budget 2, γ 0.999; trained only on own teacher-forced speech
(1-tick-delayed own voice); 500 Z. Test on other's held-out lines: top-1 36%/50%/59% after 300/1,000/3,000 dialogues;
top-5 42/55/63%; median rank 0; chance 0.2%; despite 1.1M back-term hits.
(iii) content with that accuracy (echo_gain_s1.out, echo_gain2_s1.out; idealised P→Y base +2.8):
| Echo variant | map acc. | gap |
|---|---|---|
| last heard char's Z latched (WTA), bias β 0.1–1.0 to its program at first boundary | 0.5 | +4.3 to +5.5 |
| same | 1.0 | +10.2 |
| β·recency latch | 0.6 | +4.9 to +5.1 |
| flat latch (all heard chars equal) | any | +2.5 to +3.6 (recency essential) |
Misrouted echoes lower absolute in-reply at 50–60% accuracy; confidence gate on Z margin (ASSUMED).
Still ASSUMED: the latch (last Z wins, reset by next line); false echoes and self-training of the map while listening;
β scale vs real P→Y scores; transfer onto the real-T base (plausible: echo bias independent of T).

## 3. SERIOUS S1: B7 — content memory short and volatile
Claim: "content links fade gradually, τ ≈ 1,400 dialogues, displaced only proportionally" (§5).
- Corpus A then B (forgetting_corpus.out): A = 3k pairs; exact first char on A lines 17.4% (held-out 4.2%); then +1k/+3k/
  +10k disjoint pairs → A 8.7/5.7/4.8% vs control 5.6/5.1/4.8%. A-specific gone after ~3k pairs (~2k dialogues).
- Clean synthetic (forgetting_synth.out, recall@1 after 5 epochs): associations 30/100/300/1,000/3,000 → corpus-skewed
  targets 1.00/0.84/0.60/0.39/0.12 (γ 1: 0.27); uniform —/1.00/0.90/0.64/0.31. Capacity ~100–300 associations.
- Interference vs decay (interf_gamma.out): 300 assoc. (recall 0.62) then +1k/+3k/+10k new: γ 0.999 0.44/0.12/0.04;
  γ 1 budget 2 0.55/0.43/0.34; γ 1 targets among 15 most frequent first chars 0.30/0.05/0.00; γ 1 other targets
  0.71/0.68/0.55. Frequent targets wiped by the per-target budget even without decay. No budget → 0.13 (prior only).
- Volatility (volatility_s1.out, same 2,000 held-out lines): lag 500 (~345 dial.)/1,000/2,500/5,000 pairs → first
  syllable changed for 68/79/89/89% of lines. Dominant output covers 18% (max 34%); 12 chars take turns dominating.
Fix (checked: no parameter fixes both sides): with one per-target budget, removing the prior and remembering more than
~B/(η·K) contexts per target are the same knob; γ_PY ≈ 0.9999 keeps rare associations but shifts outputs to rare chars.
Options: restate B7 ("skills permanent, content ≈ last 1–2k dialogues"), or a second slower P→Y copy with own budget
(ASSUMED).

## 4. MINOR
| # | Issue | Evidence | Fix |
|---|---|---|---|
| M1 | Lemma 4 equilibrium wrong: gain r·η/(1+L/w_ref) = loss (1−γ)L → L*(1+L*/w_ref) = η·r/(1−γ) | PROVED; Y→END equilibrium 1.97 not 2.73 | use 1.97 |
| M2 | Lemma 1 wording off by one: source must also be off on the first tick after j's last tick | PROVED (on at t−1 and t+3 after 3-tick target nets 0) | reword; all links comply |
| M3 | "Lemma 1(b) is evidence #7" unsupported: since 004031a →go has back=False; earlier (back on) teacher GO was braked (#6) so GO rarely rose | code history | drop attribution; "held sources cannot teach" only for back=True |
| M4 | toy_reply applies only forward term on Y→Y | T11 | negligible; use exact rule |
| M5 | Y recruitment uses char id (oracle); not a local rule in the spec | code | local commit: if no Y reaches full cue score 20, commit a free unit with largest D input, wired both ways to the 4 active D's; unseen chars cannot be said. Untested |
| M6 | w_init unspecified for new learned paths: net.Path default U(0,0.02) gives each Y ~5 of born current from 500 P units, swamping θ 0.02 and learned 0.1–2 | PROVED | w_init = 0 on S→GO, P→Y, Y→Y, Y→END |
| M7 | Free use with learning on locks in habits | interleaved 1:1 with teacher: absolute in-reply 0.061 vs 0.152 when free replies don't learn; gap +1.5 vs +1.8 | accept, or gate content learning on teacher/feedback (ASSUMED) |
| M8 | Breath reserve ambiguous: per-tick depletion of Y (3/syllable) → ≤3.7 syllables; τ 100 recovers only 25–40% between turns | PROVED | deplete once per syllable, or τ ≈ 30 |
| M9 | Replies ~2 syllables (corpus 6.2); later syllables carry ~no content (+0.3 full corpus); END is argmax successor for 50.5% of reply tokens | SIMULATED | known (§9.7) |
| M10 | Decay on clock time: S1→GO 2.0→1.0 after 693 steps = 69k idle ticks ≈ 87 min; model then never starts; content links 1% after ~9.6 h idle | PROVED | no decay on S→GO, or decay per experience |
Not an issue: dense branch processes full P→Y (10k×3,970) and Y→Y every tick in current() and learn(); ~3 ms/tick,
~2 min per 500 dialogues (ASSUMED acceptable).

## 5. Toy audit
No teacher info leaks into tests; held-out lines; fair baselines. Problems: §8.1 idealised codes (F1a); single-seed,
single-checkpoint numbers (F1b, F1c); toy_reply omits Y→Y back term (M4, harmless); Y allocation oracle (M5); the 20
probes in toy_circuit cannot measure content.

## 6. Files (this folder)
lab.py (shared learner/metrics); check_rule.py/.out; content_main.py → content_base_s1.out, content_base_s23.out,
content_etak_s1.out; content_track.py, content_track2.py → content_track_*.out; milestone.py → milestone_ideal.out;
gen_realT.py → realT_{a,b,c}.npz; content_realT.py → content_realT.out, content_realT_s34.out; forgetting.py →
forgetting_corpus.out, forgetting_synth.out; interf_gamma.py/.out; volatility.py → volatility_s1.out; selfreinforce.py
→ selfreinforce_s1.out; yy_back.py → yy_back_s1.out; b8.py/.out; muscle_overlap.py/.out; echo_premise.py/.out;
echo_map.py/.out; echo_gain.py, echo_gain2.py → echo_gain_s1.out, echo_gain2_s1.out; corpus_stats.out.
