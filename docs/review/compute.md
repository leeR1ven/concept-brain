(Saved by the coordinator from the reviewer's final reply; the harness blocked the reviewer's own write.)

# Whole-architecture review: computation, representation and capacity

**Reviewer:** compute (background in AI and computational theory: representation, capacity, what a learning rule can and cannot learn, and brain-inspired models such as assembly calculus, HTM, predictive coding and attractor networks compared with mainstream AI). Date: 2026-10-31.

**Files.** Every script and output is in `/tmp/claude-0/-root/e92d3d7d-5823-4f5c-a0ec-4fe302bff9af/scratchpad/arch/review/compute/`. The harness refused `report.md` ("subagents should return findings as text"), so this reply is the report.

**Labels.**
- **PROVED**: an argument with numbers.
- **SIMULATED**: a numpy run, with the file and its numbers.
- **CITED**: a paper and its finding. "via research/0x" means I took it from the panel's literature reviews and did not re-read the paper.
- **ASSUMED**: argued but not tested.

---

## 0. Verdict

**As built, the architecture cannot reach the final goal, and no parameter setting changes that.**

The body works and should be kept as it is: turn-taking, clean syllables, stopping, and staying silent while the other person speaks. It is a reliable innate reflex system. Everything that has to be *learned* runs into four structural ceilings. Each is measured or proved below.

1. **No stage on the content path learns a representation.**
   - The ear "cortex" is, for 99.4–99.6 % of neuron-ticks, a thresholded copy of one input line per neuron (SIMULATED, PROVED).
   - v1's PFC, T, is a fixed random bag of roughly the last 8 chars.
   - The hippocampus is a 1:1 mirror of the ear.
   - So two experiences can only ever be "similar" by sharing sounds.
2. **The one rule is a one-tick, first-order, excitatory-only co-occurrence counter.**
   - It cannot learn held co-activity, so it cannot form attractors, concepts or a learned working memory.
   - It cannot store negative evidence: every learned weight stays at or above its birth weight.
   - It cannot normalise for frequency, so budgets have to cap memory at about 5–25 events per target.
   - It cannot link events more than one tick apart without a hand-built one-tick gate.
3. **Nothing learns from consequences or from prediction errors.** Outside teacher-forced turns the content paths are closed. The model cannot improve during real conversation, which is the opposite of real-time learning.
4. **The testbed cannot show meaning.**
   - On never-heard lines, strong association and retrieval learners find almost no content beyond echo: 20-way top-1 ≤ 0.095 with 44k pairs, against chance 0.05 and 0.273 for plain echo (SIMULATED).
   - The corpus refers to nothing the model perceives.

**The smallest changes that would give the capacity** are all local, work with binary units, and keep the designer's spirit.

- **(a) A predictive form of the one rule.**
  - "Earlier → later strengthens" stays, but stops once the later event is already predicted.
  - A link weakens when it predicted something that did not happen.
  - Learned weights may become negative, through a learned inhibitory pathway, so Dale's law is kept.
  - Add a few-tick eligibility trace and one broadcast third factor.
- **(b) Let held co-activity be a learning target** in a few "own" populations (PFC-own, CA3-own), with fan-in of about 1,000 instead of 96. This allows assemblies and attractors to form.
- **(c) Add a slow context layer and a predictor.** The context layer is the designer's "thought range that slowly moves". The predictor is trained on everything heard, from both speakers.
- **(d) Move to a grounded micro-world** in which the answer is never in the question.

**Cheap evidence, all on the panel's own data.**
- Familiar memory (2 seeds, the exact recorded thought codes):
  - An error-driven readout recalls a familiar first syllable after one exposure 0.72–0.73, against 0.32–0.38 today.
  - After 3–6k more dialogues it keeps 0.83–0.84, against 0.46–0.50 today.
  - It needs no budget and no decay, but **only works with signed weights**. With excitatory-only weights it keeps just 0.05–0.07.
- Attractors (concepts):
  - Neither the current rule nor the panel's earlier "held co-activity neutral" variant stores a single pattern that completes from a half cue.
  - The same predictive rule, applied to recurrent links with fan-in 960, completes 97–100 % of 300 stored patterns.
  - At the model's fan-in of 96, no rule completes even 20 patterns.

---

## 1. What the whole system can learn today: a capacity audit

### 1.1 Stage by stage (v1 as specified and simulated)

| stage | what it computes | learns? | limit | label |
|---|---|---|---|---|
| ear input | 960 thermometer lines of the 8 × 6 window; exactly 480 on | no | dense (50 % on) | PROVED (code) |
| ear 3D layers 1–3, the "cortex" (20,160 neurons) | each neuron fires iff its root input line is on and it has a fixed high bias; P(on \| root off) = 0; 39 / 58 / 78 % never fire; 27 / 18 / 9 % always on (muted); about 3,960 informative neurons covering at most 836 distinct lines | feed-forward weights: no; only split children (2–10 % of always-on neurons per restructure) become 2-input units | a replicated copy of the spectrogram window: no conjunctions, categories or invariances | SIMULATED §1.2, PROVED |
| ear→ear, ear↔motor-cortex paths | one-tick transition counts; fan-out 96 / 32; budget 1 | yes | about 80 storable patterns (§1.7); next-char recall 0.16 → 0.03 → 0.01 after 20 → 100 → 400 lines (§1.6) | PROVED, SIMULATED |
| hippocampus (about 20k alive, 5 % active) | 1:1 mirror (chain weight 3.0) plus fast transitions (η 0.5, γ 0.7) | yes | 4.8 active inputs from a stored pattern onto each member (rat CA3: about 240); at most about 160 patterns even with a symmetric rule; drives no behaviour in v1 | PROVED §1.7 |
| T (10k units, 5 %) | fixed random expansion of FEAT0 (30 inputs each) plus a one-tick self-hold; a recency bag of about the last 8 chars; overwritten by every line | no (set point only) | identifies the heard line (20/20), but similarity = shared sounds; holds one line; no order or roles | CITED (ARCH_v1 §1.7; learnability proposal §0) |
| P→Y fast and slow (first syllable) | budgeted counts per program | yes | keeps about the last 5 / 25 contexts per program; retains 0.37–0.38 at +10k dialogues | PROVED (Lemma 5), SIMULATED §1.4 |
| Hc→Y, Hc→END (later syllables, stop) | bigram chain over programs; stop | yes | reply order is only one step deep; only the first syllable "sees" the heard line | PROVED (read-once lemma, CONTENT §4.3) |
| copy route (Z, latch, E→Y) | char templates learned from own voice; a bag latch of heard chars; diagonal counts | yes | identity copy, weighted by frequency; repeats 49–61 % of the heard chars (humans: 15 %) | CITED (CONTENT §0) |
| F, GO, CD, beat, programs Y, D/I/M, END | innate turn-taking and articulation machinery | no, or recruited | localist; 4,096 program slots against 3,970 chars | PROVED |

So the only learned *representations* in the whole system are Z's char templates (taught by the model's own voice) and the rare split children. Every code that could carry meaning is fixed.

### 1.2 Measurement A: the ear "cortex" is a copy of its input lines

SIMULATED (`ear_linecopy.py` → `ear_linecopy.out`). Setup: an exact numpy replica of Net3D (seed 0) fed with the real world (the voice.py stream, 300 corpus lines, 7,092 ticks including pauses). For each neuron I followed its weight-1 self input down to the input line it descends from (its "root line").

| layer | alive | on per tick | P(on \| root line off) | copy model reproduces | never on | always on (muted) | informative | distinct root lines |
|---|---|---|---|---|---|---|---|---|
| 1 | 2,880 | 864 | 0.00000 | 99.49 % | 39 % | 27 % | 994 | 836 |
| 2 | 5,760 | 1,152 | 0.00000 | 99.43 % | 58 % | 18 % | 1,420 | 809 |
| 3 | 11,520 | 1,152 | 0.00000 | 99.62 % | 78 % | 9 % | 1,546 | 771 |

The "copy model" is simply: the neuron is on iff its root line is on and it is one of the neurons whose bias is high enough to win.

**Why it holds for any seed (PROVED).**
- A neuron whose weight-1 parent is on scores at least 1.0. One whose parent is off scores at most 0.5 + 3 × 0.01 = 0.53.
- Each layer keeps fewer winners than it has parent-on candidates (40 % < 50 %, 30 % < 40 %, 20 % < 30 %, 10 % < 20 %).
- So without extra current every winner has its parent on, and the fixed biases decide which of those win. The neighbour weights (at most 0.01) only break ties.
- Over 1,500 ticks the cortex has rank 815 against 286 for the input. The top-k competition adds a little non-linearity (3–4 % of neurons sit "in between"), but no features.

**What follows.**
1. Framework item 7 ("other layers may also carry local features") does not hold with these weights.
2. About 80 % of the 20k cortex neurons never vary (never on, or always on and muted). The hippocampus still mirrors all of them.
3. Lines that are on for most chars (background-noise cells, low thermometer levels) are built into every code. So the "common features" problem sits in the encoder, not in the learning. In numbers: a char's code averages 834 of 3,831 informative neurons (22 %), any two chars share Jaccard 0.20, and 305 neurons are on for more than half of all chars (SIMULATED, `seqmem.out`).
4. Real audio or images would pass through this front end raw.

This is harmless for char identity, because every spectrogram is unique. It is harmful for the specificity of associations, and fatal for meaning and for real sensors.

### 1.3 Measurement B: what this corpus can teach about never-heard lines

SIMULATED (`ceiling.py` → `ceiling.out`). Setup:
- The panel's 20-way metric: char-set F1 of the output against the true reply and 19 others; chance is top-1 0.05 and AUC 0.50.
- The content specialist's 1,500 never-heard pairs.
- Each learner is trained on the first N pairs.

Each cell is top-1 / AUC.

| learner | 1k pairs | 3k | 10k | 44k |
|---|---|---|---|---|
| echo the heard line | 0.273 / 0.666 | same | same | same |
| say the last heard char | 0.139 / 0.554 | same | same | same |
| prior bag (你我的不了) | 0.055 / 0.500 | | | |
| 1-nearest-neighbour retrieval of a training reply (tf-idf) | 0.078 / 0.522 | 0.090 / 0.530 | 0.089 / 0.530 | 0.089 / 0.535 |
| the same with Hebbian random-indexing similarity (local, online) | 0.073 / 0.516 | 0.079 / 0.522 | 0.082 / 0.525 | 0.091 / 0.530 |
| 30-nearest-neighbour char vote ÷ prior | 0.095 / 0.546 | 0.103 / 0.545 | 0.117 / 0.566 | 0.138 / 0.591 |
| PPMI association bag, heard chars allowed (copy rate in brackets) | 0.112 / 0.552 (0.17) | 0.163 / 0.581 (0.27) | 0.240 / 0.623 (0.37) | 0.276 / 0.632 (0.43) |
| **PPMI bag with heard chars excluded = pure non-echo content** | **0.060 / 0.508** | **0.068 / 0.511** | **0.077 / 0.516** | **0.095 / 0.529** |
| real T code, error-driven reply-bag readout (`readout_never_s0.out`) | 0.056 / 0.514 | 0.067 / 0.525 | 0.069 / 0.525 | — |
| real T code, count (Hebbian) reply-bag readout | 0.055 / 0.504 | 0.051 / 0.502 | 0.053 / 0.500 | — |
| the model's copy route, recommended tuning (CONTENT, 3 seeds) | 0.151 / 0.583 | 0.172 / 0.602 | 0.202 / 0.616 | — |

**What it shows.**
- What a reply contains *beyond the heard chars* stays close to chance even with 44k pairs.
- The best association learner gains almost entirely through echo: its copy rate climbs to 0.43.
- The model's copy route already sits in this band, and gets there the same way.
- So on this corpus, a model that understands and a model that parrots cannot be told apart on never-heard lines.

Caveat: these are shallow learners. A deep model trained on about 600k chars might find a little more (ASSUMED: not enough to change the picture).

### 1.4 Measurement C: an error-driven readout on the same thought codes

SIMULATED (`readout_rule.py familiar 0|1` → `readout_familiar_s0.out` and `readout_familiar_s1.out`).

**Setup.**
- Inputs are the content specialist's exact recordings of the real front end (`arch/content/fe_probe_s{0,1}.pkl`): ear layer 0 → FEAT0 → T (4,000 units, 200 active) at L+5.
- 200 familiar probes, each heard 6 times 100 dialogues apart, inside 14.6k corpus dialogues; tested before each exposure and up to 10k dialogues after the last one.
- The protocol is the same as `eval_mem.py`, and my MEM row reproduces CONTENT §4.2 exactly.

**Results.** First-syllable recall, given as seed 0 / seed 1.

| learner (same codes, same targets) | heard 1× | 2× | 3× | 5× | +<1k | +1–3k | +3–6k | +10k |
|---|---|---|---|---|---|---|---|---|
| MEM = ARCH_v1 (count rule; fast ≈ 5-event and slow ≈ 25-event budgets) | 0.38 / 0.32 | 0.72 / 0.72 | 0.84 / 0.84 | 0.93 / 0.92 | 0.79 / 0.76 | 0.58 / 0.57 | 0.50 / 0.46 | 0.38 / 0.37 |
| **delta rule, η 0.3, signed weights, no budget, no decay** | **0.72 / 0.73** | 0.96 / 0.96 | 0.99 / 0.99 | 0.99 / 1.00 | 1.00 / 1.00 | **0.92 / 0.95** | **0.84 / 0.83** | **0.57 / 0.52** |
| same rule, excitatory-only weights (w ≥ 0) | 0.46 / 0.41 | 0.76 / 0.70 | 0.85 / 0.82 | 0.94 / 0.93 | 0.67 / 0.76 | 0.15 / 0.18 | 0.05 / 0.07 | 0.08 / 0.09 |
| perceptron (updates only on mistakes; margin 0.5; signed) | 0.65 / 0.71 | 0.90 / 0.88 | 0.94 / 0.91 | 0.95 / 0.97 | 0.96 / 0.96 | 0.74 / 0.75 | 0.61 / 0.65 | 0.44 / 0.41 |
| delta rule, η 1 (seed 0 only) | 0.78 | 0.90 | 0.93 | 0.97 | 0.90 | 0.50 | 0.26 | 0.10 |

**The rule tested.** At each program unit y: Δw_jy = (η/|P|)·P_j·(target_y − drive_y), where the target is the teacher's cue. It is local: the unit compares its own learned drive with whether it was made to fire. In brain terms this is the dendritic-prediction rule (CITED: Urbanczik & Senn 2014) and the cerebellar Purkinje-cell scheme: T is exactly a granule-like expansion, but today its readout only counts.

**What it shows.**
- **v1's memory limit is the learning rule, not the code.** The same T codes hold a familiar line after one exposure, and for thousands of dialogues, once learning corrects errors.
- **Signed weights are essential.** The error rule stores "this context is *not* y" as a negative weight, which protects a familiar line from the steady creep of frequent programs. With excitatory-only weights the same rule forgets faster than today's counts.
  - This supports, in function, the designer's item 14(1): "strongly linked → positive weight, weakly linked → negative weight".
  - The negative part should run through learned inhibitory interneurons, so Dale's law is kept.
- **Never-heard lines gain nothing measurable.**
  - First syllable: 0.04–0.07 against 0.01–0.04 for MEM, all below "always say 你" (0.10).
  - 20-way reply bag: 0.056–0.069 against 0.051–0.055 for counts (§1.3).
  - So the gain is memory capacity and normalisation, not meaning.
- **Why it works (PROVED).** Steady-state count weights are normalised co-occurrence counts, which makes the readout a Hebbian prototype classifier. For correlated codes such a classifier is dominated by shared components and by the frequency prior; that is why the panel needed binding budgets. Local error-correcting rules store correlated patterns up to the capacity of the synapses (CITED: Diederich & Opper 1987, PRL 58: local iterative rules store up to N correlated patterns where Hebbian rules fail).

### 1.5 Measurement D: can the rule store concepts (attractors)?

SIMULATED (`attractor.py`, `attractor2.py`, `attractor3.py`, `attractor4.py` and their `.out` files).

**Setup.**
- A recurrent top-k region: N = 4,000 units, k = 200 active (5 %, the same as T and the hippocampus), born weights U(0, 0.01), soft bound.
- M random patterns, each held for 3 ticks, with gaps of random "other input" between them (in attractor.py the gaps run free instead).
- Test: a 50 % cue for one tick, then 3 free ticks. The score is the overlap with the stored pattern (chance 0.05; the cue alone gives 0.5). "(x %)" is the share of patterns completed to ≥ 0.9.
- "Gated" means plasticity is on only during the held ticks of items to be remembered (a third factor).

| rule | fan-out | patterns stored | oldest 40 | newest 40 |
|---|---|---|---|---|
| exact (net.py) | 96 | 20 / 100 / 300 | 0.10 / 0.06 / 0.05 | — |
| exact | 960 | 20 / 100 / 300 | 0.06–0.07 / 0.05–0.06 / 0.05 | — |
| exact, gated | 960 | 100 | 0.05 | 0.05 |
| "held co-activity neutral" (back term only for sources that just rose; the synthesis' variant) | 96 / 960 | 20–300 | 0.05–0.10 | — |
| naive co-activity, ungated, budget scaled to 20 | 960 | 20 / 100 / 300 | 0.93 (100 %) / 0.33 / 0.05 | — |
| naive co-activity, ungated, no budget | 960 | 100 / 300 | 0.20 / 0.08 | 0.20 / 0.08 |
| naive co-activity, gated, budget 20 | 960 | 100 / 300 | 0.99 (100 %) / 0.19 | 1.00 / 1.00 |
| **predictive co-activity, gated, signed, no budget** | 960 | 100 / 300 / 1,000 | **1.00 (100 %) / 0.94 (97 %)** / 0.25 | 1.00 / 1.00 / 0.79 |
| predictive co-activity, gated, signed | 96 | 20 / 100 | 0.60 / 0.14 | 0.59 / 0.25 |

The predictive co-activity rule is Δw_ij = η·s_i(t−1)·(s_j(t) − clip(d_j/θ, 0, 1)), where d_j is j's learned recurrent drive. It is the same rule as in §1.4, with "j on during a hold" allowed as a target.

**Lessons.**
1. The current rule, and the variant the panel considered, never form a completable pattern, at any load, fan-in or gating. This is PROVED by the timing law: held co-activity is never learned.
2. A co-activity term is necessary but not enough.
   - At the model's fan-in of 96, no rule completes even 20 patterns.
   - Without gating, the "other input" between items is learned as well and swamps the store.
   - A binding budget turns the store into a recency buffer: the newest 40 are kept at 1.00, the oldest 40 drop to 0.19.
3. Fan-in of about 1,000, plus a third factor, plus error correction, stores about 300 completable patterns in 4,000 units with no budget at all, and degrades gracefully near 1,000 (newest kept best). That is a concept memory.

### 1.6 Measurement E: sensory sequence memory ("hear the start, the rest comes out")

SIMULATED (`seqmem.py` → `seqmem.out`). Setup: real ear codes (3,831 informative neurons; a char's code has about 834 of them); recurrent fan-out 96 and budget 1.0, as in ear→ear; one char per tick. I cue a char and read out which char comes next.

| lines learned (distinct chars) | exact rule: next char top-1 / top-3 / two steps ahead | predictive variant | chance |
|---|---|---|---|
| 20 (82) | 0.156 / 0.233 / 0.086 | 0.111 / 0.211 / 0.071 | 0.012 |
| 100 (303) | 0.034 / 0.056 / 0.015 | 0.036 / 0.056 / 0.010 | 0.003 |
| 400 (761) | 0.012 / 0.024 / 0.016 | 0.007 / 0.023 / 0.003 | 0.001 |

The 2,000-line run was stopped as redundant.

"Hear the start, the rest comes out" works for a handful of lines and is gone by a few hundred, under either rule. In the sensory paths the limit is fan-in times code density (§1.7), not the error term.

Sensory sequence memory needs three things that cortex has:
- sparse codes (1–2 % active);
- fan-in in the thousands;
- context-dependent cells, HTM-style.

CITED: Hawkins & Ahmad 2016; Cui, Ahmad & Hawkins 2016 (Neural Computation 28): online, unsupervised high-order sequence learning with accuracy comparable to LSTM on streaming prediction.

### 1.7 Bounds (PROVED)

- **Associative capacity is proportional to fan-in.** The formula is p_max ≈ k·C / (a·ln(1/a)), with C the fan-in, a the active fraction and k ≈ 0.2–0.3 (CITED: Rolls 2013, via research/03).
  - Rat CA3 (C 12,000, a 0.02): about 38,000 patterns.
  - Our hippocampus (C 96, a 0.05): about 160 patterns, *even with the right rule*.
  - ear→ear (C 96, a 0.16): about 80.
  - An assembly layer with C 1,000 and a 0.02: about 3,200.
  - Active inputs from a stored pattern onto each member (a·C): 4.8 in our hippocampus, 15 in the ear, about 240 in rat CA3.
- **Budgets make memory short.** Under proportional rescaling, an old event's weight halves after about 3 newer events of the same program on the fast path and about 17 on the slow path. For a first syllable used in 9.4 % of replies (你) that is about 23 and 125 dialogues.
- **There is no learned inhibition anywhere.** In `net.py` the back term floors at the birth weight (≥ 0), decay shrinks the learned part toward 0, and budgets scale it by at most 1. So the learned part is never negative, and every inhibitory path is fixed (code reading).
- **Predictions compete with sensation.**
  - In the ear's top-k competition, a neuron whose parent is off scores at most 0.53, and one whose parent is on scores 1.0–1.53.
  - Learned ear→ear current (budget 1.0, 1.5–1.7 between normalisations) plus relay returns (+0.5) can lift a parent-off neuron above parent-on ones.
  - Because k is fixed, every expected-but-absent feature that wins pushes out one that is actually present.
  - v1's decisions read the input window and layer 0, so this is contained today. Any future part that reads the ear cortex inherits the problem.
- **Data volume.** All dialogue pairs together are about 600k chars, and a 1k-dialogue toy sees about 20k. Children hear millions of words a year (CITED: Hart & Risley 1995).

---

## 2. How the architecture differs from the real brain, and which differences matter

### 2.1 The big picture

The real brain is a large learning machine wrapped around a small innate scaffold. The scaffold is sensors, effectors, reflexes and, above all, learning machinery: plasticity rules gated by neuromodulators, several memory systems, and developmental programs. Most cortical representations are learned from experience.

v1 inverts this. The scaffold (turn-taking, timing, holding, articulation, stopping) is large and hand-built. The learned part is a thin layer of co-occurrence counts at a few hand-timed junctions. That is why the body is excellent and the content is weak: the learned part has no room to grow.

### 2.2 Differences one by one

| # | aspect | real brain | here | matters for the goal? |
|---|---|---|---|---|
| D1 | where codes come from | cortex learns its codes from the statistics of experience; infants learn speech categories and word boundaries by listening before they speak (CITED: Saffran, Aslin & Newport 1996; Kuhl 2004) | every content code is fixed (ear copy, T, mirrors); only Z's char templates learn | **matters most**: nothing can come to mean more than its sound |
| D2 | cortical hierarchy | A1 → belt → parabelt → STG/STS; invariance and integration windows grow up the hierarchy (CITED via research/01: Hasson et al. 2008; Murray et al. 2014) | 4 near-copy layers plus T (about 8 chars) | matters: no invariance, no multi-timescale context |
| D3 | cell types and inhibition | PV (gain), SST (dendritic gating), VIP (disinhibition); inhibitory synapses are plastic (CITED: Vogels et al. 2011) | global top-k; fixed tonic I/D; no learned inhibition at all | top-k is a harmless abstraction; **the lack of learned inhibition matters** (§1.4) |
| D4 | dendrites | a separate compartment compares predicted with actual drive; dendritic segments detect context (CITED: Urbanczik & Senn 2014; Hawkins & Ahmad 2016) | point units | matters, and is cheap to emulate: the learned drive already exists as a separate current |
| D5 | connectivity and sparsity | about 10⁴ synapses per pyramidal cell; CA3 about 12,000 recurrent; DG about 2 % active | fan-out 30–96; ear 10–40 % active; mirrors 5–8 % | matters: capacity scales with fan-in / (a·ln 1/a) (§1.7); the designer's newest idea (variable active counts) points the right way |
| D6 | thalamus | relays plus higher-order nuclei that gate communication between cortical areas; MD sustains PFC states (CITED via research/01: Schmitt 2017) | none in v1 | moderate now; needed later for routing and attention |
| D7 | basal ganglia | action selection by disinhibition, trained by dopamine reward-prediction error; gates PFC updating (CITED: Schultz, Dayan & Montague 1997; O'Reilly & Frank 2006) | hand-built GO plus winner-take-all; nothing learns value | matters: learning *when* and *what* from outcomes |
| D8 | cerebellum | a huge fixed expansion with an error-driven readout (Marr 1969; Albus 1971); forward models | T is such an expansion, but its readout counts | matters, and is the cheapest fix (§1.4) |
| D9 | hippocampus | EC → DG (sparse, pattern separation) → CA3 (symmetric plasticity, autoassociation; CITED: Mishra et al. 2016, 58 vs 4.5 stored patterns for symmetric vs asymmetric) → CA1; one-shot plasticity over seconds (BTSP; Bittner et al. 2017); replay to cortex | 1:1 mirror; asymmetric rule; 4.8 active inputs per pattern; no recall; no replay; drives no behaviour in v1 | matters: episodic memory plus consolidation is how brains learn continually |
| D10 | PFC | layered recurrent circuits; attractor and activity-silent working memory (Wang 2001; Mongillo, Barak & Tsodyks 2008); mixed selectivity (Rigotti et al. 2013); gated by basal ganglia; the longest timescales | a fixed bag of one line; no gating; no learning | matters; §2.3 explains why depth is not the root cause |
| D11 | language streams | dorsal stream: sound ↔ articulation; ventral stream: sound → meaning (CITED: Hickok & Poeppel 2007); prediction couples comprehension and production (Pickering & Garrod 2013) | the copy route is a crude dorsal stream; there is no ventral stream | matters: this is why replies are shuffled parroting |
| D12 | speech motor system | distributed articulatory codes (Bouchard et al. 2013); auditory-feedback control learned by babbling (DIVA, Guenther) | localist programs recruited by the teacher; fixed 3-tick beats | harmless now; matters later for general motor skill |
| D13 | time and dynamics | continuous time; theta oscillations track syllables (Giraud & Poeppel 2012); adaptation | synchronous 75 ms ticks; binary units | binary units: harmless. Ticks: harmless for behaviour, but they make every learned link timing-critical |
| D14 | plasticity windows and gating | STDP within ±20 ms (inside one of our ticks); eligibility traces of 1–10 s; BTSP; all gated by third factors (via research/02: Yagishita 2014; Gerstner et al. 2018) | exactly one tick; ungated except by the teacher | matters: no credit across seconds without hand-built pulses |
| D15 | co-activity plasticity | CA3 is symmetric; high-rate co-firing potentiates regardless of order (Sjöström 2001); cells with similar tuning become connected (Ko et al. 2011) | forbidden: held co-activity cancels out | matters: no attractors (§1.5) |
| D16 | error-driven plasticity | climbing-fibre LTD; dendritic prediction errors; prediction errors in sensory cortex (Keller & Mrsic-Flogel 2018) | none | matters (§1.4, §1.5) |
| D17 | neuromodulators | dopamine (reward-prediction error), acetylcholine (encoding vs retrieval; Hasselmo 2006) and noradrenaline (surprise, reset) gate plasticity and switch modes | none; the teacher gate stands in | matters: what to store, when, and whether to encode or recall |
| D18 | homeostasis | synaptic scaling, heterosynaptic plasticity, set points | budgets, set point, output reserve | harmless and good as abstractions, but a budget must not *be* the memory (§1.4, §1.5) |
| D19 | memory systems and sleep | working, episodic, semantic and procedural memory; sleep replay interleaves new with old (McClelland, McNaughton & O'Reilly 1995; Kumaran, Hassabis & McClelland 2016) | fixed working memory; unused hippocampus; budgeted counters; no sleep | matters |
| D20 | structural plasticity | spine turnover; adult neurogenesis only in DG; cortical neurons do not split | split/delete; program recruitment | harmless as engineering, but not a source of features |
| D21 | innate vs learned | small innate scaffold plus a cortex that learns | large hand-built scaffold plus a thin learned layer | matters: the proportion is inverted (§2.1) |
| D22 | development | months of listening, then babbling, imitation and words; contingent social feedback (Goldstein, King & West 2003; Kuhl & Meltzoff 1996; Takahashi et al. 2017) | a 60-char warm-up, then teacher-forced dialogue | matters: the auditory-motor map had to be engineered (the copy route) |
| D23 | body and world | words refer to things and actions in a shared world; social reward | audio only; arbitrary sounds; no referents | **matters most for meaning** (Harnad 1990; Bisk et al. 2020) |
| D24 | data and training regime | millions of words a year, mostly overheard; learning from both sides of every exchange | about 20k chars per 1k-dialogue toy, about 600k in the whole corpus; learns content only on its own taught turn | matters |
| D25 | scale | 10¹⁰–10¹¹ neurons | about 10⁵ units | not the bottleneck now: rule, wiring and data come first |

### 2.3 The PFC question (the designer's concern)

The designer is right that a real PFC is a deep recurrent structure and ours is not. But depth and recurrence are not what is missing.

**The old design was already multi-layer and recurrent, and it failed exactly as the rule predicts.** It had a receiving layer, a thinking layer and a thalamic loop (STATE #3–#7):
- The thinking layer learned about 1e-4, because held co-activity is never learned.
- The loop held the same content-free state after every sentence (hold ≤ across).
- Every 2-tick loop alternated with period 2.

**A real PFC works because of four things.**
1. Its inputs are already learned, meaningful codes from association cortex.
2. Its synapses learn under sustained activity (co-activity, short-term facilitation).
3. Basal-ganglia gating, trained by reward, decides what to load and when to release it (PBWM; the same brake-and-release motif as our motor area).
4. Its codes are shaped by what the task needs, through prediction errors and reward.

Adding layers without these four will reproduce the old failure, at greater cost.

**Recommended build order.**
1. The predictive rule and a learnable world.
2. A slow context layer with period-1 self-holds and fixed weights. Because its weights are fixed, it cannot drift.
3. Co-activity assemblies on top of it (§1.5).
4. Learned gating, once a reward signal exists.

Depth then comes from stacking timescales (fast T, slower context), as in the cortical timescale hierarchy.

### 2.4 Failures already seen, and what the brain does instead

| failure in this project | what the brain does | missing here |
|---|---|---|
| a held thought cannot teach "speak now" (think→go ≈ 0; STATE #7) | eligibility traces plus dopamine or plateau gating bridge seconds; the basal-ganglia "go" is learned from reward-prediction error | trace plus third factor |
| shared features dominate learned associations (next/other ratio 1.06–1.93) | sparse higher-level codes; predictable input is suppressed; error-driven plasticity stops learning what is already predicted | sparse learned codes; the predictive rule |
| the PFC loop held a content-free generic state | high-dimensional mixed selectivity over learned inputs; gated loading; content-specific attractors | learned inputs; assemblies; gating |
| period-2 alternation in 2-tick loops | continuous dynamics; slow NMDA currents | already fixed in v1 by period-1 self-holds |
| familiar recall ≈ frequency; memory ≈ the last 5–25 events per target | error-correcting readouts; sparse one-shot episodic memory; consolidation | the predictive rule with learned inhibition; DG/CA3; replay |
| free speech reinforces itself (你 → 的/了) when learning is on | corollary discharge marks self-generated input (Schneider 2018); learning from outcomes | prediction targets from the world; reward |
| replies parrot the heard chars in scrambled order (早上好你好吗 → 学你好上早吗) | a ventral meaning stream; production plans message → syntax → sounds (Levelt 1989) | a meaning layer; utterance planning |

---

## 3. Problems of the whole architecture, ranked by how much they block the goal

| # | problem | kind | seen or foreseen | evidence |
|---|---|---|---|---|
| 1 | **No learned representation on the content path.** The ear cortex copies its input; T is fixed; the mirrors are 1:1. Similarity = shared sound, so no categories, invariance or meaning, and B8 (generalisation) is impossible by construction. | fundamental | seen: never-heard content is at chance without echo | §1.2, PROVED |
| 2 | **The rule's function class.** It is a one-tick, first-order, excitatory-only counter. (a) No error term → frequency prior → binding budgets → memory of about the last 5–25 events per target. (b) No learned inhibition → nothing can be ruled out. (c) Held co-activity is never learned → no attractors, concepts or learned working memory. (d) The one-tick window means every association needs a hand-built pulse (P = T∧GO, E = L∧R2, Hc delay lines). | fundamental | seen: STATE #7; the thinking layer learned about 1e-4; verifier issues BL-S1 and ML-F1 | PROVED; §1.4–1.5 |
| 3 | **No learning from consequences.** Learning is switched off in free behaviour, and there is no value system. | fundamental | seen: the C11 gate; drift (你 → 的/了) when ungated | PROVED; CITED (minimal-learning verifier) |
| 4 | **The testbed cannot show or teach meaning.** It is ungrounded, its non-echo ceiling is near chance, and it holds about 600k chars. | fundamental for the goal, cheap to fix | seen | §1.3 |
| 5 | **Working memory holds one line's bag of chars.** No multi-turn context, no order or roles, no gating; it is reset by every line. | fundamental | seen (ARCH_v1 design debt) | PROVED |
| 6 | **Memory systems.** The hippocampus is wired too sparsely to recall (4.8 inputs per pattern; at most about 160 patterns), drives no behaviour and has no replay. Long-term memory is a budgeted counter (0.37–0.38 retained at +10k). | fundamental | seen | §1.4, §1.7 |
| 7 | **Generation is a bigram chain plus a bag copy**, so word order comes out scrambled. Whole familiar replies saturate at about 0.45, because only the first syllable sees the line. | fundamental for fluent content | seen | CITED (res_mem_s0.log; CONTENT §4.3) |
| 8 | **Fan-in of 30–96 and dense codes cap every associative path** at around 10² patterns, under any rule (§1.5, §1.6). | structural (parameters and compute) | seen in toys | PROVED, SIMULATED |
| 9 | **Additive predictions compete with sensation** in top-k layers, so the ear can hear what it expects. | structural | foreseen (contained in v1) | PROVED §1.7 |
| 10 | **Localist programs recruited only by the teacher.** No babbling and no feedback control; 4,096 slots for 3,970 chars. | structural; harmless now | seen | PROVED |
| 11 | **The turn-taking reflex** cannot yield or predict turn ends. F is energy-based, so it fails with noise or a second speaker. Latency is about 450 ms. | fixable | seen (pauses ≥ 5 ticks cause overlap) | SIMULATED (ARCH_v1 stress tests) |
| 12 | **Hand-derived margins** (the ÷10 forms, cue 12, C_slot 7, the budgets) must be re-derived for every new route. | fixable, but scales badly | foreseen | PROVED (ARCH_v1 §1.3) |
| 13 | **Waste and drift.** About 80 % of ear units never vary. The hippocampus, about 29 % of all units, drives no behaviour. T's set-point drift and split/delete shift the codes that slow memories are keyed on. | fixable | seen and foreseen | §1.2; CITED (ARCH_v1 risks) |
| 14 | **No goals, self-model, planning or evaluation.** There is no substrate for "thinking" in the AGI sense. | fundamental, long-term | foreseen | ASSUMED |

---

## 4. Does it have the capacity to reach the goal? The smallest changes that would give it

**The units are sufficient. The rule, the wiring, the learning signals and the data are not.**

**Binary k-cap units are not the limit.** Binary sparse networks with local rules already:
- learn high-order sequences online, with accuracy comparable to LSTM on streaming prediction (HTM: Cui, Ahmad & Hawkins 2016);
- form and compose assemblies (assembly calculus: Papadimitriou et al. 2020);
- learn nouns, verbs and their meanings from a modest number of grounded sentences (Mitropolsky & Papadimitriou 2023);
- learn competitive features approaching backprop quality (Krotov & Hopfield 2019).

Each of them uses something our rule lacks: co-activity learning under competition, or prediction-gated learning, usually with high fan-in and sparse codes.

**The rule, given its function class (PROVED from §1.4, §1.5, §1.7 and the timing law).** The steady-state weight of a link is a normalised count of "i ends as j starts" events, never below its birth weight. So the whole system can only learn this: a fixed surface code, then one layer of normalised co-occurrence counts at hand-timed junctions. That class:
- cannot represent meaning, because there is no learned code;
- cannot hold a learned thought, because there is no attractor;
- cannot rule anything out, because there is no negative weight;
- remembers little, because of the budgets;
- cannot improve from its own experience, because outside the teacher there is no target.

**The smallest capacity package.** Each item is necessary.

| change | what fails without it | evidence |
|---|---|---|
| 1. predictive (error-correcting) rule with signed, learned-inhibitory weights | memory capped by budgets; no normalisation; learning in deployment is unsafe | SIMULATED §1.4 (+0.34 one-shot; +0.33–0.37 at +3–6k) |
| 2. held co-activity as a target, plus fan-in ≈ 1,000 and sparse codes in assembly populations | no attractors, concepts or learned working memory | SIMULATED §1.5 (0 → 97 % of 300 patterns); PROVED §1.7 |
| 3. eligibility trace plus one broadcast third factor (teacher, surprise, reward) | every association needs a hand-built one-tick gate; no credit over seconds; without gating, assemblies drown in "other input" (§1.5) | PROVED (timing law); SIMULATED (gating) |
| 4. slow context layer plus a predictor trained on everything heard | no multi-turn state; about 10× fewer learning events; no learning during conversation | PROVED (event counting); ASSUMED (the benefit) |
| 5. grounded micro-world | meaning can be neither learned nor measured | SIMULATED §1.3 |

Changes 1–3 are one rule: the predictive rule with traces and a third factor, with held activity allowed as a target in assembly areas. Later additions: replay and consolidation, basal-ganglia value learning, and learned sensory features.

**Compared with mainstream AI.**

| capability | what works in mainstream AI | brain-plausible local counterpart | here |
|---|---|---|---|
| learning representations | self-supervised prediction (next token, masked or contrastive) | predictive coding (Rao & Ballard 1999; approximates backprop: Whittington & Bogacz 2017); HTM; competitive Hebbian learning | none |
| credit assignment through depth and time | backprop / backprop through time | feedback alignment (Lillicrap et al. 2016); e-prop, i.e. traces plus broadcast learning signals (Bellec et al. 2020) | one learned layer per path; one-tick window |
| working memory | attention over context; gated RNNs | PFC attractors plus basal-ganglia gating (PBWM, which is close to LSTM gates) | fixed one-line bag |
| fast memory | retrieval memories; fast weights; modern Hopfield networks (Ramsauer et al. 2020) | CA3 autoassociation; one-shot BTSP | mirror, no recall |
| continual learning | replay; consolidation regularisers (EWC: Kirkpatrick et al. 2017) | hippocampal replay; multi-timescale synapses (Benna & Fusi 2016) | budgets plus decay |
| acting from outcomes | reinforcement learning (actor-critic) | basal ganglia plus dopamine reward-prediction error | none |
| meaning | scale plus multimodal grounding | embodied, social learning | none |

---

## 5. What to optimise, in priority order

### 5.1 Steps

| # | change | what it buys | what it costs | check before any long run (pass criterion) |
|---|---|---|---|---|
| 0 | **Testbed.** Build a grounded micro-world (see below) plus overheard dialogues. Keep the corpus for body checks and familiar recall only. | meaning becomes learnable and measurable; echo scores 0 by design | a world generator and an "eye" front end (reuse Net3D); about 1–2 days | compute ceilings first: echo ≈ 0, ideal learners high; define held-out compositions |
| 1 | **Predictive rule with signed weights at the decision layer.** Replace the P→Y fast/slow counting first, then Hc→Y and E→Y. Keep budgets only as safety bounds. | memory capacity and normalisation (SIMULATED: one-shot 0.72 vs 0.35, +3–6k 0.84 vs 0.48); no more ×10 balancing between memory and copy | one row update per boundary over units with drive; a signed (interneuron) pathway; about 10 lines in `Path.learn` | replay on the recorded codes (done here, 2 seeds); then the integrated toy with ARCH_v1 checks 1–13 unchanged (the cue margin holds because LMS drive is bounded by its 0/1 targets); a self-reinforcement test with partner-target learning on in free speech |
| 2 | **Eligibility trace (λ about 0.7–0.9) and one broadcast third factor.** | associations seconds apart without new gate units; "learn only what matters" (assemblies need it, §1.5) | one trace per unit; one scalar | unit test: learn "end of partner line → my first char" with no P gate; zero spurious learning when the factor is off |
| 3 | **Predictor Π_c (one per program).** Trained on every heard char from both speakers; the target is the char actually heard next (detected by Z). Fixed Π_c → Y_c (the copy route's mirror map, generalised). | about 10× more learning events; learning during real conversation; one content route instead of three | a Π population plus a dense context→Π table, the same size as P→Y today | next-char accuracy within a few points of an ideal n-gram counter on held-out lines; accuracy on micro-world tasks against the ceilings |
| 4 | **PFC v2a: a slow context layer.** Fixed weights; period-1 self-hold tuned so 15–30 % of units turn over per line; read out by one-tick pulses. | multi-turn topic; the "thought range that moves" (item 29) | about 10k units | after line n+1, decode line n from the context (≥ 0.8); turnover inside the band; no period-2 |
| 5 | **PFC v2b and hippocampus v2.** Predictive co-activity rule with fan-in ≈ 1,000 and 1–2 % activity in PFC-own and CA3-own; a DG-like sparse expansion in front of CA3; replay in idle phases that trains Π's slow weights. | concepts, one-shot episodic memory, consolidation | about 20M synapses per area | `attractor4.py` at GPU size (≥ 300 patterns completed from 50 % cues); one exposure → recall ≥ 0.8 at +100 dialogues; retention at +10k with and without replay |
| 6 | **Basal-ganglia value learning.** A critic plus reward-prediction error; gates context loading and program choice. | learning from consequences; autonomy | critic units; rewards from the world | bandit and foraging tasks in the micro-world |
| 7 | **Learned sensory features.** Predictive co-activity with competition over the ear lines and the eye. | invariance; real audio and vision | moderate | unsupervised syllable clustering on noisy, multi-speaker input |
| 8 | **Body extras.** An F with the own voice cancelled (efference copy) plus an abort rule; learned prediction of turn ends. | yielding to the partner; faster turns | small | interruption stress tests |

**The predictive rule, stated precisely, for step 1 and beyond.** For a link i→j:
- d_j(t) is j's learned drive from the sources at t−1; `Path.current()` already computes it.
- target_j(t) = 1 if j fires at t because of its driver input (a chain, bottom-up input, the teacher's cue, a heard char).
- Δw_ij = η · g(t) · tr_i(t−1) · [target_j(t) − p_j(t)], with p_j = min(1, d_j/θ).
- tr_i is i's eligibility trace: 1 after i was on, multiplied by λ each tick (λ = 0 gives today's one-tick window).
- g is the broadcast third factor (teacher present, surprise or reward), default 1.
- Keep the back term only for sources that turn on *after* j, i.e. true "later → earlier".
- The learned part may go negative (read it as feedforward inhibition through interneurons).
- In assembly areas, "on during a hold" also counts as a target.

This keeps the designer's direction ("earlier → later strengthens, later → earlier weakens") and adds "already predicted → no more strengthening" and "predicted but absent → weaken".

**The micro-world for step 0.**
- A scene with 3 slots. Each slot holds an object (about 12 kinds), with a colour (6) and a count (1–3). It is seen through a fixed eye net for the whole exchange.
- The partner asks or tells things about the scene, e.g. "左边是什么" → "苹果", "苹果是什么颜色" → "红色", "有几个苹果" → "两个".
- Plus social scripts, and questions across turns about a scene that has since changed (tests working memory).
- Held-out object × property combinations test systematic generalisation (as in SCAN; Lake & Baroni 2018).
- A new object introduced once ("这是X") tests one-shot real-time learning.
- A slow change of the world rules tests continual learning.
- By construction, echo scores 0 and text-only learners score at chance on scene questions.

### 5.2 Block diagram (v2)

```
 WORLD (new): partner speech + a small scene the talk is about; overheard dialogues between others
   │ sound                                    │ scene
   ▼                                          ▼
 EAR front end (fixed) ── input energy → F/GO/CD (innate)      EYE front end (fixed 3D net, new)
   │ layer 0 (FEAT0)                                            │
   ├──► Z_c char detectors (learned; exist)                     │
   ├──► T: fast "just heard" (fixed expansion; kept)            │
   └──► CTX: slow context (new) — top-k, period-1 self-hold, 15–30 % turnover per line;
          later own assemblies: predictive co-activity, fan-in ≈ 1,000, 1–2 % active
                 │ one-tick read-outs at R2 / boundaries (timing hygiene kept)
                 ▼
 HIPPOCAMPUS v2: DG (sparse ≈ 1 % expansion of Z + CTX + scene) → CA3-own (predictive co-activity, one-shot,
   novelty-gated) — recall pulses at R2; replay in idle phases ─────────────► trains Π's slow weights
                 ▼
 PREDICTOR Π_c (one per program, new): drive = T/CTX read-outs + previous char + recall + scene;
   predictive rule (signed, trace, third factor); target = the char actually heard next
   (partner, overheard speaker, taught own voice) → learns while listening, in training AND in deployment
                 │ Π_c → Y_c fixed 1:1 (the copy route's auditory→motor mirror, generalised)
                 ▼
 BODY (innate, ARCH_v1 unchanged): GO → beat → programs Y (WTA) → D/I/M → mouth → own voice → ear;
   END, breath counter; corollary discharge closes plasticity for self-produced targets
 BROADCAST (new): surprise = Σ|prediction error| → plasticity gate + CTX input gate; later a BG critic: reward → RPE
```

**Loops in v2.**
1. The CTX self-hold (period 1).
2. Speak → own voice → ear → Z → Π: a prediction check, with plasticity gated by corollary discharge.
3. Hippocampus ↔ Π and CTX: recall while awake, replay while idle.
4. Later: basal ganglia → thalamus → CTX input gate.

### 5.3 What learns, and how

| block | learns? | how | timescale |
|---|---|---|---|
| ear front end, F, GO, CD, beat, programs, D/I/M, END | no | innate | — |
| Z (auditory mirror of programs) | yes | as in CONTENT (own voice is the teacher); later the predictive form | slow |
| T | no | fixed expansion | — |
| CTX | v2a: no; v2b: yes | predictive co-activity plus set point, fan-in ≈ 1,000 | slow |
| Π (predictor) | yes | predictive rule, signed, with trace and third factor | fast and slow components |
| DG | no | fixed sparse expansion | — |
| CA3-own | yes | predictive co-activity, one-shot, novelty-gated, forgets | fast |
| replay → Π slow weights | yes | predictive rule on replayed sequences | slow |
| basal-ganglia critic and actor (later) | yes | three-factor rule gated by reward-prediction error | medium |

### 5.4 Keep

- F reading the input energy; GO, CD, beat, programs, END and the breath counter; D/I/M slaved to the held program. All of these had 0 errors.
- One-tick read-out gates as timing hygiene.
- T as the fast expansion.
- The experience clock for forgetting.
- Budgets as safety rails only.
- Closing plasticity for self-produced targets (the C11 idea).
- The Z map.

### 5.5 Risks of these proposals

- **Predictive rule.** It needs a calibrated prediction threshold and signed pathways. At η = 1 it forgets fast (§1.4); use a normalised step of about 0.3.
- **Assemblies.** They may lock, which is the designer's fear. Counter this with k-cap, a set point, input gating, and an update test (A → pause → B).
- **Traces.** They can spread credit to irrelevant sources. They need the third factor and sparse codes.
- **Micro-world.** Templated language can simply be memorised. Use held-out compositions.
- **Predictor trained on partner speech.** It may learn a generic language model that ignores whose turn it is. Give CTX speaker and turn features, and score reply relevance, not only next-char accuracy.

---

## 6. Limits of this review, and files

**Limits.**
- The readout simulations replay the recorded decision-layer inputs offline (exact codes and protocol), not inside the integrated circuit.
- The attractor and sequence toys are simplified: one pattern or char per tick, no sliding window, no output reserve.
- The ceiling learners are shallow.
- No GPU runs; one corpus.

**Files** in `/tmp/claude-0/-root/e92d3d7d-5823-4f5c-a0ec-4fe302bff9af/scratchpad/arch/review/compute/`:
- `ear_linecopy.py` → `ear_linecopy.out` (§1.2)
- `ceiling.py` → `ceiling.out` (§1.3)
- `readout_rule.py` → `readout_familiar_s0.out`, `readout_familiar_s1.out`, `readout_never_s0.out` (§1.3, §1.4)
- `attractor.py`, `attractor2.py`, `attractor3.py` (stopped early: rows M 100 and 300 only), `attractor4.py` and their `.out` files (§1.5)
- `seqmem.py` → `seqmem.out` (§1.6)
- `voice_copy.py` (a copy of `world/voice.py`)

The readout script imports the content specialist's `decide.py` and `metric.py` read-only (with `python -B`, so no files were written there).

**Sources checked on the web for this review:**
- [Cui, Ahmad & Hawkins 2016, arXiv 1512.05463](https://arxiv.org/abs/1512.05463v2)
- [Mitropolsky & Papadimitriou 2023, arXiv 2306.15364](https://arxiv.org/abs/2306.15364)
- [Diederich & Opper 1987, PRL 58:949](https://link.aps.org/doi/10.1103/PhysRevLett.58.949)

The other citations are classic papers, or come via research/01–03 as marked.

---

## summary_zh

1. 总判断：身体部分（不抢话、按时开口、字清楚、会停）已经很好，别动；但"会学的部分"容量太小——照现在的学习规则和连线，怎么调参数都到不了最终目标。
2. 实测一：听觉区"皮层"里 99.5% 的神经元只是照抄某一根输入线，没学出任何特征，约 80% 的神经元从不变化。
3. 实测二：这份语料对"没听过的话"几乎没有可学的内容——去掉"重复对方的字"后，最强的统计方法也只有 0.06–0.10（瞎猜 0.05），重复整句是 0.27；在它上面看不出模型懂不懂。
4. 实测三：把"说第一个字"的学习改成"按预测误差改"（猜对了不再加强、猜了没发生就减弱），同样输入下听一次就记住 0.72–0.73（现在 0.32–0.38），隔 3–6 千段对话还记得 0.83–0.84（现在 0.46–0.50）；但必须允许学出"抑制"（负连接），只许正连接时只剩 0.05–0.07。
5. 实测四：现在的规则存不下任何"一群同时亮、能从一半线索补全"的模式（概念）；改成"预测误差 + 持续同亮也算 + 只在要记的时候学 + 每个神经元约 1000 根连接"后，300 个模式里 97% 能补全；只有 96 根连接时 20 个都补不全。
6. 和真实大脑最大的差别：大脑是"小的先天骨架 + 大量从经验学出来的表征"，我们正好反过来——骨架很大、手工搭好，学的只是几个路口的"先后计数"。
7. 前额叶简单只是症状：以前那套多层循环前额叶失败，是因为规则学不了持续的活动、没有"什么该记"的信号，不是层数不够；只加层会重复失败。
8. 改进顺序①：先换一个能学"意义"的小世界——模型看见一个小场景，对方问的和场景有关，答案不在问题里（重复对方的话得 0 分）。
9. ②学习规则加"预测误差"、允许学出抑制，再加几拍长的痕迹和一个全脑广播的"重要/意外"信号。
10. ③让模型一直预测接下来会听到什么（两边说的都学），说话就是把预测说出来——这样在真实对话中也能边说边学。
11. ④前额叶加一层慢慢移动的"想法范围"（跨句子记住话题），再在里面形成概念（同亮可学 + 约 1000 根连接）。
12. ⑤海马体改成稀疏、能补全、空闲时回放给皮层（巩固）；以后再加"奖励"系统学会何时做什么。
13. 每一步都有不用长跑就能做的小检查（报告第 5 节），身体部分保持不变。
