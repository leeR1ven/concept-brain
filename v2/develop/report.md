(Saved by the coordinator from the reviewer's final reply; the harness blocked the reviewer's own write.)

# ARCH_v2-develop: a developmental, memory-first whole architecture (designer `develop`)

The harness blocked writing `report.md` ("subagents should return findings as text"), so this is the full report. All code and outputs are in `/tmp/claude-0/-root/e92d3d7d-5823-4f5c-a0ec-4fe302bff9af/scratchpad/arch/v2/develop/` (file list in §13).

Labels: **PROVED** (argument with numbers), **SIMULATED** (numpy run; file and numbers), **CITED** (paper and finding), **ASSUMED**.

"Bench" means the shared bench v1.2, copied unmodified into `develop/bench_copy/`. Main regime: `tutor='recast'`, world seed 0, 300 stage-A dialogues, then 2,000 dialogues each of B, C and D.

---

## 0. Verdict

**What works (SIMULATED).**
- **Teacher-free hearing and babbling (stage A).**
  - Perceptual syllable categories form from heard speech with no teacher: 65 categories for the 65 stage-A characters, one character each.
  - Every heard line is recovered exactly and in order through the real ear: 400/400 lines, in both partner voices.
  - One pass over all 35,960 four-muscle sets binds a program to every category, with 0 wrong programs.
  - A babbling memory of all 3,970 producible sounds lets a later-heard syllable be imitated at once.
  - No muscle puppeting anywhere.
- **Grounded naming beats every reference (stage B).**
  - Bench: 0.54 overall, and 0.71 in the last third of the stage.
  - The earlier run r3 reached 0.71 overall and 0.88 in its last third, which passes the B milestone.
  - Bar (v1 MEM, force) 0.12; prior 0.14; n-gram 0.13.
  - Mechanics are perfect in every run: 0 mouth ticks while a partner speaks, latency 6, every syllable exactly 3 ticks, 0 garble, 0 extra turns.
- **Slot binding (stage C).**
  - "你要X吗 → 要X" scores 0.86 on the bench, against 0.49 for the n-gram and 0.04 for the bar. Never-heard X scores 0.44, where the n-gram gets 0.00 because it cannot copy.
  - At the item level (symbolic world), never-heard frame × filler scores 0.82–1.00.
- **Relational yes/no (stage D).** 这是X吗 → 是/不是 scores 0.77, against 0.48 for the prior, 0.49 for the bar and 0.30 for the n-gram.
- **Causal ablations.** The cortex readout, learned attention, the scene focus, the visual comparator and the BG copy gate each reach behaviour.

**What does not work (SIMULATED, honest).**
1. **Catastrophic interference across stages.**
   - The frozen agent answers stage-B probes at 0.51 right after B, 0.22 after C and 0.29 after D ("what" 0.58 → 0.06 → 0.00).
   - Symbolic world: never-heard offers scored 1.00 after C and 0.38 after D.
   - Neither the replay nor the curriculum protects old skills yet.
2. **Learning on in untutored free use destroys answers.** Bench: accuracy 0.56 → 0.00 after 600 untutored dialogues. The partner's next question is learned as the "answer" to the previous one.
   - A fix (my own reply becomes a line of the dialogue, plus turn-role gated units) solved it in the symbolic world: 0.00 → 0.92.
   - On the bench the same fix cut learning 0.40 → 0.11, for a reason I could not isolate in time, so it is off in the final runs.
3. **Sleep replay does not consolidate.** In M1 it made one-shot retention worse: 0.98 → 0.62 at delay 0.
4. **Several D tasks fail on the bench.**
   - One-shot naming of a novel object: 0.00.
   - "刚才…是什么" (before): 0.01.
   - Name recall: 0.17, the prior level.
   - Symbolic world: name 1.00 and one-shot 1.00, so the mechanism works at the item level but not yet with sound and the bench's 40 single-exposure objects.
5. **Scene attribute questions are near chance on the bench.** Colour 0.14 (prior 0.14); count 0.36 (prior 0.32).
6. **Two learned parts are passengers in the final configuration.** The visual recognition group VR and the slow context CTX change nothing (CTX even lowers name recall). By the brief's own rule they must be fixed or removed.

**Bottom line.** The developmental route (hear → categories → babble → inverse map → learn answers by listening and recasts → ground words by joint attention) produces real, ablation-proven competence on stages A–C and on relational D. But the prototype is not yet a lifelong learner. Interference across stages and drift in free use are the two blocking problems; both are fundamental and are where the next work must go (§11).

---

## 1. Design in ten lines

1. **Keep the body.** The bench's `body.py` (= ARCH_v1: F/GO/CD, 3-tick beat, program WTA, D/I/M, END rule, breath counter) is unchanged. The mind supplies a score per program and for END at each boundary.
2. **Hearing first.** The fixed ear layer 0 (FEAT0) feeds perceptual syllable categories Z.
   - They form by competitive learning with a vigilance threshold (0.75), sampled by a syllable clock that is phase-reset at speech onset.
   - No teacher is involved.
3. **Babbling builds the inverse map.** One pass over all muscle sets. The own voice of each set is matched to the heard categories, and the best match ≥ 0.85 binds the set as that category's program. Every producible sound is kept in a babbling memory, so later syllables are imitated at once.
4. **Ordered utterance memory (PFC-1).** The previous line (the prompt) is held item by item with a one-shot chain (the designer's "earlier → later" rule at η = 1) until the next line ends. The current line builds beside it.
5. **Mixed selectivity (PFC-2).** Conjunction units are recruited on the first co-occurrence of two context features: prompt word, last said item, position, copy candidate, attended scene, comparator match.
6. **Cortex = signed error-driven readouts.** "Predicted → no more strengthening; predicted but absent → weaken." They learn by listening: every heard line is predicted item by item from the previous line. Speaking is reading the same readout at the body's boundaries.
7. **BG output gates (PFC-3).** Learned gates decide whether to copy the next item of a held buffer: the prompt, the recalled answer, or the recall's completion. This is slot binding.
8. **Hippocampus = episodic index.** Every heard line is stored once (DG k-WTA key + one-shot cell). It is recalled by similarity when the next line is expected, replayed in sleep, and fades slowly.
9. **Attention and grounding.**
   - The tutor's pointing gives the attended slot (innate orienting).
   - Otherwise an internal comparator decides: each word's learned visual expectation at each slot, against the scene.
   - Slow context (PFC-4) is a leaky k-WTA layer per line with a homeostatic set point.
10. **Third factors.**
    - "Heard from another voice": the target of every listening update. The corollary discharge closes it for the own voice.
    - The partner's reaction: reward → critic RPE → own-choice update.
    - Joint attention and confident comparator matches teach attention and the word→visual expectations ("primes").
    - Sleep is the offline mode.

---

## 2. Every area

Format: prototype / GPU size.

| area | role | size | activity | fan-in | persistence | learns from (rule) |
|---|---|---|---|---|---|---|
| EAR-0 + FEAT0 | fixed ear layer 0 (v1, exact) | 1,920 (FEAT0 1,153) / same | 40 % top-k | 4 | 1 tick | — (innate) |
| syllable clock | samples every 3rd tick from speech onset + 2 | 1 counter / same | — | window energy E | until silence | — (innate; plan: peak detector) |
| Z categories | perceptual syllables (all voices) | ≤ 256 / 4,096 | one-hot pulse per syllable | 1,153 (dense template) | 1 tick | competitive + vigilance 0.75, running average 0.1 (R5) |
| babbling memory | own-voice sound → muscle set | 3,970 sounds / same | — | 1,153 | permanent | one-shot, during babbling |
| PROG | syllable programs (body) | ≤ 320 / 4,096 | WTA, 1 per beat | mind scores | 3 ticks | bound by babbling / imitation |
| BUF_prev, BUF_cur (PFC-1) | ordered utterance memory | 2 lines × ≤ 24 items / same | item latches | Z | one line, held through the reply | one-shot chain (R1) |
| MIX (PFC-2) | recruited conjunctions | ≤ 600k slots (≈ 50–75k used after A–C) / 2M | ≈ 100–400 active per event (< 1 %) | 2 (coincidence of two features) | per event | recruited on first co-occurrence, deleted when unused (R3) |
| GEN | readout MIX → programs + END (cortex) | used MIX rows × 257 / sparse rows × 4,097 | drive | MIX active | weights | signed error-driven (R2) |
| GATE (PFC-3) | BG output gates: COPY_prompt, COPY_next, COPY_comp | MIX × 3 | clip(drive) | MIX | per event | R2 (target = "the heard item was the buffer's next item") |
| ATT | 3 slot units | 3 / 3 | WTA | pointing code; word→slot; comparator | one line | outstar word→slot (R7), taught by pointing or the comparator |
| PRIME | (slot, word) → expected eye units | 768 × 1,024 / 12,288 × 1,024 | — | 1 | weights | outstar, plasticity 1/n (R7), taught by pointing only |
| VR | words recognised at the attended slot | ≤ 6 of V | — | PRIME vs scene | one exchange | — (computed) |
| HIP DG | pattern separation | 3,000, k 60 / 50,000, k 1,000 | 2 % | 38 (bag 10, first 4, last 4, scene 10, CTX 10) | — | fixed random |
| HIP cells | one-shot episodes (lines) | 3,000 / 100,000 | WTA recall | DG key | γ 0.9995 per line, ×0.9 per sleep | one-shot (R4) |
| CTX (PFC-4) | slow dialogue context | 800, k 24 / 4,000, k 120 | 3 % | 6 (items + scene) | leaky λ 0.9 per line, homeostatic set point | fixed (plan: autoassociative assemblies) |
| CRITIC | value of the reply context | per MIX unit | — | MIX | weights | RPE δ = r − V (R6) |
| BODY | F, F2, GO, CD×4, R1, R2, boundary, WTA, Hp, D/I/M, END, counter | v1 | — | — | — | — (kept) |

---

## 3. The rules, exactly

All rules are local: a synapse sees its pre, its post, its own weight and at most one broadcast signal.

- **R1, one-shot sequence binding (PFC-1).**
  - When item b is detected right after item a in the same line, the link a → b is set (and line start → the first item).
  - The buffer clears at the next line's onset and rotates (current → previous) at the line's end.
  - Read-out: the successor of the last said item, at a position not yet used in this reply.
- **R2, signed error-driven readout (GEN, GATE).** For output j with the set A of active sources:
  - d_j = Σ_{i∈A} w_ij + (the other routes into j), and p_j = clip(d_j, 0, 1);
  - **Δw_ij = η · g · m_i · (t_j − p_j) / |A|**, with η = 0.35 (GEN), 0.5 (GATE), 0.2 in sleep.
  - t_j = 1 if j was driven by its teaching input (the heard item's mirror; END when the heard line ended).
  - m_i = maturation 1/(1 + use/M) (optional).
  - The prediction is the summed drive of all routes, so an item already predicted by the copy gate teaches GEN nothing: the filler stays out of the frame's weights.
  - The negative part is learned inhibition through interneurons, which keeps Dale's law.
- **R3, recruitment and deletion (MIX).** A unit is created for a pair of features the first time they co-occur; units unused for a long time are deleted.
- **R4, episodic index (HIP).**
  - Key = the DG k-WTA code; the cell is recruited once (binary key; content = the line, its scene and attention, a link to the next line's cell).
  - Recall = WTA over cells of overlap · √σ, with overlap ≥ 0.2; σ ← 0.9995 σ per stored line.
  - Recall gives two candidate sources: NEXT (the line that followed) and COMPLETION (the recalled items the cue lacks, in order).
- **R5, perceptual categories (Z).**
  - m = |f ∧ T| / |T|. If m ≥ 0.75 → that category, and T ← 0.9 T + 0.1 f; else a new category.
  - Babbling binds a category to the babbled sound with the best match ≥ 0.85.
- **R6, critic / actor.**
  - δ = r − V(first-step context); v += 0.2 δ/|A0|.
  - For every own choice (A, y) in that reply: w_iy += 0.3 δ/|A|.
- **R7, outstar (PRIME, ATT).**
  - w_i ← w_i + m_i (t − w_i), with m_i = max(1/n_i, 0.02): an exact conditional frequency whose plasticity falls with use.
  - "Confident" = w ≥ 0.6 after ≥ 6 teaching events, on eye units whose own mean activity is < 0.4.
- **Sleep.** Every 300 exchanges, 200 episodes are sampled ∝ σ. Each replays its NEXT line as heard targets, with the stored context reinstated (R2, η 0.2). Then σ ×0.9.

---

## 4. The PFC, in developmental order (the designer's concern, and both new ideas)

| layer | what | stage | gate / persistence | measured (SIMULATED) |
|---|---|---|---|---|
| PFC-1 | held prompt (items + order) + current line | A/B | held exactly through the reply and any silence; replaced only at another speaker's line end (input gate = turn units) | held 4/4 boundaries; replaced exactly by a new line; 20-way decodable 1.00 for any silence length; period-2 index 0; drift 0 (no learned weights inside; PROVED) |
| PFC-2 | recruited conjunction layer, read at every boundary | B | per event | required: additive readouts cannot compute "heard X ↔ seen Y" (PROVED in the develop review) |
| PFC-3 | BG output gates over the held buffers | C | R2 | copy gate: never-heard offers 0.44 bench (n-gram 0.00); 0.82–1.00 symbolic; ablation: rituals 0.77 → 0.12 |
| PFC-4 | slow context CTX (λ 0.9 per line, homeostatic set point, novelty-weighted input) → hippocampal key | D | leaky k-WTA | see the M7 rows below |
| PFC-5 (planned) | learned autoassociative assemblies in CTX; inner speech | E+ | self-excitation > cross-drive | period-2 condition below |

M7 measurements for the slow context CTX:
- **Before the fix it was generic**, like the old PFC: states 150 lines apart overlapped 0.456 (random: 0.030), and the newest line decoded 0.18 (20-way).
- **After a homeostatic set point and novelty-weighted input**, the overlap is 0.052 and the period-2 index is −0.14.
- 20-way decoding of the line heard d lines ago (chance 0.05): 0.60 at d = 0, 0.41 at d = 1, 0.28 at d = 2, 0.19 at d = 4, 0.09 at d = 8.

**Holding a thought by reverberation (the designer's second idea).** The old loop's four failure modes, in my design:
1. **Generic state.** PFC-1 holds the exact items; CTX was generic until the set point was added (now 0.052).
2. **Hold vs update.** The input gate is tied to the turn units: hold through the reply and silence, update at the partner's line end.
3. **Period-2** (PROVED + SIMULATED, `test_modules.py M5`).
   - In a synchronous binary loop with reciprocal links x1 ↔ x2, leak alone never removes the 2-cycle: the freshly driven pattern has potential D/(1+λ) against λD/(1+λ) for the stale one, for every λ < 1.
   - Simulated: leak 0.75 and 0.95 still give J(t,t+2) = 1.00 and J(t,t+1) = 0.00; self-excitation 0.5·D still alternates.
   - Only self-excitation 1.5·D or autoassociative weights remove it (J(t,t+1) = 1.00).
   - This corrects the synthesis claim that "leaky units kill period-2". My PFC has no reciprocal synchronous loop; any future recurrent layer must meet this condition.
4. **Drift while holding.** No learned weights in the held register; CTX has fixed weights.

Capacity: 2 lines exactly in PFC-1, plus a decaying gist in CTX, plus thousands of lines in the hippocampus (not active).

**The PFC as a recurrent next-input predictor (the designer's first idea).** GEN already is the "top state predicts the next input" layer: every heard item is a prediction target. I tested which context serves it best, with my rule on the corpus. Split as in `coord/ngram.py`: 5k training dialogues, 1k held-out, next-char top-1, outputs limited to the 1,500 most frequent characters (`nextchar.py`).

| context of the readout | fixed η 0.3 | maturing plasticity 1/n | exact n-gram (coordinator) |
|---|---|---|---|
| last char (bigram) | 0.126 | **0.228** | 0.234 |
| last 2 chars (recruited pair) | 0.217 | **0.244** | 0.249 |
| last char + recurrent top state (k-WTA of fixed random W_in·c(t) + W_rec·R(t−1)) | 0.178 | 0.199 | — |
| recurrent top state alone | 0.146 | — | — |

- A local error-driven rule with maturation matches exact n-gram statistics within 0.6 points. Hebbian counting over distributed codes lost about 8 points (coordinator).
- A fixed random recurrent state is a lossier context than an explicit conjunction of the last items at this data size.
- My recommendation: the PFC's prediction layer = an ordered buffer + recruited conjunctions + slow context. A learned recurrent layer can come later, when data volume makes longer contexts pay.

---

## 5. BG / value, and integration with the body

- **Action selection.** The body's WTA at the beat.
  - Score = GEN drive + gate × candidate.
  - END wins if its score ≥ the best program's ('max' rule).
  - Action threshold (floor) 0.3: speak only confident predictions. This also keeps the agent in the answering role: an answer is predictable from its prompt, the partner's next question is not.
- **Output gating.** Which held content to read out is learned (R2): the BG part of PBWM (CITED: O'Reilly & Frank 2006).
- **Value.** Critic + RPE on the own reply's choices (R6). Unit test: reward-only learning 1.00 vs 0.18.
- **Body integration.**
  - The scorer is called at boundaries; a released program advances the own line (efference copy).
  - The corollary discharge closes perception of the own voice, so own speech is never a teaching target.
  - Babbling is a play period.

---

## 6. Developmental stages and milestones

| stage | added structure | milestone | result |
|---|---|---|---|
| A — hear and babble | categories, syllable clock, babbling memory, programs | ≥ 90 % of heard syllables categorised and producible | **PASS** — 65/65 categories (each one char); 65/65 programs, 0 wrong; 400/400 lines recovered in order; all 140 chars needed by A–D producible after one pass |
| B — naming with a scene | GEN, MIX, attention, primes/comparator, focus | naming ≥ 0.8 | r3: last third 0.88 (**PASS**); final run: last third 0.71 (FAIL); symbolic 1.00 |
| C — item-based frames | BG copy gates over PFC-1 | never-heard frame × filler ≥ 0.7 | bench FAIL (offer 0.44, taste 0.10); symbolic **PASS** (offer 0.98–1.00, taste 0.82–0.85, like 1.00) |
| D — relations and memory | hippocampal NEXT/COMPLETION routes, CTX | relational ≥ 0.8; one-shot ≥ 0.6 | check 0.77 seen 0.82 (near); one-shot 0.00 bench / 1.00 symbolic (FAIL on bench) |

Stage lengths were fixed at 2,000 dialogues. A strict milestone gate would have kept the bench agent longer in B and C. A job testing whether B passes with more time was sent to the server but could not finish there (§12).

---

## 7. Results on the shared bench

Seed 0, recast. **Mine** = the final run `final_a`. Reference = the bar (v1 MEM, force) / prior / n-gram, from the bench README.

| task | mine (held-out) | bar | prior | n-gram |
|---|---|---|---|---|
| B: what | 0.59 (seen 0.69) | 0.06 | 0.06 | 0.03 |
| B: slot | 0.41 (seen 0.48) | 0.00 | 0.04 | 0.06 |
| B: where | 0.59 (seen 0.66) | 0.33 | 0.33 | 0.32 |
| **B: all** | **0.54 (held-out 0.09)** | 0.12 | 0.14 | 0.13 |
| C: offer | **0.86 (held-out 0.44)** | 0.04 | 0.09 | 0.49 |
| C: ritual | 0.93 | 0.51 | 0.21 | 0.99 |
| C: taste | 0.76 (held-out 0.10) | 0.46 | 1.00 | 1.00 |
| C: count | 0.36 | 0.08 | 0.32 | 0.30 |
| C: colour | 0.14 | 0.08 | 0.14 | 0.19 |
| **C: all** | **0.62 (held-out 0.20; no-scene 0.86)** | 0.19 | 0.31 | 0.54 |
| D: check | **0.77 (seen 0.82, held-out 0.48)** | 0.49 | 0.48 | 0.30 |
| D: name | 0.17 | 0.00 | 0.17 | 0.17 |
| D: before | 0.01 | 0.01 | 0.08 | 0.09 |
| D: oneshot | 0.00 | 0.00 | 0.00 | 0.00 |
| **D: all** | **0.48** | 0.21 | 0.24 | 0.16 |

- **Learning in use** (first → last third of each stage): B 0.43 → 0.71; C 0.61 → 0.73; D 0.43 → 0.49. The bar goes B 0.13 → 0.11 and C 0.34 → 0.23.
- **Mechanics** in every run: 0 overlap, latency 6, 0 non-3-tick syllables, 0 garble, 0 extra turns.
- **Other runs.**
  - The earlier run r3 (older bench copy, stricter prime thresholds) had B 0.71 (seen 0.79; what 0.81, slot 0.76, where 0.78) and C 0.66.
  - Variant `final_b` (identity-free gates + candidate-familiarity class) is within 0.01 of `final_a` everywhere. It did not fix one-shot naming.
- **My own recast-mode baselines** (server, older copy): B n-gram 0.18; C n-gram 0.66; D prior 0.34. Lookup, echo and last are ≈ 0 on B and D.

---

## 8. Ablations: every learned part must reach behaviour

Frozen agent after D; probe-only schedule; ~20–60 replies per task, so differences of about ±0.07 are noise. File: `out/ablate_final_a.txt`.

| part removed | B (full 0.29) | C (full 0.50) | D (full 0.31) | verdict |
|---|---|---|---|---|
| GEN (cortex readout) | 0.01 | 0.20 | 0.08 | **essential** |
| learned attention | 0.04 | 0.43 (colour, count → 0) | 0.34 | **essential for scene questions** |
| scene focus F | 0.08 | 0.17 | 0.26 | **essential** |
| comparator / primes | 0.19 (where 0.54 → 0.14) | 0.36 (offer 0.66 → 0.42) | 0.14 (check 0.61 → 0.19) | **essential** |
| BG copy gate (prompt) | 0.31 | 0.38 (ritual 0.77 → 0.12) | 0.32 | reaches behaviour in C |
| hippocampal recall routes | 0.28 | 0.46 | 0.26 (name 0.10 → 0.00) | weak |
| whole hippocampus | 0.30 | 0.47 | 0.30 | weak |
| visual recognition VR | 0.27 | 0.52 | 0.33 | **passenger** in this configuration |
| slow context CTX | 0.31 | 0.51 | 0.34 (name 0.10 → 0.43) | **harmful** to name recall here |

**Interference** (`out/interf_*.txt`), B probes on the frozen agent:

| agent saved after | B all | what | slot |
|---|---|---|---|
| stage B | 0.51 | 0.58 | 0.49 |
| stage C | 0.22 | 0.06 | 0.00 |
| stage D | 0.29 | 0.00 | — |

So the ablation numbers above (on the D agent) understate what each part did when it was learned.

---

## 9. Unit and module tests

**Unit tests** (`test_rules.py`): 16/16 PASS.
- R1: ordered copy of 2,000/2,000 random lines, repeats included (好不好吃, 哈哈哈).
- R2:
  - one-shot drive 1.000;
  - no change when already predicted;
  - predicted-but-absent weakens (0.50 → 0.25);
  - g = 0 → no change.
- R2, lifetime recall with frequent-target noise:

  | variant | all facts | oldest quarter |
  |---|---|---|
  | signed | 0.99 | 0.96 |
  | excitatory-only | 0.48 | 0.22 |
  | counts | 0.20 | — |
  | signed + maturation | 0.99 | 1.00 |

- R3: units recruited once; unused units deleted.
- R4:
  - one-shot exact recall 800/800;
  - partial cues 1.00 / 0.99;
  - pattern separation: Jaccard 0.94 → 0.77;
  - half-life as designed.
- R5: template detection 1.00.
- R6: reward-only bandit 1.00 vs 0.18 without the RPE.
- R7: outstar error 0.040.

**Sound level** (SIMULATED; `feat_sep.py`, `audit_order.py`, `audit_bind.py`, `babble_stats.py`).
- FEAT0 separation:
  - the same char in different voices matches ≥ 0.907 (mean 0.945);
  - different chars match ≤ 0.624 (mean 0.337);
  - misaligned windows ≤ 0.557.
- Of the 140 chars needed by A–D, 94 are produced by exactly one of 35,960 muscle sets.

  | babbling strategy | 35,960 babbles | 100k | 200k |
  |---|---|---|---|
  | random with replacement | 100/140 | 133/140 | 140/140 |
  | one pass without repeats | 140/140 | — | — |

- First-match binding left 2/67 wrong programs (客→全, 气→俄). Best-match binding after the pass leaves 0. Imitation from the babbling memory: 0 errors.

**M1, one-shot facts and retention** (symbolic; `out/m1_final.txt`). Exact recall of a fact heard once ("N喜欢什么 → X"), by delay in exchanges:

| condition | 0 | 100 | 400 | 1,500 |
|---|---|---|---|---|
| hippocampus, no sleep | 0.98 | **0.75** | 0.13 | 0.10 |
| no hippocampus | 0.93 | 0.23 | 0.13 | 0.12 |
| hippocampus + sleep | 0.62 | 0.42 | 0.17 | 0.05 |

- The index carries memory over ~100 exchanges (memory-first ✓).
- Nothing survives 400+ exchanges: the 600-cell test capacity is overwritten.
- **Sleep FAILS.** It samples ∝ strength, so the 60 facts get about 0.5 replays each per sleep; the downscaling ×0.9 weakens them; the replayed background episodes interfere.

**M4, slot binding.** Symbolic held-out: offer 0.98–1.00, taste 0.82–0.85, like 1.00, unseen object-colour combinations 1.00. The no-copy-gate ablation run did not finish (server timeout); the bench ablation (rituals 0.77 → 0.12) stands in for it.

**M5 and M7, hold and period-2.** See §4.

**M6, drift in free use** (symbolic, 2,000 untutored exchanges with learning on):

| configuration | accuracy before → after |
|---|---|
| final configuration | 0.99 → 0.00 |
| + own reply as a line + contingency window (10 ticks) | 0.99 → 0.42 |
| + turn-role gated units | 0.99 → 0.92 (0.84 of replies unchanged) |

- Bench, final configuration: 0.56 → 0.00 (`drift_bench.py`).
- On the bench, the own-reply rotation cut stage-B learning: 0.40 → 0.16, and 0.11 together with role units. A trace confirmed that contingency detection works (recast gap 6 ticks, next question 16–27 ticks), but I could not find the mechanism in time. Open.

**Symbolic curriculum** (`run_dev.py`, 3 slots, retinotopic eye).
- B test: 1.00 / 0.96 / 0.98.
- C seen: 0.94–1.00.
- D: name recall 1.00, one-shot 1.00 (prequential 0.11 → 0.94), check 0.64, before 0.14.
- With the bench-like flat eye, B reached 0.94 / 0.98 / 0.80 after the base-rate fix.

---

## 10. Iteration log: what failed, why, and the fix (in order)

1. **Naming slow (0.21).** The focus took all ~80 scene units, and attention learned only from prompts. Fix: focus on the attended slot's part of the eye, and learn attention from every line heard under joint attention → 0.80.
2. **Per-branch normalisation made it worse** (0.26). Single-unit pathways learned the marginal answer distribution fastest. Reverted.
3. **Attention as a comparator.** Word→visual primes × scene per slot, plus a direct word→slot map → slot 0.98 (symbolic).
4. **A self-reinforcing attention loop.** Internal attention taught the word→slot map, which then drove internal attention (everything drifted to slot 0). Fix: only pointing or the comparator may teach.
5. **Primes learned by the joint error rule were uninterpretable** (credit split across a line's words). Fix: outstar (R7) with 1/n maturation.
6. **The comparator summed all prompt words' expectations**, so base rates added up to "confident" everywhere. Fix: per-word matching (max over words).
7. **"Hub" eye units** (in several random codes, on in most scenes) matched every word. Fix: count only units whose own mean activity is < 0.4 (local), and require ≥ 6 units.
8. **Visual recognition by analysis-by-synthesis** (words whose expected appearance matches the attended slot): bench "what" 0.10 → 0.45.
9. **Two babbling faults.** First-match binding (2/67 wrong) and a variable-shadowing bug that misaligned the babbling memory. Fix: best-match binding after the pass; 0 errors.
10. **Primes taught by the comparator's own attention formed a loop**: degradation within a run (0.42 → 0.39). Fix: primes learn only from pointing → bench B 0.35 → 0.71 (last third 0.88).
11. **Free-use drift.** See M6 (open).
12. **The slow context was generic** (0.456). Fix: homeostatic set point → 0.052.
13. **"Check" questions name a different object than the pointed one**, which diluted the primes below the 0.9 threshold. Fix: threshold 0.6 with ≥ 6 events → check 0.44 → 0.77. Probably also why final B < r3 B.
14. **One-shot 0.00 on the bench.** The gates saw the candidate's identity, and every novel object is new. Identity-free gates did not fix it, so the cause is unresolved: most likely the recall does not select the "这是X" statement episode.
15. **Next-char.** A fixed η gives recency-weighted estimates (0.126). Maturation → 0.228 against 0.234 for the exact bigram.

---

## 11. Risks and open problems

**Fundamental.**
1. **Interference across stages / continual learning.** The fix must come from:
   - interleaving earlier stages (bench strings like 'BC', 'BCD');
   - replay that prioritises old and unconsolidated episodes (prediction error), not strength;
   - per-unit maturation of GEN (shown in the R2 unit test: oldest quarter 0.96 → 1.00);
   - context-dependent gating (CITED: Masse, Grant & Freedman 2018: context-dependent gating alleviates catastrophic forgetting).
2. **Drift in free use.** Learning by listening needs a contingency third factor: learn a heard line as an answer only if it is a contingent response to the previous turn (CITED: Goldstein, King & West 2003: responses within ~1–2 s drive vocal learning). Plus turn-role gated units. Shown to work symbolically; bench interaction unresolved.
3. **Sleep that consolidates.** Prioritised replay, several passes per novel episode, and no downscaling of unconsolidated traces.

**Fixable.**
- Scene attribute words (colour, count) are never heard under pointing in this curriculum, so they are never grounded. Grounding them by the comparator created a loop; it needs a stricter, loop-free teacher (e.g. only when the attended object was located by a different word).
- VR and CTX are passengers here; CTX in the hippocampal key hurts recall. Fix or remove.
- "Before" needs reinstatement of a recalled scene into the focus. Not built.

**Design debts.**
- The MIX pair schema is hand-chosen (task-shaped). Plan: recruit conjunctions where errors persist.
- MIX growth on the adult corpus needs deletion and maturation.
- The action threshold stands in for a learned GO gate.
- The hippocampus is already at its 3,000-cell cap after ~2,300 dialogues; GPU size must be ≥ 100k.
- Babbling relies on enumerating a finite motor space. Plan: lawful physics + auditory-error-driven search (DIVA).

---

## 12. GPU-scale cost, kept/changed vs the framework, fixed mechanisms

**Cost (PROVED arithmetic from measured prototype sizes).**
- Prototype: ~0.15 s CPU per dialogue including the world; MIX 52.6k units after A+B and 74.8k after C; 65–67 categories.
- GPU instance (4,096 programs, adult corpus), memory:

  | part | estimate |
  |---|---|
  | MIX | ≤ 2M units, deletion of unused units |
  | GEN | sparse rows, top-64 outputs per unit, fp16 ≈ 0.5 GB |
  | HIP | 100k cells × 50k-bit keys ≈ 0.63 GB |
  | PRIME | 25 MB |
  | Z | 9 MB |
  | babbling memory | 0.6 MB |
  | **total** | **≈ 1.2 GB** |

- Per tick:
  - ear and Z ≈ 1.6M MAC;
  - per event ≈ 500 hashed keys + 32k sparse MAC;
  - recall ≈ 5 G bit-ops per line (~1 ms on a V100).
- This is ≪ 1 ms per 75-ms tick: more than 50× real time per agent, so many agents or seeds can be batched on one GPU.
- The server's CPU quota (12 shared cores, ~15× slower than local for this code) is useless for it; a torch port is needed.

**Kept from the framework.**
- binary units and the fixed input layer;
- "earlier → later" sequence links (R1);
- hippocampus = fast chain region, cortex = slow;
- global downscaling as the single clean-up (in sleep);
- neuron generation/deletion (category vigilance, MIX recruitment and deletion);
- "strongly linked → positive weight, weakly linked → negative" (signed R2; framework item 14(1));
- the PFC holds the gist, the hippocampus the details;
- no speech area: all routes drive the same programs.

**Changed from the framework.**
- Readouts learn from prediction error; the outstar is used for expectations.
- The hippocampus is long-lasting, with replay.
- No teacher forcing.
- The PFC is register + conjunctions + gates + slow context, not a mutually connected pool.
- Fixed top-k is replaced by event-driven computation (top-k kept in ear 0, DG and CTX).

**Fixed (innate) mechanisms: 16.** Each with a justification and a plan to learn it.
1. body (v1);
2. ear 0 + FEAT0;
3. syllable clock (theta entrainment; CITED: Giraud & Poeppel 2012; plan: peak detector);
4. category vigilance;
5. line segmentation by silence;
6. buffer rotation at the partner's line end (plan: learned input gate);
7. pointing → attention orienting;
8. comparator parameters (κ, thresholds);
9. MIX pair schema;
10. DG projection + recall WTA + familiarity bins;
11. NEXT/COMPLETION read-outs;
12. CTX layer;
13. sleep schedule;
14. action threshold + END max rule (plan: learned GO gate from value);
15. babbling enumeration + best-match binding;
16. corollary-discharge gate on own-voice learning.

---

## 13. Files

All in `arch/v2/develop/`.
- **Rules and tests:** `rules.py` (R1–R7), `test_rules.py`.
- **Cognitive core:** `core.py`.
- **Bench agent:** `agent.py`.
- **Symbolic world:** `devworld.py`, `run_dev.py`.
- **Module tests:** `test_modules.py` (M1, M4, M5, M6, M7).
- **Bench runner and ablations:** `run_bench.py`, `ablate.py`.
- **Bench diagnostics:** `drift_bench.py`, `feat_sep.py`, `audit_order.py`, `audit_bind.py`, `audit_rebind.py`, `audit_mem.py`, `babble_stats.py`, `nextchar.py`, `dbg_*.py`.
- **Outputs** (`out/`):
  - final runs: `bench_final_a.log` / `.json`, `bench_final_b.*`;
  - earlier runs: `bench_ABCD_r3.*`;
  - isolation probes: `bench_q_*`, `bench_probe_*`;
  - ablations and interference: `ablate_final_a.txt`, `interf_*.txt`;
  - module tests: `m1_final.txt`, `modules_*.out`;
  - next-char: `nextchar_5k*.out`;
  - symbolic: `dev_*.json`.
- **Server job folders:** `arch/v2/gpu/queue/develop_*`.
  - `bars1` (baselines; the v1 reference timed out);
  - `main1` and `modules1` (timed out);
  - `final1` and `longB1` (still running when I finished; they will time out).

**Bench issues for the coordinator.**
1. `baselines()` caches into the bench folder (`bench/cache`), so other agents write into the bench's folder. I used a copy.
2. The world seed couples voice, eye codes and content. A "same items, fresh schedule" evaluation is only possible by changing `probe_frac` / `held_frac`. Suggest a separate content seed.
3. The flat eye's random codes contain hub units (on in most scenes); worth documenting. The new `eye_layout='retino'` option helps.
4. "Check" names a non-pointed object half the time. That is a fair test, but it contaminates word→visual learning; document it.
5. My server bar job timed out on RefV1 (force, 2,000 dialogues). The README's bar numbers are the ones I compared against.

---

## summary_zh

1. 按"发育、记忆优先"的思路设计并实现了完整架构。身体（轮流说话、按拍发音、停止）原样保留。
2. 先听、后咿呀、再模仿、再靠听和改说（recast）学回答，全程不推嘴，没有老师提示。
3. 听觉阶段：不靠老师自动形成音节类别（65 类，每类一个字）；听到的句子按顺序完整记住（400/400）。
4. 咿呀阶段：把 35,960 种肌肉组合各试一次，每个类别都找到正确的发音程序（0 错），以后新听到的字可以立即模仿。
5. B 阶段命名（看场景回答）：0.54，最好的一次 0.71。旧架构 v1 是 0.12，先验和 n-gram 基线约 0.13–0.14。
6. C 阶段句型带空位："你要X吗→要X" 0.86，没听过的 X 也有 0.44（n-gram 是 0）。
7. D 阶段"这是X吗→是/不是" 0.77，基线约 0.48。
8. 消融证明：皮层读出、注意、场景焦点、视觉比较器、复制门控都确实影响行为。
9. 关于"多层前额叶"：用"有序句子缓存＋组合单元＋基底节门控＋慢上下文"按发育顺序搭建。
10. 关于"靠环路保持想法"：证明了只加泄漏消除不了同步环路的 2 周期振荡，需要自兴奋大于交叉驱动或自联想学习。
11. 关于"前额叶＝预测下一个输入"：局部误差规则加"成熟"后，下一字预测准确率 0.228/0.244，接近精确 n-gram 的 0.234/0.249；固定随机递归状态较差（0.199）。
12. 失败一：学新阶段会覆盖旧本领，B 的命名在学完 C、D 后从 0.51 掉到 0.22–0.29。
13. 失败二：没人教时让学习一直开着，回答会被改坏（0.56→0）。
14. 失败三：现在的"睡眠回放"没起到巩固作用，反而有害。
15. 另外：一次性学新物体名字在测试台上 0，颜色/数量题接近随机，有两个学习部件目前没有作用（VR、慢上下文）。
16. 下一步最关键：交错复习旧阶段、按预测误差优先回放、单元"成熟"保护旧知识；用"对方是否及时回应"作为第三因子来决定学不学；把代码移植到 GPU。
