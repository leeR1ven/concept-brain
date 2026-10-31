# ARCH_v2-P — "one general predictive learner" (design part; results are in report.md)

Labels: PROVED (argument with numbers), SIMULATED (numpy run; file + numbers), CITED, ASSUMED.

## 1. Principle
- One unit type: binary output, a leaky potential v(t) = lam_v v(t-1) + I(t), k-WTA (or a threshold).
- One learning rule (PR, `pr.py`) for every learned synapse:
  `dw_ij = eta_D * theta * M_D(t) * pi_P * x_i(t-1) * (y_j(t) - p_j(t)) / n_j(t-1)`
  - x_i: presynaptic output trace x_i(t) = max(s_i(t), lam_x x_i(t-1)), population-normalised (each source area
    brings unit power: divisive normalisation by the area's own inhibition).
  - y_j: the target = what j actually did, set by its DRIVER (sensory input, efference copy, own hold):
    event groups use the event ("char c heard now"; learning only at event ticks), state groups use s_j(t)
    (held co-activity is a target -> assemblies).
  - p_j = clip(d_j / theta, 0, 1), d_j = j's learned (predictive) input -> the dendritic prediction (Urbanczik & Senn
    2014, CITED).  Predicted and present: no change; predicted but absent: weaken, below zero = learned inhibition
    (w = w+ - w-, the negative part through j's local interneuron: Dale's law kept); unpredicted: strengthen.
  - n_j = sum over all synapses onto j of pi_P x_i^2: normalisation at learning time (heterosynaptic).  One error is
    shared by all paths onto j; pi_P is a per-path plasticity factor (fast hippocampal path pi = 8, cortex 1).
  - M_D: broadcast third factor (mode gate / novelty / reward prediction error).
- Every area learns to predict its own driven activity: Z predicts its own firing (and, with the efference copy, which
  own program made the sound); ZP predicts the next heard char; Go predicts whether someone takes the floor; the
  hippocampal readouts predict the content of the episode; assemblies predict their own continued activity.
- Speaking = saying the prediction: at a beat boundary, each program's score is ZP's prediction for its sound unit;
  END is ZP's prediction that the line ends.  No separate "speech area".
- Areas differ only in wiring, sizes, timescales and plasticity factors.

## 2. Areas (prototype size / GPU size)
| area | size P / G | activity | fan-in | persistence | learns from (rule, target, M) |
|---|---|---|---|---|---|
| EAR0 / FEAT0 (fixed) | 1,153 | top 40 % of layer 0 | 4 | 1 tick + reserve | nothing (innate calibration at birth) |
| perceptual clock (innate) | 1 phase register | — | E (window energy) | reset at speech onset after silence | nothing: one sample per 3 ticks (theta-like syllable phase, Giraud & Poeppel 2012, CITED) |
| Z syllable units (STG-like word-form units) | recruited, <= 256 / <= 4,096 | WTA, p >= 0.7 | all FEAT0 | event (1 per char) | PR, state target = the winner (competitive) on others' speech; target = efference copy on own speech (inverse model); a fresh unit when no unit reaches 0.7 (the designer's "generate") |
| babble memory (forward model, lookup) | 1 entry per distinct own sound (3,618 after babbling) | — | — | permanent | one-shot store of gesture -> own sound; a newly heard unit is linked to the gesture whose sound it recognises |
| sequence buffer x1, x2 | 2 x zmax | localist | Z | the current line | fixed shift register (phonological buffer) |
| Gline (PFC line layer) | 2,048 | union of 8-unit codes (~4 %) | fixed conjunction detectors on Z (unigram + bigram) | the current line (latch) | fixed |
| Ghold (PFC working memory, BG-gated) | 2,048 | ~4 % | Gline (copy at load) | held from another speaker's line end to the next one; holds through own replies, overheard replies and recasts | prototype: gated copy; module-tested form: leaky + learned recurrence (PR state target) + reset at load (M4) |
| Gtopic (dialogue layer) | 2,048 | ~4 % | Ghold | leak 0.5 per load (several lines) | fixed |
| B prompt buffer | zmax | localist bag | Ghold chars | as Ghold | diagonal copy path into ZP (PR) |
| V scene | 1,024 eye code | ~7 % | eye | as the scene | — |
| V2 visual conjunctions | 65,536 | ~100 active (0.15 %) | 2 eye units, AND | as the scene | fixed wiring; read by ZP (PR) |
| C2 PFC x scene conjunctions | 65,536 | ~60 active | 1 Ghold unit AND 1 eye unit | as Ghold/scene | fixed wiring; read by ZP (PR) |
| H hippocampal DG (next-char path) | 4,096 | k 32 (0.8 %) | 12 random of (Ghold, Gline, x1 code, eye, start), thr 3 | per event | fast path into ZP, pi_H = 8 (one error shared with cortex) |
| R hippocampal recall index | 4,096 | k 40 (1 %) | 12 random of (Ghold, eye), thr 3 | per line end | one-shot PR readouts: RS (key = the line itself -> its chars), RN (key = the context before -> the next line's chars); recall -> Brec (reinstated chars) -> diagonal copy path into ZP |
| ZP the predictor | 1 + zmax | p in [0,1] per target | all areas above | per event | PR, target = the heard char / END of another speaker, M = 1; own events M = 0 (corollary discharge) or M = RPE |
| Go (BG Go/NoGo) | 1 | p | Ghold, B, V | per line end | PR, target = another speaker started within 4 ticks of the line end (decided before my earliest possible onset at +6, so never self-fulfilling) |
| critic (BG value; designed, unit-tested T8, not used on the bench) | 1 | p | as Go | per own turn | PR, target = reward; delta = r - V is M for own-voice targets |
| event store + sleep | <= 6,000 events | — | — | until displaced | novelty-gated store (1 - p >= 0.3) of (cortical context, target); sleep replays them interleaved into the slow paths with H silent, then downscales H (one global decay) |
| body (innate, ARCH_v1) | F, GO, CD, beat, WTA, D/I/M, END rule, counter 12, floor 0.3 | | | | nothing |

## 3. The PFC
- Multi-layer, rostro-caudal: Gline (line, fast) -> Ghold (the held prompt) -> Gtopic (dialogue, slow); conjunctive
  read-outs C2 (prompt x scene) and the reinstated hippocampal content Brec feed the same predictor at every boundary.
- Learned inputs: Z (learned by listening/babbling), Brec (hippocampal recall, learned one-shot).
- Gating: BG-like, innate in v2.0: load = reset + input at the end of another speaker's line; hold otherwise (own
  reply, overheard reply, recast).  Learned gate later (PBWM-like, from the RPE).
- Persistence per layer (leaky units) + learned recurrence (assemblies): module test M4 holds 60 ticks against
  distractors with no period-2 in 100 % of holds; without leak 36 % of holds alternate (period 2).
- Concept units without returns: assemblies in Ghold/Gtopic that drive nothing but ZP (allowed, read by the predictor).

## 4. Hippocampus
- Index theory (Teyler & DiScenna 1986, CITED): DG-like sparse expansion (1 %) of the cortical context; one-shot
  readouts (eta 1).  Two keys: the line itself (pattern completion of a line from a partial cue: 我叫什么 -> 我叫小明;
  这是什么 -> 这是X) and the context before (what came next: 刚才左边是什么 -> the earlier reply).
- Recency among competing episodes falls out of one-shot error correction (M2: newest of 5 competitors recalled 1.00).
- CA3 recurrent completion was tested (M2) and is NOT adopted: at 1,000 episodes it lowered half-cue recall
  0.68 -> 0.07 (attractors turn partial overlaps into jumps to wrong episodes).
- Sleep: replay of the novelty-gated event store into the slow cortical paths (H silent), then downscaling of the
  fast path.  Waking idle replay does not exist (no uncontrolled replay in silence).

## 5. Basal ganglia / value
- Innate: the body's WTA over programs per beat, GO from the turn units, END rule (END wins iff its prediction is the
  largest), floor.
- Learned: END (as a predicted event), Go/NoGo (who takes the floor), and (designed) a critic whose RPE is the third
  factor for own-voice targets (T8: own targets with M = 1 drift to reward 0.12; with M = r - V reach 1.00).

## 6. Motor learning, imitation, the body
- Babbling: an innate spontaneous beat plays bouts of 4-8 random 4-muscle gestures; habituation (no gesture repeated
  until all 35,960 are tried).  A gesture becomes the program of a heard syllable unit when its own voice drives that
  unit with p >= 0.75 and a margin >= 0.2 over every other unit; else its sound goes to the babble memory.
- No Z learning from babbles (self-supervision on the winner made a hub unit win 98 % of babbles: SIMULATED).
- Imitation = saying a heard char through its linked program; vocabulary growth after babbling = babble-memory lookup
  for newly heard units.
- Programs, D/I/M, beat, END rule and counter are ARCH_v1's body; the mind only supplies scores.

## 7. Teaching without muscle puppeting
- Stage A: listening (cds statements, overheard routines) builds Z and ZP; private babbling (no tutor) builds programs.
- Stages B-D: learning by listening to every speaker (model replies by other2 or the partner); recasts are heard lines
  predicted from the held prompt (the prompt stays held through the agent's own reply and the recast).
- No cue, no END clamp, no muscle drive anywhere.  Teacher forcing exists only as the v1 reference.

## 8. Developmental stages and pass thresholds (each must pass on the CPU bench before the next; all teacher-free)
| stage | what is trained | pass threshold (measured on the bench / diagnostics) |
|---|---|---|
| A0 birth | FEAT0 calibration (60 random sounds) | — |
| A hear | Z units recruited from cds + overheard speech; ZP and Go start learning by listening | Z purity >= 0.98 against the true chars; no split chars; >= 90 % of the heard lexicon has a unit |
| A babble | programs from babbling (full gesture set, habituation) | >= 95 % of heard chars have a program; 100 % of programs say their unit's char |
| B naming with a scene | ZP by listening to model replies (other2 / partner) | seen >= 0.5 and above the best baseline on seen; held above the best baseline on held |
| C frames | as B; copy gates | seen >= 0.7; held >= 0.4 (n-gram baseline 0.29-0.31) |
| D memory | hippocampal recall (RS/RN) + copy gate; surprise code; Go by open floor | name >= 0.6, oneshot >= 0.5 (1 exposure), check >= 0.7, before >= 0.5 |
| E adult corpus interleaved | everything keeps learning; sleep replay between blocks | never-heard corpus pairs echo-free above echo; B-D retention >= 90 % of their end-of-stage scores |

## 9. Cost at GPU scale (vocabulary 4,096 programs; ASSUMED sizes, PROVED arithmetic)
- ZP inputs: Gline/Ghold/Gtopic 3 x 8,192, V 1,024, V2 65,536, C2 65,536, H 16,384, x1/x2/B/Brec 4 x 4,096, S 6
  ~ 189k inputs x 4,097 targets = 7.7e8 weights = 1.5 GB in fp16.  Only active rows are read or written: ~600 active
  inputs per event -> 2.5e6 multiply-adds per prediction and per update; events every 3 ticks.
- Z: 1,153 x 4,096 = 4.7e6 weights (dense, every tick: 4.7e6 MACs).  R/RS/RN: 16,384 x 4,096 x 2 = 1.3e8 (fp16 0.27 GB).
- Per tick ~1e7 MACs => a V100 runs ~1e4 agents in parallel at real time, or one agent >1e3x faster than real time.
  The CPU prototype (zmax 256, NV2 = NC2 = 65,536) runs ~1,000-1,700 ticks/s on one core (2,000 dialogues ~3-4 min).
- Memory per agent at GPU scale ~2 GB (fp16) -> 10-12 agents (seeds/ablations) per 32 GB V100.

## 10. The designer's framework: kept, changed, removed
| framework item | ARCH_v2-P |
|---|---|
| binary neurons; fixed active fraction (#26) | kept (k-WTA); per-population normalisation keeps "how many" fixed, inputs choose "which" |
| born full of random connections (#23) | kept: the fixed random conjunction wiring (Gline codes, V2, C2, H, R) is born; learning changes weights only |
| earlier -> later strengthens, later -> earlier weakens (#9, 2026-10-30) | generalised: "predicted -> no change, unpredicted -> strengthen, predicted but absent -> weaken"; order is kept (a source after its target gets no credit, T5/T7); the state-based back clause is replaced (held sources learn the onset, T5) |
| no mutual strengthening of co-active units (#9) | relaxed in assembly populations only (state target), bounded by the prediction itself (M1: max |w| <= 0.1, no budget) |
| one global decay as the only clean-up (#3) | kept: one decay of the fast hippocampal paths per sleep; budgets are not needed (PR is self-bounding) |
| hippocampus = fast short-term chains (#2) | changed (evidence): one-shot, long-lasting episodic index with recall into the PFC and replay into cortex |
| PFC = multi-layer recurrent; deep output returns as input (#later) | kept: Gline -> Ghold -> Gtopic, recurrence with leak (M4), concept units with no returns allowed |
| motor: inhibition vs disinhibition decides action (#4); speak through muscles only | kept (the body) |
| no "speech area" (#22) | kept: speaking = the predictor's output through the auditory-motor mirror |
| split always-on / generate new units (#6) | used for perception: a fresh syllable unit when nothing explains a heard syllable |
| return strength knob (#17) | replaced by learned predictions (no fixed returns) |
| "do not add mechanisms" (#10) | every fixed mechanism is listed in §11 with a plan to learn it |

## 11. Fixed (innate) mechanisms, counted (v1 had ~30, most task-timed)
1. ear layer 0 / FEAT0 (sensory periphery; plan: learned features by the same rule on raw input)
2. perceptual clock (syllable phase reset at speech onset; plan: predict onsets from the energy envelope)
3. body (F, GO, CD, beat, WTA, D/I/M, END rule, counter, floor = ARCH_v1's scaffold, one block)
4. Z recruitment threshold (0.7) and competition threshold (0.6)
5. babbling CPG + habituation
6. babble link criterion (0.75, margin 0.2) and babble memory store/lookup
7. sequence buffer x1/x2 and the line latch Gline (unigram + bigram detectors)
8. PFC load gate (another speaker's line end -> reset + load); Gtopic leak
9. fixed random wiring of V2, C2, H, R (born connectivity)
10. gated one-to-one mirror buffer_c -> ZP_c (kappa) (the auditory-motor mirror, like Z_c == program c)
11. novelty gate of the event store; sleep schedule; downscaling factor
12. surprise thermometer code of the held line
13. third-factor routing: own voice -> M = 0 (corollary discharge)
14. Go timing windows (4 ticks / 30 ticks)
None refers to a task or a stage; none is timed to a particular junction (v1 needed P = T∧GO, E = L∧R2, Hc delay lines,
the Z clamp at B+4, latch rules, the used flag, the summed-drive END rule, the teacher gate).

## 12. Risks
- Scene binding is slow: the eye code has no location structure, so (pointing x object-at-slot) must be learned per
  (slot, object) conjunction; ~4 informative of ~100 active V2 pairs get ~2 % of each correction.
- Held-out frames need each part of the context to predict alone (stochastic release is the candidate fix).
- One-shot hippocampal readouts overwrite (palimpsest): retention of rarely repeated episodes falls with time.
- The Go unit rests on timing statistics of the tutor; a partner with other timing habits needs relearning.
- Speaking = saying the prediction means the agent says what a typical speaker would say next; a learned role/speaker
  model (who answers whom) is not built yet.
- Noise: F and the perceptual clock assume silence = 0 energy (inherited from v1); noise needs a learned onset detector.

## 13. Final state after iteration (supersedes §2-§5 where they differ)
- Hippocampal recall = recruited episode cells (one per completed line of another speaker; key = sparse codes of the held
  line, the eye, their pair conjunctions C2 and a DG code; winner-take-all by mean cosine, newest wins ties); a recalled
  cell reinstates its line's chars (Brec) and the next line's chars (Bnext).  The linear one-shot readouts RS/RN were
  replaced: they superimposed every stored line (true content 0.30 vs 1.71 for the question's own words).
- Copy gates: one PR unit per buffer (prompt B, recalled line Brec, recalled next line Bnext) predicts "the next heard
  char is in this buffer"; its output multiplies the fixed one-to-one buffer_c -> ZP_c mirror, so copying generalises
  to chars never copied before (novel words).
- Go/NoGo target: 1 if another speaker starts within 4 ticks of a line end or if others leave the floor for 30 ticks,
  0 if another speaker starts in between (never self-fulfilling: my earliest onset is at +6).
- Fast hippocampal paths into ZP: H (with the held prompt) and H2 (re-enactment, without it), pi = 8.  The audit shows
  H carries the D name task but hurts C; H2 hurts B.  Candidate for removal (pruned run in report).
- Not adopted after tests: CA3 recurrent completion (M2), reverberating PFC hold (M6/M6b: generic attractor without
  homeostasis, no lasting hold with it).  The PFC hold is a BG-gated latch.
- Speaking floor: 0.3 in training runs; a readout at 0.12 on the same knowledge raises every stage (B .135->.154,
  C .164->.255, D .092->.122): the floor should be lower (or learned from the critic's value of answering).
