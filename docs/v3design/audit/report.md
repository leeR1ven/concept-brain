# Audit: a bottom-up inventory of every mechanism the agent has today (ARCH_v2 → v2.1 → hearing → meaning → v2.2 in progress)

**Scratch:** `docs/v3design/audit/knobs.py`. It is a read-only script that counts the knobs (one process, under 100 MB). No runs. Nothing in the repo or in other agents' folders was touched.

**Labels:**
- **PROVED**: from the code or arithmetic.
- **SIMULATED**: another agent's run; I cite its file.
- **CITED**: a report or paper.
- **ASSUMED**: my design judgement. Every verdict in the table is ASSUMED unless it cites evidence.

## 0. Bottom line

1. **The inventory has 83 mechanisms** (PROVED from the code, rows in §2).
   - 62 come from ARCH_v2 (the develop, tune, think and synth layers plus the ARCH_v1 body).
   - 8 come from v2.1, 8 from hearing, 2 from meaning, and 3 from v2.2 so far.
2. **They expose 265 named configuration knobs** (PROVED, `knobs.py`):

   | layer | knobs |
   |---|---|
   | core Cfg | 85 |
   | ThinkAgent | 24 |
   | SCfg | 17 |
   | Synth agent | 6 |
   | TCfg | 5 |
   | DevAgent | 7 |
   | ARCH_v1 body | 16 |
   | RCfg (v2.1) | 26 |
   | AgentV21 | 5 |
   | hearing | 16 + 8 |
   | meaning | 20 |
   | randmix | 20 |
   | v2.2 so far | 10 |

   - There are also 25 hard-coded uses of the bench's "3": 3 slots, 3-tick syllables, `range(3)`, `% 3`, `np.eye(3)` (PROVED by grep).
   - Each ability step added 10–31 knobs.
3. **Verdicts:**

   | verdict | count | share |
   |---|---|---|
   | (a) already generic | 15 | 18 % |
   | (b) can come from a generic mechanism plus experience | 39 | 47 % |
   | (c) legitimate innate prior | 14 | 17 % |
   | (d) scaffolding to delete | 15 | 18 % |

   - The 14 (c) rows collapse into 5 budgeted priors.
   - **Of the 21 mechanisms added after ARCH_v2, only 3 are generic:** the eligibility trace, the random layer, and scene_held. One is an innate prior (the social detectors). 10 are (b) and 7 are (d).
4. **Hidden hand-coding.** 54 of the 83 rows carry at least one tag:

   | tag | rows | what it means |
   |---|---|---|
   | T | 20 | timing rule fitted to the world's script |
   | S | 18 | special-case code path or mode switch |
   | F | 16 | feature shaped like a task or a world law |
   | Tr | 10 | event trigger |
   | R | 5 | route wired for one task |
   | P | 1 | privileged world access |

   The clearest cases:
   - **Reflex 7 (`init_min_tau` 10)** was fitted, "PROVED from world.py", to the bench's 5-tick scene lead.
   - **`ral_W_other` 6** matches the bench's 3-tick model-reply gap.
   - **v2.2's `veto_W` 40** is the gap at which the world's own ground-truth log captures 100 % of corrections (`v22/an/cue_test.py` → `out/cue_test_base.txt`).
   - **Staged and life runs use different innate constants** (`THINK_FINAL` vs `THINK_LIFE`: g0_scene 0.3 vs 1.0, u_scene 0 vs 0.02, plus 4 life-only switches).
   - **Babbling reads `world.voice` offline.** Hearing's denoised binding extends the same shortcut (`AgentH._own_window`).
5. **The minimal generic set this inventory implies, bottom up, is 6 mechanisms plus 5 innate priors** (§4).
   - The 6 mechanisms:
     - C-1: area dynamics (random sparse fan-in, k-WTA, adaptation, persistence);
     - C-2: unsupervised fan-in learning (instar, use-it-or-lose-it);
     - C-3: the predictive delta rule on every projection, with an eligibility trace and one gain;
     - C-4: a fast episodic region with spontaneous replay;
     - C-5: a basal-ganglia selection loop over all areas' outputs;
     - C-6: two broadcasts, RPE and an ACh-like encoding/replay mode.
   - The 5 priors:
     - P-1: the body / brainstem;
     - P-2: the wiring plan;
     - P-3: orienting;
     - P-4: drives and values;
     - P-5: spontaneous motor activity.
   - Every (b) row maps onto these 6; none needed a seventh (ASSUMED; the ladder in §6 tests it).
6. **The 5 riskiest replacements** (§5):
   - R1. role-aware listening's cancel/commit rules;
   - R2. the clocked, vigilance-based ear;
   - R3. the four task-wired copy routes and the fixed order of thought steps;
   - R4. the turn-taking reflexes fitted to the bench's timing;
   - R5. the offline babbling inverse map.
7. **v2.2 so far shows the rule-pile dynamic directly** (SIMULATED by the v2.2 agent; CITED from its `out/`).
   - Each veto knob fixes the side effect of the previous one: veto → absorbing silence in untutored use → `veto_gate` → `veto_fast`.
   - Life free replies rise to 0.47–0.52 but stay below the echo-or-constant reference (0.61–0.64).
   - `fin` drift loss at 900 untutored dialogues is 0.45 (base −0.02).
   - `finF` loses 0.75 at 3,000 untutored dialogues.

## 1. Method

- **Generic** follows the brief's definition: the definition mentions no task, modality, word or content.
- **Hidden-coding tags:**
  - **T**: a timing constant fitted to the world's script;
  - **Tr**: an event trigger (code that fires on "line end", "quiet moment" and so on, where a continuous dynamic could do the job);
  - **R**: a route wired for one task;
  - **F**: a feature or bin shaped like a task, a metric or a world law;
  - **S**: a special-case code path or mode switch;
  - **P**: privileged world access.
- **Sources:**
  - "Added for" cites the report that justified the mechanism (CITED).
  - The parameters are read from the code (PROVED).
- **Files read:**
  - synth: `core.py`, `rules.py`, `agent.py`, `core_think.py`, `agent_think.py`, `body_think.py`, `core_synth.py`, `agent_synth.py`, `run_j.py`, `life_synth.py`, `bench_copy/body.py`;
  - v21: `core_v21.py`, `agent_v21.py`;
  - hearing: `hear.py`, `agent_h.py`;
  - meaning: `meanlayer.py`, `mcore.py`;
  - randmix: `randlayer.py`;
  - v22 (read only): `core_v22.py`, `agent_v22.py`, `jobs/`, `out/agg_steps.txt`, `out/paired_base_cf.txt`, `out/life/*.json` via `life_cmp.py`, `out/cue_test_base.txt`.

## 2. The full inventory

The replacement mechanisms C-1 … C-6 and the priors P-1 … P-5 are defined in §4.

### A. Body and brainstem
Files: bench `body.py` (Body), `body_think.ThinkBody`, `ThinkAgent._own_voice`.

| ID | mechanism (code) | added for (measured failure) | params / hand-set | tags | verdict → replacement / justification |
|---|---|---|---|---|---|
| A1 | turn unit F/F2, "other is talking": F = [0.01·E(t−1) − 10·ΣCD ≥ 0.1] inhibits GO | v1 talk-over (ARCH_v2 §5 #1) | f_w 0.01, f_thr 0.1: an absolute E ≥ 10 | F: absolute level; σ 1 noise has E ≈ 19, so F is on 0.99 of the day, and 0.16–0.52 in crowds (hearing §3, §7) | **(c)** innate vocal brake (marmoset turn-taking, Takahashi et al. 2013; PAG gating, Jürgens 2002). Its threshold must be relative (C-1 adaptation); a near-only brake comes from C-3 (P_oth) |
| A2 | GO → CD (4 ticks) → R1 → R2; first boundary = R1 ∧ ¬F | v1 0-error mechanics | cd_len 4 | – | **(c)** motor pipeline with corollary discharge (Crapse & Sommer 2008) |
| A3 | 3-tick beat, program WTA, D/I/M ramp 1-2-3, use penalty | v1 mechanics | 3 ticks, ramp | – (the body is co-designed with the world) | **(c)** syllabic CPG (MacNeilage 1998) |
| A4 | breath counter 12 syllables; max_len 8 | monologues | 12, 8 | – | **(c)** respiratory limit |
| A5 | END-vs-best "max" rule; floor 0.3 (silence if the best score is below it) | garble, extra turns | floor 0.3 | – | **(b)** END is already a predicted item. The floor is the bound of the C-5 selection loop, learned from the value of speaking vs silence (experience: outcomes of wrong vs withheld answers) |
| A6 | `brake_now`: the first syllable also needs 0.01·E(t) < 0.1 | think run 1 overlaps (a 2-tick blind window), §5 #3 | same constants | T (a patch to a pipeline) | **(c)** merge into A1: the brake reads the current input too (Schegloff 2000) |
| A7 | GO only from REL (BG release), not the v1 timer | principle 4 | – | – | **(a)** C-5 release |
| A8 | own voice = mouth active ≤ 2 ticks ago (efference copy; `own_tail`) | a refused GO ran CD and deafened the agent: B 0.56 → 0.45 (think) | 2 | T (the world's 1-tick echo + 1) | **(c)** efference copy (Eliades & Wang 2003); the window is the body's loop delay |
| A9 | line onset = E 0 → > 0; line end = the first silent tick | – (develop) | E == 0 | T, Tr: bench lines have no internal pauses | **(b)** C-3 learns an end-of-line predictor and C-1 adapts to the background, so boundaries come from prediction error (Zacks et al. 2007). Experience: pauses inside lines |

### B. Ear and perception
Files: `agent.DevAgent`, `rules.Templates`, `hearing/hear.py`, `hearing/agent_h.AgentH`, `agent_v21.VoiceCats`.

| ID | mechanism | added for | params | tags | verdict |
|---|---|---|---|---|---|
| B1 | EAR-0: fixed 3-D local random layer (1,920 units, fan-in 4, top 40 %); FEAT0 = units with birth usage ≤ 0.6 | the bench front end | 4, 40 %, 0.6 | – | **(a)** C-1 without training (the designer's local 3-D random net). Learned upper layers were rejected here (ARCH_v2 §7.11) |
| B2 | syllable clock: phase reset at onset, one sample every 3 ticks from onset + 2 (`clk % 3 == 2`) | develop design | 3, +2 | T (exactly 3-tick chars), Tr | **(b)** C-1 categorises every tick; C-3 predicts the next onset (entrainment: Lakatos et al. 2008; Giraud & Poeppel 2012). Experience: speech rhythm. B7 already shows that per-tick matching works |
| B3 | Z categories: match ≥ 0.75 → learn (rate 0.1); else a new category if > 40 bits; cap 256 | syllables without a teacher: 67/67, purity 1.00 | 0.75, 0.1, 40, 256 | F: the cap fills within one world2 day (ARCH_v2 §8.4, hearing §7.4) | **(b)** C-1 fixed-size k-WTA + C-2 instar + use-it-or-lose-it, so junk recycles. Experience: heard speech |
| B4 | babbling: one offline pass over all 35,960 four-muscle sets through `world.voice`; every own sound stored once; binding ≥ 0.85 | no puppeting | 35,960, 0.85 | **P**: privileged; the bench rules forbid reading `world.voice` | **(d)** replace by P-5 online babbling + a C-3 auditory → motor inverse predictor trained on the efference copy (DIVA: Guenther 2006). Experience: own babbling + caregiver imitation. Risk R5 |
| B5 | rebind: an unbound category is re-matched to the babbling memory at every hearing | an unbound 气; ritual 0.66 → 0.90 (§7 #2) | 0.85 | S (a patch to B4) | **(d)** vanishes with an online inverse model |
| B6 | SoundGate: μ and ν of the 2 newest slices; τ 500; trimmed μ + 4ν; on μ + 7ν, off μ + 4ν; stuck rule at 600; birth calibration | world2 σ 1: 0 lines; bench σ 1: J 0 (hearing §0) | 7, 4, 500, 4, 600 | T (600: "no line is that long"), F (2 slices = one world tick) | **(b)** C-1 area-general adaptation, subtractive with robust running statistics (Dean, Harper & McAlpine 2005). Experience: background statistics. Risk R2 |
| B7 | relock: the clock moves to a phase whose window matched ≥ 0.85 | crowds: overlapped coverage 0.06 → 0.85 | 0.85, 3 phases | T | **(d)** falls out of per-tick categorisation (B2) |
| B8 | voice boundary: a relock ends the line | a second speaker | – | F (world2 speakers share one speech rate) | **(d)** speaker boundary from the B14 area + C-3 surprise |
| B9 | no new categories in mixed lines | junk categories | – | S | **(d)** a fixed-size C-1 layer never creates units |
| B10 | conf: new categories only from windows with 3 clear pairs | junk 12 → 0 at σ 1 | – | S | **(d)** C-2's learning rate is scaled by the adapted SNR (C-1) |
| B11 | level matching: retry 1–3 levels louder, accept ≥ 0.85 | duplicates 34 → 5 | 1–3, 0.85 | F (world2's distance law subtracts integer levels) | **(b)** C-1 divisive gain control (Sadagopan & Wang 2008; Carandini & Heeger 2012) |
| B12 | denoise: cells ≤ ⌈5μ/16⌉ are zeroed; the babble memory is denoised the same way | noisy crowds matched only 0.78 | 5, /16 | – | **(b)** C-1 per-channel subtractive adaptation |
| B13 | `h_back` onset back-tracking (rejected, off) | – | 5.5 | – | **(d)** dead code |
| B14 | voice categories: speaker residual, sign-quantised, vigilance 0.7, cap 8, rate 0.05, n_min 10 | a hook for trust; unused (purity 0.91 / 0.72) | 0.7, 8 | passenger | **(b)** a C-1 area on the speaker residual, fed to the PFC as an input (needed for R1). Experience: many voices |

### C. Vision and attention
Files: `core.Core`, `core_synth.SynthCore`, `ThinkAgent._parse_eye`, `SynthAgent._parse_eye`, `core_think.think_step`.

| ID | mechanism | added for | params | tags | verdict |
|---|---|---|---|---|---|
| C1 | retinotopic eye: 3 blocks × 240, the same code in each (bench `eye_layout='retino'`) | colour 0.11 → 0.82 (§7 #5) | seg 240, 3 slots | F: the world's factorised, slot-identical code carries most of the gain (review 2 §3) | **(c)** retinotopy (maps form before eye opening: Katz & Shatz 1996) |
| C2 | slot-relative frame: one prime, recognition and focus map shared by all slots (`rel=True`) | same | – | F (weight sharing over exactly 3 slots) | **(b)** foveation + C-5 saccades put the attended thing in the fovea, so one map serves every position (the designer's foveated-vision principle; world2 `eye_mode='fovea'`). Experience: looking |
| C3 | joint-attention orienting: pointing = ≥ 4 extra bits; the attended slot is the block with most extra bits | develop; a labelling bug (B 0.30 → 0.74, §7.5) | 4 | – | **(c)** innate point and gaze following (Butterworth & Jarrett 1991; Scaife & Bruner 1975) |
| C4 | eye parser: baseline at scene onset; > 2 missing bits = scene change; Jaccard ≥ 0.6 to ≤ 3 learned pointing codes | develop | 2, 0.6, 3 | F (bench eye format, 3 codes), Tr | **(b)** C-1 adaptation: the baseline is the adapted state, a change is a transient |
| C5 | `life_eye`: > 2 new bits appearing in silence join the baseline (life only) | the partner and new objects read as pointing (§5 #23) | 2 | S (life-only path) | **(b)** the same C-1 adaptation in every world |
| C6 | PRIME outstar: word → expected eye units, taught only under external pointing (`prime_internal` 0) | grounding | m_min 0.02 | S (pointing-only gate) | **(a)** a C-2 / C-3 conditional expectation; the gate becomes "attention gates learning" |
| C7 | attD outstar: word → slot, taught by pointing or the comparator; `self_ground` (a line's own located word teaches) | where | attd_n 2 | S (the self_ground path) | **(b)** a C-3 predictor of the attended location from the held words. Experience: pointing + located statements |
| C8 | comparator MATCH and internal attention: match_thr 0.6, match_n 6, conf_min 6, rate_max 0.4, match_min 0.5, gap 0.3, κ 2, att_thr 0.5 (repeated as literals in `think_step`) | where, colour, check | 8 constants; tune found "a plateau with cliffs on both sides" | – | **(b)** top-down prediction (C-3) against the input; the lead / threshold is the C-5 WTA of a priority map |
| C9 | VR: analysis-by-synthesis (rec_thr 0.85) + cache (`rec_fix`) | what (lesion: weak, 0.62 → 0.52) | 0.85 | – | **(b)** a C-3 recognition readout eye → word (the inverse of PRIME), learned from naming |
| C10 | eye_rate base-rate filter: units on > 40 % are excluded after 50 events | dense appearance codes | 0.4, 50 | – | **(b)** C-1 adaptation |
| C11 | scene focus F (the attended block as PFC input) + slot A | what / colour (lesion F: 0.65 → 0.19) | – | – | **(a)** attention as gain (Reynolds & Heeger 2009) |
| C12 | innate scan for scene thoughts: slot k mod 3, kept if confidence rises, k ≤ 5 | self-initiated naming | 3, 5 | F | **(b)** C-5 saccade selection by value |

### D. Prefrontal cortex
Files: `rules.OneShotChain`, `rules.Delta`, `core.Core`, `core_synth.SynthCore`, `randlayer.RandMix`, `core_v21.RoleCore`, `meanlayer.MeanCodes`, `mcore.MeanCore`.

| ID | mechanism | added for | params | tags | verdict |
|---|---|---|---|---|---|
| D1 | current-line buffer + OneShotChain successor with inhibition of return | the copy routes need order | – | a symbolic pointer structure | **(b)** C-1 persistence of a sequence (held items + order via C-2 links / time cells) + C-4 one-shot sequence links |
| D2 | continuous thought: a line enters at its end (gain 1); held content × 0.3; θ 0.5; ≤ 12 items | principle 2 (it is register-equivalent, PROVED §7.3) | 0.3, 0.5, 12 | Tr (entry at line end) | **(a)** C-1 persistence + k-WTA. The entry trigger should become continuous input competition (but surprise-gated entry failed, §7 #10) |
| D3 | own speech does not enter the thought (`ct_own` 0), while v2.1 keeps a separate `T_own` slot | keep the listening target clean | 0 | S: two representations of own speech | **(d)** own speech enters as an efference input, like any line (needed for R1) |
| D4 | wandering: co-holding links (η 0.02, rows renormalised); after 200 silent ticks, every 50: fatigue 0.1 / τ 20, spread 0.5, renormalise | principle 5: favourites 0.24 → 0.82; fair baseline 0.62 and mostly function syllables (review 2) | 200, 50, 0.1, 20, 0.5 | T, Tr | **(b)** C-1 fatigue + homeostasis and C-2 links running every tick; wandering is what persistence does under weak input. Delete the trigger |
| D5 | PFC-2 = RandMix c_flat: random at birth, dendritic AND, k-WTA 1 %, instar, prune at 20,000 + regrow | MIX's growth: J 0.83 → 0.95 (default), 1.00 → 1.09 (frozen) (randmix) | 1×, fan 3, α 0.01, η 0.02, 20,000 | – | **(a)** C-1 / C-2 |
| D6 | pair schema: 14 direct groups and 31–40 pair types; population sizes from MIX's bench key counts (`alloc` mix / pow:0.5) | additive readouts cannot conjoin (PROVED in develop) | 40 pairs | F: the pair list is chosen per task; pow:0.5 builds the bench's statistics into the birth wiring | **(b)** C-1 random fan-in from every projecting group (the wiring plan names areas, not pairs) |
| D7 | hand features: N = min(n, 6); FAM bins (overlap ≥ 0.45, ≥ 0.7); M bins (match ≥ 0.5, ≥ 0.8); P1 / PL; L; NOP | name recall (FAM), check (M) | thresholds | F (bins shaped like tasks) | **(b)** C-1 population codes of the continuous values (overlap, match, position) |
| D8 | GEN readout, rule R2 (NLMS, η 0.35; END; signed; 0.2 in replay) | the core | 0.35 | – | **(a)** C-3 |
| D9 | BG copy gates COPY_prompt / COPY_next / COPY_comp over OneShotChain candidates | ritual (lesion 0.91 → 0.14), recalled answers (name) | η 0.5, gain 1 | R | **(b)** one C-5 output-gate type over "the next item of every source area" (PBWM: O'Reilly & Frank 2006). Experience: listening, which teaches which source predicts the heard item. Risk R3 |
| D10 | gateR, "say what I recognise / where it is" (2 outputs, frame-only keys, say_loc_thr 0.8, say_noprompt) | held-out what 0.61 → 0.81, where 0.81 → 0.90 (v21 §3) | 0.8 | R (what / where) | **(b)** the same as D9 |
| D11 | scene_held: a self-initiated thought reads the held thought | initiatives repeated the partner's lines | – | it removes a special case | **(a)** |
| D12 | gates read the context only (candidate identity is removed from the gate keys) | generalise to new words | – | – | **(c)** wiring prior of BG output gating: context in the striatum, content in the cortex (PBWM) |
| D13 | critic / actor, rule R6: δ = r − V(first-step context); actor η 0.3, critic 0.2, act_T 60 | reward shapes content (weak) | 0.2, 0.3, 60 | T (60) | **(a)** C-3 + C-6 RPE; act_T becomes the generic trace decay |
| D14 | meaning RI codes: D 512, kx 6, km 16, η = max(1/n, 0.002), base rate τ 20,000 + floor, hysteresis 1.2, ≤ 2 swaps per recompute, a recompute every 20 lines, placeholders, n_min 3 | class held-out 0.50 → 1.00 in the module test; the bench cannot credit it (meaning §0) | 11 constants | Tr; the drift patches | **(b)** C-2 on a context area (instar over the neighbours' lines) + C-1 base-rate adaptation (PMI). Delete the drift scaffolding. Experience: distributional statistics (Saffran et al. 1996) |
| D15 | meaning cue in the DG (mbag / mfirst groups; weight = hipcue·D / (km·V)) | wrong-class recall (0.70 → 1.00) | the formula | F (a hand-derived normaliser) | **(b)** DG fan-in from the meaning area (P-2) + C-1 divisive normalisation per input group |
| D16 | thought steps in a fixed order: recall → comparator → VR (think_k 3, ior 0) | think: honest timing; no content gain | the order | S, Tr | **(b)** C-5 selects the next internal operation, rewarded by the gain in confidence (open problem 2), or all run in parallel. Risk R3 |
| D17 | confidence e = margin of the best first item | the latency signal | – | – | **(a)** |
| D18 | listening learns only from other voices: the agent never hears itself (no detection while its own voice sounds) | own speech must not teach | – | S | **(b)** C-3 with efference prediction: one's own voice is predicted, so its error is about 0 and nothing is learned (predictive coding, Rao & Ballard 1999). Experience: hearing oneself |
| D19 | dead switches: CTX, naming target, residue, surprise entry, split, maturation, weight decay, sleep, role_units, hip_bigram, hip_novelty, contingency / own_rotate, dialogue_reset, gate_freq, rf_focus | rejected on evidence | about 40 knobs | – | **(d)** delete |

### E. Hippocampus
Files: `rules.Hippo`, `core.awake_recall` / `_replay`, `core_synth`, `core_v21`.

| ID | mechanism | added for | params | tags | verdict |
|---|---|---|---|---|---|
| E1 | DG expansion 6,000 at k 120 + 10,000 one-shot cells (the weakest is overwritten) | interference and drift: J 0.61 → 0.73 with recall + a big store (tune) | 6,000 / 120 / 10,000 | – | **(c)** the hippocampal index (Teyler & DiScenna 1986; DG pattern separation, Leutgeb et al. 2007); this is C-4 |
| E2 | DG cue groups bag / first / last / scene (fans 10, 4, 4, 10) | – | fans, weights | F | **(b)** DG fan-in from the PFC state itself (wiring) |
| E3 | recall WTA overlap·σ^ρ (ρ 2); θ 0.2; σ × 0.9995 per stored line; σ > 0.05 | recency: +0.05 J | 2, 0.9995, 0.2 | – | **(a)** C-4 completion with decaying strength |
| E4 | NEXT / COMPLETION read-outs | name recall (lesion 0.45 → 0.10) | – | R | **(b)** C-4 sequence links whose output joins the C-5 gate |
| E5 | awake recall: a quiet moment = 12 silent ticks; 8 episodes drawn uniformly; η 0.2 | the drift brake: C 0.57 → 0.00 without it (§8.3) | 12, 8, 0.2 | T, Tr | **(b)** C-4 spontaneous reactivation whose rate rises as external drive falls (C-1). Experience: quiet periods (awake replay: Foster & Wilson 2006; Carr, Jadhav & Frank 2011) |
| E6 | the thought is saved and restored around recall | keep the context | – | S: it breaks continuity; review 2: "recall saves and restores the thought" | **(d)** replay runs in the HPC → cortex loop while the thought persists; the C-6 mode signal marks it |

### F. Speech intention, turn-taking and value
Files: `ThinkAgent`, `SynthAgent`, `AgentV21`, `run_j.THINK_FINAL`, `life_synth.THINK_LIFE`.

| ID | mechanism | added for | params | tags | verdict |
|---|---|---|---|---|---|
| F1 | accumulator I ← I + G·e + u; release at I ≥ 1 | principle 4 | θ_I 1 | – (skeptic: latency 6–7 ticks under every lesion, a timer in effect) | **(a)** C-5 accumulate-to-bound (Cisek et al. 2009) |
| F2 | gains and urgency per thought kind: g0 reply 1; g0 scene 0.3 staged / 1.0 life; u reply 0.15; u scene 0 / 0.02; u_R1; urg_gate | timing; initiatives | 6 constants | S (per kind and per world) | **(b)** urgency learned from how long partners wait (a C-3 hazard predictor). Experience: waiting partners |
| F3 | thought open / close: a reply thought at another speaker's line end; a scene thought at scene / pointing onset in silence; it closes at the own utterance's end, another speaker's onset, a scene change or T_MAX 40; one utterance per thought | no monologues (0 extra turns) | 40 | Tr, T (40 vs the bench's 45–60-tick window) | **(b)** C-1 continuous thought + C-5 refractoriness after a release; no thought objects |
| F4 | transition window: a reply may release only at τ ≤ 7 (`late_release` 0; prior b 0 inside the window, 1 outside) | think: 3 and 17 overlap exchanges | 7 | T (bench wait timing) | **(b)** C-3 P_oth learns it; keep only a weak "floor taken" prior in P-1. Risk R4 |
| F5 | P_oth learned brake: target = another line starts within L 10; keys × τ ≤ 30; nogo 0.25 / 0.05; η 0.2 | turn-taking; lesion: 0 initiatives, speech after a recast 0.21 → 0.54 | 10, 0.25, 0.05 | T (L) | **(a)** a C-3 predictor |
| F6 | time cells τ ≤ 30 | interval timing | 30 | – | **(c)** striatal / hippocampal time cells (MacDonald et al. 2011; Mello et al. 2015), part of P-2 |
| F7 | turn features: kind, R ("I spoke since the last line"), pointing 'pt', + oth_kind and eye_feats (life only) | turn learners | – | F, S | **(b)** the C-1 state already holds own speech (efference) and the scene; no hand features |
| F8 | V_go → v2.1 NoGo veto: V_go < 0.2 blocks scene thoughts (replies never); no_react (target 0 after 50 ticks); prior 0.5; β 2; g_max 3 | 你好 to an empty room: 60–80 → 5–9 per tenth (v21 §4) | 0.2, 50 | S (scene only), T | **(b)** C-5 NoGo = negative learned value (C-3 + C-6). Experience: unanswered speech |
| F9 | reflex 7: a scene thought releases only after being open ≥ 10 ticks | one overlap; PROVED from world.py: the partner speaks ≤ 5 ticks after a scene appears | 10 | T (fitted to world.py) | **(d)** P_oth (C-3) learns it. Risk R4 |
| F10 | dips: no reaction within react_T 50 → r = 0, δ = −V; dip_scope | social replies; cost: held-out check 0.80 → 0.43 | 50 | T, Tr | **(b)** a TD critic over time cells predicts when the reaction is due (C-3 + C-6; reward-omission dips, Schultz et al. 1997) |
| F11 | coarse social detectors: someone spoke < 300 ticks ago; > 2 eye bits appeared / vanished in silence < 60 ticks ago | without them life social replies collapse (0.23 / 0.05 vs 0.41 / 0.43) | 300, 60, 2 | T, F | **(c)** innate person-present / arrival / departure detectors (IDEAS #17; newborn face orienting, Johnson et al. 1991), but read as C-1 recency traces, not fixed windows |
| F12 | scene_react: a line starting ≤ 10 ticks after an own scene-thought utterance is learned in the scene context (`ThinkAgent.step`) | greeting first (§8.2) | 10 | S, T, R | **(d)** the own utterance sits in the held thought (C-1), so the reply is learned in that context anyway |
| F13 | innate rewards: recast of an acceptable reply 1.0, imitation 0.2; world2 praise and eating | the value loop | given by the world | – | **(c)** P-4 drives (social reward in the striatum: Fareri & Delgado 2014) |
| F14 | eligibility windows: V_go 50, P_oth pending L 10, actor act_T 60 | credit assignment | 50, 10, 60 | T | **(b)** C-3 synaptic traces with one decay constant per area |

### G. Role-aware listening (v2.1)
File: `core_v21.RoleCore`.

| ID | mechanism | added for | params | tags | verdict |
|---|---|---|---|---|---|
| G1 | per-synapse eligibility trace of a heard line's R2 steps (`RecDelta`) | role-aware listening | – | – | **(a)** three-factor trace (Gerstner et al. 2018) |
| G2 | resolution rules: **cancel** if my mouth starts ≤ 12 ticks after the line, or another line starts ≤ 6 ticks after it; **commit** if a reward comes at onset ± 2 within 60 ticks of my utterance, otherwise at 12 ticks; cut or relink the NEXT link | drift loss at 900: 0.54 → 0.01; J +0.24; H 0.60 → 0.76 (v21 §2) | 12, 6, 60, 2 | T (6 = the bench's 3-tick model-reply gap), R | **(b)** own speech in the thought (C-1 efference input) + C-3 predicting the next line of every voice + C-6 RPE on the trace. Experience: overheard Q–A pairs and recasts. **Risk R1** |
| G3 | rejected variants: ral_q, strict cancel, own lines in the hippocampus, overlap contingency, ral_init | – | about 10 knobs | – | **(d)** |

### H. v2.2 additions so far (read only; numbers SIMULATED by the v2.2 agent)
Files: `core_v22.V22Core`, `agent_v22.AgentV22`.

| ID | mechanism | added for | params | tags | verdict |
|---|---|---|---|---|---|
| H1 | RandMix c_flat inside the v2.1 core (+ `alloc pow:0.5` tried) | step 2. base → cf: J 1.184 → 1.191 (mixed), colour −0.10, held-out taste +0.33 (`out/paired_base_cf.txt`) | – | F (pow:0.5 uses bench counts) | **(a)** as D5; keep flat only |
| H2 | the reply-veto family: veto_R1, veto_W 40, veto_eps 0.1, veto_rel, veto_gate 0.15, veto_fast 0.04, gain_reply | corrections cancelled (the agent speaks after 79–87 % of them); life 0.34 vs 0.62 | 7 knobs | **T** (40 is read off the world's ground-truth log, `an/cue_test.py`: corrections captured 1.00 at a gap ≤ 40, questions 0.33–0.35), S | **(d)**. Life free 0.47 / 0.50 (`fin`) vs echo-or-constant 0.64 / 0.61. Staged `fin`: J 1.04, drift loss at 900 0.45, at 3,000 0.90. `finF`: J 1.21, but 0.75 at 3,000 |
| H3 | dip_g 0.25; dip_scope 'pred' with a P_react predictor (dip only if P_react ≥ 0.5) | the dips' cost on check | 0.25, 0.5 | S | **(b)** the TD critic of F10 |

### I. Cross-cutting

| ID | mechanism | added for | params | tags | verdict |
|---|---|---|---|---|---|
| I1 | staged vs life parameter sets (`THINK_FINAL` vs `THINK_LIFE`: g0_scene, u_scene, oth_kind, life_eye, eye_feats, no_react) | tuning per world | 6 | S | **(d)** one agent, one parameter set |
| I2 | mode flags spread through code paths: `_replaying` (thought frozen, meaning and RAL off), `learning` / `storing`, the kind == 'scene' / 'reply' branches | replay hygiene | – | S | **(b)** one C-6 "encoding vs replay" broadcast (ACh-like; Hasselmo 1999) |
| I3 | birth calibration on `world.warmup_windows()` (FEAT0 usage, SoundGate start) | the allowed calibration | – | – | **(c)** spontaneous activity before birth calibrates sensory maps (Katz & Shatz 1996) |

## 3. Counts per verdict

| origin | rows | (a) | (b) | (c) | (d) |
|---|---|---|---|---|---|
| ARCH_v2 (incl. the v1 body, develop, tune, think, synth) | 62 | 12 | 29 | 13 | 8 |
| v2.1 | 8 | 2 | 4 | 1 | 1 |
| hearing | 8 | 0 | 3 | 0 | 5 |
| meaning | 2 | 0 | 2 | 0 | 0 |
| v2.2 (so far) | 3 | 1 | 1 | 0 | 1 |
| **total** | **83** | **15** | **39** | **14** | **15** |

- The 14 (c) rows merge into 5 priors:
  - P-1 (A1–A4, A6, A8);
  - P-2 (C1, D12, E1, F6, I3);
  - P-3 (C3);
  - P-4 (F11, F13);
  - P-5 (the replacement for B4).
- **Why the count grows** (ASSUMED, from the tags). Five recurring root causes:
  1. **Discrete symbolic interfaces force glue per route.** Item ids every 3 ticks, OneShotChain, 3 slots and per-source copy gates all need it (D1, D9, D10, E4, C2).
  2. **Event triggers instead of continuous dynamics force timing constants.** Line end, quiet moment, thought open/close and resolution windows all need them (20 T rows).
  3. **The world gives no negative experience, so it must be built in.** The bench's J = −1 rule forbids even one overlap, and probe worlds never give feedback (F4, F9).
  4. **Own speech is excluded from the predicted stream, so role rules are needed** (D3, D18, G2, H2).
  5. **Each modality and each world got its own normaliser instead of one area-general rule** (B6, B10–B12, C4, C5, C10, D15, I1).

## 4. The minimal generic set implied bottom-up

All of the following is ASSUMED design, mapped to existing code.

### Core mechanisms (fixed code, the same everywhere)

**C-1 Area dynamics.**
- **Rule:**
  - random sparse fan-in at birth (local in sensory areas);
  - k-WTA at a fixed active fraction;
  - per-unit adaptation (subtract a robust running mean / MAD of the input; fatigue);
  - per-population divisive normalisation;
  - persistence (self-excitation with leak λ; held until displaced by competition).
- **Per-area numbers** (size, fan, active fraction, λ, adaptation τ) belong to the wiring plan.
- **Exists in:** `randlayer.RandMix`, `rules.Hippo.dg`, `SynthCore._ct_input` / `tick_thought`, `hear.SoundGate`, `core.Ctx` (set point), `eye_rate`, `MeanCodes.b`.
- **Replaces:** B2, B3, B6, B11, B12, B14, C4, C5, C10, D1, D2, D4, D7, F3, F7, and the forms of F11 / E5.
- **Brain:** normalisation as a canonical computation (Carandini & Heeger 2012); adaptation to stimulus statistics (Dean et al. 2005); persistent activity (Wang 2001).

**C-2 Unsupervised fan-in learning.**
- **Rule:** instar with heterosynaptic normalisation; use-it-or-lose-it pruning with regrowth; plasticity falling with use (max(1/n, floor)).
- **Exists in:** RandMix train/prune, `Templates.learn`, `MeanCodes._learn`, `Outstar`, the T_W co-holding links.
- **Replaces:** B3, C6, D4, D14.
- **Brain:** competitive Hebbian learning (Linsker 1986); synapse overproduction then pruning (Huttenlocher 1979).

**C-3 Predictive projection learning.**
- **Rule:**
  - every projection learns by the signed delta rule to predict its target area's activity at the next event (or the reward line);
  - Δw = η·(t − p)·x/|x| × the synapse's eligibility trace × one broadcast gain;
  - one plasticity scalar per area: hippocampus one-shot, cortex slow, instinct ≈ 0 (IDEAS #17: "plasticity is a spectrum").
- **Exists in:** `rules.Delta` (GEN, gates, gateR), `agent_think.SparsePR` (P_oth, V_go, P_react, ral_q), `rules.Critic`, `core_v21.RecDelta`.
- **Replaces:**
  - A9, C7, C8, C9, D18, F2, F4, F10, F14, G2, H3;
  - the inverse model for B4.

**C-4 Fast episodic region.**
- **Rule:**
  - C-1 to C-3 at maximal plasticity: a fixed DG expansion, one-shot cells with decaying strength, completion and successor links;
  - spontaneous reactivation whose rate rises when external drive is low;
  - reactivations drive C-3 in the cortex.
- **Exists in:** `rules.Hippo`, `Core.awake_recall` / `_replay`, the recency term ρ.
- **Replaces:** E2, E4, E5, E6.

**C-5 Selection loop (basal ganglia).**
- **Rule:**
  - candidates = the most active item of every projecting area (held thought, recalled successor, recognised object, location word, motor programs, internal operations);
  - a gate per (context, source) learned by C-3 with the RPE;
  - evidence accumulates to a bound (C-1 persistence);
  - NoGo = negative value; refractoriness after a release; small exploration noise.
- **Exists in:** `ThinkAgent._intend`, `ThinkBody` (GO from REL), the copy gates in `Core._drives`, gateR, `val_veto`.
- **Replaces:** A5, C2, C8, C12, D9, D10, D16, E4, F3, F8, H2.
- **Brain:** BG selection and output gating (Redgrave, Prescott & Gurney 1999; O'Reilly & Frank 2006).

**C-6 Two broadcasts.**
- **(i) RPE** δ = r − V (dopamine-like; Schultz et al. 1997): multiplies the traces of the value, actor and gate projections.
- **(ii) An ACh-like mode signal:**
  - high on surprise or novelty, where it raises encoding into C-4 and C-3's learning rate;
  - low during replay (Hasselmo 1999).
- **Cost:** one broadcast more than the principle's "at most one".
- **It replaces:** the `_replaying` / `learning` / `storing` flags, the own-voice M_listen gate and the thought save/restore. These are rows I2, E6, D3 and D18.

### Innate priors (budget 5, fixed, never added per task)

- **P-1 Body and brainstem:** muscles, voice physics, syllable CPG, motor executor with END and breath, a *relative* vocal brake, the efference copy.
  - Evidence: the PAG / brainstem vocal network (Jürgens 2002); auditory suppression during own vocalisation (Eliades & Wang 2003); innate vocal turn-taking in marmosets (Takahashi et al. 2013).
- **P-2 Wiring plan:** areas and sizes; which area projects to which; topography (retinotopy); local 3-D sensory layers; the DG expansion; BG gates read the context; a time-cell basis; birth calibration by spontaneous activity.
  - Evidence: genetic arealisation (O'Leary & Sahara 2008); maps before eye opening (Katz & Shatz 1996).
- **P-3 Orienting:** to onsets and novelty, to faces and voices, gaze and point following.
  - Evidence: Johnson et al. 1991; Hood, Willen & Driver 1998; Butterworth & Jarrett 1991.
- **P-4 Drives and values:** innate reward for social approval or a reaction (the prosocial drive, IDEAS #17), food when hungry, negative value for distress; coarse social detectors (a person is present / arrives / leaves / addresses me).
  - Evidence: contingency detection in infants (Watson 1972); still-face distress (Tronick et al. 1978).
- **P-5 Spontaneous motor activity:** babbling and exploration noise that falls as learned value rises.
  - Evidence: canonical babbling (Oller 1980).

### What gets removed
- **The 15 (d) rows outright:** B4, B5, B7–B10, B13, D3, D19, E6, F9, F12, G3, H2, I1. That is about 70 knobs, PROVED by summing the dead and patch knobs in `knobs.py`'s layers.
- **The 39 (b) rows as code.** Their behaviour has to come back from C-1 … C-6 plus experience.
- **The worlds must supply that experience** (no new agent code):
  - pauses inside lines;
  - many voices and noise;
  - a foveated eye with saccades (`eye_mode='fovea'` exists);
  - overheard Q–A pairs between other people;
  - partners who sometimes do not wait, so that overlaps are possible early in life;
  - partners who sometimes give no feedback;
  - babbling with imitation;
  - mishearing (IDEAS #14 addendum).

## 5. The 5 riskiest replacements, and how to test each

| # | hand-built part → generic version | why it may not match (evidence) | test (existing scripts) | pass / falsify |
|---|---|---|---|---|
| **R1** | RAL cancel/commit rules (G2) → own speech enters the thought as efference input (D3), C-3 predicts the next line of every voice (speaker from B14), C-6 RPE commits the trace | RAL is the largest single gain (J +0.24, drift 0.54 → 0.01). Earlier generic tries fell short: own speech in the thought alone gave drift −0.23 vs −0.28 (skeptic); Xcut scored 0.72 / 0.80 vs RAL's 0.81 / 0.83 on the symbolic probe; the learned question detector cost J −0.20 (v21 §2.4). In life no timing cue separates corrections from questions (`cue_test`: gap ≤ 40 captures 1.00 of corrections but 0.33–0.35 of questions) | 1. `v21/regress/gen_probe_v21.py`, recast and mixed tutors, 3 seeds. 2. `run_v22.py` frozen protocol, seeds 0–4, with `ral=0` + the generic path. 3. Lives on seeds 5 / 6 with `life_cmp.py` (corrCommit / qCommit columns) | **Pass:** symbolic frozen held-out ≥ 0.75; staged drift loss at 900 ≤ 0.05; H within 0.05 of v2.1; life commits ≥ 0.8 of corrections and ≤ 0.3 of questions. **Falsify:** drift loss at 900 ≥ 0.15 on ≥ 3 of 5 seeds |
| **R2** | 3-tick clock + vigilance Z + the hand front end (B2, B3, B6–B12) → C-1 / C-2 per-tick competitive layer with area-general robust adaptation | Today the bench is bit-exact and purity is 1.00. A learned layer 2 lost information (0.71–0.83, §7.11). Untrimmed or 2.5ν-trimmed floors ran away or collapsed (hearing §6.3). Everything downstream assumes one discrete item per 3 ticks | 1. `hearing/record.py` + `ear_eval.py` grid: 2 / 16 / 64 bodies × σ 0 / 0.5 / 1 / 2. 2. `life_eval.py` with noise σ 0 → 1 → 0.5 → 1. 3. Bench clean and σ 1, 5 seeds. 4. The **same** adaptation constants applied to the eye and to the meaning base rate | **Pass:** near recall ≥ 0.95 at σ ≤ 1; purity ≥ 0.98; categories ≤ 1.15 × the noise-free count; bench J within the seed SD of 0.06; colour unchanged. **Falsify:** any deaf day (0 lines) or purity < 0.95 |
| **R3** | 4 task-wired copy routes + the fixed recall → comparator → VR order (D9, D10, D16, E4) → one C-5 output gate over every area's top item + learned operation choice (or all operations in parallel) | gateR carries naming (lesion: B 0.91 → 0.40); the prompt gate carries ritual (0.91 → 0.14). RPE-trained gating is sample-hungry (PBWM), and life rewards are sparse (16 % of heard lines) | 1. Staged frozen protocol, 5 seeds, vs v2.1 Steps 1+2 (`s12`: J 1.249). 2. Held-out what / where / slot / ritual. 3. A lesion of the unified gate must reproduce each route's collapse. 4. Latency vs difficulty (skeptic: 6–7 ticks under every lesion) | **Pass:** held-out what ≥ 0.75, where ≥ 0.85, ritual ≥ 0.95, J within 0.06. **Falsify:** what < 0.7 or ritual < 0.9 |
| **R4** | turn reflexes fitted to the bench's timing (F3 T_MAX, F4 τ ≤ 7, F9 reflex 7, A6) → C-3 P_oth with a weak "floor taken" prior and a relative brake | Principle 8: any overlap makes J −1, and learning the brake needs negative experience. Turn learners misfire when partners give no feedback (open problem 7). In crowds F is on 0.52 of the day | Delete the 3 reflexes. Run staged, lives on seeds 5 / 6, and world2 with 16 bodies. Plot overlaps per 1,000 utterances per tenth of life. **Needs a coordinator decision:** a developmental mechanics metric that allows early overlaps | **Pass:** overlaps reach 0 by the 2nd tenth of life and stay 0; v2.1's life gains hold (appropriate ≥ 0.38, social ≥ 0.30). **Falsify:** overlaps still > 1 per 1,000 in the last tenth |
| **R5** | offline enumerated babbling through `world.voice` (B4, B5) → P-5 online babbling + a C-3 auditory → motor inverse predictor | Random physics has no smooth forward model. Online exhaustive search costs 35,960 × 3 ≈ 108k ticks, about 11 % of a 10⁶-tick life or about 4.5 world2 days (PROVED arithmetic). With the full mouth a random babble hits a lexicon char at p ≈ 3·10⁻⁵ (bench README) | 1. A life with the bench's `body.babble` + `keep='reward'` (caregiver imitation), random and lawful physics. 2. Bound programs vs time; free replies | **Pass:** ≥ 90 % of lexicon chars bound by 20 % of life on lawful physics. If it works only on lawful physics, that is a world requirement, not agent code. **Falsify:** < 50 % bound on lawful physics |

**Honourable mention:** the comparator and VR constants (C8, C9). Tune found cliffs on both sides of today's values, and colour 0.93 / count 0.76 depend on them.

## 6. Falsifying the claim as a whole ("a fixed core learns everything without new code")

1. **Deletion ladder.** Go from the lowest risk to the highest:
   - first the (d) rows;
   - then the (b) rows one at a time;
   - R1–R5 last.

   At each step compare paired against the v2.2 reference: 5 seeds on the frozen staged protocol, J and J2, plus lives on seeds 5 / 6 against the same-life references. Keep a step only if it stays within the seed noise and the life targets still hold.
   - *Falsified if* any (b) row can only be recovered by adding code beyond C-1 … C-6.
2. **No-new-code test.** Freeze the code hash. Add world2 tasks that are not built in, for example:
   - requests ("去拿X");
   - the class task "X能不能吃";
   - "before" questions;
   - new caregivers.

   Ability must rise with experience alone.
3. **Scale curve.** Run 1× / 4× / 16× in world2. Today a 4× meaning layer loses 0.11 J (meaning §5), a warning that the core is not yet "bigger is better".

## summary_zh
1. **盘点结果：** 现在的智能体一共有 83 个机制、265 个可调参数；另外代码里有 25 处写死了测试台的"3"（3 个位置、3 拍一个字）。每加一种能力，就多出 10–31 个参数。
2. **四类判定：**
   - 已经通用的 15 个；
   - 可以由通用机制加经验产生的 39 个；
   - 合理的先天部分 14 个（可以合并成 5 项）；
   - 应删除的脚手架 15 个。
   - ARCH_v2 之后新加的 21 个机制里，只有 3 个是通用的。
3. **隐藏的手工规则：** 83 个机制里有 54 个带着这类规则：
   - 有的计时常数是照着测试世界的剧本定的，比如"反射 7"的 10 拍、"别人马上回答"的 6 拍、v2.2 的 40 拍（直接从世界的真实日志读出来的）；
   - "一生"模式和测试台用两套不同的先天参数；
   - 咿呀学语时偷看了世界的发音表。
4. **由下而上推出的最小通用集合，是 6 个机制加 5 项先天：**
   - 6 个机制：
     - 每个区域同一套动力学（随机稀疏连接、k-WTA、适应/归一化、持续活动）；
     - 无监督的扇入学习（instar、用进废退）；
     - 每条投射都用带资格迹的有符号预测误差规则；
     - 快速的情景记忆区，安静时自发回放；
     - 基底节选择回路，在所有区域的输出里选一个；
     - 两个广播信号：奖励预测误差，以及类似乙酰胆碱的"编码/回放"状态。
   - 5 项先天：身体和脑干、连线蓝图、朝向反射、驱力和价值、自发运动。
5. **风险最大的 5 处替换**（每处都写了检验方法和成败标准）：
   - "分角色边听边学"的撤销/保留规则；
   - 按 3 拍采样、新建类别的耳朵；
   - 按任务接好的 4 条复述通道和固定的思考顺序；
   - 按测试台时间定的轮流说话反射；
   - 离线遍历的咿呀学语。
6. **v2.2 正在发生的事，就是"规则越堆越多"：** 一个否决开关带来副作用，就再加一个开关去补。"一生"的自由回答升到 0.47–0.52，但仍低于参考的 0.61–0.64；而无人指导 3,000 段对话后，成绩几乎掉光。