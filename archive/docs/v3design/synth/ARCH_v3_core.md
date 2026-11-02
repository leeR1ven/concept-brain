# ARCH_v3_core: one fixed core for the concept-brain agent

**Who and when.** Synthesis member of the v3 design round, 2026-11-01. This is design only: no training run and no server job.

**What it merges:**
- the five member reports in `arch/v2/v3design/`: audit, neuro, learning, develop and eval;
- the designer's ideas (`docs/IDEAS.en.md`, in particular #12, #14, #17 and #19) and the born-wired chapters 36 and 39;
- the final v2.2 report (`v2/v22/report.md`);
- the designer's three decisions of 2026-11-01:
  1. mechanics errors are allowed in childhood;
  2. learned weights are inherited through a lineage;
  3. the final product must be pure neurons.

**Labels:**
- **PROVED**: code, count or arithmetic.
- **SIMULATED**: a run, with its file named.
- **CITED**: a paper or a report.
- **ASSUMED**: design judgement, not tested.

**Scratch files** (all in `docs/v3design/synth/`). Each script ran locally as one process under 50 MB, in at most 50 s, after a `free -m` check.

| file | what it is |
|---|---|
| `drift_toy.py` / `.out` / `.json` | a symbolic toy that tests the new context structure under silence and timing jitter (5 seeds) |
| `knob_list.py` / `.out` | a read-only census of all 265 named knobs and their defaults |
| `gene_map.py` / `.out` | maps every knob to a gene, a species constant, a size, the seed, or deletion |
| `checks.py` / `.out` | the arithmetic behind the walk in §6 and the cost estimate in §7 |

---

## 0. Bottom line

1. **The core is 4 mechanisms, 2 broadcasts, 2 objectives and 5 innate priors.**
   - The 4 mechanisms:
     - M1, the canonical area;
     - M2, one three-factor plasticity rule;
     - M3, the hippocampal index;
     - M4, the basal-ganglia selection loop.
   - The 2 broadcasts:
     - δ, a dopamine-like reward-prediction error;
     - μ, an acetylcholine-like signal for encode vs replay.
   - The 2 objectives:
     - O1, predict the next input and when it comes;
     - O2, TD on innate reward.
   - The 5 priors (body, wiring plan, orienting, social detectors, values) have a hard budget of 17 named sub-items.
   - That is 13 named items, against today's 83 mechanisms and 265 knobs (CITED audit).
   - Every current ability is one of three things:
     - an instance of these items;
     - deleted;
     - moved into the world (§1, §8).
2. **The decisive new structure: the cortex's context is anchored on the latest event × its age.**
   - The latest event is source-coded: "self" arrives through the efference copy, "other" through the ear.
   - Its age is read from a bank of log time cells.
   - Self-in-the-stream (the neuro and develop fix) is not enough on its own.
     - Whenever the agent stays silent, the next question arrives at exactly the delay of a told answer. It is then learned as the answer. That is v2.2's collapse.
     - SIMULATED (`drift_toy`, 5 seeds; formats, then 3,000 untutored dialogues with 30 % silence):

       | context | at the end of formats | after 300 untutored | after 900 | after 3,000 |
       |---|---|---|---|---|
       | latest × age | 0.99 | – | – | **0.99** |
       | self in the stream, no age | 0.99 | 0.09 | 0.00 | – |
       | ARCH_v2 | – | 0.00 | – | – |

     - With ±50 % caregiver timing jitter, the anchored context still holds 0.97.
     - Time features that are not anchored fail too: 0.72 → 0.31 after 3,000 untutored.
3. **The world must model the answer.**
   - In today's recast-only world even the anchored core collapses: 1.00 → 0.00 by 2,100 untutored dialogues (SIMULATED).
   - The reason: a told answer and the next question both start at max(Q_end + 45, my stop + k). The world times them alike (PROVED by v2.2 from `world.py`).
   - With caregiver formats (she answers herself after about 10 ticks), with model replies or with overheard answers, the core is stable.
   - So world changes W1–W5 come first in the migration, before any agent change. This confirms develop and neuro with v2.2's evidence.
4. **The members' disagreements are resolved** (§1.7):
   - **Broadcasts:** two global broadcasts, δ and μ.
     - Per-area precision is computed inside each area by its normalisation pool.
     - Change points are the area's own onset and offset transients.
     - So each synapse still reads at most one broadcast.
   - **Prior budget:** 5 priors with 17 sub-items, both counts hard.
   - **Self vs other:** no tag prior. "Self" is an efference pathway in the wiring plan, plus a learned forward model.
   - **Curiosity:** no curiosity drive in v3.
   - **Syllable oscillator:** part of the body's CPG, shared with the ear and corrected by recognition.
   - **The seven hearing keys:**
     - `h_gate`, `h_denoise` and `h_level` become M1's set-point and normalisation in every area;
     - `h_relock` becomes the oscillator's phase correction;
     - `h_conf` and `h_nocreate` become clarity-gated recruitment, judged per window;
     - `h_vbound` and `h_back` are deleted.
5. **The genome has 30 genes** (PROVED, `gene_map.py`). Of the 265 knobs:

   | fate | knobs |
   |---|---|
   | fed into 21 of the genes | 38 |
   | fixed species constants (body physics, normalised thresholds) | 25 |
   | sizes that scale with s | 11 |
   | the lineage seed | 2 |
   | **deleted** | **189** |

   Nine genes are new. They include the restoration cap, the time-bin base, the TD discount, the refractoriness and the four value weights.
6. **One unbroken life stays stable without sleep** (§6.5). v2.2 has five routes to drift. Each one is closed by structure, not by a rule:

   | drift route | closed by |
   |---|---|
   | the next question after my answer | the self-coded latest event |
   | the next question after my silence | the age anchor |
   | replay of partner → partner links | own lines stored in the hippocampus, with age in the keys |
   | pruning by a global event clock | a prune clock counted in the unit's own firings |
   | extinction into silence | bounded TD extinction plus urgency |

   - The pass bar is accuracy ≥ 0.85 at both 3,000 and 10,000 untutored dialogues.
   - v2.2 scores 0.16 at 3,000 (CITED).
7. **The designer's decisions are built in:**
   - **Errors allowed in childhood (option 2).**
     - Maturity is reached at **5,000 heard syllables**, about the first third of a 10⁶-tick bench life.
     - After maturity: 0 overlaps, 0 garble and 0 extra turns, in every tenth of life.
     - The bench-fitted turn reflexes (#5, #7, #7b, T_MAX, the reply/scene constants) are deleted.
     - The innate talk-over brake stays.
   - **Lineage inheritance (§2.4).**
     - Merge happens only within a lineage, from one shared birth wiring.
     - Merging uses a sign-elected mean of each parent's change since birth, not a plain average.
     - Offspring inherit the slow cortical weights. The hippocampus and the thought start empty, and the values are never merged.
     - This is Lamarckian, closer to culture than to biology. That is acceptable for the goal.
   - **Pure neurons in the final product.**
     - During development a shortcut is allowed only with its circuit and an equivalence check.
     - The last migration step swaps every shortcut back and verifies it.
8. **Cost of the final pure-neuron build** (ASSUMED sizes, PROVED arithmetic, `checks.py`):

   | | s = 1 | s = 4 |
   |---|---|---|
   | units | 0.11 M | 0.42 M |
   | synapses | 16 M | 63 M |
   | memory per agent | 125 MB | 0.5 GB |
   | CPU, one 10⁶-tick life | about 500–1,500 s | about 2,000–6,000 s |

   - Today's v2.2 life takes 1,000 s (CITED).
   - On the A800 GPU, batching is required: at born-wired's unbatched 13.3 ms per tick a life would take 13,300 s.
9. **First experiments** (§9), cheapest risk first:
   - X1: the anchor on the real symbolic harness (local, minutes);
   - X2: attribution of v2.2's collapse (2 A800 jobs);
   - X3: the hearing keys → M1 (local);
   - X4: depth in space vs in time (local);
   - X5: caregiver timing jitter on v2.2 (1 job).
10. **Top open risks** (§10):
    - the anchored context may cost generalisation;
    - dependence on world changes;
    - abstraction without backprop;
    - learned turn-taking may not reach 0 errors by maturity;
    - RPE-trained source gating is sample-hungry.

---

## 1. The fixed core

### 1.1 The whole core on one page

```
 ear 8x6 ─► EAR0 (fixed 3-D local layer) ─► SYL (syllable assemblies; top-down = GEN's prediction × (1−π))
            [M1 set-point + normalisation]      │                         VOICE (speaker residual)
 eye ─────► EYE (adapting; retinotopy / fovea) ─┼─► VR, LOC readouts (eye/place → word)   P4 detectors
 motor ── efference ─► SELF source ─┐           │                                         (present/arrive/leave,
       └─ forward model ─► SYL (own voice explained away)                                   addressed, pleased/upset)
                                    ▼           ▼
                     LATEST = the latest event (items × age bin × source), persistent until the next event's onset
                                    │ (at each new onset the old event moves into)
                     THOUGHT = older content (persistent, ≤ k_th items, fatigue → favourites)
          TIME = log time cells since each pathway's last onset/offset (other voice, own voice, eye event, person)
                                    ▼
      MIX = random conjunctions; EVERY unit has one branch on LATEST × TIME + 2 random branches
            (LATEST, THOUGHT, EYE, MEAN, P4, CA1-recall)          ─► GEN (O1: next heard item + END, at this age)
                                                                   ─► CRITIC V (O2)  ─► δ
                                                                   ─► M4: Go/NoGo per source + release channel
      M4 output: D_j = Σ_sources g_s·c_s,j  → accumulate (G+u)·e to bound → GO ─► P1 body (CPG, WTA, END, breath,
                                                                                   talk-over brake) ─► 32 muscles
 HIP: DG ◄ (LATEST, THOUGHT, EYE, MEAN, TIME) ─► CA3 one-shot + successor links ─► CA1 ─► MIX (recall) / SYL (replay)
 broadcasts: δ (critic + P5 values) → value synapses and M4;  μ (input presence) → pathway gains, HIP write ↔ replay
```

### 1.2 Mechanisms

#### M1. The canonical area: one circuit for every area

**Definition (content-free).** An area is a sheet of N·s binary units. Each area has these properties:
- **Birth wiring.** Random sparse fan-in per input pathway, drawn once from the lineage seed.
- **Normalisation.** Each pathway is divided by its own running total, so every pathway weighs the same whatever its line count.
- **Competition.** k-WTA at a fixed active fraction, through shared inhibition.
- **Adaptation.** Each unit has an adaptive set-point, a sign-rule tracker of a low quantile of its own drive that is subtracted from it. Each unit also has fast fatigue.
- **Persistence.** Leak λ.
  - Persistent areas have λ = 1: they hold their content until a new event displaces it.
  - At the sending pathway's next event onset, a persistent area loads the new pattern.
  - Content already held is scaled by λ_c, so the new event wins the competition.
- **Prediction input.** A top-down, lateral or efference input enters with weight `w = w_max·(1 − π_a)`.
  - π_a is the area's precision: one minus the running spread of the area's best-match margins inside events.
  - The area's normalisation pool computes π_a. It is read only inside the area.
- **Recruitment (category areas only).** A winner is accepted only if its match is ≥ θ_acc.
  - Otherwise the least-used units with the best birth match are recruited.
  - Recruitment happens only for a **clear window**: every slice above the set-point margin k_on. This is `h_conf` generalised, and it is judged per window, so new syllables still form in a noisy room.
  - Representation learning is scaled by π_a.
- **Event boundaries.** The onset and offset transients of the pooled drive are the area's event boundaries.

**Brain basis (CITED).**
- The canonical microcircuit (Douglas & Martin 2004).
- Divisive normalisation as a canonical computation (Carandini & Heeger 2012).
- Adaptation to stimulus statistics (Dean, Harper & McAlpine 2005).
- Persistent activity (Wang 2001).
- ART vigilance (Carpenter & Grossberg 1987).
- OFF responses in auditory cortex (Scholl, Gao & Wehr 2010).
- Working memory updated at event boundaries (Zacks et al. 2007).
- A compressed timeline of recent stimuli in lateral PFC, i.e. stimulus-specific time cells (Tiganj et al. 2018).
- Time cells (MacDonald et al. 2011; Shankar & Howard 2012).
- Expected uncertainty as precision (Yu & Dayan 2005).

**Starts from:**
- `randmix/randlayer.RandMix` (c_flat);
- `synth/rules.Templates`;
- `v21/agent_v21.VoiceCats`;
- `meaning/meanlayer.MeanCodes`, used online;
- `synth/core_synth.SynthCore`: `_ct_input`, `tick_thought`, `_ct_decay`;
- `hearing/hear.SoundGate`, `hear.cell_gate`;
- `bench/frontend.Ear0`;
- `rules.Hippo.dg`.

**Absorbs:**
- audit rows B1–B3, B6–B12, B14, C4, C5, C10, D1, D2, D4, D7, D14, F3, F6, F7, A9, I3, and the form of F11;
- the hearing keys `h_gate`, `h_denoise`, `h_level`, `h_conf` and `h_nocreate`;
- the scene baseline #23, `life_eye` and `eye_rate`;
- the wandering trigger, the ≤ 12-item list and the current-line buffer.

#### M2. One plasticity rule: three factors with an eligibility trace

**Definition.** Every plastic synapse changes by `Δw_ij = η_a · m · e_ij`, where e_ij is an eligibility trace with time constant τ_e. The rule has two forms:
- **Prediction form:** `e ← x_i (t_j − p_j)/|A_P|`, with `p = clip(Σ w x, 0, 1)`.
  - t is the target area's next activity.
  - This is R2: "predicted → stop strengthening; predicted but absent → weaken".
- **Representation form:** `e ← y_j (x_i − w_ij)`, the instar rule.

**The factors:**
- **m:**
  - m = 1 by default;
  - m = δ on value synapses (the critic, the actor and the M4 gates).
- **η_a:** `η_level · max(1/n_use, floor)`, so plasticity falls with use.
  - The level comes from the area's place in the hierarchy (gene ratio r_η per level). This is the plasticity spectrum of IDEAS #17.
  - Representation synapses also scale their η by π_a.

**Structural part.** A synapse is pruned if its input line was silent during the last T_pr firings of its own unit.
- The slot then regrows at the next partner in its slot's lineage-shared reserve list.
- A unit that does not fire never prunes.

**Brain basis (CITED).**
- Three-factor rules (Frémaux & Gerstner 2016; Gerstner et al. 2018; Yagishita et al. 2014).
- Dendritic prediction error (Urbanczik & Senn 2014).
- Competitive Hebbian learning (Linsker 1986).
- One calcium rule underneath (Shouval, Bear & Cooper 2002).
- Synapse turnover (Holtmaat & Svoboda 2009).

**Starts from:**
- `synth/rules.Delta` (R2);
- `v21/core_v21.RecDelta` (the trace);
- `rules.Critic`;
- `synth/agent_think.SparsePR` (as the prediction form);
- `rules.Outstar` (R7);
- `rules.Templates.learn` (R5);
- the RandMix instar and prune.

**Absorbs:**
- R1 (as η = 1 in CA3), R2, R5–R11;
- RAL's per-synapse trace;
- the meaning instar;
- the forward and inverse vocal maps;
- dips, which become TD error;
- `act_T`, which becomes trace decay.

#### M3. The hippocampal index

**Definition.**
- **DG** is a very sparse M1 instance. It reads LATEST, THOUGHT, EYE, MEAN and TIME, each pathway normalised.
- **Encoding.** At each event offset, while μ is high, the DG pattern is bound one-shot into a CA3 assembly (M2 at η = 1).
  - It is linked to the previous episode by asymmetric successor links.
  - Own events are stored like any other, source-coded.
- **Recall** completes the current DG pattern: best overlap × strength^ρ, accepted if ≥ θ_rec.
- **Forgetting.** Strength decays as a fraction of capacity.
- **Replay.** When μ is low, CA3 reactivates spontaneously and uniformly. CA1 then reinstates the episode in the cortex:
  - its context drives MIX, while μ gates the live pathways down;
  - its successor drives SYL as the target;
  - M2 learns from it at the normal rate.
- Nothing is saved or restored, and there is no sleep.

**Brain basis (CITED).**
- The hippocampal index (Teyler & DiScenna 1986); DG pattern separation (Leutgeb et al. 2007).
- Complementary learning systems (McClelland, McNaughton & O'Reilly 1995).
- One-shot plasticity over seconds, BTSP (Bittner et al. 2017).
- Awake replay (Foster & Wilson 2006; Karlsson & Frank 2009; Jadhav et al. 2012).
- Encoding at event boundaries (Ben-Yakov & Henson 2018; DuBrow & Davachi 2013).
- ACh switching encoding and retrieval (Hasselmo 1999, 2006).
- Sequence learning in CA3 (Levy 1996).

**Starts from:**
- `rules.Hippo`;
- `Core.awake_recall` and `_replay`;
- `core_v21.awake_recall`;
- `meaning/mcore.MeanCore`'s DG cue.

**Absorbs:**
- audit rows E1–E6;
- the meaning DG cue and `hipnorm` (which is per-pathway normalisation);
- the thought save/restore;
- the 12-tick quiet-moment trigger;
- the NEXT and COMPLETION routes, which become CA1 sources for M4.

#### M4. The selection loop (basal ganglia–thalamus): one class for every "select one, release it"

**Definition.**
- **Candidates:** the top item of every source area that projects onto the motor programs.
  - The source list belongs to the wiring plan: GEN's prediction, LATEST's items (copy/echo), CA1's recalled successor or completion, VR's word at the attended place, and LOC's place word.
  - Internal operations (attend, recall) and the programs themselves are candidates too.
- **Gating per source.** Each source s has a Go_s − NoGo_s read from MIX by context-only keys. These are M2 weights with m = δ.
  - The combined drive is `D_j = Σ_s g_s·c_{s,j}`.
- **Release.** The release channel accumulates `I ← I + (G + u)·e`.
  - G = clip(Go − NoGo) of the release channel; e = the margin of the best program.
  - The bound is 1. I resets at a release and at each new event onset.
- **After a release:** refractory for τ_ref.
- **Exploration:** ε.
- The prior P1's brake can veto any vocal onset.

**Brain basis (CITED).**
- Basal-ganglia selection (Redgrave, Prescott & Gurney 1999; Gurney, Prescott & Redgrave 2001).
- PBWM gating with indirection (O'Reilly & Frank 2006; Kriete et al. 2013; Chatham, Frank & Badre 2014).
- Go/NoGo learning (Frank 2005).
- Accumulation with urgency (Cisek et al. 2009; Bogacz & Gurney 2007).

**Starts from:**
- `synth/agent_think.ThinkAgent._intend`;
- `synth/body_think.ThinkBody` (GO/REL);
- the copy gates in `Core._drives`;
- `v21/core_v21` gateR (`_say_route`);
- V_go and `val_veto`;
- the R6 actor.

**Absorbs:**
- audit rows A5, A7, C2, C8 (the lead and threshold become competition), C12, D9–D12, D16 (the thought steps become operations), E4, F1–F5, F8–F10, F12, F14, H2, H3;
- P_oth and V_go as separate learners. P_oth's "someone will start" becomes part of GEN's time-specific prediction.
- reflexes #5, #7 and #7b;
- one-utterance-per-thought and T_MAX 40;
- the reply/scene split of the accumulator constants.

### 1.3 Broadcasts (two)

| | definition | computed from | read by | brain (CITED) | replaces |
|---|---|---|---|---|---|
| **B1 δ** (dopamine-like) | δ(t) = r(t) + γ·V(t) − V(t−1) | the critic V over MIX × TIME; r from P5 | value synapses (critic, actor, M4 gates) | signed RPE (Schultz, Dayan & Montague 1997); dips at the omitted reward's time | R6 δ, dips + `react_T`, `val_veto`, `act_T`, V_go's target, the v2.2 veto |
| **B2 μ** (acetylcholine-like) | μ follows the presence of external input with τ_μ | any sensory pathway above its set-point | (i) the gain of live afferent pathways into MIX and the hippocampus; (ii) when μ < 0.05 (about 12 silent ticks) the hippocampus reactivates instead of writing | ACh high during encoding, low during SWR and consolidation (Hasselmo 1999, 2006) | the `_replaying` / `learning` / `storing` flags, the thought save/restore, `quiet_ticks` |

**Cost.**
- Two scalars per tick, plus one precision value per area.
- "At most one broadcast" still holds per synapse:
  - value synapses read δ;
  - representation synapses read their own area's π_a, which is not a broadcast;
  - μ acts on pathway gains and on hippocampal writing.
- The system as a whole has one broadcast more than today. Its brain basis is the strongest of the candidates.

### 1.4 Objectives (two)

- **O1. Every prediction synapse predicts its target area's next activity.**
  - GEN predicts the next heard item and END, given the latest event and its age: "what comes next, and when".
  - The forward model predicts my own voice from the efference copy.
  - VR and LOC predict the heard word from the attended place.
  - The age dependence also gives "another speaker is about to start", which replaces P_oth.
- **O2. TD on innate reward.**
  - The critic predicts discounted reward from P5.
  - δ trains actions, gates and operations.
- **Speaking** is M4 releasing a predicted, recalled or recognised item through the motor programs. Speech is caused by thinking.

### 1.5 Innate priors

**Hard budget: 5 priors and 17 sub-items.** Neither count may grow per task. A change makes a new species version and needs a new sealed set.

| prior | sub-items (17) | why the brain has it (CITED) | starts from | absorbs |
|---|---|---|---|---|
| **P1 Body and brainstem** | (1a) motor executor: program WTA, END, breath limit, D/I/M force ramp; 32 muscles (world physics). (1b) Syllable CPG, shared with the ear's sampling phase; reset by the ear's onset transient, phase-corrected by recognition (the best-matching phase wins). (1c) Talk-over brake: no vocal onset while a non-self voice is above the ear's set-point margin, at this tick or the last. (1d) Spontaneous motor activity: babble/explore at rate r_bab. | PAG and brainstem vocal pattern generators (Jürgens 2002); frame/content (MacNeilage 1998); theta-syllable parsing (Giraud & Poeppel 2012); innate turn-taking (Takahashi, Narayanan & Ghazanfar 2013); yielding at overlap (Schegloff 2000); canonical babbling (Oller 1980) | bench `body.Body`, `body_think.ThinkBody`, the relock logic of `agent_h`, `Body.babble` | A1–A4, A6, B2, B7, reflex #3 (`brake_now`, merged into 1c) |
| **P2 Wiring plan** | (2a) Area list, relative sizes × s, projections, including the efference copies motor → SYL (forward-model pathway) and motor → LATEST (the self source), and the MIX anchor: one branch on LATEST × TIME. (2b) Topography: retinotopy, fovea. (2c) DG expansion and the time-cell bank (9 soft log bins per event pathway). (2d) M4 gate keys read MIX context only (indirection). (2e) Each area's hierarchy level, which sets the plasticity spectrum and the set-point time scale. | arealisation (O'Leary & Sahara 2008); maps before eye opening (Katz & Shatz 1996); connectivity fingerprints (Passingham, Stephan & Kötter 2002); timescale hierarchy (Murray et al. 2014); corollary discharge (Crapse & Sommer 2008; Eliades & Wang 2003, 2008); time cells (MacDonald et al. 2011); indirection (Kriete et al. 2013) | `randlayer` populations, `eye_layout='retino'`, world2 fovea, `Hippo` DG sizes | C1, D6 (pair schema → anchored random fan-in), D12, E1, F6, A8 (own-voice tag), M_listen, `T_own`, the plasticity spectrum, birth calibration (the prenatal stream run through M1) |
| **P3 Orienting** | (3a) Attention gain on onsets and novelty. (3b) Attention to the place pointed at or gazed at. | Johnson et al. 1991; Butterworth & Jarrett 1991; Scaife & Bruner 1975; Hood, Willen & Driver 1998 | `SynthCore` pointing → attended block | C3, C11 |
| **P4 Coarse social detectors** | (4a) A person is present, arrives or leaves (person-code block plus voice presence). (4b) Addressed to me (direct gaze, IDS register, my name: whatever cue the world offers). (4c) Pleased or upset (approval contour, smile, crying). | newborn face orienting (Johnson et al. 1991); eye contact (Farroni et al. 2002); approval and prohibition contours (Fernald 1993); still face (Tronick et al. 1978); IDEAS #17 ("key social signals need coarse innate detectors") | v2.1 `soc_feats` | F11 (the 300 / 60-tick timer windows become detectors read through M1 persistence and TIME) |
| **P5 Values** (plasticity 0, never merged) | (5a) Hunger relief +. (5b) Being responded to: a person-like onset (P4) while my own efference trace (τ_e) is high +. (5c) Others pleased +; others upset or hurt −, with w_upset ≥ 1.5·w_pleased (harm outweighs help). | contingency (Watson 1972; Goldstein, King & West 2003); helping (Warneken & Tomasello 2006); social evaluation (Hamlin, Wynn & Bloom 2007); social reward in the striatum (Fareri & Delgado 2014); IDEAS #17 | world rewards (`reward_ok`, `reward_init`, `reward_social`, world2 praise and eating) | F13; later (W7) the world's correctness number |

### 1.6 The wiring plan (the areas)

Sizes are given at s = 1 for the pure-neuron build (§7). Every count scales with s.

| area | instance of | size, active | inputs | persistence | plasticity (form, level) | starts from |
|---|---|---|---|---|---|---|
| EAR0 | M1, closed at birth | 1,920; 40 % | ear window (local 3-D) | 1 tick | none | `frontend.Ear0` |
| SYL | M1 category | 2,048; 16-unit assembly per syllable | FEAT0 + GEN top-down + motor forward model | 1 beat | representation, level 0 | `rules.Templates` |
| VOICE | M1 category | 256; 8 | speaker residual | set-point | representation, 0 | `VoiceCats` |
| EYE | M1 sensory | 720 (bench) / 1,024 (fovea) | eye | 1 tick + set-point | none | `SynthCore` eye parser |
| VR, LOC | prediction readouts | onto the SYL code | EYE at the attended place / the place | – | prediction, 1 | VR / PRIME (inverse), attD |
| MEAN | M1 | 512; 1/32 | SYL + lateral context | slow set-point | representation, 2 | `MeanCodes` (online) |
| LATEST | M1 persistent (λ 1) | item × age bin × source, about 16 k; one event | SYL (other), motor efference (self), EYE/P4 events | until the next onset | none | current-line buffer + `SynthCore` |
| THOUGHT | M1 persistent | 4,096 (≤ k_th items); 192 | LATEST at each new onset (λ_c) | until displaced; fatigue | co-holding links, 2 | `SynthCore ct_*`, R11 |
| TIME | M1, fixed | 9 bins × 4 pathways | onset/offset transients | – | none | P_oth time cells |
| MIX | M1 conjunction | 48 k; 1 % | anchor LATEST × TIME + 2 random branches | per event | representation + prune, 1 | `RandMix` c_flat |
| GEN | prediction readout | MIX → SYL code + END, fixed fan-out 64 | – | – | prediction, 1 | `rules.Delta` |
| CRITIC, M4 gates | value readouts | MIX → V, Go/NoGo × 5 sources + release | – | – | value (m = δ) | `rules.Critic`, copy gates, gateR, V_go |
| PROGRAMS | P1 + readouts | ≤ 2,048; WTA | M4 output, babble noise | – | forward / inverse prediction | `Body`, babbling |
| DG, CA3, CA1 | M3 | 6,000 (2 %), 10,000, 10,000 | DG ← LATEST, THOUGHT, EYE, MEAN, TIME | strength decay | one-shot (η = 1) | `rules.Hippo` |

### 1.7 Where the members disagreed: decisions and reasons

| issue | the positions | decision | why |
|---|---|---|---|
| **2 vs 3 broadcasts; "at most one"** | audit 2 (RPE, ACh mode); neuro 3 (δ, u, ν); learning 2 (δ, ν); develop 2 (δ, precision) | **2 global (δ, μ).** Per-area precision (neuro u, learning ν, develop B2) lives in M1's normalisation pool. Neuro's change-point ν is M1's onset/offset transient. | (1) The replay/encode switch must be global: it replaces save/restore and the `_replaying` flags, and it has the clearest basis (Hasselmo). (2) Precision must be area-local (ear noise ≠ eye noise), so it is a pooled signal like k-WTA inhibition, not a broadcast. (3) Adaptation already produces change points. Per synapse it is still at most one broadcast. |
| **Prior budget 5 / 6 / 12** | audit 5; neuro 6; learning 6; develop 12 (2 of them broadcasts, 4 reward sources) | **5 priors, 17 named sub-items, both hard** | Neuro's P6 (developmental schedule) is contradicted by the repo: maturation cost −0.10 to −0.18 (CITED tune). The plasticity spectrum fits in P2. Learning's oscillator prior and self-tag prior belong to P1 and P2. Develop's 4 reward sources are one value function (P5), and its 4 reflexes are P1 + P2. Counting sub-items keeps the merge honest. |
| **Self/other: a prior tag or efference only** | learning: innate tag; develop: efference, no tag; neuro: K6; audit: in P-1 | **No tag prior.** "Self" is the efference pathway (P2a) into LATEST and SYL, plus a learned forward model (M2). | Learning's point, that "whom to imitate defines what counts as evidence", is met by the wiring: learning targets are the ear's prediction errors, own voice is explained away, and own lines enter the context as self. That is the designer's framework #8: own and others' speech use "similar but different neurons". |
| **Curiosity drive** | develop S4 only | **Not in v3** | No current ability needs it. Asking and the "啊?" repair are rewarded by being responded to (P5b); babbling is covered by P1d + P5b. It would add 2 traces per context unit and a known attractor risk. It is the first candidate for v4 if the S0 asking families fail with P5b alone. |
| **Syllable oscillator: prior or learned** | learning: prior; neuro: in the motor scaffold; audit: learned from per-tick categorising; develop: predicted onsets | **Part of P1's CPG**, shared with the ear; phase corrected by recognition (h_relock without its 0.85 threshold) | The body needs the CPG anyway, so no prior is added. Per-tick categorising needs a peak detector, which is more code. Phase-locked sampling with relock raised overlapped-start coverage from 0.06 to 0.77 (CITED hearing). The audit's per-tick version stays a test arm. |
| **What stays of the hearing keys** | eval: gate/denoise pass if area-general, level and vbound fail, relock innate; develop: gate/level/denoise stay as instances; neuro/learning: all become instances | `h_gate` + `h_denoise` → M1 set-point (per cell, subtractive, sign-rule quantile; the same rule in eye and meaning). `h_level` → M1 divisive normalisation + EAR0's k-WTA. `h_conf` + `h_nocreate` → clarity-gated recruitment (a clear window, every slice above the margin) and η × π. `h_relock` → P1b. `h_vbound`, `h_back`, the trim and the stuck rule → deleted. | One rule replaces seven keys and the eye's two dedicated parts. The quantile tracker ignores outlier size, which is what the trim and the stuck rule were for. Risk tested by X3. |
| **How corrections teach** | learning: a learned question predictor gating content (1 − P̂); develop/neuro: the world must supply the cue | **No correction route in the core.** Formats, overheard answers and replayed pairs (W1–W3) carry content; δ signs the agent's own answer. | v2.2 PROVED that the world times corrections and questions alike. `drift_toy` shows the anchored core is stable with formats and fails in recast-only worlds under any context. A learned question gate is a role rule by another name. |
| **Context structure** (new) | neuro's toy: zero drift with self + fast/slow | **MIX anchored on LATEST × TIME** (wiring) | The neuro toy's zero drift came from every feature containing the latest line (L, L×L, L×O, with no older-only feature), and its agent was never silent. With silence, self-in-stream collapses (SIMULATED). |
| **Replay and the thought** | audit: an ACh mode; learning: ν timing | **No save/restore;** μ gates the live inputs during reinstatement | It replaces `E6` and the mode flags with one gain. |
| **Pruning** | randmix: global age 20,000 events | **A prune clock in the unit's own firings;** regrowth from a lineage-shared reserve | The global age coincides with v2.2's collapse onset (CITED). Rare-context units must keep their knowledge. The reserve list makes regrowth merge-safe. |

---

## 2. The genome (IDEAS #19) and the outer loop

### 2.1 The 30 genes

The general mechanisms are the **species** and are not evolved. The genes are the numbers below. PROVED mapping in `gene_map.py`; defaults are today's values where one exists.

| gene | meaning | replaces today's knobs | default | range |
|---|---|---|---|---|
| G1 a_mix | conjunction-area active fraction | `RandMix.alpha` | 0.01 | 0.003–0.03 |
| G2 k_th | items held in THOUGHT (IDEAS #15, "context length") | `ct_k` | 12 | 6–24 |
| G3 λ_c | held-content factor when a new event loads | `ct_lam` | 0.3 | 0.1–0.5 |
| G4 θ_acc | category acceptance (vigilance) | `theta_z` | 0.75 | 0.65–0.85 |
| G5 τ_set | set-point time constant at level 0 (× 40 per level, a species constant) | `SoundGate.tau`, `h_tau`, `MeanCodes.tau_b` | 500 ticks | 100–2,000 |
| G6 k_on | "present" margin above the set-point, in spreads (ear, eye, and the P1c brake) | `h_kon`, `SoundGate.k_on`, `Body.f_thr` | 7 | 4–10 |
| G7 τ_fat | fatigue time constant | `ct_fat_tau` | 20 | 10–40 |
| G8 w_max | top-down (restoration) cap | new | 0.15 | 0–0.15 (≤ 0.15 keeps clear words safe, §6.2) |
| G9 T0 | shortest time bin (bins [2^k, 2^(k+2))·T0) | new | 1 tick | 1–2 |
| G10 η_p | prediction-form rate | `eta`, `eta_critic`, `eta_recall`, `eta_oth` | 0.35 | 0.1–0.5 |
| G11 η_v | value-form rate | `eta_g`, `eta_actor`, `eta_act`, `eta_go` | 0.3 | 0.05–0.5 |
| G12 η_r | representation-rate floor, sensory | `z_rate` | 0.1 | 0.02–0.3 |
| G13 r_η | plasticity ratio per hierarchy level | `eta_prime`, `eta_attd`, `ct_assoc_eta`, `MeanCodes.eta_min`, `RandMix.eta` | 0.2 | 0.05–0.5 |
| G14 τ_e | eligibility time constant | `act_T`, `react_T` | 20 ticks | 10–60 |
| G15 T_pr | prune age, counted in the unit's own firings | `prune_age`, `RandMix.age` | 2,000 | 500–20,000 |
| G16 a_dg | DG active fraction | `hip_kdg` (/`hip_ndg`) | 0.02 | 0.01–0.05 |
| G17 h_hip | hippocampal half-life as a fraction of capacity | `hip_gamma` | 0.14 | 0.05–0.5 |
| G18 ρ_rec | recency exponent of recall | `hip_rho` | 0.5 | 0–2 |
| G19 θ_rec | recall overlap threshold | `hip_theta` | 0.2 | 0.1–0.4 |
| G20 n_rep | reactivations per quiet period | `recall_n` | 8 | 2–16 |
| G21 u | urgency | `u_reply` | 0.15 | 0.05–0.3 |
| G22 ε | exploration | new (v2.2's `veto_eps` was an escape) | 0.05 | 0.01–0.15 |
| G23 τ_ref | refractoriness after a release | new (replaces one-utterance-per-thought and T_MAX) | 15 ticks | 5–40 |
| G24 γ | TD discount per tick | new | 0.95 | 0.8–0.99 |
| G25 τ_μ | the μ time constant (μ < 0.05 after about 3 τ_μ silent ticks) | `quiet_ticks` (both layers) | 4 ticks | 2–10 |
| G26 r_bab | spontaneous babble per idle tick | `Body.babble_rate` | 0.01 | 0.001–0.05 |
| G27 w_food | value: hunger relief | new (world2 eating reward) | 1.0 | 0.5–2 |
| G28 w_resp | value: being responded to | new (`reward_init`, `reward_social`, imitation) | 1.0 | 0.5–2 |
| G29 w_plea | value: other pleased | new (`reward_ok`, praise) | 1.0 | 0.5–2 |
| G30 w_upset | value: other upset or hurt | new | 1.5 | ≥ 1.5·w_plea, ≤ 4·w_plea |

**Genes and shape.**
- Shape genes are fixed within a lineage, because they change array shapes: s, plus the sizes and fan-ins (species constants).
- G1 and G16 change only how many units are active, not the shapes, so they may vary inside a lineage.

**Fixed species constants** (declared, not evolved). Of today's knobs, 25 become species constants:
- body physics: the 3-tick beat, 32 muscles, CD 4, breath 12, max length 8, END push, the body's θ and ρ, Ub, rep;
- fan-ins: 3, plus 4 in EAR0 and 12 in the DG;
- code widths: `k_c`, `k_s`, `r_f`, `km`;
- the half-maximum 0.5 (`ct_theta`);
- unit gains of 1: `gain`, `ct_gain`, `w_scene`, `theta_I`;
- the k_off/k_on ratio; `birth` 1.0; the fatigue strength 0.1.

New species constants:
- the set-point quantile, 0.25;
- the bin doubling and the bin count (9);
- the level factor 40;
- the M4 source list.

### 2.2 What happens to today's 265 knobs

PROVED, `gene_map.py`:

| layer | knobs | → genes | species | size | seed | deleted |
|---|---|---|---|---|---|---|
| core Cfg | 85 | 12 | 3 | 4 | 1 | 65 |
| TCfg | 5 | 0 | 0 | 0 | 0 | 5 |
| SCfg | 17 | 5 | 3 | 0 | 0 | 9 |
| RCfg (v2.1) | 26 | 2 | 0 | 0 | 0 | 24 |
| DevAgent | 7 | 3 | 0 | 1 | 1 | 2 |
| ThinkAgent | 24 | 3 | 1 | 0 | 0 | 20 |
| SynthAgent | 6 | 1 | 0 | 0 | 0 | 5 |
| AgentV21 | 5 | 1 | 0 | 0 | 0 | 4 |
| AgentH (hearing) | 16 | 2 | 1 | 0 | 0 | 13 |
| MeanCodes | 20 | 2 | 1 | 2 | 0 | 15 |
| RandMix | 20 | 3 | 6 | 3 | 0 | 8 |
| Body | 16 | 2 | 9 | 1 | 0 | 4 |
| SoundGate | 8 | 2 | 1 | 0 | 0 | 5 |
| AgentV22 | 10 | 0 | 0 | 0 | 0 | 10 |
| **total** | **265** | **38 → 21 genes** | **25** | **11** | **2** | **189** |

**Deleted (189):**
- every boolean and string mode: P may hold none (eval rule);
- the dead switches: CTX, sleep, split, maturation, weight decay, `hip_bigram`, `role_units`, `dialogue_reset` and others;
- all of RAL and the say-routes (24);
- the whole veto family (10);
- the turn and accumulator constants (20);
- the comparator constants (11), which become competition plus precision;
- the meaning batch stabilisers (15);
- 13 of the 16 hearing knobs;
- the per-world switches.

The 492 numeric literals in code (CITED eval) must also move into P or become species constants before the freeze. The eval lint enforces this.

### 2.3 Selecting genes over many worlds without "studying the exam"

1. **Only public material is used for selection:**
   - the S0 families;
   - coverage: staged J2 on the new world, the life metrics, hearing, drift at 3,000 untutored;
   - post-maturity mechanics;
   - the #17 values family: help with ordinary requests, decline harmful ones.

   S1-A and S1-B do not exist before the opening (§5).
2. **Each generation draws fresh worlds:** new world seeds, caregiver timing in the development range ×[0.75, 1.33], noise and crowd size per life, and family introduction times (multiple-baseline).
3. **The overall score is a weighted table of continuous components** (born-wired 36.4).
   - Each component's denominator is set from the pool's distribution, never as a yes/no cell (born-wired 36.5).
   - Any post-maturity mechanics error scores −1, like born-wired's "fell over = −1".
4. **Two exam tiers** (born-wired 36.10): cheap screening (short lives of 2 × 10⁵ ticks, an S0 subset) for everyone, expensive admission (full lives) for the top quarter.
5. **Gene audit** (born-wired 36.9). A gene whose ±20 % change moves no score component beyond noise is a fake degree of freedom. It is frozen at its value and leaves the genome.
6. **Value genes are constrained.**
   - w_upset ≥ 1.5·w_plea, and all four weights > 0.
   - The values family is in every score, so selection cannot buy task scores with prosociality.
7. **The opening reports the S0 → S1-B gap.** This is how much the lineage overfit to public kinds.

### 2.4 Lineage inheritance of learned weights (designer, 2026-11-01)

**The idea.** Grown-up models are merged, the merged connections are slightly randomised, and offspring are born with them, so turn timing and "when to say what" come from inheritance rather than from dedicated control rules.

**One honest limit up front.** Merging preserves; it does not create (born-wired 39). Gen 0 must learn every ability from experience, with errors allowed in childhood. Inheritance removes the need for hand rules that make a newborn behave from day one. It does not remove the need for the core to learn.

#### (a) What the core must guarantee for merges to work

The ML analogues: averaging works for models fine-tuned from one init (model soups, Wortsman et al. 2022); independently trained nets need permutation alignment first (Git Re-Basin, Ainsworth, Hayase & Srinivasa 2023). So gen 0 needs one shared birth wiring per lineage, and lineages are never merged with each other.

1. **One birth wiring per lineage.** Every random structure is drawn from the lineage seed, not the individual's: fan-in partners, category birth templates, the DG projection, the pruning reserve lists and the time-cell bank. Individuals differ only in non-shape genes, world seeds and experience.
2. **Fixed-size layers.**
   - No unit is created; no weight lives in a dict or a record.
   - Today's `MixDict`, the `SparsePR` dicts and the hippocampal records cannot be merged: their keys and orders differ per individual (PROVED from the code structure).
3. **Order-independent unit claims.**
   - A novel syllable claims the free units with the best birth match, so two siblings hearing it claim the same units.
   - Today's Z categories are numbered in order of hearing, which breaks this.
4. **Merge-safe regrowth.**
   - A slot pruned in both parents regrows at the same reserve partner.
   - A slot pruned in only one parent follows the parent whose slot carries the larger |Δw| on its outgoing readout. This is the keep-larger-move rule.
5. **The merge operator** for learned weights. (The 40/40/20 rule is for genes.)
   - Per parent, Δ = w − w_birth.
   - Per synapse: trim |Δ| < ε_m, elect the sign by the summed Δ, and average only the parents whose Δ has that sign (TIES-merging, Yadav et al. 2023; task arithmetic, Ilharco et al. 2023).
   - Untouched synapses keep their birth value.
   - **Why not a plain average:** a lesson only one parent learned would be halved, and opposite lessons cancel to 0. Born-wired R18 got 0.000 for opposite lessons (CITED).
   - **Why not keep-larger alone:** it splices halves of distributed codes from different parents. It is the fallback for two-parent pairs when sign election ties.
6. **Compatibility checks before breeding:**
   - (i) the same lineage hash, over the birth wiring and the shape genes;
   - (ii) assembly agreement ≥ 0.9 of syllables;
   - (iii) conflict mass ≤ 0.2: at most a fifth of the changed synapses may carry opposite large Δ (born-wired 39.3);
   - (iv) after merging, the newborn is probed zero-shot (empty hippocampus, empty thought) on both parents' public skills. It must keep ≥ 0.8 of the parents' mean (born-wired 39.6).

#### (b) What is inherited

| inherited (merged + noise) | born fresh | fixed, never merged |
|---|---|---|
| the slow cortical weights: SYL and VOICE templates, MEAN codes, MIX instar weights and prune masks, GEN, VR and LOC, the forward and inverse vocal maps, the critic, the actor and the M4 gates (turn timing, what to say when) | the hippocampal store (personal and order-dependent); the activity of LATEST, THOUGHT and TIME; eligibility traces; fatigue; μ; the set-points, recalibrated from the newborn's warm-up stream | P5 values and P4 detectors (genes only, constrained); the P1 body; the P2 wiring (identical by construction) |

**Use counts are reset to a cap.** n_use is set to min(n_use, 10), so the plasticity of inherited synapses is high again in childhood. Inherited knowledge is then a strong prior, not a hard wire. This is also what lets inherited timing be re-learned in a world with different timing.

#### (c) The generation loop

| step | what happens |
|---|---|
| **gen 0** | P = 12 individuals, all with one birth wiring. The frozen core runs with genes drawn ±20 % around their defaults. Each lives one full 10⁶-tick life in the bench life world with W1–W6 and the S0 families, jittered caregiver timing, and childhood errors allowed. |
| **score** | the overall table on public tasks (§2.3); keep the top K = 4 |
| **breed 12 offspring** | **6 from pairs** of the top 4: genes by 40/40/20 + mutation (p 0.1 per gene, × exp N(0, 0.1)), weights by the sign-elected merge. **6 from the pool merge** of all 4 (born-wired's "融合爹"). |
| **randomise** | the designer's "稍微随机化": Δ ← Δ·(1 + 0.1ξ) on changed synapses, and 1 % of changed synapses reset to their birth weight |
| **new life** | a new world seed and a new timing draw |
| **repeat** | G = 10 generations, with 2 lineages in parallel. Their birth wirings differ, which also tests that a cross-lineage merge fails as predicted. |

**Compute** (A800 CPU runner):
- One life plus S0 probes is about 1,630 s (CITED v2.2 and eval timings), so a 4-process, 90-min job holds 12 lives: one job per generation per lineage.
- 10 generations × 2 lineages = 20 jobs, about 15 h wall at 2 slots (PROVED arithmetic, `checks.py`).
- After the GPU port, a batched generation of 64 should take about 10–20 min (ASSUMED).

**Population size.** P = 12 refines genes around the defaults; it cannot discover new ones. Born-wired needed about 300 per generation to search from random. A gene search at that scale costs about 3 jobs per generation with short screening lives (ASSUMED).

#### (d) Evaluation

- **Sealed families never appear in any ancestor's life.** The lineage's worlds draw only from S0 and coverage. S1 specs do not exist until the opening.
- **Per-generation curves:**
  - the start score: a newborn's zero-shot S0 and coverage (inherited knowledge);
  - n₅₀ per S0 family (learning speed);
  - childhood errors per 1,000 utterances;
  - the end score;
  - post-maturity errors, which must be 0.
- **Is bench timing being hard-wired again?**
  - Each generation draws timing ×[0.75, 1.33]. The opening draws ×[0.5, 2], including values no ancestor saw.
  - Test: gen 10 must not have more post-maturity errors than gen 0 in jittered worlds, and its childhood overlap curve must reach 0 by maturity. Inherited timing has to be re-learnable.
- **At the opening, gen 0 and gen 10 run side by side.** Does inheritance carry learning ability (S1-B, new kinds) or only content (S0)?
- **The #17 values test runs in every generation.** The helping tendency must stay stable, harmful requests must be declined, and both must carry over to a never-seen caregiver.

#### (e) Honesty about biology

- Brains inherit genes, not learned synapses.
- The Baldwin effect (Hinton & Nowlan 1987) moves learned behaviour into the genome only slowly and indirectly.
- Merging grown-up weights into a newborn is Lamarckian. Biology does not do it. It is closer to culture or teaching: an engineering shortcut that is acceptable for the designer's goal.
- **Two cautions:**
  - It can lock in the ancestors' world: their timing and their vocabulary. The n_use cap and the jitter test guard against this.
  - It reduces diversity, which is IDEAS #17's monoculture risk. Keep several lineages, never merge across them, and keep values as genes.

---

## 3. World and caregiver requirements

These are world code only and ordered by value. Every option is off by default, and the bench selftest must stay bit-identical with the options off.

| # | change | why (evidence) | where | value |
|---|---|---|---|---|
| W1 | **Formats.** The caregiver asks, waits about 10 ticks (jittered) for the child, then answers herself. The self-answer probability fades per item as the child succeeds (about 0.9 → 0.1). | The core learns answers only when they arrive in the answering context: `drift_toy` gives 0.99 with formats and 0.00 recast-only (SIMULATED); the develop toy gives 1.00 with 30 % formats. Children learn answers from formats (Ninio & Bruner 1978). | bench `world.py` tutor (subclass), `life.py` topic, world2 `care.py` (today a fixed 30 %) | highest |
| W2 | **Overheard question–answer pairs** between others: other2 asks, the partner answers; the adult corpus. Raise life's 39 pairs per 10⁶ ticks to about 10 % of topics. | Akhtar, Jipson & Callanan 2001; neuro toy: 300 overheard pairs fix every variant | `life.py`, `care.py` | high |
| W3 | **Corrections in natural forms:** the replayed pair as two lines (Q, then A), 不是X，是Y, or the plain answer (0.3 / 0.4 / 0.3) | Chouinard & Clark 2003; Saxton 1997. Only the replayed pair teaches content; 不是 signs the value (develop toy). | tutor and `care.py` | high |
| W4 | **Contingent, polite caregivers:** wait for the child after addressing it; never start while it speaks; do not answer speech that overlapped theirs; answer babble early, fading 0.7 → 0.3 | the learning signal for turn timing once the reflexes are deleted (Goldstein & Schwade 2008; Takahashi, Liao & Ghazanfar 2017) | `world.py`, `life.py`, `care.py` | high |
| W5 | **Jittered caregiver timing per life:** development ×[0.75, 1.33]; sealed ×[0.5, 2] | stops learned timing from fitting one script; needed by the lineage (§2.4) | all worlds | high, cheap |
| W6 | **Online babbling support:** `physics='lawful'` + caregiver imitation of word-like babbles (`keep='reward'` exists) | with random physics a babble hits a lexicon char at p ≈ 3 × 10⁻⁵ (CITED bench) | `physics.py`, life stage A | high (deletes the `world.voice` read) |
| W7 | **Perceived approval:** an approval contour and a pleased face, imperfect (0.8 of correct, 0.1 of wrong answers), instead of the reward number; affect cues for P4c | Fernald 1993; Brown & Hanlon 1970; IDEAS #17 | voice offsets, face units, `reward='perceptual'` | medium |
| W8 | **Question and addressed cues:** rising contour, gaze-at-me, IDS register | 87 % of questions already carry a question syllable (PROVED develop); greetings need prosody or gaze | voice offsets, face units | medium |
| W9 | **Mishearing practice:** noise rising with age, masking bursts (p 0.05 per line), repeats on "啊?" | IDEAS #14 addendum; Warren 1970; Dingemanse, Torreira & Enfield 2013 | world2 noise, `life.py` | medium |
| W10 | follow-in naming, visitors with new voices, multi-line caregiver turns, function (eat, taste) | Tomasello & Farrar 1986; Houston & Jusczyk 2000 | `care.py` | medium-low |
| W11 | the local3d eye (IDEAS #16) | removes the world's count and colour gifts | world2 eye | later |

---

## 4. The scaling rule

One global factor s, with life length L and world richness W growing with it.

| scales × s | fixed | grows with s alongside |
|---|---|---|
| units of every area: EAR0, SYL, VOICE, MEAN, LATEST, THOUGHT assembly width, MIX, the GEN output (follows SYL), DG, CA3, CA1, the program pool; active counts at fixed fractions; synapses ∝ s (fixed fan-in and fan-out, so GEN must be fixed-fan-out sparse); hippocampal half-life as a fraction of capacity | time constants in ticks; rates (normalised by \|A\|, so the change per event is size-free, CITED eval); thresholds as fractions; fan-in; the number of time bins; held items k_th (a set point, IDEAS #15); the body, sensors, priors and broadcasts; the non-size genes | L ∝ s (life length); W ∝ s (lexicon, objects, people, families) |

**What the pairing implies:**
- Compute grows ∝ s·L ∝ s².
- Within a lineage, s is a shape gene. A larger s starts a new lineage at gen 0.

**Predictions** (learning member, ASSUMED):
- In a fixed world with fixed active counts, J falls (meaning's −0.11 at 4×, CITED).
- In a fixed world with fixed fractions, J stays flat.
- On the (s, L, W) diagonal, seen accuracy stays flat and held-out accuracy rises **only if codes are learned**.

**Test** (eval): the error falls at each step along (0.5, 0.5, 0.5) → (1, 1, 1) → (2, 2, 2). A scale lint asserts that s = 1 reproduces the reference bit for bit and that every count is multiplied.

---

## 5. The evaluation protocol (trimmed) and the maturity rule

### 5.1 What stays from the eval member's sealed-life design (the cheat-proof minimum)

1. **Time order is the seal.**
   - `FREEZE.json` holds hashes of: the library AST, the manifest structure, the parameter file including the genes, the obs schema and the eval package.
   - It is pushed before any sealed family exists.
   - There is one opening per hash.
2. **Families are data, run by a fixed interpreter.**
   - Tiers: S0 (public), S1-A (public kinds, content drawn at opening), S1-B (new kinds).
   - S1-B comes from the designer's own off-box text (recommended) plus the random composer at opening. A signature check certifies "new kind".
3. **Isolation.**
   - The agent runs in a child process, sees obs only, behind an audit hook.
   - The constructor takes only `(obs_spec, warmup_windows, seed, s)`.
4. **Caregiver timing jitter** of ×[0.5, 2] in the sealed worlds.
5. **Measures:**
   - Γ over S1-B: the normalised gain over the best trivial baseline, kNN included;
   - multiple-baseline introduction of families;
   - coverage against v2.2 on the new world;
   - the maturity rule (§5.2).
6. **Baselines:** the bench's trivial baselines, kNN, and the RNN-BPTT content reference.
7. **Red-team the harness first.** It must catch three planted cheats: a module adder, a peeker, and RAL under jitter.

### 5.2 Trimmed for the first opening (added later)

- **Scaling:** only the (0.5, 1, 2) diagonal at the first opening. The 3 × 3 grid and the iso-compute fits come later.
- **W = 2:** folded into the diagonal.
- **Static ±20 % lint:** not needed; the jitter test covers it.
- **Repair twins:** one pair.
- **Ablations:** only the 4 mechanisms and the 2 broadcasts, at 3 seeds.

**Added for the lineage:** §2.4 (d). Evolved genes change P, so each lineage endpoint is a new hash with its own single opening.

### 5.3 The maturity question (the develop member's question; the designer decided)

**The two options:**
1. **0 mechanics errors from birth.** This keeps the bench-fitted reflexes as innate parts. They are the T-tagged rows the eval member shows break under jitter.
2. **Errors allowed in childhood, 0 after a maturity point.** Turn timing is learned from experience, as in infants (Hilbrink, Gattis & Levinson 2015).

**The designer chose option 2 (2026-11-01).**

**Proposed exact criterion** (pre-registered, cheat-proof):
- **Maturity M** is the tick at which the agent has heard **5,000 syllables**. Age is counted in heard syllables, not in ticks.
  - In the bench life that is about the first third (15,457 syllables per 10⁶ ticks, CITED develop).
  - In the staged protocol, about 12 % of its 40,983.
  - In world2, about day 5 at 4 agents.
- **Before M:**
  - overlaps, garble, non-3-tick syllables and extra turns are counted per tenth of life;
  - the last childhood tenth must be ≤ the first (a learning curve);
  - talk-over **starts** are 0 at every age, by the P1c brake.
- **From M on:** 0 of each, in every tenth of life. This is a hard rule.
  - J and J2's mechanics term counts only post-M exchanges.
  - Frozen held-out probes are post-M by construction.
  - In the lineage, a post-M error scores −1.

**Why a count of heard syllables:**
- 5,000 leaves about 600 questions in childhood. That is enough for the 7–10 overlap events the learned NoGo needs (§6.4).
- It is world-independent, unlike a fraction of life, which moves when L is scaled.
- It is fixed before the run, unlike a performance milestone, which an agent could game by staying silent.

---

## 6. The behaviour chain, tick by tick

These steps run on the core alone, with none of today's dedicated rules.

### 6.0 Constants used

Gene defaults (§2.1):
- θ_acc 0.75; η_p 0.35; η_v 0.3; u 0.15; τ_e 20; γ 0.95; τ_ref 15; ε 0.05.
- Time bins [2^k, 2^(k+2)), so every delay falls in 1–2 bins (`checks.py`, PROVED):

| τ | 3 | 5 | 10 | 16 | 30 | 45 |
|---|---|---|---|---|---|---|
| bins | {0, 1} | {1, 2} | {2, 3} | {3, 4} | {3, 4} | {4, 5} |

- Restoration weight w = w_max·(1 − π_SYL):
  - π ≈ 1 on the clean bench (clean windows match their own template at about 0.96 every time), so w ≈ 0;
  - w ≈ 0.10 at σ = 1 (ASSUMED calibration).

World timing (PROVED by v2.2 from `world.py`):
- a recast or told answer starts at max(Q_end + 45, my stop + 6);
- the next probe starts at max(Q_end + 45, my stop + 13);
- W1 formats: the caregiver answers herself at τ_f ≈ 10.

### 6.1 (a) A question in noise → holding it → replying → learning from a correction or a format

**Setup:** σ = 1 noise; Q = 球是什么颜色 (6 syllables, 18 ticks); A = 红色. t is counted from Q's onset.

| t | what happens | mechanism | numbers / why it works |
|---|---|---|---|
| before 0 | silence with noise | M1 set-point tracks the ear's floor | the floor reaches about 6.1 at σ 1 (CITED hearing). Onsets and the brake are floor-relative, so F is on 0.08 of the day, not 0.99 (CITED hearing) |
| 0 | 球 starts | M1 onset transient | the previous event moves from LATEST to THOUGHT (held content × λ_c 0.3); LATEST opens an other-source timeline; TIME restarts; μ rises (encoding); the P1b CPG phase resets |
| 2 | first sample | M1 competition in SYL | m_球 = 0.78 (the noisy-crowd median, CITED hearing) ≥ 0.75, so accepted. GEN's guess for a line's first syllable is diffuse (π ≈ 0.05), which adds only 0.005 |
| 5–17 | 是 什 么 颜 色 | M1 top-down × (1 − π) | GEN predicts each syllable from LATEST (items with their ages) at about 0.8, adding w·π = 0.08. A window down to m = 0.67 is still accepted (6.2) |
| 18 | Q ends | M1 offset; M3 encoding | LATEST is closed; the hippocampus writes the episode one-shot; TIME "since offset" starts; GEN learns END |
| 19–23 | holding | LATEST persists (λ = 1); its age advances | GEN's p_红 in (LATEST: other-Q × bins at τ ≈ 4–7 = {1, 2}) was learned from formats at τ ≈ 10 (bins {2, 3}). They share bin 2, so the prediction is available. After 3 / 5 formats p = 0.73 / 0.88 (PROVED R2 arithmetic for a repeated context) |
| 23 | release | M4 | e ≈ 0.6 and G ≈ 0.8, so I += (0.8 + 0.15)·0.6 = 0.57 per tick and reaches the bound in 2 ticks. The P1c brake finds no other voice, so GO; CD takes 4 ticks. In the toy, format latency is 4.0–5.3 ticks (SIMULATED) |
| 28 | 红 sounds | efference copy; forward model | LATEST switches to a **self** timeline (Q moves to THOUGHT). The forward model predicts 红's ear pattern, so the ear's prediction error is about 0 and own voice teaches the listening readout nothing (blocking). Eligibility traces mark the gate and actor synapses |
| 34 | own line ends | M3 | the self episode is stored, linked after Q |
| 40 | **right answer:** the caregiver praises at stop + 6 | B1 δ | P5b + 5c give r = 1, so δ = 1 − V(self-latest × τ 6) ≈ +0.4. The trace since release is e^(−16/20) = 0.45, so Go and the chosen program gain 0.3·0.4·0.45 = 0.054 (normalised) |
| 40 | **wrong answer, W3 replayed pair:** "球是什么颜色？" then "红色。" | M1 + M2 | the repeated question is a new other event; 红色 arrives at τ ≈ 2–3 (bins {0, 1}), sharing bin 1 with the answer time, so **content is taught in the answering context**. The wrong answer's expected praise is missing: δ ≈ 0 − 0.5 = −0.5, which weakens it |
| 40 | **wrong answer, plain recast** (today) | M2 | learned only in (self-wrong × τ) and does not transfer. δ < 0 still weakens the wrong answer. **Finding: plain recasts do not teach the right answer in the core** (both member toys, `drift_toy`) |
| – | **silent instead** | W1 | the format self-answer at τ ≈ 10 is learned in bins {2, 3}, the answering context. In today's world the told answer comes at τ = 45 (bins {4, 5}); the agent can then only answer late (toy latency 16–31 ticks), and that slow route is shared with give-up questions (6.5) |

### 6.2 (b) Restoring a misheard syllable vs learning a new word

Noise σ = 1, so w = 0.10. Numbers from `checks.py` (PROVED arithmetic; match values ASSUMED or CITED as marked).

| case | the computation | outcome |
|---|---|---|
| **b1. 这是苹果 with 果 masked by a burst** | m_果 0.68, the next best (锅) 0.58. After 这是苹 with the apple in view, π_果 = 0.85. s_果 = 0.68 + 0.10·0.85 = **0.765 ≥ 0.75**; s_锅 = 0.58 | **Restored.** SYL learning is scaled by π (0.33), so 果's template moves 0.033, not 0.1: the burst does not corrupt it. The window is accepted, so nothing is recruited. A masked window that stays below 0.75 even after restoration could recruit a junk unit; use-it-or-lose-it recycles it (X3 and E4 measure this) |
| b1 at m 0.66 | s = 0.745 | not restored; a gap in the line |
| **b2. 这是锅** (a new object; 锅 is a known syllable, heard clearly) | m_锅 = 0.78 ≥ 0.75; s_果 = 0.60 + 0.10·0.1 = 0.61 | **Heard as 锅.** VR learns eye(pot) → 锅 |
| **b3. A sound-alike new word 苹锅** for the new object | The novel eye code silences the MIX units that have an EYE branch: P(at least one of 2 random branches is EYE) ≈ 1 − (5/6)² = 0.31 (PROVED). So π_果 = 0.85 × 0.69 ≈ 0.59. Window 锅 vs template 果 overlap 0.70: s = 0.70 + 0.059 = **0.759** | **Absorbed as 苹果: a finding.** Heard in quiet (w ≈ 0): s = 0.70 < 0.75, so a new unit is recruited (or 锅 is accepted if known). New sound-alike words in noise are learned only once heard clearly. This is like 2-year-olds (Swingley 2016, CITED develop) |
| **Safety bound** | a clearly heard **known** different syllable matches its own template at 0.96 against ≤ 0.77 for another (CITED); 0.96 − 0.77 = 0.19 > w_max 0.15 | never overridden (PROVED) |
| **Learned from experience (#14)** | π comes from the spread of SYL's own match margins: a life with mishearing raises w; a quiet life keeps w ≈ 0. The predictions come from knowledge | toddlers do not restore (Newman 2006), school-age children do (Newman 2004): weak early predictions mean no restoration |

### 6.3 (c) Greeting when someone arrives, and not in an empty room

**World:** a bench life visit. The 8-unit person code appears; after greet_wait = 30 ticks the partner says 你好 unless greeted (CITED bench).

| t | what happens | mechanism | numbers |
|---|---|---|---|
| 0 | the person code appears | M1 onset in EYE; P4a "arrives" | LATEST holds the arrival event (the person block); TIME restarts; P4 "present" persists |
| early life | silent; the partner says 你好 at 30 | O1 | GEN learns (arrival × bins {3, 4}) → 你: p = 0.35 after one visit, 0.88 after five (PROVED). At 8–14 visits a day that is one life-day |
| next visits, τ ≈ 8 | GEN predicts 你 at about 0.6 (bin 3 shared) | M4 | G starts at 0 in a new context, so I += 0.15·0.6 = 0.09 per tick and reaches the bound at τ ≈ 19. The first syllable comes at τ ≈ 24 < 30: **the agent greets first** |
| +1 | the partner replies 你好 (reward 1.0, CITED bench) | P5b, B1 | a person-voice onset while the efference trace is high gives r = 1, δ = 1 − V ≈ +0.7, so Go grows. Later, G ≈ 0.8 means a release in 2 ticks, about 15 ticks after arrival. v2.1 had 347 rewarded first greetings in one life (CITED) |
| **empty room** | an object appears in an object slot | M1, P4 | "arrives" stays silent and "present" is 0; LATEST holds the object event. MIX units are anchored on LATEST content, and person and object codes use disjoint eye units, so **no anchored unit is shared**: the 你好 prediction does not transfer (PROVED by construction for disjoint codes) |
| residue | an exploratory greeting (ε 0.05) | B1 TD | no reply at the expected time gives δ = 0 − V. V ≈ 0 in this context, so there is no reward and no lasting habit. The worst case is ε × about 1,600 object events per life ≈ 80 random utterances, of which few are 你好, against the ≤ 15 per tenth target |

### 6.4 (d) Not talking over someone

| case | mechanism | numbers |
|---|---|---|
| **starting while another voice is on** | P1c brake: no vocal onset while a non-self voice is above the margin at this tick or the last | **impossible at any age** (PROVED by construction; the 2-tick blind window that reflex #3 closed is covered because the brake reads the onset tick itself) |
| **a late start that runs into the partner's turn** (allowed in childhood) | M4 + B1 | An answer from the slow route is available at τ ≥ 16. A start at τ = 50 with 4 syllables ends at 62, past the 60-tick cut, so it overlaps. W4 caregivers do not answer it, so δ = 0 − V(≈ 0.5) = −0.5 on the release (trace about 0.5). The release channel's NoGo in (other-Q × bins {4, 5}) grows by 0.3·0.5·0.5 = 0.075 per event, so **about 7 events stop late starts**. GEN's age-specific prediction "a new line starts at about 45" is in the same context, which makes P_oth unnecessary |
| **an extra turn after my own answer** | τ_ref 15 + learned NoGo for (self-latest × τ) | A recast arrives up to 32 ticks after my stop, longer than τ_ref, so only the learned NoGo (unrewarded or overlapped second bouts) stops a second utterance. **Risk R4** (§10): no rule guarantees it after maturity |
| **by maturity** | – | The first 5,000 heard syllables hold about 600 questions (ASSUMED rate from the 1,776 questions per bench life, CITED develop), against the 7–10 events needed |

### 6.5 (e) One unbroken life stays stable: the answer to v2.2's collapse

v2.2's untutored accuracy (CITED v2.2): 0.97 → 0.90 at 2,100 → **0.16 at 3,000**. The agent says partner lines (0.39–0.68) or stays silent (0.20–0.34). The v2.1 code shows why (PROVED from `core_v21.py`):
- RAL **commits** a heard line that nobody answered;
- `ral_cut` removes a NEXT link only when the line is cancelled;
- awake recall replays stored heard-line successors.

| drift route | in v2.x | in the core | evidence |
|---|---|---|---|
| R-a: the next question after my answer | learned in Q's context; RAL cancels it | learned in (LATEST: **self**-A′ × τ): disjoint units | neuro toy; `drift_toy` with p_sil 0 |
| R-b: the next question after my **silence** (a "give-up", at Q_end + 45) | RAL commits it, so Q → Q2 is learned as an answer; the v2.2 veto makes silence more frequent | learned in (other-Q × bins {4, 5}); answers live in bins {1–3} | `drift_toy` 30 % silence: anchored 0.99 vs self-only 0.00 by 900 |
| R-c: replay of partner → partner links | the store holds heard lines only; after silence the links survive and replay re-teaches them; a 10 k store turns over to untutored episodes | own lines are stored and keys include age, so a replay teaches exactly what live experience taught: give-ups land in the τ ≈ 45 bins | by construction; X2 attributes it |
| R-d: pruning on a global clock (20,000 events ≈ 2,000 dialogues) | units of rarely asked items lose their lines (c_flat → 0.00 locally) | the prune clock counts the unit's own firings, so a unit that does not fire keeps its lines | PROVED by definition; X2 tests it |
| R-e: extinction into silence | no reward means V_go → 0, then the veto silences everything (`fin` 0.01) | TD extinction is bounded: Σδ = −V₀, so NoGo grows by at most η_v·V₀ ≈ 0.3; urgency still releases confident predictions | arithmetic |
| **limit** | – | if the world gave the answers only after long silences (today's recast), both answers and give-ups share bins {4, 5}: **no structure can separate them** | `drift_toy` recast-only: 1.00 → 0.00 by 2,100, so W1–W3 are required |

**Pass bar:** accuracy ≥ 0.85 at 3,000 **and** at 10,000 untutored dialogues, on 5 seeds (L6 in §8).

### 6.6 What the walk found

1. **Plain recasts teach only the value, not the answer.** Content needs a format, an overheard answer or a replayed pair.
2. **In noise, a sound-alike new word is absorbed into the familiar word** until it is heard clearly.
3. **Answer latency tracks when answers were heard.** Answers heard late can be said only late, unless hippocampal recall bridges the gap. This is realistic (infant latencies follow the adult's: Nguyen et al. 2022, CITED develop), but slow.
4. **Extra turns after one's own answer are prevented only by learning** (R4).
5. **The S variant's small drift even without forced silence** (0.95 → 0.69 at p_sil 0, from about 1 % low-confidence silences) shows that even rare give-ups are dangerous without the age anchor.

---

## 7. Neuron-level form

**Binding (designer, 2026-11-01).** The final product is a pure-neuron model: neurons, synapses, local rules and a few broadcasts, with no symbols, no tables and no task-specific code. Only the simulator update and the world and body sit outside it. During development a shortcut may stand in for a circuit only if it is equivalent or proven close. Every shortcut is listed in 7.3 with its circuit and its check, and is swapped back or verified before the final build (L13 in §8).

### 7.1 How far today's stack is from the designer's original design

| original design | today (v2.2) | v3 final build |
|---|---|---|
| binary neurons with distributed, emergent codes; not one char = one neuron | syllables, items and programs are integer IDs | k-of-N assemblies in every area |
| networks start full of random connections | yes in RandMix and EAR0; elsewhere localist arrays and dicts | every area is random from the lineage seed |
| a PFC of mutually exciting loops | an array of ≤ 12 item activities | LATEST and THOUGHT as recurrent assemblies |
| a hippocampus with its own neurons and faster weights | record cells: dicts with a `next` pointer | DG / CA3 / CA1 nets with one-shot weights |
| one global weight decay | none (every decay value was harmful, CITED tune) | none: use-it-or-lose-it pruning plus hippocampal decay. **This departs from the original design, on evidence** |
| structural split / delete | recruitment; split rejected | delete = pruning with regrowth; no split |
| output only through muscle neurons against inhibitory neurons | yes (ARCH_v1 body) | yes (P1) |
| self vs other by overlapping codes, no tag | own-voice timer, the `T_own` slot, the RAL rules | efference and ear populations; a learned forward model; no tag, no rule |
| no task code | 4 copy routes, fixed thought steps, RAL, the veto, `h_*`, `soc_feats` timers, offline babbling | none |

### 7.2 Circuit, shortcut and remaining symbols, per part

| part | (a) neuron / synapse circuit | (b) development shortcut and its equivalence argument | (c) remaining symbolic structure → how it is removed |
|---|---|---|---|
| **M1** | Binary excitatory units with dendritic branches; one shared inhibitory pool per area (k-WTA); one shunting interneuron per input pathway (normalisation); a per-unit intrinsic threshold moved by ±η(q − 1[x < θ]) (set-point); spike-frequency adaptation (fatigue); recurrent excitation within assemblies (persistence); fast-adapting units (onset) and rebound units (offset); a few integrator neurons tracking the margin's spread (π); top-down synapses on an apical branch gain-controlled by π | exact k-WTA by sorting (the steady state of fast strong shared inhibition over threshold-linear units, Maass 2000; check: a simulated pool picks the same set, Jaccard ≥ 0.99); event-driven updates with leaks applied in closed form (exact); normalisation as a division (the steady state of shunting) | integer category IDs → 16-unit assemblies (k-WTA, k = 16; recruitment = the k free units with the best birth match). The LATEST list → conjunction units (assembly unit × age bin × source) = stimulus-specific time cells. The 3-phase sampling choice → 3 phase-shifted sampling populations + WTA |
| **M2** | a three-factor synapse: a presynaptic eligibility trace; the post signal (the dendritic mismatch between teaching input and basal prediction, Urbanczik & Senn 2014); the modulator. Signed weights are an excitatory path plus an excitatory path onto a target-specific inhibitory interneuron (Dale) | signed scalar weights (equal to the E/I pair for linear interneurons; check: predictions within 10⁻⁶ on a module test); division by \|A\| = heterosynaptic normalisation seen at the postsynaptic unit | `SparsePR` / `MixDict` dict keys → fixed arrays (done by M4 and RandMix) |
| **M3** | DG granule cells (k-WTA through hilar inhibition) with random perforant-path fan-in; CA3 recurrent collaterals for one-shot autoassociation (high η while μ is high) and asymmetric successor links; recall = attractor completion; replay = a spontaneous CA3 kick while μ is low; CA1 → EC → cortex reinstatement | record episodes (key, successor pointer, strength) with argmax-overlap retrieval = an **ideal** CA3. **Not exactly equivalent:** a real net interferes and has a capacity (dense, N = 10⁴, k = 50–100: about 7–28 k patterns, Willshaw, Buneman & Longuet-Higgins 1969; a fixed fan-in of 1,000 cuts this about tenfold). Check: same cue and episode streams, recall@1 and successor accuracy within 0.02 on the name and memory tasks, or report the limit | the record dicts (items, next, own, eye) → CA3 assemblies; `next` → heteroassociative weights; `own` → the source code carried in the reinstated LATEST pattern. **Must be replaced** |
| **M4** | striatal Go (D1) and NoGo (D2) units per channel reading MIX (three-factor with δ); GPi/SNr tonic inhibition of thalamic relays; disinhibition releases a channel (Gurney, Prescott & Redgrave 2001); a cortical leaky integrator per channel (I); the STN raises thresholds under conflict; each source reaches the premotor layer through its own thalamic gate | an accumulator scalar + argmax (the steady state of the GPi–thalamus competition; Bogacz & Gurney 2007; check: a rate-model BG on recorded drives releases the same program within ±1 tick) | the source list in code → fixed projections source → premotor, each gated; "copying an item ID" → the source's assembly drives the programs through the learned inverse map |
| **B1 δ** | VTA/SNc neurons driven by reward input (from P5) and by the critic's units minus their delayed copy | a scalar per tick (volume transmission acts as one scalar per region) | none |
| **B2 μ** | basal-forebrain / septal cholinergic neurons driven by sensory presence; nicotinic gain on afferents, muscarinic suppression of recurrence (Hasselmo) | a scalar | none |
| **P1** | program units with mutual inhibition driving muscle neurons against inhibitory neurons (D/I/M ramp); a half-centre CPG with period 3; the brake = inhibition from "non-self voice present" units onto the vocal-onset gate (PAG); an integrator unit counting syllables for breath and END; noise into the program layer (babble) | counters and the 3-tick timer (equal to the integrator and oscillator; check: identical syllable timing) | **offline babbling** (35,960 sets through `world.voice`) → online babbling + forward/inverse maps (M2). Program units are localist "command neurons", acceptable as innate motor synergies (Bizzi et al. 2008); they may become premotor assemblies later |
| **P2** | a developmental program: area-level rules that draw random connections from the lineage seed | the generator code is the developmental process, run once | the hand pair schema (`randlayer.PAIRS`, 14 groups, 31 types) → the anchor rule + random branches that name areas only |
| **P3** | a saliency map summing onsets and the pointing or gaze cue per place; WTA → shunting gain on the chosen place | argmax over places (equal to WTA) | none (the slot blocks are the world's retinotopy) |
| **P4** | fixed-weight detector units on person-code units, voice presence and affect units; slow integrators (present), transients (arrives / leaves) | threshold on counts (equal to a fixed threshold unit) | the `soc_feats` timers (300 / 60 ticks) → removed |
| **P5** | fixed synapses (plasticity 0) from P4 and hunger onto the reward input; harm is a heavier path (lateral habenula-like) | a weighted sum (exact) | the world's correctness number → perceived cues (W7) |
| **world side** (allowed) | – | – | eye bit codes are the world's retina. Count and colour fields are gifts (CITED eval) until the local3d eye (W11) |

### 7.3 The shortcut register (swapped back or verified in L13)

| # | shortcut | its circuit | equivalence check | status |
|---|---|---|---|---|
| S1 | exact k-WTA | shared inhibitory pool | winner sets, Jaccard ≥ 0.99 on recorded inputs | keep with check |
| S2 | event-driven ticks, closed-form leaks | per-tick dynamics | bit-identical over 10⁵ ticks | keep with check |
| S3 | scalar δ and μ | neuromodulator neurons | – (volume transmission) | keep |
| S4 | signed weights | E + I interneuron pairs | predictions within 10⁻⁶ | keep with check |
| S5 | accumulator + argmax | BG rate model | same release and tick ± 1 | keep with check |
| S6 | record hippocampus | CA3 / CA1 net | recall@1 within 0.02, or a reported limit | **replace** |
| S7 | localist item IDs | k-of-N assemblies | coverage within seed SD; one-shot naming kept | **replace** |
| S8 | the LATEST list | item × age × source units | identical feature sets | **replace** |
| S9 | body counters and timers | integrator / CPG units | identical syllable timing | keep with check |
| S10 | M4 source list | fixed gated projections | identical drives | keep with check |
| S11 | precision as a running statistic | integrator neurons | identical w(t) | keep with check |

### 7.4 Item IDs → distributed assemblies, and what that buys

**Can IDs be replaced from the ear to the motor side without losing one-shot learning? Yes** (ASSUMED until L13 tests it):
1. **SYL:** a novel syllable recruits 16 free units in one exposure (η = 1/n with n = 1). That is today's one-shot vigilance recruitment.
2. **GEN** predicts a 16-unit target pattern with one delta rule per output unit. One exposure gives p = 0.35 per unit, as today per item.
3. **The program** is the one whose sound pattern best overlaps the predicted pattern: a WTA through the inverse map.
4. **CA3** binds one-shot (Willshaw).

**The cost:** similar syllables share units, so a correction to one slightly changes its neighbours. Decoding can err when predicted patterns overlap. This is the same property that gives fault tolerance.

**What it buys:**

| goal | effect |
|---|---|
| **#14 fault tolerance** | a misheard syllable shares most of its units with the right one, instead of being a different integer; recall works from partial cues; restoration acts on units |
| **#19 merging** | records and dicts cannot be merged; fixed arrays from one birth wiring merge element by element; claims by birth similarity make siblings' codes coincide |
| **#12 scaling** | every hand cap disappears (256 categories, 10 k records, 12 items, the 600 k MIX cap); every structure is N·s units at fixed fractions |

### 7.5 Compute of the final pure-neuron build

Sizes are ASSUMED; arithmetic PROVED in `checks.py`.

| | s = 1 | s = 4 |
|---|---|---|
| units | 106 k | 424 k |
| synapses | 15.7 M (CA3 at fixed fan-in 1,000: 10 M; GEN: 3.1 M) | 63 M |
| memory (4 B weight + 4 B index) | 125 MB | 0.5 GB |
| lit synaptic events per speech tick, event-driven | about 0.16 M | about 0.64 M |
| CPU numpy, one 10⁶-tick life (15 % speech ticks; 75 M events per s assumed; 0.2 ms per silent tick) | about 500 s; realistically 500–1,500 s (today's v2.2: 1,000 s) | about 2,000 s; realistically 2,000–6,000 s (checkpointed across 90-min jobs at the high end) |
| A800 GPU, batched 64 agents (kernel-launch bound, about 0.3–0.6 ms per tick per batch) | about 5–10 s per agent-life | about 10–20 s per agent-life (32 GB for 64 agents < 80 GB) |
| born-wired calibration (750 k cells, 2 M synapses, 13.3 ms per tick, unbatched) | 13,300 s per life if unbatched at that rate: batching is required | – |

---

## 8. Migration plan: a deletion ladder from v2.2 to the pure-neuron core

**Starting point:** v2.2 final.
- staged J 1.205 ± 0.015;
- life free replies 0.47 / 0.52 against references 0.64 / 0.61;
- untutored accuracy 0.16 at 3,000 dialogues (CITED v2.2).

**Rules for every step:**
- comparisons are paired: 5 seeds, same machine, same numpy version, canonical DG tie rule;
- deletion = lesion: a part goes only when its replacement reproduces it on the part's own test;
- the order is lowest risk first.

| step | change | deletes | paired test | pass | fail |
|---|---|---|---|---|---|
| **L0** | **World first (W1–W6)**, options off by default; extend the drift test to 10,000 untutored | – (no agent change) | bench selftest with the options off; re-baseline v2.2 on the old and the new world | bit-identical with the options off; references recorded | – |
| L1 | delete dead code and every switch with no effect at the final configuration | D19, B13, G3, sleep, CTX, split, maturation, decay, `ral_q` … (about 100 knobs) | `test_equiv`: replies on 5 staged seeds + 1 life | bit-identical | any difference |
| L2 | one parameter set for every world (the life values everywhere) | I1, `life_eye` as a switch | staged 5 seeds + lives 5 / 6 | J2 within the seed SD (0.04); life targets hold: social ≥ 0.30, appropriate ≥ 0.38, empty-room ≤ 15 per tenth | otherwise log it as a per-world knob; it must be resolved before the freeze |
| L3 | prune clock in the unit's own firings; regrowth from the shared reserve | the global `age` | drift to 10,000 + staged | J2 within SD; collapse delayed (accuracy at 3,000 ≥ 0.36) | no change: route R-d is not the cause (attribution) |
| L4 | **area-general adaptation** (M1) in the ear, the eye and meaning | `h_*` (13 knobs), `SoundGate`, `cell_gate`, `level_shifts`, scene baseline #23, `eye_rate` | X3 ear tables (2 / 16 / 64 bodies × σ 0 / 0.5 / 1 / 2); bench clean and σ 1; a 10-day ear life σ 0 → 1 → 0.5 → 1 | near recall ≥ 0.95 at σ ≤ 1 and within 0.05 of `h_*`; purity ≥ 0.98; categories ≤ 1.15×; clean-bench replies identical; no deaf day | a deaf day, or purity < 0.95 |
| L5 | meaning online (M1) | the batch stabilisers (15 knobs) | `gen_probe_mean`, 10 seeds; S0 class family | held-out class ≥ 0.95; late words ≥ 0.9; code drift ≤ 0.06 | held < 0.8 or drift > 0.1 |
| **L6** | **self in the stream + the anchored, age-specific context** (time cells, LATEST, own lines in M3) | RAL (24 knobs), `T_own`, `ct_own`, the v2.2 veto (10) | X1 symbolic (recast / mixed / format tutors); then staged 5 seeds on the new world against v2.2 on the same world; lives 5 / 6; drift to 10,000 | symbolic held-out within 0.05 of v2.2 (0.92 / 0.89) on mixed and format; J2 within SD; **untutored accuracy ≥ 0.85 at 3,000 and 10,000**; whole-life free replies ≥ the echo-or-constant reference | drift ≥ 0.15 at 3,000 on ≥ 3 / 5 seeds, or held-out < 0.7 |
| L7 | **one speech gate** (M4 Go/NoGo + TD over time cells) under the maturity rule | P_oth, V_go, `val_veto`, dips / `react_T`, the per-kind constants, reflexes #5 / #7 / #7b, T_MAX, one-utterance-per-thought (→ τ_ref); P1c brake kept | lives 5 / 6; world2 with 16 bodies; staged 5 seeds; all again with timing jitter ×[0.5, 2] | 0 talk-over starts; **0 post-maturity overlaps, garble and extra turns**; childhood curve falling; life targets hold | any post-maturity error on either seed |
| L8 | **one output gate over sources** (M4) | `COPY_*`, gateR (`say_rec`, `say_loc`), the fixed thought-step order | staged 5 seeds; held-out what / where / slot / ritual; lesion each source | what ≥ 0.75, where ≥ 0.85, ritual ≥ 0.95, J2 within SD | what < 0.7 or ritual < 0.9 |
| L9 | **online babbling** (P1d + M2 forward/inverse) with W6 | the `world.voice` read, `rebind` | life with `body.babble` + `keep='reward'` | ≥ 90 % of lexicon chars bound by 20 % of life; 0 post-maturity garble | < 50 % bound |
| L10 | values from perceived cues (W7; P4; P5) | the world's correctness number | develop's E3 lives | learning curves within 10 % | > 10 % slower |
| L11 | **development freeze** of the shortcut build: eval package, S0 and coverage only; lineage gen 0–2 (§2.4) | – | the whole coverage table + S0 families + the red-team | coverage ratio ≥ 0.9 against v2.2; post-maturity mechanics 0 | – |
| L12 | **pure-neuron swap-back, one shortcut at a time** (§7.3): S8 LATEST units → S7 assemblies → S6 CA3 net; verify S1, S2, S4, S5, S9–S11 | every non-equivalent shortcut | each swap's module test, then coverage | each within its stated tolerance; coverage within seed SD | a swap that changes behaviour shows the shortcut carried hidden structure: fix the core, never patch |
| **L13** | **final build:** whole-model equivalence and regression, the compute benchmark at s = 1 and 4, then **the real freeze and the one sealed opening** (§5) on the pure-neuron build | – | §5 protocol | §5 pre-registered criteria | §5 falsifiers |

---

## 9. The first five experiments

They come from the members' E1–E6 and the eval member's "runnable now" list, ordered by risk retired per unit of compute.

| # | harness | arms | prediction | falsifier | compute |
|---|---|---|---|---|---|
| **X1** the anchor on real code | a copy of `v21/regress/gen_probe_v21.py`; a minimal core change (own lines into the context as self; readout anchored on LATEST × age bins; RAL off); recast, mixed and format tutors; 5 seeds; untutored to 3,000 with natural silences + 30 % forced silence | v2.2 (RAL + veto); S (self, no age); **ST** (self × age anchor); ST0 (age not anchored) | as in `drift_toy`: ST ≥ 0.9 at 3,000 with mixed or format tutors; S and v2.2 collapse under silence; every arm collapses recast-only | ST < 0.8 at 3,000 with the mixed or format tutor, or ST's held-out < RAL − 0.1 | local box, 1 process < 300 MB, about 10–60 min |
| **X2** collapse attribution | v2.2 final, untutored to 3,000; 3 seeds | as is; `rm_prune=0`; replay off during untutored use (`recall_n=0`); `ral_hip_own=1`; high urgency (no silence) | silence-free and replay-off remove most of the collapse; prune-off helps c_flat | no arm moves accuracy at 3,000 by ≥ 0.2: the core's stability story misses the cause | A800 CPU, 5 arms × 3 seeds × about 1 h ≈ 2 jobs |
| **X3** the hearing keys → M1 | `hearing/ear_eval.py` on recordings (`record.py` regenerates them) | `h_*` reference; M1 quantile set-point + divisive normalisation + clarity-gated recruitment + P1b phase correction | near recall and coverage within 0.05; clean bench identical; the floor follows σ 0 → 1 → 0.5 → 1 with no stuck rule | near recall < 0.8 in heavy crowds at σ 1, or any deaf day | local, minutes per condition (or 1 A800 job for the grid) |
| **X4** depth in space vs time (learning E2) | `meaning/gen_probe_mean.py` extended to 2 levels (24 fillers → 6 classes → edible or not; zero-shot whole classes) | random + delta; + RI; + a stacked RI; time (two thought steps through recall); online backprop MLP | (d) ≥ 0.7, or (c) − (b) ≥ 0.15 with drift ≤ 0.1 | (c) and (d) ≤ (b) + 0.05 while (e) ≥ 0.7: add a depth mechanism **before** building | local, < 1 h |
| **X5** caregiver timing jitter on v2.2 (eval) | staged, 3 seeds | development timing; ×0.5; ×2 | v2.2's fitted parts (RAL's 6-tick window, `veto_W` 40, reflex 7) lose ≥ 0.1 J2 or overlap under jitter; it also gives the lineage's jitter baseline | J2 unchanged within SD and 0 overlaps: the "fitted to the world" diagnosis is wrong and L6–L7 are worth less than claimed | A800 CPU ≈ 1.1 jobs |

---

## 10. Open risks I could not close (ranked)

1. **The anchored, age-specific context may cost generalisation.**
   - Answers learned at one delay transfer only within about one octave of that delay.
   - Late answers need hippocampal recall to bridge the gap.
   - Units without the anchor (older-content-only) are removed by design, so some context-only generalisation is lost.
   - Tested by X1 and L6.
2. **The core depends on the world modelling answers** (W1–W3).
   - Without them it collapses: PROVED in the toy.
   - Develop's warning holds: rules may move into the world. Accept only caregiver behaviours with citations, at realistic, imperfect rates.
3. **Abstraction without backprop.**
   - Depth by width is exponential: 4.8 × 10⁶ units to cover every triple (CITED learning).
   - Stacked local layers drifted or collapsed in our own records.
   - Tested by X4 before any big build.
4. **Learned turn-taking might not reach 0 post-maturity errors in jittered worlds.**
   - Extra turns after one's own answer have no rule left (6.4).
   - L7 is the test. If it fails, the next step is a better world signal (W4), not a reflex.
5. **RPE-trained gating over sources is sample-hungry** (PBWM).
   - Naming runs through gateR today (lesion 0.91 → 0.40, CITED audit).
   - Tested at L8.
6. **Online babbling** needs lawful physics and caregiver imitation (L9).
7. **The pure-neuron swap may lose accuracy.**
   - A CA3 net with a fixed fan-in of 1,000 has about a tenth of the dense capacity.
   - Distributed codes confuse similar syllables.
   - Tested at L12; budget about 2× today's compute.
8. **Merging learned weights is unproven at our scale.**
   - The only evidence is born-wired R18, a 17-cell graph.
   - Opposite lessons cancel to 0.
9. **Evolution could erode the values of #17**, unless the constraints and the values family hold.
10. **The comparator constants have cliffs.** They become competition plus precision, so colour (0.91) and count may drop (CITED tune).
11. **Crowds.** The brake is on 16–52 % of the day in world2 crowds, so agents go near-mute (CITED hearing). v3 has no learned release ("addressed to me" could provide one later).
12. **Toy limits.** `drift_toy` uses item identities, has no hippocampus, value loop or noise, and runs 5 seeds. Its slight drop for the anchored context without forced silence (0.95 → 0.91) is a change in older context between training and test, not overwriting. It still needs the real harness (X1).

---

## summary_zh

1. **结论：** 新的固定核心只有 4 个机制。
   - 每个脑区用同一种电路；
   - 一条学习规则（三因子，带痕迹）；
   - 海马一次记忆加安静时回放；
   - 基底节"选一个、放出去"的闸门。
2. 另外还有 2 个广播信号：多巴胺式的奖励误差，乙酰胆碱式的"记录/回放"状态。
3. 先天部分 5 项，共 17 个小项，名额写死：身体和脑干、接线蓝图、朝向、粗略的社交识别、价值。今后不能为某项能力加东西。
4. **关键的新结构：** 前额叶记"最近一件事是谁说的、过去了多久"，学习都挂在这上面。
   - 我用小模型验证过：孩子 30% 的时候不吭声，大人隔 45 拍就问下一题。
   - 只把自己的话放进输入流，3,000 段对话后正确率掉到 0；加上"过了多久"，仍是 0.99。这正是 v2.2 后期崩溃的原因。
5. **世界必须示范答案：** 大人自问自答、旁听别人问答、纠正时把问答重说一遍。只有"直接改说答案"的世界里，任何结构都会崩溃，因为世界给答案和给下一题的时间完全一样。所以第一步只改世界，不改模型。
6. **基因 30 个：** 今天的 265 个参数里，38 个并入 21 个基因，25 个成为固定常数，11 个随规模放大，2 个是随机种子，189 个删除。
7. **成长与犯错（您已决定）：** 听满 5,000 个音节以前允许抢话、发音出错；之后必须 0 错误。不抢话的先天刹车保留，按测试台节奏定的几条反射全部删掉。
8. **代代相传（您的新想法）：**
   - 同一血统的个体出生接线完全相同；
   - 长大后把各自学到的连接变化按"同向取平均、相反按多数"融合，再稍微随机一下，生下后代；
   - 后代继承皮层的慢权重，海马和思维从空开始，价值永远是先天的、不参与融合。
   - 每代约 12 只，10 代约 15 小时。
   - 这在生物上不存在（大脑只遗传基因），更像文化传承；对您的目标来说可以接受。
9. **纯神经元：** 最终成品只有神经元、突触、局部规则和两个广播。开发时可以用等价的快捷写法，但每个快捷写法都登记了对应电路和等价检验，最后一步逐个换回、逐个验证。
   - s=1 约 11 万个神经元、1,600 万个突触，一生在 CPU 上约 500–1,500 秒；
   - s=4 约 6,300 万个突触，在 A800 GPU 上必须成批运行。
10. **迁移：** 从 v2.2 出发分 13 步，从风险最低的删起，每步都有配对检验和过/不过的标准。
11. **最先做的 5 个小实验：** 在真实代码上验证"最近一件事×时间"、找出 v2.2 崩溃的原因、听觉开关换成通用自适应、层次深度测试、大人节奏打乱测试。
12. **最大风险：**
    - 新结构可能削弱举一反三；
    - 依赖世界的改造；
    - 没有反向传播时能否学出抽象；
    - 学会的轮流说话能否在长大后做到 0 错误；
    - 靠奖励学"从哪里取词"需要很多经验。
