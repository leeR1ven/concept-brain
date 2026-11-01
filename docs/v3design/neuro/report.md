# Neuro report: top-down from the brain. A fixed core for ARCH_v3

**Role.** neuro (design round v3design, phase 1: design only). I ran no training and no server job.

**My one local run.** A symbolic toy, `arch/v2/v3design/neuro/role_toy.py`:
- outputs `role_toy.out` and `role_toy.json`;
- 1 process, about 11 s, well under 100 MB;
- run as `python3 -I role_toy.py role_toy.json`.

**Labels.**
- PROVED: an argument or arithmetic from the files.
- SIMULATED: my toy run.
- CITED: a paper, or another report in this repo.
- ASSUMED: not checked.

Repo paths are relative to `v2/`. The v2.2 path is the working tree's `arch/v2/v22/`, which I read but did not touch.

---

## 0. Bottom line

1. **The brain learns its many abilities with a small, repeated toolkit, not one circuit per ability** (CITED, §1).
   - The cortex repeats one microcircuit. Areas differ mostly in their inputs and outputs, and in a few parameters that change gradually across the cortex. These parameters are time constant, fan-in and when plasticity closes in development.
   - Around the cortex sit a handful of other circuits, each reused for everything:
     - basal-ganglia loops: select and release, for actions and for working memory alike;
     - the hippocampus: one-shot binding plus replay;
     - a few broadcast neuromodulators;
     - one family of plasticity rules plus homeostasis and pruning.
   - A short list of innate priors makes the right experience likely:
     - the wiring plan;
     - brainstem motor rhythms;
     - orienting to faces and voices;
     - drives and social reward;
     - a few protective reflexes.
2. **Proposed core: 6 generic mechanisms (K1–K6), 3 broadcast scalars (K7), and 6 innate priors (P1–P6).** Section 4 maps the roughly 75 named parts, rules and keys of ARCH_v2 → v2.2 onto it.
   - Most dedicated parts become instances of K1 (the canonical area) or K4 (the gate).
   - Deleted outright:
     - role-aware listening's cancel/commit timing rules;
     - the seven `h_*` hearing keys;
     - the meaning layer's batch recompute rules;
     - P_oth, V_go and the veto as separate learners;
     - the four copy routes as separate code paths;
     - every v2.2 veto key.
3. **Toy result for the biggest deletion, role-aware listening (SIMULATED, 5 seeds per cell).** The drift is a context-structure problem, not a missing rule.
   - Put the agent's own speech in the input stream, coded by source (self or other), and keep the latest line apart from the older one. Then untutored use causes zero drift with no cancel rule: accuracy 1.00 → 1.00 after 900 untutored exchanges.
   - Under ARCH_v2's context, accuracy goes 1.00 → 0.00.
   - Mixing own speech into the same bag (≈ `ct_own`) only slows the collapse: 0.78 after 300, 0.02 after 900. That matches the skeptic's earlier finding (CITED).
4. **The same toy shows v2.2's open problem is about information, not mechanism (SIMULATED).**
   - A recast-only world where the child talks after every correction reproduces v2.1's life failure. The RAL-like rule falls from 1.00 to 0.53.
   - No local commit rule I tried recovers those corrections: the generic variants land at 0.51–0.61.
   - Overheard third-party answers fix it completely: with 300 overheard exchanges first, every core variant reaches and keeps 1.00.
   - Children learn this way too. They overhear conversations (Akhtar, Jipson & Callanan 2001) and get reformulations that repeat their own words (Chouinard & Clark 2003). So the world must supply the cue (overheard answers, prosody, reformulations), and the core supplies one learner for whichever cue exists.
5. **Correcting a mishearing should be learned.** Toddlers do not show perceptual restoration (Newman 2006, CITED). School-age children do (Newman 2004; Walley 1988).
   - In the core it is K1's top-down prediction entering the syllable competition. Its weight is the area's own noise estimate (K7 u, ACh-like).
   - On the clean bench that weight is exactly 0, because the floor's μ = ν = 0 there (hearing report §6, CITED). The clean bench is therefore unchanged (PROVED).
   - With a weight ≤ 0.15, a clearly heard novel word can never be "restored away" (PROVED in §5.4 from the match statistics in the hearing report). The meaning report's objection is answered: restoration is driven by acoustic uncertainty, not by contextual surprise.
6. **Biggest risks.**
   - Generic learned replacements are slower than fitted rules.
   - Fitted reflexes guarantee the 0-overlap mechanics today. I keep only the talk-over brake as a prior, so the learned remainder must be checked in life.
   - Three broadcasts break today's "at most one broadcast" rule. Per synapse it is at most two (§2, K7).

---

## 1. What the brain uses (evidence, compressed)

### 1.1 Cortex: one circuit, many areas

**Shared across areas (CITED):**
- A canonical recurrent microcircuit: excitation amplified by local recurrence and kept in check by shared inhibition (Douglas & Martin 2004).
- Stereotyped laminar classes and connectivity, described as "themes and variations" (Harris & Shepherd 2015).
- A proposed laminar split for predictive coding: superficial layers send prediction errors forward, deep layers send predictions back (Bastos et al. 2012). Feedforward gamma and feedback alpha/beta fit this (Bastos et al. 2015).
- Divisive normalisation as a "canonical computation" in every modality (Carandini & Heeger 2012).

**The strongest evidence that the algorithm is generic: rewiring and cross-modal reuse (CITED).**
- Retinal input routed into ferret auditory cortex builds orientation maps there (Sharma, Angelucci & Sur 2000) and supports visual behaviour (von Melchner, Pallas & Sur 2000).
- In congenitally blind adults, occipital cortex processes language (Bedny et al. 2011).
- Domain areas need experience plus a prior connectivity map:
  - monkeys raised without seeing faces develop no face patches, on an innate retinotopic proto-map (Arcaro et al. 2017);
  - where the visual word form area will appear is predicted by connectivity before the child can read (Saygin et al. 2016).
- In general, an area's function follows from its connectivity fingerprint (Passingham, Stephan & Kötter 2002).

**What really differs between areas (CITED):**

| kind | example | in the core |
|---|---|---|
| inputs / outputs | connectivity fingerprints; corollary-discharge routes (Crapse & Sommer 2008) | P1 wiring plan |
| graded parameters | spines per pyramidal cell rise roughly 16× from V1 to PFC (Elston 2003); a hierarchy of intrinsic timescales (Murray et al. 2014; Chaudhuri et al. 2015); large-scale gradients of gene expression and microstructure (Burt et al. 2018; Huntenburg, Bazin & Margulies 2018) | K1 parameters (fan-in, τ, sparsity) set by P1 |
| maturation timing | synaptogenesis peaks in auditory cortex at about 3 months, in PFC years later (Huttenlocher & Dabholkar 1997); PFC spines keep their juvenile excess into the third decade (Petanjek et al. 2011); development runs along a sensorimotor → association axis (Sydnor et al. 2021) | K3 schedule, P6 |
| real exceptions | agranular frontal and motor cortex has no L4 (Shipp, Adams & Friston 2013); some glutamatergic transcriptomic types are area-specific while GABAergic types are shared (Tasic et al. 2018); Betz cells | accepted: the core treats motor output as P2 plus K1 |

**Verdict.** One cortical algorithm with three or four knobs, plus wiring, is a defensible approximation. It is not literal uniformity (§6).

### 1.2 The loops around the cortex

**Cortico–basal ganglia–thalamo–cortical loops.** The same parallel circuit serves the motor, oculomotor, prefrontal and limbic loops (Alexander, DeLong & Strick 1986, CITED).

**One gate for actions and for working memory (CITED):**
- The basal ganglia gate what enters working memory (input gating) by the same Go/NoGo learning that selects actions (PBWM: O'Reilly & Frank 2006).
- The gating is hierarchical across rostro-caudal loops (Frank & Badre 2012).
- Output gating selects which held item drives behaviour (Chatham, Frank & Badre 2014).
- Gates act on roles, not contents ("indirection"), which generalises to new role–filler pairs (Kriete et al. 2013).
- Human evidence: basal-ganglia activity predicts filtering of distractors from working memory (McNab & Klingberg 2008).
- The loop can implement near-optimal evidence accumulation (Bogacz & Gurney 2007). The subthalamic nucleus raises the threshold under conflict (Frank 2006).
- Striatal plasticity is three-factor: dopamine within about 0.3–2 s after the Hebbian event (Yagishita et al. 2014; Reynolds, Hyland & Wickens 2001).

**Thalamus.**
- The mediodorsal thalamus amplifies and sustains PFC connectivity (Schmitt et al. 2017) and switches cortical representations between contexts (Rikhye, Gilra & Halassa 2018).
- The pulvinar regulates attentional synchrony (Saalmann et al. 2012).
- In the core the thalamus is part of the gate loop (K4) and of the gain (K1, K7 u). It is not a separate mechanism.

**Cerebellum.**
- Marr–Albus–Ito: a huge random expansion (granule cells, about 4 inputs each) read out by an error-driven delta rule with a climbing-fibre teacher (Raymond & Medina 2018). It provides forward models and timing (Sokolov, Miall & Ivry 2017).
- The optimal in-degree of such expansion layers is small (Litwin-Kumar et al. 2017). The same motif appears in the insect mushroom body, the dentate gyrus and piriform cortex.
- **randmix's layer + GEN is architecturally this motif** (fan 3 per branch, random, flat, plus a delta readout; CITED randmix).
- So "random expansion + delta readout" is a generic brain motif. The core uses it in K1 and K2 and adds no cerebellum module now.

### 1.3 Hippocampus

- Pattern separation (DG) plus a one-shot index of the cortical state (Teyler & DiScenna 1986).
- Complementary learning systems: fast hippocampus, slow overlapping cortex (McClelland, McNaughton & O'Reilly 1995; Kumaran, Hassabis & McClelland 2016).
- One-shot CA1 plasticity over seconds (BTSP: Bittner et al. 2017).
- The same circuit also learns statistics (Schapiro et al. 2017).
- Awake replay exists and matters: disrupting awake sharp-wave ripples impairs learning (Karlsson & Frank 2009; Jadhav et al. 2012).
- Replay is plausibly prioritised by gain (Mattar & Daw 2018). But this repo's tune found uniform replay better than strength- or recency-weighted replay (0.62 vs 0.46, CITED ARCH_v2 §5 #22).

### 1.4 Neuromodulators: a few broadcast "meta-parameters"

**Dopamine (CITED).**
- A signed reward-prediction error: bursts and dips (Schultz, Dayan & Montague 1997).
- Also novelty and information-seeking (Bromberg-Martin & Hikosaka 2009).
- Contested (§6).

**Acetylcholine (CITED).**
- Expected uncertainty (Yu & Dayan 2005).
- Raises afferent drive and suppresses intracortical recurrence and feedback ("encoding mode"; Hasselmo 2006).
- In effect it sets the bottom-up vs top-down weight.

**Noradrenaline (CITED).**
- Unexpected uncertainty and change points (Yu & Dayan 2005).
- A phasic "interrupt / reset" (Dayan & Yu 2006; Bouret & Sara 2005).
- Adaptive gain (Aston-Jones & Cohen 2005).
- Pupil-linked arousal tracks the learning rate at change points (Nassar et al. 2012).

**Serotonin (CITED).**
- Patience and waiting for delayed reward (Miyazaki et al. 2014).
- Aversive and opponent roles (Daw, Kakade & Dayan 2002).
- Doya (2002) frames dopamine, serotonin, noradrenaline and acetylcholine as δ, discount γ, inverse temperature β and learning rate.
- Not in my core (§6).

**Two modulators on one eligibility trace.** In visual cortex the same Hebbian eligibility trace is converted to LTP by noradrenaline and to LTD by serotonin (He et al. 2015, CITED). This is precedent for one synapse reading two broadcasts.

### 1.5 Plasticity: a zoo with one underlying family

**The families (CITED):**
- Hebbian and spike-timing plasticity ("earlier → later" strengthens: Bi & Poo 1998).
- Three-factor rules with eligibility traces (Frémaux & Gerstner 2016; Gerstner et al. 2018).
- Homeostatic scaling and intrinsic plasticity (Turrigiano 2008), and the BCM sliding threshold (Bienenstock, Cooper & Munro 1982).
- Inhibitory plasticity for E/I balance (Vogels et al. 2011).
- Short-term depression and facilitation (Tsodyks & Markram 1997).
- Structural turnover (Holtmaat & Svoboda 2009), and overproduction then pruning (Huttenlocher 1979; complement- and microglia-tagged pruning, Schafer et al. 2012).

**One family underneath (CITED).** Calcium-based models reproduce most STDP variants with one rule (Shouval, Bear & Cooper 2002; Graupner & Brunel 2012). Dendritic error models make a local delta rule plausible: the apical (prediction) input compared with the somatic (actual) activity (Urbanczik & Senn 2014; Sacramento et al. 2018).

**In the core:** one parametric rule (K2), homeostasis plus structure (K3), and short-term adaptation inside K1.

### 1.6 Development

- Critical periods open and close by PV-interneuron maturation (Hensch 2005). They exist in speech perception too: perceptual narrowing at 6–12 months (Werker & Tees 1984; Kuhl et al. 2006; Werker & Hensch 2015).
- Synapses are overproduced, then pruned by use (Huttenlocher 1979; Changeux & Danchin 1976). randmix reproduced this: born at 1.5×, about 23 % of units survive, and nothing new becomes active after stage B (CITED).
- "Starting small" as a curriculum (Elman 1993) is disputed (Rohde & Plaut 1999).

### 1.7 Innate priors for language and social learning (CITED)

**Faces.**
- Newborns orient to face-like patterns (Johnson et al. 1991; CONSPEC/CONLERN, Morton & Johnson 1991). Fetuses do too (Reid et al. 2017).
- Newborns prefer eye contact (Farroni et al. 2002).
- An alternative: a general top-heavy bias (Macchi Cassia, Turati & Simion 2004).

**Voices and speech.**
- Newborns prefer the mother's voice (DeCasper & Fifer 1980) and speech over non-speech (Vouloumanos & Werker 2007).
- They are sensitive to native prosody (Mehler et al. 1988).
- The preference for infant-directed speech is robust but modest (d ≈ 0.35) and grows with age and native-language exposure (ManyBabies Consortium 2020). So it is partly learned.

**Turn-taking.**
- Proto-conversation appears by 2–3 months (Trevarthen 1979; Jaffe et al. 2001).
- Response gaps reorganise with development: fast at 3 months, slower around 9 months as language processing enters, then faster again (Hilbrink, Gattis & Levinson 2015).
- Contingent social feedback shapes babbling (Goldstein, King & West 2003).
- Marmosets turn-take with coupled-oscillator dynamics, and parental contingency speeds vocal development (Takahashi, Narayanan & Ghazanfar 2013; Takahashi, Liao & Ghazanfar 2017).
- The ~200 ms gap needs prediction (Levinson 2016; Stivers et al. 2009).

**Statistical learning of syllables.**
- 8-month-olds segment by transitional probability (Saffran, Aslin & Newport 1996).
- Already in neonates (Teinonen et al. 2009).
- Domain-general: in vision (Kirkham, Slemmer & Johnson 2002) and in tamarins (Hauser, Newport & Aslin 2001).
- Can be outweighed by stress cues at 8 months (Johnson & Jusczyk 2001).
- Implication: statistical learning is the generic predictor (K1 + K2), not a language prior.

**Repairing misheard speech.**
- Adults restore masked phonemes (Warren 1970; Samuel 1981). Human STG represents the restored phoneme, and frontal activity before the sound predicts which one is heard (Leonard et al. 2016).
- Toddlers do not show restoration (Newman 2006, Perception & Psychophysics 68:625–642). School-age children do, with a benefit like adults' but lower accuracy (Newman 2004, Applied Psycholinguistics 25:481–493). Younger children restore less (Walley 1988).
- 14-month-olds recognise mispronounced familiar words, but less well (Swingley & Aslin 2000).
- Conversational repair ("huh?") is universal (Dingemanse, Torreira & Enfield 2013).
- Adults reformulate children's errors, and children take up the reformulations (Chouinard & Clark 2003).
- Restoration therefore grows with lexical knowledge and experience. It is not innate. This supports the designer's IDEAS #14 addendum.

---

## 2. The proposed core: 6 mechanisms + 3 broadcasts

Every definition below mentions no task, modality or word. "Area" means any instance of K1.

### K1. The canonical area: one circuit, every area

**Definition.** A sheet of N units. Its parameters come from P1: N, fan-in f, sparsity k/N, time constant τ_a.
- **Birth.** Random sparse fan-in from its input pathways, overproduced. The pathways are flat and mixed, with no pair schema.
- **Competition.** Shared inhibition gives k-WTA. Each input pathway is divisively normalised by its own running total, so every pathway weighs the same whatever its line count.
- **Adaptation.** Each unit has two adaptive variables:
  - fast: short-term depression or fatigue, τ about 10–20 events;
  - slow: a homeostatic set-point that tracks a low quantile of the unit's own drive by a sign rule, `θ ← θ + η(q − 1[x < θ])`. This is a robust, trim-free floor (ASSUMED form).
- **Persistence.** Recurrent self-excitation with time constant τ_a. A long τ_a makes a "thought" area that holds content until it is displaced by competition. Nothing is cleared.
- **Prediction input.** A top-down or lateral input from higher areas, and an efference input from motor areas, that predicts the area's next activity. It biases the competition with weight `w = w_max · u_a` (u from K7). The area reports its match margin (confidence) and its prediction error.

**Brain.** The canonical microcircuit, normalisation, adaptation, the timescale hierarchy, and predictive feedback (§1.1). The efference input is the corollary-discharge circuit:
- self-produced speech suppresses auditory cortex (Houde et al. 2002; Eliades & Wang 2008);
- self-tickle is cancelled (Blakemore, Wolpert & Frith 1998).

**Instances, mapped to code:**

| instance | file and class | settings |
|---|---|---|
| ear layer 0 | `bench/frontend.py` `Ear0`, FEAT0 | plasticity closed at birth (P6) |
| syllable categories | `synth/rules.py` `Templates` (Z) | |
| voice categories | `v21/agent_v21.py` `VoiceCats` | on the speaker residual |
| meaning area | `meaning/meanlayer.py` `MeanCodes` | lateral input = the context |
| conjunction layer PFC-2 | `randmix/randlayer.py` `RandMix` | |
| the thought PFC-1 | `synth/core_synth.py` `SynthCore` (`_ct_input`, `tick_thought`, `_ct_decay`) | long τ |
| fast / slow context pair | new: two τ's, replacing the "previous line" bag | fast = latest line, slow = older lines |
| eye areas, comparator / VR | `synth/core.py` `_match`, `_recognize`; `synth/core_synth.py` | |
| DG | `rules.py` `Hippo.dg` | very sparse |
| time cells | `agent_think.py` P_oth keys | a bank of τ's: generic temporal context (MacDonald et al. 2011) |

### K2. One local plasticity rule (three-factor, with eligibility)

**Definition.** Δw_ij = η_a(age) · m(t) · e_ij, where e_ij is an eligibility trace with area time constant τ_e. There are two forms of the same rule:
- **Prediction synapses:** e_ij ← trace of x_i (t_j − p_j), with p_j = clip(Σ_i w_ij x_i, 0, 1). This is R2 and the designer's "earlier → later" rule in prediction-error form. "Predicted → stop strengthening; predicted but absent → weaken."
- **Representation synapses:** e_ij ← x_i y_j − y_j w_ij. Instar with normalisation; Hebbian.

m is a broadcast (K7). Its default is 1, so most learning is ordinary two-factor learning.

**Brain.** §1.5.

**Subsumes:**
- R1 (one-shot order is K2 at η = 1 in a fast area), R2 (GEN, the copy gates), R5 (template learning), R6 (critic and actor, with m = δ), R7 (outstars PRIME and attD), R8 (P_oth, V_go), R11 (co-holding links);
- randmix's instar;
- meaning RI's instar M;
- the babbling forward and inverse map (R9, which becomes K2 on motor → sensory synapses);
- v2.1's `RecDelta` (that is exactly e_ij);
- the hippocampal one-shot write (K2 with large η and a seconds-long trace, as in BTSP).

### K3. Structure, homeostasis and development

**Definition.**
- Use it or lose it: a synapse unused for T_prune events is replaced by a random new line (randmix (c) or c0).
- Overproduction at birth.
- η_a(age) falls with maturation along the P1 gradient: sensory areas close early, association areas late. These are the critical periods.
- Recruitment is not a rule. An unused unit has a low set-point (K1), so it wins a novel input that matches nothing. "Vigilance" emerges from this.
- No global weight decay. The repo found every decay value harmful (CITED tune).

**Subsumes:**
- vigilance recruitment (ARCH_v2 innate #13);
- MIX recruitment R3;
- randmix pruning;
- the category cap with no recycling (hearing open item 4): junk categories become prunable units;
- R11's fatigue;
- the scene baseline (#23): eye bits present in silence adapt away, by the same set-point that sets the ear's floor.

### K4. The gate loop (basal ganglia–thalamus): one circuit for every "select one and release it"

**Definition.**
- Candidates are proposed by areas: any active representation, or any action program.
- Each candidate's Go − NoGo is read from cortical features (K1 outputs, including temporal context and the P5 detectors), with K2 weights at m = δ.
- Evidence accumulates to a threshold with urgency. The STN-like threshold rises under conflict.
- The winner is released. A protective brake (P3) can veto.

**One class, four uses.** Only the candidate set differs:
- **action release:** speech GO and motor actions;
- **input gating:** what enters the thought, at event boundaries;
- **output gating:** which active representation drives the mouth, such as the prompt copy, the recalled continuation, the word recognised at the attended slot, or the slot's location word;
- **operation selection:** recall vs look vs compare.

**Brain.** §1.2: PBWM, output gating, Bogacz & Gurney, indirection.

**Subsumes:**
- `body_think.ThinkBody` GO/REL;
- the intention accumulator: `agent_think.ThinkAgent._intend`, `_gain`, g0 and urgency (#8);
- P_oth and V_go (`agent_think.SparsePR`) and the veto (`val_veto`);
- the copy gates COPY_prompt, COPY_next and COPY_comp, and v2.1's `gateR` (`core_v21._say_route` / `_say_sets`);
- the fixed thought steps (#10, `core_think.think_step`);
- the thought-open/close rules (#6);
- reflexes #5 (τ ≤ 7 window) and #7b (≥ 10 ticks open);
- v2.2's `veto_R1`, `veto_W`, `veto_rel`, `veto_gate`, `veto_fast`, `veto_eps`, `gain_reply` and `P_react` (`agent_v22.AgentV22._intend`).

### K5. The hippocampal index

**Definition.**
- The DG (a K1 instance at about 2 % activity) reads the whole association-cortex state, one pathway per area, each divisively normalised.
- One-shot K2 binding of the state to an episode cell.
- Recall by partial cue (best overlap, recency as now).
- Awake replay in quiet moments reinstates an episode and teaches cortex through K2 (no sleep).
- Replay starts uniform. Prioritising by |δ| or surprise is a test, not a default (tune's evidence).
- Episodes include the agent's own lines, as source = self. The NEXT links then follow the real dialogue.

**Brain.** §1.3.

**Subsumes:**
- `rules.Hippo` and R4;
- R10 awake recall (`core_v21.awake_recall`);
- the recency σ² term;
- the meaning DG cue and `hipnorm`. That rule is exactly per-pathway divisive normalisation, so it becomes K1's normalisation, not a rule (CITED meaning §6.9).

### K6. Self-prediction through the efference copy

I keep this separate only to make it visible. It is K1's efference input plus P1's wiring.

**Definition.** Every motor command drives, through K2-learned motor → sensory synapses, a prediction of its own sensory result. Predicted input is explained away and represented in a separate "self" source population.
- The PE is small, so K2 hardly learns from it: own speech does not teach the listening readout.
- Only the unexpected part of own speech teaches, such as a garble heard back. That is the auditory-feedback error used for vocal learning (Houde & Jordan 1998; Tourville & Guenther 2011).

**Designer.** The framework's #8 says own speech and others' speech activate similar but different neurons (CITED `docs/research/FRAMEWORK.md`).

**Subsumes:**
- the own-voice tag (#4, "mouth active ≤ 2 ticks");
- the M_listen third factor;
- RAL's `T_own` SELF slot;
- the babbling binding R9 (the inverse of the same map), which also removes the inherited `world.voice` shortcut once babbling runs online in the body.

### K7. Three broadcast scalars

| broadcast | brain | computed from (local to the area that sends it) | read by | replaces |
|---|---|---|---|---|
| δ (dopamine-like) | signed reward-prediction error, bursts and dips | primary values P5 vs the critic's V | K4 Go/NoGo, the actor, K2 commits of eligibility traces | R6 δ, dips, `val_veto`, `act_T` expiry (as trace decay) |
| u_a (acetylcholine-like, per sensory area) | expected uncertainty, i.e. precision | the area's own noise and spread estimate (K1's set-point spread) and its recent match margins | K1 top-down weight w; the area's η (learn less from ambiguous input) | `h_conf`, `h_nocreate`, comparator thresholds (#17) as a precision weight |
| ν (noradrenaline-like) | unexpected uncertainty, a change point | an area's input leaves its predicted family: a voice change, a scene change, or a heard line not predicted by the open exchange | K4 input gate (a new event opens), eligibility traces of the old event are closed, a brief rise in η | `h_vbound`, line segmentation (#14), the thought's open/close (#6), RAL's cancel |

**Cost.**
- Today a synapse sees at most one broadcast.
- With K7, PFC readout synapses see δ and ν (precedent: He et al. 2015); sensory top-down synapses see u; striatal synapses see δ.
- Compute cost is trivial: three scalars per tick.
- The conceptual cost is two more global variables, each with one generic definition.
- Serotonin is left out: patience can be the gate's urgency parameter (§6).

---

## 3. The innate prior budget (fixed; not to be extended per task)

| # | prior | why the brain has it (CITED) | current parts it keeps |
|---|---|---|---|
| P1 | **Wiring plan and maps.** Which areas exist and what feeds what: ear → categories → meaning → PFC; motor → sensory efference; hippocampus reads association areas; a gate loop from every area. Topography: tonotopy, retinotopy, the fovea. The parameter gradients (N, f, k, τ_a, η schedule) | protomap and area patterning, connectivity fingerprints, timescale and spine gradients (§1.1) | retinotopic eye #16, world2 fovea, area sizes |
| P2 | **Motor scaffold.** Brainstem vocal and motor pattern generators: the 3-tick syllable beat (theta rate), program WTA, breath, force ramp. One theta oscillator, shared with the ear's sampling clock and phase-reset by onsets and recognition | PAG and brainstem vocal control (Jürgens 2009); the frame/content jaw cycle (MacNeilage 1998); theta-rate parsing (Giraud & Poeppel 2012) | ARCH_v1 body (#1), syllable clock (#12), `h_relock` as phase reset |
| P3 | **Protective reflexes.** Do not start vocalising while a near voice is active; the first-syllable now-brake | early vocal turn-taking and marmoset antiphony (Jaffe et al. 2001; Takahashi et al. 2013); yielding at overlap (Schegloff 2000) | F/F2 brake, reflex #3 |
| P4 | **Orienting biases**, as attention gain plus mild intrinsic reward: to faces and persons, to voices, speech and infant-directed speech, to the caregiver's pointing and gaze, to novelty | §1.7; gaze following may be partly learned (Triesch et al. 2006) | innate pointing → attended slot (#16) |
| P5 | **Primary values and coarse social detectors.** Food and hunger. Being responded to is rewarding (Goldstein et al. 2003; Watson 1972). Prosocial value: others pleased → +, upset or hurt → −, with harm-avoidance outweighing help. Detectors: person present, addressed to me, pleased, upset | IDEAS #17; amygdala-like broad tendencies; the still-face response (Tronick et al. 1978); spontaneous helping in toddlers (Warneken & Tomasello 2006) | v2.1 `soc_feats` (as detectors), world2 rewards |
| P6 | **Birth calibration and the developmental schedule.** Warm-up statistics (prenatal hearing; DeCasper & Spence 1986); the overproduction ratio; the maturation order | §1.6 | `warmup_windows()`, randmix c0's 1.5× |

**Current "innate" items that are not justified priors.** They become learned (K2, K4) or generic (K1, K7):
- comparator constants (#17);
- thought-step order (#10);
- the MIX pair schema (#18);
- the transition window (#5);
- reflex 7 (#7b);
- the accumulator constants (#8);
- vigilance 0.75 (#13);
- line segmentation by silence (#14);
- the scene baseline (#23);
- the SoundGate trim and stuck rules;
- RAL's windows (12 / 6 / 60 ticks);
- every v2.2 veto key.

---

## 4. Accounting: current part → core → fate

PROVED by reading the code and reports; whether each replacement works is the job of §7.

| current part (file, class or key) | core | fate |
|---|---|---|
| ARCH_v1 body, beat, program WTA, END, breath, D/I/M ramp (`bench/body.py`, `synth/body_think.ThinkBody`) | P2 | keep |
| F/F2 talk-over brake, first-syllable now-brake (#3) | P3 | keep |
| EAR-0 / FEAT0 (`bench/frontend.Ear0`) | K1 (plasticity closed) | keep, reinterpreted |
| syllable clock (#12) + `h_relock` | P2 oscillator + phase reset by K1 recognition | merge |
| `hearing/hear.SoundGate` (`h_gate`: μ/ν, k_on/k_off, trim, stuck) | K1 per-unit quantile set-point + K7 u | **delete** |
| `h_denoise` (`hear.cell_gate`) | K1 per-cell subtractive adaptation | **delete** |
| `h_level` (`hear.level_shifts`, `AgentH._level_match`) | K1 divisive normalisation | **delete** |
| `h_conf`, `h_nocreate` | K7 u (precision-weighted η) | **delete** |
| `h_vbound` | K7 ν (change point of the voice category) | **delete** |
| Z categories (`rules.Templates`, vigilance #13, cap 256) | K1 + K2 instar + K3 recruitment and pruning | merge, remove the vigilance constant |
| `v21/agent_v21.VoiceCats` | K1 on the speaker residual | keep as an instance |
| babbling memory + binding R9 (`DevAgent.babble`, `_bind_from_memory`) | K6 forward map (K2) | replace; online babbling |
| current-line buffer R1 | K2 at η = 1 in a fast area (or K5) | merge |
| PFC-1 thought (`SynthCore._ct_*`, R11, wandering) | K1 long τ + K2 Hebbian lateral + K3 fatigue | keep, reinterpreted |
| "previous line" context (`_ct_view` bag; `ct_own`) | K1 fast/slow pair, source-coded | **change** (§5.1) |
| MIX `rules.MixDict` / `RandMix` + pair schema (#18) | K1 random flat mixed fan-in | keep RandMix; drop the schema later |
| GEN readout (`rules.Delta`, R2) | K2 prediction synapses | keep |
| copy gates COPY_prompt/next/comp + `gateR` (`core_v21._say_route`) + `say_rec`, `say_loc`, `scene_held` | K4 output gating, one class with an open candidate list | **merge into one** |
| thought steps (`core_think.think_step`, #10) | K4 operation gating (innate default order at first) | replace later |
| intention accumulator (`ThinkAgent._intend`, `_gain`, #8) | K4 accumulation | merge |
| P_oth, V_go (`agent_think.SparsePR`), `val_veto`, dips (`AgentV21._dip_check`), `act_T` | K4 Go/NoGo + K7 δ | **merge into one actor** |
| reflexes #5, #7b, prior "floor taken" #7 | K4 learned NoGo | **delete** (keep P3) |
| v2.2 `veto_R1`, `veto_W`, `veto_rel`, `veto_gate`, `veto_fast`, `veto_eps`, `gain_reply`, `dip_scope`, `dip_pthr`, `P_react` (`agent_v22.AgentV22`) | K4 + K7 δ | **delete** |
| RAL (`core_v21.RoleCore._ral_open`, `_ral_close`, `_ral_resolve`, `_ral_commit`, `_ral_cancel`, `ral_mouth_onset`, `ral_tick`; W 12, W_other 6, T 60, reward contingency) | K6 self in the stream + K1 fast/slow context + K2 trace + K7 δ and ν | **delete the rules**; keep `RecDelta` as e_ij |
| `T_own` SELF slot | K6 | merge |
| ATT / attD / PRIME (R7) / MATCH / VR, comparator constants (#17) | K2 top-down synapses + K1 match + K7 u | merge; constants → precision |
| `soc_feats` | P5 detectors + K1 temporal context | keep as P5 |
| Hippo DG / R4 / recall / σ² | K5 | keep |
| awake recall R10 | K5 replay | keep |
| meaning `MeanCodes` (RI: batch recompute every 20 lines, hysteresis 1.2, 2 swaps, placeholder) + `MeanMix` | K1 meaning area, online k-WTA with set-points (the base-rate division b is the set-point) | **delete the batch rules** |
| `mcore.MeanCore` DG cue + `hipnorm` | K5 reading the meaning area through K1 normalisation | merge |
| eye scene baseline (#23) | K3/K1 adaptation (the same rule as the ear's floor) | **delete** |
| line segmentation by silence (#14) | K7 ν / K1 sound↔silence change point | merge |

**Rough count (PROVED by listing).** About 75 named parts and keys:
- ARCH_v2: 23 innate items, 11 rules, 7 third factors;
- v2.1: about 11;
- hearing: 7;
- meaning: about 6;
- randmix: 3;
- v2.2: about 10.

In the core this becomes 6 mechanisms + 3 broadcasts + 6 priors = 15. Everything else is a parameter setting of an instance.

---

## 5. Five abilities from the core plus experience

### 5.1 Who is speaking, and role-aware listening

**Today.**
- ARCH_v2 learns the first line after its own reply as the answer, because own speech enters neither the context nor the hippocampus.
- v2.1 patched this with RAL's timing rules (cancel if I or someone answers within 12 / 6 ticks; commit on reward).
- v2.2 is now adding veto keys.

**Core account.** Three ingredients; none of them mentions dialogue.
1. **K6, self in the stream.** My own utterance enters the input stream as a source = self population, because the efference prediction explains it. Other voices are K1 voice categories. There is no special role variable.
2. **K1, a fast/slow context pair.** The latest line is held by a fast population; older content is held by a slow one. Nothing is cleared.
   - The next question Q2, heard after my reply A′, is learned in the context (latest = A′ from self, older = Q). That context is disjoint from the context "latest = Q from other", in which I answer.
   - So untutored use cannot overwrite the answer.
3. **K2 + K7, learning from reactions.** My production leaves an eligibility trace. The next other-voice line commits to it with m = δ (a burst on praise) or "same event" (¬ν). A dip weakens what I said.

**What experience supplies.**
- Overheard answers by others: model replies, and caregiver-to-caregiver chat in world2.
- Rewards.
- In a richer world: perceivable question cues (prosody, the particles 吗 / 什么) and reformulations.
- Speaker identity then lets K4 learn whose lines to imitate (trust; Koenig & Harris 2005).

**Toy evidence (SIMULATED; `role_toy.py`, 5 seeds, 4 frames × 8 fillers, one delta-rule readout; frozen accuracy over all 32 items):**

| context, commit rule | model 600 → 300 / 900 untutored | recast-only 1,500 → 900 untutored | recast, child talks after every reaction → 900 untutored | role confusion in untutored use |
|---|---|---|---|---|
| A: ARCH_v2 (own speech excluded) | 1.00 → 0.00 / 0.00 | 1.00 → 0.00 | 1.00 → 0.00 | 0.59–0.69 |
| B: own reply in the same bag (≈ `ct_own`) | 1.00 → 0.78 / 0.02 | 1.00 → 0.00 | 1.00 → 0.00 | 0.53–0.77 by 900 |
| R: RAL-like (cancel if I speak next, unless rewarded; + RPE) | 1.00 → 1.00 / 1.00 | 1.00 → 1.00 | **0.53 → 0.54** | 0.01–0.02 |
| C0: self + fast/slow, no production route | 1.00 → 1.00 / 1.00 | 0.00 | 0.00 | 0.16 |
| C1: + commit every reaction | 1.00 → 0.00 / 0.00 | 1.00 → 0.00 | 1.00 → 0.00 | 0.56–0.73 |
| C2: + commit on δ burst, dips | 1.00 → 1.00 / 1.00 | 0.56 → 0.62 | 0.51 → 0.52 | 0.02 |
| C3: + commit if the line is predicted by my answer readout (¬ν) | 1.00 → 1.00 / 1.00 | 0.64 → 0.64 | 0.61 → 0.61 | 0.02 |
| C4: + commit if the line is in a learned answer family (outstar) | 1.00 → 1.00 / 1.00 | 0.54 → 0.59 | 0.56 → 0.58 | 0.02 |
| all C0–C4, with 300 overheard model exchanges before 600 recasts | 1.00 at every point (C1 → 0.00 after untutored use) | | | |

**Readings.**
- **The drift is a context-structure problem.** Self in the stream plus the fast/slow pair gives 0 drift with no cancel rule, PROVED for the toy by construction: Q2 is learned in a disjoint context.
- **The bag form fails, as the skeptic found** with `ct_own` (drift loss −0.23 vs −0.28; CITED review2 SYNTHESIS).
- **RAL's cancel works only while the child stays silent after corrections.** At take-up 1.0 it falls to the reward-only level. That is v2.1's life failure, with 79–87 % take-up (CITED v21 §2.3).
- **No local commit rule recovers unrewarded corrections the child talks over** (0.51–0.61). The learned answer-family gate (C4) falls into a confirmation trap. It only learns answers it has already been rewarded for: in the outstar dump, frame f0's family holds the 2 of 4 colours ever rewarded.
- **Overheard answers remove the problem entirely.** Infants learn words by overhearing (Akhtar, Jipson & Callanan 2001), and adults hand them reformulations (Chouinard & Clark 2003).
- **Recommendation.** For life, give the world third-party conversation and reformulations, not a new rule. Build the core's C3 (the ν "same event" test) for whatever cue the world offers.

**Caveat.** This is a toy. The real agent also has hippocampal NEXT links, replay and conjunction interference. The next test is §7 E1.

### 5.2 The ear's noise floor

**Today.** Seven `h_*` keys in `hearing/hear.py` and `agent_h.py`.

**Core account.** Each key becomes an instance of one rule that holds in every area:

| key | core instance |
|---|---|
| floor (`h_gate`) | each ear unit's K1 slow set-point tracks a low quantile of its drive; "sound" = drive above the set-point by the unit's spread. Every trigger becomes floor-relative automatically, as hearing showed. A sign-rule quantile tracker is robust to bursts by construction (it ignores outliers' size), which is what the trim and the stuck rule were added for. ASSUMED that it is enough |
| `h_denoise` | the same set-point, subtractive |
| `h_level` | K1 divisive normalisation of the window |
| `h_conf`, `h_nocreate` | K7 u: η scaled by precision, so learn less from ambiguous or mixed input |
| `h_vbound` | K7 ν on the voice category |
| `h_relock` | P2 oscillator phase-reset by K1 recognition |

**Same rule elsewhere.** The eye's scene baseline (#23) is the same adaptation, so one rule replaces two dedicated parts.

**Brain (CITED).**
- Adaptation to the stimulus-level distribution (Dean, Harper & McAlpine 2005).
- Contrast gain control (Rabinowitz et al. 2011).
- Noise-invariant A1 representations (Rabinowitz et al. 2013; Mesgarani et al. 2014).
- Adaptation of human auditory cortex to changing background noise (Khalighinejad et al. 2019).
- Divisive normalisation (Carandini & Heeger 2012).

**Clean-bench safety (PROVED).** The bench's warm-up silence is exactly 0 (hearing §6). A quantile set-point therefore starts at 0, and every speech pair has e2 ≥ 21. The gated window equals the raw window, as with `h_gate`.

**Still open (CITED hearing §7).** The brake F in crowds needs relative loudness. In the core it is a K4 NoGo over near/far features, and own-voice loudness (K6) is the reference.

### 5.3 Meaning codes

**Today.** `meaning/meanlayer.MeanCodes` (RI): an instar M, top-16 of M/max(b, floor), plus drift fixes (hysteresis 1.2, 2 swaps per recompute, recompute every 20 lines, placeholders). `mcore` adds the DG cue with `hipnorm`.

**Core account.** A K1 area between the syllable categories and the PFC, whose lateral input is the context.
- Its units learn by K2 instar.
- Each unit's K1 set-point divides by its base rate. That is exactly RI's PMI division by b, so RI already is K1 + K2, computed in batch.
- Online k-WTA with slow set-points replaces the recompute, hysteresis and swap limits.
- The hippocampus reads this area like any other, through per-pathway normalisation (`hipnorm` is that normalisation).

**Brain (CITED).**
- Items in the same temporal community get similar codes in the hippocampus, IFG and ATL (Schapiro et al. 2013).
- Successor-like codes (Stachenfeld, Botvinick & Gershman 2017; Garvert, Dolan & Behrens 2017).
- Infants learn word classes from frequent frames (Mintz 2003; Lany & Saffran 2010).

**Risk.** RI's drift fixes existed because rank flips caused drift (code drift 0.043 RI vs 0.082 TC; CITED meaning). Online k-WTA may drift more; E3 measures this.

### 5.4 Correcting a mishearing

**Today.** Nothing. The meaning report rejected top-down restoration "by construction", because a misheard item looks like a novel combination.

**Core account.**
- K1's top-down input carries the PFC's prediction of the next syllable, π_j. GEN already computes it while listening.
- It enters the syllable competition: s_j = m_j + w·π_j, accepted if ≥ θ (0.75).
- The weight is w = w_max · u_ear: the ear's own noise and spread estimate.
- What experience supplies:
  - a better π, through more lexical and contextual knowledge;
  - a calibrated u, through exposure to noise;
  - conversational repair ("啊?") as a K4 action rewarded when the partner repeats.

**Why novel words are safe (PROVED from CITED match statistics).**
- A clean window matches its own template at about 0.96 (hearing §2). Different lexicon characters overlap ≤ 0.77 even under lawful physics (bench README; 0.48 under random physics).
- Overriding a clearly heard item needs w(π_other − π_true) > 0.96 − 0.77 = 0.19.
- With w_max ≤ 0.15 that is impossible. A clearly heard novel filler is never restored away, and the class test stays intact.
- On the clean bench u = 0, so w = 0. The bench is bit-identical, and the clean-substitution fault test is unaffected. That test is not mishearing: humans hear a clearly said wrong word as the wrong word.

**Gain in noise (PROVED arithmetic).**
- In noisy crowds own-template matches fall to a median of 0.78 (CITED hearing §2).
- A window at m = 0.70 is rescued if π_true ≥ 0.33 (with w = 0.15); a window at 0.65 if π ≥ 0.67.
- Restoration therefore acts exactly on degraded windows within about 0.15 below threshold, and in proportion to the agent's knowledge.
- This is the developmental signature: none in toddlers (Newman 2006), adult-like benefit in school age (Newman 2004).

**Brain (CITED).**
- Warren 1970; Samuel 1981.
- Leonard et al. 2016: STG holds the restored phoneme.
- Sohoglu & Davis 2016: perceptual learning of degraded speech minimises prediction error in STG.
- Disputed against: the Merge model, in which lexical effects are post-perceptual (Norris, McQueen & Cutler 2000; §6). For our agent both readings use the same K1 combination; they differ only in where it happens.

### 5.5 When to speak

**Today.**
- The intention accumulator, P_oth (time cells × thought), V_go, the veto, dips, `soc_feats`.
- Reflexes #3, #5, #6, #7, #7b.
- v2.2's ten veto and dip keys.

**Core account.**
- The speech GO is a K4 action. Go − NoGo is read from cortical features: thought confidence, temporal context (time since the last onset, as a bank of τ's), P5 social detectors and source features.
- One K2 actor is trained by K7 δ:
  - + for the partner's rewarded reaction;
  - a dip when an expected reaction does not come (δ = r − V, so extinction is automatic as V falls);
  - − for being overlapped. Overlap is aversive as a prediction error on hearing another voice while speaking (ASSUMED as a P5 value).
- Evidence accumulation and urgency are the gate's own dynamics.
- Only P3, the talk-over brake, stays innate.
- Merged or deleted:
  - P_oth (the NoGo for "someone will start") and V_go (Go) become one actor;
  - the τ ≤ 7 window and reflex 7b become learned NoGo;
  - v2.2's vetoes are deleted.

**Brain (CITED).**
- Basal-ganglia action selection with dopamine Go/NoGo (Frank 2005; Bogacz & Gurney 2007).
- Turn-taking timing develops (Hilbrink, Gattis & Levinson 2015).
- Parental contingency trains vocal timing (Takahashi, Liao & Ghazanfar 2017; Goldstein et al. 2003).
- An innate antiphonal tendency (Takahashi et al. 2013).

**Evidence that the merge is safe (CITED v2.1 §4).**
- Empty-room greetings stopped once V_go's "no reaction → 0" reached a NoGo.
- The lesion "life without P_oth learning → 0 initiatives" shows the brake's learned part carries initiative. In the core that is the same actor.

---

## 6. Where the brain evidence is weak or disputed

1. **Cortical uniformity.**
   - Against: area-specific excitatory transcriptomic types (Tasic et al. 2018); agranular frontal cortex.
   - Explicit prediction-error neurons have mixed support: mismatch responses in V1 L2/3 exist (Keller & Mrsic-Flogel 2018), but a systematic review finds limited evidence for predictive processing as the mechanism of perception (Walsh et al. 2020).
   - K1's top-down input is a modelling bet.
2. **How much language needs innate structure.**
   - For: early left-lateralised speech responses at 3 months (Dehaene-Lambertz, Dehaene & Hertz-Pannier 2002); the poverty-of-stimulus debate.
   - The core bets on generic learning plus P1–P6. If language needs a hierarchical-structure bias, that would be a seventh prior.
3. **Dopamine = RPE.**
   - Contested by value and motivation accounts (Hamid et al. 2016; Berke 2018), by functional heterogeneity (Engelhard et al. 2019), and by retrospective causal-association accounts (Jeong et al. 2022).
   - K7 δ is the RPE form. A test is to swap in an alternative.
4. **ACh = expected uncertainty, NE = unexpected uncertainty** (Yu & Dayan 2005).
   - This is a computational hypothesis. Acetylcholine's sensory-gain role is solid (Hasselmo 2006). The uncertainty mapping and the specificity of the locus-coeruleus–pupil link are weaker (Joshi & Gold 2020).
5. **Serotonin.** No consensus (Cools, Nakamura & Daw 2011; Dayan & Huys 2015). Left out of the core.
6. **PBWM gating.**
   - The model has indirect support (McNab & Klingberg 2008; Chatham et al. 2014).
   - Alternatives hold working memory in activity-silent synaptic traces (Mongillo, Barak & Tsodyks 2008) or in thalamocortical loops (Schmitt et al. 2017).
7. **Sleep vs awake replay.** Sleep-ripple disruption impairs consolidation (Girardeau et al. 2009). "No sleep" is the designer's choice; awake replay covers part of it (Jadhav et al. 2012).
8. **Innate priors.**
   - Face preference may be a general top-heavy bias (Macchi Cassia et al. 2004).
   - The infant-directed-speech preference grows with exposure (ManyBabies 2020).
   - Gaze following may be learned (Triesch et al. 2006).
   - Turn-taking is part innate rhythm, part learned timing.
   - How much statistical learning explains real segmentation is debated (Johnson & Jusczyk 2001).
9. **Restoration in development.** Data are sparse and behavioural: Newman 2006 on toddlers, Newman 2004 and Walley 1988 on children. Whether the top-down effect is perceptual feedback or post-perceptual merging is disputed (Norris, McQueen & Cutler 2000 vs Leonard et al. 2016; Sohoglu & Davis 2016).
10. **Role learning is hard for children too.** Role-reversal imitation and pronoun reversal are learned (Carpenter, Tomasello & Striano 2005). A generic learner may need rich social input, not just the core. This is consistent with the toy's result.
11. **Replay priority.** Prioritised replay (Mattar & Daw 2018) conflicts with tune's measurement (uniform better). It is unresolved for this agent.

---

## 7. Falsifying experiments (cheapest first; each names the harness)

| # | claim | experiment | pass | falsified if |
|---|---|---|---|---|
| E1 | role-awareness = K6 + K1 fast/slow context, no RAL rules | `v21/regress/gen_probe_v21.py` (recast and mixed tutors, 3 seeds): RAL off; own utterance enters as a source = self line; context = fast (latest) + slow (older) groups; trace committed by δ and ¬ν. Then the staged bench, 5 paired seeds, frozen protocol | symbolic frozen held-out ≥ 0.75 (RAL 0.81 / 0.83; Xcut 0.72, CITED); staged J within the seed SD of v2.1; drift loss after 900 ≤ 0.05; role confusion ≤ v2.1's 0.08 | held-out ≤ 0.5 or drift ≥ 0.2: the hand rules carry information the stream does not. Control: own reply in the same bag must fail (as in the toy and the skeptic's run); if it does not, slot separation is not the active part |
| E2 | the seven `h_*` keys = K1 adaptation + normalisation + K7 u, ν | `hearing/ear_eval.py` on the existing recordings (`record.py` regenerates `rec/`): replace SoundGate, cell_gate and level_shifts by a per-cell quantile set-point + divisive normalisation + precision-weighted category learning. Same tables | near recall and coverage within 0.05 of `h_*` at σ 0.5 / 1 for 2 and 16 bodies; categories ≤ 1.15×; clean bench bit-identical (PROVED by the μ = 0 argument, then run); floor follows σ 0 → 1 → 0.5 → 1 with no stuck rule | the floor runs away in heavy crowds at σ 1 (near recall < 0.8): a quantile is not enough and a gating rule must stay |
| E3 | meaning = an online K1 area | `meaning/gen_probe_mean.py`, 10 seeds, `late=0.8`, `w2=1`: MeanCodes' batch recompute replaced by online k-WTA with slow set-points | class held ≥ 0.95; late words ≥ 0.9; code drift (last third) ≤ 0.06 | held < 0.8 or drift > 0.1: the batch stabilisers are doing real work |
| E4 | restoration is learned from experience, through K1 top-down × K7 u | world2 recordings at σ 1 and 2 and in crowds; agents frozen early in life (after day 1) vs late (day 10); w_max ∈ {0, 0.15}; plus a "raised in noise" vs "raised in quiet" pair | late-life gain on degraded windows ≥ +0.1 coverage at σ 2, about 0 early; quiet-raised agents gain less; class test held stays 1.00; clean-bench fault test unchanged | gains appear early (not learned), or novel-word or class accuracy drops (the prior overrides clear input) |
| E5 | when to speak = one K4 actor + P3 | life, seeds 5 and 6, 10⁶ ticks: P_oth, V_go, veto, dips and reflexes #5 / #7b replaced by one Go/NoGo actor with δ (reward +, missing expected reaction = dip, overlap −); v2.2 keys off | 0 talk-over starts (P3); overlaps from the other starting while I speak ≤ v2.1's rate after the first tenth; empty-room 你好 ≤ 15 per tenth; appropriate initiatives ≥ 0.38; social ≥ 0.30 | any of these fails on both seeds: the dedicated predictor P_oth carries something the single actor cannot learn |
| E6 | corrections need world information, not a rule (toy §5.1) | life with world changes only: caregivers sometimes answer each other (overheard routines), and corrections reformulate the child's words; core E1 agent | life free replies above the same-life echo-or-constant reference (0.62) on seeds 5 and 6 | still below with overheard answers: the core lacks a mechanism (then reconsider a learned question cue) |
| E7 | **the decisive test: new abilities without new code** | freeze the code after E1–E5. In world2 (16 agents, 207-character lexicon, fovea), compare 1× and 4×. Add task types no mechanism was designed for (X大不大, 你拿着什么, 这是谁, requests) | each new task rises above its same-life reference with a learning curve like the old tasks'; 4× ≥ 1× once the world is large | any new task needs a code change, or 4× < 1× persistently (CITED: meaning 4× −0.11 in the small world, so judge only in the large world) |

**Rule for every deletion: deletion = lesion.** A dedicated part is removed only after its core replacement reproduces the part's behaviour within noise on the part's own test (paired, 5 seeds, same machine and numpy version, canonical DG tie rule).

---

## 8. Costs, risks and the order of work

**Order.**
1. E2 and E3: module tests on existing recordings and probes. No agent runs.
2. E1: symbolic, then staged.
3. E5 and E6: life.
4. E4: world2.
5. E7: the scaling test.

**Compute.**
- The fast/slow context adds one more input group to the conjunction layer: about 1.3× populations (ASSUMED).
- The top-down prior reuses GEN's already-computed drive: one gather per sample tick.
- K7 adds three scalars per tick.

**Risks.**
1. **Learned replacements of fitted reflexes may produce overlaps early in life.** J's rule (any overlap → −1) then punishes them. P3 is kept so that no start ever overlaps.
2. **Generic rules are slower than hand rules.** The toy's recast-only world shows this: 0.5–0.6 vs 1.00 for "commit always", which has no drift protection.
3. **The CTX lesson.** A slow context in the hippocampal key hurt name recall (CITED ARCH_v2 §7 #16). The slow area must enter the DG through a normalised pathway, and E1 must report name recall.
4. **Surprise-gated thought entry harmed B** (0.69 → 0.15, CITED). K7 ν must mark perceptual change points (sound/silence, voice, scene) and open the gate at full gain. It must not scale entry by content surprise.
5. **Two broadcasts reach the PFC readout synapses.** That exceeds today's limit. Brain precedent: He et al. 2015.

## 9. Files (all in `arch/v2/v3design/neuro/`)

- `role_toy.py`: the toy, with variants A, B, R, C0–C4 documented in its docstring.
- `role_toy.out`: the printed table.
- `role_toy.json`: means and SDs per schedule, variant and phase.

**Web sources checked for the restoration citations:**
- [Newman 2004, Cambridge](https://www.cambridge.org/core/journals/applied-psycholinguistics/article/perceptual-restoration-in-children-versus-adults/5FBD4E565309E3788569E251AAEA41C8)
- [Nagaraj & Magimairaj 2020 review, PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC7641369)
- [Newman CV (Newman 2006, P&P 68:625–642)](https://hesp.umd.edu/sites/hesp.umd.edu/files/cv/CV%20Newman%20shortform%202025.pdf)
- [ScienceBlogs summary of Newman 2006](https://scienceblogs.com/cognitivedaily/2008/05/01/do-toddlers-fill-in-the-gaps-w)

Every other citation is from my own knowledge of the literature and was not re-checked online.

---

## summary_zh

1. **大脑靠一套固定的通用机制学会很多本领。**
   - 皮层各区用的是同一种局部电路。各区的差别主要是"接哪些输入输出"，再加几个渐变的参数：时间常数、连接数、可塑期什么时候关闭。
   - 视觉信号改接到听觉皮层后，听觉皮层也能长出视觉特征。先天失明的人，视觉皮层会用来处理语言。
   - 皮层外面只有几样通用电路：
     - 基底节环路：既管动作的选择和放行，也管工作记忆里放进什么、拿出什么；
     - 海马：一次记住，事后回放；
     - 几种广播式的神经调质；
     - 一类可塑性规则，加上稳态调节和修剪。
2. **我提的核心：6 个通用机制，加 3 个广播信号，加 6 条先天先验。**
   - 6 个机制：
     - 同一种"标准区"：出生时随机连接、名额竞争、除法归一化、自适应底线、自上而下的预测；
     - 一条三因子学习规则（带资格迹）；
     - 结构与发育：用进废退、先多长再修剪、关键期；
     - 一个"选一个、放出去"的闸门（基底节）；
     - 海马索引和清醒回放；
     - 用"传出副本"预测自己的声音。
   - 3 个广播信号：多巴胺（有正有负的奖励预测误差）、乙酰胆碱（输入可靠程度）、去甲肾上腺素（意外的变化，即事件边界）。
   - 6 条先天先验：接线图、发声节律、不抢话的刹车、朝向人脸和人声、基本价值和亲社会价值、出生校准和发育时间表。
   - 现在约 75 个专门的部件和开关，大多变成这几样机制的实例。可以删掉的有：分角色倾听的计时规则、听觉的 7 个开关、词义层的批处理规则、三个分开的"说不说"学习器、四条复制通道、v2.2 的全部否决开关。
3. **小实验（符号世界，5 个种子）：**
   - 只要把"自己说的话"按来源放进输入流，并把"最新一句"和"更早的内容"分开存放，无人指导时就完全不漂移，不需要任何撤销规则（900 段后正确率仍是 1.00；ARCH_v2 的方式掉到 0）。
   - 把自己的话混进同一个袋子只能延缓漂移（300 段后 0.78，900 段后 0.02）。
   - 孩子每次被纠正后都接着说话时，v2.1 式规则掉到 0.53，我试的各种通用规则也只有 0.51–0.61。但只要孩子能先旁听到别人回答问题，所有通用版本都是 1.00。
   - 所以"分不清纠正和下一个问题"缺的是世界里的信息，不是机制。世界里要有旁听的对话、改述式的纠正、疑问语调。
4. **听错后的纠正应该靠经验学会。**
   - 两三岁的幼儿还没有"音素修复"，学龄儿童才有（Newman 2006、2004）。
   - 做法：前额叶的预测参与音节竞争，权重等于耳朵自己估计的噪声。
   - 安静的测试台上权重正好是 0，结果完全不变；权重不超过 0.15 时，听得清楚的新词永远不会被"改成"熟词。
5. **需要先做的检验**（从便宜到昂贵）：
   - 用已有的录音检验听觉的 7 个开关能不能换成通用的自适应；
   - 用小测试检验在线的词义层；
   - 用符号测试和分阶段测试检验"自己进入输入流 + 快慢两层上下文"；
   - 在"一生"里检验一个统一的"说不说"闸门；
   - 在 world2 的噪声里比较一生早期和晚期的修复能力；
   - 最后是决定性检验：代码冻结后，只让世界变大、模型变大，看能力是否继续上升、新本领能否不加代码就学会。
6. **证据薄弱或有争议的地方：**
   - 皮层是否真的统一；
   - 多巴胺是不是奖励预测误差；
   - 乙酰胆碱和去甲肾上腺素是否对应"预期的不确定"和"意外的不确定"；
   - 血清素的作用；
   - 基底节是否真的管工作记忆的闸门；
   - 不睡觉能否完全替代睡眠里的回放；
   - 各条先天偏好有多少其实是后天学来的。