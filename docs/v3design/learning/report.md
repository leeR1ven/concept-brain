# Learning-theory member: what a fixed core needs in order to learn, and the smallest core that might

Scratch folder: `docs/v3design/learning/`. It holds one analytic script, `width_vs_depth.py`, and its output `width_vs_depth.out` (one process, a few MB, no training). I ran no training and no server jobs.

Labels: **PROVED** = argument, arithmetic or code count. **SIMULATED** = a numpy run (here only runs from earlier reports, with the report named). **CITED** = a paper; "(v)" means I checked the abstract this round. **ASSUMED** = argued but not tested.

The coordinator's note on global scaling is answered in §7.

---

## 0. Bottom line

1. **"Learn anything" cannot be met literally.** Recurrent networks and neural assemblies are computationally universal (Siegelmann & Sontag 1995; Papadimitriou et al. 2020; Dabagia et al. 2024 (v)). Learnability is a separate question:
   - learning is intractable in general (Blum & Rivest 1992);
   - every learner needs an inductive bias (Wolpert 1996).

   **The goal that can be met:** a core whose description does not grow with the number of abilities. It learns whatever the world makes learnable within one life, given a small fixed budget of innate priors.
   - The operational test is a **held-out-ability protocol**: freeze the code, then give the agent task types it was never built for (§8).
2. **Today's agent is far from that.** I counted about 60 named mechanisms and **243 configuration keys** (PROVED by count, `width_vs_depth.out`):
   - `core.py` Cfg alone has 84 keys;
   - `RCfg` 26, `ThinkAgent` 30, `h_*` 19, `MeanCodes` 22, `RandMix` 21.
3. **Proposed core: predict, value, memory (the P-V-M core).**
   - 2 learned objectives:
     - O1: predict the next input everywhere;
     - O2: predict and maximise discounted innate reward (TD).
   - 1 rule template with 3 instances: delta rule, three-factor rule, competitive Hebbian with one-shot recruitment.
   - 2 broadcast signals: δ (global reward-prediction error) and ν (per-area surprise or precision).
   - 6 architecture motifs and 6 budgeted innate priors.
   - That is 19 named items, and each new ability adds none. Almost every v2.x part maps onto them (table in §5.6).
4. **Credit assignment (§3).** None of the seven proposed families is local, online, life-long and deep all at once.
   - **Chosen:** delta-rule readouts (exact for one layer), three-factor rules for actions and gates, and local self-supervised or competitive learning for hidden areas.
   - **Rejected for now:** equilibrium propagation (needs phases and settling), forward-forward (needs negative data or sleep), and feedback alignment (needs a vector error broadcast).
5. **Objectives (§4).**
   - The hearing front end and the value loop's social features can become instances of generic motifs: adaptation and normalisation in every area, a bank of traces at fixed time constants, a TD critic, and a Go/NoGo gate.
   - Role-aware listening (RAL) can probably lose its hand timing windows and its reward exemption. What cannot be learned away is a small prior about **whom to imitate**: the self/other tag. This is experiment E1.
6. **Biggest theoretical risk: abstraction without backprop in one unbroken life.**
   - **Depth by width is exponential** (PROVED, `width_vs_depth.out`, randmix geometry, 10 responding units per tuple). Covering every pair needs about 1.1×10⁵ units, every triple 4.8×10⁶, every 4-tuple 1.5×10⁸.
   - **Stacked local layers drifted or collapsed in our own records:**
     - the meaning round's competitive layer (CL) collapsed;
     - its bootstrapped temporal-context code (TC) drifted 0.082 vs 0.043;
     - a deeper ear layer lost information.
   - **Composition in time (thought steps) does not reach content yet:** with the thought steps in a fixed order, more thinking time added nothing (think report).
   - E2 tests this before any big build.
7. **Second risk: data in one life.** A staged run hears about 2–4×10⁴ syllables, a 10⁶-tick life about 10⁵, and a world2 life about 5×10³ per agent.
   - A 5-year-old has heard about 6×10⁷ words (Frank 2023 (v)).
   - With about 1.2×10⁷ GEN weights that is about 10⁻³–10⁻² syllables per weight, against about 20 tokens per parameter at Chinchilla's optimum (Hoffmann et al. 2022).
   - Generic learners must discover from data what the hand rules gave for free, so they cost experience. E1 measures that cost.
8. **Scaling (§7).** Define one global factor s:
   - unit counts and active counts scale with s, so active fractions stay fixed;
   - time constants, set points, the body and the rules stay fixed;
   - world diversity and life length grow with s.

   **Predictions:**
   - in a fixed world, growing s with a fixed active *count* lowers J; this explains meaning's −0.11 at 4×;
   - in a fixed world, growing s with a fixed active *fraction* leaves J flat;
   - if world diversity and life length grow with s, seen accuracy stays flat and held-out accuracy rises **only if hidden codes are learned**. That held-out slope is the test of "the architecture is sound". This is experiment E3.

---

## 1. What "a fixed set of mechanisms that learns anything" requires

### 1.1 Universality is cheap; learnability is not
- **Expressive power** (CITED).
  - Recurrent nets with rational weights are Turing-complete (Siegelmann & Sontag 1995).
  - A random recurrent pool with trained readouts is universal for fading-memory filters (Maass, Natschläger & Markram 2002).
  - Random graph + Hebbian plasticity + k-cap inhibition, the "assembly calculus", can project, associate and merge (Papadimitriou et al. 2020). It learns sequences and any finite-state machine from presentations, and is universal with long-range interneurons (Dabagia, Papadimitriou & Vempala 2024 (v)).
  - Our motifs are k-WTA areas, random projections, Hebbian rules and recurrence. They lie in this class, so **the architecture class excludes no ability** (PROVED by citation).
- **Learnability** (CITED).
  - Training even a 3-node network is NP-complete (Blum & Rivest 1992).
  - Averaged over all worlds, no learner beats another (Wolpert 1996).
  - So the core must carry a bias that matches the world's structure:
    - compositional (frames × fillers);
    - hierarchical (syllable → word → class);
    - temporally local, with sparse causes;
    - social (others' behaviour predicts reward).
- **Brain analogue.** Evolution is the outer loop that sets the bias, and it must pass through a "genomic bottleneck": wiring rules and a few numbers, not per-task circuits (Zador 2019). For us, the designer plus the tune sweeps are that outer loop. The budget rule follows: **priors are allowed only if they are few, fixed, content-free or broadly social, and never added per task.**

### 1.2 Five requirements a learn-anything core must meet (ASSUMED synthesis of the above)
1. **A label-free objective** that turns any regularity in the stream into a learning target. Next-input prediction does this.
2. **A value objective** for what the stream does not say is good. Innate reward feeds TD.
3. **Fast non-parametric memory** for one-shot specifics, with slow consolidation (complementary learning systems: McClelland, McNaughton & O'Reilly 1995; Kumaran, Hassabis & McClelland 2016).
4. **Credit assignment** across space (hidden units), across time (delayed outcomes) and across "which operation was used" (thought actions).
5. **Stability with plasticity always on:** sparse codes, a plasticity spectrum, replay and homeostasis.

v2.x meets 1–3 partly. It hand-codes most of 4 and 5 per ability.

### 1.3 Depth by width is exponential (PROVED, `width_vs_depth.py`)
- **Setup.** randmix geometry: an item is 16 of 512 lines, a branch draws 3 lines, and a unit fires if every branch has at least one active line.
- **Per-unit probability.** A branch hits a given item with p₁ = 0.091, so a unit responds to a specific m-tuple with probability p₁ᵐ.
- **Units needed for 10 responders per specific tuple, for every tuple type over the 14 groups:**

| order m | units per type | types | total |
|---|---|---|---|
| 2 | 1,207 | 91 | 1.1×10⁵ |
| 3 | 13,260 | 364 | 4.8×10⁶ |
| 4 | 145,677 | 1,001 | 1.5×10⁸ |

- **Today:** about 48k units over 45 populations, i.e. about 8.8 responders per specific pair.
- **So** a fixed random basis plus a trained readout (a kernel machine) is enough for pairs. Every further order costs about 11 × (the number of new types) more units.
- Abilities that need 3- and 4-way structure must come from one of:
  - (a) **learned factorised codes** (class codes), which shrink the effective number of items;
  - (b) **composition in time**: several shallow steps held in the thought.

  These are the two arms of E2.

### 1.4 The data budget of one life (PROVED arithmetic; the voice-tick share 0.2–0.4 is ASSUMED)
- **One life hears very little.** A staged run hears about 2–4×10⁴ syllables, a 10⁶-tick life about 0.7–1.3×10⁵, and a world2 10-day life about 5×10³ per agent (≈ 130 lines a day, from the hearing report). A child by age 5 has heard about 6×10⁷ words.
- **Syllables per GEN weight** are about 10⁻³–10⁻².
- **What follows (ASSUMED reading, consistent with every size result in the repo):**
  - The agent sits deep in the data-limited regime.
  - That is why "size fixes none of the failures": develop and brainlike at 2–4×; randmix at 4× and 16× gained nothing; meaning at 4× lost 0.11.
  - It is also why generic replacements for hand rules must be checked for **experience cost**, not only end accuracy.

---

## 2. Precedents: what each shows and what it costs

- **Meta-RL in the PFC** (Wang et al. 2018 (v)).
  - Slow weights, trained by DA-like RL, shape a recurrent net whose activity dynamics then learn new tasks with **weights frozen**. That is "new abilities without new code" in its purest form.
  - **Cost:** the slow loop was backprop-through-time with an actor-critic (A3C), over thousands of tasks drawn from a family. Neither the outer loop nor the task distribution exists in one unbroken life with local rules.
  - **Lesson.** Read the picture as layered timescales:
    - evolution, i.e. design plus tune, sets the priors;
    - within life, slow learning of **what to gate and hold** (L2 on M5 below) shapes the dynamics;
    - the fast "learning" lives in the thought and the hippocampus.

    Do not copy the mechanism.
- **Leabra / O'Reilly** (O'Reilly 1996; O'Reilly et al. 2012, *Computational Cognitive Neuroscience*).
  - One fixed set of equations is used across models of perception, attention, memory, language and executive control:
    - k-WTA inhibition;
    - error-driven learning from local activation differences between a minus and a plus phase (GeneRec, equivalent to backprop for symmetric weights);
    - Hebbian learning mixed in;
    - PBWM gating of PFC by the basal ganglia;
    - a hippocampal model.
  - DeepLeabra supplies the plus phase without a teacher: the actual next input arrives about every 100 ms through the pulvinar. The IT pathway then learns invariant object categories from raw video by prediction alone (O'Reilly et al. 2021 (v)). **This is exactly IDEAS #1**: the top state predicts the next input, and the next input is the teacher.
  - **Gap.** Each Leabra model is a separate network trained offline for many epochs, not one life. Its depth comes from bidirectional error-driven learning, which we do not have yet.
  - **Lesson.** The core below is essentially "Leabra in one unbroken life". The precedent says the *set* is sufficient for many cognitive phenomena. Whether it works without epochs is our risk 2.
- **Cortical uniformity** (CITED: Mountcastle 1978; Sharma, Angelucci & Sur 2000; von Melchner, Pallas & Sur 2000). Auditory cortex rewired to receive visual input develops orientation maps and mediates visual behaviour. One cortical algorithm plus the input statistics makes the modality. This is the strongest biological support for "fixed mechanisms, function from experience" (for the cortex only).
  - **The brain is not one circuit, though.** It has a handful of distinct circuit types (cortex, basal ganglia, hippocampus, thalamus, cerebellum, brainstem) and about 4 broadcast neuromodulators. These act like the meta-parameters of one learning machine: DA = TD error, 5-HT = discount, NE = exploration, ACh = learning rate (Doya 2002). The budget I propose has the same shape.
- **Spaun** (Eliasmith et al. 2012) is the counter-example. It does many tasks with hand-compiled per-task modules, which is exactly what the designer wants to avoid.
- **Predictive learning as the one objective.** How far does it carry? (ASSUMED, from our bench evidence)
  - **It covers:** perception and categories, sequence and frame learning (GEN is already a next-item predictor), turn-taking (P_oth is a predictor), class codes (meaning RI is a distributional predictor), a world model of others' reactions to my own utterances, and phonemic restoration (#14: prediction as a prior on hearing; Warren 1970).
  - **Theory.** Even *linear* next-token predictors are universal learners when the data contain step-by-step intermediate tokens (Malach 2023 (v)). The computation is supplied by the sequence, which is the depth-in-time route of §1.3.
  - **Where prediction alone fails:**
    - (i) **what to do.** Initiative "says the partner's lines" because the readout predicts what is heard next (ARCH_v2 §12.3). Choosing what to say needs value.
    - (ii) **whom to imitate.** The role confound (review 2) is a prediction confound: without a self/other tag, the line after my answer is predicted as the answer.
    - (iii) **one-shot specifics.** These need the store.
    - (iv) **deep credit.**
  - **Unification.** TD is itself a prediction rule (Sutton 1988), and values are linear in a predictive map of future states (Dayan 1993; Stachenfeld et al. 2017). So O2 is O1 on a discounted target. The core is "predict at two horizons; act so the long-horizon prediction is high".
- **RL over thought actions and motor actions** (CITED).
  - Gating working memory can be learned as actions by DA-like TD: Todd, Niv & Cohen 2008; PBWM (O'Reilly & Frank 2006).
  - Attention-gated tagging (AuGMEnT; Rombouts et al. 2015 (v)) learns working-memory tasks with tags plus one TD broadcast.
  - Indirection (one PFC stripe holding a pointer to another) gives role/filler systematicity beyond experience (Kriete et al. 2013 (v)).
  - **Cost:** sample complexity. Node perturbation is slower than gradient descent by a factor equal to the number of output units (Werfel, Xie & Seung 2005 (v)).
  - **Lesson.** Thought operations should be M5 actions learned by L2. That is slow, so use dense learning signals: the critic's δ after every reply, not only end-of-episode reward.
- **Episodic memory for one-shot learning** (CITED).
  - Episodic control wins early in learning: Lengyel & Dayan 2007; Blundell et al. 2016; Pritzel et al. 2017; Gershman & Daw 2017.
  - Ours, SIMULATED: one-shot naming is 0, because the completion gate never learns to copy the recalled answer; with the gate forced open it is 0.60 (develop follow-up).
  - Also SIMULATED: an entity-keyed sparse store learned 10k facts one-shot with no forgetting (llm2hebb).
  - **Lesson.** The store works; **using** it is a credit-assignment problem for the gate (L2), not a memory problem.

---

## 3. Local credit-assignment options

The question for each: is it local, online, workable in one unbroken life, and does it scale in depth?

| family | local? | online? | one unbroken life? | scales in depth? | evidence (CITED; ours SIMULATED) | verdict for the core |
|---|---|---|---|---|---|---|
| **three-factor** (eligibility trace × broadcast) | yes: pre, post, trace and one scalar | yes | yes, no phases | alone, poorly: variance grows with the units involved (Werfel 2005 (v)). With attention-gated feedback from the chosen action it is deep: BrainProp ≈ backprop on CIFAR-10/100 and Tiny ImageNet (Pozzi, Bohte & Roelfsema 2020 (v)) | Frémaux & Gerstner 2016; Gerstner et al. 2018; Izhikevich 2007; AuGMEnT (v). Ours: R6 actor, RAL's trace | **Keep as L2** for the policy, gates and thought operations. Feedback-gated tags are the upgrade path for depth. |
| **e-prop** (Bellec et al. 2020 (v)) | needs a per-neuron learning signal (random or symmetric feedback of the output error), i.e. a vector, not one scalar | yes | yes | in time it truncates credit through recurrent paths, and the RFLO relative struggles at long timescales (Murray 2019 (v)); shallow in space | TIMIT: competitive with BPTT for LSTMs and spiking nets; learns slower than BPTT | **Not now.** Learn *what to hold* by gating (L2, as PBWM does) first. e-prop is the fallback if gating cannot learn it. |
| **equilibrium propagation** (Scellier & Bengio 2017) | yes, a contrastive-Hebbian difference between two equilibria; symmetric weights | per example; needs a free phase and a nudged phase (holomorphic EP turns them into an oscillation) | **poor:** assumes static input and settling to a fixed point, while our thought flows and never settles | good: ImageNet 32×32 on par with backprop (Laborieux & Zenke 2022 (v)) | — | **No.** Phases and settling clash with continuous thought, and compute is heavy. |
| **predictive-coding nets** (Rao & Ballard 1999; Whittington & Bogacz 2017; Millidge et al. 2022) | yes, error units × pre | an inner inference loop per input | mostly. "Prospective configuration" settles activity first and reduces interference (Song et al. 2024 (v)), which suits one life | weak: scalability with depth is the field's stated open problem (Pinchetti et al. 2024 (v)). Ours (llm2hebb): standard PC diverged; the fixed-prediction variant equals backprop (cosine 1.0) but needs non-local timing | — | **Partial.** Take the per-area error-unit form of O1, not deep PC inference. |
| **target propagation** (Lee et al. 2015) | layer-local, with learned inverse or feedback weights | yes | the inverses must track drifting forward weights | plain variants fail on ImageNet (Bartunov et al. 2018 (v)); with a special feedback-training scheme, DTP reaches about 60.6 % top-5 on ImageNet 32 (Ernoult et al. 2022 (v)) | — | **Use where an inverse is natural: speech.** Heard syllable as target → inverse model → muscle set (goal babbling, Rolf et al. 2010; DIVA, Guenther 2006). This replaces the offline `world.voice` babbling shortcut. |
| **feedback alignment / DFA** (Lillicrap et al. 2016; Nøkland 2016) | yes, given a vector error broadcast through fixed random weights | yes | yes | FA fails on deep convnets and ImageNet (Bartunov 2018); DFA comes close to backprop on many tasks, with a larger gap on transformers (Launay et al. 2020 (v)). Ours (llm2hebb): random feedback has cosine ≈ 0 with the gradient when grafted onto an existing net | — | **Not now.** Needs a vector broadcast, and it pays only for deep supervised nets we do not have. |
| **forward-forward** (Hinton 2022 (v)) | yes, a layer-local "goodness" | needs negative data | negatives come from a separate phase (Hinton suggests sleep) or a generator, which clashes with "no sleep" | small locally connected nets: test error only slightly worse than backprop on CIFAR-10; depth unproven | — | **No, as stated.** Its useful idea, "contrast real input with my own prediction", is a contrastive form of O1. |
| *(added)* **local self-supervised / competitive layers** (LPL; CLAPP; SoftHebb; Hebbian + k-WTA) | yes. CLAPP needs one broadcast; LPL is purely local (predictive + Hebbian + decorrelation terms) | yes | in principle yes; collapse needs variance or decorrelation terms. Ours: the meaning round's CL collapsed and its TC drifted | moderate. CLAPP readout accuracy rises with layer up to 5–6 layers (Illing et al. 2021 (v)); LPL learns invariant object codes in deep nets (Halvagal & Zenke 2023 (v)); SoftHebb reaches 80.3 % CIFAR-10 but 27.3 % ImageNet (Journé et al. 2023 (v)) | — | **Keep as L3 plus area-local O1** for hidden areas. E2 tests stacking in one life. |

Our own records (SIMULATED, `side/llm2hebb`, `v2/meaning`, ARCH_v2 §7.11) agree with the literature:
- the delta rule is exact only at the last layer;
- cheap local rules are far from the gradient (layer-local delta cosine about 0.20);
- coupled plastic layers drift or collapse unless anchored.

---

## 4. Can the hand-made v2.x parts become 1–3 learned objectives?

### 4.1 Role-aware listening (`v21/core_v21.py` RoleCore, `ral_*`)
- **What RAL encodes:**
  - (a) a per-synapse eligibility trace on a line's answer-learning. This is generic: L1 with a trace. Keep it.
  - (b) **role from realized timing:** cancel if I speak within 12 ticks, or another speaker within 6. Hand-set.
  - (c) a **reward exemption** (commit a rewarded line within 60 ticks of my utterance). Hand-set.
  - (d) cutting the NEXT link. Hand-set.
  - (e) a SELF slot `T_own`. This is a prior.
- **Why the timing rules were needed.** Own speech is excluded from the stream. `ct_own` is 0 by default, and when it is on, own items enter the *same* untagged item assemblies (PROVED, `core_synth.py` lines 14, 319). So "the line after my answer" is learned in the context of the question.
  - The skeptic's `ct_own` 0.6 arm failed (drift −0.23 vs −0.28) because nothing distinguishes my item from a heard item.
  - Xcut (own line enters as a line, NEXT follows the real dialogue) already reached 0.72 frozen / 0.80 live held-out on the symbolic mixed tutor (SIMULATED, review 2). RAL reached 0.83 / 0.80.
- **Proposed replacement** (ASSUMED; tested by E1):
  1. **The full stream with a SELF tag.**
     - My utterance enters the thought like any other line: the same entry gain, displacing old content (×0.3).
     - It is tagged by the efference copy (prior P3) and also enters the hippocampal NEXT chain, so replay follows the real order.
     - A heard line is then learned in its **true** context. "Moving on" after my answer teaches "what follows my answer", which the SELF-tagged features absorb, as error-driven learning does (Rescorla–Wagner blocking).
  2. **Self-caused input teaches the policy, not the world model.** My own lines update only through δ (L2; this is the existing R6 actor). It mirrors the brain's cancellation of self-generated sensation (Blakemore, Wolpert & Frith 1998; Eliades & Wang 2008). The `M_listen` gate becomes this general routing rule.
  3. **A learned question predictor replaces the 12/6-tick windows.**
     - It is an instance of O1 (the same learner class as P_oth, `agent_think.py` SparsePR).
     - Its target: "another speaker, not the line's speaker and not me, starts within the transition window after line L". Targets come from overheard routines and caregiver chat, so my own compulsive replies cannot contaminate them.
     - Answer-learning commits with weight 1 − P̂(L).
     - This differs from the rejected `ral_q`, whose targets were sparse reward labels and which misread 你好.
  4. **No reward exemption.** The agent speaks after 79–87 % of corrections because of its policy (urgency `u_R1` = 1). In the core, every utterance passes the same value gate (M5 with O2). Unrewarded speech right after a reaction is extinguished, and the recast then commits under item 3.
- **What stays innate:** the SELF tag and "imitate other speakers, not myself" (P3). Neither can be learned from prediction, because they define what counts as evidence.
- **Removed if E1 passes:** `ral_W`, `ral_W_other`, `ral_T`, `ral_cont='r'`, `ral_cut`, and the `ct_own` = 0 special case.
- **Falsifier:** E1.

### 4.2 The value loop's social features (`v21/agent_v21.py` AgentV21: `soc_feats`, `val_veto`, `dip`, `act_T`)
- `soc_feats` = "someone spoke within 300 ticks" plus "something appeared or disappeared in silence within 60 ticks". These are hand-picked traces of two generic channels: voice onset, and eye change (the latter is already what innate rule #23 computes).
  - **Replacement:** motif M4, a trace bank at fixed log-spaced time constants (3, 10, 30, 100, 300, 1000 ticks) on every area's output. V and P_oth read all traces and learn which matter.
  - The same bank supplies the time cells (τ ≤ 30, innate #9), `act_T` (eligibility duration), the 12-tick quiet moment and the 50-tick reaction window (CITED: Laplace-domain time cells, Shankar & Howard 2012; Howard et al. 2014).
- **Dips** are just TD. A V(t) that reads the trace of my own utterance predicts the reaction, so a missing reward gives δ < 0 with no special case.
- **The veto threshold 0.2** becomes the NoGo pathway of one BG gate, Go vs NoGo learned from ±δ (Frank 2005; Collins & Frank 2014).
- **Kept as prior P4:** coarse innate social detectors (person present, arrived, left, addressed me). IDEAS #17 asks for them so that the innate drive's input is reliable.
- **Removed:** 3 features with their windows, the dip special case, `val_veto`, `act_T`.
- **Cost:** many more inputs to V, so V learns more slowly. That matters because the empty-room effect already took most of a life to develop.
- **Falsifier:** life seeds 5 and 6 must keep v2.1's gains: empty-room 你好 ≤ 15 per tenth, social ≥ 0.30, appropriate ≥ 0.38. If they fail, the three features stay as named P4 detectors (budgeted, not per task).

### 4.3 The hearing front end's triggers (`hearing/hear.py` SoundGate; `agent_h.py` AgentH, `h_*`)

| key | generic form |
|---|---|
| `h_gate` (μ/ν floor, 4ν/7ν hysteresis, trim, 600-tick stuck rule), `h_level`, `h_denoise` | **M1:** every area adapts and divisively normalises its own input (subtract a running mean, divide by a running deviation). Level invariance is the defining property of divisive normalisation (Carandini & Heeger 2012); the auditory evidence is the source already cited (Dean, Harper & McAlpine 2005). "Sound present" = normalised energy above one area-general set point. |
| `h_conf`, `h_nocreate` (and the vigilance rule) | **precision-weighted plasticity:** recruitment and learning rate scale with ν, the area's confidence (Yu & Dayan 2005; Feldman & Friston 2010). |
| `h_relock` | part of **P2** (the syllable-rate oscillator, Giraud & Poeppel 2012): its phase is chosen by best recognition, i.e. the parse with the lowest prediction error. |
| `h_vbound` | **boundary = silence or a surprise peak** of the voice predictor (event segmentation, Zacks et al. 2007; transitional-probability dips, Saffran, Aslin & Newport 1996). The same rule replaces innate #14 (segmentation by silence). |

- **Risk** (ASSUMED): surprise-based segmentation needs a good predictor before it can work, so P2 stays innate.
- **Falsifier:** the world2 ear tables (σ 0 / 0.5 / 1, 16 bodies heavy chatter). Near recall must stay ≥ 0.9, junk categories must not exceed hearing's, and the clean bench J must stay within noise.

### 4.4 Other parts
- **Meaning RI** is the instar rule L3 applied to an item's context. It becomes the rule of *every* item-coded area, not one module.
- **The DG's meaning cue** becomes "the DG key is the whole cortical state", not hand-picked groups.
- **The `say_rec` / `say_loc` routes** become one M5 copy gate whose candidates are the active items of every area.
- **The fixed thought steps** (`core_think.py` ThinkCore) become M5 thought operations learned by L2. This is the open, hardest part.

---

## 5. The proposed core (P-V-M)

### 5.1 Objectives
- **O1: predict the next input everywhere.**
  - Each area predicts its own next input from the traces of its inputs.
  - The top (PFC) predicts the whole next multimodal input: heard items with a speaker tag, my own voice, the eye, the body, and the reward channel.
  - Speaking = releasing the predicted next own-voice input through the inverse model (ideomotor / active-inference reading; Hickok, Houde & Rong 2011). Speech stays caused by thinking.
- **O2: predict discounted innate reward (TD), and choose actions and thought operations that raise it.**
- **The store** (not an objective): every event is written once, and recall is an action.

### 5.2 One rule template, three instances
Δw_ij = η · M · a_i · b_j, where a_i is the presynaptic trace (from M4) and M is a modulator.

| instance | b_j | M | replaces |
|---|---|---|---|
| **L1 delta / prediction** | target − prediction at the unit's teaching input (dendritic mismatch; Urbanczik & Senn 2014), normalised by \|A\| | 1, or ν | R2 `Delta`, `RecDelta`, R8 `SparsePR`, R7 `Outstar` (a single-input delta rule), the critic (L1 with a TD target) |
| **L2 three-factor** | the eligibility tag of the selected unit (feedback from the chosen action, AuGMEnT-style) | δ | R6 actor, BG gates, V_go, thought operations |
| **L3 competitive Hebbian** | post activity with instar normalisation y_j (x_i − w_ij), k-WTA, homeostatic set point; one-shot recruitment of the least-used unit when nothing matches (η = 1 in the hippocampus) | ν | R5 `Templates`, R3 / `RandMix` instar + pruning, R4 `Hippo`, R1 `OneShotChain`, R11, meaning `MeanCodes`, `VoiceCats` |

### 5.3 Broadcast signals (two, i.e. one more than today's rule allows)
- **δ:** the global reward-prediction error (DA-like).
- **ν:** per-area surprise or precision, computed from the area's own error (NE/ACh-like). It sets learning rate, recruitment, episodic write strength and replay timing ("quiet" = low ν).
- **Justification:** Doya 2002; Yu & Dayan 2005.
- **Cost:** two scalars per tick, plus a trace per plastic synapse. RAL's `RecDelta` already stores such traces.

### 5.4 Motifs (content-free, the same everywhere)
- **M1:** a sparse k-WTA area with a fixed active fraction, divisive normalisation and adaptation (the bench's binary Region, DG k-WTA, the thought's ≤ 12 items).
- **M2:** random-at-birth sparse projections per the wiring plan, overproduced, with use-it-or-lose-it pruning (randmix `c_flat`).
- **M3:** recurrent self-excitation plus fatigue, which holds content (the continuous thought, favourites as tuned attractors).
- **M4:** the trace bank at fixed time constants.
- **M5:** a basal-ganglia selector with Go/NoGo and evidence accumulation with urgency (Cisek et al. 2009) over **all** candidates:
  - motor programs;
  - copy routes from any area;
  - thought operations: gate into the thought, hold, recall, imagine, attend;
  - speech release.
- **M6:** the hippocampal store: DG separation, one-shot cells, cue completion, awake replay when ν is low, prioritised by prediction error (Mattar & Daw 2018).

### 5.5 Innate priors (budget 6; each with the brain's reason)
- **P1 Body and brainstem scaffold:** muscles, voice physics, syllable beat, program WTA, breath. Vocal pattern generators are brainstem circuits (Jürgens 2002).
- **P2 Syllable-rate oscillator:** theta entrainment (Giraud & Poeppel 2012).
- **P3 Efference self-tag**, with "self-caused input teaches the policy, not the world model" (Blakemore et al. 1998; Eliades & Wang 2008).
- **P4 Drives:** hunger, and prosocial reward (caregiver pleased or praising; distress negative), with coarse social detectors (IDEAS #17; newborn face and voice preferences: Johnson 2005; Vouloumanos & Werker 2007).
- **P5 Orienting and turn reflexes:**
  - orient to sound onset and to joint attention (follow pointing and gaze);
  - one turn brake: do not start while another is talking, and yield on overlap.

  Turn timing itself develops with experience (Hilbrink, Gattis & Levinson 2015), so the 7-tick window and reflex 7b become learned via O1/O2.
- **P6 Wiring plan, global size s, set points and the plasticity spectrum:**
  - set points = active fractions, time constants, normalised thresholds;
  - plasticity spectrum = hippocampus fast, cortex slow, instincts about 0; this includes metaplasticity (Benna & Fusi 2016) and critical periods (Hensch 2005).

### 5.6 Mapping to code, and what is removed (removal assumes the named test passes)

| v2.x part (file, class / key) | becomes | removed |
|---|---|---|
| GEN R2 (`synth/rules.py` Delta; `core_v21.RecDelta`) | L1 | nothing (already generic) |
| P_oth, V_go (`agent_think.SparsePR`); critic and actor (`rules.Critic`, actor in GEN) | L1 / O2 critic, L2 actor | V_go as a separate gain (already a passenger) |
| MIX R3 (`rules.MixDict`) | M2 (`randmix/randlayer.RandMix` c_flat) | MIX recruitment (v2.2 is merging this) |
| meaning RI (`meaning/meanlayer.MeanCodes`), DG cue (`mcore.MeanCore`) | L3 in every item area; M6 key = whole state | the module-specific code path |
| `rules.Templates` R5, VoiceCats | L3 + ν | vigilance 0.75 → a set point |
| `rules.Hippo` R4, recall R10 | M6 | 12-tick / 8-episode / uniform schedule → ν-triggered and prioritised (test priority vs uniform; tune found uniform better than strength-weighted) |
| continuous thought (`core_synth.SynthCore`, `ct_*`) | M3 + M5 input gate; own line enters with the same gain, SELF-tagged | `ct_own` = 0 special case |
| thought steps (`core_think.ThinkCore`), intention (`ThinkAgent` θ_I, g0/u per thought type) | M5 operations learned by L2; one accumulator | the fixed step order; the reply/scene split of constants |
| RAL (`core_v21.RoleCore` `ral_*`) | §4.1 | windows, exemption, NEXT cut (E1) |
| `say_rec`, `say_loc` (`gateR`) | M5 copy gate over all areas | the two dedicated routes |
| value loop (`agent_v21.AgentV21`) | §4.2 | `soc_feats`, `val_veto`, `dip`, `act_T` |
| hearing (`hear.SoundGate`, `agent_h.AgentH`) | §4.3 | 6 of the 7 kept `h_*` keys' rules (`h_relock` stays inside P2) |
| innate #3, #5, #6, #7, #7b, #8, #9, #10, #13, #14, #17, #19, #22, #23 | P5 / M4 / M5 / ν set points / M1 | the per-mechanism constants |
| #1 body, #4 own voice, #12 clock, #16 retinotopy, #18 pair schema | P1, P3, P2, P6 (wiring) + P5 | — |
| #15 babbling read from `world.voice` offline | L3 one-shot store of my own (sound, muscle set) from real babbling + L1 forward model / inverse (target-prop form) | the privileged shortcut |
| #11 EAR-0 / FEAT0 | M2 layer 0 (fixed) | — |

---

## 6. Why it is general, where it fails, and the biggest risk

### 6.1 Why it is general (ASSUMED argument; parts PROVED or CITED)
1. **Coverage.** Every v2.x ability is an instance (§5.6), so the core is at least as expressive *by construction*.
2. **Closure.** A new ability needs one of:
   - new structure in the stream (learned by O1 / L3);
   - new combinations of actions (M5 + L2), including thought operations;
   - a new value, which comes only through the fixed P4.

   None needs code, because every motif is content-free and M5 selects over every area's outputs instead of over named routes.
3. **The class is universal** (§1.1), and sequential composition through the thought is universal even for linear predictors (Malach 2023).
4. **It matches the brain's pattern:** one cortical algorithm (rewiring experiments), a few circuit types, about 4 neuromodulators, and Leabra's breadth.

### 6.2 Where it would fail
- **(a)** High-order structure that is neither factorisable nor taught step by step (parity-like). Width is exponential (§1.3), and no local gradient builds the codes.
- **(b)** Long-horizon credit with sparse reward. L2 is slow, and eligibility traces stop at the longest time constant in M4.
- **(c)** Abilities whose evidence is rare within one life. Only priors help there.
- **(d)** Deep codes in a world that keeps changing, since upper layers chase drifting lower layers.
- **(e)** Systematic binding beyond seen fillers, which needs learned indirection (Kriete 2013).
- **(f)** One-shot use. The store holds the answer, but the gate must learn to use it (our oneshot 0 vs 0.60 forced open).

### 6.3 The biggest theoretical risk
**Abstraction without backprop in one unbroken life.** L1 and L2 train readouts and actions *on top of* representations, but L3 forms those representations without knowing what the readouts need. If abilities need task-shaped intermediate codes, the core needs one of three things:
- exponential width (PROVED cost);
- stacked local layers that stay stable with learning always on (unproven; our records show collapse and drift);
- learned multi-step thought (RL-slow; not yet shown to reach content).

In that case "bigger" buys memorisation, not generalisation, which is the opposite of principle "bigger is better once sound". Data per life (§1.4) is the second risk, and it is fused with this one: width-based abstraction also needs every tuple to be seen.

---

## 7. One global size factor s, and the scaling curve (coordinator's note)

| quantity | under s | reason |
|---|---|---|
| units N in every area (ear upper layers, random conjunction layer, meaning codes, thought population, DG, CA cells, critic / actor / gate units, P_oth / V features) | × s | the whole model, never one part |
| active count k per area | × s (fixed **fraction**) | each unit stays active as often as before, so per-unit training is unchanged |
| item code width (16 of 512) | 16s of 512s | the same overlap statistics |
| fan-in per branch | fixed (3) | expected active inputs per unit stay fixed, so thresholds stay fixed |
| step per synapse η/\|A\| | ∝ 1/s; total change per event fixed | more averaging, less noise |
| hippocampal forgetting 1 − γ | ∝ 1/s | storage horizon ∝ capacity |
| items held in the thought | fixed (assemblies grow × s) | a context-length set point (IDEAS #15) |
| motor program pool | grows with the world's vocabulary | — |
| fixed | time constants, set points, the plasticity spectrum, rules, motifs, broadcasts, priors, the body (32 muscles), ear window | the world's tempo and the design, not the model's size |

- **Experience that must go with it.**
  - World diversity (lexicon, object kinds, people, rooms, frames) and life length both grow with s. Exposures per item then stay constant while novel content fills the extra capacity.
  - This is the analogue of Chinchilla's D ∝ N (Hoffmann et al. 2022). Our "data" is *novel content*, not ticks: for a local learner the binding quantity is exposures per used conjunction.
  - Compute per tick grows with s, so total compute grows with s².
- **Predictions** (ASSUMED; E3 tests them):
  - **P-S1.** Fixed world, fixed active *count*: J falls with s, because per-unit exposures fall by 1/s. This explains meaning's −0.11 at 4×.
  - **P-S2.** Fixed world, fixed fraction, all areas scaled together: J flat within noise for s ≥ 1 (the knee is already passed, as randmix 4×/16× showed).
    - Unit-level noise tolerance should improve with the population size (ASSUMED).
    - Item-level mishearing should not improve (it needs O1 in hearing), as found.
  - **P-S3.** Along the diagonal (world and life × s): seen accuracy and once-heard retention stay flat. **Held-out (frame or class) accuracy rises as a power of s only if L3 learns shared codes**, because more fillers per class give better codes. A memorising core stays flat. This slope is the measure of soundness.
  - **P-S4.** Growing the world without growing s: once-heard retention falls once load exceeds capacity (today 10k cells, half-life ≈ 1,400 lines).

---

## 8. Cheap experiments before a big build

All use existing harnesses, run locally on the CPU in minutes to hours, under 700 MB.

### E1. Role from prediction: can RAL's hand timing go? (the experience-cost risk)
- **Harness:** `v2/v21/regress/gen_probe_v21.py`. Copy it into a new folder; it uses RoleCore.
  - 4 frames × 12 fillers;
  - recast and mixed tutors;
  - 900 dialogues, 3 seeds;
  - frozen and live held-out;
  - plus 900 untutored dialogues for drift.
- **Arms:**
  - ARCH_v2 (0.00);
  - RAL (reference 0.81 / 0.83 frozen);
  - **S0:** own lines enter the thought and the NEXT chain SELF-tagged; own lines teach only through δ; no windows;
  - **S1:** S0 + the learned question predictor of §4.1;
  - **S0-untagged:** a control that should reproduce the `ct_own` failure.
- **Also report** held-out at 300 / 600 / 900 dialogues: the experience cost.
- **Pass:** S0 or S1 within 0.1 of RAL on frozen held-out, drift loss ≤ 0.1 after 900 untutored, reached in ≤ 2× the dialogues.
- **Fail:** RAL's timing is a needed prior. It joins P5 as one "turn structure" prior, still per mechanism, not per ability.
- **Then:** if it passes, run 5-seed staged and the life pair (seeds 5 / 6) with v2.2's correction target.

### E2. Depth in space vs depth in time (the biggest risk)
- **Harness:** extend the class probe `meaning/gen_probe_mean.py` to two levels:
  - 24 fillers → 6 classes (fruit, vegetable, snack / toy, animal, vehicle) → 2 superclasses (edible / not);
  - one class frame each (X甜不甜, X要不要煮, …);
  - cross-class frames that link classes only at the second order (X洗一洗 for fruit + vegetable; X放哪里 for toy + vehicle);
  - 能不能吃 taught for 4 classes;
  - vegetable and vehicle are **never** asked it (zero-shot whole classes).
- **Arms:**
  - (a) random layer + delta readout;
  - (b) + one RI meaning layer;
  - (c) + a second stacked local layer (RI on first-layer codes);
  - (d) **time:** the stream also contains statements such as X是蔬菜 / 蔬菜能吃; the answer comes from two thought steps (hippocampal recall X→class, then readout class→answer);
  - (e) an online one-pass two-hidden-layer MLP trained by backprop. This is a scratch reference only: is the information in the stream?
- **Measures:**
  - level-1 held fillers and level-2 held classes;
  - code drift over a 10× longer stream (meaning's metric: 0.043 for RI);
  - code collapse (code entropy).
- **Pass:** (c) − (b) ≥ 0.15 on level 2 with drift ≤ 0.1, **or** (d) ≥ 0.7.
- **Fail:** (c) and (d) both ≤ (b) + 0.05 while (e) ≥ 0.7. The risk is then real: add a depth mechanism (DeepLeabra-style next-input plus phase, or BrainProp tags) before building. If (e) also fails, the task is at fault, not the core.

### E3. The global size factor (the scaling theory)
- **E3a.** Meaning module test and the short protocol (nS 500), s ∈ {1, 2, 4}, two seeds.
  - Arms: fixed active count vs fixed fraction with all areas scaled.
  - Predicts P-S1 vs P-S2 (−0.11 reproduced vs flat).
- **E3b.** The symbolic diagonal:
  - fillers 12s and dialogues ∝ s;
  - ambiguity tuned so that s = 1 sits at 0.6–0.8 held-out (avoids the module test's ceiling of 1.00);
  - arms (a) vs (b).
  - Predicts a rising held-out curve for (b) and a flat one for (a) (P-S3).
- **Falsifier:** a flat held-out curve for (b) means more size buys only memorisation. Scaling must then wait for E2's fix.

### Acceptance test for the whole core
**Held-out abilities.** Freeze the code. Then run world2 task types that no part was built for:
- fetch, give, come;
- 你拿着什么 (what are you holding);
- 这是谁 (who is this);
- 这是哪里 (where is this);
- 大不大 (is it big).

**Success:** above `Lookup` on new-situation items for at least half of the new types, with zero code changes.

---

## 9. Sources (verified this round, abstract level)
- [Bellec et al. 2020, e-prop](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7367848/)
- [Laborieux & Zenke 2022, holomorphic EP](https://papers.nips.cc/paper_files/paper/2022/hash/545a114e655f9d25ba0d56ea9a01fc6e-Abstract-Conference.html)
- [Pinchetti et al. 2024/25, predictive-coding benchmark](https://arxiv.org/abs/2407.01163)
- [Bartunov et al. 2018, scalability of biologically motivated algorithms](https://arxiv.org/html/1807.04587)
- [Hinton 2022, forward-forward](https://arxiv.org/pdf/2212.13345)
- [Halvagal & Zenke 2023, LPL](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10620089/)
- [Illing et al. 2021, CLAPP](https://proceedings.neurips.cc/paper/2021/hash/feade1d2047977cd0cefdafc40175a99-Abstract.html)
- [Journé et al. 2023, SoftHebb](https://arxiv.org/pdf/2209.11883)
- [Dabagia, Papadimitriou & Vempala 2024](https://proceedings.mlr.press/v237/dabagia24a.html)
- [Malach 2023, next-token predictors are universal learners](https://arxiv.org/html/2309.06979v3)
- [Pozzi, Bohte & Roelfsema 2020, BrainProp](https://proceedings.neurips.cc//paper/2020/file/1abb1e1ea5f481b589da52303b091cbb-Paper.pdf)
- [O'Reilly et al. 2021, deep predictive learning](https://arxiv.org/pdf/2006.14800)
- [Wang et al. 2018, PFC as meta-RL](https://www.biorxiv.org/content/10.1101/295964v1)
- [Launay et al. 2020, DFA scales](https://papers.nips.cc/paper/2020/hash/69d1fc78dbda242c43ad6590368912d4-Abstract.html)
- [Song et al. 2024, prospective configuration](https://www.nature.com/articles/s41593-023-01514-1.pdf)
- [Ernoult et al. 2022, scaling DTP](https://proceedings.mlr.press/v162/ernoult22a.html)
- [Cui, Ahmad & Hawkins 2016, HTM online sequence learning](https://arxiv.org/pdf/1512.05463)
- [Frank 2023, the data gap](https://sites.socsci.uci.edu/~lpearl/colareadinggroup/readings/Frank2023_ChildrenLLMs.pdf)
- [Murray 2019, RFLO](https://elifesciences.org/articles/43299)
- [Rombouts et al. 2015, AuGMEnT](https://pmc.ncbi.nlm.nih.gov/articles/PMC4351255)
- [Kriete et al. 2013, indirection](https://pmc.ncbi.nlm.nih.gov/articles/PMC3799308)
- [Werfel, Xie & Seung 2005](https://proceedings.nips.cc/paper_files/paper/2003/file/f8b932c70d0b2e6bf071729a4fa68dfc-Paper.pdf)
- [Clark & Chouinard 2003, adult reformulations](https://cambridge.org/core/journals/journal-of-child-language/article/adult-reformulations-of-child-errors-as-negative-evidence/AC1DA451B23BDDCE589273617D8B4F4C)

All other citations are well-known primary papers cited from memory with author and year: Siegelmann & Sontag 1995; Maass et al. 2002; Blum & Rivest 1992; Wolpert 1996; Zador 2019; Papadimitriou et al. 2020; O'Reilly 1996; O'Reilly et al. 2012; O'Reilly & Frank 2006; Doya 2002; Yu & Dayan 2005; Frémaux & Gerstner 2016; Gerstner et al. 2018; Lillicrap et al. 2016; Nøkland 2016; Scellier & Bengio 2017; Whittington & Bogacz 2017; Millidge et al. 2022; Rao & Ballard 1999; Lee et al. 2015; Sutton 1988; Dayan 1993; Stachenfeld et al. 2017; Hoffmann et al. 2022; Carandini & Heeger 2012; Dean et al. 2005; Zacks et al. 2007; Saffran et al. 1996; Sharma et al. 2000; von Melchner et al. 2000; Mattar & Daw 2018; Frank 2005; Collins & Frank 2014; Benna & Fusi 2016; Hensch 2005; McClelland et al. 1995; Kumaran et al. 2016; Hilbrink et al. 2015; Blakemore et al. 1998; Eliades & Wang 2008; Urbanczik & Senn 2014; Lengyel & Dayan 2007; Blundell et al. 2016; Pritzel et al. 2017; Gershman & Daw 2017; Todd et al. 2008; Guenther 2006; Rolf et al. 2010.

---

## summary_zh
- **"学会任何事"本身做不到。** 神经网络能"算"任何东西，但"学会"需要先天偏好。
- **可以做到的目标：** 机制数量不随能力增加。检验办法是冻结代码，在 world2 放进从没为之写过代码的新任务（拿东西、给东西、这是谁……），看能否学会。
- **现状：** 约 60 个具名机制、243 个配置键。
- **提议的核心"预测-价值-记忆"：**
  - 两个目标：各区预测下一刻输入；用 TD 预测并争取先天奖励。
  - 一个规则模板的三种用法：误差修正、三因子、竞争 Hebb。
  - 两个广播：奖励误差 δ、各区"意外度" ν。
  - 六个通用结构和六个先天先验。
  - v2.x 的各部件大多可归为它们的实例：听觉前端那几个开关 → 各区自适应和归一化；社交特征 → 多时间常数痕迹 + TD；"谁回答谁"的时间窗 → 自我标记的完整对话流 + 学到的"这句是不是问题"。
  - 不能学掉的：区分自己和别人、只模仿别人（先天）。
- **最大风险：没有反向传播，在一生连续学习中能否学到抽象的中间表示。**
  - 只靠加宽是指数代价：覆盖全部三元组合要约 480 万单元，四元约 1.5 亿。
  - 叠加的局部层在我们自己的实验里会漂移或塌缩。
  - 第二风险：一生的经验太少。一次台架训练只听到约 2–4 万个音节，一段 10⁶ tick 的生活约 10 万，远少于五岁儿童。
- **放大的规则：** 一个全局系数 s。所有区的神经元数和激活数按 s 同比放大（激活比例不变）；时间常数、设定点、规则、身体不变；世界的丰富度和寿命也按 s 增长。
- **预测：**
  - 世界不变时，只增加单元而激活数不变 → 变差（解释了 4× 时 −0.11）；按比例整体放大 → 持平。
  - 世界和寿命同步增长时，见过的项目持平；没见过的项目只有在学到共享编码时才会随 s 上升。这条斜率就是"架构是否健全"的检验。
- **大规模构建前先做三个小实验：**
  - E1：取消人为时间窗的角色学习；
  - E2：空间深度（叠层）对比时间深度（多步思考），两级类别任务；
  - E3：全局放大系数。
