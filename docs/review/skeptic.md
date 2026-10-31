(Saved by the coordinator from the reviewer's final reply; the harness blocked the reviewer's own write.)

# Skeptic review of the whole architecture (framework + ARCH_v1): a systems engineer's first-principles view

The harness blocked writing `report.md`, so the full report is below. The scripts and their outputs are saved in `arch/review/skeptic/` (§9).

Labels: **PROVED** = argument with numbers or code reading; **SIMULATED** = numpy run in `arch/review/skeptic/` (file and numbers given); **CITED** = paper and finding; **ASSUMED** = my judgement, untested.

---

## 0. Verdict in ten lines

1. **The body works by construction, not by learning.**
   - The 0-error turn-taking, articulation and stopping come from about 30 hand-set units and rules, with several dozen constants.
   - All of them are tied to a noise-free world with 3-tick syllables.
   - The toy proves there is no implementation bug. It says nothing about learning (PROVED).
2. **The brain areas that learn all the time have no effect on behaviour.**
   - These are the ear cortex, the hippocampus with its returns, and the motor cortex with its paths.
   - When I switch all of them off, every decision weight stays bit-identical and 30/30 free replies are identical.
   - They use 2/3 of the toy's compute (SIMULATED).
   - Behaviour is steered by six lookup tables. Five of them learn only when the teacher drives them.
3. **The content numbers measure echo and lossy memorisation, not meaning.**
   - Never-heard lines score 0.12. Simply repeating the heard line, with no learning at all, scores 0.24 on the same set.
   - With an echo-free version of the metric, every learner I tried is at chance at 1k dialogues: T, retrieval, association counts.
   - At 29k dialogues the best of them reach only AUC 0.52–0.53 (SIMULATED).
4. **On this corpus, at this scale, no architecture can show meaning.**
   - 1,000 dialogues = 19,396 chars, about one day of a toddler's input.
   - B4 and B8 need a different instrument (§5.3, Stage 0).
5. **The memory side is the bottleneck, not the thought code.**
   - An "episode cell" store keyed on v1's existing T code recalls the whole reply 0.98–0.99 after one exposure.
   - It still recalls 0.97–0.98 after 10k more dialogues (SIMULATED, on the content specialist's own probe protocol).
   - v1 with fast and slow memory (MEM): first syllable 0.25–0.34 after one exposure, 0.39–0.40 at +10k dialogues; whole-reply top-1 0.45.
6. **One clause of the learning rule is the root cause of the clockwork.**
   - The back term is "state-based": a source that stays on counts as firing again. This creates the timing law.
   - With an onset-based back term (depress only when the source turns on again), a held thought can be learned.
   - The changed rule still refuses ping-pong and reverse order (SIMULATED).
7. **The rule also lacks four things a general learner needs.**
   - Credit across delays, normalised statistics, learned inhibition, and learning from consequences (a broadcast modulator).
   - Each missing piece was replaced by a fixed patch.
8. **Capacity is structurally small.**
   - With 96 synapses per neuron and 10–40 % of units active, a sheet stores only about 30–300 specific transitions.
   - With ≤5 % active and 10× more synapses, it stores ≥1,000 (SIMULATED).
   - The brain uses sparse codes and 10³–10⁴ synapses per neuron.
9. **The PFC worry is right, but depth is not what is missing.**
   - v1's PFC is a fixed hash of raw sound, read once.
   - What is missing: a learning signal, learned inputs, and a gate that decides what to hold and when to output.
   - A multi-layer loop updated in synchronous ticks will alternate with period 2 again (PROVED).
10. **What to do, in order.**
    - Fix the instruments and the learning rule first.
    - Then give memory and representation learning a real path to behaviour: episode store → predictive cortex with replay → gated PFC and basal ganglia (BG).
    - Keep v1's clockwork as an overridable "brainstem" reflex and retire it piece by piece.

---

## 1. New evidence in this report (all files in `arch/review/skeptic/`)

| id | check | result |
|---|---|---|
| S1 | `ablate_passengers.py` → `.out`. ARCH_v1 toy (MEM + echo slot), seed 0, 120 dialogues. FULL vs STRIP (`ear_learn`, `hip_on`, `returns`, `mctx_on` all off). | Max \|diff\| **0** for P→Y fast, P→Y slow, Hc→Y, Hc→END, E→Y, FEAT0→Z and the program table. Free replies identical on **30/30** never-heard lines. Speed **14.6 vs 4.9 ms/tick**. Also PROVED by code: the boundary scores read only P (from T, from ear layer 0, which gets no learned or return current), Hc, and the slot (FEAT0→Z). See the `archv1.py` step function, lines ≈396–470. |
| S2 | `baselines_corpus.py` → `.out`. run.py's exact 500 never-heard lines and its 20-way metric, plus an **echo-free** variant: the heard line's chars are removed from the output and from all 20 candidate replies. | Echo: 0.240 / AUC 0.646 (echo-free: 0.050 / 0.500). Last heard char: 0.113 / 0.540. Text retrieval (TF-IDF, reply of the nearest stored prompt): 0.055 / 0.520 at 1k → 0.094 / 0.541 at 29k; echo-free 0.049 / 0.501 → 0.077 / 0.518. PMI association (top-5, no echo), echo-free: 0.072 / 0.508 → 0.092 / 0.530. Reply chars that occur in the heard line: 13.3 %. |
| S3 | `retrieval_T.py` → `.out`. Episode cells keyed on the **real T code**, using the content specialist's recorded front-end runs. 3 seeds, 1,500 never-heard lines, 1k / 3k / 10k dialogues. | Never-heard: top-1 0.054–0.082, AUC 0.510–0.526. Echo-free AUC **0.499–0.506**: T similarity carries no reply information beyond echo. |
| S4 | `episode_familiar.py` → `.out`. The same store on the content specialist's familiar-probe protocol (200 probes, 6 exposures, 14.6k dialogues, 2 seeds). | Whole reply exactly right **0.98–0.99 after ONE exposure**. Retention **0.97–0.98 at +10k** (n ≈ 255 per seed). For comparison, CONTENT MEM: first syllable 0.25–0.34 after 1 exposure, 0.90 after 5, 0.39–0.40 at +10k; whole reply 0.45. |
| S5 | `rule_v2_test.py` → `.out`. net.py's rule vs the same rule with an onset-based back term plus slow "source on, target did not rise" depression. 200 scripted trials. | Held thought (source on ticks 0..9, target pulse at 6): **0.000 → 0.106**. Fresh pulse: 0.712 in both. Constant source: 0.000 → 0.058 (≈ base rate). Ping-pong: 0.005 in both. Reverse order: 0 in both. |
| S6 | `capacity.py` → `capacity.out`, `capacity_nobudget.out`. Order-rule associative memory, N = 2,000, random transitions. | **96 synapses per neuron:** recall is near chance by 100–300 stored transitions at every activity level (a = 0.4: 0.45 at M = 100, chance 0.40; a = 0.05: 0.66 at 100, 0.37 at 300). **960 synapses, a = 0.02, no budget:** 1.00 at 300, 0.91 at 1,000. **The per-target budget** costs 3–5× (a = 0.05: 0.89 → 0.30 at M = 300). **With 20–40 % active**, 10× more synapses barely help (≈30–100 transitions). |

---

## 2. Do the claimed results show what they say?

| claim | what it actually shows | verdict |
|---|---|---|
| "Body works: 0 errors in turn-taking, articulation, stopping, B1" | F, GO, CD, beat, programs and D/I/M are fixed logic, so 0 errors is a theorem; the toy confirms there is no implementation bug. Two weak points: **(1) A noise-free world is assumed.** F = [0.01·E − 10·ΣCD ≥ 0.1], so a constant background of 10 level-units (< ½ the minimum speech energy of 22) keeps F on forever. GO never fires and the model is mute (PROVED). **(2) Strict alternation is assumed.** When the partner pauses 5–9 ticks mid-turn, the model already talks over them: 29–75 mouth ticks per 6 trials. | True, but says nothing about learning, and brittle |
| "B2: T decodes 20/20" | T is a fixed random expansion of ear layer 0 with a self-hold: a recency-weighted hash of the sound. Lines ending in the same char get the same first syllable 92.8 % of the time (learnability verifier). | Identity, not "thought" |
| "Familiar lines (6×): 0.57" | This is lossy memorisation of 40 pairs. The adopted default MEM scores **0.425**, worse than base. By design, no line-specific source reaches later syllables ("read-once"). An episode store on the same T gives **0.98–0.99 after one exposure** (S4). | The memory mechanism is the bottleneck |
| "Never-heard 0.12 with echo route" | Below the zero-learning "repeat the line" policy (**0.240 / 0.646**) and about equal to "last char" (0.113 / 0.540). Echo-free it is at chance: first syllable ≈0.055 vs ≈0.040 shuffled at 1k–10k (CONTENT §3.1). | Measures echo; the metric rewards parroting, and the design was steered toward it |
| "B7: real-time learning, stable, bounded" | It is stable because the decision paths are small gated tables, and the continuously learning areas are passengers (S1). Budgets are enforced only every 100 ticks: hippocampus 13–17 vs budget 2, ear 1.7 vs 1.0. Idle replay rewrites hip/ear weights (Δw 6.9 / 0.8 per 1,000 silent ticks) for no consumer. In free speech the decision paths are gated off (`g_now = (not gate) or Yext`), so **the deployed model never learns what to say from its own conversations** (PROVED). | "Real-time" only under teacher forcing |
| "The PFC loop drifted and alternated, so remove it" | In a pure two-layer synchronous loop, A(t) = f(B(t−1)) and B(t) = g(A(t−1)), so A(t) = f∘g(A(t−2)). Even and odd ticks are two independent copies, and any phase mismatch persists as period 2 (PROVED; cf. Bruck 1990, Proc IEEE: synchronous threshold networks end in fixed points or 2-cycles). The alternation was largely a modelling artefact. Removing the loop also removed the only place meaning could be learned. | Right for v1; wrong lesson for the PFC (§5.5) |
| "Hippocampus = fast memory" | It has no causal role (S1). Fan-in 96 and γ 0.7 per 100 ticks give a span of a few dialogues (S6). | Intent not realised |
| "Timing law: a learned source must be a 1-tick pulse" | True for net.py, but it comes from the **state-based** back clause only (S5). It is not a law of order-based learning. | Root of most fixed mechanisms; removable |

---

## 3. Differences from the real human brain, and whether each matters for the final goal

| aspect | real brain | ARCH_v1 | matters? |
|---|---|---|---|
| Scale, connectivity, sparsity | 10³–10⁴ synapses per neuron; <5 % of A1 strongly active (Hromádka 2008, CITED) | Fan-out 96; ear layers 10–40 % active | **Yes.** Capacity ≈30–300 transitions (S6). Use sparse codes (≤5 %) and ~10× synapses where sequences or episodes are stored. Absolute size matters less. |
| Cortical microcircuit, dendrites | Layers (L4 input, L2/3 association, L5 output); apical vs basal compartments do prediction and coincidence detection (Larkum 2013, CITED) | Point units. "Drive sets how many fire, extra current sets which" is a crude two-compartment proxy. | **Partly.** Missing: a local predicted/unpredicted state. Active-but-unpredicted = surprise, and sequence context lives in which cells are predicted (Hawkins & Ahmad 2016, CITED). Cheap and binary-compatible. |
| Inhibitory cell types, inhibitory plasticity | PV / SST / VIP interneurons; plastic inhibition (Vogels 2011); learned cancellation of one's own sounds (Schneider 2018) (CITED) | k-WTA and tonic motor I. **No learned inhibition**: learned weights are always ≥ w0 ≥ 0. | **Yes.** No negative associations or exceptions; no learned own-voice cancellation, so F needs a timer and the model cannot yield. |
| Thalamus | Loops for every area; MD sustains and switches PFC states | Removed | **Low for now.** A self-hold plus a BG gate can cover it. |
| Basal ganglia | Tonic inhibition + disinhibition; dopamine three-factor learning of what and when; gates working memory (PBWM, O'Reilly & Frank 2006, CITED) | Fixed GO and beat; teacher-gated winner-take-all tables | **Yes.** This is where "when / what to say" should be learned from consequences. |
| Cerebellum | Forward and inverse models, timing, error-driven learning (Wolpert, Miall & Kawato 1998, CITED) | None. CD is a fixed 4-tick timer; the beat is fixed at 3 ticks. | **Yes for the final goal** (motor learning, robots, yielding); no for the toy. |
| Hippocampus | DG separates patterns; CA3 stores in one shot; indexing theory (Teyler & DiScenna 1986); replay consolidates into cortex (McClelland 1995; Kumaran, Hassabis & McClelland 2016) (CITED) | 1:1 mirror; forgets in ~300 ticks; **no behavioural role**; replay uncontrolled | **Yes.** One-shot episodes (S4: 0.98 / 0.97 at +10k) and replay-driven consolidation are how a real-time learner avoids forgetting. |
| PFC | Multi-layer recurrent; mixed selectivity (Rigotti 2013); attractor + synaptic working memory (Mongillo 2008); BG-gated update; top-down biasing (Miller & Cohen 2001); hierarchy (Badre 2008) (CITED) | T = fixed random expansion of raw layer-0 sound + self-hold; read once | **Yes.** But first: a learning signal, learned inputs, gating and top-down use. Depth comes after (§5.5). |
| Time and dynamics | Continuous, asynchronous; membrane integration; adaptation | Synchronous binary ticks; no membrane memory | **Partly.** Binary units are harmless. 2-tick synchronous loops give period 2 (PROVED). A leaky potential per unit gives integration, smooth holding and a natural eligibility trace. |
| Plasticity | STDP depends on rate: at high rates LTP regardless of order (Sjöström 2001). CA3 windows are symmetric (Mishra 2016). Plasticity over seconds, one-shot (Bittner 2017). Heterosynaptic and homeostatic plasticity. (CITED) | One order rule with a state-based back term, plus reserve, set point, budget, decay | **Yes, most important.** Held state → action onset is unlearnable; no delayed credit; no normalisation (copy weights are counts, CONTENT §3.4). |
| Neuromodulators | Dopamine (reward prediction error); acetylcholine (encode vs retrieve); noradrenaline (surprise, reset) (Hasselmo 2006; Yu & Dayan 2005; He 2015, CITED) | One binary "teacher drove this" gate | **Yes.** No learning from consequences, no novelty-gated encoding, no listen/speak mode. |
| Memory systems and sleep | Working, episodic, semantic, procedural memory; complementary learning systems; sleep replay and synaptic downscaling (Tononi & Cirelli 2014, CITED) | Fast/slow lookup tables; decay on an experience clock; replay that nothing uses | **Yes.** Decides lifelong learning: retention 0.39 at +10k; P→Y holds only 100–300 associations. |
| Innate vs learned | Brainstem pattern generators / PAG for vocal patterns (Jürgens 2002); infant turn-taking is early but shaped by contingent feedback (Hilbrink 2015; Goldstein 2003; Takahashi 2017) (CITED) | Body, turns and articulation are innate and **cannot be modulated**; representations fixed; only lookup tables learn | **Yes: the ratio is inverted.** Innate scaffolds are fine if learned systems can tune or override them. |
| Development | Babbling → imitation → words → combinations; synapse overproduction then pruning | No babbling; program units are created by teacher cues (a motor oracle) | **Yes.** The auditory→motor inverse model learned by babbling (DIVA, Guenther 2016) is how speech is learned without a puppeteer. The Z map is half of it already. |
| Body and world | Noisy; compositional phonology (Mandarin ≈400 base syllables from initials, finals and tones); coarticulation; interactive partners | Silence = exactly 0; each char is an arbitrary sound + unique arbitrary 4-muscle set; fixed 3-tick syllables; one partner; corpus replay (nothing responds to the model) | **Yes.** No sublexical structure → no generalisation in hearing or speaking. No noise → F is trivially perfect. No contingency → nothing can be learned from consequences. |
| Training regime | Order of 10⁴ words per day for 1–2 years before word combinations, with feedback and context (Hart & Risley 1995; Gilkerson 2017, CITED) | 1,000 dialogues = 19,396 chars (PROVED by count) ≈ one toddler-day; one exposure each; teacher-forcing in motor space | **Yes.** No learner shows meaning at this scale on open chit-chat (S2, S3). |
| Harmless simplifications | — | Binary units; no Dale's law; fixed early sensory layers (brain-like after critical periods); no glia; no explicit oscillations; a localist syllable inventory (if learned) | Harmless |

---

## 4. Problems of the whole architecture, ranked by how much they block the final goal

F = fundamental (needs a structural change); x = fixable (parameters or details).

| # | problem | evidence | F/x | what to do instead |
|---|---|---|---|---|
| 1 | **Learning and behaviour are decoupled.** The parts that learn all the time have no effect; the parts that behave do not learn. | S1; code | F | Every learned component must have a causal path to output, checked by ablation. First path: an episode store. |
| 2 | **The rule cannot learn what a general learner needs.** It cannot learn: held state → onset, effects after a delay, the probability of the target given the source, inhibition, features, or consequences. Each gap became a fixed patch: P = T∧GO, E = L∧R2, Hc delay lines, CD timer, END rules, teacher gate. | Timing law; S5; CONTENT §3.4 | F | **Rule v2**, five parts: (a) onset-based back term; (b) eligibility trace; (c) slow depression when the source is on and the target does not rise; (d) a broadcast modulator; (e) plastic inhibition through inhibitory interneurons (Dale-compliant). Unit-test each. |
| 3 | **No learned representation → no meaning → no B8 by design.** T is a fixed surface hash; ear feed-forward weights are fixed; nothing predicts. | First syllable ≈0.055 vs 0.040 shuffled at 10k; S3 echo-free AUC ≈0.50 | F | Predictive sparse cortex using the same rule, plus hippocampal replay into cortex, so lines that predict similar continuations get similar codes. |
| 4 | **The teaching signal is in motor space, and there is no contingency.** The teacher cues exact muscle sets and clamps END; free speech never learns. | ARCH_v1 §1.5; gate code | F | Babbling → inverse model → a teacher who only speaks; modulators from imitation match, feedback and surprise; an interactive scripted partner. |
| 5 | **The instrument cannot see meaning.** Echo dominates the metric. Non-echo ceilings are at chance at 1k dialogues and only +0.02–0.03 AUC at 29k for every simple learner. | S2, S3; 19,396 chars per 1k dialogues | F (for the plan) | Echo-free metrics; baselines on the same stream; a synthetic curriculum with held-out combinations and multi-turn binding (§5.3, Stage 0). |
| 6 | **Task-specific fixed mechanisms keep accumulating.** About 30 units/rules and several dozen constants are tuned to 3-tick syllables. They break easily: noise mutes F; mid-turn pauses produce 29–75 overlapping mouth ticks; there is no multi-turn context. And they hide what the learner cannot do. | ARCH_v1 §1.3–1.4, §6; stress test | F for generality, x for the toy | Keep a **generic** reflex layer (rhythm, turn gap, tonic inhibition) that a learned gate can tune or override. Every new fixed part must come with a plan for learning it. |
| 7 | **Memory capacity and code density.** K = 96 and 10–40 % active → tens to hundreds of transitions. P→Y holds 100–300 associations. MEM retention is 0.39 at +10k. The program pool is localist (4,096). | S6; learnability verifier; CONTENT §4 | x (structural inside areas) | Episode cells (S4). Sparse ≤5 % codes with ~1,000 synapses per neuron where sequences are stored. Forget through interference and sleep-time downscaling, not through budget rescaling every 100 ticks. |
| 8 | **Synchronous-update artefacts.** Period 2 in any pure inter-layer loop; it will recur in a multi-layer PFC. | PROVED (§2) | x | Give each layer its own persistence (leaky potential or self-hold), or stagger the updates. |
| 9 | **Budgets act every 100 ticks; idle replay is uncontrolled.** | hip 13–17 vs budget 2; idle Δw 6.9 | x | Normalise on every update. Put replay and downscaling into explicit sleep blocks. |
| 10 | **Compute goes to passengers.** | S1: 14.6 vs 4.9 ms/tick | x | Connect them (Stages 2–3) or drop them from runs until they matter. |
| 11 | **Single-pair stimulus→response protocol.** There is no conversation state; T is overwritten by the own voice from B1+2. | ARCH_v1 trace | F for dialogue | Multi-turn tasks; PFC and hippocampal context across turns. |
| 12 | **The world has no phonology and no noise.** Untaught chars cannot be recognised or said; nothing transfers between chars. | world/voice.py | x (world change) | Compositional syllables (initial + final + tone, with shared muscle synergies), background noise, variable syllable durations. |

### 4.1 Hidden assumptions (each is false in the real world)
1. Silence is exactly zero energy.
2. A teacher can drive the model's muscles at the model's own syllable boundaries.
3. Syllables last exactly 3 ticks.
4. One char = one atomic sound + one unique muscle set.
5. One partner, strict alternation, and the partner's replies never depend on what the model said.
6. Decision learning stops in free use, yet the system is called "real-time".
7. The 20-way char-F1 metric measures content.
8. 1,000 dialogues can show content.
9. Synchronous ticks are harmless (not for loops).
10. Budgets bound the weights (only every 100 ticks).

### 4.2 Mechanism audit: what is fixed, what learns, what matters

- **Fixed and task-specific (≈30):**
  - turn units: F, F2, GO, R1, R2, CD1–4;
  - beat: the boundary rule and phase counter; the force ramp;
  - motor: D/I/M slaving; Hp hold; program recruitment; the Hc delay line;
  - stopping: the END summed-drive rule (×0.9ⁿ); the 12-syllable counter; floor θ; the used flag;
  - thought: T (fan-in 30, hold 3.0, set point, tie-break); P = T∧GO; FEAT0 selection;
  - echo route: the Z clamp at B+4; the own-voice gate on Z; the Z threshold 0.75·B; latch set and clear rules; E = L∧R2; diagonal-only E→Y;
  - teacher and scaling: slot cap 7; ×10 memory/copy scale; cue 12; END clamp; teacher-drive gate;
  - clocks and world: the experience clock; the re-articulation world patch.
- **Learned and causal (6 tables):** P→Y fast, P→Y slow, Hc→Y, Hc→END, FEAT0→Z, E→Y.
- **Learned but non-causal:** ear→ear, mctx→mctx, ear→mctx, mctx→ear, hip→hip, and the relays (S1).

For a "general real-time learner" this inventory is upside down: the general parts are passengers, and the specific parts carry everything.

---

## 5. What to optimise, in priority order

### 5.1 Principle
Few, general mechanisms. Every one of them learns, and every one of them matters.

- An innate scaffold is allowed, and is brain-like, but it must be generic and overridable by a learned system.
- Before each long run, two things must exist:
  - an instrument that tells learning apart from lookup and echo;
  - an ablation showing that each learned part actually changes behaviour.

### 5.2 Target architecture (block level)
```
           broadcast modulators M(t): teach/reward (DA-like) · surprise (NE/ACh-like, from unpredicted units) · mode listen|speak
                 │ every plastic synapse: eligibility trace (rule v2) × M → weight change
 sound+noise ─► EAR-0 (fixed, as now) ─► CORTEX sheets: sparse ≤5 %, predicted/unpredicted state per unit,
   ▲ own voice                             learns feed-forward features (competitive + set point) and order (rule v2)
   │                                           │        ▲ top-down bias, replay
   │                                           ▼        │
   │                         HIPPOCAMPUS: DG-like sparse code → EPISODE CELLS (recruited, one-shot) → sequence replay
   │                           (retrieval at the end of a heard line; replay in pauses / "sleep" → cortex consolidation)
   │                                           ▼
   │                         PFC: input layer (learned from cortex + hip) → hold layer (self-recurrent, leaky),
   │                           update/output gated by the BG; biases cortex, hippocampal retrieval and the BG
   │                                           ▼
   │                         BG GATE: tonic inhibition; disinhibits ONE candidate per beat; learns what/when (rule v2 × M)
   │                                           │  ▲ innate defaults = v1's F/GO/CD/beat as a "brainstem reflex";
   │                                           ▼  │   the learned gate may delay, advance or abort it
   │                         MOTOR: syllable/synergy programs learned by babbling; inverse model heard sound → program (= Z);
   │                           cerebellar forward model program → predicted own voice (cancels it in F; its error teaches)
   └──────────────────────── muscles ─► mouth ─► own voice
```

How things learn in this design:
- One rule (v2) everywhere. Areas differ only in speed: cortex slow, hippocampus one-shot, PFC gated.
- One modulator vector.
- Unit recruitment (already accepted for programs) is reused for episode cells.
- Global decay stays the single clean-up mechanism, now applied in sleep blocks.

### 5.3 Stages: what each changes, buys and costs, and how to check it before any long run

**Stage 0 — instruments (CPU, days).**
- Changes:
  - a causal audit: ablate each learned part (template: `ablate_passengers.py`);
  - echo-free metrics plus baselines on the same stream (echo, last char, retrieval, association);
  - a synthetic curriculum: ~60 chars; templates "X是什么颜色→X是Y色" (facts), "我叫N→你好N" (binding/copy), "A加B→C" (rule), two-turn "N三岁 … N几岁→N三岁" (context); 30 % of combinations held out (cf. SCAN, Lake & Baroni 2018, CITED);
  - a noisy world: background noise, ±1-tick syllable jitter, mid-turn pauses.
- Buys: we can tell learning from lookup and echo.
- Cost: small.
- Check: the lookup baselines must fail the held-out and two-turn items, and echo must score 0 on the echo-free metric. If not, the curriculum is too easy.

**Stage 1 — rule v2 and modulators (CPU unit tests, minutes).**
- Changes:
  - (a) the back term fires only when the source turns on again (S5);
  - (b) an eligibility trace on forward events, decaying over ~5–20 ticks;
  - (c) slow depression when the source is on and the target does not rise, so weights estimate the probability of the target's onset given the source;
  - (d) a broadcast modulator scales all weight changes;
  - (e) plastic inhibition through inhibitory interneurons, first used to cancel the own voice (Schneider 2018).
- Buys: P = T∧GO, E = L∧R2, the Hc delay lines and the CD timer are no longer needed. Delayed credit, normalised copy weights and learning from consequences become possible.
- Costs:
  - one trace per synapse (memory ×1.5–2) and one broadcast signal;
  - the designer's clause "a held neuron counts as firing again" is replaced. S5 shows its purpose survives: constant sources get only base-rate weights, and ping-pong is still blocked.
- Check:
  - unit tests: held → onset; delay of k ticks; probability estimate; no ping-pong; no reverse order; no runaway from constant sources; predictable input is cancelled;
  - then the v1 toy with rule v2 must keep every mechanics check at 0 errors and must not lose familiar or echo scores.

**Stage 2 — the hippocampus becomes the memory route.**
- Changes:
  - recruited one-shot episode cells, keyed on a sparse code of the heard line (T or the DG-like S code);
  - the output is the reply as a sequence, via episode × position conjunctions or an episode-specific chain;
  - retrieval acts only above a familiarity threshold (same-line T Jaccard ≈0.8 vs 0.05 for different lines); otherwise the default routes play.
- Buys:
  - whole-reply familiar recall of 0.98–0.99 after one exposure and 0.97 at +10k (S4), instead of 0.45 / 0.39;
  - the hippocampus gets a causal role;
  - P→Y fast/slow and their budgets can likely be deleted (check by ablation).
- Cost: about 1.5 units per dialogue (≈48k for the whole corpus) and one sparse dot product at each line end.
- Check: the S4 numbers on the GPU build; a hippocampus ablation that changes behaviour; unfamiliar lines unchanged.

**Stage 3 — predictive cortex plus consolidation (the only stage that can produce meaning).**
- Changes:
  - sparse sheets (≤5 % active, ~1,000 synapses per neuron) with a predicted/unpredicted state per unit;
  - competitive learning of feed-forward features;
  - offline "sleep" blocks where the hippocampus replays episodes into cortex;
  - global decay applied during sleep.
- Buys: codes that reflect statistics; a surprise signal for the modulators; retention through consolidation.
- Cost: the largest compute item, but it replaces passengers we already pay for.
- Check, on the curriculum:
  - held-out combinations scored above the lookup baselines;
  - the codes of role-equivalent chars (all object names, all colours) become more similar to each other than random pairs;
  - retention with sleep is better than without.

**Stage 4 — the PFC as gated context, with the BG learning what and when (details in §5.5).**
- Check: two-turn binding scores above the lookup baselines; a PFC ablation drops it to chance; the period-2 check passes.

**Stage 5 — motor learning and teaching by sound only.**
- Changes:
  - babbling: random programs, the model hears its own voice;
  - this trains the inverse model Z and a forward model;
  - then imitation of heard syllables;
  - then the teacher only speaks the reply;
  - compositional phonology in the world.
- Buys: no motor oracle; learning works in deployment; the approach transfers to robots.
- Check:
  - syllables the model has never said can be imitated, thanks to shared components;
  - F with learned own-voice cancellation works under noise;
  - overlaps during the partner's mid-turn pauses fall.

**Stage 6 — retire the scaffolding.**
- For each fixed mechanism:
  - train its learned replacement, using the fixed one as default or teacher;
  - switch the fixed one off and compare.
- Success: the count of fixed mechanisms falls, and a second, non-speech task runs on the same code.

### 5.4 What not to do
- Do not scale the current v1 on the GPU expecting content. The main content route is flat from 1k to 10k dialogues (0.14–0.15 top-1). The best variant rises only together with parroting (0.15 → 0.20 while the parrot index goes 0.41 → 0.59).
- Do not add more SLOT patches, flags or scale ratios tuned on the 20-way metric.
- Do not add PFC layers before a learning signal exists. The old loop already showed the outcome: a content-free generic state.

### 5.5 The designer's PFC question, answered directly
The concern is correct: v1's T is a fixed hash of raw sound that never learns and is read only once. The fixes, in order:
1. Feed the PFC from learned codes (cortex and hippocampal retrieval), not from ear layer 0.
2. Learn what to hold, with a BG-gated update driven by rule v2 × the modulator (as in PBWM, O'Reilly & Frank 2006).
3. Learn the readout from held states. This becomes possible once the back term is onset-based and an eligibility trace exists (S5).
4. Hold within each layer: a leaky, self-recurrent deep layer, so that 2-tick loops cannot alternate (PROVED §2).
5. Only then add layers: a superficial input/selection layer and a deep hold/output layer. Later, a hierarchy whose front layers hold slower and more abstract context (Badre 2008).

Depth without steps 1–3 repeats the old content-free generic state.

---

## 6. Two reference architectures

**R1 "v0-lookup" — the bar any brain-inspired version must clear.**
- Parts: v1's turn clockwork + episode cells keyed on T + an echo fallback.
- Familiar lines: whole replies 0.98–0.99 after one exposure, 0.97 at +10k (S4).
- Never-heard lines: equal to echo (0.24–0.27).
- It has fewer mechanisms than v1's content side.
- Meaning: v1's neural content machinery currently adds nothing over a lookup table plus a parrot. Report R1 next to every result.

**R2 — the general learner of §5.2.**
- One sheet type: sparse k-WTA, set point, leaky potential, predicted/unpredicted state.
- One rule (v2) and one modulator vector.
- Three speeds: cortex slow, hippocampus one-shot with replay, PFC gated.
- A BG gate, a learned motor periphery, and v1's clockwork as an overridable reflex.
- Everything task-specific is learned; the same code should run a non-speech task.

---

## 7. Checks that must pass before any long training run
1. Causal audit: every learned part changes behaviour when ablated.
2. Echo-free content above the baselines on the curriculum's held-out items.
3. The rule-v2 unit tests pass.
4. Mechanics at 0 errors in the noise-free world.
5. Mechanics measured, not assumed, in the noisy world.
6. Familiar recall ≥0.9 after one exposure, and still ≥0.9 at +10k dialogues.
7. A period-2 check on every recurrent loop: Jaccard of tick t with t+1 and with t+2.
8. Learning is on in free use and does not drift. Test: the minimal verifier's self-reinforcement test, where the first syllable drifted 你 → 的/了, now with learning gated by the modulators.
9. No compute is spent on parts with no measured effect.

---

## 8. Where I may be wrong
- The designer may accept the clockwork as innate "brainstem". Then problem 6 matters less; problems 1–5 remain.
- S4 relies on T identifying lines. With noise or other speakers, T similarity will drop and retrieval must become similarity-based.
- The echo-free ceilings come from count and retrieval learners. A learner with real semantics, trained on far more data, could beat them. That is exactly why the instrument must change.
- Rule v2 may create new failure modes, for example held generic states picking up prior mass. It must be unit-tested and then run in the integrated toy before any GPU run.
- S6 uses random patterns. Correlated speech codes can help (generalisation) or hurt (interference).

---

## 9. Files (`/tmp/claude-0/-root/e92d3d7d-5823-4f5c-a0ec-4fe302bff9af/scratchpad/arch/review/skeptic/`)

| check | files |
|---|---|
| S1 passenger ablation | `ablate_passengers.py` → `ablate_passengers.out` |
| S2 corpus baselines, echo-free metric | `baselines_corpus.py` → `baselines_corpus.out` |
| S3 episode store on T, never-heard | `retrieval_T.py` → `retrieval_T.out` |
| S4 episode store on T, familiar | `episode_familiar.py` → `episode_familiar.out` |
| S5 learning-rule variant | `rule_v2_test.py` → `rule_v2_test.out` |
| S6 capacity | `capacity.py` → `capacity.out`, `capacity_nobudget.out` |

`arch/synth/archv1.py` and `arch/content/*.pkl` were only read (`python3 -B`, so no `__pycache__` was written there).

---

## summary_zh
1. 身体（轮流说话、发音、停止）0 错误，是因为它们是手工搭的固定结构（约 30 个固定单元和规则）。这证明代码没写错，不证明模型会学；而且只在"安静时声音正好为 0"的理想世界成立。
2. 关键发现：把听觉皮层的学习、海马体和回程、运动皮层全部关掉，模型说的话和学到的权重一模一样（30/30 相同）。这些部分却用了 2/3 的计算量。现在真正决定说话的只有 6 张主要靠老师示范来学的查表。
3. "没听过的话 0.12"其实是重复听到的字：什么都不学、直接把听到的话重复一遍，能得 0.24。去掉重复因素后，所有方法在 1000 段对话时都等于瞎猜，3 万段也只略高于瞎猜。
4. 1000 段对话约等于小孩一天听到的话，这个语料在这个规模上根本测不出"意义"。需要换测试：人工设计、有规律可学的对话，加上留出的新组合和多轮对话。
5. 记忆的瓶颈不在"想法"本身：用现有的 T 编码做"一段经历一个记忆单元"（海马体式），听一遍就能完整说出回答（0.98），1 万段对话后仍有 0.97；现在的方法只有 0.45 和 0.39。
6. 根本原因在学习规则的一条子句："一直亮着算再亮一次"。改成"真正再次亮起才减弱"，被托住的想法就能学会触发动作；同时仍然不会来回激活、不会学反顺序。
7. 规则还缺：跨时间的记账（资格痕迹）、按比例而不是按次数、学来的抑制、根据结果学习（多巴胺式信号）。每缺一样，就多一个固定补丁。
8. 每个神经元只有 96 根连接、10–40% 同时亮，最多记几十到几百个转移；大脑是稀疏激活、每个神经元上千上万个连接。
9. 前额叶的担心是对的，但缺的首先不是层数，而是：输入要来自学出来的编码，要学会"该记住什么"（基底节闸门），要能从托住的状态学出动作。同步更新的多层回路一定会两拍来回跳，每层要有自己的"保持"。
10. 建议顺序：先换测试和做因果检查 → 改学习规则并做单元测试 → 海马体变成真正的记忆通路 → 皮层学预测并在"睡眠"中巩固 → 前额叶加基底节闸门 → 咿呀学语、只用声音教 → 逐步拆掉固定结构。
11. 不要再在 GPU 上放大训练现在的 v1 来期待内容变好，也不要继续给 20 选 1 指标打补丁。