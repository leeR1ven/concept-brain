# Whole-architecture review — coordinator's synthesis of the five reports (2026-10-31)

Sources: `arch/review/{skeptic,learning,neuro,develop,compute}/report.md` (each with scripts and outputs in its folder).
Every one of the five reviewers answered the same questions for the whole architecture. Agreement is very high.

## A. Consensus (5/5 unless noted)
1. **Keep the body** (F/GO/CD, 3-tick beat, programs + WTA, END/counter, D/I/M) as an innate reflex scaffold. It is
   brain-like (brainstem / BG / thalamic GO). Skeptic and neuro: later make it tunable or overridable by learned gates.
2. **Learning is disconnected from behaviour.** Ear cortex, hippocampus and motor cortex learn but reach no decision.
   Ablating them all changes no decision weight and none of 30/30 free replies (skeptic S1; PROVED by code in neuro,
   learning, compute, develop). They use 2/3 of the toy's compute.
3. **No real-time learning in use.** What to say, what comes next, when to stop and what to copy are learned only while
   the teacher drives the muscles; in free conversation they are frozen.
4. **No learned representation on the content path.** The ear "cortex" is a thresholded copy of its input lines for
   99.4–99.6 % of neuron-ticks (compute; neuro PROVED from net3d.py). T is a fixed hash weighted toward the last ~8 chars.
   The hippocampus is a 1:1 mirror with a ~15 s half-life. So experiences can only be similar by sharing sounds → no meaning, no B8.
   Neuro: T + P→Y is a cerebellum / mushroom-body design (memorises, never abstracts). v1 has the dorsal "hear → repeat" route but no ventral "hear → understand" route.
5. **The learning rule lacks what a general learner needs:**
   - held co-activity cannot be learned → no assemblies, concepts or learned working memory. With the exact rule, 0 patterns complete from half cues. With co-activity or predictive learning plus an immediate budget, 97–100 % of 300 patterns complete (fan-in 960) — compute, neuro.
   - no credit across delays → every link needs a hand-built 1-tick gate (eligibility trace needed);
   - no negative evidence, prediction error or learned inhibition → weights are frequency counts with budget artefacts. An error-driven, signed readout gives one-shot 0.72 vs 0.35 and retention 0.84 vs 0.48 (compute); excitatory-only gives 0.05;
   - no third factor (neuromodulators: reward, surprise, novelty) → only "freeze" (teacher gate) or "drift" (你→的/了);
   - budgets are enforced only every 100 ticks → overshoot (hip 13–17 vs 2) → super-attractors once co-activity exists.
   - Skeptic: the timing law comes from the state-based back clause only. An onset-based back term lets a held state learn to trigger an onset and still blocks ping-pong and reverse order.
6. **Memory.**
   - The hippocampus should be a one-shot, long-lasting episodic index with offline replay into cortex (consolidation, complementary learning systems). Its current role is the reverse of the clinical evidence (H.M.).
   - Evidence:
     - an episode store keyed on T gives whole replies 0.98–0.99 after ONE exposure and 0.97 at +10k dialogues (skeptic S4);
     - local growth (full units mature, fresh ones join) gives 0.83–0.88 vs 0.41–0.50 (learning).
7. **Capacity.** Fan-in 96 with 10–40 % activity stores only ~30–300 transitions. Sparse (≤5 %) codes with ~1,000 synapses per neuron store ≥1,000 (skeptic, compute).
8. **The testbed cannot show meaning.**
   - Echo-free ceilings on never-heard lines are ≈ chance even for ideal retrieval or association learners with 40k pairs; echo alone scores 0.24–0.27.
   - The adult corpus is ≈ 0.4 M words (weeks of a child's input), 99.7 % of pairs are unique, and it refers to nothing the model perceives. Prompt → first reply char carries 0.01–0.03 of 7.85 bits (develop).
   - Develop: even an ideal child-directed curriculum fails with v1's machinery ("你要X吗 → 要X" on unseen combos 0.00), because of the unordered latch, recency-bag T, no slot binding and an additive readout. A mixed-selectivity layer lifts naming 0.15 → 0.55.
   - Neuro: the corpus does contain learnable categories (a local learner gets purity 0.34 vs 0.006 chance: 我→你他她, 红→绿紫蓝).
9. **Teaching.** Stop puppeting the muscles. Instead:
   - babbling builds the inverse model (Z already is half of it);
   - then imitation;
   - then learning by listening: predict what will be heard next, from both speakers and overheard dialogues — speaking = saying the prediction;
   - plus contingent partner feedback (recasts) as a third factor.
   - World: make articulation → sound lawful (develop: r = −0.019 today), add noise, later compositional phonology.
10. **PFC (the designer's concern): right, but depth is not the first missing thing.**
    - The old multi-layer loop failed because:
      - its input was raw sound with a shared core;
      - the rule cannot store held activity;
      - there was no "what to hold" signal;
      - 1-tick synchronous units alternate with period 2 (PROVED, a known artefact class).
    - What the PFC needs:
      - learned inputs (lexical / meaning layers, hippocampal recall);
      - per-layer persistence (leaky units);
      - BG-like gating (update while the partner speaks, hold during the reply — reuse the turn units);
      - assemblies via co-activity;
      - readout at every boundary;
      - mixed selectivity;
      - slow context layers (line ~20 ticks, dialogue ~100 ticks) — the rostro-caudal "multi-layer" part.
    - Build it after the lexical layer (the PFC matures last).
11. **Offline "sleep" phases** for replay and downscaling; stop uncontrolled replay during waking silence.
12. **Process.** Every step gets a CPU bench test with pass thresholds → a 50-dialogue integrated smoke run with ARCH_v1's checks → only then a long run. A causal ablation must show every learned part changes behaviour.

## B. Where the reviewers differ (to resolve in design)
- Order of the first steps. All start with the testbed plus the rule, but:
  - skeptic and compute: instruments → rule → memory → cortex → PFC;
  - develop: world/curriculum first, then structure;
  - learning: teacher-free prediction first;
  - neuro: world + per-area rule → LEX + GIST → production through perception.
- How the rule is extended:
  - one predictive rule everywhere (compute, skeptic), vs
  - per-area rules: the current order rule in sequence areas, co-activity in association areas, trace × third factor on decision paths (neuro).
- The first PFC:
  - fixed random mixed-selectivity layer + ordered utterance memory (develop);
  - leaky gist layer with co-activity (neuro);
  - fixed slow context layer, then assemblies (compute).
- Episode key: T directly (skeptic) vs a DG-like sparse expansion (neuro, compute).

## C. Framework changes that need the designer's decision
1. **Learning rule:**
   - (a) back term on source onset, not "held counts as firing again";
   - (b) prediction error: "already predicted → stop strengthening; predicted but absent → weaken";
   - (c) learned inhibition (negative effective weights via inhibitory interneurons);
   - (d) "fire together → strengthen" in association areas (relaxes framework #9 there);
   - (e) eligibility trace + one broadcast "important / surprise / reward" signal.
2. **Hippocampus:** a one-shot, long-lasting episodic memory with replay to cortex, not only short-term memory.
3. **Training:** a staged, child-like world with a scene; no muscle puppeting; the adult corpus last.
4. **Connectivity:** sparse activity and ~1,000 inputs per neuron in memory/association areas (GPU cost).

## D. Next phase (proposed)
Design ARCH_v2 as a whole plus a staged roadmap with bench tests: several independent full designs → adversarial
verification → synthesis. Body kept; rule, memory, representation, PFC, teaching and world redesigned per A/C.
