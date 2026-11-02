(Saved by the coordinator from the reviewer's final reply; the harness blocked the reviewer's own write.)

# Review round 2: the whole ARCH_v2, through a systems-neuroscience lens

Reviewer: neuro. Labels: **PROVED** (argument with numbers), **SIMULATED** (computed by me; file and numbers), **CITED** (paper and finding), **ASSUMED**.

The harness blocked writing `report.md`, so the full report is below.

**Where my numbers come from.** I ran no agent. Every new number comes from one script in my folder:
- script: `docs/review2/neuro/log_checks.py`
- output: `docs/review2/neuro/log_checks.out`

It only reads files that already exist:
- the seed-5 life logs `synth/out/life/{final_noctx,final_novgo,final_nooth}_s5_log.pkl` and `final_noctx_s5.json`;
- the five staged runs `synth/out/runs/final_noctx_s{0..4}.json`.

It ran as one process, in under a minute and under 200 MB. All other paths below are relative to `scratchpad/arch/v2/`. "ARCH_v2" means the `final_noctx` configuration.

## 0. Verdict

ARCH_v2 is a real step forward:
- it learns by listening with a signed, local error rule;
- it keeps a one-shot episodic index and makes memories firm by awake recall;
- its attention is brain-like: it follows pointing, compares what it expects to see with what it sees, and uses a retinotopic layout;
- the innate body makes no mechanics errors.

In brain terms, though, the content path is still a lookup machine:
- one unit per syllable, with no shared or distributed codes;
- an exact dictionary of pairwise conjunctions, read out by a linear layer;
- an exact table of episodes;
- three thought steps in a fixed order.

Three gaps block the goal more than any missing brain area does.

1. **The self is missing from its own context.**
   - The agent's own speech enters neither the thought nor the episodic memory. That is what causes free-use drift.
   - Awake recall is counted as the drift brake. It will make drift worse once the store it replays from fills with untutored episodes.
2. **There is no signed value loop.**
   - When an expected reward does not come, nothing reaches the critic or the actor (`synth/agent_think.py:398`). The agent cannot unlearn anything through consequences.
   - The learned brake only predicts "will someone else start talking". An empty room is exactly where that answer is "no".
   - Result: 626 of the agent's 1,337 self-initiated 你好 were said to an empty room, and the rate does not fall over the whole life.
3. **There is no content-independent routing and no learned similarity.**
   - The agent generalises only where the world hands it shared input features: held-out colour 0.60, count 0.37.
   - Where the answer must name something, it does not: held-out what 0.09, slot 0.05, where 0.04, taste 0.00.
   - Life memory facts stay at 0.00 even after 9 or more encounters.

**Two showcase numbers do not mean what they claim.**
- The "favourite topics" are the most frequent function syllables. Their base rate is 0.63, not 0.14, and they never reach speech.
- "Greeting first is learned" is half true. What was learned is "the scene changed → say 你好", not "a person arrived → greet".

Scaling up (the GPU port, 16× units) before closing these three gaps will mostly buy more memorised conjunctions.

## 1. ARCH_v2 compared with the real brain

**1. Content codes — matters most.**
- ARCH_v2: one unit per syllable category. Contexts are exact pair keys in a dict (rule R3), and the readout is linear in those keys.
- Brain: distributed, overlapping population codes, with similarity learned from use.
  - Temporal-context codes (Howard & Kahana 2002) and successor-like codes (Stachenfeld et al. 2017).
  - Mixed selectivity from broad connectivity (Rigotti et al. 2013).
- Why it matters: this is why held-out answers stay at 0.0–0.2, and why scaling will not generalise (§2.3).

**2. Word level — matters for the adult corpus and for naming.**
- ARCH_v2 has none: items are syllables.
  - 小 is shared by 小狗, 小猫, 小鸟 and all 8 names.
  - 子 is shared by 鸭子, 盒子, 杯子, and so on.
- Brain: word-form areas (STG/STS) and lexical-semantic areas (MTG/ATL) on the ventral route (Hickok & Poeppel 2007).

**3. PFC dynamics — matters for "keeps thinking" and for dialogue across turns.**
- ARCH_v2: the thought is a passive buffer of the last heard line.
  - Competition λ 0.3 is below the threshold θ 0.5, so it behaves as a register (the designer's own equivalence proof).
  - It never decays, and it changes only when a line arrives.
  - There is no slower context layer (CTX was removed), so the context is exactly one line.
- Brain:
  - recurrent dynamics that generate sequences;
  - a front-to-back hierarchy of timescales (Koechlin et al. 2003; Badre & D'Esposito 2009);
  - updates gated by the basal ganglia (O'Reilly & Frank 2006).
- CTX failing only shows that a *fixed random* layer feeding the hippocampal key is harmful. It says nothing against a learned slow layer.

**4. Basal ganglia — matters.**
- ARCH_v2:
  - 3 copy gates, whose keys include the identity of the candidate being copied;
  - an evidence accumulator for release;
  - a learned brake;
  - V_go, which has no measurable effect (a passenger).
- Brain: Go and NoGo pathways, trained by dopamine bursts and dips (Frank et al. 2004). Gates act on roles, not contents ("indirection"), which generalises to new role–filler pairs (Kriete et al. 2013).

**5. Dopamine / reward-prediction error — matters.**
- ARCH_v2: δ = r − V is computed only when r ≠ 0, so there are no dips.
- Brain: the dip at a missing reward is the core signal (Schultz, Dayan & Montague 1997).

**6. Neuromodulated plasticity — matters moderately (drift, interference).**
- ARCH_v2: one fixed learning rate everywhere; every heard line is stored and learned with equal weight. The designer's "inputs of different strength" item is still open.
- Brain: ACh/NE signal surprise, set the learning rate and mark event boundaries (Yu & Dayan 2005; Bouret & Sara 2005). Hippocampal storage is gated by novelty (Lisman & Grace 2005).

**7. Hippocampus — the exactness does not matter now; the missing self and what gets replayed do.**
- ARCH_v2:
  - DG pattern separation (good);
  - recall is a global argmax that returns an exact record;
  - NEXT is a pointer;
  - the agent's own lines are not stored;
  - replay is uniform over a window of about 6,000 lines.
- Brain:
  - CA3 auto-association with interference, and a CA1 match/mismatch comparison;
  - episodes include one's own actions;
  - replay is prioritised by reward and novelty (Mattar & Daw 2018), and awake replay exists (Karlsson & Frank 2009).

**8. Consolidation — matters (tied to item 1).**
- ARCH_v2: replay copies line → next-line pairs into the same conjunction table. That memorises; it does not abstract.
- Brain: complementary learning systems. The cortex extracts structure because its codes overlap (McClelland et al. 1995).

**9. Self-monitoring — matters.**
- ARCH_v2: own speech only sets an efference tag. ct_own = 0, and own lines are not in the hippocampus.
- Brain: the sound of one's own voice is damped, but its content stays represented (Houde et al. 2002). One's own acts are part of episodes.

**10. Social perception — matters for world2 and for IDEAS #17, whose innate values need reliable social inputs.**
- ARCH_v2: no person or addressee detector. The partner's code is 8 random eye bits, and only the turn-taking learners see them.
- Brain: innate face orienting below the cortex (Johnson 2005); STS/TPJ for agents and gaze.

**11. Perception — not for the bench; yes for world2, real audio and real images.**
- ARCH_v2: a fixed ear layer 0 (fine here); syllable categories with no gain control; a clean eye code with one dedicated unit set per attribute and a separate pointing channel.
- Brain: divisive normalisation (Carandini & Heeger 2012), learned invariance, separating sound sources, voice identity.

**12. Attention and body — brain-like; keep.** Innate gaze-following, the expectation-vs-scene comparator, retinotopic gain, the innate beat, turn units, programs and reflexes.

**13. Sleep — matters only through capacity.**
- The designer chose no sleep. The brain uses sleep to scale synapses down (Tononi & Cirelli 2014).
- With pruning off, MIX keeps growing: 29k → 66.7k units over the life, still about +1.2k per tenth at the end.
- A design without sleep needs an awake clean-up: synaptic scaling, pruning unused conjunctions (Turrigiano 2008).

**14. Does not matter:** 76-ms ticks, binary units, no cerebellum or thalamus model. The 4.4 Hz beat matches theta-rate syllable tracking.

Also right: lines enter the thought at their end. This matches encoding at event boundaries (Ben-Yakov & Henson 2018; Baldassano et al. 2017).

## 2. Problems of the whole architecture

### 2.1 Logically unsound or internally inconsistent

**P1. The learned brake frees the floor where nobody listens.** PROVED.
- Rule R8 trains P_oth to predict "another speaker starts within 10 ticks". A release needs its estimate p_o to be low (`agent_think.py:291`).
- When the agent is alone, no one ever starts speaking, so p_o goes toward 0 in exactly those contexts.
- Without this learning there are no initiatives at all (`final_nooth`: 0). So all initiative comes from a predictor that is most permissive in an empty room.

SIMULATED (`log_checks.out` §2):
- 626 of 1,337 self-initiated 你好 were said with nobody present.
- Per tenth of life: 3, 57, 87, 70, 57, 65, 74, 61, 88, 64. There is no extinction.
- 244 of them came right after the partner **left**, and 382 after scene changes (median lag 12 ticks).
- Removing V_go changes nothing: 748 to an empty room.

**P2. There is no negative reward-prediction error.** PROVED.
- `if r and self.learning: core.reward(r)` (`agent_think.py:398`): the critic and actor update only on ticks where r ≠ 0. When the reward is missing (r = 0), δ < 0 is never computed. The actor can only strengthen.
- V_go does get target 0 when no reaction comes (`no_react`), but it only scales the gain: G = clip(1 + 2(V_go − r̄), 0, 3).
  - G reaches 0 only if V_go ≤ r̄ − 0.5, which is impossible while the average reward r̄ is below 0.5.
  - So value can slow a release but never veto one. There is no NoGo pathway.
- Consequence: nothing can be unlearned by its consequences. That rules out IDEAS #17's "don't upset people" brake.

**P3. Self-initiated speech ignores the thought, so the favourite thoughts are inert.** PROVED from the code.
- A scene thought (the source of every unprompted utterance) gets P = [] (`core_think.py:122`). Its readout sees only the scene, never the held thought.
- At the next heard line, competition multiplies the held content by 0.3 (`core_synth.py:239`). Any wandered content falls below threshold before a reply is read out.
- So the co-holding links (R11) and the wandering are learned but reach no speech. That violates principle 7, and it explains "J identical by construction, life within noise".
- Their only effect is a side effect: the wandered content becomes the learning context for the first heard line after a long silence.

**P4. The drift brake wears off and then reverses.** PROVED arithmetic; the size of the effect is ASSUMED.
- Awake recall samples uniformly among episodes with σ > 0.05 (`core.py:695`). σ shrinks by 0.9995 per stored line, so only the last **5,989 lines** can be replayed.
- The agent's own replies are not stored. In an untutored dialogue, question Q1's NEXT pointer is therefore the next question Q2 (`core_synth.py:445`). Replaying it teaches Q1 → Q2, which is the drift itself.
- In the 300-dialogue drift test the replay window is still mostly tutored, so recall works as a brake (C 0.77 → 0.01 without it).
- After about 6,000 untutored lines, every replay rehearses drift. At the life's rate (3,967 heard lines in 21 h) that is about 1.5 life-days. Life never shows it only because the recast tutor never leaves.

**P5. "Continuous thinking" has no length knob.**
- With λ < θ the thought is the previous-line register (the designer's equivalence test: 592/592 records).
- The readout's keys put every held item into one bag (`_ct_view`). Older content therefore confounds the new line instead of adding separate context. This is the likely cause of the −0.05 when residue was tried (ASSUMED).
- So IDEAS #15 ("excitability = context length; bigger is better") has nothing to turn yet. A length knob needs keys that tell held lines apart: a separate group for the older line, or keys weighted by activation.

**P6. Memory cannot be said in contexts that were never answered.**
- Copy-gate keys include the candidate's identity (`gate_noid` is False by default).
- The gates learn only when the next heard item equals the candidate.
- In life, memory questions are never answered afterwards. So the completion gate never gets a positive target in those contexts.
- Result: fact_name 0/51 and fact_person 0/62 at **9 or more** encounters (SIMULATED, §6). This is not one-shot learning failing; it is a routing failure.

### 2.2 Hidden teachers and shortcuts

- **Agent code is clean except for babbling.** By grep over `agent.py`, `agent_think.py`, `agent_synth.py`, `run_j.py` and `life_synth.py`, the agent reads only `obs`, the allowed `world.warmup_windows()`, and `babble(world.voice)`. The babbling shortcut is broader than "reads world.voice":
  - `voice.decode(m)` returns the symbolic character;
  - sounds are de-duplicated by that character's identity;
  - the own-voice pattern is synthesised in a clean stream rather than heard.

  Doing it online is affordable: 35,960 muscle sets × 3 ticks ≈ 108k ticks ≈ 2.3 h of life time.
- **Innate constants were read off `world.py`:**
  - reflex 7's 10 ticks, against the world's 5-tick scene lead;
  - the reply window τ ≤ 7, against the world's 12-tick wait;
  - a syllable clock of exactly 3 ticks per character.

  "0 overlaps" is therefore a property of instincts fitted to this one protocol, not evidence of social learning. The designer allows this, but each constant must be re-derived for world2.
- **The eye is clean and factorised:** 8 units per attribute value, the same code in every slot block, plus a dedicated pointing code. Colour 0.93 and count 0.76 depend on this.
- **Life is tutored forever:** recasts all life long, and reward = exact correctness. So "no drift in life" says little about an open world.

### 2.3 What will not survive an open world, many bodies, real images or an adult corpus

**Adult corpus and scale.** PROVED arithmetic.
- The bench has 141 syllables and 3,967 lines, and ends with 66.7k conjunction units.
- With about 4,000 syllables, one pair type alone (held item × last item) has 16M possible keys.
- The GPU port hashes MIX into 2¹⁹ = 524k slots, so the load is far above 1. Collisions are random: units shared by collision carry interference, not similarity.
- Prediction for the scale test P6: held-out H ≤ 0.25 at 4× and at 16×, unless change #4 lands first.

**Many bodies.**
- Syllable categories hit their cap within a day (ARCH_v2 §8.4).
- Every voice that is not the agent's own is treated as a teacher, so peers' errors become training data. There is no speaker identity and no trust. IDEAS #17's "containing bad behaviour" fails at the first synapse.
- Splitting lines by silence, and a syllable clock locked to speech onset, both break when speakers overlap.
- Caregivers who do not recast will cause drift (P4).
- The agent's own motor acts are not remembered, so "where did I put it" cannot be answered.

**Real images.** The scene-focus keys, the expected-appearance maps, `rate_max` and `conf_min` were all tuned on clean binary attribute codes.

**Being alone in an open world.** P1 + P2: the agent talks to empty rooms indefinitely.

### 2.4 Numbers that do not mean what they claim

All SIMULATED from `log_checks.out`.

**"Favourite topics 0.81 vs a base rate of 0.14."**
- 0.14 is the *type* share (20 of 141 syllables). The *token* share of the top 20 in the partner's speech is **0.63**.
- The top 20 are 是这什么小再见吗我在好叫你子里要哪不放了: function syllables and farewells.
- Spreading activation over row-normalised co-occurrence converges on these hub items. That is the "one generic state" the designer wants tuned away, not a foodie's topic. It also never reaches speech (P3).

**"1,891 initiatives; greeted first 347."** 1,337 of them are 你好. 626 went to an empty room, at a flat 57–88 per tenth of life, and 244 came right after the partner left.

**"0 overlaps."** Guaranteed by reflexes tuned to this world (§2.2).

**"Name recall 0.76."** This is recall, within one dialogue, of the most recent 我叫 episode. Across visits (life fact_person) it is 0.00 even at 9+ encounters.

**"Awake recall is the drift brake."** True only while the replay window is mostly tutored (P4).

**"Honest timing."** Real but narrow:
- 60 % of replies start at the 6-tick pipeline floor, with accuracy 0.55 there against 0.25 for later replies;
- so thinking takes at most 4 ticks.

## 3. What to change next (ranked)

### #1 Put the self in the loop, and let an event-continuity signal choose where a heard line is learned

**Mechanism.**
- The agent's own line enters the thought at its end with gain ≈ 0.6, tagged *self*. The question stays held (1 − 0.7·0.6 = 0.58 ≥ θ).
- Readout keys carry the role of each held entry (`role_units`, extended so the thought view keeps roles).
- The hippocampus stores own lines with a self bit, so the chain becomes question → own answer → reaction.
- A heard line after the own line is always learned in the self-tagged context: "what follows what I say".
- It is learned in the question's context only in proportion to a continuity match m: its overlap with the held own reply and question. This is a match/mismatch signal like CA1's (Lisman & Grace 2005); a low match marks an event boundary (Zacks et al. 2007).

**Why it is the bottleneck.**
- Without a teacher, "learning all the time" fails: C loses 0.14–0.33 after 300 untutored dialogues, and P4 says it gets worse.
- develop already fixed drift symbolically with "own reply as a line + role units" (0.00 → 0.92). The open problem was losing recast learning.
- The agent can perceive the cue that solves this (SIMULATED, §3 of the output):
  - a confirmation is identical to the agent's own reply (Jaccard 1.00, n 496);
  - the match m separates confirmations from next lines with AUC 0.96, and corrections from next lines with AUC 0.72;
  - corrections contain a question syllable 0 % of the time, against 59 % for next lines. The agent could learn this as "lines after which the partner waits" (ASSUMED).
- This is a general event-segmentation mechanism, not a special timing window. The designer's "no special mechanism yet" was about timing windows, which this bench cannot support anyway.

**Prediction.**
- Drift loss over 300 dialogues ≤ 0.10 (now 0.14–0.33), with C3 within −0.03.
- In a 3,000-dialogue drift test, ARCH_v2 falls to ≤ 0.2 (P4) while the change stays ≥ 0.5.

**Test.** `run_j.py`, seeds 0–1, with `drift=300` and `drift=3000`, with and without the change (≈ 20 min per run).

### #2 A signed value loop with a NoGo veto, plus coarse innate social detectors

**Mechanism.**
- When the partner reacts, or 50 ticks pass with no reaction, update the critic and actor even if r = 0. This gives δ = −V: a dopamine dip.
- Release requires a learned value above a NoGo level, so V_go becomes a veto rather than a gain.
- Innate coarse features "person present / arrived / left" feed both the readout's keys and the value keys, as IDEAS #17 asks.
  - In life they come from the existing eye-change parser (`life_eye`).
  - In world2 they come from the caregiver cells.

**Why it is the bottleneck.**
- World2 runs on consequences: requests, eating, praise.
- IDEAS #17's innate values ("help", outweighed by "don't harm") need a signed signal and a veto.
- Today nothing can be unlearned (P2), and initiative is tied to an empty floor (P1).

**Prediction** (life seed 5):
- 你好 to an empty room falls from about 65 per tenth to ≤ 15 in the last tenth;
- rewarded greetings stay ≥ 30 per tenth;
- the share of appropriate initiatives rises from 0.19 to ≥ 0.4;
- social replies stay ≥ 0.30.

**Test.** One life, `life_synth.py`, seed 5 (≈ 9 min).

### #3 Content-independent routing (basal-ganglia indirection), with the thought as the context of unprompted speech

**Mechanism.**
- Add a fourth copy route, "say what I recognise": its candidate is the word recognised at the attended slot, in heard order.
- Make all copy gates read only the context and how familiar the candidate is, never its identity. The switches exist (`gate_noid=1 gate_freq=1`) but were never tested together with such a route.
- Self-initiated thoughts take the held thought as their context, so unprompted speech is about what the agent is thinking about.

**Why it is the bottleneck.** One root cause — missing role-based output gating (Kriete et al. 2013) — explains three open problems at once:
- held-out what / slot / where are 0.09 / 0.05 / 0.04, while colour is 0.60 (SIMULATED, §5);
- only 18 of 1,891 initiatives are namings, against 244 "这是什么";
- memory facts are 0.00 at 9+ encounters (P6).

**Prediction.**
- Held-out what and slot ≥ 0.35.
- Life namings ≥ 100, and "这是什么" initiatives cut by half.
- Life fact questions > 0.10, provided the completion gate transfers from answered 'what' episodes.

**Test.** Bench seeds 0–1, plus life seed 5.

### #4 Learned distributed word codes: a ventral lexical-semantic layer

**Mechanism.**
- Recruit word units from syllable chains that recur.
- Give each word a sparse code (for example 32 of 2,000 units) that drifts toward the average thought in which the word occurs. This is a local temporal-context rule (Howard & Kahana 2002).
- Pair these code units with the sentence frame in MIX, so that 芒果 inherits weights that 苹果 earned.
- This is IDEAS #1's "still missing" piece and IDEAS #5. It also makes hashing in the GPU port meaningful: units shared by similar words would carry similar meaning.

**Why it is the bottleneck.** Without it, the adult corpus and the scale test P6 only add memorised conjunctions (§2.3), and held-out offer / taste stay at 0.17 / 0.00.

**Prediction.**
- Nearest-neighbour category purity of the word codes ≥ 0.6 on the bench lexicon, and ≥ 0.25 on the corpus (round 1 reached 0.34 with a simpler learner).
- Held-out offer and taste ≥ 0.4.
- Gains in held-out generalisation from 1× to 16× appear only with this layer.

**Test.** A module test on existing logs and the corpus (minutes), then one bench run.

### #5 Hearing for many bodies: gain control, voice identity and selective trust

**Mechanism.**
- Divisive normalisation of the ear window before syllables are categorised. Recruit a new category only from loud, clean samples.
- A voice-identity code: one prototype per voice.
- Per-speaker trust, learned as the fraction of that speaker's lines that caregivers confirm or that the agent predicts. Trust scales the listening learning rate (selective trust in children: Koenig & Harris 2005).

**Why it is the bottleneck.**
- The next phase is many bodies.
- Categories hit their cap within a day.
- Peers' errors are learned exactly like a caregiver's lines.

**Prediction.**
- In world2, the number of categories levels off at ≤ 1.5× the lexicon size.
- With 2 nonsense-babbling peers among 16 agents, a focal agent's accuracy on caregiver questions drops ≤ 0.05 (≥ 0.15 without trust).

**Test.** `world2_smoke.py` (156 s for 4 agents × 1.5 days), then 16 agents after the port.

## 4. Are the designer's ideas implemented faithfully?

**Implemented faithfully**
- **IDEAS 8 / principle 1 (one unbroken life):** only the staged evaluation freezes copies.
- **IDEAS 11 / principle 9 (one-shot naming not engineered):** respected.
- **Principle 8 (zero mechanics errors):** met.

**Met in form only, or stretched**
- **IDEAS 1 (PFC = next-input predictor; "the prediction becomes the next input"):** one readout layer over a hand-built state. Predictions are rolled out only through the mouth, never internally, so there is no inner speech or imagined continuation. Middle layers do not learn (#4).
- **IDEAS 2 / 9 (a thought held by a loop; continuous thinking):** a buffer with no decay, equivalent to the old register (PROVED by the designer). There are no loops and no length knob (P5).
- **IDEAS 6 / principle 4 (speech caused by thinking):**
  - every thought is opened by an outside event (a line ending, or a scene or pointing onset);
  - the thought itself never opens an utterance;
  - unprompted speech ignores the thought (P3).
- **IDEAS 7 / principle 5 (favourite thoughts as a feature):** built, but mis-measured and inert.
  - The wandering converges on hub syllables (base rate 0.63) and never reaches speech (P3).
  - A faithful version would let the self-initiated thought read the held thought (#3). It would also weight spreading by each item's own usage (a homeostatic set point, as CTX had) and by value, so that a foodie returns to *food*, not to 是.
- **IDEAS 10 / principle 3 (no sleep; recall makes memories firm):** faithful as a mechanism. But "getting worse without a teacher becomes rare" is not reached, and P4 predicts it will get worse. An awake homeostatic clean-up is missing.

**Under-used or not yet implemented**
- **IDEAS 3 (senses and muscles as the PFC's input and output; a words-and-meaning layer):** no word or meaning layer (#4).
- **IDEAS 4 (local connectivity):** only in ear layer 0. The learned local fields were rejected, correctly for this bench.
- **IDEAS 5 (things used alike get similar codes):** not implemented. This is the main gap (#4); held-out stays at 0.0–0.2.
- **IDEAS 12 (bigger is better once the architecture is sound):** scaling now is premature. I predict no held-out gain at 16× (§2.3).
- **IDEAS 13 (world2, many agents):** started. Hearing and trust are still needed (#5).
- **IDEAS 15 (excitability = context length):** the context is fixed at one line, so there is nothing to sweep (P5).
- **IDEAS 16 (3-D visual network):** not yet; the bench eye is symbolic.
- **IDEAS 17 (innate values; instincts as connections that are hard to change):** reward is positive-only, and there is no veto and no social detector (#2).

**Violated**
- **IDEAS 14 (fault-tolerant sparse codes):** violated in the core. Items are IDs and keys are exact dict entries, so one misheard syllable changes every key it appears in. Fault tolerance exists only in the ear and the DG.
- **Principle 7 (every learned part reaches behaviour):** violated twice. V_go is a passenger but kept, and the co-holding links and wandering never reach speech (P3).

**Mostly met**
- **Principle 6 (local rules only):** awake recall saves and restores the thought globally. That is acceptable as activity-silent working memory (Stokes 2015).

## summary_zh

1. **大方向对**：边听边预测、带正负号的局部误差规则、一次记住的海马体加清醒回忆、注意力和先天身体都很像大脑。但内容通路本质上还是"查表"：每个音节一个单元、精确的成对组合字典、精确的情节表、固定三步思考。

2. **最要紧的三个缺口**：
   - **"自己"不在自己的上下文里。** 自己说的话不进想法、不进海马体，这正是无人教时漂移的原因。清醒回忆只回放最近约 6,000 句；等这些句子都变成无人教的对话，回忆反而会加速漂移。
   - **没有带正负号的价值回路。** 得不到奖励时从不产生负信号，所以什么都"改不掉"。学到的刹车只预测"别人会不会开口"，没人的时候它最放行：1,337 次主动"你好"里有 626 次是对空房间说的，一生都不减少。
   - **没有与内容无关的输出路由，也没有学到的相似性。** 只有世界直接给了共享特征的题才会泛化（没见过的颜色题 0.60）；要说出名字的题几乎为零（这是什么 0.09、在哪里 0.04）。"事实"类记忆题听过 9 次以上仍然是 0。

3. **两个数字言过其实**：
   - "最爱话题 0.81 对 0.14"：按出现次数算，基准其实是 0.63，最爱的是"是、这、什么、再见"这类虚字，而且从来没有进入说话。
   - "学会先打招呼"：学到的是"场景一变就说你好"，不是"有人来了才打招呼"。

4. **建议按顺序改五件事**（每条都给了可以证伪的预测和便宜的测试）：
   - (1) 让自己的话和动作进入想法和记忆，并用"是否同一件事"的匹配信号决定新听到的句子记在哪个上下文里；
   - (2) 加入"该有奖励却没有"时的负信号和"不做"的否决通路，再加上先天的"有人在 / 来了 / 走了"粗检测；
   - (3) 加"说出我认出的东西"的复制通路，闸门只看情境、不看内容；主动说话时读取当前想法；
   - (4) 学出分布式的词语编码，让意思相近的词编码也相近，这样放大模型才有意义；
   - (5) 多个身体时的听觉：增益控制、分辨说话人、按可靠程度决定信谁。

5. **在这些改好之前就放大规模**，多半只会多记一些组合，不会更会举一反三。