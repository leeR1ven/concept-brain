# ARCH_v3_core, revision 2 (r2b)

**Who and when.** The synthesis member, 2026-11-01. Design and toys only: no training run and no server job.

**What r2b adds.** r2b replaces the first r2. It folds in:
- the coordinator's mapping onto the designer's macro architecture;
- the designer's working rule: look in the designer's own sources before adding anything.

**Inputs:**
- the four reviews: develop, ml, neuro and skeptic;
- `REVISION_BRIEF.md` and `DESIGNER_SPEC.md` (40 rows);
- `designer_msgs_clean.txt`;
- IDEAS #12–#23;
- the repos ana, chain-neuron-architecture, char-neuron-network, char-neuron-voice and local3d (read only).

**Labels:**
- **PROVED**: code, count or arithmetic.
- **SIMULATED**: a toy run, with its file named.
- **CITED**: a source.
- **ASSUMED**: judgement, not tested.

**Source marks** on every element:
- **[D]**: the designer's;
- **[D·p]**: the designer's, parameterised;
- **[N]**: new, because no designer source covers it.

**Toys** (all in `docs/v3design/synth/`). Each ran as one local process under 40 MB, after `free -m`, in the develop reviewer's caregiver world (read-only import), with 5 seeds.

| toy | what it tests |
|---|---|
| `age_toy` | age codes under the delta rule |
| `hebb_toy` | the designer's rule against delta |
| `r2b_toys` | the count code under the designer's rule; release by reward-scaled links |
| `r2b_toys2` | age codes under the designer's rule with a relative readout, like for like |

---

## 0. Bottom line

1. **The core is the designer's macro architecture with very few new parts.**
   - **One area type, M1 [D]:**
     - binary units;
     - thermometer-pair inputs, half on;
     - 3-D local random sensory nets;
     - per-neuron thresholds that follow each neuron's own received signal;
     - a quantile global threshold, so the active count varies;
     - gain control;
     - mutual excitation;
     - two-layer weights.
   - **One plasticity rule, M2 [D]:**
     - temporal Hebbian learning on neurons that just changed;
     - concave, uncapped growth;
     - one global decay by depth, with birth edges exempt;
     - a floor of about 1 %;
     - slow split and delete.
   - **Wiring [D]:**
     - chain neurons to the PFC and the hippocampus;
     - a return path;
     - a hippocampus with its own population and fast weights;
     - a motor area with muscle neurons against inhibitory neurons;
     - direct inter-area links.
   - **Reward is a tool [D·p].** It scales the Hebbian change of the motor and inhibitory links that were just active.
   - **There is no separate BG, TD critic, time module, event boundary, refractory period, μ or error-driven readout.**
2. **Mapping point 1 is confirmed in the toy: LATEST comes for free from the designer's rule.**
   - Only neurons that just changed link to neurons turning on now. Own speech is heard as similar-but-different neurons.
   - So my own answer, not the question, links to the partner's next line.
   - Under the designer's rule, both age codes give **1.00 after 3,000 untutored dialogues in W1**, with 30 % silence (SIMULATED, `r2b_toys2`).
3. **The age code needs one new element.** Same rule, same relative readout; realistic caregivers, care continuing 50 %, after 3,000 untutored dialogues (SIMULATED, `r2b_toys2`):

   | age code | accuracy | echo | silent |
   |---|---|---|---|
   | the designer's literal count, "how many of the assembly's neurons are still on" | **0.01** | 0.56 | 0.43 |
   | **fatigue handover** [N] | **0.75** | 0.02 | 0.07 |

   - The handover uses unequal fatigue rates, so different units of the held assembly win at log-spaced ages.
   - It is the one new element kept, and it is flagged as **Q1**. The designer's "later-active inhibit earlier-active" (message 119) would give the same handover, but the designer called it temporary (message 122).
4. **Mapping point 2: on seen items, Hebbian competition is as sharp as NLMS.**
   - W1: 1.00 against 0.99.
   - Realistic caregivers: 0.69–0.75 against 0.68, with echo 0.01–0.02 against 0.33.
   - Corrections are 2–4× slower: changed answers right after 4 / 8 / 16 re-teachings in 0.10 / 0.25 / 0.60 of cases, against 0.65 / 0.90 / 0.85.
   - Held-out generalisation is **untested** (X2).
   - The designer says firm memories are meant to be hard to change, and that this is a matter of growth and decay parameters (message 426). So correction speed is **[D·p]**, risk R3 and question Q2.
5. **Mapping point 3: no TD critic is needed, but reward must be signed** (SIMULATED, `r2b_toys` part 2: arrivals and empty-room objects share an "onset" feature):

   | release rule | empty-room releases per tenth of life | arrival greetings |
   |---|---|---|
   | reward-only Hebbian + decay | 160 / 160 by the 3rd tenth: every object event (fails) | 1.00 |
   | **signed:** reward minus its running mean; negative grows blocking links | **0.2–3.8** | **0.6–0.9** |

   The sign comes from a running mean, which is the only [N] detail in the value path (Q3).
6. **The five FATAL items of the review stay closed:**

   | FATAL | fix | source |
   |---|---|---|
   | release cannot stop | inhibition beats muscle drive by default; release when the drive is sharp | [D] (message 44; char-neuron-voice) |
   | the seed changes the physics | a lineage-fixed physics seed | [N], a lineage lesson |
   | replay collides with the held context | replay enters the PFC only by winning the thinking-range competition | [D·p] |
   | tick-level O1 collapses | only neurons that just changed learn | [D] (plastic2) |
   | 0 errors after maturity is unreachable | babble inside the motor competition; brake and yield; only agent-caused exchange errors count | [D·p] |

7. **Kept as lessons, not mechanisms:**
   - the world models answers (formats, overheard pairs);
   - own speech in the stream;
   - the maturity rule (#20);
   - whole-model scaling (#12);
   - sealed tests;
   - a lineage with a fixed physics seed (#19).

   The designer confirms that **a world that keeps teaching** is the normal case: "在现实世界里人始终都在不停的学习外界…没人教越来越坏可能也正常" (message 426).
8. **Biggest remaining risk (mapping point 8):** abstraction and multi-step reasoning without error-driven learning. It is tested in X7 before any migration.

---

## 1. Change log: every FATAL and SERIOUS review item

| # | item (severity) | verdict | how, with source mark |
|---|---|---|---|
| develop 1 | 0 errors after maturity impossible (FATAL) | accepted | Babble is spontaneous muscle-neuron activity *inside* the inhibition competition, fading as the motor maps sharpen [D·p, char-neuron-voice]. Only agent-caused exchange errors count after maturity [N] |
| develop 2 | P5b contiguity | accepted | A response counts only if unexpected: onset × (1 − predicted) [N] |
| develop 3 | W1/W4 timing rules moved into the world | accepted | Realistic caregivers (re-ask, move-on, politeness 0.8) plus care continuing [D, message 426]. Training with long reply windows [D, message 158] |
| develop 4 | maturity currency | accepted | 300 contingent exchanges, capped at 15,000 syllables [D·p, #20] |
| develop 5 | μ replay in answer windows | closed | No μ; competition decides [D·p] |
| develop 6 | crowds | accepted | One engaged voice drives the brake [D·p, #20 brake] |
| develop 7 | #17 values under TD | accepted | Values are innate, near-unchangeable links (#17, #38) [D]; "upset" drives the inhibitory neurons directly [D·p] |
| develop 8 | disjoint self/other | accepted | Overlapping codes [D, #34]; the auditory area completes the line [D, #12] |
| develop 9 | experience per lineage | accepted | Family mixtures per sibling; L grows [D·p, #19] |
| ml 1 | release has no stop (FATAL) | closed | Inhibition dominates [D] |
| ml 2 | the seed changes the physics (FATAL) | accepted | A physics seed per lineage [N] |
| ml 3 | MIX sparsity | moot | No RandMix AND units; PFC convergence fan-in follows source activity [D·p, ana's `pfc_fan_in`] |
| ml 4 | CA3 capacity | accepted | The designer's hippocampus is short-term [D]; its horizon is sized and tested first (X4) |
| ml 5 | merge prerequisites | accepted | Alignment inside the merge [N]; one birth wiring [D, #19] |
| ml 6 | gating generalisation | closed | No source gates; the inhibitory neurons also read the pooled size of the held assembly [D·p, spec 2] |
| ml 7 | new kinds | partly | X7 first; fallback pre-registered |
| ml 8 | scaling inert | accepted | Hippocampal connectivity ∝ s [D·p, #12] |
| ml 9 | stability scope | accepted | Whole-core tests (X1–X3) |
| ml 10 | metaplasticity | closed | Concave growth **is** #22 [D] |
| neuro 1 | replay vs held context (FATAL) | closed | Competition [D·p] |
| neuro 2 / 3 | CA3 size; P5b | as ml 4 / develop 2 | – |
| neuro 4 | M4 credit, ε | moot | No M4. Reward scales only the links just active [D·p, born-wired's "this moment counts" cell] |
| neuro 5 | M2 blocks corrections | changed | The designer's rule; corrections slower [D·p]; Q2 |
| neuro 6 | ungated LATEST | closed | No loading rule; inhibitory partners keep the thought range [D, message 83] |
| neuro 7 | P4 on world codes; innate name | accepted | Sensor-physics detectors [D·p, #17]; the name is learned |
| neuro 8 | binary forms | closed | Binary, variable count [D, specs 1 and 2] |
| skeptic 1 | O1 tick vs event (FATAL) | closed | Neurons that just changed [D] |
| skeptic 2 | value loop untested | accepted | X3 runs it [N test] |
| skeptic 3 | maturity cannot be met | accepted | As develop 1; the brake also yields [D·p] |
| skeptic 4 | compute | accepted | §4 |
| skeptic 5 | representation swap last | accepted | Now first in the ladder (§5) |
| skeptic 6 | hidden items | accepted | Source gates, TIME pathway and τ_ref deleted; detectors on sensor physics |
| skeptic 7 | §6 numbers | accepted | r2b walk numbers are marked ASSUMED; toy numbers SIMULATED |

---

## 2. Conformance with DESIGNER_SPEC (all 40 rows)

Every row is answered, with the source mark in brackets.

| row | mechanism | r2b |
|---|---|---|
| 1 | binary neurons | conforms [D] |
| 2 | distributed codes | conforms [D] |
| 3 | thermometer pairs | conforms [D] |
| 4 | variable count, global threshold | conforms [D]; the age handover is re-tested under the threshold (X1) |
| 5 | random birth connections | conforms [D] |
| 6 | local connectivity | conforms [D] |
| 7 | chain neurons; always-on units not sent | conforms [D] |
| 8 | return path, rational ↔ instinct dial | conforms [D]; also restoration (#14) |
| 9 | char-network PFC | conforms [D] |
| 10 | PFC as a recurrent predictor | conforms [D], by temporal Hebbian links |
| 11 | inhibitory partners | conforms in principle [D, message 83]. ana measured no association gain (+0.010 vs +0.013), but that was a global negative-edge form. **Retest in the partner form** (X1 arm); Q4 |
| 12 | rough intention; the areas complete the sequence | conforms [D] |
| 13 | direct links; instinct vs reason | conforms [D] |
| 14 | temporal Hebbian | conforms [D] (`hebb_toy`, `r2b_toys2`) |
| 15 | concave growth | conforms [D] |
| 16 | hard to change, never frozen | conforms [D, ana's floor] |
| 17 | one global decay by depth | conforms [D·p]. v2 tune decayed NLMS readouts, not Hebbian edges, so its negative result does not apply. Retest (L9); Q5 |
| 18 | no forgetting, memory or time modules | conforms. The **handover** is a fatigue spread, not a module [N]; Q1 |
| 19 | two-layer weights | conforms [D] |
| 20 | no phenomenon-specific mechanisms | conforms |
| 21 | dopamine is a tool | conforms [D·p]; the sign from a running mean is [N]; Q3 |
| 22 | the designer's hippocampus | conforms [D]; sparse recode in its own population [D·p] |
| 23 | long-term memory = the cortex via recall | conforms [D] |
| 24 | split and delete | conforms [D], slow; local3d found little gain unless capacity is short |
| 25 | motor inhibition dominates | conforms [D] |
| 26 | blocking vs sharpness | conforms [D] |
| 27 | babbling by self-hearing | conforms [D] |
| 28 | think in sounds | conforms [D] |
| 29 | learned stop after a line ("先激活肌肉神经元，再激活抑制神经元") | conforms [D, message 101] |
| 30 | the model decides when to reply | conforms [D] |
| 31 | continuous thought | conforms [D] |
| 32 | no event boundaries | conforms [D] |
| 33 | no refractoriness | conforms [D] |
| 34 | overlapping self/other codes | conforms [D] |
| 35 | favourites are tuned | conforms [D] |
| 36 | excitability = context length | conforms [D·p] |
| 37 | local3d vision | deviates for now: thermometer world codes until local3d |
| 38 | instinct = hard-to-change links | conforms [D] |
| 39 | evolution and merging | conforms [D·p] |
| 40 | whole-model scaling | conforms [D] |

**The brief's 12 points:**

| point | covered by |
|---|---|
| binary | row 1 |
| variable count | row 4 |
| random birth | row 5 |
| chains | row 7 |
| hippocampus | row 22 |
| decay | row 17 |
| split | row 24 |
| motor | rows 25–26 |
| self/other | row 34 |
| no boundaries | row 32 |
| no refractoriness | row 33 |
| IDEAS #12–#23 | rows 15–16, 22–23 and 36–40 |

- **#22:** the designer's concave growth with the 1 % floor [D].
- **#23:** the hippocampus stores everything, with saturating growth [D].

---

## 3. The core, every element source-marked

### 3.1 Areas, M1

- **Units.** Binary units [D]. Inhibitory units are ordinary units with negative weights [D, ana appendix 5].
- **Thresholds:**
  - per-neuron thresholds that follow each neuron's own received signal [D, chain rule 4]. This replaces the ear's `h_*` rules: no event triggers, robustness through experience in noise;
  - one quantile global threshold per area, smoothed, so the count varies and strength = the number active [D, char-net #5];
  - gain control [D, char-net #8].
- **Connections:**
  - mutual excitation and spreading activation [D, #9];
  - two-layer weights [D, char-net #6].
- **Inputs:** thermometer pairs, half on; 3-D local random sensory layers, birth-fixed [D, ana kernels 1–2].
- **Fatigue spread [N].** Unequal fatigue rates among PFC units, the age handover. It is the only new dynamic. Evidence is in §0.3; question Q1.
- **Not used:**
  - fatigue for moving the thought on in general. The designer: a stuck thought means too few rich connections (messages 80, 86) [D];
  - refractoriness [D, #33].

### 3.2 One rule, M2

- **Temporal Hebbian learning** on neurons that just changed [D, plastic2]:
  - falling → rising strengthens;
  - the reverse order weakens;
  - co-rising units strengthen each other;
  - always-on units grow nothing.
- **Concave growth:** Δw = η·a_i·a_j/(1 + w/w_ref)^pow [D].
- **One global decay** by depth (ana's table), with birth edges exempt [D·p].
- **Floor:** about 1 % [D].
- **Split and delete:** slow, rarer near the input [D].
- **Reward as a tool [D·p]:**
  - Reward multiplies the change of motor and inhibitory links that were just active, as in born-wired's "this moment counts" cell.
  - Its sign comes from the reward's running mean [N, needed by `r2b_toys` part 2].

### 3.3 Wiring

| part | what it does | source |
|---|---|---|
| chain neurons | every layer of every area sends 1:1 copies to the PFC and the hippocampus; always-on units are filtered out | [D] |
| return path | PFC → sensory layers, never L1; forward weight > return weight (message 119) | [D] |
| hippocampus | its own chain-fed population with fast growth and decay; short-term only; a sparse recode (DG-like) through its own threshold | [D]; [D·p] for the recode, mapping point 5 |
| meaning | PFC association clusters, like the char-net's food cluster; no meaning layer | [D·p], mapping point 5 |
| motor | muscle neurons against inhibitory neurons; inhibition wins by default; selection among competitors by lateral inhibition; reward-shaped release; learned stop | [D] (messages 44, 101; char-neuron-voice) |
| PFC input gating | the PFC's own inhibitory partners | [D·p], mapping point 3 |
| brake | the engaged voice drives the inhibitory neurons, including yielding at the next syllable | [D·p] (#20 brake) |
| innate values and detectors | P4/P5 on sensor physics; near-unchangeable links | [D] (#17, #38) |
| contingency | a response counts only if it was unexpected | [N] |
| maturity counting | as in §1, develop 4 | [N] |

### 3.4 How the behaviours come out (ASSUMED until X1–X3)

- **(a) Answering.** Q's assembly is held; its age is carried by the handover.
  - A format answer heard at τ ≈ 10 links Q_τ → A, by concave increments 0.10 / 0.19 / 0.27.
  - Later, Q spreads to A. A is heard inwardly through the return path, then sharpens the muscles, and is released when the drive beats inhibition.
  - The auditory area completes 红 → 色. The learned stop links the line end to the inhibitory neurons.
- **(b) Restoration.** The return path biases a degraded syllable. Experience grows the predictions (#14).
- **(c) Greeting vs empty room.** Rewarded releases grow drive links. An unrewarded release falls below the running mean and grows blocking links (SIMULATED: 0.2–3.8 empty-room releases per tenth).
- **(d) No talk-over.** The brake reads the engaged voice. An overlapped late start brings no unexpected reply, so blocking grows.

---

## 4. Numbers (unchanged from r2)

- **Compute:**
  - s = 1: 1,000–2,000 s per 10⁶-tick life on CPU, from the skeptic's measured ticks;
  - s = 4: with L ∝ s, 16,000–32,000 s per life, i.e. 3–6 chained jobs;
  - GPU batching is unproven (gpuport: 259 ms per tick at batch 64);
  - lineage: 30–45 h on CPU.
- **Knobs:** about 146 in v2.2 → about 70 in r2b (about 30 genes). This is an estimate; the freeze lint does the census.
- **Concave weights:** w = √(1 + 0.2n) − 1, giving 0.10 / 0.41 / 1.19 at n = 1 / 5 / 19 (toy constants).

## 5. Migration ladder: mapping tests first, then the swap, then the lineage

| step | what | pass |
|---|---|---|
| T1–T3 | **X1–X3 below** (local, minutes to hours) | each meets its prediction |
| T4 | **X4**, the hippocampus in the designer's form | recall ≥ 0.9 within the horizon |
| L1 | thermometer ear, per-neuron thresholds, sound assemblies, chain neurons | hearing tables within 0.05 of `h_*` at σ ≤ 1 |
| L2 | **X5 merge** and **X6 tick benchmark** | disjoint lessons ≥ 0.9; budget confirmed |
| L3 | char-network PFC + M2 + handover (or the count code, if X1 passes it) in the bench agent | untutored ≥ 0.85 at 3,000 and 10,000 with care continuing; echo ≤ 0.1; J2 within SD of v2.2 on the new world |
| L4 | motor (inhibition vs drive, babbling, brake, learned stop) | ≥ 90 % of lexicon bound by 20 % of life; 0 post-maturity agent errors |
| L5 | values (#17 tests) | helps; declines harm |
| L6 | decay by depth and split/delete retests | no loss |
| L7 | development freeze; lineage gen 0–2 with a Darwinian control | per-generation curves |
| L8 | shortcut verification | final freeze and one sealed opening |

## 6. First experiments (front-loaded, local)

| # | harness | arms | prediction | falsifier |
|---|---|---|---|---|
| **X1** (mapping point 1) | real symbolic harness (`gen_probe_v21` copy) with an r2b PFC | age: count code vs handover vs handover under the global threshold; silence 0 / 30 %; W1 / realistic caregivers / care on-off; inhibitory partners on/off; multi-syllable answers; replay that takes over vs mixes | the count code holds in W1 and fails realistically, as in the toy; handover ≥ 0.7 realistic | the count code ≥ 0.7 realistic: drop the [N] handover |
| **X2** (mapping point 2) | the same, plus held-out frame × filler pairs | the designer's Hebbian vs NLMS | Hebbian ≈ NLMS on seen items; held-out unknown | Hebbian held-out < NLMS − 0.15: the competition is not sharp enough |
| **X3** (mapping point 3) | the life greeting world (symbolic) | reward-only vs signed reward; shared onset features | signed ≤ 15 empty-room releases per tenth; reward-only fails | signed > 15 per tenth |
| **X4** | the designer's hippocampus on a recorded v2.2 stream | horizon, sparse recode on/off | recall ≥ 0.9 within the horizon; recode removes misrecall by identity | < 0.7 |
| X5 | merge of siblings | alignment on/off | disjoint lessons ≥ 0.9 | < 0.7 |
| X6 | tick benchmark | s = 1, 4; CPU and torch | within §4 | GPU not faster: CPU only |
| **X7** (mapping point 8) | depth in space vs time (learning member's E2) | as before | (d) ≥ 0.7 | both ≤ (b) + 0.05 while backprop ≥ 0.7 |

## 7. Open risks, ranked

1. **Abstraction and multi-step reasoning without error-driven learning** (X7).
2. **The age code.** The designer's literal count fails with realistic caregivers in the toy. The kept handover is [N] and needs the designer's consent (Q1).
3. **Correction speed** is 2–4× slower. That is "learn once, keep", parameterised (Q2).
4. **Held-out generalisation** under Hebbian competition is untested (X2).
5. **The hippocampus horizon** against long-delay memory (X4).
6. **Moving thoughts without fatigue.** The designer's claim is that rich connections suffice; it is untested.
7. **0 turn-taking errors after maturity** under inhibition vs drive is untested (L4).
8. **Compute and GPU.**
9. **Merging distributed assemblies.**
10. **Values under evolution; crowds.**

## 8. Questions for the designer

- **Q1.** "How many neurons are still on" fails with realistic caregivers in the toy (0.01). Unequal fatigue among PFC units (handover) works (0.75). Is that acceptable? The alternative is your "later-active inhibit earlier-active", which you called temporary.
- **Q2.** Corrections are 2–4× slower under your rule. Is that acceptable, as "firm memories are meant to be hard to change"?
- **Q3.** Release learning needs a signed reward: a release that brings less than the usual reward grows inhibitory links. Is a running-mean baseline acceptable as part of "reward as a tool"?
- **Q4.** Retest your inhibitory partners in the partner form, for distractor resistance and for choosing between competing answers?
- **Q5.** Retest the global decay in your form (Hebbian edges)? v2's negative result decayed error-driven readouts.

---

## summary_zh

1. **这一版完全以您的整体架构为准：**
   - 感觉区是立体局部随机网络，输入一半亮；
   - 链式神经元送到前额叶和海马体；
   - 前额叶是文字网络式的循环预测网络；
   - 海马体是一群自己的神经元，连接长得快、掉得也快；
   - 有回程；
   - 运动区是肌肉神经元对抗抑制神经元；
   - 区与区之间有直连。
2. 学习只有一条：时序赫布（只有刚变化的神经元参与）、越长越难长、全脑统一降权、两层权重；奖励只是工具。没有基底节模块、时间模块、事件边界、不应期。
3. **我逐条查了您原来的说法：** "说完一句就先激活肌肉、再激活抑制神经元"直接照用；"想法卡住是因为连接不够丰富"所以不加通用疲劳；"现实世界一直在教"也就是大人要持续教。
4. **小实验（5 个种子）：**
   - 按您的规则，"自己说的话进入输入流"自然做到"最近一件事"，无人指导 3,000 段后正确率 1.00。
   - 但按字面"还亮着几个神经元表示过了多久"，在真实大人行为（重问、跳过）下会崩到 0.01；让神经元疲劳快慢不一、轮流占上风，则是 0.75。这是本版唯一新加的东西，请您决定能不能用（或用您说过的"后亮抑制先亮"，但您说那是临时办法）。
5. **您的学习规则：** 已学会的答案不忘、不乱说对方的话（0.01 对 0.33），但改正学错的答案要多 2–4 倍教学次数，这符合"学会一次就不忘"。
6. **奖励作工具就够了，不需要额外的价值评估器。** 但奖励要有正负：开口后得到的比平常少，就加强抑制连接；否则空房间里的打招呼会越来越多。
7. **最大风险：** 没有误差修正的学习，能不能学出抽象和多步推理。接下来先用小实验测这几件事，再改真代码。
