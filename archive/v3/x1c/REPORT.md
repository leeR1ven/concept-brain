(Saved by the coordinator from the agent's final hand-back. In the repo, the per-run results that the code writes to `out/` are copied to `results/`.)

# X1c report: does the designer's own core work on a continuous dialogue stream?

Labels: **PROVED** = code, counters or exact numbers from runs; **SIMULATED** = a toy or pilot run (file named); **CITED** = a source; **ASSUMED** = my judgement, not tested.
Source marks: **[D]** the designer's; **[D·p]** the designer's, parameterised; **[N]** new.

## 0. Bottom line

1. **No arm learned to answer.** I built a continuous, tick-by-tick world and core and ran 12 arms × 5 seeds (60 lives). PROVED (`out/*.json`, `out/AGG.json`):
   - accuracy is **0.00 ± 0.00 in every arm**: seen pairs, held-out pairs, after teaching, and through untutored use (3,000 exchanges for B, B_real, B_loop; 1,000 for the others);
   - **first-syllable accuracy is also 0.00**;
   - corrections: no arm ever produced a changed answer in 300 correction exchanges (about 11–15 re-teachings per changed pair). Latency is undefined, because the old answers were never produced either;
   - what the agent actually says is **mostly garble**: 0.4–2.5 garbled utterances per exchange, plus silence. In probes it often talks during the question (0.4–1.0 of probes). Echo is about 0 (≤ 0.02).
2. **The activity-control question** (the coordinator's arms), PROVED, 5 seeds, whole run:
   - **(C) no PFC inhibition, thresholds a rising fraction of each neuron's in-weight sum.** At birth (SIMULATED, `dyn3.py`) it is bounded during speech (L2 6–16 %, L3 3–10 %) but **dies out** in silence. Once the top→L1 predictions add drive to L1 it **runs away**: L2 and L3 average 398 of 400 units and are saturated 99 % of ticks. The thought is frozen (1 distinct state). It never speaks.
   - **(C2) the designer's corrected form: per-neuron thresholds from each neuron's own received signal, count preserved layer to layer, no PFC inhibition.** It is **unbounded both ways**: L2 is empty 62 % of ticks and saturated 30 %; L3 is empty 67 % and saturated 32 %. It costs 4.4 ms per tick.
   - So under C, PFC activity does **not** stay bounded when the prediction links add drive on top of the chain input. Of the listed failure modes, both "runs away" and "dies out" occur. By undecided item 10, inhibition is therefore needed.
   - **(A) local feedback inhibition + bounded homeostatic thresholds + partners.** Bounded on average (L3 mean 39) but with **bursts up to 370 of 400**. Thoughts keep moving (stuck20 0.40–0.45; about 500–650 distinct states). It babbles: 2.0 garbled utterances per exchange in teaching, 0.9 in untutored use. Cost 1.5 ms per tick.
   - **(B) global quantile threshold.** Bounded by construction (L3 36 of 400, never empty, never saturated), but **the thought is stuck**: stuck20 0.87–0.88, turnover over 5 ticks 0.03.
   - **(B_loop) the coordinator's fix: κ = 0, a random birth loop top→L1 (gain 8, tuned once on seed 0), lateral w33 = 0.4.** Activity stays alive with κ = 0 (L3 33 of 400, empty 1 % of ticks), and **thoughts move**: stuck20 0.47–0.56, turnover5 0.16–0.18, about 1,157 distinct states against about 115–150 for B. Answers are still 0.00, and it garbles more (2.5 per exchange in teaching).
3. **Diagnosis.** The core does not bridge question → pause → answer. Under the binding learning rule, held units grow nothing, so the units that carry the question through the pause cannot link to the later answer. Section 5 gives the mechanism. This should go to the designer as a question.

## 1. What was built (final `core3.py`, `world.py`, `run.py`)

**Layer counts and sparsities (PROVED, B arm, whole-run means):**

| layer | size | active (mean) | notes |
|---|---|---|---|
| A0 auditory input | 360 (180 on/off pairs) | exactly 180 on (50 %) | thermometer on/off pairs |
| L1 | 212 chain lines | 18.8 (9 %) | 150 content on-units + 30 voice on-units + 24 muscle copies + 8 stop-neuron copies; off-units filtered as always on |
| L2 | 400 | 18.7 | target 30 (7.5 %) |
| L3 (top) | 400 | 36.2 | target 40 (10 %) |

Roughly count-preserving, as the designer later asked.

### World [continuous stream; no symbolic Q/A/age events reach the brain]

- **Stream** [D: IDEAS #24.1; ARCH §二.11]. Each syllable lasts 3 ticks; caregivers speak, pause and react. The brain receives only a binary auditory vector per tick plus innate value detectors.
- **Sounds** [D #34]: 34 syllables, each 10 of 150 content units. Each voice has its own 6 voice units, and content is shared between own and other voices.
- **Task** [N]: 4 frames × 8 fillers with compositional answers of 1–3 syllables (香 / 不能吃, 要 / 不要, colour+的, 大 / 小). 10 pairs are held out. Narration (30 %) supplies the colour and size facts that the held-out pairs need.
- **Caregivers** [D·p: statistics from `caregiver_toy.py`; ARCH §二.11]:
  - format: answers herself after 12 ticks;
  - realistic: two voices, re-ask 30 %, move-on 15 %, delay jitter ±50 %, overheard adult Q→A 15 %;
  - corrections restate both question and answer (ARCH §二.11);
  - politeness 0.8.
- **Values** [D: IDEAS #24.3]: pleased +1 (p 0.8 after a correct answer); upset −1 (p 0.25 after a wrong answer, p 0.5 after an interruption); hunger relief when the child says 要 to offered food. **The amount of response is never used.** The probabilities are [N].
- **Other conditions** [N]:
  - empty room: 300 ticks every 60 exchanges;
  - distractors: far voices at loudness 0.4 (signal strength = active count, [D #4]);
  - body: 24 muscles, a 4-muscle set per syllable, mouth rule clear / mumbled / garble [D·p: bench / voice-repo body]; own voice heard 1 tick later.

### Brain

- **Neurons and input** [D §二.1, §二.3]: binary neurons. A0 is the on/off-pair layer. 1:1 chain pairs (forward and return) exist for feature neurons only; units on more than 60 % of the time are cut [cut value D·p].
- **Layered PFC** [D, binding of 11-01]: L1 = copies (forward chain, or internally activated). L2 and L3 have local random feed-forward weights that are fixed. The top→L1 Hebbian prediction links are local (only nearby L1, about 21 % of pairs). Lateral L3 links are random at birth and not learned [D, binding #2]. The optional "Hebbian middle layers" arm was **not run**.
- **One rule** [D: msg 48, msg 50, plastic2; ARCH §二.8]:
  - when j rises, units already on (or fallen within 2 ticks) strengthen →j;
  - when j falls, units held across its fall weaken →j;
  - co-rising units strengthen each other;
  - growth is concave, Δw = η/(1+w/w_ref), with a 1 % floor;
  - one global multiplicative decay by depth plus pruning [D·p: 0.998 per 100 ticks on top→L1]; birth links exempt.
- **Two-layer weights** [D char-net]: propagation divided by out^0.5 · in^0.5.
- **Output only through return chains** [D, binding]:
  - auditory return = inner hearing, which reaches the muscles through the direct auditory→motor link;
  - motor return = intention to act;
  - PFC return to the phasic stop (inhibitory) neurons = intention to stop. These stop neurons have chain pairs; the tonic blockers, which are always on, have none [D·p].
  - The direct auditory→motor link also acts on heard sound without the PFC [D binding]. It is a **birth-given stand-in for the babbling result**, not learned [D·p, deviation].
- **Motor area**: tonic blocking wins by default [D]. Release depends on sharpness, via divisive gain plus selection relative to the maximum [D·p, char-neuron-voice]. The innate brake is driven by a loud partner voice [D·p #20].
- **Values** amplify the just-made top→muscle-copy increments; upset weakens them and links the release context to the stop copies [D·p §二.8].
- **Inner hearing** [D·p]:
  - threshold = a fraction of each neuron's own top-down in-weight;
  - internally active L1 units drive L2 at half strength ("return ≤ half of forward", ARCH §5).
- **Comparison mechanisms**:
  - homeostatic per-neuron thresholds, bounded [D·p §二.1; the bound is N];
  - inhibitory partners [D msg 27];
  - fast synaptic component [D·p #24.2];
  - hippocampus: own population, chain-fed, fast links, recall re-lights sensory and motor neurons through return chains [D binding #3; thresholds N];
  - NLMS readout (comparison only).
- **Self-hold κ in B** [N]. The coordinator correctly flagged it as not in the spec. B_loop removes it.

## 2. Problems met, sources checked, decisions (the designer's working rule)

| # | problem (SIMULATED, seed-0 pilots unless noted) | sources checked | decision |
|---|---|---|---|
| 1 | Flat random PFC pool is chaotic: the same question gives Jaccard 0.2–0.3 at its end and about 0 after 10 ticks (`dyn.py`, `core.py`) | SPEC #9, char-net README, msgs 83/84 | superseded by the binding layered PFC |
| 2 | Equal birth weights cause threshold ties, which history resolves | ana kernels ("random at birth") | continuous random weights [D·p] |
| 3 | Persistence on chain-copy units held the last heard syllable as fake imagery, causing echo | char-net decay applies to thought units; §5 return weaker | persistence on the top layer only |
| 4 | Smoothed quantile oscillates (counts 8 → 144) and drops held units at sound offsets | char-net θ_lr 0.5; ana level-wise calibration | count bounded to [k/2, 2k] [D·p], later exact tracking with a floor (B is a shortcut, item 10) |
| 5 | Stale predictions of held question syllables (echo risk) | msgs 48, 50; plastic2 | rule as in §1 [D] |
| 6 | Unbounded homeostasis: silent units' thresholds fall until they fire regardless of input; the top state collapses to one shared state (cross-question Jaccard 0.92) | ARCH §二.1 | bounded offsets [N] |
| 7 | Imagination drowns input (L1 imagery about 40 units against about 16 heard) | ARCH §5 "回程边不超过前向边的一半" | internal L1 drives L2 at 0.5; relative imagery threshold |
| 8 | Always-on blocking-neuron copies feed a constant L1 signal | ARCH §二.3 | no chain for tonic blockers; phasic stop neurons with chains |
| 9 | No question→answer link across the 12-tick pause | IDEAS #24.1, msg 158, ARCH §五 ("the world must demonstrate Q then A") | tested a 4-tick self-answer: still 0 |
| 10 | Coordinator: stuck thought caused by κ and the empty birth loop | ARCH §二.4 | B_loop; gain tuned on seed 0 (table below) |

**B_loop tuning on seed 0** (300 exchanges, empty room, SIMULATED `tune_loop.py`):

| g_loop | stuck20 | distinct states | probe garble |
|---|---|---|---|
| 0 | 0.59 | 478 | 0.09 |
| 2 | 0.59 | 407 | 0.41 |
| 4 | 0.45 | 665 | 0.46 |
| 8 | 0.29 | 954 | 1.00 |
| 16 | 0.40 | 744 | 1.00 |

- κ = 0 alone does **not** kill activity under the global threshold, so no minimum loop gain was needed to keep it alive.
- I chose g = 8 as the most moving. It also gives the most garble.

## 3. Results (5 seeds each, mean ± SD; PROVED from `out/AGG.json`)

### 3.1 Answers, and speech during teaching and untutored use

Teaching: probe accuracy on seen pairs was 0.00 and on held-out pairs 0.00 in all 12 arms, so those columns are omitted.

| arm | silent / garble in probes | talk during Q | garble per exchange (teach) | extra turns per exchange (teach) | untutored (first 300): response rate / garble per exchange |
|---|---|---|---|---|---|
| B | 0.58 / 0.34 | 0.42 | 1.59 | 1.19 | 0.50 / 0.48 |
| B_real | 0.45 / 0.53 | 0.62 | 1.59 | 1.13 | 0.47 / 0.46 |
| B_nlms | 0.60 / 0.40 | 0.42 | 0.78 | 0.54 | 0.46 / 0.45 |
| B_nopart | 0.23 / 0.76 | 0.80 | 1.78 | 1.22 | 0.59 / 0.62 |
| B_linear | 0.41 / 0.59 | 0.60 | 2.11 | 1.51 | 0.47 / 0.48 |
| B_fast | 0.23 / 0.77 | 0.64 | 2.49 | 1.77 | 0.64 / 0.66 |
| B_hip | 0.40 / 0.60 | 0.60 | 1.90 | 1.34 | 0.49 / 0.49 |
| B_dist | 0.71 / 0.29 | 0.40 | 1.87 | 1.31 | 0.45 / 0.57 |
| B_loop | 0.37 / 0.63 | 0.81 | 2.49 | 1.67 | 0.77 / 0.91 |
| A | 0.37 / 0.63 | 1.00 | 2.05 | 1.51 | 0.54 / 0.86 |
| C / C2 | 1.00 / 0.00 | 0 | ≈ 0 | ≈ 0 | 0 / 0 |

- **Echo** (saying partner lines) ≤ 0.02 everywhere.
- **Held-out frame × filler**: 0.00 everywhere.
- **Drift**: untutored accuracy stays 0.00, so drift cannot be measured.
- **Hebbian vs NLMS**: equal, both 0.00.
- **Partners on/off**: no accuracy difference; with partners off the agent garbles more and talks during the question more (0.80 against 0.42).
- **Two correction mechanisms**: no effect, since nothing was learned in the first place. In B_hip the recall route **never fired** (0.00 of speech onsets), so that arm did **not** test recall (PROVED from the route counter).

### 3.2 Mechanics

There is no maturity, because the agent never answers. Rates were measured through teaching and untutored use (PROVED):
- talk-over: 0.01–0.03 per exchange;
- **failure to yield: 0.00 in every arm** (the innate brake works);
- garble: 0.4–2.5 per exchange;
- extra turns inside exchanges: 0.3–1.8 per exchange.

### 3.3 Empty-room speech under innate-values-only reward (reported, not patched)

Utterances per 100 empty-room ticks, teaching / untutored:

| arm | teaching | untutored |
|---|---|---|
| B | 0.48 | 0.02 |
| B_loop | 0.52 | 0.39 |
| A | 0.94 | 0.66 |
| B_dist | 1.09 | 0.69 |
| B_real | 0.25 | 0.02 |
| C, C2 | ≈ 0 | ≈ 0 |

It is all garble (for example `~~蕉~~~`).

### 3.4 Speech routes (share of motor drive at each speech onset; PROVED)

| route | share |
|---|---|
| PFC motor return (intention) | 0.53–0.80 |
| PFC inner hearing → direct auditory→motor link | 0.17–0.45 |
| heard sound driving the muscles directly (reflex) | 0.02–0.03 (0.16 with distractors, 0.32 in C2) |
| hippocampal recall | 0.00 |

### 3.5 Thought motion without refractoriness (empty room)

| arm | stuck20 | turnover5 | moves? |
|---|---|---|---|
| B (κ self-hold) | 0.87–0.88 | 0.03 | no |
| B_real | 0.90 | 0.02 | no |
| B_loop | 0.47–0.56 | 0.16–0.18 | yes, partly |
| A | 0.40–0.45 | 0.31–0.33 | yes |
| B_dist | 0.64 | 0.05 | partly, driven by the distractor voices |
| C | 1.00 | — | frozen at saturation |
| C2 | 0.35–0.58 | — | oscillates |

### 3.6 Cost (PROVED)

- 0.7–1.5 ms per tick; C2 4.4 ms (saturated layers).
- One life of 600 + 300 + 3,000 exchanges takes 2.5–3.6 min.
- Peak RSS ≤ 70 MB, one process at a time.

## 4. Records with exact numbers (SIMULATED, seed 0, `dyn3.py` pilots)

- **Layered top state at birth, at question end**: Jaccard 0.42–0.61 for the same question, against 0.33–0.52 for the same frame with another filler. It is frame-specific; the filler is weakly represented.
- **After 300–800 teaching exchanges**: across most variants the state either collapsed (cross-question Jaccard 0.8–1.0, with one constant "hub" inner-hearing syllable predicted after every question) or became unspecific (Jaccard 0.2–0.5).
- **Format self-answer delay of 4 ticks instead of 12**: still 0.

## 5. Diagnosis

1. **The rule and the hold conflict** (PROVED from the rule's form; consequence SIMULATED).
   - The binding rule gives no link from units held across an event: held units count as "A then B then A" (msg 50), and "always-on units grow nothing".
   - To answer after a pause, the question has to be held, and the held units are exactly the ones excluded.
   - The only units that change at the answer's onset are the subset the answer displaces. That subset depends on history, so the question→answer link is noise.
2. **Unspecific predictions feed back** (SIMULATED).
   - Once the top→L1 links exist but are not question-specific, their inner hearing floods L1. The top state then collapses, or the agent babbles: 53–80 % of speech comes from PFC motor intentions that carry no answer.
   - Returning internal activity at half strength and using the relative threshold reduced this but did not remove it.
3. **Holding vs moving trades off at this scale.**
   - A self-hold gives a stuck thought (B).
   - Removing it and adding the birth loop gives moving but unspecific thought (B_loop, A).
   - Neither gives an input-locked, reproducible trajectory that could carry the question to its answer.

**Possible ways forward (ASSUMED, untested; for the designer):**
- (a) Measure "just changed" from the earlier unit's own onset, within a longer window. My pilot of that form brought back hub predictions.
- (b) Reproducible, input-locked moving trajectories, so that question-carrying units change near the answer.
- (c) The designer's view that the model is too small: scale up.
- (d) A learned babbling route, so speech is not driven by a fixed stand-in.

## 6. Limits and deviations

- Teaching was 600 exchanges. Pilots up to 800 showed no trend toward answers.
- Untutored use was 1,000 exchanges for 9 arms. A run.py bug repeats the 1000 mark when n_free = 1000, so their last mechanics window was empty; those mechanics come from the first 300 untutored exchanges. Probes and the empty-room and motion statistics are unaffected.
- **The hippocampus arm did not exercise recall**: its threshold was never reached.
- The direct auditory→motor link is a stand-in, not learned.
- No Hebbian-middle-layer arm was run.
- The earlier `core.py` (flat pool) and `core2.py` are kept only as records of the pilots.
- The repo was not modified. One local process at a time, ≤ 70 MB.

## 7. Files

All in this folder (`v3/x1c/`). The pilot cores `core.py` and `core2.py` and the pilots `dyn.py`, `dyn2.py` and `dyn4.py` were not copied into the repo:
- final core: `core3.py`
- world: `world.py`
- driver and arm definitions: `run.py`
- batches: `batch.py`, `batch2.py`
- aggregator: `agg.py`
- loop-gain tuner: `tune_loop.py`
- pilot scripts: `dyn3.py` (and the earlier `dyn.py`, `dyn2.py`, `dyn4.py` in the same folder)
- per-run results: `results/` (60 run files)
- aggregate: `results/AGG.json`, text form `agg_all.txt`
- logs: `batch.log`, `batch2.log`

## summary_zh

1. **做了什么：**
   - 按 ARCHITECTURE.md（中途收到的 8 次更正都已照做）搭了一个连续的世界和连续的网络：每拍一个声音，大人说话、停顿、回应；
   - 听觉区是正反对编码；链式神经元成对（去程、回程）；前额叶分三层，中间层是局部随机连接，顶层到第一层做局部赫布预测；只有一条学习规则；
   - 运动区里抑制默认占上风；先天刹车；价值只来自先天（高兴、难过、吃饱）；
   - 共 12 组、每组 5 个种子。
2. **结果：** 所有组的正确率都是 0（练过的、没见过的组合、无人教之后都是 0）；改正答案也一次都没学会。孩子说出来的大多是乱码，或者不说。
3. **前额叶怎么控制激活量：**
   - 不用抑制神经元（C、C2）：一加上"顶层预测第一层"，就会全部亮满，或者全亮和全灭来回跳，不受控；
   - 局部抑制（A）：大体有界，偶尔爆发，想法会动；
   - 全局阈值（B）：有界，但想法卡住（自我保持项造成的）；
   - 去掉自我保持、加出生随机回路（B_loop）：活动不会熄灭，想法动起来了（卡住比例 0.87 → 0.5），但仍然答不对。
4. **根本原因：** 规则规定"一直亮着的神经元不长连接"。可要在停顿之后回答，问题必须一直保持着，而保持住问题的那些神经元正好被这条规则排除。答案到来时变化的，只是被答案挤掉的那一小部分，跟问题关系不大，所以"问题→答案"连不起来。预测不针对具体问题时，还会反过来淹没输入，变成乱说。
5. **其他测量：**
   - 说话的来源：前额叶回程到运动区占 53–80%，内心声音经直连到运动区占 17–45%，直接听到就说约 2%；
   - 对方开口时都能让出话轮（失败为 0）；
   - 空房间里会说一点乱码；
   - 每拍 0.7–1.5 毫秒。
6. **需要设计者定：** 停顿期间一直亮着的神经元，能不能和之后到来的答案建立连接？或者，想法要怎样"每次走同一条路"？