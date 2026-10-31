(Saved by the coordinator from the reviewer's final reply; the harness blocked the reviewer's own write.)

# Review round 2, skeptic lens: the whole ARCH_v2 architecture

The harness blocked the write to `arch/v2/review2/skeptic/report.md`, so the full report is below.

**My files** (in `arch/v2/review2/skeptic/`):
- `drift_own.py` and `drift_own.log`: my only run. One process, about 210 MB, about 6.5 min, on the saved seed-0 agent `synth/out/agent_final_s0.pkl`.
- `fav_base.py`: reads the life log; takes about 1 s.

Labels: **PROVED** (argument with numbers), **SIMULATED** (my run), **CITED** (paper), **ASSUMED**.

## 0. Verdict

The ARCH_v2 numbers are honestly logged and reproducible. My run reproduced the seed-0 drift: C 0.79 → 0.51. But the numbers measure less than the text claims.

In function, ARCH_v2 is three things:
- a conjunction table keyed on exact tuples of discrete syllable IDs (`MixDict` keys such as `('P', item, 'L', item)`, read out by an NLMS rule);
- a perfect symbolic replay buffer (the hippocampus stores Python lists, and recall copies them verbatim);
- a turn-taking scaffold whose timing constants were read off the bench's source code.

Its competence rests on three things the world supplies:
1. **A tutor whose next line is the right answer** (recasts).
2. **Pre-digested senses:**
   - syllables segmented to exactly 3 ticks;
   - an eye code that is factorised and translation-invariant, with fixed sub-blocks for colour, count, shape and "pointing";
   - an articulation table that babbling reads offline.
3. **Massed drill:** 3,300 dialogues of the same 12 frames.

Remove any one of these and the existing files already show results at chance or below trivial baselines:
- untutored use: C falls 0.79 → 0.31 after 900 dialogues (my run);
- held-out fillers;
- life mode;
- world2.

Do not port to GPU before the changes in §3. A faster copy of this design mostly scales up the lookup table.

## 1. ARCH_v2 vs the real brain

| Difference | Matters for the goal? |
|---|---|
| **Own speech is invisible.** `ct_own=0` keeps own lines out of the thought. `end_line` stores only heard lines, and the hippocampal NEXT links skip the agent's own turns (`core_synth.py:316-325, 426-448`). In the brain, efference copy plus hearing yourself make what you said part of context. | **Critical.** It is the root of free-use drift and of incoherence across turns (§2.1). |
| **Symbols after perception.** Items are category IDs, and MIX units are exact ID tuples (`rules.py:151-217`). No distributed or overlapping code exists downstream of Z. | **Yes.** There is no transfer of even a constant answer to a new filler (§2.2). This contradicts IDEAS #5 and #14. |
| **No recurrent PFC state.** The "thought" is the bag of items in the last heard line (register-equivalent by the designer's own test), plus explicit position and last-item features. | **Yes.** It blocks context beyond one line, IDEAS #1 and #15, and use of an adult corpus. |
| **The hippocampus is a lossless database.** Records are exact `dict(items, eye, next pointer)` (`rules.py:254-262`, `core_think.py:143-153`). There is no interference and no reconsolidation, and the effective horizon is about 6,000 lines (§2.6). | **Partly.** It flatters the recall numbers, and the drift brake depends on it. |
| **Hand-fitted hearing.** The syllable clock samples at onset+2+3k because the world's SOUND is 3. Lines are segmented by silence. Vigilance is fixed and there is no gain control. | **Yes, for world2 and real audio.** In world2 the categories hit their cap within one day (ARCH_v2 §8.4). |
| **Vision supplied by the world.** In `retino`, every slot block carries an identical code with fixed sub-ranges (`world.py` Curriculum: shape 0–159, colour 160–207, count 208–231, pointing 232–239). | **Yes, for real images (IDEAS #16).** The colour and count "solutions" depend on it (§2.4). |
| **Motor learning is a table lookup.** Babbling enumerates all 35,960 muscle sets through `voice.decode`, synthesises its own hearing with `voice.hear`, and de-duplicates by the world's character label (`agent.py:209-257`). | **Yes, for any new body or voice**, including world2. |
| **Fixed thinking and timing.** Thinking is a fixed 3-step pipeline, and release is forced by τ=7. In the brain, the basal ganglia choose cognitive operations and can hold or plan. | **Yes**, for "speak because it thinks". |
| **Value is one exact scalar.** Reward is `reply in acceptable` (`world.py:722`). | Medium. It matters for IDEAS #17 and for a life without a tutor. |
| No sleep, no spikes, Dale's law only as a note, a single body. | **No.** Sleep is excluded by the designer's decision; the rest is detail. |

## 2. Problems of the whole architecture

### 2.1 The teacher is built into the readout

**Mechanism.** Speaking means predicting the partner's next line, and the agent's own speech is invisible. GEN learns each heard item from the thought held at line onset. The thought holds only lines by other speakers:
- `own_finish` returns before anything enters (`core_synth.py:316-325`);
- the episodic NEXT pointer follows `last_rec`, the last heard line (`core_synth.py:443-447`).

So in "partner Q1 → agent A1 → partner L2", L2 is learned as the continuation of Q1, whatever A1 was.
- **With a recast tutor**, L2 is the correct answer. This is supervised learning with the label supplied by the world.
- **With any other partner**, the agent learns "after Q1, say L2" (a new question or a comment). R2 then weakens the old answer: predicted but absent → weaken.

ARCH_v2 §12.1 says "no timing cue can separate" recast from next question. That is not the real cause. The cause is that the agent's own turn is deleted from its context. The difference is perceivable: a recast repeats or corrects A1 (efference copy), and a new question does not.

**Evidence: drift over 5 seeds (PROVED, `out/runs/final_noctx_s*.json`, n=789).** C after 300 untutored dialogues:

| task | before | after |
|---|---|---|
| ALL | 0.804 | 0.541 |
| colour | 0.885 | 0.497 |
| offer | 0.789 | 0.479 |
| count | 0.786 | 0.565 |

**Evidence: longer untutored use and own speech in the thought (SIMULATED, `drift_own.log`; seed-0 agent, C probe n=159).**
- **As ARCH_v2:** 0.79 → 0.51 after 300 dialogues → **0.31 after 900**. Ritual falls 0.93 → 0.21 and taste 0.67 → 0.24. Drift keeps going: awake recall delays it but does not stop it.
- **Own speech in the thought** (`ct_own=0.6`, no retraining): 0.79 → **0.57** after 300 dialogues, versus 0.51. The loss is −0.23 vs −0.28, about 1.5 SE; colour 0.39 → 0.50, count 0.43 → 0.60.
- So adding own speech to the context helps a little. Most of the loss remains, because the keys of the question are still taught the next question.
- The missing part is routing learning by contingency (§3.1).

**Evidence: facts in life (PROVED, `out/life/final_noctx_s5.json`).**
- All life memory facts score 0.00 (0/381), including `fact_name` at 9 or more encounters (0/51).
- A statement such as 我的小狗叫N is never followed by its question and answer. A learner that only predicts the next line cannot use it.
- Most adult speech is statements of this kind.
- "No drift in life" holds only because 908 of that life's 1,731 QA exchanges were recasts.

### 2.2 "Generalisation" is lookup: the frame never generalises

**Evidence (PROVED, 5 seeds, held-out pairs):**
- **taste** (X好不好吃): **0/30**. 好吃 is acceptable for every food.
- **offer** (你要X吗): 9/52 = 0.17. 要 is always acceptable.
- **check** (这是X吗): 29/122 = **0.24**, below a coin or "always 是" (about 0.5).
- **slot / where / what:** 0.05 / 0.04 / 0.09.
- **colour / count:** 0.60 / 0.37. These transfer only because the answer is a symbolic eye code.

**Likely cause (ASSUMED, from the key schema in `core_synth.py:135-178`).** A new filler X brings X-keys whose weights were trained on X's other frames (colour, where). These outvote the frame keys, and nothing represents "the frame regardless of the filler".

H ≈ 0.2 hides that the agent is below chance on binary held-out items. For an open world or an adult corpus, where almost every line is new, this is the key missing capability. develop's corpus test agrees:
- next-character accuracy 0.228 against 0.234 for an exact n-gram;
- the recurrent top state lost, at 0.199 (`develop/report.md` §4).

### 2.3 Life mode — the mode of the goal — is at chance on most scene tasks

Source: `final_noctx_s5`, eye layout `retino`. References are from `bench/report_life.md` (seed 0, model mode, so indicative only).

**Scene task accuracy:**

| task | ARCH_v2 | chance or trivial reference |
|---|---|---|
| colour | 0.20 | chance 0.17 |
| count | 0.39 | chance 0.33 |
| check | 0.51 | chance 0.50 |
| where | 0.20 | chance 0.33 (below chance) |
| taste | 0.49 | constant 1.00 |
| offer | 0.54 | constant 1.00 |
| slot | 0.62 | clearly above |
| what | 0.37 | clearly above |

- **Only slot and what are clearly above chance.**
- **Constant-answer comparison (PROVED arithmetic).** On the same 1,151 scene QA items, a constant answer per task type scores **0.40**; ARCH_v2 scores **0.43**.
- **Staged results do not survive life.** Staged colour 0.89 drops to 0.20 here, where colour is asked only 91 times in the whole life.

**Social and free replies:**
- Social replies score 0.32. Echo scores **1.00**: every social answer repeats the prompt.
- Free replies score 0.34; echo scores 0.53.

**Initiative vs the privileged random namer** (which is told the scene; ARCH_v2 has visual recognition VR):

| | ARCH_v2 | random namer |
|---|---|---|
| initiatives per 1,000 quiet ticks | 1.94 | 2.31 |
| appropriate share | 0.19 | 0.21 |
| appropriate share with partner present | 0.34 | ≈0.8 |
| reward | 365 | 454 |

**"Last tenth of life 0.47".** The rise is carried by social replies (0.05 → 0.62–0.80), which echo solves.

### 2.4 The retinotopic gain is mostly the world becoming disentangled

- **Different eye settings.** tune's best ran on the default `random` eye (`out/runs/base_s0.json` has no world key). ARCH_v2 runs on `retino`.
- **The agent reads the world's configuration.** It checks `world.p['eye_layout']` and hard-codes seg=240 and 3 slots (`agent_synth.py:54-56`).
- **Attention reads a label.** The attended slot is the block holding the 8 pointing units (`agent_synth.py:95-106`).

**ΔJ of +0.10, decomposed (PROVED from §8.1):**
- F +0.08, almost all from C3 +0.23 (colour and count) divided by 3;
- DR +0.04 after weighting, since the drift level is higher only because the level before is;
- RET −0.01 and H −0.01.

So the headline gain is the eye encoding plus removing CTX, a part this lineage had itself added. Real images, and even world2's 25 cells, do not supply such a frame; the world2 smoke run fell back to the flat eye.

### 2.5 Innate constants reverse-engineered from the bench

- **Reflex 7** (`init_min_tau=10`) is justified as "PROVED from world.py" because the world's `scene_lead` is 5.
- **The transition window** τ≤7 with `late_release=False` is fitted to the bench's waits.
- **The 3-tick syllable clock and silence segmentation** follow from the world's SOUND=3 and lines that contain no internal pause.
- **The dialogue boundary is a blank eye.** The world blanks the eye for 5 ticks between dialogues (`world.py:_run_dialogue`), and the parser resets on that (`agent_think.py:414-419`).

The designer defines instinct as broad connections over category features (IDEAS #17). These are script constants, and each one is a place where a new world breaks silently.

"0 mechanics violations" is partly by construction, because the world also yields:
- `World._wait` never starts the partner's line while the agent speaks;
- world2's caregivers wait up to 30 ticks out of politeness.

It is a valid regression check. It is not evidence of learned turn-taking.

### 2.6 The hippocampus is a short replay buffer; recall dilutes rather than consolidates

**Short horizon (PROVED).** Recall and replay need σ>0.05 (`rules.py:279`, `core.py:695`). σ falls by ×0.9995 per stored line, and 0.9995^5,990 = 0.05. So the "long-lasting" index covers only the last ~6,000 lines, which is a few days of an adult life.

**No consolidation:**
- Replay samples uniformly at η 0.2.
- Recall does not strengthen the trace (`rec_boost=0`), and plasticity never falls with use (`mature=0`); GEN keeps η 0.35.
- The drift brake is therefore dilution. It is not IDEAS #10's "recalled often → firm".
- Untutored Q1→Q2 episodes are replayed too.

The predicted continued drift is confirmed: 0.31 after 900 untutored dialogues (§2.1).

**Non-local operations.** Thought state is restored exactly after recall (`core_synth.py:464-470`), and content is copied through pointers. These are data-structure operations, which stretches principle 6.

### 2.7 The continuous thought is a register, and initiative ignores it

**Register (by ARCH_v2's own equivalence test).**
- λ 0.3 is below θ 0.5, so every new line pushes the old content below threshold. The test gives 592/592 identical records.
- The residue variant (λ 0.6) was rejected because no bench task needs more than one line.
- Principle 2 holds in letter only: functionally the thought is cleared and reloaded at each line end.

**Self-initiated speech never reads the thought.** `think_new` uses `prev` only for `kind=='reply'` (`core_think.py:122`, `core_synth.py:301-305`). A scene thought has P=[] whatever PFC-1 holds. Favourite thoughts never reach speech, and initiative content comes from the scene, not from thinking.

**Favourites metric (PROVED, `fav_base.py` on the life log).**
- The base rate used was 20 of 141 item types (0.14).
- The token share of the 20 most frequent heard characters is **0.63**.
- Those characters are 是这什么小再见吗我在好叫你子里要哪不放了: function characters, not topics.
- So the result is 0.81 vs 0.63, not 0.81 vs 0.14. The "foodie" analogy is unsupported.

### 2.8 The speech intention is a timer in disguise

**PROVED from `THINK_FINAL` and `agent_think.py:274-294`:**
- urgency u=0.15 per tick forces I≥1 by τ=7;
- `late_release=False` blocks later releases;
- thinking ends at τ=3;
- whether to speak is set by the body floor (0.3), and content by GEN.

**From `out/ablate_s0.log`:**
- latency is 6.0–7.4 ticks under every lesion;
- correct vs wrong: 6.01 vs 6.11 (B) and 6.10 vs 6.33 (C);
- the v1 bar timer gave 6 on 100% of replies.

V_go is a passenger (ARCH_v2's own lesion). Think's "honest timing" exists only in think's unbounded variant.

### 2.9 Statistics

- **Small probe samples.** Per-task probe n per seed is 15–78 (name 17, taste 21, one-shot 16); held-out n per task is 4–31. Life is one seed and one life per configuration.
- **"Scene questions in new situations".** Almost every question is a new exact (prompt, scene, pointing) combination, so this is not a generalisation test.
- **"Items heard once 0.36".** It counts (task, X) encounters, and frame-level answers such as taste → 好吃 count as hits. It is not one-shot learning.
- **J** has a maximum of 1.5, so it is not an accuracy.

### 2.10 Scale, open world, adult corpus

**MIX.**
- One unit is recruited per distinct tuple, never pruned (`prune_age=0`).
- At the cap, MIX silently stops recruiting (`rules.py:183-184`).
- The adult corpus has 99.7% unique pairs and about 100 keys per event, so distinct keys should grow roughly linearly with tokens, well beyond 2M (ASSUMED).
- The GPU budget "MIX ≤2M, 0.8 GB" therefore means either a hard stop on new combinations or growing interference from hash collisions in 2^19 slots.

**Vocabulary.** V is capped at 256 items, and in world2 the perceptual categories hit their cap of 256 within a day.

**world2 smoke run.**
- `world2_smoke.py:24,45` gives the agent `w.voice`.
- world2's README (line 20) allows only obs plus `warmup_windows()`.
- Its results are below echo: free replies 0.03 vs 0.13, social 0.00 vs 1.00.

## 3. Ranked changes

### 3.1 Self-monitoring with contingency-routed learning

**Mechanism:**
- Own utterances enter PFC-1 as efference-tagged assemblies (role keys: `role_units` exists, plus `ct_own` above θ).
- Own utterances are stored as episodes, so the NEXT links follow the real dialogue.
- Two readouts:
  - a partner model (what they say next) that always learns;
  - a policy (what I say) that learns from a heard line only if it is contingent on my turn, or from overheard exchanges between two other speakers.
- Contingency is a local, perceivable signal: the content overlap between the heard line and the efference copy of my reply.

**Why it is the bottleneck:**
- drift −0.26 after 300 untutored dialogues and −0.48 after 900;
- life facts 0;
- no coherence across turns;
- world2 and the adult corpus will be mostly untutored.

My run shows `ct_own` alone is not enough (−0.23 vs −0.28).

**Prediction.** On seeds 0–4: loss ≤ −0.10 after 300 untutored dialogues and ≤ −0.15 after 900, with F within −0.03.

**Cheap test:**
- `drift_own.py` with contingency gating, about 7 min;
- then `run_j.py ct_own=0.6 role_units=1` on seeds 0–1.

### 3.2 A learned item / meaning layer (IDEAS #3, #5, #14)

**Mechanism:**
- Each item gets a sparse code from local co-occurrence (k-WTA). develop showed 我→你他她.
- MIX and DG keys pair these codes, not raw IDs.
- Frame keys (the prompt minus its novel item) are added.

**Why it is the bottleneck.** A lookup table cannot face open data: held-out taste is 0/30 and held-out check 0.24.

**Prediction.** Held-out taste ≥0.7, check ≥0.55, offer ≥0.5, and slot/where >0.2, with F not lower.

**Test.** J runs on seeds 0–1, reporting held-out results per task.

### 3.3 Remove the world's gifts one at a time

**Mechanism:**
- **Hearing:** divisive gain control and a learned onset detector in place of the 3-tick clock. This is needed for world2 anyway (P4d).
- **Babbling:** the bench's `body.babble` with `keep='reward'` in place of `voice.decode`.
- **Vision:** a non-factorised eye (the `random` layout, or world2's fovea) with a slot frame the agent must learn.

**Why it is the bottleneck.** Without this ladder, nobody knows which results come from the architecture and which from the world.

**Prediction.** With noise 2, body babbling and the random eye:
- fewer than 67 programs after stage A;
- colour ≤0.3 without a learned frame.

If colour stays ≥0.8 with a learned frame, the retino result is vindicated.

**Test.** `run_j.py w_eye_layout=random w_noise=2` plus body babbling, one seed each.

### 3.4 Retrieval practice and stabilisation in place of next-line replay

**Mechanism:**
- Quiet-moment replay becomes a cloze: 我的小狗叫 → ?, training the completion gate on the hippocampus's own completion.
- Replay is prioritised by prediction error.
- A successful recall lowers the plasticity of the GEN rows it used (metaplasticity; IDEAS #10 "recall often → firm").

**Why it is the bottleneck:**
- knowledge told only as statements is unusable (facts 0/381);
- the drift brake rests on a buffer of about 6,000 lines.

**Prediction.** Life `fact_name` at 9+ encounters rises from 0/51 to ≥0.3; drift after 900 untutored dialogues is at most half of today's.

**Test.** One life run (about 9 min) plus `drift_own.py` at 900 dialogues.

### 3.5 Make speech come from the thought

**Mechanism:**
- Scene thoughts read PFC-1 (held or wandering content plus VR).
- The basal ganglia select the next thought operation (recall, compare, look at another slot, wait), rewarded by the confidence gain.
- Urgency is learned from how long partners wait, instead of being fixed so that release is forced by τ=7.

**Why it is the bottleneck.** IDEAS #6, #7 and #15 are outside the loop today.

**Prediction:**
- appropriate initiatives with the partner present rise from 0.34 to ≥0.6 (random namer ≈0.8);
- correct and wrong latencies separate by ≥2 ticks.

**Test.** One life run plus the latency split from the `ablate_synth.py` output.

### Precondition for all five

- **Same-condition trivial references.** Every headline number is reported next to echo, a task-type constant answer, the random namer and `life_refs.py`, all on the same seed and tutor. These take seconds.
- **Held-out results per task**, not only H.
- **Port gate.** Do not start the GPU port until ARCH_v2 beats the task-type constant on life scene questions and echo on social replies.

## 4. Fidelity to the designer's ideas

| Idea or principle | Status |
|---|---|
| #1 PFC as a recurrent next-input predictor (binding) | **Stretched.** There is no link from top state (t−1) to input (t); it is a conjunction table plus a one-line register. The recurrent variant lost in develop. |
| #2 holding by loops | **Letter only.** A stored array, no loop dynamics. |
| #3 words-and-meaning layer | **Missing.** |
| #4 local connectivity | Only EAR-0 and the DG fan-in. The learned local layers were rejected because the bench's ear is already perfect, which is a limit of the bench. |
| #5 random net plus Hebbian middle layer | **Not implemented.** This causes §2.2. |
| #6 speech caused by thinking | **Partly.** Replies come from a timer-like accumulator; initiatives come from the scene. |
| #7 favourite thoughts | Wrong base rate, and they never reach speech. Principle 7 says connect it or drop it. |
| #8 one unbroken life | Holds in life mode; life results are weak (§2.3). |
| #9 continuous thinking | **Letter only.** Equivalent to clear-and-load. |
| #10 no sleep, memories firmed by recall | No sleep: yes. "Firm" is missing: constant η, and replay dilutes. |
| #11 one-shot naming not urgent | Respected. |
| #12 bigger model | Untested. The dict-based MIX does not scale. |
| #13 world2 | Smoke run only, using the forbidden `w.voice`. |
| #14 fault-tolerant codes | **Unused after Z.** One wrong ID gives a different key. |
| #15 excitability as context length | **Untestable on this bench.** Every task is one line. |
| #16 3-D visual network on real images | Not yet. The current eye is factorised by the world. |
| #17 innate values; instinct as broad connections | Reward is a correctness oracle. The 23 "innate" items are mostly bench timing constants. |
| No puppeting; programs from babbling | Babbling reads the articulation table and the world's labels. |
| Principle 6 (local rules) | Mostly holds. Exceptions: pointer copy, perfect records, snapshot/restore, dict lookup. |
| Principle 7 (every learned part reaches behaviour) | V_go is a passenger. The T_W links and the hippocampus add about +0.01 pooled. All are kept. |
| Principle 8 (0-error mechanics) | Holds, partly by construction and because the world yields. |

## 5. Evidence index

**Code:**
- `synth/agent.py:209-257` (babbling)
- `synth/agent_synth.py:54-56, 95-106` (eye layout read from the world; pointing parse)
- `synth/core_synth.py:301-305, 316-325, 426-470` (thought, own speech, restore)
- `synth/core_think.py:120-153`
- `synth/rules.py:151-295`
- `synth/agent_think.py:274-294` (intention)
- `bench_copy/world.py`: `_fill`, `_exchange`, `_wait`, the Curriculum retino codes

**Results:**
- `synth/out/runs/final_noctx_s{0..4}.json` (my 5-seed per-task and held-out aggregation)
- `synth/out/life/final_noctx_s5.json`
- `synth/out/ablate_s0.log`
- `bench/report_life.md`
- `bench/world2/README.md`
- `bench_copy/README.md` (baselines)

**Mine:**
- `review2/skeptic/drift_own.py` and `drift_own.log` (SIMULATED)
- `review2/skeptic/fav_base.py` (PROVED from the life log)

## summary_zh

1. ARCH_v2 的数字是真实、可复现的，但它们测到的东西比文字说的少。功能上它是三样东西：
   - 按精确音节编号查组合的表（类似 n-gram）；
   - 一个完美的符号式回放库（海马体存的是 Python 列表）；
   - 再套上按测试台脚本时间常数调好的轮流说话机制。
2. 成绩主要来自世界送的三样东西：
   - 每次答完都给正确答案的复述老师；
   - 预先处理好的感官：恰好 3 拍一个音节；颜色、数量、形状、"指向"各占固定位置、每个位置编码相同的眼睛；离线读取世界发音表的咿呀学语；
   - 大量重复练习。
3. 最关键的结构问题：自己说的话不进入想法，也不进入海马体序列，所以"对方下一句"被当成"上一个问题的答案"来学。
   - 有复述老师时，这就是有标签的监督学习；
   - 没有老师时就漂移。我的实验：无人教 300 次对话后 C 从 0.79 降到 0.51，900 次后降到 0.31，仍在下降；只把自己的话放进想法（ct_own=0.6）只到 0.57，改善很小。
   - 一生模式中，听过 9 次以上的事实仍然 0/51 答对。
4. 泛化其实是查表：
   - 没教过的"X好不好吃"0/30，而"好吃"对所有食物都对；
   - 没教过的"这是X吗"只有 0.24，低于抛硬币。
5. 一生模式（最接近目标的模式）：
   - 颜色、数量、"这是X吗"、在哪里都在随机水平或更低；看图题总体 0.43，按题型说固定答案就有 0.40；
   - 社交回答 0.32，鹦鹉学舌是 1.00；
   - 主动说话的合适比例不如随机说物体名的参考（对方在场时 0.34 对约 0.8）。
6. 其他问题：
   - 视网膜式注意的提升主要来自世界换成了"已分解好的"眼睛编码；
   - "连续想法"在功能上等于旧的寄存器，而且主动说话根本不读想法；
   - "最爱话题 0.81 对 0.14"用错了基准，按出现次数算基准是 0.63，最常见的是虚词而不是话题；
   - 说话意图实际上是个计时器，延迟约 6 拍，对错几乎一样；
   - 海马体只覆盖最近约 6,000 句；
   - world2 冒烟测试用了 world2 规则禁止的发音表。
7. 建议按顺序做：
   1. 自我监控：自己的话进入想法和记忆，并按"对方是否在回应我"分配学习；
   2. 学到的词义层，取代按编号查组合；
   3. 逐个去掉世界送的东西：增益控制、身体咿呀学语、不分解的眼睛；
   4. 用"回忆练习"（填空式回放）加回忆后变稳固，代替"下一句"回放；
   5. 让说话真正来自想法。
   在一生看图题超过"按题型固定答案"、社交超过鹦鹉学舌之前，不要移植 GPU。
