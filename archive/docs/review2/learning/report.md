(Saved by the coordinator from the reviewer's final reply; the harness blocked the reviewer's own write.)

# ARCH_v2 through a learning-theory lens (review round 2, reviewer "learning")

Labels: **PROVED** (argument with numbers), **SIMULATED** (my run: file and numbers), **CITED**, **ASSUMED**. One process at a time, each under 2 min and 100 MB. Paths are relative to the scratchpad.

My files are in `arch/v2/review2/learning/`:
- `gen_probe.py`, `gen_probe2.py`: symbolic probes, outputs in `gen_probe_out.jsonl` / `gen_probe2_out.jsonl`; `summ.py` summarises them. `gen_probe_out_v1.jsonl` holds the first, superseded runs (live measure only).
- `held_table.py`: reads the staged runs.
- `life_errors.py`, `life_firstask.py`: read ARCH_v2's life log.

The harness blocked writing `report.md`, so this text is the report.

## 0. Verdict

1. **Same objective, different learner (PROVED from code).**
   - What ARCH_v2 has from the language-model analogy:
     - the objective: predict every heard item, and speaking means saying the prediction;
     - the output half of the rule: R2 is exactly normalised LMS, the back-propagation step of one linear layer (`rules.py` L87–116).
   - What it lacks: everything else an LM learns.
     - No learned embedding: every context feature is an item identity, hashed into recruited pair units (`core_synth.py` L135–176).
     - No learned recurrent state: on the bench the continuous thought is the previous-line register. `ct.residue_items = 0` over 10,827 lines (`synth/out/runs/final_noctx_s0.json`).
     - Its own prediction never returns as input (`ct_own` = 0).
   - In LM terms it is three things: a sparse log-linear n-gram over hand-picked identity conjunctions, kNN retrieval (the hippocampus), and innate pointer ("copy") heads.
   - develop measured this on the corpus: 0.244 next-char accuracy vs 0.249 for an exact n-gram. The literal "fixed random recurrent top state" did worse: 0.178–0.199.
2. **It generalises better than its numbers say, and learning in use destroys it (SIMULATED).**
   - In a symbolic frame × filler probe, ARCH_v2's own core answers never-taught pairs perfectly when they are only tested: frozen held-out **1.00 on 3/3 seeds**.
   - If the same pairs are asked during life without feedback, as the bench and life do, held-out falls to **0.00 on 3/3 seeds**. Seen pairs stay at 0.92–1.00.
   - The cause: the agent learns the line it hears after its own answer as the answer. Two routes do this:
     - cortex: the line is learned in the context of the question;
     - hippocampus: the line is stored as the question's NEXT, then replayed and copied.
   - That line is the partner's next question. So the bench's H ≈ 0.2 and life's held-out ≈ 0.1 measure **drift on unrewarded items**, not the capacity to generalise. ARCH_v2's open problem 5 is open problem 1.
3. **A perceivable LLM-style turn rule fixes most of it (SIMULATED).**
   - The rule: never learn across your own turn; answers are learned from other speakers' answers.
   - With a tutor that gives 30 % model replies, held-out goes 0.00 → **0.72 frozen / 0.80 live** (3 seeds). Seen accuracy stays at 1.00.
4. **Similarity-based generalisation is still missing (SIMULATED, small n).**
   - Test: an answer that depends on a new word's class (foods → 好吃, toys → 不能吃, class learnable only from other frames).
   - ARCH_v2 says 好吃 to everything: 0.50.
   - Adding a learned or even an oracle class code to its pair schema gives 0.25–0.75 (2 seeds). It also copies role-confused transitions ("你玩猫吗").
5. **The real teacher is the recast protocol.** world2 answers a correct reply with praise (`bench/world2/care.py` L475–479). ARCH_v2 will learn that praise as the answer (PROVED, §2.4).

## 1. ARCH_v2 vs the real brain (learning lens)

**Differences that matter for the goal**

**a. Complementary learning systems are inverted.**
- Brain: a fast, sparse hippocampus and a slow, overlapping cortex. Interleaved replay lets the cortex extract structure (CITED McClelland, McNaughton & O'Reilly 1995; Kumaran, Hassabis & McClelland 2016).
- ARCH_v2: both stores are fast and sparse. NLMS moves the active context's drive by η·error on every event, whatever its history (PROVED, η 0.35):
  - one contradiction drops a learned answer from 0.9 to 0.58;
  - a second drops it to 0.38, below the competitor's 0.58.
- So cortical knowledge is a moving average over about 1/η ≈ 3 events per context.
- Long-term knowledge lives in the hippocampal replay window: σ > 0.05 for ≈ 5,990 stored lines (ln 0.05 / ln 0.9995, PROVED).
- **Why it matters.** The roles are swapped. Replay can only rehearse; it has no slow, shared substrate to distil into. tune found recall halves C held-out (0.19 → 0.09).

**b. No learned representation between hearing and meaning.**
- Brain: graded, learned similarity on the ventral route.
- ARCH_v2: Z is one category per syllable, and everything after it keys on identity. This is round-1 consensus item A.4, still open and not on ARCH_v2's own list.
- **Why it matters.** Nothing is "alike in meaning" (class probe at chance). A split category is a new word with zero transfer: in the world2 smoke run, 3 of 4 agents hit the 256-category cap within a day.

**c. Nothing learns what to hold.**
- Brain: the PFC learns what to hold, and the BG learns when to update (CITED O'Reilly & Frank 2006).
- ARCH_v2: the thought dynamics are innate.
- **Why it matters.** "Excitability = context length" cannot pay off without learned relevance. The residue variant cost −0.05 (§7 #9).

**d. No inner speech.**
- Brain: the efference copy of speech reaches auditory and language cortex, so people think in words (CITED Alderson-Day & Fernyhough 2015).
- ARCH_v2: own speech never enters the thought, and thinking writes nothing.
- **Why it matters.** There is no multi-step thought. Open problem 2 ("thinking does not improve content") is structural.

**e. No synaptic consolidation.**
- Brain: consolidated synapses stabilise (CITED Fusi, Drew & Abbott 2005; Benna & Fusi 2016).
- ARCH_v2: no consolidation variable; maturation was tried and removed.
- **Why it matters.** Idea #10's "recall it often and it sticks" has nothing that sticks.

**f. No multi-step credit.**
- Brain: eligibility traces plus TD dopamine (CITED Schultz, Dayan & Montague 1997; Yagishita et al. 2014).
- ARCH_v2: a bandit critic V(first-step context); the actor's eligibility is one utterance (`core.py` L648–661).
- **Why it matters.** Multi-step acts in world2 cannot be credited.

**g. One learning stream with no roles.**
- A child knows that "what is said to me" is not "what I should say".
- ARCH_v2 learns a single target: the next other-speaker line, skipping its own.
- **Why it matters.** Role confusion, drift and the held-out collapse (§2.1).

**Differences that do not matter now.** Binary units and the sign-split Dale's law; event time instead of continuous dynamics; hashed conjunction units (the identity keys are the problem, not the hash); the innate retina, body, beat and reflexes (legitimate instinct); the BG accumulator; no sleep (awake replay can do sleep's job, because target and schedule matter, not the brain state).

## 2. Problems of the whole architecture

### 2.1 The held-out collapse is learned, not structural (SIMULATED, `gen_probe.py`)

**Setup.**
- Core: ARCH_v2's `SynthCore` with the final core keys (ct, ρ 2, CTX off, awake recall 8 / 0.2 / uniform, 10k hippocampus, floor 0.3). No audio, body or scene.
- Frames:
  - copy frames: 你要X吗 → 要X, 你喜欢X吗 → 喜欢X;
  - constant-answer frames, filler first: X好不好吃 → 好吃, X在哪里 → 不知道.
- 12 fillers; each is held out of exactly one frame and taught in the other three.
- 900 dialogues of 1–3 exchanges. Recast tutor: reward 1 if acceptable, then the correct reply. 15 % seen probes; awake recall after every exchange.
- At the end, a frozen copy answers every held-out pair 3 times.

| condition (seeds 0/1/2) | seen (frozen) | **held-out (frozen)** | held-out (live, last third) | share of failures that are partner lines |
|---|---|---|---|---|
| held-out pairs never occur in the life | 1.00 | **1.00 / 1.00 / 1.00** | — | — |
| held-out pairs asked during life (always probes, as bench/life) | 0.97 | **0.00 / 0.00 / 0.00** | 0.01 | 0.69 |
| + own reply enters the thought (`ct_own` 1) | 0.97 | 0.00 ×3 | 0.01 | 0.72 |
| + turn-role tag on every key (seed 0) | 1.00 | 0.00 | 0.04 | 0.61 |
| ORACLE: the line after an unanswered reply is learned after the own reply (cortex only) | 1.00 | 0.00 ×3 | 0.01 | 0.72 |
| ORACLE: only its hippocampal NEXT link is cut | 1.00 | 0.17 / 0.08 / 0.00 | 0.06 | 0.72 |
| ORACLE: both | 1.00 | **0.83 / 0.58 / 1.00** | **0.87** | 0.20 |
| tutor with 30 % model replies, ARCH_v2 as is | 0.97 | 0.00 ×3 | 0.00 | 0.78 |
| same tutor + **Xcut** (perceivable: after any own utterance the own reply enters the thought and the next line is not linked as NEXT) | **1.00** | **0.42 / 0.75 / 1.00** | **0.80** | 0.22 |

`gate_noid` (BG gates read context only) changed nothing: live held-out stayed at 0.007.

**Reading.**
1. Generalisation does exist. The frame-only pair units, e.g. (P=要, N=0) and (P=好, L=BOS), plus the copy gates carry a frame to a filler never heard in it.
2. Each unanswered probe teaches the following line as the answer, by two routes, and both must be closed.
   - The hippocampal route is the stronger: the probe is stored with the next prompt as its NEXT (`core_synth.py` L443–448).
   - Awake recall then replays that pair as if heard (`core.py` L663–682).
   - At the next ask, recall finds the exact episode (FAM 3), and COPY_next says the stored line. Example: 狗在哪里 → "你喜欢鸟吗".
3. Xcut costs nothing on seen items (live seen 1.00 vs 0.94–0.98 for base), because model replies teach answers and drift is gone. The cost: recasts no longer teach the prompt's answer.

**The bench has every ingredient.**
- Training worlds keep the default `held_frac` 0.15 with learning on (`run_j.py` L119; `world.py` L62), and held-out pairs are always probes.
- The frozen probes (`run_j.py` L64) then run on agents that have already stored those episodes.

**Life shows the signature** (`life_firstask.py`):

| | held-out accuracy (2nd half) | n |
|---|---|---|
| 0 earlier asks | 0.33 | 3 |
| 1 earlier ask | 0.33 | 6 |
| 2–3 earlier asks | 0.28 | 18 |
| 4–7 earlier asks | 0.11 | 47 |
| ≥ 8 earlier asks | 0.07 | 91 |

Seen items in new situations, which get recasts, rise with asks: 0.03 → 0.53.

**Consequence for the designer's 2026-10-31 addendum to idea #5.** The addendum says "no unit stands for 'this frame, whatever the filler'". That is not accurate for ARCH_v2: such units exist and work. A random-at-birth MIX is good for fixed size, the GPU and fault tolerance (#14). But it will not raise bench held-out while the learning target is unchanged.

**Prediction.** A random MIX under today's protocol shows the same live collapse.

### 2.2 The central claim, component by component (PROVED from code)

| LM component | what it buys | ARCH_v2 |
|---|---|---|
| learned embedding | similar tokens → similar codes | absent (one-hot identities) |
| learned recurrence (BPTT) | what to keep | absent (innate thought; equals the register on the bench) |
| learned attention | which past item matters | absent; every held item enters pairs equally |
| output layer | state → next token | **R2 = NLMS, exact** |
| autoregressive feedback | multi-step generation and thought | absent |
| induction/copy heads | binding new tokens in context | present, innate (OneShotChain + gates) |
| retrieval (kNN-LM) | rare facts | present (hippocampus) |

- **"Pull one hair and the whole body moves" is the mechanism of generalisation, not a defect.** Interference and transfer are the same overlap (CITED McClelland et al. 1995; French 1999). ARCH_v2 learns in real time without interference because every identity pair gets its own unit. That is why it cannot transfer by similarity.
- **"Bigger" is a no-op for the readout at bench scale.** MixDict creates a unit for every new key until its cap: 48,632 of 600,000 were used. A 4×/16× MIX in P6 changes nothing here; only data and shared representations can.

### 2.3 Numbers that do not mean what they claim

**H ≈ 0.2 hides ≈ 0 for every task whose answer is a word.** Seeds 0–4, `held_table.py`:

| task | held-out | n |
|---|---|---|
| what | 0.08 | 89 |
| slot | 0.04 | 82 |
| where | 0.04 | 78 |
| offer | 0.16 | 52 |
| taste | 0.00 | 30 |
| check | 0.23 | 122 |
| colour | 0.59 | 45 |
| count | 0.36 | 57 |

- **check 0.23 is below** the 0.5 a constant 是 or 不是 would score.
- **Colour and count survive because the world hands over invariance.** Their answer bits are shared by all objects, and the world gives the identical 240-unit code in every slot (world2 does the same in its 25 cells). Real images (#16) will not give position-identical codes. The +0.08 J of retinotopy is mostly that gift.
- **J weights H at 0.25, so selection could not see generalisation.**
  - brainlike: offer held-out 0.78, what 0.33.
  - develop: offer 0.44, check 0.48.
  - ARCH_v2: offer 0.16, what 0.08, check 0.23.
- **Life is mostly memorisation** (`life_errors.py`, 2nd half):

  | situation | accuracy | n |
  |---|---|---|
  | repeated situations | 0.93 | 82 |
  | new situations (seen items) | 0.41 | 654 |
  | held-out pairs | 0.12 | 165 |

  Held-out errors: partner lines 39 %, echo 13 %, silence 29 %. "Accuracy keeps rising" is real only for items that get recasts.
- **"Favourite topics 0.82" describes the thought, not behaviour.** Behaviour stays within noise, so the co-holding links T_W are a learned passenger by principle 7.
- **"Continuous thought, PROVED equivalent" means principle 2 is met only in form.** No measured behaviour depends on continuity.

### 2.4 Hidden teacher and shortcuts

- **The recast protocol is the teacher.** Own speech is kept out of the thought (`ct_own` 0, `core_synth.py` L316–323), so the first other-speaker line after a reply lands in the question's context. The bench guarantees that line is an acceptable reply (README, Tutor modes).
- **world2 breaks this (PROVED).** After a correct reply, world2 says 真棒 / 对 / 很好 (reward 1). By R2 each success teaches the praise as the answer and cuts the correct answer's drive by ≈ 0.35 × 0.9 ≈ 0.3.
  - **Prediction:** a success → praise → error → recast → success cycle, with praise words given as answers. 对 is also an acceptable reply to check, which will inflate check.
- **The hippocampal NEXT link means "what came next" with no role**, so it carries the same flaw (§2.1).
- **Perception is perfect symbols.** Z purity is 1.00 only because the bench's syllables are fixed, noise-free templates. Every downstream key is an identity, so idea #14's fault tolerance is used only in the ear and the DG.
- **Babbling reads `world.voice`** (known). The world2 smoke Shim passes `w2.voice` too.
- **Awake recall saves and restores the whole thought** (`core_synth.py` L464–470). This is a clear-and-reload around every quiet moment (≈ 0.76 per heard line), which goes against the spirit of principle 2.

### 2.5 What will not survive the open world, many bodies or an adult corpus

- **The drift brake is a rehearsal race (PROVED arithmetic, ASSUMED rates).**
  - Per quiet moment each eligible episode is replayed with probability ≈ 8/6,000.
  - In the 300-dialogue drift test, a C item is contradicted about as often as it is rehearsed, and at a higher rate (η 0.35 vs 0.2). The fixed point is p* ≈ 0.2r / (0.2r + 0.35c) ≈ 0.36.
  - Once an untutored stretch is longer than the ~6k-line window, only drifted episodes remain and replay accelerates the drift.
- **Adult corpus.**
  - With 4,096 items there are 16.8M (P, L) keys per pair type, and 99.7 % of corpus pairs are unique, so most units see one event.
  - Hashing into 2¹⁹ slots gives random sharing, not similarity.
- **Multi-step requests in world2** (fetch / give within 400 ticks) need TD credit, which ARCH_v2 does not have.
  - **Prediction:** requests stay at the silent/echo floor.
- **Many agents overhearing each other learn each other's wrong answers** (port item P4e). This is recursive training on generated data, which degrades (CITED Shumailov et al. 2024).
- **Data scale.** Even 10–100M-word BabyLM learners stay far below LLMs (CITED Warstadt et al. 2023). A 10⁶-tick life holds ≈ 10⁴–10⁵ syllables. Expect child-like competence, carried by inductive biases.

## 3. What should change next (ranked)

**1. Turn-aware learning targets (who answers whom).**
- **Mechanism.**
  - The default is the efference rule tested as Xcut: after the agent's own utterance, the own reply enters the thought as a line (old content ×0.3), and the next heard line is not linked as the prompt's hippocampal NEXT.
  - Answers are then learned from other speakers' answers: model replies, overheard dialogues, world2's p_model 0.3.
  - A contingency gate lets a heard line marked as a reaction to the own reply (praise or reward; content overlap with the efference copy, as real recasts reformulate the child's words, CITED Chouinard & Clark 2003) also teach the prompt's answer.
- **Why it is the bottleneck.** It explains the held-out collapse, role confusion, free-use drift and world2's praise problem with a single cause.
- **Prediction.** With the bench tutor 'model' (or alternating model/recast dialogues):
  - H ≥ 0.45 and drift loss ≤ 0.05, with F within −0.05;
  - in world2, praise words appear in < 1 % of answers instead of > 5 % after two days.
  - It is falsified if H stays < 0.3.
- **Cheap test.** `run_j.py` with an Xcut agent key and tutor 'model' (≈ 17 min per seed). The symbolic version is already done: 0.00 → 0.72.

**2. Measure generalisation honestly.**
- **Mechanism.**
  - Probe held-out pairs only with frozen copies: set `held_frac` 0 in the training worlds, `run_j.py` L119.
  - Give H the same weight as F.
  - Report per-task held-out and a role-confusion rate (partner lines and echoes).
  - Keep `gen_probe*.py` as regression tests.
- **Why it is the bottleneck.** Today the bench cannot see any change in generalisation; it sees drift.
- **Prediction.** ARCH_v2 unchanged, re-scored: H rises from 0.18–0.22 to ≥ 0.35, with what/offer/taste ≥ 0.3. If it does not, the scene/VR route has its own failure.
- **Cheap test.** One `run_j` seed (≈ 17 min).

**3. A learned meaning layer, after change 1.**
- **Mechanism.**
  - Items get sparse codes learned online from what precedes and follows them: random indexing / successor-representation Hebbian (CITED Kanerva et al. 2000; Stachenfeld et al. 2017; Levy & Goldberg 2014).
  - These codes become the input of the conjunction layer, replacing identities. This is the addendum's random-at-birth MIX, reading learned codes.
- **Why it is the bottleneck.** Without it, nothing generalises by meaning (class probe 0.50). Bolting class bits onto the identity schema is not enough: 0.25–0.75 over 2 seeds, and without the turn fix it generalised role confusion too.
- **Prediction.** After change 1, class-dependent held-out ≥ 0.8.
- **Cheap test.** `gen_probe2.py` (≈ 2 min per run).

**4. A slow, consolidating cortical path (CLS done right).**
- **Mechanism.**
  - A second readout over the meaning codes, with η ≈ 0.02, trained mostly by awake replay.
  - Per-synapse consolidation that grows with each successful recall (cascade synapses), so idea #10's firming has a variable.
- **Why it is the bottleneck.** Today's cortex forgets in about 3 contradictions, and the 6k-line buffer is the only long-term store.
- **Prediction.**
  - ARCH_v2 now: C after 3,000 untutored dialogues ≤ 0.1, and an item untouched for 10k lines is retained ≤ 0.3.
  - With the slow path: ≥ 0.5 for both.
- **Cheap test.** `run_j` with `drift=3000`.

**5. Close the autoregressive loop, with TD credit.**
- **Mechanism.**
  - Inner speech: with the mouth gated off, predicted items enter the thought tagged as imagined; they never become learning targets.
  - The BG chooses thought operations, reinforced by a TD critic with eligibility traces over states.
- **Why it is the bottleneck.** Thinking cannot change content, and multi-step acts cannot be credited.
- **Prediction.** 刚才S是什么 and two-hop memory facts > 0.3; world2 requests above the floor.
- **Cheap test.** A symbolic two-hop probe; world2 requests with 4 agents.

## 4. Are the designer's ideas implemented faithfully?

| idea / principle | verdict | evidence |
|---|---|---|
| #1 PFC = an LM's RNN | **stretched** | objective and last-layer rule yes; recurrence, learned state and prediction→input no |
| #2 a loop holds the thought | faithful | §7.3 |
| #3 a words-and-meaning layer | **under-used** | Z is one-hot |
| #4 local connectivity | ear only | content path is all-to-all identity hashing |
| #5 + addendum | **diagnosis partly wrong** | frame-only units exist and generalise (1.00); the loss comes from the learning target |
| #6 speech caused by thinking | faithful in timing | content = next heard line → role confusion |
| #7 favourite thoughts | faithful | behavioural passenger |
| #8 one unbroken life | faithful | — |
| #9 continuous thinking | **in form only** | register-equivalent; recall saves/restores the thought |
| #10 firming by recall | **stretched** | rehearsal of ≈ 6k lines; nothing firms |
| #11 one-shot naming not urgent | respected | — |
| #12 bigger is better | **no-op** for MIX as built | 48,632 of 600,000 units used |
| #13 world2 | port feasible | praise and missing TD break learning |
| #14 fault tolerance | **under-used** | identities after Z |
| #15 excitability = context length | **not realisable yet** | needs learned relevance |
| #16 3-D vision | untested | the slot invariance is a world gift |
| #17 innate values | **under-used** | actor lesion leaves J unchanged; no value pathway |
| §9 principles | 1, 3, 4, 5, 6, 8, 9 met | 2 is stretched (save/restore); 7 is stretched (T_W and the continuous thought have no behavioural effect) |

## summary_zh

1. ARCH_v2 实现了"像语言模型那样预测下一个输入、说话就是说出预测"这个目标，读出层的局部误差规则也正确（等于单层反传）。但语言模型能泛化靠的三样东西它都没有：学出来的词向量、学出来的循环状态、"预测再当输入"。实际上它是"按词身份组合的查表式 n-gram + 海马体检索 + 先天复制通道"。
2. 新发现：它其实会泛化。在符号测试里，从没教过的"句型 × 词"组合只测不学时，3 个种子全对（1.00）。但像测试台和一生模式那样在生活中被问、又没人反馈时，就降到 0.00。原因是它把"我回答之后听到的下一句"（往往是对方的下一个问题）当成答案来学：皮层学一次，海马体存成"下一句"、回放、再复制出来。所以测试台的"没见过组合约 0.2"测到的是无人教时的漂移，不是泛化能力。
3. 一条像语言模型"角色标签"的规则，"不跨过自己的发言去学答案，答案只从别人的回答里学"，不用任何先知信息，就把它恢复到 0.72–0.80，已学内容不受损。
4. world2 里答对后照料者说"真棒/对/很好"，ARCH_v2 会把这些当成答案学进去（已证明）。
5. 建议按顺序做：①按轮次区分学习目标；②只用冻结副本测泛化，并提高它在 J 里的权重；③在①之后加学出来的"意义层"（按上下文学的词编码）；④加慢速、会巩固的皮层通路；⑤让预测回流成"内心语言"，并加时间差分奖励学习。
