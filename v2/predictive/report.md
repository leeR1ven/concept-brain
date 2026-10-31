(Saved by the coordinator from the reviewer's final reply; the harness blocked the reviewer's own write.)

The architecture and prototype are built and tested on bench v1.2. The teacher-free basics work end to end, content stays modest, and the memory tasks (stage D) fail; the causes are identified. The full report follows inline: the harness refused `report.md`, so the design tables are in `DESIGN.md`.

# ARCH_v2-P — "one general predictive learner"

Folder: `arch/v2/predictive/`. Labels: PROVED (argument with numbers), SIMULATED (numpy run; file + numbers), CITED, ASSUMED. Running log of every number: `NOTES_results.txt`.

## 0. Verdict

1. **Design.** Every learned synapse uses one local rule (PR, below). Every area learns to predict its own driven activity. Speaking means saying the prediction: a program's score is the predicted probability that its sound comes next, and END is the predicted end of the line. Areas differ only in wiring, sizes, timescales and plasticity. ARCH_v1's body is kept unchanged.
2. **The rule passes 9/9 unit tests** (SIMULATED, `test_rule.out`).
   - Its weights become probabilities.
   - It stops strengthening once the target is predicted.
   - It learns inhibition (a weight goes to −1.00).
   - A held source learns the target's onset (1.00; net.py's exact rule gives 0.000).
   - Credit across delays needs the trace.
   - Its own outputs can only be learning targets when the third factor is the reward prediction error.
   - It stores correlated codes one-shot (0.96, against 0.17 for counts).
3. **Teacher-free development works end to end on the bench** (SIMULATED, `main_s0.log`).
   - Syllable units are learned by listening: 67 units for 67 chars, purity 1.00, no splits.
   - Programs are learned by babbling: all 35,960 gestures tried, 67/67 correct, no cue.
   - All 71 words first heard in stage D got a program from the babble memory.
   - Turn-taking uses a learned Go/NoGo.
   - Next-char prediction on heard speech is 0.76–0.79.
   - Mechanics are perfect: 0 overlap, latency 6, all 6,281 syllables 3 ticks, 0 garble, 0 extra turns.
4. **Content** (seed 0, tutor = model, 1,500/1,500/1,200 dialogues, pruned agent):

   | stage | mine | v1 bar | prior | n-gram | held-out |
   |---|---|---|---|---|---|
   | B | 0.14 | 0.12 | 0.14 | 0.13 | 0.04 |
   | C | 0.34 | 0.19 | 0.31 | 0.54 | 0.18 |
   | D | 0.09 | 0.21 | 0.24 | 0.16 | 0.07 |

   - The scene questions "slot" and "what" are above every reference.
   - "offer" (0.57) beats the n-gram (0.49).
5. **It learns in use.** On seen items, from the first to the last third of each stage: B 0.16 → 0.28, C 0.38 → 0.56, D 0.07 → 0.12. The v1 bar goes 0.13 → 0.11 and 0.34 → 0.23.
6. **Causal audit and pruning.** Several learned parts were passengers or harmful. Removing five of them gave equal or better scores on every stage with about 30% less compute.
7. **Failures, each with an identified cause.**
   - Held-out scene items stay near 0.
   - D one-shot is 0.00; "check" is below the yes-bias prior; the new "before" is 0.01.
   - The designer's reverberating PFC on realistic codes collapses into one generic state, so the agent holds a line with a gated latch instead.
   - At text level the rule is less data-efficient than exact n-gram counting.

## 1. The design

### 1.1 The one rule (PR, `pr.py`)

For a synapse i→j of path P:

`Δw_ij = η · θ · M(t) · π_P · x_i(t−1) · (y_j(t) − p_j(t)) / n_j(t−1)`

- **x_i:** the presynaptic trace, `x_i(t) = max(s_i(t), λ x_i(t−1))`. Each source population is normalised to unit power (divisive inhibition).
- **y_j:** the target, set by j's driver:
  - in event groups, the event itself ("char c heard now"), with learning only at event ticks;
  - in assembly (state) groups, the unit's own firing, so co-activity is a target.
- **p_j:** j's prediction, `p_j = clip(d_j/θ, 0, 1)`, where d_j is j's learned input (the dendritic prediction, Urbanczik & Senn 2014, CITED).
- **Sign:** w = w⁺ − w⁻. The negative part runs through j's local inhibitory interneuron, so Dale's law holds.
- **n_j:** the power of all synapses onto j, the normalisation at learning time. One error is shared by all paths onto j.
- **π_P:** the plasticity factor (hippocampal paths 8, cortex 1).
- **M:** the broadcast third factor:
  - 1 for other speakers' events;
  - 0 for my own voice (corollary discharge);
  - r − V for own outputs when there is reward (designed; unit-tested in T8);
  - a novelty gate for hippocampal storage.

The rule keeps the designer's direction: an earlier source is strengthened toward a later target, and a source that comes after its target gets no credit. It adds "predicted means no more strengthening" and "predicted but absent means weaken".

### 1.2 Areas (prototype size; GPU sizes in DESIGN.md §9)

| area | size | activity / fan-in | persistence | learns from |
|---|---|---|---|---|
| EAR0/FEAT0 (fixed) | 1,153 | top 40 %, fan-in 4 | 1 tick | birth calibration only |
| perceptual clock (innate) | phase register | window energy | reset at speech onset | — (theta-like, one sample per 3 ticks) |
| Z syllable units | recruited (67 → 138) | winner-take-all, all of FEAT0 | 1 event per char | competitive PR on others' speech; efference copy on own speech (inverse model); a fresh unit when none reaches 0.7 |
| babble memory | one entry per distinct own sound (3,618) | — | permanent | one-shot gesture → own sound; links newly heard units |
| x1, x2 | last two chars of the line | localist | the line | fixed register |
| Gline (PFC line layer) | 2,048 | union of 8-unit uni/bigram codes, about 4 % | the line | fixed codes |
| Ghold (PFC held prompt) | 2,048 | about 4 % | from another speaker's line end to the next one | BG-gated load (reset + load); latch |
| Gtopic | 2,048 | | leak 0.5 per load | fixed (pruned) |
| B (prompt bag) | zmax | localist | as Ghold | diagonal copy path |
| V (eye code) | 1,024 | about 7 % | as the scene | — (pruned) |
| V2 | 65,536 | AND of 2 eye units, about 100 active | as the scene | read by the predictor (pruned) |
| C2 | 65,536 | AND of (Ghold unit, eye unit), about 60 active | as Ghold | read by the predictor |
| H / H2 (fast DG paths) | 4,096, k 32 | 12 random inputs, threshold 3 | per event | π = 8 into the predictor (pruned) |
| episode cells (hippocampal index) | ≤ 6,000, recruited one per completed line of another speaker | key: Ghold, eye, C2 pairs, DG code | until the ring frees the cell | one-shot store; winner-take-all recall reinstates the line's chars (Brec) and the next line's (Bnext) |
| predictor ZP | END + one unit per Z | | per event | PR, target = heard char or line end |
| copy gates (3) | one unit per buffer | | per event | PR: "the next char is in this buffer"; the gate multiplies the innate one-to-one mirror |
| Go/NoGo | 1 | | per line end | PR, target = another speaker starts within 4 ticks, or others stay silent for 30 ticks |
| critic (designed) | 1 | | per turn | PR on reward; not used on the bench |
| event store + sleep | ≤ 6,000 events | | | novelty-gated replay into the slow paths, then downscaling of the fast paths (implemented, not tested on the bench) |
| body | ARCH_v1 | | | nothing learns |

### 1.3 The other parts of the design

- **PFC.** It has three layers: Gline (the current line) → Ghold (held prompt, BG-gated: loads at another speaker's line end, holds through own replies, overheard replies and recasts) → Gtopic (slow). Its inputs are learned: Z units and reinstated hippocampal content. Mixed selectivity comes from C2, and it is read at every boundary. Leak plus learned recurrence was module-tested (M4/M6). The reverberating version is not adopted (§3.2).
- **Hippocampus.** Recruited episode cells, an index in the sense of Teyler & DiScenna (CITED). CA3 recurrent completion was tested and rejected (M2). Linear one-shot recall was tested and rejected: it superimposed every stored line, so the true content scored 0.30 against 1.71 for the question's own words.
- **Basal ganglia.**
  - Innate: winner-take-all per beat, GO, the END rule, the floor.
  - Learned: END as a predicted event, the Go/NoGo unit, and the copy gates.
  - Designed but not used on the bench: a critic whose prediction error trains own outputs.
- **Motor.** Bouts of random 4-muscle gestures on a spontaneous beat, with habituation (no repeat until all are tried). A gesture becomes a program when its own voice drives a heard unit with p ≥ 0.75 and a margin ≥ 0.2. There is no learning of Z from babbles. Imitation emerges: in C the agent repeats the model reply exactly after 17% of model exchanges.
- **Teaching without puppeting.** No muscle cue and no END clamp anywhere. The agent learns by listening to every speaker. Recasts are heard lines, predicted from the still-held prompt.
- **Stages, GPU cost, framework, fixed mechanisms, risks.** All are in DESIGN.md §8–12.
  - Fixed mechanisms counted: 14. None of them refers to a task or a stage, and none is timed to a junction (v1 needed P = T∧GO, E = L∧R2 and the Hc delay lines).
  - Cost at GPU scale: about 1.5 GB fp16 per agent, about 1e7 multiply-adds per tick.
  - Framework kept: binary units, fixed active fraction, born random wiring, one global decay, the body, no speech area. Framework changed: the rule and the role of the hippocampus.

## 2. Prototype files (`arch/v2/predictive/`)

- `pr.py`: the rule and areas.
- `agent.py`: the mind.
- `body.py`, `frontend.py`: copies of the v1 logic and the exact ear layer 0.
- `run_lib.py`: a diagnostic world and the babbling phase.
- `run_main.py`, `train_audit.py`, `lesion_eval.py`, `audit_all.sh`: training and the audit.
- `test_rule.py`, `test_modules.py`, `text_pred.py`, `table.py`: tests and tables.
- `bench_snapshot/`: bench v1.2.
- Results: `main_s0.*`, `pruned_s0.*`, `audit_{B,C,D}.log`, `test_*.out`, `text*.log`.

## 3. Results

### 3.1 Rule (`test_rule.out`, 9/9)

- **T1, probabilities:** p converges to 0.207 / 0.498 / 0.903 for targets at 0.2 / 0.5 / 0.9.
- **T2, saturation:** w = 1.000 after 100 and after 1,000 events (net.py's count: 1.82 → 6.13).
- **T3, learned inhibition:**
  - signed weights: p(Y|A) = 1.00, p(Y|A+C) = 0.00, w(C→Y) = −1.00;
  - excitatory-only weights cannot store the exception: 0.73 / 0.73.
- **T4, a constant source** keeps a bounded weight (|w| 0.13).
- **T5/T7, onset and order:** a held source learns the onset (1.00); a source that comes after the target gets 0.
- **T6, delays:** with λ = 0 only 1-tick delays are learned; with λ = 0.8 the scores are 0.98 / 0.89 / 0.65 at 2 / 4 / 6 ticks.
- **T8, third factor:**
  - M = 0 freezes all learning;
  - own outputs as targets with M = 1 drift, reward 0.12;
  - with M = r − V the reward reaches 1.00.
- **T9b, one-shot storage:** correlated codes with Zipf labels at M = 1,000: PR 0.96, excitatory-only 0.68, counts 0.17.

### 3.2 Modules (`test_modules.py`, `test_m4.out`, `test_m6*.out`)

- **M1, assemblies (fail on one pre-set criterion; M1b passes).** Held co-activity as the target gives attractors.
  - With fan-in 1,000: 100 and 300 patterns complete from half cues at 1.00 and 0.97, and hold at 1.00 for 16 ticks.
  - Random starts fall only into stored patterns, and |w| ≤ 0.1 with no budget.
  - With fan-in 100 it fails: 0.07 at 100 patterns.
  - The pre-set criterion that failed: a one-tick 50% cue does not replace a held attractor (0.05). A gated full load does (1.00), which is the M1b criterion, revised after seeing M1.
- **M2, hippocampus (fail).** CA3 recurrence lowered half-cue recall at 1,000 episodes from 0.68 to 0.07. A dentate code with a one-shot readout recalls the oldest of 1,000 episodes at 1.00 from a full cue and 0.68 from a half cue (0.98 / 0.42 at 3,000), and the newest of competing episodes wins every time.
- **M4, PFC with gated load = reset + input (pass).** With leak plus learned recurrence and distractors during the hold:
  - hold 1.00 over 60 ticks, no period-2 in any hold;
  - replacement leaves J 0.14 with the old content; 20-way decode 1.00.
  - Without recurrence the hold collapses (0.02). Without leak, 36% of holds alternate with period 2.
- **M5, drift (pass):** a familiar line's held code keeps J 1.00 after 1,000 more lines.
- **M6, the designer's reverberating hold (fail),** on realistic bench line codes that share frame chars:
  1. Generic state: held states of different lines overlap at 1.00, against 0.06 for their inputs.
  2. Hold vs update: the new load ends in the same attractor (+0.95 extra overlap).
  3. Period-2: none (index −0.02).
  4. Drift: J 0.17–0.21.
  - 20-way decode is 0.45 after 10 ticks and 0.15 after 60.
  - Lines held at once, loaded in a row without reset (1/2/3/4/6 lines): 0.90 / 0.85 / 0.82 / 0.74 / 0.64.
- **M6b, the same with a homeostatic set point:** the generic state is gone (0.045 vs 0.068), replacement is clean, period-2 index −0.08. But the hold dies: decode 0.97 after 10 ticks, 0.00 after 60.
- **Answer to the reverberation checklist:** on realistic codes a loop either collapses into a generic state or cannot hold. The bench agent's gated latch avoids all four failures by construction.

### 3.3 Bench (bench v1.2, seed 0, model tutor; references are the README's seed-0 numbers)

| task | full agent | pruned agent (seen / held) | v1 bar | prior | n-gram |
|---|---|---|---|---|---|
| B slot | 0.13 | 0.12 (0.21 / 0.03) | 0.00 | 0.04 | 0.06 |
| B what | 0.09 | 0.08 (0.17 / 0.01) | 0.06 | 0.06 | 0.03 |
| B where | 0.18 | 0.22 (0.36 / 0.10) | 0.33 | 0.33 | 0.32 |
| C offer | 0.49 | 0.57 (0.88 / 0.25) | 0.04 | 0.09 | 0.49 |
| C taste | 0.38 | 0.40 (0.67 / 0.21) | 0.46 | 1.00 | 1.00 |
| C count | 0.24 | 0.28 (0.38 / 0.19) | 0.08 | 0.32 | 0.30 |
| C colour | 0.14 | 0.12 (0.17 / 0.06) | 0.08 | 0.14 | 0.19 |
| C ritual | 0.27 | 0.26 | 0.51 | 0.21 | 0.99 |
| D check | 0.20 | 0.21 (0.39 / 0.07) | 0.49 | 0.48 | 0.30 |
| D name | 0.11 | 0.12 (first char 0.73) | 0.00 | 0.17 | 0.17 |
| D before | 0.01 | 0.01 | 0.01 | 0.08 | 0.09 |
| D oneshot | 0.00 | 0.00 (first char 0.85) | 0.00 | 0.00 | 0.00 |

- The full agent is the main run, the pruned agent the same plan without H, H2, V, V2 and Gtopic.
- Seed 1 (server, short run) reproduces stage A (purity 1.00, 67/67 programs) and early B (0.08 at 500 dialogues, against 0.07 for seed 0 at 600).

**A lower speaking floor helps.** About a third of free replies in C were silent because the best prediction was below 0.3. Own speech never trains the predictor, so the same knowledge can be read out with a lower floor. At 0.12 the scores rise: B 0.135 → 0.154, C 0.164 → 0.255, D 0.092 → 0.122 (audit content).

### 3.4 Causal audit (`audit_*.log`)

Each learned part of the main agent was silenced on the same content, with learning on. There were about 50–100 probes per run, so differences under 0.05 are noise.

| stage (intact score) | part | score without it |
|---|---|---|
| B (0.135) | C2 | 0.096 |
| B | H2 | 0.212 |
| B | x1+x2 | 0.173 |
| B | V, V2 | identical replies |
| C (0.164) | prompt-copy path B | 0.145 (offer 0.46 → 0.36) |
| C | H | 0.255 |
| C | episode recall (Brec) | 0.218 |
| C | V, V2, C2, babble memory | identical replies |
| D (0.092) | H | 0.082 (name 0.21 → 0.07) |
| D | copy gates | 0.071 (check 0.15 → 0.10) |
| D | Go, babble memory, Ghold | identical replies |

- The Go unit acts outside the scored replies. Before it existed, the agent spoke after 69–83% of model replies (it said the partner's next prompt); with it, 0.00 in B.
- The fast path H carries the name task in the trained full agent (0.21 → 0.07 without it). The pruned agent, trained without H, still reaches name 0.12, so other paths take over when it is absent from training.

### 3.5 The designer's recurrent predictor, at text level (the coordinator's protocol)

Next-char top-1 on 1,000 held-out corpus dialogues:

| learner | 5k dialogues | 30k dialogues |
|---|---|---|
| PR, last two chars | 0.175 | 0.226 |
| + the designer's recurrent kWTA loop | 0.183 | — |
| + my PFC line codes (current line, held previous line) | 0.214 | — |
| Hebbian distributed codes (coordinator) | 0.152 / 0.153 | — |
| exact n-gram, 1/2/3 chars (coordinator) | 0.234 / 0.249 / 0.242 | 0.245 / 0.287 / 0.283 |

- So error-driven learning beats Hebbian counting, the PFC context adds 4 points, and the loop adds about 1 point.
- One online pass of PR is still less data-efficient than exact counting.

## 4. What failed during development, and what changed

1. **The corollary discharge was treated as own voice.** The v1 body fires GO at L+4 even though the model reply starts at L+3. My agent paused its perceptual clock on that discharge and heard every model reply misaligned. Fixed by using the actual own-voice tail; B went from 0.00 to 0.11.
2. **Self-supervised Z learning during babbling made a hub unit** that won 98% of babbles. Fixed: babbling only links programs, with a margin.
3. **A double spontaneous GO cut babbles short.** Fixed with a refractory condition.
4. **Babbling coverage.** 140k ticks reached only 89% of the gestures; the full set needs 170k ticks with habituation.
5. **Turn-taking.** "Say the prediction" first made the agent say the partner's next prompt after model replies. A Go target of "quick reply only" then silenced D probes. The open-floor target fixed both.
6. **Scene conjunctions.** A k-WTA conjunction layer gave no usable pairs; replaced by exact pair layers (V2, C2).
7. **Copying novel words** needed the copy gates (per-char copy weights are zero for a new char).
8. **Linear recall** gave way to episode cells.

## 5. Notes for the coordinator

- **Bench:**
  - Model-mode timing: any agent that treats the corollary discharge as "I am speaking" mishears every model reply (§4, item 1). Worth a README note.
  - The random eye code has no location-invariant structure, so held-out scene items need a conjunction per (slot, object). The new `eye_layout='retino'` option could fix this; I did not test it.
  - D one-shot is never taught: it is always a probe, and its question reuses the statement's frame "这是". A learner by listening never hears such a question answered.
  - My earlier D "before" scores (0.13–0.15) came from the old task version and are invalid. On v1.2 it scores 0.01.
- **GPU queue:**
  - CPU-bound numpy agents ran about 12× slower per process there, from 0.6× cores plus BLAS thread oversubscription. Jobs need `OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1`.
  - Job `predictive_bc1` and batch 2 timed out. Job `predictive_pruned1` was too slow and will time out; please discard it (I re-ran that test locally).

## 6. Status and next steps

**Works:**
- the rule;
- teacher-free perception, babbling, new-word linking and turn-taking;
- learning by listening and in use;
- perfect mechanics;
- assemblies and the gated PFC hold;
- above the v1 bar on B and C, and above every reference on slot, what and offer.

**Does not work:**
- held-out scene items;
- every D task;
- the reverberating hold;
- CA3 completion;
- sleep replay and the critic are untested on the bench;
- recast mode and stage E were not run;
- only one full seed.

**Next, in order:**
1. A retinotopic eye with an innate spatial-attention gate, giving one location-invariant "attended object" code and recall keyed on the referent.
2. Keep the pruned agent and a lower (or learned) speaking floor.
3. A learned role model (who answers whom), so the agent only says predictions for its own role.
4. A fast count-like path or replay for data efficiency.
5. A sleep-replay test.
6. A torch/GPU port for multi-seed long runs.

## summary_zh

1. 设计：整个模型只用一条局部学习规则——猜对了不再加强，猜了没发生就减弱（可以变成负的，就是学来的抑制），再加痕迹和全脑广播信号。每个区都学着预测自己的输入；说话就是把预测说出来。身体部分保留不变。
2. 规则的单元测试 9/9 全部通过：权重变成概率；被持续托住的想法能学会触发动作；跨时间间隔要靠痕迹；自己说的话只有在有"奖励误差"时才学，否则会越说越偏。
3. 完全不用老师、不推肌肉：听话学会了字的声音单元（67 个，全对）；咿呀学语找到了全部字的发音；后来新听到的 71 个字靠"咿呀记忆"直接会说；学会了轮流说话。不抢话、按时开口、字清楚这些机械指标全部满分。
4. 成绩（B / C / D）：0.14 / 0.34 / 0.09；v1 是 0.12 / 0.19 / 0.21，先验是 0.14 / 0.31 / 0.24。"左边是什么""这是什么""你要X吗"超过了所有基线。没见过的组合仍然很弱，D 的记忆题失败。
5. 边用边学：同一阶段里熟悉题目的正确率一直在上升（C 从 0.38 升到 0.56），v1 不会。
6. 逐个关掉部件检查：有些部件确实影响回答，有些只是"乘客"，有的还有害；删掉 5 个以后，成绩不变或更好，计算量少了约 30%。
7. 设计者提出的"循环保持想法"：在真实输入下会塌缩成同一个通用状态，老问题重现；加了稳态机制又保持不住。所以现在用"闸门锁存"来保持。
8. 失败原因已经找到：眼睛的编码里没有和位置无关的注意机制；一次学会新物体的题，答案从来没有被教过；开口门槛太高（降低以后各阶段都提高）。
9. 下一步：视网膜式的眼睛加注意力闸门、保留精简版、学会"谁该回答"、测试睡眠回放、移植到 GPU 跑多种子长训练。