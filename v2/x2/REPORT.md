(Saved by the coordinator from the agent's final hand-back.)

**In the repo:** this folder holds X2's own files plus its patched `rules.py` and `randlayer.py` (the stability rule). They ran on a flat copy of `v2/v22`, `v2/v21` and `v2/synth`; to rerun, copy those folders' files flat into one folder and these files over them. Results: `results/server/<job>__<run>.json` (the A800 jobs), `results/local/`, `results/life/`; job lists in `jobs/` (`fork_*` = the `x2fork` code copy used from F1b on). Pickles are not committed. Paths below of the form `.../arch/v2/...` are the coordinator's scratch folder.

# X2 final report: why ARCH_v2.2 collapses after about 3,000 untutored dialogues

Labels: **PROVED** = from code, counters or exact equality. **SIMULATED** = numpy runs; "server" means the A800, nS 1000, seeds 0–2, and "local" means nS 500, seed 0. **CITED** = from the designer's sources or earlier reports. **ASSUMED** = inference I did not test.

## 0. Bottom line

1. **The hippocampal horizon causes the collapse.** It ends the rehearsal of tutored episodes, and v2.2's slow readouts cannot hold the answers without that rehearsal.
   - **The arithmetic (PROVED).** A trace's σ decays by 0.9995 per stored line, and recall and replay need σ > 0.05. So a trace is usable for at most 5,989 lines. An untutored dialogue stores 2.0 heard lines, so every tutored trace is gone after about 2,990 dialogues. On every run, the tutored part of the replay pool reaches exactly 0 at the 3,000 mark.
   - **Causal tests, horizon shortened (SIMULATED).**
     - Removing all tutored traces at the start of untutored use: C accuracy 0.03 after 300 dialogues (0.00 / 0.07 / 0.01).
     - Halving the horizon (γ 0.999, local): 0.91 at 900, 0.84 at 1,200, then **0.22 at 1,500**.
   - **Causal tests, horizon lengthened (SIMULATED).** Doubling it (γ 0.99975 from the start of untutored use):
     - **0.91 at 3,000** (0.92 / 0.85 / 0.95), against 0.16 for the base;
     - then 0.46 at 4,500, and 0.00 at 6,000, 8,000 and 10,000 (seed 0).
     - So a longer horizon only delays the collapse.
   - **The two forces.** Untutored use pushes the agent toward partner lines from the first dialogue. Awake replay of tutored episodes, about half of which are question → correct answer, is the only counterweight.
     - With replay switched off during untutored use, C is 0.44 at 300 and 0.02 at 900 (3 seeds). Replay protects the agent; it is not the cause.
   - **One variable tracks C (SIMULATED).** Across 116 marks from 10 arms, C follows the untutored share of the replay pool (Pearson −0.95, Spearman −0.91).
     - Share below 0.05: C 0.93. Share 0.2–0.3: C 0.48. Share above 0.8: C 0.06.
     - Every replayable untutored cell is "partner question → partner's next question". These links survive only when the agent stayed silent; a decode of the local pool at 2,100 found 137 of 137 such cells (PROVED).
2. **Verdicts on the interim report's predictions.**

   | Prediction | Verdict |
   |---|---|
   | Replay off and no forced silence remove most of the collapse | **Refuted.** Replay off collapses about 3× sooner (0.02 at 900); no forced silence gives 0.21 vs 0.16 at 3,000 |
   | `ral_hip_own=1` reduces it | **Refuted.** 0.01 at 3,000, and the decline starts earlier |
   | `rm_prune=0` does little | **Confirmed.** 0.13 vs 0.16 |
   | The stability rule delays the collapse but does not stop it | **Refuted.** It brings the collapse forward |
   | Pruning clock | 71–85 learning events per untutored dialogue, so `rm_age` 20,000 ≈ 235–280 dialogues (PROVED). Pruning runs throughout the drift and does not coincide with the onset |

3. **Routes ruled out (SIMULATED unless marked).**
   - **Test-time hippocampal recall is not the route.** From mark 300 on, at least 99% of probe-time recalls come from untutored-era cells (PROVED, counters). Lesioning the NEXT copy route or all hippocampal recall at 3,000 never restores C (base 0.04 / 0.13 / 0.31 → 0.03 / 0.12 / 0.30 and 0.06 / 0.15 / 0.41). The damage is learned into the slow readouts.
   - **The reward-absence dip is not the route.** With `dip:0`, the fork is bit-identical to the in-job base drift on every counter at every mark, on all 3 seeds (PROVED). So dips never fire in this world. The same equality also proves the forks start from exactly the base trajectory.
4. **Self-rehearsal works in v2.2 but locks answers in.** The fix stores the agent's own replies as self cells and replays "question → my own answer". It uses two existing keys, `ral_hip_own=1` and `ral_replay_own=1`.
   - **During untutored use only:** **0.95 at 3,000** (0.95 / 0.89 / 0.99); seed 0 holds 0.93–0.95 out to **10,000**. Silence and unanswered lines drop to about 0.
   - **From birth, staged:** J **1.240 ± 0.022** vs 1.208 ± 0.016 for the base; 0.95 at 3,000.
   - **Correction fails after long self-rehearsal.** After 3,000 self-rehearsed dialogues, 20 recasts teach only 0, 1 and 2 of 4 changed answers (seeds 0, 1, 2). The trained agent before drift learns 3–4 of 4 in 2–12 rounds.
     - Confound: the comparison is "after 3,000 dialogues" vs "trained", with no 3,000-dialogue non-self-rehearsing control that is still healthy.
   - **One of two lives was destroyed.**
     - Seed 5 matched the reference: free replies 0.47 / 0.47, last tenth 0.60 / 0.65, corrections committed 0.87 / 0.81.
     - Seed 6 locked into its own babble from the first chunk: free replies 0.02, it spoke after 98% of corrections, so nearly every heard line was cancelled.
   - **I did not adopt it as is.** The obvious next test, a weaker self-replay weight (`ral_replay_own` < 1), was not run.
5. **Stability rule (designer #22, built on v2.2's error-correcting readout): harmful.**

   | Arm | J | Colour held-out | C at 2,100 (base 0.88) | C at 3,000 |
   |---|---|---|---|---|
   | β 0.03 | 1.100 ± 0.035 (−0.11) | 0.10–0.44 (base 0.78–0.90) | 0.23 | 0.00, mostly silence (0.54–0.91) |
   | β 0.3 | 0.981 ± 0.044 (−0.23) | — | 0.07 | 0.00 |

   - It makes the agent quieter from the start: online silence 0.15–0.23 vs 0.02–0.04. That sends more unanswered-question links into the pool.
   - Relearning a changed answer becomes stable about 3 rounds later (11.3 vs 8.2 rounds); it is not blocked.
6. **(T), wider and more persistent thought: harmful (3 seeds).**
   - No staged cost: J 1.193 vs 1.208.
   - The collapse comes earlier: 0.40 at 1,500 and 0.11 at 2,100. Silence and unanswered lines rise early, by the same route.
7. **hor2 life (seed 5, γ 0.99975 from birth):** free replies 0.40 vs 0.47, social 0.45 vs 0.67. It only delays the collapse, so it is not recommended.

## 1. What ran

**Server jobs, all in `gpu/queue_a800/`:**

| Job | Contents |
|---|---|
| `x2_base` | Base seeds 0–2 and (T) seed 0 (previous agent) |
| `x2_stab1` | β 0.03 on seeds 0–2, β 0.3 on seed 0 (previous agent) |
| `x2_F1b` | Seed 0: all forks plus amnesia and hor2. Seed 1: replay-off and no-silence |
| `x2_F2b` | Base relearning (fixed probe). Seed 1: hor2, amnesia, `rm_prune=0`, `ral_hip_own=1`. Seed 2: replay-off, no-silence, hor2, amnesia |
| `x2_F3` | Relearning for the stability arms; (T) seeds 1–2; hor2 to 6,000; hor2 life seed 5 |
| `x2_F4` | β 0.3 seeds 1–2; seed 2 `rm_prune=0` and `ral_hip_own=1`; self-replay seeds 0–2 |
| `x2_F5` | hor2 6,000 → 10,000; no-dip ×3; replay-off + no-dip ×3; self-replay life seed 5 |
| `x2_F6` | Self-replay 3,000 → 10,000; relearning after 3,000 self-rehearsed dialogues ×3; self-replay life seed 6 |
| `x2_F7` | Self-replay from birth, staged, ×3 |

- `x2_F1` crashed in 32 s (see §6).
- At most 2 of my jobs were queued or running at any time. I never used SSH. All 58 run records show 0 mechanics violations, and the lives report 0 overlap ticks.
- **The four required forks** (`rm_prune=0`; replay off during untutored use only; `ral_hip_own=1`; no forced silence) all ran on 3 seeds. They ran from the `x2base` trained pickles using the code that trained them (folder `x2fork`).
- **Optional runs.** Seed 0 showed a clear (harmful) effect for both β 0.3 and (T), so both were added on seeds 1–2. hor2 and self-replay stopped the collapse at 3,000, so each got a 10,000 run and lives.
- **Local runs (one process each):**
  - the as-is drift (collected);
  - a 2,100 → 3,000 fine-grained continuation;
  - replay off from 2,100;
  - half horizon;
  - self-replay.
- **Arms I added beyond the brief** (one line each):
  - amnesia: a direct causal test of the horizon;
  - hor2 and the half horizon: dose-response;
  - self-replay: the v3 rule "own speech enters the input";
  - no-dip: the v3 rule "good and bad not taken from the amount of response";
  - self-replay from birth: its staged cost.

## 2. The collapse, measured (SIMULATED)

**Base, server, mean over 3 seeds:**

| dialogues | 300 | 900 | 1,500 | 2,100 | 3,000 |
|---|---|---|---|---|---|
| C accuracy | 0.97 | 0.94 | 0.91 | 0.88 | 0.16 (0.04 / 0.13 / 0.31) |

- This matches the v2.2 report's 0.16 ± 0.11.
- The failure shape at 3,000 is partner lines 0.39–0.60 and silence 0.25–0.34.

**Local cliff, 2,100 → 3,000:**

| dialogues | 2,100 | 2,250 | 2,400 | 2,550 | 2,700 | 2,850 | 3,000 |
|---|---|---|---|---|---|---|---|
| C accuracy | 0.77 | 0.70 | 0.60 | 0.53 | 0.34 | 0.28 | 0.01 |
| tutored cells in the replay pool | 887 | 731 | 590 | 434 | 294 | 144 | 0 |

- The resumed run reproduced the original 3,000-mark result exactly (0.0126, same top wrong replies), so mark splits and pickle resumption are exact (PROVED).
- Replay switched off at 2,100 collapses faster than the base: 0.49 (2,400), 0.03 (2,700), 0.01 (3,000).

**The decoded replay pool (local; PROVED as counts, the text labels use the agent's own syllable categories):**
- At the end of training, 1,289 of 3,039 replayable cells are question → answer.
- At 2,100:
  - tutored-era cells: 449 of 887 are question → answer;
  - untutored-era cells: all 137 are prompt → next prompt, for example 早上好 → 葡萄是什么颜色.

## 3. All arms (C accuracy, server, mean with per-seed values at 3,000)

| arm | 300 | 900 | 1,500 | 2,100 | 3,000 |
|---|---|---|---|---|---|
| base | 0.97 | 0.94 | 0.91 | 0.88 | 0.16 (0.04 / 0.13 / 0.31) |
| `rm_prune=0` | 0.96 | 0.97 | 0.90 | 0.84 | 0.13 (0.05 / 0.23 / 0.11) |
| replay off (untutored only) | 0.44 | 0.02 | 0.00 | 0.00 | 0.00 |
| `ral_hip_own=1` | 0.94 | 0.94 | 0.89 | 0.77 | 0.01 |
| no forced silence | 0.96 | 0.95 | 0.90 | 0.89 | 0.21 (0.09 / 0.09 / 0.44) |
| amnesia | 0.03 | 0.01 | 0.00 | – | – |
| hor2 (γ 0.99975) | 0.99 | 0.95 | 0.95 | 0.93 | **0.91** (0.92 / 0.85 / 0.95) |
| self-replay (untutored only) | 0.95 | 0.95 | 0.96 | 0.94 | **0.95** (0.95 / 0.89 / 0.99) |
| self-replay from birth | 0.94 | 0.95 | 0.95 | 0.95 | **0.95** (0.96 / 0.92 / 0.99) |
| no-dip | identical to base | | | | |
| stability β 0.03 | 0.84 | 0.79 | 0.58 | 0.23 | 0.00 |
| stability β 0.3 | 0.75 | 0.65 | 0.49 | 0.07 | 0.00 |
| (T) | 0.90 | 0.82 | 0.40 | 0.11 | 0.00 |

**Relearning** (fixed probe; first round in which the frozen copy says the new answer; "never" counts as 21):

| state | mean first round | mean stable round |
|---|---|---|
| base | 7.7 | 8.2 |
| stability β 0.03 | 8.5 | 11.3 |
| self-replay from birth (before drift) | 7.2 | — |
| after 3,000 self-rehearsed dialogues | 18–21 | — |

## 4. Mechanism

- **The force toward partner lines.** In untutored use, a question the agent leaves unanswered keeps its NEXT link to the partner's next question. RAL also commits that next question as the "answer".
  - That link is the only kind of untutored episode replay can reach (PROVED, decoded pool).
  - I assume the RAL-commit route is the main online damage, based on commits rising from 0.23 to 0.52 per dialogue as silence rises when replay is off. I did not isolate it (ASSUMED).
- **The counterweight.** Replay of tutored episodes keeps the readouts right while those episodes are live.
- **The cliff.** The horizon removes the counterweight on a fixed count of about 2,990 dialogues. As the untutored share of the pool rises, partner lines and silence feed each other, because more silence means more question → question links.
- **Why (T) and the stability rule fail.** Anything that raises early silence collapses sooner, because it raises the inflow of bad links.
- **Why self-replay works.** Rehearsing the agent's own answers fills the pool with question → own answer. That keeps C up after the horizon, but it also rehearses its errors and its babble.

## 5. What this means for v3 (ARCHITECTURE.md §二)

1. **The horizon will exist in v3 by design.** §二.6 says the hippocampal links "长得快，也掉得快" (grow fast and fade fast) and handle short-term memory only. So v3 must not rely on the hippocampus to hold tutored answers.
2. **The cortex must hold them itself.**
   - §二.6 and DESIGNER_SPEC #23 (CITED): "长期记忆就是皮层自己慢慢学会的：靠反复回忆形成难改的连接" (long-term memory is the cortex slowly learning by itself, forming hard-to-change connections through repeated recall).
   - X2 shows v2.2's error-correcting cortex never reaches that state. After full training, removing the hippocampal traces gives 0.03 within 300 dialogues. §三 already lists the error-correction readout as wrong.
   - The tune report already saw this (CITED): "only awake recall moved it, and only when the store still holds old tutored episodes."
   - **A test v3 must pass:** train, apply amnesia, run 3,000 untutored dialogues.
3. **Under v3's single rule, something must keep re-activating the answers.**
   - §二.8 (CITED): one global decay, and "常用的连接靠反复加强保持住" (frequently used connections are kept by repeated strengthening). So an unused question → answer link fades in v3 too.
   - In v2.2 the only re-activation is hippocampal replay. The agent's own answers never teach: RAL cancels the heard line after an answer, and own replies are not stored.
   - In v3, own speech enters the input (§五, CITED), and temporal Hebbian learning (A then B strengthens A→B) turns every correct answer into rehearsal. This is what X2's self-replay emulates, and it stops the collapse to 10,000 (SIMULATED).
   - So the specific cause would not arise in v3 as long as the agent keeps answering (ASSUMED).
4. **The lock-in risk transfers.** In v3, "用得越多越难改" (the more used, the harder to change; §二.8) plus self-rehearsal is exactly the "if wrong things are also locked in, they can't be corrected" risk that IDEAS #22 warned about.
   - X2 measured it: corrections fail after 3,000 self-rehearsed dialogues, and one infant life locked into its own babble.
   - So the weight of the own-voice signal relative to heard input, and the "new learning first lands on fast connections" correction route (§二.8: 改正有两种机制), are the parameters to test first in v3.
   - §二.1 says own and other voices share content neurons and differ in speaker neurons. That gives a natural handle for weighting them differently (ASSUMED).
5. **The bad links themselves.** The link that hurts is the partner's next line after the agent's *silence*. RAL already cuts the link after an answer, so this refines §五's "自己回答之后对方说的下一句" (the partner's next line after my own answer).
   - In v3, temporal Hebbian learning would learn question → next question as a true regularity of an untutored world.
   - It becomes harmful only if speaker identity does not separate "the other will say X" from "I say X" (ASSUMED).
6. **A world that keeps teaching.**
   - §二.11 and designer message 97 (CITED): "没有这种世界让其不停学习，所以会没人教越来越坏可能也正常" (without a world that keeps teaching, getting worse with no teacher may be normal).
   - X2 agrees. The collapse needs at least one full horizon of world without question → answer examples. In tutored lives, v2.2 never collapses (tune report, CITED).
   - The designer also expected tuned growth and decay parameters to prevent it. In v2.2 none did: the stability rule made it worse, the longer horizon only delays it, and pruning and the brakes are irrelevant. The route that worked comes from the learning rule itself (own speech is learned), not from a new mechanism.
7. **Good and bad not taken from the amount of response (IDEAS #24.3).** The "no reaction" dip never fires in this world, and removing the reply brakes changes nothing. So the collapse does not come from response-amount signals.
8. **Time as order (IDEAS #24.1).** The horizon is counted in stored lines, i.e. events, not ticks. That is consistent with "time = order".
9. **#22 and #23 in v3.** X2's harm from the stability rule (built on NLMS) and from (T) (v2.2's thought implementation) does not test the designer's concave Hebbian rule or the designer's prefrontal dynamics. Do not carry those verdicts over.

## 6. Problems found and fixed (PROVED)

- **The relearning probe was invalid in `x2_base` and `x2_stab1`.**
  - It built its worlds with seeds seed+31 and seed+57. The world seed sets the voice and the eye codes, so the agent could not hear: "before" replies were noise like 'h什假' or silence, and every item showed as "never learned".
  - I fixed it to use the same seed and re-ran it from the pickles. The trained agents now give the old answers first (不客气, 晚安, 好吃, 要香蕉).
- **`x2_F1` crashed.**
  - The `x2base` pickles predate the `hip_dup` key in `core_synth.py`, so loading them under the current code raised AttributeError.
  - I resubmitted the forks as `x2_F1b` with the `x2_base` code copy (folder `x2fork`) plus the new `drift_x2.py`.
- **`drift_x2.py` saved clock0 as None in its drift pickles.** Resumed diagnostics would have used the wrong tutored/untutored boundary. I fixed it and added a `clock0=` override.
- **Additions to `drift_x2.py`:**
  - the `dset` keys `hip.<attr>` and `amnesia`;
  - `trim=1`, which drops the non-state logs before probes, to cap local memory.
- **Local memory went slightly over the cap twice.** Two local runs peaked at 602.8 MiB and 608.5 MiB, before I moved trimming in front of the probes and added a watchdog. Later local runs peaked at 520–532 MiB. The previous agent's as-is drift peaked at 659 MB.
- **Repo untouched.** The agent modified nothing in the repo and pushed nothing.

## 7. Files

Base folder: `.../arch/v2/`

- **Code:**
  - `.../arch/v2/x2/drift_x2.py` (fixed);
  - `.../arch/v2/x2fork/`: the code used for every new job, with job files in `.../arch/v2/x2fork/jobs/` (`F1.txt`, `F2b.txt`, `F3.txt`–`F7.txt`);
  - new tools: `.../arch/v2/x2/tab_x2.py` (summary tables), `.../arch/v2/x2/an_pool.py` (pool decoder), `.../arch/v2/x2/relearn_many.py`.
- **Server results:**
  - `.../arch/v2/gpu/queue_a800/x2_{base,stab1,F1b,F2b,F3,F4,F5,F6,F7}/result/out/runs/*.json`;
  - lives: `.../arch/v2/gpu/queue_a800/x2_F3/result/out/life/L_hor2_s5.json`, `.../arch/v2/gpu/queue_a800/x2_F5/result/out/life/L_selfrep_s5.json`, `.../arch/v2/gpu/queue_a800/x2_F6/result/out/life/L_selfrep_s6.json`.
- **Pickles:**
  - `.../arch/v2/gpu/queue_a800/x2_F4/result/ck/x2selfrep_s{0,1,2}_d3000.pkl.gz`;
  - `.../arch/v2/gpu/queue_a800/x2_F3/result/ck/x2hor2L_s0_d6000.pkl.gz`;
  - the base trained pickles copied to `.../arch/v2/x2/ck/srv/`.
- **Local results:** `.../arch/v2/x2/out/runs/loc_drift_{lasis,lfine,lnorec2100,lhor05,lselfrep}_s0.json`, with logs in `.../arch/v2/x2/loc/`.

## summary_zh
- **崩溃原因已确认：海马体的"记忆期限"。**
  - 每条痕迹每存一句衰减 0.9995，低于 0.05 就不能再回放。所以约 2,990 段无人教对话后，训练时的问答片段全部失效（数字推算，实测一致）。
  - 皮层（v2.2 的误差修正读出层）自己记不住答案，只靠回放训练时的问答来维持。
  - 无人教时，"没回答的问题 → 对方下一个问题"不断被学成答案。问答片段一失效，就崩溃。
  - 因果验证：
    - 开始时删掉训练片段：300 段内崩；
    - 期限减半：约 1,500 段崩；
    - 期限加倍：3,000 段时仍 0.91，但 6,000 段时崩（只是推迟）；
    - 关掉回放：900 段内崩（回放是保护，不是原因）。
  - 不是原因的：修剪、强制沉默、测试时的海马回忆、"没回应就扣分"。
- **设计者的"越常用越难改"：** 加在误差修正规则上有害。综合分降 0.11–0.23，崩得更早，因为模型更沉默。
- **(T) 更宽的想法：** 没有训练代价，但也崩得更早。
- **有效的办法：让模型自己的回答也被回忆（自我复习）。** 3 个种子到 3,000 段都在 0.95，到 10,000 段不崩；从出生开始用，综合分还略高。
  - 代价一：复习 3,000 段后几乎改不过来（20 次纠正只学会 0–2/4）。
  - 代价二：两次生活里一次陷进自己的咿呀声。
  - 所以暂不采用，需要先测更弱的自我复习权重。
- **对 v3 的意义：**
  - v3 的海马体本来就"掉得快"，所以长期记忆必须真正落在皮层。可以用"训练后删掉海马记忆，再无人教 3,000 段"来检验。
  - "自己说的话进入输入 + 赫布规则"本身就会产生自我复习，能防止这种崩溃。
  - 但必须保证外界纠正能赢过自我复习，否则会锁死错误。这正是设计者"两种改正机制"要解决的问题。
  - 设计者说"世界一直在教才正常"，X2 结果与此一致。
