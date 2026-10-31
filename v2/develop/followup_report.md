(Saved by the coordinator from the reviewer's final reply; the harness blocked the reviewer's own write.)

# Follow-up: separating "too small / untuned" from "structural" (ARCH_v2-develop)

All runs were local, strictly one process at a time.
- Setup for every variant: recast tutor, bench v1.2 (my copy), stage A = 300 dialogues + one babbling pass, then 1,000 dialogues per stage.
- After each stage, the frozen agent answered the B, C and D probe schedules: `World(stage, probe_frac=1, held_frac=0.3)`, 80 dialogues each. B and C were scored with learning off; D with episode storage on but readouts frozen.
- Driver: `develop/sizetune.py`; outputs in `develop/out/st_<variant>_s<seed>.log`.
- Each run took 5–6 minutes, much shorter than planned.
- All numbers are SIMULATED. Each probe set has ~150–200 replies, so differences under about ±0.06 are noise.

## 1. The table

Variant definitions:
- **Baseline** = the final configuration of my report.
- **(a) SIZE:** hippocampal cells 3k → 30k, DG 3k/60 → 6k/120, MIX cap 0.6M → 2M, CTX 800/24 → 4,000/120.
  - Hippocampus use after B / C / D: 4.0k / 7.7k / 10.9k cells, so it never filled.
  - Programs (≤ 320) and categories (≤ 256) were never at their caps.
- **(b) maturation:** GEN plasticity m = max(1/(1 + use/50), 0.05).
- **(c) interleaved curriculum:** stage strings 'B', 'BC', 'BCD'.

Columns: the frozen B / C / D probe scores after each stage. In the interleaved variants, stage 2 is 'BC' and stage 3 is 'BCD'. One-shot is the D task `oneshot` after the last stage. Drift is C probes before → after 300 untutored C dialogues with learning on.

| variant | seed | after stage 1 (B) | after stage 2: B / C | after stage 3: B / C / D | D: check / name | one-shot | drift |
|---|---|---|---|---|---|---|---|
| baseline (sequential) | 0 | 0.55 | 0.37 / 0.64 | 0.35 / 0.38 / 0.41 | 0.78 / 0.24 | 0.00 | 0.38 → 0.02 |
| baseline (sequential) | 1 | 0.57 | **0.18** / 0.56 | 0.50 / 0.54 / 0.55 | 0.74 / 0.47 | 0.00 | 0.54 → 0.01 |
| (a) SIZE | 0 | 0.49 | **0.24** / 0.58 | 0.30 / 0.37 / 0.44 | 0.82 / 0.29 | 0.00 | 0.37 → 0.01 |
| (b) maturation | 0 | 0.42 | 0.53 / 0.64 | 0.24 / 0.47 / 0.30 | 0.50 / 0.35 | 0.00 | 0.47 → 0.08 |
| **(c) interleaved** | 0 | 0.55 | **0.62** / 0.53 | **0.67 / 0.57** / 0.34 | 0.58 / 0.41 | 0.00 | 0.57 → 0.01 |
| **(c) interleaved** | 1 | 0.57 | **0.71** / 0.60 | **0.70 / 0.67** / 0.52 | 0.69 / 0.47 | 0.00 | 0.67 → 0.01 |
| (d) b + c | 0 | 0.42 | 0.58 / 0.57 | 0.55 / 0.59 / 0.17 | 0.24 / 0.35 | 0.00 | 0.59 → 0.08 |
| (e) a + b + c | 0 | 0.42 | 0.57 / 0.59 | 0.67 / 0.61 / 0.08 | 0.04 / 0.35 | 0.00 | 0.61 → 0.05 |

Further notes on the table:
- In the seed-1 baseline, B partly recovers during D. D's routines contain B/C filler exchanges, which act as partial interleaving.
- In (d) and (e), maturation freezes exactly the shared context units that stage D must re-use. "Check" then collapses (0.24, 0.04): the stability–plasticity trade-off.

## 2. Interference across stages: structural (the training regime), not size

**SIZE does not help.**
- B still falls after C: 0.49 → 0.24, against the baseline's 0.55 → 0.37.
- C is lost after D: 0.58 → 0.37.
- The hippocampus never filled and the MIX cap never bound, so capacity was not the limit.
- The forgetting is in the cortical readout's shared conjunction units (scene focus, recognised objects, line start, position). These are re-trained by the next stage's pointing tasks, for example 你要X吗 toward 要.

**Tuning (maturation) helps partly and costs plasticity.**
- C is kept better after D: 0.47 against 0.38.
- B even improves during C: 0.42 → 0.53.
- But B still collapses during D (0.24), initial learning is slower (0.42 vs 0.55), and combined with interleaving it blocks D learning.

**Interleaving removes it, in both seeds.**
- B keeps rising instead of falling: 0.55 → 0.62 → 0.67 (seed 0) and 0.57 → 0.71 → 0.70 (seed 1).
- C is retained: 0.57 / 0.67.
- The cost is less training on the newest stage: D 0.34 / 0.52 against 0.41 / 0.55.

**Verdict.** This is the textbook property of a distributed error-driven learner (catastrophic interference without interleaved experience; CITED: McCloskey & Cohen 1989; McClelland, McNaughton & O'Reilly 1995). It is not size. In the architecture, the interleaving must come from hippocampal replay of old episodes or from a curriculum that keeps old routines (as children's daily life does). My sleep does not yet supply it (§5). Interleaving is the fix available now.

## 3. One-shot naming (bench 0.00 vs symbolic 1.00): the cause is the BG completion gate, not size

The privileged audit (`diag_oneshot.py`, `out/diag_oneshot_base.txt`) checked each candidate cause in turn:

| candidate cause | finding |
|---|---|
| missing programs or categories for the novel characters | no: categories and programs exist for all 26 novel objects, via imitation from the babbling memory |
| recall selection | partly: about half the time the hippocampus recalls the "这是X" statement, and its completion holds exactly the new word (眼镜, 青蛙, 大象, 书包) |
| the gate | yes: the completion gate sits at 0.00–0.07, while the prompt-copy and recalled-next gates (0.3–0.8) push 这, giving "这是 / 这是什么" echoes |

**Intervention** (`diag_gate.py`). Forcing the completion gate open whenever a completion exists:
- one-shot 0.00 → **0.60** (n = 35);
- name 0.31 → 0.38;
- check 0.82 → 0.55, because the forced copy overrides yes/no.

So recall and programs are adequate, and the learned gate is the bottleneck. The remaining 40 % are recall-selection errors.

**Why the gate never learns.**
- The bench asks one-shot questions only as probes, by design, so their answer is never heard and the gate never gets a positive teaching event in the 这是什么 context.
- The gate's inputs are conditioned on the prompt's words, so the completion skill that the "name" task trains (我叫什么 → N) does not transfer.
- In my symbolic world, the one-shot question was answered by the tutor, which is why it reached 1.00 there.

**Size and tuning.** No variant changed it: 0.00 in all seven runs. Bigger stores only make the recall problem easier, and recall is not the limit.

**Mechanism needed.** A prompt-independent completion gate. It would be driven only by structural features: the recall's familiarity, whether the recalled episode's scene matches the current scene, and whether the cue is a question. Then the skill learned in one context ("answer with what the recalled episode adds to the cue") generalises to new questions.

## 4. Drift in untutored free use: structural; size and tuning only slow it

- Every variant still collapses: C 0.37–0.67 → 0.01–0.08. Maturation slows the loss slightly (→ 0.08).
- Lowering both learning rates during free use (`drift_eta.py`, base agent): ×1.0 → 0.02; ×0.3 → 0.03; ×0.1 → 0.16 after 300 dialogues. That is slower, not stopped.
- The cause is a missing role/contingency mechanism. The partner's next question is learned as the "answer" to the previous one.

**Why the own-reply fix hurt bench learning** (A 150 + B 700 probes).

| configuration | B overall | slot | what | where |
|---|---|---|---|---|
| rotation off | 0.34 | 0.21 | 0.36 | 0.44 |
| rotation on | 0.19 | 0.10 | 0.18 | 0.28 |

Both rows have a context reset at every dialogue boundary.
- **Excluded:** the cross-dialogue explanation (my after-recast utterance becoming the context of the next dialogue's first question), because the harm survives the reset.
- **Found, but partial:** a real asymmetry in attention teaching. After rotation, the previous line is often my own answer, an object name. The comparator locates that object, and the location then mis-teaches the next question's slot words. This can explain slot and where, but not "what", which uses the tutor's pointing.
- **Unresolved:** the rest of the harm lies in the readout's own learning. I could not isolate it in the time available.

## 5. Sleep: prioritised replay removes the harm; consolidation stays weak

Prioritised sleep (implemented in `core.py`, `sleep_mode='priority'`):
- every 150 exchanges, take 400 candidate traces;
- priority = the cortex's prediction error on the episode's answer × trace strength;
- the top 60 with error > 0.2 get 3 replay passes each;
- only traces the cortex now predicts (error < 0.2) are downscaled.

Symbolic M1: exact recall of a fact heard once, by delay in exchanges (`out/m1_sleep.txt`, `out/m1_nosleep.txt`):

| condition | 0 | 100 | 400 | 1,500 |
|---|---|---|---|---|
| no sleep | 0.92 | 0.65 | 0.17 | 0.13 |
| old sleep (sample by strength, downscale all) | 0.58 | 0.38 | 0.23 | 0.13 |
| **prioritised sleep** | 0.88 | 0.65 | 0.20 | **0.20** |

- The old sleep was harmful at short delays.
- Prioritised replay matches no-sleep at 0–100 and adds about 7 points at 1,500 exchanges.
- But retention beyond 400 exchanges stays near 0.2 in every condition, so consolidation into the cortex is weak. This is a mechanism-strength issue (more passes, replay of older traces), not a size issue: the 600-cell test index was the same in all conditions.

## 6. Conclusion: size/tuning versus mechanism

| failure | size? | tuning? | needs a mechanism? |
|---|---|---|---|
| interference across stages | no (SIZE alone: B 0.49 → 0.24 after C) | partly (maturation keeps C 0.47 vs 0.38, slows learning, blocks D when combined) | **yes**: interleaving or replay of old episodes (CLS). With an interleaved curriculum it is solved now: B 0.67 / 0.70 and C 0.57 / 0.67 at the end, seeds 0 / 1 |
| one-shot naming | no (0.00 in all 7 runs) | no | **yes**: a prompt-independent BG completion gate (forcing it open → 0.60) |
| drift in free use | no | only slows it (×0.1 rate: 0.38 → 0.16) | **yes**: a contingency/role third factor. Worked symbolically (0.92); the bench interaction is unresolved |
| sleep / consolidation | no | partly: prioritisation fixed the harm (0.58 → 0.88 at delay 0) and gave +7 points at 1,500 | **yes**: stronger replay of unconsolidated and old traces |
| stage-D gains | SIZE helps D a little (0.44 vs 0.41; check 0.82, name 0.29) | — | — |

**Answer to the designer.** Size is not the cause of any of the four failures. Tuning (maturation, learning rate) moves the numbers a little but does not remove any failure. Two failures need a regime change that already works: interleaving fixes interference; prioritisation fixes the sleep harm. Two need new mechanisms that are designed but not yet working on the bench: a generic completion gate for one-shot naming, and contingency/role gating for free-use drift.

## Files

All in `/tmp/claude-0/-root/e92d3d7d-5823-4f5c-a0ec-4fe302bff9af/scratchpad/arch/v2/develop/`.
- **Driver and run chains:** `sizetune.py`, `batch_st.sh`, `batch_post.sh`, `batch_post2.sh`, `batch_seed1.sh`.
- **Diagnostics:** `diag_oneshot.py`, `diag_gate.py`, `drift_eta.py`.
- **Code changes:**
  - `core.py`: the capacity knobs, prioritised sleep and the dialogue-reset option;
  - `rules.py`: index-based hippocampal keys; unit tests still 16/16.
- **Outputs** (`out/`):
  - variant runs: `st_*_s0.log`, `st_*_s1.log`;
  - one-shot diagnosis and gate intervention: `diag_oneshot_base.txt`, `diag_gate.txt`;
  - drift versus learning rate: `drift_eta.txt`;
  - sleep: `m1_sleep.txt`, `m1_nosleep.txt`;
  - rotation probes: `bench_rot_reset.log`, `bench_norot_reset.log`, `bench_q_*.log`.
- My two old server jobs time out on their own, as instructed.

## summary_zh

1. 体积不是任何一个失败的原因。海马体扩大 10 倍、组合层上限扩大 3 倍后，学 C 时照样忘掉 B（0.49→0.24）；漂移和一次性学名字也都不变。
2. "参数没调好"只起一部分作用。单元"成熟"能让 C 记得更牢（0.47 对 0.38），但学得变慢；和交错课程一起用时，反而让 D 学不会。降低学习率只能让漂移变慢（×0.1 时 0.38→0.16），不能消除。
3. 阶段间遗忘是结构问题：这种分布式学习器需要交错的经历。用交错课程（B、BC、BCD）就解决了：最后 B 0.67/0.70、C 0.57/0.67（两个种子），原来是 0.35/0.50 和 0.38/0.54。将来应由海马体回放旧情景来提供交错。
4. 一次性学名字为 0 的真正原因找到了。新字的类别和发音程序都在（26/26），海马体约一半时候也回忆起了"这是X"。卡在"补全门"从来不打开：测试台只把这种问题当探针，不给答案，门永远学不到。强制打开门后从 0 升到 0.60。需要一个不依赖具体问题词的通用补全门。
5. 无人教时的漂移是结构问题，需要"对方是否及时回应 + 轮次角色"这个第三因子机制。这个机制在符号世界有效（0.92），但在测试台上反而伤害学习，原因只找到一部分。
6. 睡眠：按预测误差优先回放、每条多回放几遍、不削弱未巩固的痕迹之后，不再有害（0.58→0.88），长时保持略升（0.13→0.20），但巩固仍然很弱。
