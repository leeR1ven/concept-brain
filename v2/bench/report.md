(Saved by the coordinator from the reviewer's final reply; the harness blocked the reviewer's own write.)

The bench is finished and in use: READY says v1.2, and `selftest.py` passes 29/29 (68 s), including a tick-by-tick match with ARCH_v1's own toy. Writing `report.md` was blocked by the harness, so the full report follows inline; README.md (usage, the API as implemented, every [clar], reference numbers) is the persistent spec.

Two things need your action:
1. **Old `before` task in the designers' copies.** I redesigned D's `before` task at 02:56 because it could be passed without any scene memory (RefV1 scored 0.38 by repeating the answer it had just heard). The develop copy (02:20), the predictive copy (02:49) and brainlike's `benchcopy_v02` still have the old version, so their D `before` scores are not comparable. Brainlike's current copy has the fix. They should recopy `world.py`.
2. **The V100 runner gives jobs one CPU.** `nproc` inside a job printed 1, so the advertised 48 cores aren't available to jobs. My 12-worker jobs ran about 6× slower per process than this box and hit their time limits. I ran the bar locally instead, one process at a time.

The bar's core result: the ARCH_v1-equivalent agent has perfect mechanics, but its replies are at or below text-only baselines. Scene and memory questions are at chance or 0, and with no teacher it never speaks.

---

# ARCH_v2 shared bench: report (bench agent)

## 0. Bottom line

1. **Everything in API.md is implemented.**
   - Physics: random and lawful, `noise=σ`, the `other2` voice.
   - Senses: ear, eye, reward, cue.
   - `frontend.py`, and `body.py` with both cue-based and babble-based program creation.
   - `World` with stages A–E, tutor modes model / recast / force plus probes, and the turn protocol.
   - The log, `evaluate()`, `baselines()`, the v1 reference agent with its scores, and `selftest.py`.
   - Releases: v0.1 (B, C, model and force modes) came first, then v0.2 (A, D, E, recast, `body.py`, `frontend.py`, `ref_v1.py`), v0.3 (lawful-physics checks, selftest), v1.x (the bar, curriculum helper, two options a designer asked for).
2. **The validity checks pass** (SIMULATED):
   - Echo and exact lookup score 0.00 on held-out items and scene questions in B, C, D and E: 0.00±0.00 across 3 seeds, in both force and model mode.
   - The oracle scores 1.00 on every task, through the real mouth, decoder and timing.
   - A fixed seed reproduces the stream and log exactly, and the content schedule does not depend on the agent. The fingerprint is identical under numpy 1.26 (local) and 2.1.2 (V100).
3. **Mechanics agree with ARCH_v1's toy, in the strongest form.** Three runs give tick-identical mouth output:
   - (a) `archv1.py`'s own Talker;
   - (b) the bench world driving archv1's Brain;
   - (c) the bench world driving `RefV1`, built from `body.py` + T + P→Y + Hc→Y/END + the echo slot.

   The match holds over 8,097 ticks (mem config, seed 0: 108 taught pairs + 40 free replies, 2,313 mouth ticks) and 8,107 ticks (base config, seed 1). The bench's mechanics metrics equal run.py's audit exactly: 0 overlap, latency 6 in 40/40, 771 syllables all 3 ticks, 0 garble, 0 restarts, stops floor 35 / END 5, guided 108/108. On the corpus the bar also reproduces ARCH_v1's published content numbers: 20-way top-1 0.115 vs 0.121, AUC 0.572 vs 0.573, copy 0.44 vs 0.46.
4. **Speed:** the world costs 17–19 µs per tick, so the trivial agent runs 2,000 dialogues in 3.5 s (target under 180 s). `RefV1` costs 0.8–0.9 ms per tick: 2,000 dialogues take 2.5–4 min; stage E costs 2.2 ms per tick.
5. **The bar** (v1's adopted MEM default, teacher-forced, 3 seeds × 2,000 dialogues per stage):
   - Acceptable-reply rate is B 0.12±0.01, C 0.19±0.01, D 0.19±0.02.
   - That is at or below the text-only baselines: prior 0.14 / 0.30 / 0.22, n-gram 0.14 / 0.52 / 0.15.
   - Scene questions sit at chance and the memory tasks are about 0.
   - Mechanics are perfect in all 9 runs.
   - In model or recast mode v1 never speaks: 0 of 1,075 and 0 of 3,984 free replies.

## 1. What was built (`arch/v2/bench/`, numpy only, about 4,100 lines)

| file | contents |
|---|---|
| `physics.py` | `Voice`: `world/voice.py` copied (verbatim copy in `voice_orig.py`), built over the 3,970 sorted corpus chars, so every sound and muscle set is the real model's; vectorised, cached mouth rule; `other2`; lawful physics; `mouth_vocab`. `Stream`: 3 sources mixed by max, ARCH_v1's re-articulation onset, `noise=σ`, read-only window. Corpus path from `BENCH_CORPUS`, `/root/work/corpus` or `~/v2lab/data`. |
| `world.py` | `World`, `Curriculum` (held-out split, eye codes), `CorpusSets`, stages A–E, tutor modes, `Track` (mouth tracker), `run`, `run_script`, `run_curriculum`, `warmup_windows`. |
| `metrics.py` | `evaluate`, `headline`. |
| `baselines.py` | echo, last char, prior, exact lookup, char n-gram, oracle, silent, plus diagnostic `lastreply`; `baselines()` (cached), `run_agent`. |
| `frontend.py` | `Ear0` (ear layer 0, bit-identical, with reserve), `feat0`, `feat0_v1`, `WorkingMemoryT` (v1's T), `v1_tie`. |
| `body.py` | ARCH_v1's body as a module with a mind interface. |
| `ref_v1.py` | The v1-equivalent agent ("the bar"). |
| `v1_crosscheck.py`, `selftest.py`, `physics_check.py` | Checks. |
| `bar.py`, `bar_local*.py`, `summarize_bar.py`, `update_readme_bar.py` | Bar runners and tables. |
| `results_all/` (132 JSON), `checks/` | Raw results and check outputs (selftest, cross-checks, physics checks, bar tables). |
| `archv1_copy.py`, `voice_v1copy.py`, `echo_slot_orig.py`, `echo_slot_copy.py` | Copies; only import paths changed, and in the last one a single speed edit verified exact against the pristine copy. |

## 2. The API as implemented (README.md is the reference; deviations marked [clar] there)

- **Constructor:** `World(stage, seed=0, physics='random'|'lawful', noise=0.0, tutor='model'|'recast'|'force', vocab=None, prev=None, **params)`.
  - Stage strings may combine letters, e.g. `'BCD'`.
  - `prev=` continues a curriculum: clock, exposure history, novel objects already named, corpus position, fresh content stream. `run_curriculum(agent, [(stage, n), …])` wraps this.
  - Runs: `run()` (repeatable on one world), `run_script(agent, items)`.
  - Main params: `probe_frac` 0.15, `held_frac` 0.15, `held_ratio` 0.3, `gap_answer` 45, `after_stop_test` 12, `pause_after` 5, `reply_cap` 60, `gap_model` 3, `model_wait` 12, `eye_layout`, `mouth_vocab`.
- **obs and agent:** `obs = {'ear': (8,6) int read-only, 'eye': (1024,) uint8 read-only, 'reward': float, 'cue': (32,) or None}`. The agent returns `step(obs) -> (32,)` muscles and may set `agent.tick_event` to report its stop reason.
- **Vocabulary:** the Voice covers all 3,970 corpus chars (the real model's Voice); the micro-world lexicon is 141 of them.
- **Re-articulation [clar]:** a new syllable starts when the active muscle set changes, after silence, or when the force on the same set drops.
- **Eye [clar]:** 3 slots × (12 objects + 40 novel, 6 colours, counts 1–3). Each (slot, value) code is 4 units shared across slots plus 4 slot-specific units. The tutor's pointing stays on from the prompt until the exchange ends. Option `eye_layout='retino'`, proposed by the brainlike designer, uses codes identical to their copy.
- **Cue [clar]:** the force-mode END push is an all-zero cue vector.
- **Turn protocol [clar]:**
  - Free replies: the window ends at max(45 ticks after the prompt, 13 silent ticks after the agent stops), cut at 60.
  - Model mode: the model reply starts 3 ticks after the prompt; the agent then gets 12 ticks to start, and if it speaks, 5 silent ticks after it stops.
  - Force mode: the cue advances on each mouth onset; the next line comes at stop + 6.
  - Held-out pairs are never taught, only probed.
- **Tasks:**
  - B: what, slot, where.
  - C: offer, taste, colour, count, ritual.
  - D: check, name (我叫N … 我叫什么), before (redesigned so it needs scene memory), oneshot (40 novel objects, each named once, then retention probes).
  - A: child-directed statements, overheard two-speaker routines, play periods where the tutor imitates babbles with reward 0.2.
  - E: corpus training dialogues from corpus[:-3000]; 500 never-heard test pairs and 40 familiar pairs; interleaved 50/50 with B–D.
- **`evaluate()`:** per stage and task:
  - acceptable rate (exact and first-char), split held-out / seen / familiar;
  - echo-free score, only on items where every acceptable answer keeps a char after removing prompt chars [clar], and the parrot index;
  - mechanics, guided speech, imitation after the model reply, learning early vs late, one-shot retention by delay, play stats;
  - ARCH_v1's 20-way metric, plus `top1_distinct`, which draws distractors only from different reply texts.
- **Rules for agents:** use only `obs`, plus `world.warmup_windows()` once at birth. The bench cannot enforce this.

## 3. Validity checks (selftest, 29/29 PASS)

| area | what is checked |
|---|---|
| Physics | Voice and mouth rule equal `voice_orig.py` (seeds 0 and 1). Stream equals ARCH_v1's over 4,000 random ticks. Silence energy is exactly 0. Lawful physics: layer-0 Jaccard 0.12 / 0.24 / 0.40 / 0.63 for 0–3 shared muscles (random: 0.21 flat); corr(sound distance, shared muscles) −0.74 (random −0.02); max pairwise Jaccard 0.77 over the lexicon and 0.89 over all 3,970 chars, no pair above 0.9, own voice identified 100%. |
| Timing | Determinism; agent-independent schedule; probe and model gaps; force-mode repeats 哈哈哈 / 好好 / 对对对 / 嗯嗯 said exactly. |
| Metrics | Hand-made cases and a synthetic log. |
| Baselines | Echo and lookup 0 and oracle 1 in B, C, D, E. |
| Memory tasks | Repeating the last heard reply scores 0.03 on `before`, 0.00 on `name`, 0.00 on `oneshot`. |
| Body (babbling) | Programs appear with no cue at all. Stage A run: 144 babbles, 54 imitated by the tutor, 54 kept, all real chars. |
| Options | `eye_layout='retino'` and `mouth_vocab='lexicon'`. |
| Speed | 2,000 dialogues of the trivial agent well under the 180 s target. |
| ARCH_v1 | Tick-identical cross-check and metric agreement with run.py's audit. |

## 4. Numbers (2,000 dialogues per stage; ALL / held-out / scene acceptable rate; 3 seeds, mean±sd)

| stage | bar (refv1_mem, force) | echo | prior | lookup | ngram | oracle |
|---|---|---|---|---|---|---|
| B | 0.12±.01 / 0.11 / 0.12 | 0 / 0 / 0 | 0.14 / 0.11 / 0.14 | 0 / 0 / 0 | 0.14 / 0.11 / 0.14 | 1 / 1 / 1 |
| C | 0.19±.01 / 0.07 / 0.09 | 0.10 / 0 / 0 | 0.30 / 0.29 / 0.23 | 0 / 0 / 0 | 0.52 / 0.29 / 0.24 | 1 / 1 / 1 |
| D | 0.19±.02 / 0.46 / 0.22 | 0 / 0 / 0 | 0.22 / 0.50 / 0.23 | 0 / 0 / 0 | 0.15 / 0.15 / 0.15 | 1 / 1 / 1 |

The `last` baseline scores 0 everywhere.

Per task, bar / prior / n-gram:

| task | bar | prior | n-gram |
|---|---|---|---|
| B: what | 0.06 | 0.06 | 0.04 |
| B: slot | 0.00 | 0.07 | 0.06 |
| B: where | 0.32 | 0.31 | 0.32 |
| C: colour | 0.08 | 0.16 | 0.17 |
| C: count | 0.10 | 0.31 | 0.30 |
| C: offer | 0.04 | 0.07 | 0.48 |
| C: ritual | 0.62 | 0.20 | 1.00 |
| C: taste | 0.37 | 1.00 | 1.00 |
| D: check | 0.46 | 0.48 | 0.31 |
| D: name | 0.00 | 0.12 | 0.12 |
| D: before | 0.01 | 0.08 | 0.08 |
| D: oneshot | 0.00 | 0.00 | 0.00 |

- **Learning in free use** (seen items, first → last third): C drops 0.34 → 0.23; D drops 0.12 → 0.08; B goes 0.11 → 0.14.
- **Bar mechanics over the 9 runs:** 10,895 free replies all at latency 6, 90,173 syllables all exactly 3 ticks, 0 overlap, 0 garble, 0 extra turns, guided speech exact (1.00) in all 9, 0 back-term hits.
- **Stage E, 20-way metric on never-heard corpus pairs** (seed 0): bar 0.11 / AUC 0.57 / copy 0.44 (shuffled 0.07). Echo scores 0.29 / 0.67: the metric rewards echoing, as the develop review found. Prior 0.05, lookup 0.05, n-gram 0.06. Inside stage E the bar's micro-world replies collapse to B 0.03 and C 0.06.
- **v1 without a teacher:** silent in model mode (B) and recast mode (C).

## 5. Bench issues found and fixed during the phase

- **`before` was solvable by recency.** RefV1 scored 0.38 by repeating the answer it heard just before. Now the changed slot is never named before the change, and half the time the new object is named right before the question. RefV1 now scores 0.01 (base config 0.11), the prior 0.08.
- **Echo-free definition.** It was penalising correct answers that are fully contained in the prompt; such items are now excluded. Fixed before v0.1.
- **Syllable counting at window edges.** Syllables that started before a window or were still sounding at its end were counted with the wrong length; now only complete syllables count.
- **20-way metric ties.** Duplicate reply texts among the distractors cap even the oracle below 1.00 (0.98, 0.73 on familiar pairs). I kept ARCH_v1's definition and added `top1_distinct`.
- **ARCH_v1's FEAT0 is not the usage rule.** archv1.py's warm-up adds the post-silence state seven times, so its FEAT0 is 1,152 neurons. The correct usage rule gives 1,153. `feat0_v1` reproduces archv1's version for exact equivalence.
- **Designer requests adopted:**
  - `eye_layout='retino'`.
  - `mouth_vocab='lexicon'`: a babble reaches 35 of the 141 lexicon chars after 323 babbles, versus 5 through the full 3,970-char mouth.
  - Two traps documented: v1's corollary-discharge timer runs after a blocked GO, and 30% of model replies come from the partner itself.

## 6. Known limits

- The anti-cheating rules can't be enforced; reviewers must check agent code.
- Some tasks can be solved without understanding: taste and rituals by a fixed answer, offer by copying the prompt. Read scores per task against the prior and n-gram columns. The real tests are scene questions (echo-free), held-out pairs and the D memory tasks.
- The micro-world is small: 12 objects, 6 colours, 3 counts, 3 slots, 8 names, 40 novel objects. That gives at most 40 true one-shot namings per world, and most retention probes come at long delays.
- With the full mouth, random babbling rarely reaches lexicon chars; use `mouth_vocab`, or lawful physics plus a learned inverse model.
- Noise is added as a max; with σ ≥ 0.5 the v1 body's talk detector hears speech all the time (by design).
- Learned float32 weights can differ in the last bit across numpy versions (scalar promotion changed in numpy 2), so long agent runs reproduce statistically, not bit-for-bit.

## summary_zh

测试台已完成并在使用中（READY v1.2，自检 29/29 通过）：
- **完成的功能：** 五个阶段、三种教学方式、两种发音物理、噪声、第三说话人、日志、评分、六个基线、身体模块、听觉前端和 v1 等价参考智能体都已实现。
- **有效性：** 重复问句和查表在没教过的题和看图题上都是 0，神谕是 1.00；固定种子完全可复现；v1 等价智能体与 ARCH_v1 原模型逐拍输出完全相同。
- **速度：** 2000 段对话只需约 3.5 秒。
- **基准线（v1）：** 机械方面全对，但内容与只看文字的基线持平或更低；看图题是瞎猜水平，记忆题为 0；没有老师推嘴时一句话也不说。
- **修过的问题：** "刚才"题原来靠重复上一个答案就能答对，已改；按设计者建议加了视网膜式视觉编码和限制发音字表两个选项。
- **需要处理：** 部分设计者的副本早于"刚才"题的修改，需重新拷贝；GPU 服务器的任务实际只有 1 个 CPU。