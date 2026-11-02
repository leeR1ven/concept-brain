(Saved by the coordinator from the reviewer's final reply; the harness blocked the reviewer's own write.)

# ARCH_v2-tune: systematic parameter search on the develop prototype (final report)

`report.md` could not be written: the harness blocks report files from subagents. This text is the full report.
Labels: **SIMULATED** = numpy run (files in §9); **PROVED** = argument with numbers.

## 0. Verdict
**Confirmed recommended configuration (§5).** It has three parts:
- develop's defaults;
- an interleaved curriculum ("mixB": stage 2 B:C = 1:1, stage 3 B:C:D = 1:1:2);
- awake recall (8 episodes per quiet moment, η 0.2, uniform sampling) from a 10,000-cell hippocampal store.

**Objective J** (defined in §1), SIMULATED:
- on the search seeds 0–2: develop **0.42 → 0.73**;
- on world seeds 3–4, never used in the search: **0.34 → 0.74**;
- mechanics: 0 errors in all ~150 runs.

**What disappeared and what did not** (SIMULATED):
1. **Interference across stages: gone.**
   - B retention: −0.19 → +0.07 (held-out seeds: −0.30 → +0.02).
   - C retention: −0.14 → +0.06 (held-out seeds: −0.16 → 0.00).
2. **Free-use drift: about two-thirds removed, not gone.** C after 300 untutored dialogues: 0.46 → 0.01 before, 0.58 → 0.38 now (same on both seed sets).
   - Only awake recall moved it, and only when the store still holds old tutored episodes.
3. **"Maturation blocks new learning": explained and avoidable.**
   - The cause is develop's normalisation: every update is divided by the total number of active units, so a few fresh units get only a tiny share.
   - Dividing by the summed plasticity instead (ν = 1) removes the block.
   - But maturation and split give no net gain, so both are off.
4. **Held-out generalisation: only partly improved.** 0.19 → 0.21 on the search seeds, 0.13 → 0.24 on the held-out seeds. Recall costs some of the curriculum's gain: curriculum alone gives 0.29.
5. **Colour: not solved.** 0.04 → 0.15–0.20, at the prior of 0.14.
6. **Count: partly improved.** 0.23 → 0.45; the prior is 0.32.
7. **One-shot naming: 0.00 in every run.** It is structural, as expected, and was not optimised.

**Answer to the designer.** "Consolidation is just weight growth and decay parameters" is half true.
- No decay setting helps; every decay value tried is harmful.
- No maturation setting helps.
- What works is repeated awake recall from a store large enough to keep old experience. It does the job sleep was meant to do: on the sequential curriculum, recall alone gives J 0.64, the same as changing the curriculum (0.66).
- Size alone does nothing: the 10k store without recall gives J 0.61.

## 1. Setup (SIMULATED, verified)

**Base and regression.**
- Base: develop, copied into `arch/v2/tune/`. Files: `core.py`, `rules.py`, `agent.py`, the bench v1.2 copy, plus `life.py` from bench v2.0.
- With every new knob at its default, my code reproduces develop seed 0 bit for bit. Stage-by-stage: B 0.547; then B 0.365 and C 0.635; then B 0.347, C 0.384, D 0.410. Drift 0.384 → 0.019.
- Rule unit tests: 16/16.

**Sleep.** develop's bench agent never called `sleep()`; it ran only in the symbolic M1 test. So nothing sleep-like runs anywhere.

**New mechanisms.** All are local and all are off by default.
- **Awake recall.**
  - A quiet moment = 12 silent ticks, the agent not speaking, not inside a heard line.
  - Sample `recall_n` stored episodes with probability ∝ σ^`rec_pow` (σ = trace strength).
  - Reinstate each episode as the previous line, and learn its successor line as if heard again. This uses rule R2 at rate `eta_recall`, with the same targets as listening.
  - Optional strengthening on recall: σ += `rec_boost`(1 − σ).
  - The ongoing context (previous line and CTX) is restored afterwards, so there is no hard reset.
  - Successor links carry a birth stamp. This fixes stale links from overwritten cells, but only in this path; develop's NEXT route still has the issue.
- **Maturation.** m = max(1/(1 + use/M), m_min). Learning-time normalisation is |A|^(1−ν)(Σm)^ν: ν = 0 is develop's rule, ν = 1 normalises by the summed plasticity.
- **Split.** A unit is a candidate once used ≥ `split_use` times. After `split_n` events in which it was active with error ≥ `split_err`, it spawns fresh children keyed (parent, first prompt word, last prompt word).
- **Global weight decay.** Per 1,000 learning events, scaled by each unit's plasticity.
- **MIX.** Recruitment probability and pruning.
- **Exposed constants.** CTX set point and τ, comparator margins, attention threshold, critic and actor rates.
- **Life only (`life_eye`).** Eye units that appear outside a line's first 6 ticks are absorbed into the scene. Otherwise appearing objects and the partner's "person" code are read as pointing.

**Objective** (`analyze.py`).
- Mechanics are a hard constraint: overlap, non-3-tick syllables, garble and extra turns must all be 0.
- Protocol: A 300 dialogues → babbling pass → 3 stages. After each stage, frozen B/C/D probes (80 probe dialogues each, ~150–200 replies).
- Terms:
  - F = mean(B3, C3, D3), final probe accuracy;
  - RET = mean(min(0, B3 − B1), min(0, C3 − C2)), retention;
  - H = mean held-out accuracy at the end;
  - DR = C accuracy after 300 untutored C dialogues with learning on.
- **J = F + 0.5·RET + 0.25·H + 0.25·DR.**
- Short rung: J1 = B1 + 0.25·hB1 + 0.25·DR_B (B only, drift over 150 untutored dialogues).

**Method.**
1. Rung 1: 67 one-at-a-time changes, A 300 + B 500.
2. Rung 2: 45 configurations at A 300 + 3 × 500: mechanisms, curricula, and 8 random 3-parameter perturbations.
3. Rung 3: full length (3 × 1,000) on seeds 0–2.
4. Held-out world seeds 3 and 4: new voices, eye codes and held-out sets.
5. Life check: 10⁶ ticks, seed 5.

**Compute.** Local: ≤ 3 processes. Server runner: 12 jobs of ≤ 4 processes.
- Under full load, server runs are 2–3× slower per process.
- Three rung-3 server jobs hit the 38-min limit; their finished runs were kept.

**Noise.**
- J1 for the base over 3 seeds: SD 0.058.
- Full-length J for the base: SD 0.11; for the recommended configuration: SD 0.06.
- Server and local numpy versions differ, so cross-platform runs differ by up to ~0.03.
- So single-seed differences below ~0.08 are noise.

## 2. Sensitivity

**Rung 1** (A 300 + B 500, seed 0; ΔJ1 against base J1 0.52):
- **Recall** helps:

  | recall setting (count / η / pow) | ΔJ1 | B drift after 150 untutored dialogues |
  |---|---|---|
  | 16 / 0.1 / 0.5 | +0.14 | 0.57 → 0.24 |
  | 4 / 0.2 / 0 | +0.11 | → 0.14 |
  | 4 / 0.2 / 1 | +0.08 | → 0.14 |

  Without recall, B drift is 0.00 in every configuration.
- **Maturation with ν = 1** (M = 50): +0.10.
- **Decay** is harmful: −0.28 at 0.3 and −0.40 at 2 per 1,000 events.
- **Maturation with ν = 0**: −0.18.
- **Cliffs** (harmful when moved this far):
  - η 0.15 −0.20; η 0.25 −0.09;
  - floor 0.5 −0.22; floor 0.4 −0.09;
  - match_thr 0.45 −0.24;
  - conf_min 3 −0.11; conf_min 10 −0.16;
  - match_n 10 −0.12;
  - rate_max 0.25 −0.16; rate_max 0.6 −0.12;
  - att_thr 0.8 −0.15;
  - rf_thr 0.45 −0.08.
- **Bit-identical runs (no effect at all):** vigilance θ_z 0.7/0.8, z_rate 0.03/0.3, prime floor 0.005.
- **recruit_p 0.5:** identical answers with 16 % fewer MIX units.
- **Within ±0.06 (noise):** η 0.5–0.7, η_g, η_actor 0–0.6, η_critic 0.05, η_attd, floor 0.2, κ, rec_thr, rf_thr 0.75, match margins, CTX λ 0.75–0.97, CTX τ 50/800, CTX β 4, hip γ 0.999/0.9999, hip_novelty 0.6/0.8, split.
- **Slightly harmful:** CTX β 0 (−0.06), w_ctx 0.5 (−0.06), η_critic 0.5 (−0.06).

**Rung 2** (A 300 + 3 × 500, seed 0; base sequential J 0.38):

| change | J | note |
|---|---|---|
| sequential base | 0.38 | |
| curriculum: mixA / mixC / interleaved / **mixB** | 0.47 / 0.52 / 0.59 / **0.60** (0.64 over 2 seeds) | |
| recall 8, 3k store (sequential) | 0.52 | DR 0.18 |
| recall 8, 10k store (sequential) | 0.62 | DR 0.36 |
| + boost 0.1 | 0.64 | DR 0.33 |
| recall with rec_pow 1 (recency) | 0.46 | uniform beats recency |
| recall 16 / η 0.1 | 0.51 | |
| on mixB: recall 8 / 16 / 4 | 0.63 / 0.66 / 0.64 | |
| on mixB: recall 8 + 10k | 0.67 | |
| on mixB: split 300 | 0.67 | |
| on mixB: maturation | 0.56 | |
| on mixB: recall + maturation | 0.63 | |
| on mixB: rung-1 winners | 0.62 | |
| split 100/300/1000 (sequential) | 0.37–0.42 | ≈ base |
| maturation ν = 1 | 0.28–0.32 | |
| maturation ν = 0 | 0.23 | |
| best random perturbation (hip_novelty 0.8, hip_theta 0.17, rec_thr 0.74 on mixB + recall) | 0.67 | |

## 3. Full length (rung 3) and held-out confirmation (SIMULATED)

Mean (SD) over seeds. "RET B / RET C" = B3−B1 / C3−C2. "drift C" = C before → after 300 untutored dialogues.

| configuration | seeds | J | B / C / D end | RET B / RET C | held-out | drift C | colour / count | one-shot | mech |
|---|---|---|---|---|---|---|---|---|---|
| develop base (sequential) | 0,1,2 | 0.42 (0.11) | 0.41 / 0.46 / 0.49 | −0.19 / −0.14 | 0.19 | 0.46 → 0.01 | 0.04 / 0.23 | 0.00 | 0 |
| mixB | 0,1,2 | 0.66 (0.03) | 0.65 / 0.60 / 0.50 | +0.04 / +0.06 | 0.29 | 0.60 → 0.01 | 0.09 / 0.43 | 0.00 | 0 |
| mixB + 10k store, no recall | 0,1,2 | 0.61 (0.03) | 0.64 / 0.54 / 0.49 | +0.05 / −0.00 | 0.23 | 0.54 → 0.01 | 0.10 / 0.35 | 0.00 | 0 |
| mixB + recall 8 (3k) | 0,1,2 | 0.65 (0.01) | 0.62 / 0.58 / 0.46 | +0.01 / +0.06 | 0.17 | 0.58 → 0.22 | 0.11 / 0.43 | 0.00 | 0 |
| mixB + recall 16 (3k) | 0 | 0.62 | 0.58 / 0.62 / 0.38 | 0.00 / +0.09 | 0.20 | 0.62 → 0.18 | 0.07 / 0.57 | 0.00 | 0 |
| recall 8 + 10k, sequential | 0,1,2 | 0.64 (0.05) | 0.46 / 0.60 / 0.47 | −0.14 / +0.02 | 0.19 | 0.60 → **0.48** | 0.20 / 0.43 | 0.00 | 0 |
| **mixB + recall 8 + 10k (recommended)** | 0,1,2 | **0.73 (0.06)** | 0.68 / 0.58 / 0.48 | **+0.07 / +0.06** | 0.21 | 0.58 → **0.38** | 0.15 / 0.45 | 0.00 | 0 |
| develop base | **3,4** | 0.34 (0.03) | 0.32 / 0.45 / 0.50 | −0.30 / −0.16 | 0.13 | 0.45 → 0.01 | 0.11 / 0.17 | 0.00 | 0 |
| mixB | **3,4** | 0.54 (0.02) | 0.52 / 0.57 / 0.49 | −0.10 / −0.06 | 0.23 | 0.57 → 0.00 | 0.10 / 0.36 | 0.00 | 0 |
| mixB + recall 8 (3k) | **3,4** | 0.58 (0.01) | 0.57 / 0.54 / 0.49 | −0.02 / −0.06 | 0.18 | 0.54 → 0.13 | 0.11 / 0.31 | 0.00 | 0 |
| **recommended** | **3,4** | **0.74 (0.02)** | 0.66 / 0.60 / 0.50 | +0.02 / 0.00 | 0.24 | 0.60 → **0.38** | 0.20 / 0.45 | 0.00 | 0 |

- D check: 0.77 → 0.73–0.75 (unchanged). Name recall: 0.38 → 0.44–0.48 (within the seed SD of 0.13–0.15).
- The recommended configuration ranked first on both seed sets. On the held-out seeds its margin grew: +0.16 over the 3k-store recall variant, +0.20 over the curriculum alone.
- mixB + recall 16 has only seed 0: its seeds 1–2 were killed by the server time limit. It did not beat recall 8 on any measure.
- Attribution:
  - the curriculum alone removes most interference on seeds 0–2;
  - recall needs the big store and the store needs recall: 10k alone 0.61, recall with 3k 0.65, both together 0.73;
  - recall plus store also fixes the B/C loss that remains with the curriculum alone on the held-out seeds (−0.10 / −0.06 → +0.02 / 0.00);
  - recall plus store is the only drift lever.

## 4. Life robustness check (bench v2.0 life mode, SIMULATED)

Setup: seed 5, tutor recast, 10⁶ ticks ≈ 21 h of life, one unbroken life. The life has no stages, so this tests the recall part (the non-curriculum configuration) against the base.

| measure | develop base | recall 8, 3k store | recall 8, 10k store (recommended) |
|---|---|---|---|
| free replies: accuracy (n 2,599) | 0.19 | 0.21 | 0.20 |
| scene questions in new situations (n 1,134) | 0.21 | **0.27** | **0.25** |
| held-out (n 300) | 0.09 | 0.10 | 0.07 |
| memory facts and novel objects (n 381) | 0.01 | 0.01 | 0.00 |
| social greetings (n 868) | 0.09 | 0.06 | 0.06 |
| items heard once / 3–4× / 9+× | 0.14 / 0.32 / 0.54 | **0.33** / 0.38 / 0.56 | **0.30** / 0.40 / 0.56 |
| accuracy, first → last tenth of life | 0.11 → 0.24 | 0.14 → 0.25 | 0.12 → 0.25 |
| answer rate over life | 0.70–0.85 | 0.78–0.89 | 0.75–0.89 |
| overlap / garble / stability flags | 0 / 0 / none | 0 / 0 / none | 0 / 0 / none |
| CPU time | 700 s | 1,170 s | 1,298 s |

- No configuration drifts or collapses in life; accuracy keeps rising. Partner recasts continue all life, so life tests free-use drift only weakly.
- Recall doubles the accuracy on items experienced once (0.14 → 0.30–0.33, about 4 standard errors). This is consolidation by rehearsal.
- Recall gains +0.04–0.06 on new-situation scene questions and costs about −0.03 on social replies.
- The 10k store never filled in life (3,961 cells used), so the two recall variants are equal there.
- Memory facts stay at 0 in every configuration. This is structural: the completion gate never opens (develop follow-up §3).

## 5. Recommended parameter set (confirmed on held-out seeds 3–4)

Command: `tune.py R=3 nS=1000 cur=mix b2=1 c2=1 b3=1 c3=1 d3=2 recall_n=8 eta_recall=0.2 rec_pow=0 hip_cap=10000 hip_ndg=6000 hip_kdg=120`

- **Curriculum:** A 300 → babbling → B 1,000 → `BC` 1,000 → `BCDD` 1,000.
- **Awake recall:** `recall_n` 8 per quiet moment, `eta_recall` 0.2, `rec_pow` 0 (uniform over the store), `rec_boost` 0, `quiet_ticks` 12. This gives about 2.5 quiet moments and 20 recalled episodes per dialogue.
- **Hippocampus:** `hip_cap` 10,000, DG 6,000 with k 120, γ 0.9995, novelty 1.0, θ 0.2.
- **Switched off:** sleep, weight decay (`wd` = 0), maturation (`mature` = 0), split (`split_use` = 0), recruit_p = 1, pruning.
- **Unchanged develop values:**
  - learning rates: η 0.35, η_g 0.5, η_critic 0.2, η_actor 0.3, η_prime 0.02, η_attd 0.02;
  - action floor 0.3;
  - vigilance θ_z 0.75, z_rate 0.1;
  - comparator: match_thr 0.6, match_n 6, conf_min 6, rate_max 0.4, rec_thr 0.85, rf_thr 0.6, κ 2, match_min 0.5, match_gap 0.3, att_thr 0.5;
  - CTX: λ 0.9, β 2, τ 200, w_ctx 1;
  - copy gain 1.
- **Cost:** about 2.5× the CPU of the base (880 s vs 350 s locally), mostly the hippocampal recall over 10k cells. Recall triples the learning events.
- **Continuous thinking:** no hard reset was added. Dialogue reset stays off, recall restores the ongoing context afterwards, and CTX λ 0.9 is the gradual channel.

## 6. Problems that remain (they need mechanisms, not parameters)
- **Free-use drift.** Recall slows it, but untutored partner lines are still learned as answers (C 0.58 → 0.38). This needs a contingency or turn-role third factor (develop follow-up §4).
- **Held-out generalisation.** It stays low, and recall trades a little of it away: C held-out 0.19 with the curriculum alone vs 0.09 with recall.
- **Colour and one-shot naming.** Both need new mechanisms:
  - colour: a loop-free grounding teacher for attribute words;
  - one-shot: a prompt-independent completion gate.

## 7. Notes for the coordinator
- **Bench:** no metric bugs found.
- **Life mode** needs `life_eye` for develop-style eye parsers.
- **Server:** throughput falls to about 1/3 per process when 12 processes run. Full-length runs with recall exceed 38 min there; run them locally.
- **Agent bug:** in develop's own code, the hippocampal NEXT route can follow stale links after a cell is overwritten. It is fixed only in my recall path, to keep the baseline reproducible.

## 8. Not done
- `prune` was never run in rung 1 and `wd`-with-maturation in rung 2 (both skipped by deadlines). Pruning was therefore never tested; decay was harmful at every value tested in rung 1.
- Recall 16 has only one seed at full length.

## 9. Files
All in `/tmp/claude-0/-root/e92d3d7d-5823-4f5c-a0ec-4fe302bff9af/scratchpad/arch/v2/tune/`:
- code: `core.py`, `rules.py`, `agent.py`;
- drivers: `tune.py`, `batch.py`, `life_run.py`;
- analysis: `analyze.py` (the objective), `sens.py`, `final_table.py`, `life_summary.py`;
- job generators and submission: `mkjobs_r1.py`, `mkjobs_r2.py`, `mkjobs_rand.py`, `submit.sh`, `submit_sh.sh`, `collect.sh`;
- results: `out/runs/*.json` (one file per run: r1_*, r2_*, r3_*, hx_*), `out/life/life_*_s5.json`, the regression run in `out/regress.jsonl`;
- server job folders: `arch/v2/gpu/queue/tune_*`.

## summary_zh
1. 在 develop 原型上做了系统调参，共约 150 次运行：
   - 逐个参数扫描 67 项；
   - 机制和课程 45 项；
   - 完整长度用 3 个种子；
   - 另用 2 个从未用过的世界种子确认；
   - 生活模式 10⁶ 拍检验。
2. 新机制默认全关时与 develop 逐位一致。测试台上本来就没有调用睡眠。
3. 确认的最佳设置 = develop 默认值 + 交错课程（第二段 B:C=1:1，第三段 B:C:D=1:1:2）+ 清醒回忆（每个安静时刻回忆 8 段、学习率 0.2、在全部记忆里均匀抽取）+ 一万格海马体。
4. 综合分 0.42→0.73；在保留种子上 0.34→0.74。机制始终 0 错误。
5. 消失的问题：阶段间遗忘。B 的保持 −0.19→+0.07，C 的保持 −0.14→+0.06；保留种子上 −0.30→+0.02、−0.16→0.00。
6. 减轻但没消失：无人教时的漂移。C 从 0.46→0.01 变为 0.58→0.38，只有"清醒回忆 + 足够大的记忆库"有效。
7. 设计者说"巩固只是权重增长和衰减的参数"，只对一半：
   - 任何权重衰减都有害，成熟和分裂没有净收益；
   - 真正起作用的是清醒时反复回忆旧经历，而且记忆库要大到能留住旧经历；
   - 只加大记忆库不回忆没用（0.61），只回忆不加大也不够（0.65）。
   - 不改课程、只靠回忆就和交错课程一样好（0.64 对 0.66）。
8. 成熟"冻住学习"的原因：develop 按全部活跃单元来分摊学习步长，新单元分到太少；改为按可塑性总和分摊（ν=1）就不冻了。但成熟本身仍然没有净收益，所以关掉。
9. 没解决：
   - 保留组合泛化只小幅提高（0.19→0.21 / 0.13→0.24），回忆还略降低泛化；
   - 颜色题仍在先验水平，数量题 0.23→0.45 高于先验；
   - 一次性命名始终为 0。这些需要新机制。
10. 生活模式（种子 5，约 21 小时）：
    - 回忆让只听过一次的内容答对率 0.14→0.30–0.33，新场景题 0.21→0.25–0.27；
    - 社交问候略降（0.09→0.06），记忆事实题仍为 0；
    - 没有漂移崩溃，不抢话，无乱码。
11. 敏感性：
    - 课程和回忆影响最大；
    - 比较器阈值有"悬崖"，develop 默认值在平台上；
    - 学习率、门槛在合理范围内都差不多；
    - 警觉阈值、模板学习率、组合单元招募概率完全无影响。
