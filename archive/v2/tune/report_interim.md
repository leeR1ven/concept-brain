(Saved by the coordinator from the reviewer's final reply; the harness blocked the reviewer's own write.)

# ARCH_v2-tune: parameter search on the develop prototype (interim report; the harness required handback before the search finished)

**Status.** Rung 1 and most of rung 2 of the successive-halving search are done. Three steps are not done:
- the top configurations at full length on 2 more seeds;
- confirmation on held-out seeds;
- the coordinator's 10⁶-tick life run.

Round-A jobs were still running at handback: 24 runs in `gpu/queue/tune_r2A/B/C` and 4 runs plus 2 baseline seeds locally. The commands to finish are in §8. `report.md` was not written: handback was required immediately, so this text is the report.

Labels: SIMULATED = numpy run, file named; PROVED = argument with numbers.

## 1. Setup (SIMULATED, verified)

**Base.** The develop prototype, copied to `arch/v2/tune/` (`core.py`, `rules.py`, `agent.py`, bench v1.2 copy plus `life.py` from bench v2.0).

**Regression check.** With every new knob at its default, my code reproduces develop's seed-0 baseline bit for bit: B1 0.547; B 0.365 and C 0.635 after stage 2; B 0.347, C 0.384, D 0.410 after stage 3; drift 0.384 → 0.019; MIX 69,438 units. Rule tests pass 16/16.

**Sleep.** develop's bench agent never called `sleep()`; it ran only in the symbolic M1 test. So "sleep off" was already true on the bench.

**New mechanisms** (all local, all off by default):
- **Awake recall.** A quiet moment = 12 silent ticks, not speaking, not inside a line.
  - Sample `recall_n` stored episodes with probability ∝ σ^`rec_pow`.
  - Learn each episode's successor line as if heard again (rule R2 at `eta_recall`).
  - Optional strengthening on recall: σ += `rec_boost`·(1−σ).
  - The ongoing context is restored afterwards: no reset.
  - Successor links carry a birth stamp, because overwritten hippocampal cells left stale links. develop's NEXT route still has this stale-link issue; I left it unchanged to keep the baseline reproducible.
- **Maturation.** m_i = max(1/(1 + use_i/M), m_min). New normalisation at learning time: |A|^(1−ν)·(Σm)^ν. With ν = 1, the plastic units share the full step.
- **Split.** A unit used ≥ `split_use` times, and active in ≥ `split_n` events whose error is ≥ `split_err`, spawns fresh children keyed (parent key, first prompt word, last prompt word). New learning goes to the children.
- **Weight decay.** Global decay per 1,000 learning events, scaled by each unit's plasticity.
- **MIX.** Recruitment probability and pruning.
- **Exposed constants.** CTX set point and τ, comparator margins, attention threshold, critic and actor rates.
- **Life-mode eye** (life runs only). Eye units that appear outside the first 6 ticks of a heard line are absorbed into the scene. Without this, develop's parser reads appearing objects and the partner's person code as pointing gestures.

**Objective** (`analyze.py`). Mechanics are a hard constraint: overlap, non-3-tick syllables, garble and extra turns must all be 0.
- Full curriculum:
  - F = mean(B3, C3, D3) of the frozen probes after the last stage;
  - RET = mean(min(0, B3 − B1), min(0, C3 − C2)), i.e. retention against just after each stage was learned;
  - H = mean held-out accuracy at the end;
  - DR = C probes after 300 untutored C dialogues with learning on;
  - **J = F + 0.5·RET + 0.25·H + 0.25·DR**.
- Short rung (A300 + B500): J1 = B1 + 0.25·hB1 + 0.25·DR_B, where DR_B is measured after 150 untutored B dialogues.
- One-shot naming is reported, not optimised.

**Noise.**
- Base J1 over seeds 0, 1, 2: 0.52 / 0.495 / 0.605 (SD 0.058).
- The same configuration and seed differs between server numpy 2.1 and local numpy 1.26: B1 0.477 vs 0.447.
- So single-seed differences under about 0.08–0.10 are noise.

**Mechanics.** 0 violations in every run so far (> 100 runs, tens of thousands of exchanges). Latency is always 6.

## 2. Rung 1: A300 + B500, seed 0, 67 one-at-a-time changes (`out/runs/r1_*.json`, `sens.py r1_`)

| change | ΔJ1 | note |
|---|---|---|
| recall 16 / η 0.1 / pow 0.5 | **+0.14** | B1 0.57; B drift 0.57 → **0.24** (base → 0.00) |
| recall 4 / η 0.2 / pow 0 | **+0.11** | drift → 0.14 |
| maturation M 50, floor 0.05, ν = 1 | **+0.10** | B1 0.56 |
| recall 4 / pow 1 | +0.08 | drift → 0.14 |
| η_actor 0, η 0.7, floor 0.2, M 200 with ν = 1, CTX β 4 | +0.03 to +0.06 | noise |
| θ_z 0.7/0.8, z_rate 0.03/0.3, η_prime 0.005 | **0 (identical runs)** | no effect at all in this world |
| recruit_p 0.5 | 0 | identical probe answers with 16 % fewer MIX units |
| hip_novelty, hip γ, CTX τ/λ, κ, match margins, η_g, η_attd, rec_thr, split | ±0.04 | noise |
| maturation with ν = 0 (M 50) | −0.18 | reproduces develop's "maturation blocks learning" |
| η 0.25 / 0.15 | −0.09 / −0.20 | |
| floor 0.4 / 0.5 | −0.09 / −0.22 | |
| match_thr 0.45 | −0.24 | |
| conf_min 3 / 10 | −0.11 / −0.16 | |
| match_n 10 | −0.12 | |
| rate_max 0.25 / 0.6 | −0.16 / −0.12 | |
| att_thr 0.8 | −0.15 | |
| **weight decay 0.3 / 2** (per 1k events) | **−0.28 / −0.40** | decay erases learning |

B drift was 0.00 in every configuration without recall.

## 3. Rung 2: A300 + 3 × 500, full objective (`out/runs/r2_*.json`)

| configuration | J | B3 | C3 | D3 | RET | H | DR (C after drift) |
|---|---|---|---|---|---|---|---|
| develop base, sequential (2 seeds) | 0.338 | 0.29 / 0.35 | 0.42 / 0.24 | 0.47 / 0.48 | −0.13 / −0.23 | 0.21 / 0.18 | 0.01 / 0.00 |
| **mixB curriculum** (stage 2 B:C 1:1; stage 3 B:C:D 1:1:2; 2 seeds) | **0.641** | 0.55 / 0.74 | 0.61 / 0.58 | 0.37 / 0.53 | 0.00 / −0.01 | 0.35 / 0.26 | 0.01 / 0.00 |
| interleaved B, BC, BCD | 0.589 | 0.66 | 0.62 | 0.23 | 0.00 | 0.34 | 0.01 |
| mixC (BCC, BCDDD) / mixA (BCCC, BCDDDD) | 0.524 / 0.470 | | | | | | |
| **awake recall** 8 / η 0.2 / pow 0, sequential | **0.520** | 0.33 | 0.54 | 0.50 | −0.10 | 0.27 | **0.18** (from 0.54) |
| split 100 / 300 / 1000 (with maturation ν = 1) | 0.39 / 0.37 / 0.42 | | | | | | 0.01–0.03 |
| split 300 without maturation | 0.37 | | | | | | |
| maturation ν = 1, M 50 / M 200 / with gate | 0.32 / 0.28 / 0.31 | | | | | | |
| maturation ν = 0, M 50 | 0.23 | B1 0.30 | | 0.23 | | | |

- Recall also gave check 0.80 and name 0.71 on seed 0.
- Split and maturation with ν = 1 are within noise of the base.

**Life test** (seed 5, 10⁵ ticks, `gpu/queue/tune_lifet1`):
- base: free accuracy 0.10, new scenes 0.06;
- recall 8: free accuracy 0.15, new scenes 0.17;
- both: 0 overlap, 0 garble, latency 6, no stability flags;
- speed: 56–79 s per 10⁵ ticks, so a 10⁶-tick life takes about 10–15 min.

## 4. Per failure: develop baseline vs the best so far (rung 2; not yet confirmed at full length or on held-out seeds)

| failure | develop baseline | best so far | verdict |
|---|---|---|---|
| interference (RET) | −0.13 / −0.23 (full length, seed 0: B 0.55 → 0.35, C 0.64 → 0.38) | mixB 0.00 / −0.01; recall alone −0.10 | **disappears with an interleaved curriculum** (cheap and robust); recall alone halves it |
| free-use drift (C after 300 untutored dialogues) | 0.42 → 0.01 | recall: 0.54 → **0.18**; B drift 0.57 → 0.24 | **reduced, not removed.** Awake recall is the only knob that moved it; learning rates, maturation, decay and split do not |
| held-out generalisation | H 0.21 / 0.18 | mixB 0.35 / 0.26; interleaved 0.34 | improved by interleaving; still low (B held-out about 0.1–0.2) |
| colour / count | 0.04 / 0.00 and 0.20 / 0.43 (prior 0.14 / 0.32) | 0.11 / 0.00 and 0.37 / 0.37 | **not solved**: prior level (structural: no grounding teacher for attribute words) |
| one-shot naming | 0.00 | 0.00 in every run | not optimised; structural (the completion gate never gets a teaching event) |
| maturation blocks D | reproduced (ν = 0: D 0.23) | ν = 1 removes the block | the cause is the normalisation over all active units, not maturation itself. But maturation gives no net gain |
| mechanics | 0 | 0 | kept |

## 5. Sensitivity
- **Matters a lot (> 0.1):**
  - curriculum interleaving (+0.25 to +0.30);
  - awake recall (+0.14, and it is the only drift lever);
  - any weight decay (harmful);
  - maturation with ν = 0 (harmful);
  - the comparator thresholds. match_thr, conf_min, match_n, rate_max and att_thr are cliffs, and the develop defaults sit on the plateau;
  - η ≤ 0.25 and floor ≥ 0.4 (harmful).
- **Does not matter:** vigilance θ_z, z_rate, the prime-plasticity floor, MIX recruitment probability, hippocampal novelty and γ, CTX λ/τ, κ, comparator margins, gate, attention and critic rates.
- **Neutral within noise:** split, maturation with ν = 1, η_actor 0, η 0.5–0.7, floor 0.2.

**Answer to the designer.** Two problems largely disappear with parameters alone: interference (via the curriculum mix) and part of the drift (via awake recall). Held-out improves. Drift is not removed. Colour/count and one-shot naming do not move with any parameter.

## 6. Recommended parameter set (provisional)
- develop defaults (η 0.35, η_g 0.5, floor 0.3, match_thr 0.6, match_n 6, conf_min 6, rate_max 0.4, hip_cap 3,000, CTX λ 0.9);
- curriculum mixB (stage 2 'BC', stage 3 'BCDD');
- awake recall: `recall_n` 8–16, `eta_recall` 0.1–0.2, `rec_pow` 0–0.5, `quiet_ticks` 12;
- no weight decay;
- maturation off (if on, ν = 1, never ν = 0);
- split off.

This combination has not finished running: it is in round A (`r2_mixB_rec8`, `rec16`, `rec32` and the random perturbations). It must be confirmed at full length (nS = 1000) on seeds 0, 1, 2 and then on held-out seeds 3 and 4 against `r3_base`.

## 7. Notes for the coordinator
- **Server speed.** Under full load (12 processes) the server is about 3× slower per process than local: rung-2 runs took 700–1,080 s. Ten rung-2 jobs were skipped by my batch deadline and were resubmitted.
- **Bench.** No metric bugs found.
- **Agent-level note.** Life mode needs `life_eye`, otherwise the develop agent's eye parser treats appearing objects and the person code as pointing.

## 8. How to finish (all scripts are in `arch/v2/tune/`)
1. `./collect.sh`, then `python3 analyze.py "out/runs/r2_*.json" --R 3` (round A results).
2. Rung 3 for the top 3–4 configurations, nS = 1000, seeds 0, 1, 2: `tune.py tag=r3_X seed=S R=3 nS=1000 evaln=80 drift=300 <cfg>`, submitted with `./submit.sh <label> <jobs.txt> 4 38`. The baseline is `r3_base` (seed 0 already done; seeds 1 and 2 are running locally).
3. Held-out seeds 3 and 4: the best configuration vs `r3_base`.
4. Life robustness check: `python3 life_run.py tag=best seed=5 ticks=1000000 <non-curriculum cfg>` and the same with the base config, then `python3 life_summary.py "out/life/*.json"`.

**Files** (all in `/tmp/claude-0/-root/e92d3d7d-5823-4f5c-a0ec-4fe302bff9af/scratchpad/arch/v2/tune/`):
- code: `core.py`, `rules.py`, `agent.py`;
- drivers: `tune.py`, `batch.py`, `life_run.py`;
- analysis: `analyze.py`, `sens.py`, `life_summary.py`;
- job generators: `mkjobs_r1.py`, `mkjobs_r2.py`, `mkjobs_rand.py`;
- results: `out/runs/*.json`, regression run `out/regress.jsonl`.

## summary_zh
1. 在 develop 原型上做了系统调参（逐个参数扫描 + 逐级淘汰）。新加的机制默认全关时，结果和 develop 完全一致。测试台上原本就没有调用睡眠。
2. 新机制都是局部规则：
   - 安静时刻的清醒回忆：按强度抽取旧情景，当作重听一遍来学；
   - 使用成熟（发现"按全部活跃单元归一化"才是阻塞学习的原因）；
   - 分裂；
   - 全局衰减；
   - 生活模式的眼睛处理。
3. 交错课程（第二段 B:C=1:1，第三段 B:C:D=1:1:2）基本消除阶段间遗忘：保持度从 −0.13/−0.23 变为 0.00/−0.01，综合分 0.34→0.64。
4. 清醒回忆是唯一能减轻漂移的参数：C 在无人教 300 段对话后从 0.54 只降到 0.18，原来降到 0.01；综合分 +0.14。漂移没有完全消失。
5. 任何权重衰减都有害；ν=0 的成熟有害；分裂和 ν=1 的成熟都在噪声内，没有好处。
6. 比较器阈值有"悬崖"，develop 的默认值正好在平台上。听觉警觉阈值、模板学习率、组合单元招募概率等完全没有影响。
7. 颜色/数量题和一次性命名不随任何参数变化，是结构问题。
8. 机制（不抢话、每字 3 拍、无乱码）在所有运行中始终为 0 错误。
9. 未完成：完整长度三种子复核、保留种子确认、10⁶ 拍生活模式检验。剩余作业已在跑，命令见 §8。生活模式短测（10⁵ 拍）中回忆把正确率从 0.10 提到 0.15。
