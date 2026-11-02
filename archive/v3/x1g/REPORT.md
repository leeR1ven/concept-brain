(Written from the agent's two hand-backs (seed-0 pilots) and the 3-seed results that came back afterwards. In the repo: X1f/X1d/X1c files are imported from `../x1f`, `../x1d` and `../x1c`; `wire8.py` is X1g's copy of X1f's `wire.py`; `tables/` holds the aggregates (the per-life JSON are not committed). Every verdict is conditional on the audit in §8 and `AUDIT.md`, which found that the test model did not follow ARCHITECTURE.md strictly.)

# X1g report: does a sparser, larger PFC let the one Hebbian rule learn question-specific Q → A? (ARCH §四.20)

Labels: **PROVED** means 3-seed runs (seeds 0–2, mean ± SD) or an exact code check. **SIMULATED (seed 0)** means one seed: a pilot or a diagnostic. **ASSUMED** means my judgement, not tested. **NOT RUN** means the job list was written but never submitted.

## 结果（简明）

1. **四条及格线全部不及格**（3 个种子）。
2. **前额叶放大到 1,600 个神经元、每刻只亮约 30 个**：各个字的编码保持不同，没有掉进 X1e/X1f 的"万能状态"。但问题一结束 3 拍，前额叶就熄灭：12 拍时 96–97% 的情况一个神经元都不亮。所以停顿后认不出是哪个问题（0.005，及格线 0.3）。
3. **因此几乎答不出来**：听过的题第一个字 0.06–0.09（及格线 >0.2），针对性 ≤ 0.026，和随机打乱一样（随机的 95% 线 0.03）。内部连接学或不学，结果都一样。
4. **对照组**：手动给每道题接好一条完美思路，每组 20 或 30 个神经元。学到的"预测答案"已经针对问题了：针对性 0.20–0.33，明显高于随机的 0.09–0.11。X1f 那种每组 5 个神经元时，这个数是 0。
5. **但对照组的新题仍说不出来**：第一个字 0.06–0.12（及格线 0.5），73% 的时候不说。错字得到的推动有答案的 0.65–0.9 倍，心里同时想到几个音，"阻断连接"就让它不说。换别的读出方式最好也只有 0.33。
6. **种子 0 的诊断**：那个错字几乎总是答案的第二个字，因为学习规则里"刚熄灭的也算"把第二个字也连了上去。去掉这一项后，答案总是推动最强的字，但孩子还是多半不说（0.18）。只有一个种子、一个检查点，原因没查清。
7. **出生时的回程环路**（种子 0）：只有把"说不说"的门槛放得很松，才能在停顿里保住想法。可学 300 段后又熄灭，孩子还会跟着重复听到的音。说答案要门槛严，保持想法要门槛松，两者矛盾。
8. **没跑的**：6,400 个神经元的主实验、门槛放松的回程环路组、400 个神经元的各组。任务单都写好了，没有提交。6,400 只有种子 0 的结果：内部连接不再两两全连（27%），但停顿时同样熄灭。
9. **断点**：第一处是"在停顿里保住问题"，第二处是"把想到的答案说出来"。光靠稀疏解决不了第一处。
10. **这些结论有前提，而审计发现测试模型没有严格照架构搭**（§8、`AUDIT.md`）：最关键的是安静时前额叶没有输入，想法保持的回路转不起来。所以这里的"学不会"不能算作架构不行。

## 0. Verdict table (criteria fixed in `PLAN.md` before the batches; best checkpoint, 3-seed mean)

| criterion | target | best value | verdict |
|---|---|---|---|
| 1. Positive control: new-pair first syllable | ≥ 0.5 | <ul><li>PC1600_m20: 0.12 ± 0.04 (3,000 and 10,000 dialogues)</li><li>PC6400_m30: 0.09 ± 0.00 (300–3,000)</li><li>post-hoc readout swaps on the same weights: best 0.33 ± 0.11 (a 0.5 + relative, PC1600)</li></ul> | **FAIL** (PROVED) |
| 2. Learned path: context-varied question decoding at lag 12, with within-path J < 0.8 | ≥ 0.3 | <ul><li>M1600_k30_on: 0.005 ± 0.008</li><li>M1600_k30_off: 0.005 ± 0.008</li><li>lag 0: 0.21–0.25; P(L3 empty) at lag 12: 0.96–0.97</li></ul> | **FAIL** (PROVED) |
| 3. Fair specificity on seen pairs, above the permutation null's 95 % | ≥ 0.10 | ≤ 0.026 ± 0.018 (M1600_k30_off at 3,000; null 95 % = 0.03) | **FAIL** (PROVED) |
| 4. Seen-pair first syllable | > 0.2 | 0.09 ± 0.04 (M1600_k30_on at 10,000) | **FAIL** (PROVED) |
| Addendum: birth return loop holds the path (same as criterion 2) | ≥ 0.3 | seed 0 only: 0.000 in all 5 loop lives (th1 0.015 with g_loop 4 / 16 / 64, L33 on / off; th1 0.005 with g_loop 16) | no verdict: the 3-seed loop arms were **NOT RUN**; seed 0 points to FAIL |

Where the chain breaks (PLAN: code → path → specific prediction → release):
- **Code:** fixed by sparsity (seed 0, §5.1).
- **Path:** breaks at every size and sparsity tried.
- **Specific prediction:** appears only when a path is hand-wired with realistic assembly sizes.
- **Release:** then blocks the answer.

## 1. What was built

All options are in `core8.py` and are off by default. The pre-registered details are in `PLAN.md`.

- **`core8.py`** is X1f's `core6.Brain` plus options.
  - **Defaults reproduce core6 bit-exactly** (PROVED, `eqtest8.py`): no tick differs over 6,037 ticks.
  - **`lat_fly`** computes the same ARCH two-layer L3 → L3 propagation every tick from the current weights, so N = 6,400 fits in numpy.
    - With L33 off it is identical to core6 (PROVED, 6,068 ticks, 0 differences).
    - With L33 on it differs from tick 1, because core6 refreshes its frozen P33 only every 25 ticks. The statistics match: L3 mean 96 vs 93, L33 positive-pair fraction 0.76 vs 0.77 (SIMULATED, seed 0).
  - **`homeo2` / `homeo3`:** each L2 / L3 unit's threshold slowly follows its own long-run activity toward k / N. This is ARCH §四.10's sanctioned next step ("按长期活跃程度自我调整门槛"). With k = 30 the active fraction falls as N grows: 7.5 % at 400, 1.9 % at 1,600, 0.47 % at 6,400.
  - **`a31` / `a33`:** the readout split, propagation = w / (out^a · in^(1−a)). a = 0.5 is the ARCH / X1f form; a = 1 is out-dominant.
  - **`rel1`:** release by sharpness instead of X1f's absolute floor 0.35. A line is predicted if its share x / (1 + Σx) ≥ th1 (0.015) **and** x ≥ 0.6 · max x.
  - **`fell_win31`** (added after the pilots, diagnostic only): the fall window of the top → L1 rule alone.
- **`wire8.py`:** X1f's wiring plus an option for overlapping assemblies.
- **`run8.py`:** the driver.
  - Stages: birth → A1 + A2 speech basics → (positive-control arms only) X1f G0P paths → overhearing with checkpoints at 300 / 1,000 / 3,000 / 10,000 dialogues → a frozen live world.
  - Measures at every checkpoint:
    - learned top → L1 drive into the answer vs the best wrong syllable;
    - fair specificity (X1e `other1`) with a 200-permutation null;
    - path decoding and motion;
    - accuracy by pair set;
    - child utterances while overhearing;
    - L2 / L3 counts.
  - Every checkpoint also re-probes the same weights under 10 readouts (7 pre-registered, 3 post-hoc with a stricter max-relative cut).
- **Code version.** The 3-seed jobs and `x1g_pilot1` ran `run8.py` / `core8.py` as they were when submitted. The published files add:
  - per-pair answers and a drive breakdown (`top_single`, `top_multi`, `best_wrong_is_a2`);
  - the 3-tick-window decoding in the output;
  - `fell_win31`.
  With defaults these change no dynamics. PROVED by diff: the added code only adds output fields, or is skipped when `fell_win31` is None.

## 2. What ran

| arm | N | seeds | overheard dialogues | where | status |
|---|---|---|---|---|---|
| M1600_k30_on / _off (sparse k 30, a 1, relative release; L33 on / off) | 1,600 | 0–2 | 10,000 | A800 job `x1g_sB` | PROVED |
| PC1600_m20 (positive control, 20-unit assemblies) | 1,600 | 0–2 | 10,000 | `x1g_pcA` | PROVED |
| PC6400_m30 (positive control, 30-unit assemblies) | 6,400 | 0–2 | 10,000 | `x1g_pcA` | PROVED |
| PL1600_on_g4 / _g16 / _g64, PL1600_off_g16 (birth return loop, th1 0.015) | 1,600 | 0 | 1,000 | `x1g_sB` | SIMULATED (seed 0) |
| P_s400 (= M400_k30_on) | 400 | 0 | 1,000 | local | SIMULATED (seed 0) |
| P_s1600 (= M1600_k30_on) | 1,600 | 0 | 3,000 | `x1g_pilot1` | SIMULATED (seed 0) |
| P_s6400 (= M6400_k30_on) | 6,400 | 0 | 1,000 | `x1g_pilot1` | SIMULATED (seed 0) |
| P_kp1600 (= M1600_kp_on, k = 0.2 N = 320) | 1,600 | 0 | 1,000 | local | SIMULATED (seed 0) |
| P_pc1600 (= PC1600_m20) | 1,600 | 0 | 3,000 | `x1g_pilot1` | SIMULATED (seed 0) |
| P_pc400x1f (= PC400_m5, X1f G0P replicated in core8) | 400 | 0 | 3,000 | `x1g_pilot1` | SIMULATED (seed 0) |
| PL1600_on_g16_th005 (= one ML1600_k30_on life) | 1,600 | 0 | 1,000 | local | SIMULATED (seed 0) |
| DPC1600, DPC1600_fw (diagnostic: fall window of the top → L1 rule) | 1,600 | 0 | 1,000 / 300 (stopped) | local | SIMULATED (seed 0), not pre-registered |
| M400_ref, M400_k30_on / _off, PC400_m5, M1600_kp_on / _off, M1600_k30_on_a05 / _a075 | 400 / 1,600 | 0–2 | – | `jobs/restD.txt` | **NOT RUN** |
| M6400_k30_on / _off | 6,400 | 0–2 | – | `jobs/main6400.txt` | **NOT RUN** |
| ML1600_k30_on / _off, ML6400_k30_on (birth return loop, th1 0.005) | 1,600 / 6,400 | 0–2 | – | `jobs/loopC.txt` | **NOT RUN** |

- `jobs/main400.txt`, `main1600a.txt` and `main1600b.txt` are the first job lists.
  - `main1600a` ran as `sB` + `pcA`.
  - `main400` and `main1600b` are contained in `restD`, which was not run.
- The local seed-0 command lines are in `jobs/local_seed0.txt`.

## 3. Main arms: the learned path (PROVED, 3 seeds; `tables/agg8_3seed.txt`, `tables/sets_3seed.txt`)

The probes are frozen. "Same-snapshot decoding" uses two noise repetitions of the muted probe at that checkpoint. "Context-varied" is X1e's phase-1 decoding, at birth and at the end.

| arm, checkpoint | seen first syl | fair spec (null 95 %) | hub share | same-snapshot decoding lag 0 / 6 / 12 | drive answer / best wrong | answer top-driven | L3 mean | P(L3 empty) | child utterances per dialogue |
|---|---|---|---|---|---|---|---|---|---|
| L33 on, 300 | 0.03 ± 0.04 | 0.011 (0.03) | 0.38 | 0.22 / 0 / 0 | 0.34 / 0.64 | 0.12 | 36 | 0.50 | 0.78 |
| L33 on, 1,000 | 0.08 ± 0.11 | 0.015 (0.03) | 0.37 | 0.23 / 0 / 0 | 0.39 / 0.67 | 0.14 | 25 | 0.51 | 0.96 |
| L33 on, 3,000 | 0.05 ± 0.06 | 0.011 (0.03) | 0.36 | 0.28 / 0 / 0 | 0.45 / 0.81 | 0.14 | 28 | 0.51 | 1.03 |
| L33 on, 10,000 | 0.09 ± 0.04 | 0.006 (0.03) | 0.37 | 0.27 / 0 / 0 | 0.47 / 0.80 | 0.14 | 28 | 0.51 | 1.08 |
| L33 off, 300 | 0.05 ± 0.04 | 0.008 (0.03) | 0.37 | 0.20 / 0 / 0 | 0.39 / 0.64 | 0.12 | 36 | 0.48 | 1.13 |
| L33 off, 1,000 | 0.05 ± 0.06 | −0.026 (0.03) | 0.37 | 0.15 / 0 / 0 | 0.45 / 0.78 | 0.09 | 24 | 0.51 | 1.27 |
| L33 off, 3,000 | 0.06 ± 0.06 | 0.026 (0.03) | 0.39 | 0.21 / 0 / 0 | 0.48 / 0.83 | 0.09 | 28 | 0.51 | 1.25 |
| L33 off, 10,000 | 0.06 ± 0.06 | −0.001 (0.02) | 0.40 | 0.27 / 0 / 0 | 0.48 / 0.82 | 0.09 | 28 | 0.51 | 1.27 |

- **Context-varied decoding at lags 0 / 6 / 12:**
  - birth: 0.14 / 0 / 0, P(L3 empty) at lag 12 = 0.99;
  - end, L33 on: 0.21 / 0.02 / 0.005, P(empty) 0.96;
  - end, L33 off: 0.25 / 0.02 / 0.005, P(empty) 0.97.
  - Chance is 1/32 = 0.031. Lag 0 is about frame level.
- **L33 learning:** at N = 1,600 it connects 97 % of all L3 pairs (0.973 ± 0.004), and it changes nothing measurable.
- **Held-out pairs:** first syllable 0.00–0.10.
- **Readout swaps at 10,000** (same weights): no readout gives more than 0.15 ± 0.12 (L33 on, th1 0.008). The absolute floor at a = 1 turns silence into garble (0.12–0.30).
- **Live world, frozen, 60 exchanges:**
  - response rate 0.34, accuracy 0.04;
  - talk-over ≤ 0.011 per exchange, failure to yield 0, garble ≤ 0.07.
- **While overhearing** the child speaks 0.8–1.3 times per dialogue, with garble ≤ 0.02. The pre-registered criteria do not cover this.

## 4. Positive control: readout with a hand-wired path (PROVED, 3 seeds; DIAGNOSTIC)

- Hand-wired held, question-specific paths (B) for all 22 seen pairs, in non-overlapping assemblies.
- Inner-hearing links (C) for the 11 base pairs only. "New" = the other 11, whose answers must be learned by the Hebbian readout.
- L33 is off.

| arm, checkpoint | new first syl | new silent | new fair spec (null 95 %) | new drive answer / best wrong | new answer top-driven | base first syl |
|---|---|---|---|---|---|---|
| PC1600_m20, 300 | 0.09 ± 0.07 | 0.76 | 0.30 (0.09) | 0.60 / 0.39 | 0.73 | 0.85 |
| PC1600_m20, 1,000 | 0.06 ± 0.04 | 0.76 | 0.33 (0.10) | 0.56 / 0.37 | 0.76 | 0.88 |
| PC1600_m20, 3,000 | 0.12 ± 0.04 | 0.73 | 0.33 (0.09) | 0.53 / 0.35 | 0.76 | 0.88 |
| PC1600_m20, 10,000 | 0.12 ± 0.04 | 0.73 | 0.33 (0.09) | 0.52 / 0.34 | 0.73 | 0.91 |
| PC6400_m30, 300 | 0.09 ± 0.00 | 0.64 | 0.26 (0.11) | 1.04 / 0.88 | 0.61 | 0.79 |
| PC6400_m30, 1,000 | 0.09 ± 0.07 | 0.76 | 0.22 (0.10) | 0.93 / 0.78 | 0.73 | 0.70 |
| PC6400_m30, 3,000 | 0.09 ± 0.00 | 0.70 | 0.22 (0.10) | 0.96 / 0.84 | 0.64 | 0.79 |
| PC6400_m30, 10,000 | 0.06 ± 0.04 | 0.73 | 0.20 (0.09) | 1.01 / 0.90 | 0.61 | 0.76 |

- **The path is held:** context-varied decoding at lag 12 is 0.89 ± 0.07 (PC1600) and 0.88 ± 0.07 (PC6400), at birth and at the end. P(L3 empty) = 0.
- **Readout swaps at 10,000, new first syllable** (same weights):

| readout | PC1600_m20 | PC6400_m30 |
|---|---|---|
| a 0.5 + absolute floor | 0.00 | 0.03 |
| a 0.5 + relative | **0.33 ± 0.11** | 0.15 ± 0.11 |
| a 0.75 + relative | 0.18 ± 0.07 | 0.15 ± 0.15 |
| a 1 + absolute floor | 0.12 (garble 0.12) | 0.09 (garble 0.55) |
| a 1 + relative (pre-registered) | 0.12 ± 0.04 | 0.06 ± 0.04 |
| a 1 + relative, th1 0.008 / 0.03 | 0.09 / 0.00 | 0.12 / 0.06 |
| post-hoc ρ 0.8: a 1 / a 0.5; ρ 0.9: a 0.5 | 0.09 / 0.12 / 0.12 | 0.06 / 0.15 / 0.12 |

- **Live world** (format caregiver / realistic caregiver):

| arm | base pairs: accuracy (response rate) | new pairs: accuracy (response rate) |
|---|---|---|
| PC1600 | 0.65 (0.83) / 0.79 (0.92) | 0.03 (0.21) / 0.06 (0.33) |
| PC6400 | 0.35 (0.85) / 0.38 (0.91) | 0.10 (0.33) / 0.13 (0.37) |

  - Talk-over ≤ 0.006 per exchange, failure to yield 0, garble ≤ 0.033.
  - At N = 6,400 the hand-wired base answers degrade during overhearing (live accuracy 0.35–0.38). Even at birth, the forced whole answer is said only 0.92 ± 0.11 of the time. ASSUMED cause: the generic L3 part's learned drive (best wrong 0.9) interferes.
- **X1f form replicated** (P_pc400x1f, SIMULATED, seed 0; 5-unit assemblies, N = 400, X1f thresholds and readout):
  - new first syllable 0 at every checkpoint;
  - new-pair fair specificity 0.000;
  - drive into the answer 0.12–0.13 vs best wrong 0.10–0.12 (X1f: 0.12–0.19 vs 0.12–0.23).

## 5. Seed-0 pilots and diagnostics (SIMULATED, seed 0; `tables/agg8_seed0.txt`, `tables/sets_seed0.txt`)

### 5.1 Code (calib.py, N = 400, after 100 overheard dialogues; console output not kept, numbers from the agent's first hand-back)

| setting | L3 mean / p90 | P(L3 empty) | Jaccard between syllable codes, birth → after learning |
|---|---|---|---|
| X1f thresholds, L33 on | 97 / 120 | 0.005 | 0.23 → **0.67** (one generic state) |
| X1f thresholds, L33 off | 77 / 100 | 0.006 | 0.23 → **0.63** (the inner-hearing loop alone does it) |
| homeostasis k = 30, L33 on | 30 / 79 | **0.44** | 0.05 → **0.04** (codes stay distinct) |

- Raising fixed thresholds instead of using homeostasis gives very uneven codes. At N = 1,600, f2 0.22 / f3 0.12, syllable codes have 0 / 12 / 51 units (min / median / max).

### 5.2 Assemblies for the positive control (learning off, N = 1,600)

| assemblies | base-pair accuracy | path decoding at lag 12 |
|---|---|---|
| m = 20, non-overlapping | 1.00 | 0.94 |
| m = 30, overlapping, pool 1,200 | 0.36 | 0.22 |
| m = 5, overlapping, pool 800 | 0.64 | 0.50 |
| m = 10, overlapping, pool 800 | 0.18 | – |

So the positive control uses non-overlapping assemblies (a pilot decision, written in `PLAN.md` before the batches).

### 5.3 Other sizes and k (main-arm form, L33 on)

| life | seen first syl | fair (null 95 %) | decoding lags 3–12 | P(L3 empty) | L33 pairs > 0 | drive answer / best wrong |
|---|---|---|---|---|---|---|
| N 400, k 30, 1,000 dialogues | 0.00 | 0.004 (0.02) | 0 | 0.54 | 1.00 | 0.63 / 0.88 |
| N 1,600, k 30, 3,000 | 0.04 | 0.023 (0.02) | 0 | 0.53 | 0.97 | 0.52 / 0.73 |
| N 6,400, k 30, 1,000 | 0.04 | 0.023 (0.03) | 0 | 0.57 | **0.27** | 0.53 / 0.54 |
| N 1,600, k 320 (proportional), 1,000 | 0.00 | −0.018 (0.02) | 0 | 0.47 | 1.00 | 2.59 / 3.37 |

- At 6,400, context-varied decoding is 0.19 / 0.03 / 0.00 at lags 0 / 6 / 12. Live accuracy is 0.13 / 0.20 (response 0.25 / 0.37); with fair specificity ≈ 0, these are frame-level guesses (ASSUMED).
- At 6,400, L33 is no longer saturated, but it built no holding assemblies by 1,000 dialogues.
- A 10,000-dialogue life takes about 3 min at N 400, 6–7 min at 1,600, and 27–30 min at 6,400 (A800 CPU, one process).

### 5.4 Birth return loop (coordinator's request; `birth31=recip`, no self-hold, no fatigue)

- **At birth** (`dbg_loop.py`, console output not kept):
  - With th1 0.015 the returned inner voice never passes the release. Its share per line is 0.008–0.017, 0–2 lines come on, and L3 dies 3 ticks after the input stops.
  - With th1 0.005 the loop reverberates (L3 → inner voice → L2 → L3):
    - N = 400: 12–24 L3 units held through 8 silent ticks;
    - N = 1,600: only intermittent (155, 128, 1, 0, 0, 87, 0, …).
  - g_loop 16 and 64 behave identically at birth (scale-free weights).
- **Learning, th1 0.015** (4 lives on the A800: g_loop 4 / 16 / 64 with L33 on, and g_loop 16 with L33 off; 1,000 dialogues):
  - seen first syllable 0–0.09; fair −0.023 to 0.032 (null 0.02–0.04);
  - context-varied decoding at lag 12 = 0, P(empty) at lag 12 = 0.99–1.00;
  - forced whole answer said at birth 0.59 (1.00 without the loop).
- **Learning, th1 0.005** (local, g_loop 16, L33 on):
  - context-varied decoding 0.20 / 0 / 0, in 3-tick windows 0.34 / 0.02 / 0 at birth and 0.25 / 0 / 0 at the end;
  - P(L3 empty) 0.49–0.52 over the life;
  - the child echoes 0.95–1.06 times per dialogue;
  - forced whole answer said 0.41.

### 5.5 Which wrong syllable competes (DPC1600 vs DPC1600_fw; not pre-registered)

Same seed, same positive-control setup as PC1600_m20. New pairs at 300 dialogues:

| top → L1 rule | best wrong = answer's 2nd syllable | answer top-driven | drive answer / best wrong | new first syl | new silent |
|---|---|---|---|---|---|
| as in ARCH (fall window: A that fell ≤ 2 ticks ago still strengthens A → B) | 1.00 | 0.82 | 0.61 / 0.37 | 0.45 | 0.36 |
| fall window off for top → L1 only (`fell_win31 = −1`) | 0.00 | 1.00 | 0.77 / 0.28 | 0.18 | 0.82 |

- The fall window explains *which* syllable competes: the answer's second syllable.
- Removing the fall window did **not** make the new answers come out. This is 11 pairs at one checkpoint; the same seed without the change fell from 0.45 at 300 dialogues to 0.09 at 1,000. The `fell_win31` run stopped after the 300-dialogue checkpoint (the agent was told to hand back).
- Why a clean, top-driven prediction is still mostly silent is **not resolved**.

## 6. Diagnosis

1. **Code (fixed).** Sparse homeostatic codes keep syllables distinct (Jaccard 0.04–0.05 vs 0.63–0.67 for the dense X1f regime). SIMULATED, seed 0.
2. **Path (the main failure).** PROVED at N = 1,600; SIMULATED at 400, 6,400 and with proportional k.
   - With k = 30, L3 is empty in about half of all ticks over the life. In the pause after a question it is empty at lag 12 in 96–97 % of probes, and decoding beyond lag 0 is 0.
   - L33 learning on or off makes no difference.
   - Dense codes (X1f thresholds, proportional k) instead fall into one generic state, as in X1e/X1f.
   - ASSUMED: nothing in this PFC holds the question through a silence. At fixed k, the birth lateral drive per unit (about g33 · k / N) shrinks as N grows. In silence there are no input rises, so the onset rule has nothing to link.
3. **Specific prediction (works only with a given path).** PROVED.
   - With hand-wired 20–30-unit assemblies, new-pair fair specificity is 0.20–0.33 (null 95 % 0.09–0.11). The answer is the top-driven syllable for 0.61–0.76 of the new pairs.
   - With X1f's 5-unit assemblies it is 0 (seed 0).
   - ASSUMED from these numbers: X1f's "a small specific state cannot push" was mainly assembly size. The out-dominant split (a = 1) did not help; a = 0.5 with relative release gave the best swap.
4. **Release (blocks the positive control).** PROVED.
   - The best wrong syllable gets 0.65 (N 1,600) to 0.9 (N 6,400) of the answer's drive, so it passes the 0.6 max-relative cut. A2 then silences the mixture: new pairs are silent 0.73 of the time.
   - Seed 0 (§5.5): the competitor is the answer's 2nd syllable, linked by the rule's fall window. Removing it did not release the answers.
5. **Loop (seed 0).** The birth return loop holds the pause only with a loose release (th1 0.005), but answers need a strict one. Even loose, at N = 1,600 the hold is gone by 300 dialogues, and the child echoes about once per dialogue.

## 7. Planned but not run

- `jobs/main6400.txt`: M6400_k30_on / _off, 3 seeds. Only the seed-0 pilot exists (§5.3).
- `jobs/loopC.txt`: ML1600_k30_on / _off and ML6400_k30_on (loop, th1 0.005), 3 seeds. Only one seed-0 life exists (§5.4).
- `jobs/restD.txt`: all N = 400 arms (M400_ref = X1f G0 in core8, M400_k30_on / _off, PC400_m5), proportional k (M1600_kp_on / _off), and a = 0.5 / 0.75 trained (M1600_k30_on_a05 / _a075). Seed-0 pilots exist for M400_k30_on, PC400_m5 and M1600_kp_on only.
- The plan's "5 seeds for the N = 400 arms if the queue allows" did not happen.

## 8. 此测试模型与 ARCHITECTURE.md 的出入（审计，11-02，`AUDIT.md`）

设计者 11-02 问："会不会是其他智能体测试时没有严格做好架构导致出了问题。"逐条对照（`AUDIT.md`，复查脚本 `audit/c1.py`、`audit/c2.py`）后，结论是**没有严格照架构搭**，所以本报告的结论只说明"这个测试模型"做不到什么，不能算作对架构本身的否定。

- **最大的出入：想法保持的回路转不起来。** 没有听觉区深层，输入层常亮的一半又没有链式神经元，所以安静时前额叶第一层只有 0–2 个亮；中间层要约 7 个才亮，3 拍后全灭（出生时和学 1 万段后实测都是这样）。违背"安静时前额叶照样有输入"（设计者 10-30）和"各层输入多少、输出多少"（§二.4）。上面第 6 节的"停顿里保不住问题"就是这个。
- 前额叶中间层和最后一层用了感觉区的门槛形式（§二.1 只给感觉区），再加按长期活跃度调门槛和每小片减平均。
- 回程不是一对一送回原神经元：内心的声音没有回到听觉区，直接接到了肌肉映射上。
- 两层权重只按出边摊薄；最后一层内部连接降权最快、也不按凹形变慢。
- 手动接的说话连接不降权；"3–5 块肌肉才放行"是硬门槛（§二.7 说没有硬门槛）。
- 缺：海马体（关掉）、"很少一起亮的互相抑制"、听觉区深层、概念区。
- 只放大了前额叶两层，其他区不变（违背 §五"整体一起放大"；设计者 11-02：前额叶应和全脑成正比）。

下一步是严格照 ARCHITECTURE.md 搭、全脑用一个倍数一起放大的测试模型（X10），跑之前先逐条对照检查。

## 9. Files (all in `v3/x1g/`)

- **Plan:** `PLAN.md` (pre-registered arms and criteria, plus the addendum written before the loop batches).
- **Audit:** `AUDIT.md` (X1 test models vs ARCHITECTURE.md, 11-02) and its quick checks `audit/c1.py`, `audit/c2.py` (usage: `python3 -I audit/c1.py 400 2`).
- **Code:**
  - `core8.py`: X1f core6 + the options above.
  - `wire8.py`: X1f's `wire.py` + overlapping assemblies.
  - `run8.py`: the driver. Usage: `python3 -I run8.py <name> <seed> key=value ...`; it writes `out8/<name>_s<seed>.json`.
  - `eqtest8.py`: core8 vs core6 equivalence. `calib.py`: sparsity calibration. `dbg_loop.py`: return-loop check at birth.
  - `agg8.py`, `agg8_sets.py`: tables. Usage: `python3 -I agg8.py <out8 folder> [...]`.
- **Jobs:** `jobs/*.txt` (one life per line: `<name> <seed> key=value ...`) and `jobs/job.sh`, which runs `jobs.txt` in the current folder, 4 lives at a time.
- **Tables:**
  - `tables/agg8_3seed.txt`, `tables/sets_3seed.txt`: 3 seeds;
  - `tables/agg8_seed0.txt`, `tables/sets_seed0.txt`: seed-0 pilots and diagnostics.
- **Not copied:** per-life JSON and logs, `pilot0.py`, the submit script, and the copies of X1c/X1d/X1f code, which are imported from their own folders.
- **Compute:**
  - A800 CPU jobs `x1g_pilot1`, `x1g_pcA` (34 min) and `x1g_sB` (14 min), 4 lives at a time.
  - Local: one process at a time, at most about 95 MB.
