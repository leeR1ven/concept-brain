(Saved by the coordinator from the agent's hand-back; partial — batch 2 was still running. In the repo, `out1/` is `results1/` and `out2/` is `results2/`; `core3.py`, `world.py` and `run.py` are imported from `../x1c/`.)

# X1d report: can the moving thought follow the same path for the same question, and does question → answer then form?

Labels: **PROVED** = counters or exact numbers from runs (files named); **SIMULATED** = a pilot or a single seed; **ASSUMED** = my judgement, not tested.

**I was told to hand back early, so this is a partial report.**
- **Complete:** phase 1, and phase 2 for 4 arms × 5 seeds.
- **Still running in the background, not reported:** batch 2, with the shaping arms (RNpm_shape, R1e_shape), the growth/decay-only arm (RNg) and the oracle diagnostics (RNpm_oracle, R1e_oracle). It writes to `x1d/out2/`, log `batch_2.log`. One early number from RNpm_shape seed 0 at teach 600: pleased 0.003 per exchange, responses 0.09, answer-syllable hits 0.
- **Not done:** the phase-1 confirmation seeds and the ×4 scale arm (queued as lists `conf` and `x4` in `sweep1.py`).

## 0. Bottom line

1. **Phase 1 (birth, no learning): no setting reaches the target** (PROVED for seed 0; 63 regimes, `out1/`).
   - The best realistic regime has same-question Jaccard only about 0.02–0.05 above cross-question at lag 12.
   - Question decoding at lag 12 is ≤ 0.08 (chance 0.031), and frame decoding ≤ 0.46 (chance 0.25).
   - Every regime falls into one of four failure modes:
     - **chaotic:** damage spreads and same-question similarity drops to the cross level within 3–6 ticks;
     - **context-locked:** the path follows the pre-question state, not the question;
     - **generic attractor:** all questions go to one shared state;
     - **dead or saturated.**
   - X1c's chosen g = 8 is chaotic, as the coordinator suspected: turnover about 0.8 per tick, and damage goes 0.81 → 0.19.
2. **Phase 2 (learning, corrected onset rule, 4 arms × 5 seeds): accuracy is 0.00 everywhere** (PROVED, `out2/AGG2.json`):
   - seen pairs, held-out pairs and first syllable: all 0;
   - the correction phase: 0 new or old answers.
3. **Where the chain breaks** (PROVED unless marked):
   - **Link 1, reproducible path: fails.** After teaching, same ≈ cross at lag 12 (R1e 0.30 vs 0.29; RNpm 0.39 vs 0.38), and question decoding at lag 12 is 0.03–0.04, i.e. chance.
   - **Link 2, specific prediction of the answer's inner hearing: fails.** The answer's first-syllable code is predicted equally after its own question and after others (own 0.05–0.13 vs other 0.04–0.11). Predictions are dominated by one hub syllable (hub share 0.67–0.85).
   - **Link 3, motor release: mostly works.** In the hand-wired diagnostic, the answer's first-syllable inner hearing is forced for 3 ticks. The mouth then produces that syllable within 3 ticks in 84–87 % of cases in R1e and 51–65 % in RNpm.
   - But the response is masked by ongoing garble, so first-syllable accuracy stays at about 0–0.13. The best single run (RNpm seed 0) reached full-answer accuracy 0.14 under forcing.
4. **Activity control without PFC inhibition** (the designer's main arm):
   - **At birth:** strictly bistable, either dead in silence or saturated at 400/400. The switch happens within Δf1 = 0.05 (f1 0.70 saturates, 0.75 dies).
   - **With learning:** growth/decay alone did not stop the runaway (RNe, mass-preserving weights; and the scale-free form at g31 = 1 also saturated). The patch-mean fallback (RNpm, X3's gain control) stays bounded over whole lives: mean L3 48–70, bursts to 350–360 of 400, saturation 0.000–0.001.
   - The growth/decay-only arm RNg with γ = 0.99 stayed bounded in its pilot but died in silence. Its 5-seed run is still pending.

## 1. What I built (all in `x1d/`; X1c copied, not edited)

- **`core4.py` = core3.Brain unchanged, plus options (all off by default).** PROVED that the defaults reproduce core3 exactly: L3 states and L31 identical over 3,000 ticks, for both the global and local arms.
  - `rule='onset'`: the coordinator's held-unit rule.
    - A→B grows when B rises while A is on (new, held, or fell within 2 ticks).
    - A→B weakens only when A rises after B was already on.
    - The fall clause is dropped.
  - `post='ext'`: the post side is the next chain **input** (heard sound including own voice, actual muscles, stop neurons), not imagery. This follows "最后一层…和下一刻第一层的**输入**做赫布连接". Without it, the top layer learns to predict its own predictions, and a hub syllable takes over (pilot: hub share 0.91, path same 0.75 vs cross 0.70). This is my reading of the spec [ASSUMED correct].
  - `birth31='recip'`: the birth loop is wired back to the L1 units that drive each top unit (msg 84, "G returns to A"), with the same mask, gain and fan.
  - `nrm31='arch'`: the two-layer weights for top→L1 in ARCH form, g31·w/√(out·in), with no mass rescale.
  - `dgain`: the coordinator's slow bounded divisive gain.
  - `loc_same`: same-tick local inhibition.
  - `pm`: X3 patch-mean subtraction of the prediction and lateral drive.
  - Hand-wired DIAGNOSTIC hooks, never used in learning: `_force` (forced inner hearing) and `_oracle` (an oracle path).
  - Babbling (`babble`) is implemented but **not used**. The designer now allows hand-wired speech basics, so X1c's stand-in auditory→motor link is kept.
- **`phase1.py`:**
  - A birth period first: 200 world exchanges with every Hebbian and value rate at 0, so only homeostasis settles. Without it, every input gives the same state (same 0.94 vs cross 0.90).
  - Then, frozen, for 32 questions × 4 world-like preceding contexts, it measures:
    - same-question and cross-question Jaccard by lag;
    - decoding of question, frame and filler-given-frame (also over 3-tick windows);
    - a noise-only repeat (same context, fresh render noise);
    - damage spreading (10 % and 25 % of active L3 units moved, identical input);
    - motion over a long silence.
- **`run2.py`:**
  - X1c's protocol: birth 200, teach 600, correction 300, untutored 1,000. **The n_free mark bug is fixed** (marks are now unique and increasing).
  - It also includes the `ShapingWorld` (graded praise, standards rising with age), prediction specificity, release statistics, forced-release and oracle diagnostics, and phase-1 path metrics on the trained brain.
- **Other files:**
  - `regimes.py` (arms), `sweep1.py` (phase-1 lists), `batch.py`, `agg2.py`, `tab1.py`;
  - `fftest.py`: feed-forward similarity test.

## 2. Phase 1 results (seed 0, 4 contexts, birth 200; PROVED from `out1/*.json`, summaries in `out1/SWEEP_v1..v5_s0.txt`)

### 2.1 Feed-forward stack (`fftest.py`, PROVED)

| layer | same syllable, fresh noise | different syllables |
|---|---|---|
| L1 | 0.84 | 0.23 |
| L2 | 0.63 | 0.15 |
| L3 | 0.58 | 0.16 |

- Fan-in (8–100) and k (15–100) change this by less than 0.05.
- So at question end there **is** a usable gap of about 0.4. The losses come in the pause.

### 2.2 Regimes (lag 12 = mean of lags 10–12; damage = Jaccard to the unperturbed run at lag 12)

| regime | control | same@0 | cross@0 | same@12 | cross@12 | gap@12 | decode Q@12 | decode frame@12 | damage 10 % | within-path J(t,t+1) | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| G_loop8 (= X1c B_loop, g 8) | global | 0.14 | 0.10 | 0.08 | 0.08 | +0.006 | 0.021 | 0.31 | 0.39 | 0.15 | chaotic; mouth on 0.90 |
| G_loop4 | global | 0.14 | 0.08 | 0.09 | 0.06 | +0.022 | 0.060 | 0.32 | 0.33 | 0.13 | chaotic |
| G_fan (fan 60/80) | global | 0.21 | 0.17 | 0.17 | 0.15 | +0.022 | 0.057 | 0.36 | 0.58 | 0.27 | |
| G_rec4_lat (reciprocal loop; = R1 at birth) | global | 0.11 | 0.07 | 0.08 | 0.06 | +0.024 | 0.076 | 0.32 | 0.55 | 0.17 | best realistic |
| G_lowK (fan 8/8/6/4) | global | 0.31 | 0.11 | 0.07 | 0.06 | +0.006 | 0.047 | 0.29 | 0.12 | 0.12 | |
| G_lowK, own voice muted | global | 0.33 | 0.16 | 0.18 | 0.12 | **+0.054** | **0.081** | **0.46** | 0.43 | 0.23 | best overall (not realistic) |
| G_loop4, muted | global | 0.41 | 0.34 | 0.62 | 0.60 | +0.021 | 0.054 | 0.35 | 0.80 | 0.59 | generic attractor |
| G_K16, muted | global | 0.22 | 0.17 | 0.23 | 0.21 | +0.018 | 0.039 | 0.35 | 0.67 | 0.45 | context-locked (noise-only 0.62) |
| G_loop8rel / G_k80 | global | 0.78 / 0.67 | 0.69 / 0.59 | 0.80 / 0.64 | 0.80 / 0.63 | ≈0 | 0.04 | 0.23–0.26 | 0.89 / 0.78 | 0.74 / 0.70 | generic frozen |
| G_K12, g_int 0.2 (stronger input reset) | global | **0.45** | 0.14 | 0.06 | 0.06 | +0.003 | 0.023 | 0.24 | 0.18 | 0.10 | input-locked at the end, chaotic in the pause |
| G_lat0.4 / G_lat1 (lateral only) | global | 0.43 / 0.28 | 0.15 / 0.11 | 0.06 | 0.06 | 0.000 | 0.04 | 0.26–0.29 | 0.18 / 0.06 | 0.08 | fully chaotic, turnover 0.94 |
| G_rec4_lat + divisive gain | global | 0.07 | 0.06 | 0.13 | 0.13 | 0.000 | 0.041 | 0.27 | 0.42 | 0.17 | the gain does not help |
| A_g4 (local, 1-tick lag) | local | 0.12 | 0.06 | 0.03 | 0.02 | +0.008 | 0.024 | 0.16 | 0.31 | 0.00 | bursts to 185; 40 % empty |
| A_lowK | local | 0.31 | 0.13 | 0.05 | 0.04 | +0.007 | 0.042 | 0.27 | 0.50 | 0.01 | bursts to 171 |
| A_same (same-tick inhibition) | local | 0.01 | 0.01 | 0 | 0 | 0 | 0 | 0 | 0 | — | dies: 96 % empty |
| N_C (= X1c C, no loop) | none | 0.33 | 0.10 | 0 | 0 | 0 | 0 | 0 | — | — | dead after lag 2 |
| N_f1.6 / 0.65 / 0.70 (loop) | none | 1 | 1 | 1 | 1 | 0 | 0.016 | — | 1 | 1 | saturated 400/400 |
| N_f1.75 (loop) | none | 0.31 | 0.10 | 0 | 0 | 0 | 0 | 0 | — | — | dead |
| N_own (= C2) | none | — | — | — | — | — | — | — | — | — | dead 100 % |
| N_pm1 (patch-mean) | none + patch-mean | 0.32 | 0.09 | 0.01 | 0.00 | +0.010 | 0.018 | 0.02 | — | 0.04 | bounded (max 100), but 84 % empty |

What the regimes show:
- **The own voice matters.** At birth the child babbles in 70–90 % of pause ticks, because the random loop predicts muscle copies and the muscle→blocker latch holds.
  - Muting raised the gap (G_lowK 0.006 → 0.054).
  - Switching babble off with motor knobs did not (G_lowK_m2, gap 0.012).
  - So the gain comes from the latched muscle copies acting as a sustained, question-dependent self-input [SIMULATED interpretation].
- **The trade-off is clear** (PROVED by the g_int and fan series): input-locking at question end comes with chaos in the pause, and order in the pause comes with context-locking or a generic attractor.

### 2.3 Full curves

Lags 0, 1, 2, 3, 4, 6, 8, 10, 12 (same → cross; damage 10 %):
- **G_loop8 (X1c B_loop, birth 300, 6 contexts, `out1/Bloop8_s0.json`):**
  - same: 0.16 .15 .17 .15 .11 .11 .10 .10 .11;
  - cross: 0.10 .09 .09 .11 .08 .09 .07 .08 .09;
  - damage: 0.81 .88 .91 .64 .56 .45 .34 .30 .28 (diverging);
  - J(t,t+1): about 0.15.
- **G_lowK_mute (best overall):**
  - same: 0.33 → 0.18 at lag 12; cross: 0.16 → 0.12;
  - damage: 0.81 → 0.43;
  - frame decoding stays at 0.39–0.46.
- The remaining per-lag curves are in each `out1/<name>_s0.json` (keys same, cross, dec_q, dec_q_win3, damage, within_J1, n3_lag). `python3 -I tab1.py curves <names>` prints them as markdown.

## 3. Phase 2 results (5 seeds, mean ± SD; PROVED, `out2/AGG2.json`)

### 3.1 Arms

- **R1e:** the best realistic phase-1 regime. Global threshold (shortcut reference), reciprocal birth loop, lateral 0.4, κ = 0, onset rule, post = input.
- **RNpm:** no PFC inhibitory neurons. Thresholds are a fraction of each neuron's own in-weight (X1c C values), there is no birth loop, ARCH-form two-layer weights, and the X3 patch-mean on the prediction and lateral drive.
- Each arm was run with the format caregiver and the realistic one ("_real").

### 3.2 Results table

| metric (teach 600 unless marked) | R1e | R1e_real | RNpm | RNpm_real |
|---|---|---|---|---|
| seen / held-out / first-syllable accuracy | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| untutored 1,000: seen / first | 0 / 0 | 0 / 0.03±0.05 | 0 / 0 | 0 / 0 |
| correction: new / old | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| probes: silent / garble / talk during Q | 0 / 1.00 / 1.00 | 0.01 / 0.99 / 1.00 | 0.37 / 0.63 / 0.61 | 0.47 / 0.53 / 0.40 |
| garble per exchange | 2.95 | 2.95 | 2.17±0.92 | 1.62±0.89 |
| talk-over per exchange | 0.036 | 0.023 | 0.021 | 0.022 |
| failure to yield | 0.000 | 0.000 | 0.000 | 0.000 |
| speech routes: heard / inner / PFC motor / recall | .01 / .26 / .72 / 0 | .01 / .26 / .72 / 0 | .01 / .44 / .55 / 0 | .02 / .44 / .54 / 0 |
| answer code predicted: own vs other questions (max over lags) | 0.07 / 0.07 | 0.05 / 0.04 | 0.13 / 0.11 | 0.12 / 0.11 |
| answer is the top predicted syllable at lag 12 | 0.04 | 0.03 | 0.01 | 0.02 |
| hub share of predictions (lags 10–13) | 0.80 | 0.67 | 0.79 | 0.85 |
| path after teaching: same@0/6/12 | .25/.28/.30 | .52/.52/.52 | .52/.39/.39 | .44/.32/.32 |
| path after teaching: cross@0/6/12 | .23/.26/.29 | .50/.51/.51 | .38/.37/.38 | .31/.29/.30 |
| path: decode Q @0/6/12 (chance .031) | .05/.05/.04 | .07/.05/.04 | **.16**/.05/.03 | **.17**/.04/.03 |
| path: damage 10 % @12; within J(t,t+1) | 0.81; 0.87 (near-stuck) | 0.87; 0.95 (stuck) | 0.47; 0.45 (moving) | 0.43; 0.30 (moving) |
| FORCED inner hearing at lag 12: said within 3 ticks | 0.87 | 0.84 | 0.51±0.28 | 0.62±0.38 |
| FORCED: first-syllable / full accuracy | 0.00 / 0 | 0.01 / 0 | 0.05 / 0.03 | 0.05 / 0 |
| L3 count: mean / max / P(empty) / P(saturated) | 34 / 39 / 0 / 0 | 34 / 39 / 0 / 0 | 70 / 350 / 0.32 / 0.001 | 48 / 361 / 0.40 / 0 |
| pleased per exchange (exact-only world) | ≤ 0.005 | 0 | ≤ 0.002 | ≤ 0.002 |

Notes on the table:
- **Garble over time** (R1e, quality_over_time): 0.94 → 0.89 per exchange. Clear-syllable responses rose 0.33 → 0.58, and answer-syllable hits stayed at 0.02–0.04.
- **RNpm varies a lot across seeds.** The best was RNpm_real seed 4: path same 0.13 vs cross 0.03 at lag 12 while turning over 60–90 % per tick. That is moving and partly reproducible, but at low absolute overlap [SIMULATED, one seed].
- **Held-out accuracy** is expected to need the abstract concept area (ARCH §四 item 14), which was not built. Seen-pair accuracy is the main criterion here.

## 4. Diagnosis

1. **The first missing link is the reproducible path, and the existing knobs cannot fix it** (PROVED over 63 birth regimes; the theory is ASSUMED).
   - With fan-in of 4–100, a random binary k-WTA or threshold net is expansive. Each flipped unit flips several downstream units, so it is chaotic.
   - Making it ordered (lower gain, a stronger generic bias, a fixed-threshold form) gives one of three things: a generic attractor, context-locking, or dying/saturating with no inhibition.
   - In silence, chains send almost nothing, so no input can re-lock the path.
   - The coordinator's divisive gain and the reciprocal loop (msg 84) did not change this.
2. **Learning did not create the path.** With the corrected rule plus post = input, the top→L1 links predict a hub syllable after every question, because hub links settle 9–12× higher than specific ones (§5).
   - The inner hearing of the hub then pulls every pause into the same state (generic, as in R1e), or the path stays chaotic (RNpm).
   - Q→A specificity (own vs other) is ≤ 0.02 in every arm.
3. **The motor side releases a correct inner hearing** (forced diagnostic, 51–87 %). In R1e, however, the child babbles continuously (talk during the question 1.00, about 3 garbled utterances per exchange). PFC motor intentions drive 72 % of speech onsets. So even a correct prediction would come out inside garble.
   - This is a second problem, behind link 1.
   - The no-inhibition arm (RNpm) is quieter and produced 3–14 % full answers when forced, in its best seeds.
4. **Reward never shaped anything in the exact-only world.** The child was pleased at most 0.005 times per exchange. The shaping-arm data are still running (batch 2), so the comparison is not reported.

## 5. Activity control and the growth/decay equilibrium (coordinator item 2)

The equilibrium is L* = [−w_ref + √(w_ref² + 4fηw_ref/(1−γ))]/2, with f = confirmations per 100 ticks.
- **X1c's values (η 0.02, w_ref 0.5, γ 0.998):**
  - a specific Q→A pair (f ≈ 0.11): L* = 0.53;
  - f = 1: L* = 2.0;
  - a hub (f = 5): L* = 4.75, about 9× the specific pair.
- **RNg (γ 0.99):** 0.165 / 0.78 / 2.0, about 12×.
- Under the scale-free ARCH two-layer weights only this ratio matters, so growth/decay cannot set the absolute drive; it can only set how concentrated the weights are.
- **Runs:**
  - Mass-preserving weights saturated (RNe pilot: L3 400, 184 L1 units predicted). The ARCH form at g31 = 1 also saturated.
  - g31 = 0.3 with γ = 0.99 (RNg) stayed bounded in its pilot but died in silence (5 seeds pending).
  - The patch-mean fallback (RNpm) stayed bounded in all 10 lives.
- So the PFC count was not held by growth/decay alone in what I tested. The reader's argument, that a neuron's threshold never sees the layer total from speech to silence, matches the data [ASSUMED as the explanation].

## 6. Deviations and limits

- Phase 1 used 1 seed and 4 contexts per regime; the seeds 1–2 confirmation and the ×4 scale arm were not run.
- Phase-2 lives are 600 teaching exchanges, not 1,500.
- The update and distractor-recovery tests were not done.
- `post='ext'` is my reading of "输入".
- The reciprocal birth loop is a choice of birth wiring, and the patch-mean is X3's gain control (the coordinator's suggested fallback); both are labelled as such.
- All forced and oracle arms are hand-wired diagnostics. No answers are wired anywhere.
- One python process at a time, peak below 100 MB. No server jobs. The repo was not modified.

## 7. Files

All in this folder (`v3/x1d/`):
- **Code:**
  - core: `core4.py` (plus copied `core3.py`, `world.py`, `run.py`);
  - drivers: `phase1.py`, `sweep1.py`, `run2.py`, `regimes.py`, `batch.py`;
  - tools: `agg2.py`, `tab1.py`, `fftest.py`, `sm.py`.
- **Phase 1:** `out1/*.json` (63 regimes, full curves) and `out1/SWEEP_v1..v5_s0.txt`.
- **Phase 2:** `out2/*.json` (20 lives plus batch 2 in progress) and `out2/AGG2.json`.
- **Logs:** `sweep_v1..v5.log`, `batch_main.log`, `batch_2.log`.

## summary_zh

1. 第一步（出生时，不学习）：试了 63 种参数组合（回路强度、横向连接、前向连接、输入强度、思考范围大小、三种控制方式），**没有一种能让"同一个问题每次走同一条路"**。
2. 原因：随机连接的网络要么太乱（一点差别很快放大，同一个问题停顿 12 拍后和别的问题一样）；要么太稳（所有问题都掉进同一个状态，或者只记得问题之前在想什么）。问题刚说完时其实分得开（相似度约 0.57 比 0.16），是停顿期间丢掉的。
3. 前额叶不加抑制时：要么停顿时熄灭，要么全部亮满，中间没有稳定的档；参数差 0.05 就从一头跳到另一头。
4. 第二步（学习，用您更正后的规则：一直亮着的前额叶神经元也长连接）：4 组、每组 5 次一生，**答对率全是 0**。
5. 断在哪里：①路线不重复：学完后同一个问题和别的问题的停顿状态几乎一样；②于是预测也不针对问题：总是预测同一个"万能字"（占 67–85%）；③手动把正确答案"放进内心声音"时，嘴巴 51–87% 能说出来，所以说话这一环基本是通的，但孩子一直在咿呀乱说，正确的音被乱码盖住。
6. 不加抑制、靠权重增长和降权：在我试过的参数里没能单靠它控制住；加一个局部"减去周围平均"的增益控制（海马体实验里用过的那种）后，整个一生都没有亮满。
7. 奖励：只在完全答对时高兴的话，孩子几乎从来得不到奖励（每次对话不到 0.005 次）。"说得接近也夸"的那组还在跑，结果没来得及放进报告。
8. 结论：问题不在学习规则本身，而在出生时的随机网络撑不住一条可重复、又在走的思路。可能要更有结构的连接，或者先让思路自己形成可重复的路线，再学问答。这需要您来定方向。