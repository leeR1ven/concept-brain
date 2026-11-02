# X11, interim: overhearing a large dialogue corpus (curve 1, s = 1, seeds 0 and 1)

Plan, data conversion and scoring rules: `PLAN.md`. Count-model reference curve: `results/ngram_curve.json`. Job results: `results/curve1/j<n>/`. **MEASURED** means a run in `results/`.

## 结果（简明）

1. **数据：** 下载了 LCCC 中文对话库（682 万段），全部转成"音节 + 声调"，再用合成声音（4 个大人）念给模型听。保留 660 万段，2 亿个音节。数据本身不放进仓库。
2. **测试方法：** 模型一直旁听，学习不停。在 1 千、3 千、1 万、3 万、10 万段时，用固定的 200 段没听过的对话考它：听到前面的音时，前额叶能不能预先想到下一个音。同时用最简单的"数次数"方法（二元、三元）当参照。
3. **1 千段：还没变好。** 前额叶预测下一个音的准确率和刚出生一样，都在瞎猜水平（约 0.1%）。同样的 1 千段，数次数方法已经能猜对 14%。活动量稳定，没有失控。
4. **速度：** 每个模型每小时约听 1,600 段。听到 1 万段约 7 小时，听到 10 万段约 70 小时。
5. **可能的原因（还没验证）：** 前额叶里亮的主要是几拍前听到的音的"回声"，不是对下一个音的预测；最后一层亮得太多、各音共用，学到的东西被平均掉了；海马体在这个设置下从来没被内部点亮过。下一步先做一次冻结诊断（约 15 分钟）来分清是哪个原因，曲线本身不改。

## 1. Speed (learning on, two seed processes per GPU job)

- 20.4–20.8 ms per tick per process; GPU peak 2.5 GB.
- About 1,600 dialogues per hour per seed. One 74-minute job moves each seed about 1,750–1,800 dialogues.
- 10k dialogues take about 6 jobs, 30k about 17 and 100k about 56 (about 70 h of chained GPU time).

## 2. Held-out results at 1k dialogues (MEASURED, seed 0 / seed 1)

The held-out set has 200 dialogues, 5,008 scored occurrences and 1,262 candidate toned syllables. A uniform guess scores 0.0008 top-1 and 0.004 top-5.

| | newborn | 1k dialogues |
|---|---|---|
| PFC x_int t0+1..3, top-1 | 0.001 / 0.000 | 0.002 / 0.001 |
| PFC x_int t0+1..3, top-5 | 0.008 / 0.006 | 0.007 / 0.007 |
| PFC toneless, top-5 | 0.028 / 0.020 | 0.021 / 0.015 |
| late window t0+2..4, top-1 / top-5 | | 0.001 / 0.004; 0.000 / 0.002 |
| per-tick J(x_int, copies that rose) | 0.016 / 0.013 | 0.021 / 0.019 |
| internally lit sound copies per 3-tick window | 44 / 51 | 78 / 71 |
| hippocampus h_int | empty in every window | empty in every window (H1int = 0 on every tick) |
| reply's first syllable (n = 375), PFC top-1 / top-5 | | 0.000 / 0.005–0.008 |

- **Readout ceiling:** ranking the heard rises themselves gives top-1 0.34–0.35 and top-5 0.74.
- A good predicted distribution also survives this readout: the interpolated count model, ranked through the codes, scores top-1 0.095.

Count models trained on the same dialogues (direct top-1 / top-5):

| model | 1k | top-1 at 100k |
|---|---|---|
| unigram | 0.031 / 0.144 | |
| bigram, lag 1 | 0.137 / 0.277 | 0.160 |
| trigram, lag 1 | 0.140 / 0.281 | 0.222 |
| bigram, lag 2 | 0.041 / 0.126 | 0.057 |
| trigram, lag 2 | 0.037 / 0.118 | 0.076 |
| reply-initial unigram | 0.093 / 0.275 | |
| across-turn bigram | 0.013 / 0.115 | |

**Matched reference.** By the pipeline latency (PLAN §D), x_int on t0+1..3 sees mostly the syllable before last. So the matched reference is the lag-2 rows, 0.04–0.08.

## 3. Health (MEASURED)

All 175 training windows per seed pass:
- layer ratios are in [0.5, 2];
- Hs stays within 2 × kh;
- no layer goes over 0.41 on.

Over the first 1k dialogues:
- L4 rose from 656 to about 900 (L4/L3 1.15–1.35).
- Hs is 70–79 (kh 50).
- L1int is 77–104 per tick, against an L1ext of about 690.

## 4. Candidate causes (not yet tested)

1. **Echo.** x_int is mostly the birth loop's echo of what was heard 4–5 ticks earlier. The newborn already lights 44 copies per window, all echo, and learning adds about 30. Ranking x_int against the syllable codes then points at the recent past.
2. **Shared L4.** L41 learns from an L4 that is about 23 % on and mostly shared across syllables, so the sequences average away. This is the X1e / X10 (b) problem (ARCH §四.20).
3. **Latency.** Even a perfect predictor in this window sees only syllables 1.5 or more back. So the matched ceiling is the lag-2 reference.
4. **Hippocampus.** With the learned gain g_h1 = 0.1 (R2, X10 stage a), the learned Hs→H1 drive never reaches θ_h1. The hippocampus therefore never recalls into H1.

**Next.** A frozen diagnostic (learning off, about 15 min) runs between jobs 2 and 3, on the newborn and on the 1k checkpoint. It measures:
- x_int against the current and earlier syllables (the echo test);
- the learned L41 drive alone, both at threshold and as a graded ranking;
- the distribution of H1's internal drive against θ_h1.

Curve 1 stays unchanged. Any lever goes into a new curve only after it has been checked against ARCHITECTURE.md.

## 5. Job 2 (3k dialogues) and the frozen diagnostic (11-02)

**3k held-out, seed 0 / seed 1 (MEASURED):**
- PFC top-1 0.003 / 0.001; top-5 0.009 / 0.007. Still at chance (0.0008 / 0.004).
- Lag-1 trigram 0.165 / 0.312; lag-2 bigram 0.047 / 0.142.
- H1int is still 0 on every tick.
- L4 drifts to 1.13–1.15 × L3.

Curve 1 is stopped. The designer (11-02) asked for the model to be fixed first: PFC counts equal at every layer, and a working hippocampus.

**Diagnostic** (`results/diag1/`, learning off, on copies of the newborn, 1k and 3.4k states):

| | 1k | 3.4k |
|---|---|---|
| L4→L1 synapses above 0 | 98 % (861,648 / 882,008) | 98 % |
| Hs→H1 synapses above 0 | 96 % | 96 % |
| Hs↔Hs synapses above 0 | 81 % | |
| learned L41 drive alone, per-tick max (median / p90) | 0.42 / 0.63 θ1 | 0.45 / 0.63 θ1 |
| H1 internal drive, per-tick max (median / p90) | 0.006 / 0.010 θ_h1 | 0.007 / 0.013 θ_h1 |
| learned L41, graded top-5, next syllable / previous syllable | 0.014 / 0.023 | 0.013 / 0.023 |

- **Every internally lit PFC copy is birth-loop echo.** The learned links alone light nothing.
- **Cause: too many units on at once.** About 250 of 3,900 copies rise per tick, while about 35 % of L4 and 3 % of Hs are eligible. So the one learning rule strengthens almost every possible pair. The two-layer normalisation then turns the near-uniform weights into a tiny, flat drive.
- **Arithmetic (coordinator, ASSUMED until the sweep).** The density is about 0.4 base hits per Hs→H1 synapse per 200 ticks. At that density no gain or decay setting lets a one-shot pair stand out. That needs about 15 copy rises per tick, not 250. This is the designer's own picture (11-02: about 10 inputs per frame).
- **Next.** Fast parameter sweeps in X10:
  - gains only, as the check of the arithmetic;
  - then a sparser input per frame (sound rates, or only the deep layers sent through the chains) × hippocampal decay and gain.

**Corrected layer specificity** (`diag_l4.py`; L1–L4 grouped by latency; same-syllable minus different-syllable J; MEASURED, same at newborn and 3.4k):

| L1 copies | L2 | L3 | L4 |
|---|---|---|---|
| +0.15 | +0.08 | +0.06 | +0.047 |

- L4 is 15–25 % on, and about 90 % of its code is shared across syllables. The fixed random middle layers wash the difference out, and learning does not change it.
- Sound-area decoding (toneless, nearest centroid): A0 0.92, A1 0.92, A2 0.88, A3 0.73, A4 0.42, PFC L1 copies 0.75. The deep layers lose information in this voice.
