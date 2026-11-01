# X1d addendum: batch 2

All numbers are PROVED: 5 seeds per arm, mean ± SD, at teach 600 unless marked. Sources are `results2/*.json` and `agg2_final.txt`. No new arms were started.

## 1. Graded praise vs exact-only caregiver

**Graded praise does not help.** The caregiver now gets pleased often (0.04–0.08 per exchange, against ≤ 0.005 for the exact-only caregiver). But garble does not fall and answer-syllable hits do not rise, and accuracy stays 0.

Pleased reactions per exchange over time (untutored phases have no caregiver reactions, so 0 by design):

| arm | teach100 | teach300 | teach600 | correction | free300 / free1000 |
|---|---|---|---|---|---|
| R1e, exact-only | 0.000 | 0.005±0.010 | 0.001±0.001 | 0.000 | 0 / 0 |
| R1e, shaping | 0.070±0.025 | 0.058±0.015 | 0.079±0.011 | 0.054±0.013 | 0 / 0 |
| RNpm, exact-only | 0.000 | 0.002±0.004 | 0.001±0.001 | 0.002±0.004 | 0 / 0 |
| RNpm, shaping | 0.056±0.042 | 0.035±0.026 | 0.040±0.026 | 0.045±0.029 | 0 / 0 |

Answer-window speech, teach100 → teach600 → free1000:

| arm | garbled response | answer-syllable hit | first syllable right / full answer |
|---|---|---|---|
| R1e, exact-only | 0.94 → 0.90 → 0.89 | 0.02 → 0.03 → 0.04 | 0 / 0 |
| R1e, shaping | 0.92 → 0.90 → 0.91 | 0.01 → 0.04 → 0.05 | 0 / 0 |
| RNpm, exact-only | 0.62 → 0.69 → 0.79 | 0.02 → 0.07 → 0.04 | 0 / 0 |
| RNpm, shaping | 0.68 → 0.63 → 0.71 | 0.03 → 0.01 → 0.00 | 0 / 0 |

- Accuracy is still 0 everywhere: seen, held-out and first syllable, at teach600 and free1000.
- Prediction specificity does not move. The answer's first-syllable code is predicted as much after other questions as after its own: R1e_shape 0.04 vs 0.04; RNpm_shape 0.18 vs 0.15.
- Value can only amplify top→muscle-copy increments that were just made. When those increments are babble, the praise strengthens babble.

## 2. RNg: no inhibition, growth/decay only (g31 0.3, decay 0.99 per 100 ticks)

**Not bounded in a usable way.** In 4 of 5 seeds it is bounded but mostly dead. In 1 of 5 it saturates.

| seed | L3 mean / max | P(empty) | P(saturated) | L3 count at lag 0 / 6 / 12 after a question |
|---|---|---|---|---|
| 0 | 15.8 / 255 | 0.62 | 0 | 47 / 12 / 2.7 |
| 1 | 6.5 / 169 | 0.75 | 0 | 20 / 0.7 / 1.2 |
| 2 | 5.0 / 186 | 0.76 | 0 | 11 / 0.4 / 0.0 |
| 3 | 5.7 / 149 | 0.74 | 0 | 14 / 0.0 / 0.0 |
| 4 | 397.7 / 400 | 0.00 | 0.99 | 400 / 400 / 400 |

- Question decoding at lag 12 is 0.00. It is 0.10 at lag 0.
- Accuracy is 0, and the child responds in only 0.33 of exchanges.
- So growth/decay alone gives the same bistability as at birth: the pause dies, or the layer runs away. The patch-mean arm (RNpm) remains the only no-inhibition form that stayed bounded in all 10 lives.

## 3. Oracle-path diagnostic (hand-wired; no answers wired)

In each post-question pause, L3 is replaced by a fixed random moving code per (question, lag) — a perfectly reproducible path. This is used in teaching and in probes.

| metric | R1e (no oracle) | R1e_oracle | RNpm (no oracle) | RNpm_oracle |
|---|---|---|---|---|
| seen / first-syllable accuracy | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| answer code: own vs other question (max over lags) | 0.07 / 0.07 | **0.20 / 0.09** | 0.13 / 0.11 | 0.07 / 0.07 |
| answer code: own vs other, per-lag mean | ≈0.06 / 0.06 | 0.04–0.07 / 0.05 | — | 0 after lag 2 |
| answer is the top prediction (best lag) | 0.05 | 0.07 | 0.01 | 0 |
| hub share of predictions | 0.80 | **0.14** | 0.79 | 1.00 |
| answer-syllable hit, teach100 → teach600 | 0.02 → 0.03 | **0.07 → 0.15** | 0.02 → 0.07 | 0.02 → 0.01 |
| garble in probes / talk during question | 1.00 / 1.00 | 1.00 / 0.82 | 0.63 / 0.61 | 0.48 / 0.61 |
| response rate (child speaks in the wait) | 0.97 | 1.00 | 0.69 | 0.50 |
| forced inner hearing @12 → said; first syllable | 0.87; 0.00 | 0.83; 0.00 | 0.51; 0.05 | 0.91; **0.39** |

With a perfect path, the hub largely disappears in R1e (share 0.80 → 0.14). The child also says answer syllables 3–5× more often (0.15 vs 0.03). But the prediction is still only weakly specific, so links 2–3 still fail. Two reasons are visible:
- **(a) The child babbles through almost every wait.** It responds in 99–100 % of exchanges, so the caregiver's answer comes after a recast (Q, 2 ticks, A) instead of after the 12-tick pause. The oracle step that precedes the answer is therefore usually lag 1–2, not lag 11. In addition, the child's own garble is the most frequent next input after every question, so top→L1 learns to predict it.
- **(b) In RNpm, the scale-free learned drive from a 40-unit code stays below the absolute prediction floor.** Nothing is predicted after lag 2.

When the right inner hearing is forced in the quieter RNpm_oracle child, the first syllable comes out right 0.39 of the time. So release works once the child is quiet.

**Diagnosis update.** A reproducible path is necessary, but not sufficient. The motor area's default blocking must also hold during the wait. Otherwise constant babble cuts the pause short and becomes the "answer" that gets learned. This needs fixing before Q→A can be tested, for example in X1e. The next knobs are the existing motor blocking knobs: motor return gain g_ret_mot, muscle→blocker latch c_mb, and threshold theta_m. The designer allows hand-wired speech basics here.

## summary_zh

1. "说得接近也夸"：大人高兴的次数从几乎 0 升到每次对话 0.04–0.08，但乱码没有减少，说出答案音节的比例也没有上升，答对率仍是 0。
2. 不加抑制、只靠权重增长和降权（RNg）：5 次一生里 4 次停顿时熄灭，1 次全部亮满，没有稳定的中间档；只有"减去周围平均"那种增益控制能一直不失控。
3. 手动给一条完全可重复的思路（诊断用，不给答案）："万能字"基本消失，说出答案音节的次数多了 3–5 倍，但预测仍不够针对问题，答对率还是 0。
4. 原因：孩子在大人等待时几乎一直在咿呀，打断了停顿，自己的乱码成了"最常出现的下一个声音"；孩子安静时，把正确答案放进内心声音，第一个音有 39% 能说对。
5. 所以除了可重复的思路，还要先让运动区"默认压住不说"在等待时真正起作用，再学问答。