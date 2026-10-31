(Saved by the coordinator from the reviewer's final reply; the harness blocked the reviewer's own write.)

The latch and the continuous loop both finished. Here is the follow-up report.

# Follow-up: the reverberating PFC hold — parameter map, continuous update, bench

The coordinator asked me to tune the loop between M6 (collapse) and M6b (no hold) rather than replace it, and then (designer's addendum) to do it without any reset at a new line.

- **Map.** I mapped 51 settings on the realistic bench line codes, 1,500 training lines at Zipf frequency (`reverb_sweep.py`; outputs `sweep1.out`, `sweep2.out`, `sweep_c1.out`, `sweep_c2.out`; notes `REVERB_NOTES.txt`).
- **Target.** A continuous setting meets the target: content decodable ≥ 0.8 for 120 ticks, a new line takes over within 3 ticks, part of the previous line stays as context, held states are no more alike than their inputs, no period-2.
- **Bench.** The continuous loop does not beat the gated latch: it is about 0.03 lower on B and C and equal on D (seed 0). It does win on rituals, and it gives the first non-zero one-shot score.

## 1. Why M6 collapsed

- **Main cause: learning on the first tick of a new line.**
  - On that tick the learning source is still the old thought, so every line gets wired to whatever preceded it.
  - Over 1,500 random successions, everything links to the shared and frequent units, and that makes one generic basin.
  - Removing only this raised decode at 60 ticks from 0.03 to 0.77.
- **Second cause: more active units (k = 120) than the line drives (~59).**
  - The free slots are filled by what the old thought or the frequent lines predict.
  - Setting k ≈ line size (40–60) brings the overlap between held states of different lines down to the inputs' own (0.05–0.11).
- **M6b's set point acted too fast** (τ 500), so it suppressed the very units being held. A slow one (τ 20,000) only helps once the two causes above are fixed.
- **Learning while a line is heard** (old and new content mixed) also collapses the continuous regime (overlap 0.54–1.00). Learning in the settled state after the input does not.
- **The period-2 index** lies between −0.031 and +0.002 across all 51 settings. The leak prevents alternation everywhere.

## 2. The map

Decode is 20-way (chance 0.05). Overlap is the Jaccard between held states of different lines; the inputs' own overlap is about 0.06.

| regime | where it lives | behaviour |
|---|---|---|
| collapse | learning at line onset, or while hearing; weak input (gain ≤ 2); k well above line size; partial reset | overlap 0.4–1.00; decode ≤ 0.2 |
| sticky (continuous) | leak 0.9, input gain 2–4 | the old thought's potential (about 10× its drive) wins; takeover 0.2–0.6 |
| balanced, gated (reference only; uses a reset) | no onset learning; k 40 with normalisation of frequent frame features | decode 0.97 / 0.87 / 0.80 at 10 / 60 / 120 ticks; overlap 0.05; clean replacement (+0.00) |
| **balanced, continuous** | leak 0.8; input gain 4–6 for others' speech; k ≈ line size; learning only after the input | new line decoded 0.83–0.97 at 3 ticks; hold ≥ 0.83 for 120 ticks; line n−1 kept as context 0.10–0.27; overlap at input level |
| favourites | spare active units (k 120), or fast fatigue (0.2, τ 50) | long silences drift toward often-heard lines: 33–67% of 500-tick silences end near a top-10% frequent line (chance 10%), at the cost of takeover (0.67) or of more overlap (0.36) |

Silent drift is slow in every balanced setting: the held state at 60 vs 500 ticks keeps J 0.88–0.99.

## 3. Recommended continuous setting (no reset)

**Setting:** k 60; input gain 4 for others' speech and 0 for my own (my own speech is predicted); leak 0.8; fatigue 0.1 with τ 100; learning only in the first 10 silent ticks after an input.

| measure | value |
|---|---|
| new line decoded at 3 / 6 / 10 ticks after its end | 0.87 / 0.87 / 0.87 |
| line n−1 still decodable as context | 0.27 / 0.27 / 0.23 |
| hold at 30 / 60 / 120 ticks | 0.83 / 0.83 / 0.83 |
| overlap between held states (inputs) | 0.09 (0.06) |
| silent drift, J(60, 500 ticks) | 0.94 |
| period-2 index | −0.001 |

Faster fatigue gives the slow wandering toward favourites the designer described, but at the cost of takeover.

## 4. Bench: latch vs continuous loop

Seed 0, the pruned agent, plan A600 → babble → B800 → C800 → D600. The two runs differ only in the PFC hold; references are from the bench README.

| task | latch | continuous loop | v1 bar | prior | n-gram |
|---|---|---|---|---|---|
| B overall (held) | 0.13 (0.05) | 0.10 (0.02) | 0.12 | 0.14 | 0.13 |
| B where | 0.23 | 0.18 | 0.33 | 0.33 | 0.32 |
| C overall (held) | 0.29 (0.18) | 0.26 (0.14) | 0.19 | 0.31 | 0.54 |
| C offer | 0.45 | 0.34 | 0.04 | 0.09 | 0.49 |
| C ritual | 0.19 | 0.33 | 0.51 | 0.21 | 0.99 |
| C count | 0.27 | 0.20 | 0.08 | 0.32 | 0.30 |
| D overall | 0.09 | 0.09 | 0.21 | 0.24 | 0.16 |
| D name | 0.10 | 0.12 | 0.00 | 0.17 | 0.17 |
| D one-shot | 0.00 | 0.03 | 0 | 0 | 0 |

- Mechanics are identical and perfect in both runs: 0 overlap, latency 6, every syllable 3 ticks, no garble.
- Both keep learning in use (seen items, early → late third):
  - C: latch 0.31 → 0.45, loop 0.33 → 0.49;
  - B: latch 0.17 → 0.29, loop 0.14 → 0.22.

**Likely cause of the gap: a context mismatch in learning by listening** (ASSUMED, consistent with offer being the main loss).
- While the model reply is being overheard, it flows strongly into the thought, so the reply is learned in the context "prompt fading + reply so far".
- When the agent answers itself, its own speech barely enters, so the context is "prompt".
- The latch holds the prompt through any reply, so its training and use contexts match.
- Continuity helps where carry-over matters: rituals, and the first one-shot answers.

## 5. Recommendation and next step

- **Keep the continuous loop as the PFC's thought.** It meets the designer's requirements and the module target.
- **Next test:** gate the input gain by the predictor's own surprise, so a well-predicted reply barely enters the thought and only unexpected content moves it. That should remove the train/use mismatch.
- **Until that is tested,** the latch scores about 0.03 higher on B and C.
- **Caveat:** one seed. The B and C differences are about two standard errors (n ≈ 430–450 probes each).

## summary_zh

1. 循环保持之所以塌缩成同一个状态，主要不是循环本身的问题，而是两点：新句子到来的第一拍就学习，这时的源还是"旧想法"，于是每句话都和前一句连在一起；同时亮的神经元比句子本身多，空位被常见的内容填满。改掉这两点后，内容能保持住，也不再塌缩。
2. 按设计者的要求不清空、让新句子连续流入（只靠疲劳和竞争让旧想法淡出），找到了合适的参数：遗忘 0.8、对方说话的输入强度 4、疲劳 0.1。新句子说完 3 拍后就能被识别（0.87）；上一句仍作为背景保留一部分（0.27，瞎猜是 0.05）；120 拍内内容都能识别（0.83）；没有塌缩，也没有两拍来回跳。
3. "偏爱的想法"也能出现：疲劳快一些、或者亮的神经元多一些时，长时间安静后会慢慢走向常听到的话题（33–67%，偶然只有 10%）；代价是新句子接管得慢一些。
4. 放回模型里测试（同一种子、同样的训练）：连续循环版 B 0.10、C 0.26、D 0.09，闸门锁存版 B 0.13、C 0.29、D 0.09。循环版在打招呼题上更好，"一次学会新物体"题第一次有了得分，但总体还没有超过锁存。
5. 原因很可能是：旁听别人回答时，回答也流进了"想法"，可自己回答时，自己的话基本不进入"想法"，于是学的时候和用的时候背景不一样。下一步：让输入强度由"意外程度"决定，已经预测到的话不太进入想法。