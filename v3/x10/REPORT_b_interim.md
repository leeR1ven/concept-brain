# X10 stage (b), interim: speech basics (s = 1, seed 0)

Stage (a) is in `REPORT_a.md`. Plan and registered revisions are in `PLAN.md`; results are in `results/` (folder per job). These are interim numbers, all from seed 0. **MEASURED** means a run in `results/`.

## 结果（简明）

1. **设计者纠正（11-02）：** 声音和动作要经前额叶连，听觉区不直连运动区。旧的"听觉区 → 肌肉"已删掉，自检确认没有直连。
2. **按规律强制说话学"音 → 肌肉"，目前学不会：** 正确肌肉只比别的强约 1.1 倍。原因是动肌肉那一刻，前额叶最后一层有约 25% 亮，大多是各音共用的背景，学习规则把它们全连上了。把强制说话对准前额叶"正想着这个音"的那一拍（大人说完后第 4–8 拍），也只好一点。
3. **运动区"落定"（设计者的表征坍缩）：** 每个音配一群兴奋、一群抑制神经元，会降权。两个音 9:7 混合时，从全是乱码变成大多落到较强的那个；倾向一停，音就停。打平时还是常出乱码。
4. **前额叶倾向原来不带"是哪个音"的信息，根源在输入：** 听觉区前两层经常亮的神经元也被送进了前额叶。把前两层的上限从 90% 降到 60%，前额叶最后一层不同音的重合从 0.32 降到 0.14，什么都不想时也不再推肌肉。
5. **手动接按"心里想到这个音时亮"的神经元来挑：** 正确的音在倾向里排第一 83%（刚想到的几拍 100%）。
6. **完整说话测试比以前都好，但还没过关：** 想一个音说对 9%，乱码 59%；听大人说话后跟着出声 59%，听对话时自己乱开口每段 1.7 次。正在逐拍查原因。

## 1. Sound → action goes through the PFC (designer, 11-02)

- The old G1 (sound A1 → force neurons) was removed.
- Speech now runs PFC L4 → PFC L1 force copies → 1:1 returns → force neurons → competition with the blockers.
- Unit check 9 confirms that no table maps sound-area units to force neurons.

## 2. Learned route: forced regular speaking (designer, 11-02)

The scaffold forces the muscles only; learning stays on.

**MEASURED:**

| scaffold | learned true / other weight | x ranked first in the tendency |
|---|---|---|
| F1 (x x x), m 8 / 32 | 1.04 / 1.18 | ≈ chance |
| F3 (adult x, then the child is forced to say x x), m 32 | 1.58 (raw) | 0.02 on P1 |
| F4 (adult x, one forced x at the PFC's "thinking x" tick 6) | 1.12 | 0.09 (peak 0.12) |

- At a forced onset, L4 has 840–990 units on (about 25 %). x's inner code shares only J 0.21 with that state, against 0.12 for the other syllables. The one rule links about 3,000 L4 units to the force copies, with no selection.
- The PFC's "thinking x" window after hearing x is ticks 4–8 (x first 0.46–1.00, frozen). The PFC does not re-light x's sound copies; the state lives in L4.

## 3. Motor collapse, "MC pattern pools" (designer: 表征坍缩, inhibitory neurons included)

Each syllable has two groups:
- a 16-neuron excitatory group that drives its own force neurons;
- a 16-neuron inhibitory group that inhibits the force neurons of muscles outside the syllable.

Both groups sample one force neuron per muscle. They are generation 0 and plastic (γ_motor). The input is 0/1, as it is in the brain.

**MEASURED on the GPU, m = 8:**

| input | with MC | without MC |
|---|---|---|
| clean x (said, onset garble) | 1.00, 0 | 1.00, 0 |
| 60/40 mixture → the stronger | 0.98 | 1.00 |
| 90/70 → the stronger | 0.71 (m 32: 0.93) | 0.14 |
| 80/80 → one clear syllable | 0.54 | 0 |
| 50/50 → one clear syllable / garble | 0.45 / 0.54 | 0 / 1.00 |
| 2 of 4 muscles | garble 1.00 | garble 1.00 |
| ticks still on after the input stops | 0 | 0 |

- Pairwise muscle links ran away: all 24 muscles came on within 3 ticks. They were removed.
- Inhibition between rarely co-occurring muscles alone did not help.
- Within-muscle support and "release only with the pool" (P2) each cost the 60/40 and 90/70 cases, so they were dropped.

## 4. Where the PFC's syllable overlap comes from

Birth overlap trace, J between different syllables in the same voice. **MEASURED:**

| configuration | A1 | A2 | L1ext | L4 |
|---|---|---|---|---|
| chain band ≤ 0.9 (medium) | 0.41 | 0.55 | 0.53 | 0.32 |
| A1 + A2 band ≤ 0.6 (**P1**) | 0.20 | 0.37 | 0.38 | **0.14** |
| sound rates × 0.5 | 0.43 | 0.59 | 0.58 | 0.63 |

- An earlier figure of 0.84 came from an older measurement on base C. It is not the birth overlap.
- On P1, G1P no longer drives the force copies when nothing is thought: 0.68 → 0.00.
- P1 passes a1 and a 300-dialogue a1L.

## 5. G1P selection and the real tendency (P1, ticks 10–16 while the PFC thinks x)

**MEASURED:**

| G1P units | x ranked first | total lit | lit when nothing is thought |
|---|---|---|---|
| heard x | 0.18 | 0.67 | 0.00 |
| heard ∩ inner (median 3 units per syllable) | 0.39 | 0.35 | 0.00 |
| **inner x** (adopted as generation 0) | **0.83** (1.00 at ticks 10–13) | 0.25 | 0.02 |

## 6. Stage (b) on P1 (m 8, b_block 1.2, g_ret_force 1.6)

**MEASURED:**

| arm | b1 said | garble | b4 echo per line | b7 utterances per dialogue | b6 brake |
|---|---|---|---|---|---|
| G1P inner + MC | 0.09 | 0.59 | 0.59 | 1.70 | 0.88 |
| G1P inner, no MC | 0.03 | 0.69 | 0.66 | 1.42 | 1.00 |
| heard ∩ inner + MC | 0 | 0.59 | 0.91 | 1.47 | 0.94 |

The tick-by-tick diagnosis (`diag_say.py`) is running. It separates three candidate causes:
- several rivals at about half strength;
- the child's own voice feeding back;
- heard-path echo.
