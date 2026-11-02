# X10 stage (a): newborn checks of the strict whole-brain build (s = 1, seed 0)

Plan, criteria and conformance table: `PLAN.md` (written and reviewed against ARCHITECTURE.md before any run). Code map: `CODEMAP.md`.
Tables: `python3 agg_a.py results/a_loop`, `python3 agg_a.py results/a_size`, `python3 agg_sens.py results`.
Labels: **MEASURED** = a run in `results/`; **ASSUMED** = judgement.

## 结果（简明）

1. **第一次在出生时（还没学习）就做到：问题停顿 12 拍后还能认出。** 用局部抑制、互相对应的回路、回路强度 1.0：问题刚结束 1.00，停 6 拍 0.64–0.71，停 12 拍 0.34–0.40（瞎猜 0.03）。以前 X1 的模型这里是 0.00–0.08。
2. **前额叶各层"进多少出多少"成立**：相邻层数量比 0.9–1.05。没有局部抑制、回路又强时会失控（95% 全亮）。
3. **前额叶大小**：中档保持得最好（0.40）；大档在同样参数下反而差（0.12–0.17）。但大档只用了中档的回路强度，没单独调，所以"大前额叶更差"还不能下结论。
4. **学习时活动上涨的来源**：逐个参数试过以后，只有"前额叶最后一层学到的内部连接强度"和"海马体学到的内部连接强度"有大影响；感觉区稀疏度、前额叶大小、学习率等都改不了。把这两个强度从 1 降到 0.1，300 段对话内活动稳定。
5. **不及格的一项**：海马体在问题结束时 32 选 1 只认对 0.30–0.44（要求 0.5）。原因是海马体没有出生回路，只编码当前听到的（多是句式共享的最后一个字）。按设计者"相同经历分不开先不管"（§四.19），改用"同一问题、不同上下文的海马编码更相近"来衡量，实测差值 0.11–0.17。
6. **说话（b）的毛病找到了**：强制想一个音时 4 块肌肉只有 3 块到位（含糊）；说完后自己的声音又点亮这个音 82% 的编码，再推起一块肌肉，于是乱码。问题在"声音→肌肉"的接法，不在前额叶。

## 1. Loop grid (48 configurations; P0/P1 inhibition × reciprocal/random loop × g_loop {0.25, 0.5, 1, 2} × w44 {0, 0.5, 1})

Every configuration was calibrated online at birth with its own loop and lateral links on, learning off (PLAN addendum).
**MEASURED**, top rows by lag-12 hold:

| config | a1 | a2 | a3 | L4 32-way decoding, lag 0 / 6 / 12 (chance 0.03) | filler-given-frame, lag 12 (chance 0.125) | Hs decoding, lag 0 |
|---|---|---|---|---|---|---|
| P1, reciprocal, g_loop 1.0, w44 0 | pass | fail | pass | 1.00 / 0.64 / 0.34 | 0.34 | 0.36 |
| P1, reciprocal, g_loop 1.0, w44 0.5 | pass | fail | pass | 1.00 / 0.61 / 0.32 | 0.35 | 0.32 |
| P0, reciprocal, g_loop 0.25, w44 0.5 | fail (L4 peaks at 0.55 on) | fail | pass | 0.94 / 0.48 / 0.36 | 0.38 | 0.30 |

- a1 passes in 22 of 48, all but one with P1 (one non-learning inhibitory neuron per patch).
- With P0 and g_loop ≥ 1, the PFC either saturates (L1 3,577 of 3,762 on, L4 all on) or freezes.
- Random birth loops hold no better than 0.16 at lag 12. The reciprocal loop (msg 84) is the one that holds.
- Count preservation holds in every P1 row: L2/L1, L3/L2, L4/L3 = 0.90–1.05 in silence and in speech.

## 2. PFC size × sensory threshold fractions (P1, reciprocal, g_loop 1, w44 0)

| PFC preset (N1) | sound rates r_l | a1 | a3 | lag 0 / 6 / 12 | L1 silence / speech |
|---|---|---|---|---|---|
| medium (3,762) | 0.20 / 0.12 / 0.08 / 0.05 (default) | pass | pass | 1.00 / 0.71 / **0.40** | 589 / 778 |
| medium (3,657) | 0.30 / 0.20 / 0.12 / 0.08 (denser) | pass | pass | 1.00 / 0.73 / 0.31 | 615 / 803 |
| small (1,913) | 0.10 / 0.06 / 0.04 / 0.025 (sparser) | pass | fail (0.30, rounding) | 1.00 / 0.36 / 0.30 | 484 / 490 |
| medium (3,652) | sparser | pass | fail | 0.98 / 0.45 / 0.18 | 627 / 770 |
| large (8,178) | default | pass | fail | 1.00 / 0.71 / 0.17 | 758 / 1,129 |
| large (8,178) | denser / sparser | pass | fail | 1.00 / 0.69 / 0.17; 1.00 / 0.34 / 0.12 | |
| chain band at every depth (8,150) | default | pass | fail | 1.00 / 0.55 / 0.14 | 693 / 1,100 |

- **ASSUMED:** the large preset ran with the loop gain tuned for medium. Its internally lit L1 copies are fewer (29 against 41), so its loop is relatively weaker. "Large holds worse" is not yet a fair test of the designer's large-PFC intuition (§二.4). A g_loop sweep for the large preset is queued.

## 3. Learning on: a1L stability and the remedy ladder (300 dialogues, §四.10 order)

| config | rung | a1L | last window L4/L3 | last window Hs (target kh = 50) |
|---|---|---|---|---|
| w44 0 | R0 as chosen | fail | 2.09 | 855 |
| w44 0 | R1 learned gains g44, g_hh, g_h1 = 0.3 | fail | 1.68 | 251 |
| w44 0 | R2 gains 0.1 | fail (one Hs window 100.4 > 100) | 1.15 | 69 |
| w44 0 | R3 = R2 + η / 4 | pass | 1.17 | 76 |
| w44 0.5 | R0 / R1 | fail | 2.21 / 2.29 | 831 / 169 |
| w44 0.5 | R2 | pass | 1.40 | 83 |

Activity budget (mean active units per tick, silence / speech; **MEASURED**):

| config | when | A1 | A2 | A3 | A4 | L1ext | L1int | L4 | Hs |
|---|---|---|---|---|---|---|---|---|---|
| w44 0, R0 | birth | 225 / 352 | 221 / 447 | 165 / 307 | 123 / 186 | 606 / 765 | 41 / 40 | 691 / 643 | 47 / 59 |
| w44 0, R0 | 100 dialogues | 254 / 375 | 331 / 541 | 275 / 425 | 235 / 319 | 773 / 927 | 147 / 147 | 1,735 / 1,734 | 801 / 801 |
| w44 0, R1 | 300 dialogues | 425 / 531 | 749 / 920 | 827 / 961 | 927 / 1,019 | 1,297 / 1,443 | 213 / 218 | 1,810 / 1,802 | 254 / 248 |
| w44 0, R2 | 100 → 300 dialogues | 263 → 239 / 386 → 357 | | | 290 → 188 / 396 → 247 | | | 1,139 → 910 / 1,117 → 833 | 86 → 65 / 98 → 73 |
| w44 0.5, R2 | 100 → 300 dialogues | 245 → 268 / 369 → 388 | | | 210 → 310 / 265 → 403 | | | 1,011 → 1,290 / 920 → 1,273 | 64 → 80 / 73 → 87 |

- At R0 the extra activity enters at L4 and Hs: L4 rises about 2.5× and Hs about 16×, and both become the same in silence and speech (self-sustained). The sound layers and L1 rise only about 25 %.
- At R1, it then flows back through the 1:1 returns into the sound area (A4 rises 7×).
- w44 0 / R2 falls back between 100 and 300 dialogues, but w44 0.5 / R2 is still rising at 300 (L4 1,011 → 1,290, A4 210 → 310). **ASSUMED:** for one unbroken life the trend matters more than the 300-dialogue window, so the next step runs the stability check longer.

## 4. Sensitivity (20 one-at-a-time variations around P1 / reciprocal / g_loop 1 / w44 0)

| knob | effect on learning-on inflation (last-window L4/L3, Hs; R0 = 2.13, 829) | effect on birth hold (lag 12; base 0.34) |
|---|---|---|
| g44 0.3 | **1.58**, 966 | 0.30 |
| g_hh, g_h1 0.3 | 2.21, **110** | 0.34 |
| g_ih × 2 | 2.21, 330 | 0.31 |
| patch gain +0.5 | 1.77, 807 | **0.05** |
| sound rates × 2 | 1.83, 823 | 0.35 |
| sound rates × 0.5 | 2.06, 753 | 0.21 |
| size small / large | 1.69 / 2.49 | 0.18 / 0.20 |
| band at every depth | 2.03, 1,008 | 0.23 |
| g_loop × 0.5 / × 2 | 2.83 / 1.13 (saturated at birth, L4 46 %) | **0.06** / 0.16 |
| g31 0.3, η × 0.25, w_ref × 4, decay faster, kh 25 / 100, w44 0 / × 2 | 2.08–2.36 | 0.20–0.36 |

- The PFC activity at birth is set by the sound area (L4 fraction 0.18 at default rates, 0.23 at × 2, 0.34 for the small preset), as the designer said (11-02). The inflation during learning is not: it comes from the learned L4 ↔ L4 and Hs ↔ Hs gains. R2 fixes it at that source.
- The birth hold is fragile to g_loop, patch gain and decay.

## 5. Why a2 fails, and the criterion change

- a2's PFC part passes: lag-0 decoding 1.00, Jaccard margin 0.19–0.32.
- Its hippocampal part fails in every configuration: Hs 32-way decoding at question end is 0.30–0.44 against a 0.5 bar, and no knob moves it.
- Cause: with ρ = 0 (§四.19), the hippocampus has no birth recurrence, so its code at question end holds only the current chain input, mostly the frame-shared last syllable. The PFC's loop is what carries the filler.
- The Hs Jaccard margin (same question in another context minus another question of the same frame) is 0.11–0.17 in 9 of the 11 variations that report it. That is §四.19's "same input → same code".
- **Decision (coordinator, 11-02):** a2's hippocampal bar becomes "Hs Jaccard margin ≥ 0.1", following §四.19 and the designer's 11-02 note that a small model not separating alike experiences is normal. The 32-way bar is kept as a reported number.

## 6. Stage (b) diagnosis (`results/diag_b.json`; P1 / reciprocal / g_loop 1 / w44 0.5 / R2)

- Forced syllable: only 3 of 4 own muscles fire. The weakest own drive (0.97) sits under the threshold raised by 1–2 recruited blocking units (1.05–1.15). Result: a mumble.
- One tick after speaking, the child's own voice lights 82 % of the syllable's A1 code (302 of 369 units). That tail pushes one muscle over threshold: the "~" garble (5 of 8 syllables).
- The same result with the loop and lateral off and with the hippocampus removed (`pilot/b_noloop`), so the fault is in the G1 sound → muscle wiring, not in the PFC.
- **Next (approved, generation-0 detail, §二.11.1):** build G1 only from A1 units that the four adult voices light and the child's own voice does not. G1 stays in the decaying part and keeps learning, and no gate is added. Then rerun the registered g_ret × G1-scale sweep.

## 7. Next steps

1. **Seeds.** Three more seeds each for both candidates: P1, reciprocal, g_loop 1.0, medium, default rates, with either w44 0.5 + R2 or w44 0 + R2/R3. Run a + a1L, with a1L extended to 1,000 dialogues to see the trend.
2. **Choice.** Among the candidates that pass a1, revised a2 and a1L on every seed, take the highest mean lag-12 (the PLAN rule). The other becomes the (c) arm.
3. **Large PFC.** A fair test with g_loop {1.5, 2} at birth, and a1L with R2.
4. **Speech.** Stage (b) with the G1 fix. If (a) and (b) pass, freeze the code and run (c).
