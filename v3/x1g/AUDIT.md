# X1 audit against ARCHITECTURE.md (version 2026-11-01)

Labels: **P** = PROVED (code read, result file, or quick check); **A** = ASSUMED. File paths refer to the X1g code (in the repo: `v3/x1g/` and the folders it imports from). Quick checks: `audit/c1.py` (birth dynamics, seed 0) and `audit/c2.py` (100 overheard dialogues, N 400).

**What X1g ran (P).**
- Class and code: `core8.Brain` (core3 → 4 → 5 → 6 → 8), driven by `run8.py`. The copies sent to the server (jobs `x1g_pilot1`, `x1g_pcA`, `x1g_sB`) differ from the local code only in instrumentation and in an unused `fell_win31` option.
- Runs: M1600_k30 on/off, 3 seeds; PC1600 and PC6400, 3 seeds; loop pilots PL1600; seed-0 pilots P_s1600, P_s6400 and P_pc400x1f. N 6,400 is seed 0 only, because `main6400`, `loopC` and `restD` were never submitted.
- Main-arm config (`M1600_k30_on_s0.json`):
  - thresholds: frac, f2 0.18, f3 0.08;
  - homeostasis: on L2 and L3, k = 30, offset clipped to [−0.3, +3];
  - patch mean: pm = pm_ff = 1;
  - off: κ, partners, hippocampus;
  - weights: a31 = a33 = 1, g33 2;
  - release: rel1 (th1 0.015, ρ1 0.6);
  - decay: dec_31 0.998, dec_33 0.95 per 100 ticks;
  - g_int 0.5;
  - speech: A1 + A2 (gate 3–5 muscles).
- The ML loop arms also use th1 0.005 and a reciprocal birth loop with g_loop 16.

## 1. Clause table

| § | clause | verdict | evidence |
|---|---|---|---|
| 2.1 | binary neurons; distributed codes; on/off pairs, input half on; own vs other voices | FAITHFUL | world.py:32-37; core8.py:173 |
| 2.1 | in-weight-fraction threshold is sensory only ("不用在前额叶") | DEVIATES | used in PFC L2 and L3, core8.py:211-215, 243 |
| 2.2 | 3-D local sensory net, deeper layers sparser; concept area | MISSING | A0 → PFC L1 directly; concept area deferred by §4.14 |
| 2.3 | chains from every layer to PFC and hippocampus | DEVIATES | one sensory layer, no hippocampus |
| 2.3 | 1:1 return to the same S | DEVIATES | muscle copy → its muscle 1:1 (core6.py:195); stop copies → all stops via the mean (core6.py:196, 206); content copies re-light no S, inner hearing drives muscles via A1 (core6.py:193); voice copies' return goes nowhere |
| 2.3 | no chain for always-on / never-on units | FAITHFUL / minor DEVIATES | rate > 0.6 cut (core3.py:79); never-on units still get chains |
| 2.4 | "中间各层": fixed random local; split/delete | DEVIATES (one middle layer); FAITHFUL (fixed); MISSING (split/delete) | core3.py:97-100 |
| 2.4 | top(t) → L1 input(t+1), Hebbian, local | FAITHFUL | post = forward-chain input, core4.py:143-147; M31 radius 0.3, core3.py:108-109 |
| 2.4 / §3 | thought held by top → L1 → mid → top loop | DEVIATES | M arms: no top → L1 drive at birth (B31 = L31 = 0); after learning the loop cannot close (Q5). ML arms: decay-exempt reciprocal birth loop (ADDED, core4.py:37-49) |
| 2.4 | roughly count-preserving | DEVIATES (P) | N 400 at birth, during speech: L1 ≈ 25 → L2 52–70 → L3 57–81; silence 0 → 0 → 0 |
| 2.4 | no PFC inhibitory neurons at first | ADDED (allowed by §4.10) | patch-mean subtraction on the L1 prediction and on L3 feed-forward and lateral drive (core8.py:187-188, 235-239); homeostasis |
| 2.4 | rarely co-active units inhibit each other | MISSING | partners off (core3.py:185-186); L33 floored at −B (core8.py:147) |
| 2.4 | thinking range of 20–30, continuous | DEVIATES (P) | 30 is a long-run per-unit average; per tick 0 in silence (P(empty) 0.50–0.53) and 50–170 in speech |
| 2.4 | two-layer weights divided by out and in | DEVIATES | a = 1, out only (run8.py:48; core5.py:48-54); a = 0.5 only in M400_ref and PC400; follows the §4.20 suggestion |
| 2.5 | output = L1 lit internally; tunable return | FAITHFUL (form) | `pred`; g_ret_aud 1, g_ret_mot 1 |
| 2.5 | return weaker than next-item link | DEVIATES / ADDED | g_int 0.5 halves an internally lit L1 unit's drive into L2 (core8.py:208-210) |
| 2.5 | prediction biases hearing | MISSING | — |
| 2.6 | hippocampus | MISSING | asserted off, core8.py:43; phase1.py:17 |
| 2.7 | default inhibition; competition, "no hard threshold" | DEVIATES | A2 is a count rule (core6.py:210-213); tonic blocks act only through their mean (core6.py:204-208) |
| 2.7 | blocks without chains, stops with chains; brake reads the input layer | FAITHFUL | core8.py:177; core6.py:199-202 |
| 2.7 / 2.11 | sound → muscle hand-wired in generation 0 | FAITHFUL; decay-exempt, which DEVIATES from the §4.1 X1f lesson | A1, wire.py:35-44 |
| 2.8 | held A learns; a fall does nothing; just-fallen A counts; co-rising units strengthen | FAITHFUL (P) | core4.py:114-129; core8.py:116-149 |
| 2.8 | weaken only when A re-lights after B | DEVIATES (minor) | only if B is still on (core4.py:124-129) |
| 2.8 | concave growth, 1 % floor; concave-slowed weakening | FAITHFUL | core3.py:189-193 (w_ref 0.5, floor 0.01) |
| 2.8 | global decay slowed by the same concave factor | DEVIATES | plain multiplication (core8.py:100-114) |
| 2.8 | one decay; birth links exempt; deeper slower, hippocampus fastest | DEVIATES (exemption FAITHFUL) | L33 ×0.95 every 100 ticks is the fastest in the brain; L31 ×0.998 (common6.py:16) |
| 2.8 | reward over 十几到二三十拍, on all recent changes | DEVIATES; inactive | λ 0.85/tick (≈ 6 ticks), top → muscle copies only (core3.py:408-413, 477-485); r = 0 in overhearing (world6.py:24-55) |
| 2.8 | fast-changing links for new learning | MISSING | `fast` off, hippocampus off |
| 2.11 | continuous life, overhearing, systematic steps 1–3 | FAITHFUL | run8.py:200-208; world6.py |
| 2.12 | registered hand-wiring; test only unwired content | FAITHFUL | run8.py:189 |
| 2.12 | scale the whole brain together | DEVIATES | only L2 and L3 (as §4.20 asked) |
| — | max-relative L1 release (rel1) | ADDED (§4.20 suggestion) | core8.py:189-195; in effect a global "≥ 60 % of max" rule over L1 |
| — | learned self-links L33[i,i] | ADDED (unintended) | core8.py:132-136. P: after 100 dialogues at N 400, all 400 units have one, at 0.2 % of their out-weight (negligible) |

## 2. Answers

**Q1. Areas and sizes (P).**
- A0 input: 360 units (180 pairs: 150 content, 30 voice).
- Motor: 40 units (24 muscles, 8 tonic blocks, 8 stops).
- PFC: L1 has 212 units (150 + 30 copies + 24 + 8); L2 = L3 = N (default 400; X1g used 400, 1,600 and 6,400).
- Hippocampus, concept area and values: 0 neurons (r is a world scalar).
- PFC : rest = (212 + 2N) : 400, so 2.5, 8.5 and 32.5.
- Only L2 and L3 changed with N. Patches went 12 / 48 / 193 and the target fraction 30/N. Fan-ins (24, 30, 20) and the M31 radius stayed fixed, so connection density fell as N grew.

**Q2. Chains and L1 input (P).**
- Chain pairs: the 180 "on" units of A0 (every "off" unit is cut), 24 muscles and 8 stops.
- L1 during silence: 0–2 noise lines, plus 8 stop copies on the tick after an adult stops. Predicted lines: 0 (M arms at birth), 0–2 (ML arm at N 1,600).
- L1 during speech: about 23–25 lines, i.e. ≈ 15 auditory plus 8 stop copies (the brake fires on every adult voice). Predicted lines add 0–9; their time average after learning is 1.8–2.0.

**Q3. How each layer's activity is set.**
- L1: share ≥ th1 and ≥ 0.6 · max, after the patch mean is subtracted.
- L2: drive − h ≥ 0.18 · in-sum (= 2.16). Internal L1 lines count half.
- L3: (feed-forward − patch mean) + (lateral − patch mean) − h ≥ 0.08 · in-sum, where the in-sum includes learned lateral weight (≈ 1.44).
- In both L2 and L3, h tracks 30/N (τ 1,000) and is clipped to [−0.3, +3].
- Against §2.4 and §4.10: local inhibition and homeostasis are sanctioned fallbacks. The L1 max-relative rule is close to a global threshold, which §4.10 allows only as a shortcut.
- The clip lets h lower a threshold by only 0.3, so homeostasis trims speech bursts but cannot rescue silence. P: at lag 3 the largest L3 drive is 0.52–0.60 against a minimum threshold of 1.10.

**Q4. Learning (P).** Held form, fall and concave clauses: see the table.
- L31: +0.02 · s(w) if A was on at any of the 4 ticks before B's forward-chain onset; −0.01 · s(w) if A rises while B is held.
- L33: the same, plus co-rising; floored at −B.
- Decay: plain multiplication, L31 ×0.998 and L33 ×0.95 every 100 ticks.
- The reward never fires.

**Q5. Prediction, lateral links, middle layer and gains.**
- With a = 1, each active L3 unit pushes exactly 1.0 into L1 and 2.0 laterally, however much it has learned.
- L33 saturates: 100 % of pairs after 100 dialogues at N 400 (P), 95–97 % at N 1,600 (P), 27 % at N 6,400 (reported).
- Gains: g_ret 1, g_int 0.5, g_dir 0.25.
- The loop cannot close (P):
  - one internal L1 line gives an L2 unit 0.25;
  - the L2 threshold is ≥ 1.86, so a unit needs about 7 internal lines among its 24 inputs;
  - 0–2 lines are lit, and L2 = 0 from lag 2.

**Q6. Motor.**
- A muscle fires if dn > 0.22 + 0.5 · (fraction of blocks on, ≈ 0.5) + 1.0 · (stop fraction) and dn ≥ 0.6 · max.
- A2 then silences any count outside 3–5.
- Brake 3.0 when ≥ 60 % of an adult voice's lines are on.
- A1 (598 links, least squares from the physics table) and A2 are fixed and decay-exempt, so they sit outside the plastic part.
- P, M1600 seed 0: inner hearing drove 97.5 % of onsets; A2 gated 6,406 releases.

**Q7. Leaks: none in the main arms (P).**
- `step` gets only the sound vector and r, and r = 0 throughout overhearing.
- A1, the brake's voice-line layout and the content flag are registered birth wiring.
- Each voice has 6 dedicated units, which is a cleaner speaker label than §2.1 implies (A: harmless here).
- Diagnostic hooks only: `_force_lines`, `_oracle`, `wm`, H33/H31/H13.

## 3. Deviations ranked by likely effect

1. **The loop cannot close.** rel1 allows 0–2 internal lines, g_int halves them, and L2 needs about 7. The M arms have no loop at birth at all. → (a) P. → (c) A: at answer onset only the frame-constant last syllable's code remains.
2. **Lateral L3 links cannot sustain activity.** a = 1 normalisation, L33 saturating at 95–100 %, and the patch mean leave lag-3 drive at about half the threshold. → (a) P; (b) A.
3. **No hippocampus.** X6 bridged the pause through it (0.64 against 0.02). → (a)(c) A.
4. **No partner inhibition and no net-inhibitory learning.** → (b)(c) A.
5. **Activity control gives bursts, not a steady 20–30.** With X1f thresholds (about 95 active) codes collapse, as X1g reported. → (a) P; (b) A.
6. **L33 decay is the fastest in the brain, and the decay is not concave.** → (b)(c) A.
7. **a31 = 1 scale-free readout.** Generic units push as much as specific ones; the fall window also links Q1 → a2. → (c)(d) A.
8. **ρ1 0.6 plus the A2 gate.** A wrong syllable at 60–70 % passes, and A2 silences the mixture. → (d) P by code.
9. **Low effect:** reward unused; only the PFC scaled; no sensory hierarchy or concept area; one middle layer; self-links.

**Earlier versions.**
- X1c: fall weakening and κ = 2 self-hold (both wrong by the document); quantile and local inhibition; learned partners; no L33 learning; the hippocampus read and recalled L1 directly.
- X1d: onset rule; RNpm (frac thresholds, patch mean, a 0.5, floor 0.35).
- X1e: L33 learning.
- X1f: A1/A2; a hand-wired WM part (diagnostic).
- X1g: homeostasis, a = 1, rel1, larger N.

## 4. summary_zh

1. X1g 最终用 core8（主组：前额叶第二、三层各 1600 个，目标每层约 30 个亮）。
2. 照文档做对的：只有亮和不亮；输入层正反对一半亮；先亮后亮规则（保持的也长、熄灭不减弱）；凹形增长加 1% 可塑；最后一层只连附近第一层；旁听世界；手动接线已登记。
3. 最大偏离：文档要的"最后一层→第一层→中间层→最后一层"回路转不起来。第一层被内部点亮的只有 0–2 个，还只算一半，中间层要 7 个左右才亮。
4. 所以输入一停，3 拍后最后一层全灭（出生时、学 1 万段对话后实测都一样）。
5. 最后一层内部连接撑不住：每个神经元总推力固定，学习后又几乎两两全连，每条分到的几乎为零。
6. 缺的：海马体（完全关掉）、"很少一起亮的互相抑制"、感觉区多层网络、概念区、分裂删除。
7. 加的：每小片减平均、按长期活跃度调门槛、第一层"只放最强的六成以上"、"3–5 块肌肉才放行"硬规则。
8. 参数不符：最后一层内部连接降权最快（每 100 拍乘 0.95），文档要越深越慢；降权也没按凹形变慢。
9. 只放大了前额叶两层；手动接的说话连接不降权，不符合 X1f 的教训。
10. 世界没有把标签或事件送进大脑；奖励在旁听阶段从没起作用。
