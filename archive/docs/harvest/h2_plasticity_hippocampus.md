(Saved by the coordinator from the reader's final hand-back.)

# Harvest: what in the plasticity, hippocampus and neuro/ML review documents can refine the designer's architecture

Sources: ARCH = `docs/ARCHITECTURE.md`; 02 = `research/02_plasticity.md`; 03 = `research/03_hippocampus.md`; NR = `v3design/review/neuro/report.md`; MR = `v3design/review/ml/report.md`. I also checked `v3/x1c`, `v3/x345` and `v3/x4b`. Read only, no runs.

**Already in ARCH, so not listed as new.** 02 and 03 were written against the old code. Several of their main proposals are now the designer's design: co-active units strengthen each other; the hippocampus has its own population with birth recurrence (X3: required; with chain copies only, recall is a coin flip); two-layer weights; gain control. Below, these appear only where a refinement is still left.

## A. Refinements, ranked by expected value

**1. A held neuron counts once, as "before"** (02 §1.2, §1.4, §7 F1; 03 §8.2)
- Part: the rule in §二.8. Answers §四.1 and §四.6.
- Change: when A stays on through B's whole life, count A once, as "A before B" (strengthen when B rises). Do not also count it as "A, B, A again", which weakens A→B when B falls. X1c used that second reading of msg 50, and it cancels every held link. Co-rising units keep their symmetric term. The ratio η_co : η_seq becomes a per-area coefficient, with a larger η_co share in the hippocampus (CA3-like).
- Evidence:
  - X1c §5.1: held units grew nothing, and every arm answered 0.00.
  - 02 §1.4 (arithmetic on the old code): held think-layer links learned about 0.0001, against 0.12 at birth.
  - Brain: co-firing at ≥40 Hz gives LTP whatever the order (Sjöström 2001, strong). CA3 STDP is symmetric: 58 vs 4.5 storable patterns (Mishra 2016, medium–strong).
- Verdict: **ADOPT.** It only makes the designer's 11-01 correction exact. Check it in the X1c rerun: is Q→A specific, and does runaway back-and-forth activation come back?

**2. A slow, bounded, divisive gain for each PFC neuron** (03 R1 and exp6; 02 5-1 and §8.2)
- Part: §二.4. Answers §四.6 (same question → same path). It is the designer's own fallback in §四.10 ("按长期活跃程度自我调整门槛").
- Change: each L2/L3 neuron divides its feed-forward drive by its own long-run mean drive. The gain is capped, and its time constant is much longer than a dialogue. Units that win for every question lose their edge.
- Evidence:
  - 03 exp6 (numpy replica, 80 sentences, one seed): cross-sentence overlap fell from 0.226, with 242 "generic" units, to 0.040 with none (chance 0.026).
  - X1c: after 300–800 teaching exchanges the top state collapsed (cross-question Jaccard 0.8–1.0). Unbounded threshold homeostasis also collapsed (0.92), so the gain must be bounded and divisive.
  - Brain: Turrigiano 1998; Hengen 2013 (strong).
  - Risk: slow drift of what units mean.
- Verdict: **TEST.** Add the gain to X1c's B_loop arm and compare same-question with cross-question Jaccard at answer onset (target ≥0.6 against ≤0.3).

**3. Set growth and decay from the equilibrium, not by trial** (02 §1.3; X3's γ sweep)
- Part: §二.8. Answers §四.4 and §四.10 ("find the right growth/decay parameters").
- Change:
  - Under concave growth plus one decay γ, a link confirmed f times per decay period settles at L* = [−w_ref + √(w_ref² + 4fηw_ref/(1−γ))]/2 (pow = 1).
  - Choose η, w_ref and γ per area so that (i) at L*, the learned top→L1 drive stays below the chain drive, and (ii) the hippocampus stays inside its palimpsest window. X1c C/C2 ran away exactly when the predictions added drive on top of the chain input.
  - Also PROVED: two-layer propagation w/√(S_out·S_in) does not change when all weights scale together. So growth and decay act through the learned : birth ratio and through how concentrated the weights are, not through raw size.
- Evidence:
  - The arithmetic above.
  - X3, γ per 200 ticks: 0.95 → 2k-tick horizon; 0.99 → 10k; 0.999 → saturated, with fresh recall only 0.3–0.47. Forgetting is necessary.
  - At L* (old-code values) the concave factor for a well-used cortical link is already about 8 %. So the 1 % floor is a backstop that is rarely reached.
- Verdict: **ADOPT** the method; **TEST** the values in the X1c rerun.

**4. The two correction mechanisms with no new part** (MR §10; NR §3; 02 §5.3)
- Part: §二.8 "改正有两种机制". Answers §四.7 and covers IDEAS #22 and #24.2.
- Change:
  - (a) "Strong means hard to change" stays strictly per synapse, through that synapse's own w: the concave factor applies to both increases and decreases, with the ~1 % floor. Because w grows only when A→B is confirmed, this already is MR's recommended "confirmed-use counting". A new answer's synapse starts weak, so it learns at full rate in the same context. Never apply it per neuron or per input unit; v2's per-unit "maturity" brought no net gain.
  - (b) The "fast connections" are the hippocampus. Hippocampal recall through the areas expresses a correction at once. The adult's pleasure (an innate value) amplifies the just-made changes through the existing reward tool. Repeated recall then writes the correction into the cortex.
- Evidence:
  - MR PROVED that a per-synapse rate factor only rescales time; the fixed point stays the same. In MR's toy, the designer's |w| + use rule survived 75 presentations against 10, but correction latency rose from 2 to 7.
  - NR toy: rigidity alone corrected 0 of 20 changed answers after 8 re-teachings. A synapse can tell a correction from a wrong lesson only by salience or repetition; here that means reward plus hippocampus.
  - Roxin & Fusi 2013: the gain comes from transfer from the fast store to the slow one (theory, medium).
- Verdict: **ADOPT** this reading. **TEST** it with X2's correction measure: change 4 answers and count the re-teachings needed to switch, with and without the hippocampus.

**5. The hippocampus carries the question across the pause and teaches the cortex Q→A** (03 R2, R5, §5.4, exp4)
- Part: §二.6. Answers §四.1 and §四.13.
- Change (no new wire):
  - While the PFC holds Q, its return chains re-light Q's auditory and concept neurons (the inner voice). Their chains already feed the hippocampus.
  - So the hippocampal units still on when A arrives are specific to Q, and they link to A. Nothing changes during the pause, so nothing else is learned.
  - Later, cued recall puts Q and A back to back in the areas, so the PFC's top→L1 links learn Q→A with no gap.
- Evidence:
  - 03 exp4: an idealised sentence-specific input at chain strength raised correct recall from 13/40 to 40/40. If it is too strong, recall sticks at the sentence start, which matches the designer's rule that the return must be weaker than the next-step link.
  - Brain: Ito 2015 (PFC → reuniens → CA1); Place 2016; Suh 2011 (entorhinal layer III needed to bridge gaps). Medium–strong.
  - X1c B_hip: recall never fired, so the hippocampal return strength needs calibration.
- Verdict: **TEST.** Use the X3 harness, put 12 silent ticks between a Q line and an A line, and supply the held Q as inner-voice input. Measure A at the auditory neurons from a Q cue, then whether PFC→auditory links learn Q→A from recall alone.

**6. Encoding against recall by a strength ratio, not a mode switch** (03 §6 #4, §8.2 H2)
- Part: §二.6. It removes an X3 shortcut.
- Change: X3 switched learned links off during encoding; leaving them on cut fresh recall to 0.08. The pure-neuron version: chain drive greater than the maximum learned recurrent drive while input is present, and learned drive above threshold in silence.
- Evidence:
  - Old code: "the first sentence hijacks the later ones", fixed by raising the chain strength to 3.0.
  - Acetylcholine suppresses recurrent input much more than external input (89 % vs 40 %; Hasselmo & Schnell 1994, medium).
- Verdict: **TEST.** Rerun X3 with learned links always active and sweep the chain strength; the target is fresh recall ≥0.9.

**7. Replay without sleep = recall cued by the current thought, through the areas** (03 R5, §3.5; NR #2 and §4; X3)
- Part: §二.6 (repeated recall, no sleep); IDEAS #23.
- Change: no random replay generator and no "useful/useless" filter.
  - In quiet times the inner voice or imagery cues the hippocampus. Its recall re-lights the areas, the thought moves, and that gives the next cue.
  - Episodes that were rewarded are stronger, so they come back more often (the existing reward tool).
  - Guard: recalled content that the current thought did not cue teaches "this thought → unrelated content".
- Evidence:
  - NR toy: random kicks reached a stored episode 0/30 times; a 30 % cue did 30/30.
  - X3: only 11–22 % of random kicks reached real heard sequences (11 % probe bigrams, 22 % world bigrams); most of the rest were chimeras. Cued recall was 1.00.
  - NR #1: replay learned in an unrelated live context collapsed v3's core from 0.99 to 0.00.
  - v2/tune: quiet awake replay worked as well as interleaving (0.64 vs 0.66), but that used a record store.
  - Brain: Jadhav 2012; Buch 2021.
- Verdict: **ADOPT** the reading. **TEST** the chimera rate, the wrong pairings learned per 1,000 quiet ticks, and speech onsets driven by recall (must stay 0; §四.8).

**8. Adults learn fast through the hippocampus because it stays undiluted** (03 R5; X4b)
- Part: §四.13.
- Change: none, only a check. X4b traced slow adult learning to two-layer dilution: frequent units have large S_in and S_out, so a new link gets a small share. The hippocampus's fast decay keeps its sums small, so it learns like a child at any age. Size its horizon to the time consolidation takes (a few recalls), not to a lifetime of episodes.
- Evidence: X4b: a parent needs about 9 repetitions for the other parent's lessons, a fresh birth 2.3. X3's horizon is 10–20k ticks (about 50–100 lines).
- Verdict: **TEST.** X4b harness plus the X3 hippocampus: adults' repetitions to correct, with and without recall.

**9. Two-layer weights are the brain's fast compensation: compute them every tick** (02 §3.1–3.2, §1.6)
- Change: keep S_out and S_in as running sums. No periodic per-neuron budget.
- Evidence:
  - Zenke 2013/2017: compensation slower than seconds to minutes leads to oscillation or saturation (theory).
  - SORN without normalisation develops seizure-like bursts.
  - Synaptic normalisation within the same event in living animals: El-Boustani 2018; Oh 2015 (strong).
  - X3: plain weights 0.0 recall; two-layer weights work.
- Verdict: **ADOPT.**

**10. Hippocampal sizing under whole-model scaling** (NR #2; MR #4; 03 §1.2)
- Part: §二.12. Answers §四.9.
- Change: the hippocampal fan-in is the one fan-in that grows with s, so each unit gets about 10–30 active learned inputs. The active count stays roughly fixed.
- Evidence:
  - NR/MR toys: capacity ≈ 0.1/f², flat in N.
  - X3 partly disagrees: its horizon grew ∝ s, and C = 3,100 gave 20k.
  - Cost (X5): the hippocampus already takes about half of each tick, and s = 4 with 4× fan-in does not fit at batch 64.
- Verdict: **TEST.** One X3 sweep (k and C, fixed vs ∝ s) settles the disagreement.

**11. Reward window ≈ 1–2 s** (02 §1.7, §2.6)
- Part: §二.8, reward as a tool.
- Change: "the connections that were just active together" = changes made in the last ~15–30 ticks, fading.
- Evidence: Yagishita 2014 (dopamine 0.3–2 s after pairing, strong); He 2015.
- Verdict: **ADOPT** (a parameter).

**12. A fast part inside cortical synapses, only if #4 fails** (02 3-2 and §8.5; NR §3)
- Change: w = birth + F + S. Learning writes F, and the one decay acts on F. F moves into S when the same synapse is strengthened repeatedly, or at an innate-reward moment.
- Evidence:
  - NR tag + capture: changed answers right 1.0; still right later 0.85/0.75; but old answers kept only 0.68/0.49.
  - MR: the diffusive two-variable Benna–Fusi form was worst (survived 1–3 presentations).
  - Frey & Morris 1997 (strong).
  - Cost: one more state value per synapse.
- Verdict: **TEST**, as the second line.

**13. A wider learning window in the hippocampus only** (02 §2.6; 03 §2.4)
- Change: the window width is a per-area coefficient. A window of 1–2 s (BTSP-like) is an alternative way to bridge the pause in #5.
- Evidence: Bittner 2017 and Milstein 2021 (strong, CA1). In X1c, a longer window in the PFC brought hub predictions back, so keep it out of the PFC.
- Verdict: **TEST** with the same pause test as #5 (window 2 vs 15 ticks).

**14. Split/delete** (02 F4, F11, §8.12; X3, X4)
- Change:
  - After a split, the slow gain (#2) pulls the children's usage back, so a common input such as silence does not double its active count.
  - In the PFC middle layers (§四.11), splitting over-used units gives the selective units that associations need.
  - Check that siblings' codes still agree before merging.
- Evidence:
  - Reed 2011; Ganguli & Simoncelli.
  - X3: 3–5 % split/delete leaves recall from a stored cue unchanged; input cues lose 0.09–0.18.
  - X4: 3 % different units → PFC codes about 60 % different.
- Verdict: **TEST.** Measure the active count of the most common input before and after a split.

**15. Smaller items**
- Vision: "strong" = local contrast, and the active count saturates (Yeh 2009; Carandini & Heeger 2012). **ADOPT** when vision is built.
- The designer's learned-partner inhibition form has brain support (Vogels 2011). §四.3, **TEST** as already planned.
- Diagnostics, **ADOPT**:
  - a hub map: learned weight per unit as a source and as a target;
  - a genes-only control lineage;
  - a speaking-rate term in the breeding score, so silence doesn't win.

## B. REJECT
- Short-term depression on synapses, PFC facilitation, cell adaptation (02 5-2; 03 §8.1): fatigue/refractoriness (§三).
- Error-driven weakening (02 5-5) and feedback alignment (MR #7): error-correction learning.
- A soft upper bound and weakening birth links below their birth value (02 3-4): these contradict "no upper bound" and "birth links not decayed".
- Heterosynaptic depression as a new rule term (02 5-3): it is a new rule, and the two-layer weights already cover it.
- A sleep phase (02 3-5): the designer wants no sleep.
- A surprise gain on all learning rates, or a novelty comparator (02 §9; 03 §2.4): a new broadcast signal.
- Drifting time context or time cells (03 R2, §3.3).
- A direct hippocampus → PFC line (03 R3): it contradicts §二.6.
- Salience-weighted writes and uniform random replay (NR #2).
- μ, the replay switch after 12 silent ticks, and teaching-only replay (NR #1): a v3 module and a timer.
- Capture driven by a critic's prediction error (NR §3): a separate value module.
- A dense bit-matrix CA3 (MR #4): its binary weights replace concave growth.
- Activation alignment before merging: X4 showed it has no effect.
- Neurogenesis, a sharp-wave-ripple generator, event-boundary rules (03 R6).

## C. Ideas that replaced the designer's design (do not repeat)
- NR and MR reviewed ARCH_v3_core: LATEST × age time cells, μ, the M4 Go/NoGo gate with a critic, the P5b reward, the MIX layer, NLMS readouts, and a record-store CA3. Their fixes belong to that design. NR's "the anchor is the cure" maps onto §五 in the designer's frame.
- 02 and 03 assumed fixed top-k quotas everywhere and a CA3 made of 1:1 copies.
- v2's record-store hippocampus with uniform replay.

## summary_zh
1. 旧调研里有不少能直接用在您的架构上，不用换架构。最有用的三条：一是前额叶里一直亮着的神经元只算一次"先亮"，不算"再亮一次"，否则保持住的问题永远连不上后来的答案（这是 X1c 答对率为 0 的直接原因）；二是前额叶每个神经元按自己长期收到的信号慢慢调增益，让不同问题走不同的路（小实验里"通用状态"从 0.23 降到 0.04）；三是用凹形增长和整体降权的平衡点公式直接算参数，不让学到的连接把前额叶推满。
2. 改正的两种机制不用加新东西。每个突触按自己的权重变难改，这本身就等于"被证实的使用次数"；新答案的突触从弱开始，学得快。新学的东西先落在海马体上，大人高兴时由奖励放大，再靠反复回忆写进皮层。只靠"越强越难改"会把错的也锁住（小实验里改动的 20 个答案一个也没改过来）。
3. 海马体：前额叶保持住问题时，经回程（内心的声音）让对应的感觉区神经元亮着。它们本来就有链式神经元通到海马体，所以答案来的时候，海马体还知道是哪个问题。以后回忆时问题和答案紧挨着出现，皮层就能学到"这个问题答这个词"。
4. 大人学得慢，是因为皮层里常用单元的连接被摊薄了。海马体降权快、不被摊薄，所以大人靠海马体仍然学得快。
5. 回忆不需要睡眠，也不需要随机回放：由当前的想法提示，经各区想起来，再推动想法往前走。随机回放几乎回不到真实经历（0/30）。
6. 有真实输入时，链式输入要强过学到的回返连接；安静时由回返连接接管回忆。这靠强度比例实现，不需要模式开关。
7. 整体放大时，海马体的连接数要跟着放大，这是唯一要随规模增长的扇入，具体用 X3 扫一次确定。
8. 不采用：短时抑制、细胞适应（属于疲劳）、误差修正、软上限、睡眠期、时间漂移和时间细胞、海马直连前额叶、按重要性写入，以及 v3 的 μ、基底节和评价模块。
