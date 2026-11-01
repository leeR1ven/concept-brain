(Saved by the coordinator from the reader's final hand-back.)

# Harvest: ideas from earlier rounds that improve the designer's architecture without replacing it

I read ARCHITECTURE.md in full, IDEAS #19–24 and all the assigned documents. To check numbers I also looked at `v3/x1c/REPORT.md`, `v3/x345/REPORT.md` and `docs/review/skeptic.md`. Base path: `docs/`. Nothing was run and nothing was modified.

**How to read the list.** For question → answer to be learnable with Hebbian learning, three things must hold:
- the neurons that hold the question can link to the answer that comes later (#1);
- the same question leads to the same held state each time (#2);
- the world shows Q → A, and own speech is in the input (§B).

X1c fails the first two.

## A. Refinements of existing parts, ranked by expected value

**1. Make the one rule exact for held neurons** (review/SYNTHESIS A.5, C.1(a); review/skeptic.md S5; ARCH_v1 §3)
- **Part:** §二.8. Answers §四.1.
- **Change:**
  - A→B grows when B rises while A is on, whether A just rose, is held, or just fell.
  - A→B weakens only when A itself rises after B was on.
  - A held neuron does not count as "firing again" on the weakening side, and a target's fall weakens nothing.
  - Every update still involves one neuron that just changed, so "只有刚变化的神经元参与" still holds.
  - Why it matters: X1c used plastic2's fall clause ("units held across j's fall weaken →j"). That clause can erase the Q→A gain a few ticks later, when the answer syllable ends while Q is still held.
- **Evidence:** in S5, the held-source link went 0.000 → 0.106, and ping-pong (0.005) and reverse order (0) were unchanged. X1c §5.1 shows from the rule's form that the question carriers are exactly the units the old rule excluded. Moderate.
- **Verdict:** **ADOPT**. This is the designer's 11-01 wording made exact. **TEST** at once: rerun X1c B_loop, 5 seeds, and report Q→A link specificity and accuracy on seen pairs.

**2. Tune the loop for "same question, same path", not for "most motion"** (learning v3 §1.1: fading memory, Maass et al. 2002; assembly calculus, Papadimitriou 2020 / Dabagia 2024; X1c §2, §4)
- **Part:** §二.4 loop. Answers §四.6.
- **Change:**
  - Keep the birth loop L_last→L1 contractive: gain per circuit below 1, and the loop weaker than the input while the question is heard. The state then depends on the question, not on where it started.
  - In silence, the thought moves along learned loop links of repeated sequences, not through birth chaos.
  - Pick g_loop and input gain by a pre-registered score: trajectory similarity across repeats of the same question, minus similarity across different questions, with stuck20 kept below a bound.
- **Evidence:** X1c picked g = 8 because it moved the most, and it also garbled the most (probe garble 1.00). A flat random pool kept no trace of the question after 10 ticks (Jaccard ≈ 0). The theory is cited, not tested here.
- **Verdict:** **TEST**, a sweep on the X1c harness. Pass if the score rises with repetitions of the same question.

**3. Concave on the way down too** (review2/learning §1e, #4: Fusi 2005, Benna & Fusi 2016; review2/skeptic 3.4; IDEAS #22 "越高越难上升，也越难下降")
- **Part:** §二.8 floor and decay; §四.4, §四.7. r2b's claim that "concave growth is #22" covers only growth.
- **Change:**
  - Divide the reverse-order weakening and the global decay by the same (1+w/w_ref)^pow factor as growth, keeping the 1 % floor.
  - Together with the hippocampus as the fast store, this gives the designer's two correction mechanisms with no new variable.
  - Add a separate fast synaptic component only if this pair fails the correction test.
- **Evidence:** v2's NLMS rule loses an answer in about 3 contradictions (0.9 → 0.58 → 0.38). Under the designer's rule, changing an answer took 2–4× more re-teachings (r2b `hebb_toy`). Weak to moderate.
- **Verdict:** **TEST**, as an X1d arm. Measure retention after 10k untouched ticks and the number of re-teachings needed to change an answer (#22 asks for both).

**4. Reward scales only the last moment's Hebbian increments, signed by innate value** (review/SYNTHESIS C.1(e); neuro v3 §1.2, Yagishita 2014; r2b §3.2, born-wired's "this moment counts"; skeptic v3 #7)
- **Part:** §二.8 reward, §二.10.
- **Change:**
  - Pleased or fed: amplify the increments made in the last few ticks (the designer's "刚才").
  - Upset: reverse those increments, and grow links from the active release context to the motor inhibitory neurons.
  - No baseline and no critic. The short window exists only on this reward path; a trace on every tick diverges.
- **Evidence:**
  - v2 had positive-only reward: 626 of 1,337 self-started 你好 went to an empty room, with no extinction (review2 neuro P1/P2).
  - X1c with innate values only: empty-room speech 0.48 → 0.02 per 100 ticks (arm B).
  - Moderate.
- **Verdict:** **ADOPT** the form. **TEST** empty-room speech with upset plus global decay only.

**5. The innate brake reads only the input layer, return chains never reach it, and the brake also yields** (ARCH_v1 §3 BD-F1 and §6.4; r2b §3.3, msg 119; skeptic v3 #3)
- **Part:** §二.7, §二.5. Answers §四.8.
- **Change:**
  - The "someone else is talking" drive onto the inhibitory neurons comes from the sensor-set input layer.
  - Return chains end above the input layer, so inner speech can neither fake nor mask a voice.
  - The same innate link also acts while the agent is speaking: when another voice rises, the agent stops at the next syllable.
- **Evidence:** in v1, imagery made the cortex "hear" speech in silence, so the model never spoke; this was the root cause. In X1c, failure to yield was 0.00 in every arm (PROVED). Strong.
- **Verdict:** **ADOPT**.

**6. Babbling happens inside the muscle-vs-inhibition competition** (skeptic v3 #3; r2b develop 1; audit R5)
- **Part:** §二.7 咿呀学语, §四.8, §二.12.
- **Change:**
  - Babbling is random spontaneous muscle-neuron activity that must still beat the inhibitory neurons. It fades on its own wherever inhibitory links have grown, with no schedule.
  - Generation 0 gets the sound → muscle basics hand-wired, and descendants inherit them.
- **Evidence (PROVED arithmetic):**
  - Babble that bypasses the competition gives about 5,700 onsets after maturity.
  - Online exhaustive babbling costs about 108k ticks, about 11 % of a life.
  - A random babble hits a lexicon character with p ≈ 3·10⁻⁵.
- **Verdict:** **ADOPT**.

**7. Recall reaches speech through the return chains, with no copy gates** (review2/neuro P6; review2/skeptic §2.1; learning v3 §2)
- **Part:** §二.6, §二.5. Answers §四.13.
- **Change:**
  - Nothing new is added. A recalled item lights its auditory neurons (inner hearing), then reaches the PFC through the chains and the muscles through learned links.
  - It enters the PFC only by winning the thinking-range competition (r2b [D·p]).
  - This route has never yet produced speech, so it must be tested.
- **Evidence:**
  - In v2 life, facts scored 0/381 (0/51 at 9 or more encounters) because a copy gate never learned to say the recalled answer. Forcing the gate open lifted one-shot naming from 0 to 0.60.
  - In X1c B_hip, the recall route gave 0.00 of speech onsets.
  - In X3, re-heard openings were recalled at 0.93 at the sensory level.
  - Moderate.
- **Verdict:** **TEST**. Say "我的小狗叫N" once, then ask "我的小狗叫什么" 1–10k ticks later. Score N at the auditory neurons and at the mouth, in a newborn and in an adult.

**8. Hippocampus sizing and dynamics** (review/SYNTHESIS A.7; learning v3 §7; eval §4.3; audit E5; neuro v3 §1.3)
- **Part:** §二.6, §四.9.
- **Change:**
  - sparse activity (≤ 5 %) with large fan-in (about 1,000);
  - decay set as a fraction of capacity (∝ 1/s);
  - recall starts by itself when outside input is weak (no quiet-moment timer);
  - no recall priority;
  - own speech is stored like any heard line.
- **Evidence:**
  - fan-in 96 at 10–40 % activity stores about 30–300 transitions, while ≤ 5 % activity with about 1,000 inputs stores ≥ 1,000;
  - X3: the memory horizon grows with s, and a too-slow decay fills the store;
  - v2: uniform replay beat prioritised replay, 0.62 vs 0.46.
- **Verdict:** **ADOPT** the sizing. **TEST** decay ∝ 1/s at s = 1, 2, 4.

**9. Build the concept area from existing parts** (neuro v3 §5.3; review2/neuro #4, Howard & Kahana 2002; review2/learning #3)
- **Part:** §二.2. Answers §四.14.
- **Change:**
  - The area is its own population with local random links, fed by chain copies.
  - The one temporal rule links the context neurons at time t to the next item's neurons. Items that follow the same contexts then light similar neurons.
  - Two-layer weights plus per-neuron thresholds do the base-rate division that made v2's RI layer work, with no batch recompute.
- **Evidence:** v2's meaning layer raised class held-out from 0.50 to 1.00, and the meaning cue raised recall-by-meaning from 0.70 to 1.00. The risk: v2's competitive layer collapsed, and its temporal-context code drifted (0.082 vs 0.043). Moderate.
- **Verdict:** **TEST** offline on recorded streams first (10 seeds; class held ≥ 0.95, drift ≤ 0.06). Add it to the model only after the PFC learns seen pairs, which is the designer's order.

**10. Two-layer weights computed per source area where areas converge** (neuro v3 K1/K5; audit D15)
- **Part:** §二.4 at PFC L1 and at the hippocampus.
- **Change:** compute the "divide by total" separately for each area's chain lines, so a large visual area cannot swamp the concept area.
- **Evidence:** v2's `hipnorm` is exactly this, and it gave the 0.70 → 1.00 above. Weak to moderate.
- **Verdict:** **TEST**, and only if the concept area's signal turns out to be swamped.

**11. A range for the return-strength dial** (neuro v3 §5.4; skeptic v3 #12)
- **Part:** §二.5.
- **Change:**
  - Return drive must stay below the margin between a clean match and the best rival match (about 0.19 on bench statistics, so ≤ 0.15).
  - A clearly heard new word is then never "corrected" into a familiar one, while degraded input still gets filled in.
  - Restoration then grows with learned predictions.
- **Evidence:** PROVED arithmetic. In children, restoration is absent in toddlers and present at school age (Newman 2006, 2004). Caveat: at σ = 1, a window with a margin below 0.10 can still be overridden.
- **Verdict:** **ADOPT** this range for the dial. **TEST** it with repair twins.

**12. Whole-model scaling rule plus a scale lint** (learning v3 §7; eval §4.3)
- **Part:** §二.12.
- **Change:**
  - Every count × s; active fractions, fan-in, time constants, the rule and the body stay fixed.
  - Hippocampal decay ∝ 1/s. World richness and life length grow ∝ s.
  - A lint builds the model at s = 0.5, 1 and 2 and checks these invariants.
- **Evidence:** v2's meaning layer scaled 4× alone lost 0.11; randmix 4× and 16× was flat in a fixed world; X3's horizon grows ∝ s. Prediction: along the (s, world, life) diagonal, held-out rises only if shared codes are learned. That slope is the soundness test.
- **Verdict:** **ADOPT**.

**13. Keep compute linear in s** (eval §4.3; skeptic v3 #4; X5)
- **Part:** §四.9.
- **Change:**
  - The designer's "只和位置相近的第一层神经元" gives a fixed fan-out (a dense readout would grow ∝ s²).
  - Update learning only over the neurons that just changed.
  - Checkpoint and resume long lives.
- **Evidence:** silent ticks cost 2.45 ms, mostly dense per-unit work. At s = 4 with life ∝ s, one life takes 16–32k s. X5: the hippocampus is half the cost, and the A800 runs 0.25 ms per agent-tick at B = 64.
- **Verdict:** **ADOPT**.

**14. Fixed random middle layers, sized by the width arithmetic** (learning v3 §1.3; neuro v3 §1.2, Litwin-Kumar 2017; audit D5)
- **Part:** §二.4. Answers §四.11 and §四.5.
- **Change:**
  - Keep the middle layers random and fixed, with few inputs per neuron and sparse activity. This covers frame × filler pairs.
  - Higher-order structure must come from the concept area or from thought steps, not from more width.
- **Evidence:** 10 responders per pair need about 1.1×10⁵ units, per triple 4.8×10⁶, per 4-tuple 1.5×10⁸ (PROVED). randmix's random layer with prune and regrow gave J 0.83 → 0.95.
- **Verdict:** **ADOPT** as the default. **TEST** split/delete against no structural change in the middle layers.

**15. PFC stability from weight parameters before adding inhibitors** (review/SYNTHESIS A.5; ARCH_v1 §6.10; X3; X1c)
- **Part:** §二.4 / §二.8. Answers §四.10 and §四.3.
- **Change:**
  - Apply the decay often relative to growth (every tick or every few ticks).
  - Use X3's "总激活越多越难再往上推" in the PFC too.
  - Add the designer's inhibitory partners only if runaway activity remains.
- **Evidence:**
  - Batch normalisation every 100 ticks let weights overshoot (13–17 vs a target of 2) and form super-attractors.
  - X3's gain control raised fresh recall from 0.8 to 1.0.
  - In X1c, turning partners off raised talking during the question from 0.42 to 0.80.
- **Verdict:** **TEST** in X1d arms: decay every tick vs every 100; gain control on/off; partners on/off.

**16. A depth-in-time test for §四.2** (learning v3 E2)
- **Change:** a two-level class world (X是蔬菜 / 蔬菜能吃) in which whole classes are never asked. Compare:
  - two thought steps through recall;
  - a two-layer concept area;
  - an online backprop network, only to check that the information is in the stream.
- **Verdict:** **TEST**, after #1 and #2 pass.

**17. Sealed tests and lineage statistics** (eval §2–6; review2/learning §2.1; skeptic v3 #9)
- **Part:** §二.12 新题封存, IDEAS #19.
- **Change:**
  - Freeze first: hash the code, numbers only, no booleans, one parameter file for every world. Then open each sealed set once.
  - The designer writes 3–4 toddler routines off the box (hash them now).
  - Jitter caregiver timing U(0.5×, 2×).
  - Score held-out only on frozen copies.
  - Baselines: echo, constant, n-gram, kNN and an online RNN with the same experience.
  - Add a gift audit and a staggered introduction of families.
  - Exclude hand-wired gen-0 content and every ancestor's life from the sealed sets.
  - Score descendants by exposures-to-criterion against a newborn.
  - Use the same world seeds within a generation, with ≥ 2 screening lives each.
- **Evidence:** v2's held-out score was 1.00 on frozen copies but 0.00 when the same items were asked live with no answers given. With 100 probes × 5 seeds, the smallest detectable gain is 0.12–0.15.
- **Verdict:** **ADOPT**.

## B. What v2 proved that still holds for the designer's architecture

- **Own speech must be in the input.** With self in the stream, the role toy stayed at 1.00 after 900 untutored dialogues; a mixed bag fell to 0.02. Under the designer's rule, "latest" comes free (r2b: 1.00 after 3,000).
- **The world must show Q → A.** With 300 overheard exchanges first, every variant kept 1.00. No local commit rule recovered corrections the child talked over (0.51–0.61).
- Random conjunction units generalise on frozen copies (J 0.83 → 0.95).
- Meaning codes generalise by class (0.50 → 1.00) and fix recall by meaning (0.70 → 1.00).
- Awake recall is a drift brake (C 0.57 → 0.00 without it), but only if own lines are stored.
- Uniform recall beats prioritised recall (0.62 vs 0.46).
- Scaling one module alone hurts (−0.11).
- Floor-relative thresholds fix hearing in noise.
- Surprise-gated entry into the thought hurt (0.69 → 0.15).
- On seen items, Hebbian competition is as good as NLMS (1.00 vs 0.99), with echo of 0.01 vs 0.33.
- One mechanism per ability piles up knobs: 83 mechanisms and 265 knobs by v2.2.

## C. r2b's [N] parts

- **Keep:**
  - the lineage-fixed physics seed (now [D] in §二.12);
  - "after maturity, only errors the agent causes count" (an evaluation rule).
- **REJECT:**
  - the fatigue handover as an age code (fatigue and time coding; #24.1);
  - the reward sign from a running mean (§三);
  - the "response counts if unexpected" contingency (reward taken from the response; it needs a predictor);
  - the agent-side maturity counter;
  - alignment inside the merge (X4: no gain);
  - "no meaning layer" (replaced by the designer's concept area).

## D. Ideas that replaced the designer's design (do not repeat)

- **v3 learning, the P-V-M core:** delta-rule O1, TD critic, δ/ν broadcasts, a fatigue motif, a trace bank / time cells, a BG selector, one-shot episode cells.
- **v3 neuro, K1–K7:** k-WTA with self-excitation and fatigue, a three-factor delta rule, a BG gate, DG plus episode cells, efference explain-away, ACh/NE broadcasts (event boundaries), η(age) critical periods, a theta oscillator, "no global decay".
- **Audit C-1–C-6:** a fixed-fraction k-WTA, a delta rule, a BG loop, two broadcasts.
- **The first ARCH_v3_core:** time bins and a TIME pathway, the μ replay threshold, the M4 accumulator, the "responded-to" reward.
- **Review 1:** a long-lasting hippocampus, sleep phases, an error-driven readout, BG gating, fixed slow context layers.
- **Review 2:** Xcut, the answered-line veto, contingency-routed two readouts, event segmentation by continuity match, copy routes, TD dips and a value table, cloze or prioritised replay, a learned trust rate, a separate slow readout.
- **ARCH_v1:** replaced the learning PFC loop with a fixed self-holding register, a GO timer, localist syllable programs, and learning gated by the teacher.

## summary_zh

1. 以前各轮的成果里，真正能"在您的架构上优化"、又不加新模块的，主要是下面几条。
2. 最重要的是把唯一的学习规则写准：被保持住的神经元，也能连到后来亮起的神经元；"后亮减弱"只在源神经元真的重新亮起时才算，目标熄灭不减弱。X1c 里问题连不到答案，很可能就是这个原因，要马上测。
3. 回路强度要按"同一个问题每次走同一条路"来挑，不要挑"动得最多"的那档。没有输入时，想法的流动靠学到的连接，不靠出生时的乱跳。
4. "越强越难改"也用在减弱和降权上，再加上海马体，就是您说的两种改正机制。奖励只放大"刚才"那一下的变化，好坏只看先天价值（高兴或难过）。
5. 运动区：不抢话的刹车只看输入层，回程到不了输入层；自己说话时别人一开口，刹车也会让。咿呀学语就是肌肉神经元的随机活动，同样要赢过抑制神经元。
6. 海马体要稀疏、入边多，降权速度按容量定。回忆起的东西经回程变成"心里听见"再说出来，这条路以前从没真正说出过话，要专门测一次。
7. 抽象概念区可以用现有零件搭：自己的一群神经元、局部随机连接、同一条时序赫布规则、两层权重。
8. 整体放大、算力、封存新题都有现成、可检查的做法。基底节模块、时间细胞、误差修正、疲劳交替、"和平常比"定奖励等，都属于替换，不再用。
