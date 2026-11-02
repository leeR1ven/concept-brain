(Saved by the coordinator from the reader's final hand-back.)

## Harvest from 00_synthesis, 01_pfc_cortex and 05_inhibition_attention (FRAMEWORK, STATE and DEPTH skimmed): what can refine ARCHITECTURE.md

**Context.** These documents were written for the old PFC: receiving layer + thinking layer + thalamus (STATE.md, code 004031a). Many of their proposals target parts that no longer exist. Below I list only what carries over to the designer's layered PFC.

Our own numbers come from:
- X1c (`v3/x1c/REPORT.md`, `core3.py`, which I read for the rule and the inhibition);
- X3 (`v3/x345/REPORT.md`, `x3_hippo.py`);
- the old full model (STATE.md §4).

Read only; no runs.

### Ranked by expected value to the designer's architecture

**1. Held PFC units must end up strengthening their links to what comes next (the rule implemented as the designer corrected it)**
- **Source:** 00 §5 row 1 and §3 B3; 01 §1.5 and §8 #3; STATE §4.7.
- **Refines:** §二.8 (the 11-01 correction). Answers §四.1.
- **Change:**
  - In the PFC, A→B weakens only when A actually rises after B, in true B-then-A order.
  - A merely staying on while B rises or falls does not count as "lighting again".
  - X1c's `_rule` still weakens A→B for every unit held across B's fall, with the same η as the rise term. A question unit held through an answer syllable therefore nets about 0. This is exactly the cancellation the 11-01 correction removes.
  - "Always-on units don't learn" stays only for shallow sensory neurons.
- **Evidence:**
  - The old model measured the same cancellation: the held-thought→target link averaged 0.0045 (max 0.06) after 1,000 dialogues.
  - X1c diagnosis §5.1 points to the same cause.
  - Brain: with sustained co-firing, strengthening wins and the reverse-order weakening disappears at high rates (Sjöström 2001). Like-to-like links grow with experience (Ko 2011, 2013).
  - Strength: strong (code logic plus a measured effect).
- **Risk:** hubs, because held units link to everything that follows (X1c pilot (a) brought hub predictions back). The first defence is the existing two-layer weights and concave growth; the second is item 6.
- **Verdict: ADOPT.** Smallest test: the X1c world, B_loop arm, with only this change. Measure learned weight from question-held top units to the correct answer's L1 copies against other answers' copies, then first-syllable accuracy.

**2. Drivers set how many neurons fire, modulators choose which: balanced (zero-mean) loop and prediction drive, applied in the same tick**
- **Source:** 00 §2.2, which quotes the designer's own old principle "外来电流只改变哪些亮，不改变亮多少"; 05 §3.1 and §4.2–4.5.
- **Refines:** §二.4 (how the thinking range is maintained). Answers §四.10.
- **Change:**
  - In each PFC layer, or each local patch, subtract the patch mean of the top→L1 prediction drive and of the loop/lateral drive within the same tick, before each neuron's own threshold.
  - Learned links can then reshape which neurons fire but cannot add to how many.
  - This is exactly X3's hippocampal gain control (`d += rho*(r − r.mean())`; ARCH §五 "总激活越多越难再往上推"), so one mechanism serves both regions.
  - In neuron form it is the least inhibition possible: one non-learning inhibitory unit per patch, reading the patch's total drive. The mean subtraction is its program equivalent, to be registered under IDEAS #21.
- **Why growth/decay parameters alone cannot do it:**
  - Keeping the count roughly constant from speech (many chain inputs) to silence (about 0) needs something that reads the layer's total activity.
  - A threshold computed from a neuron's own in-weights never sees the total. So a fixed loop gain either dies when input stops or runs away once predictions add on top of the input.
  - X1c shows both: C dies in silence at birth, then sits at 398 of 400 units, saturated 99 % of ticks. C2 is empty 62 % of ticks and saturated 30 %.
- **A code finding in X1c arm A:** its local inhibition is computed from this tick's spikes and applied on the next tick (`inh2` is updated after `l2` is decided). A one-tick lag in feedback is the textbook cause of bursts, likely its bursts up to 370 of 400. Brain PV inhibition acts within about 1 ms, well inside one 75 ms tick.
- **Evidence:**
  - X3: the runaway (code sizes 5 → 128) was fixed, and fresh recall went from 0.8 to 1.0.
  - Cortex is inhibition-stabilised; without feedback inhibition it runs away (Sanzeni 2020; Haider 2013).
  - PV inhibition is a dense, non-specific blanket: 71–92 % of pyramidal cells within 100 µm (Packer & Yuste 2011).
  - PFC pyramidal firing stays constant with engagement; what changes is fast-spiking inhibition and connectivity (Schmitt 2017).
  - Strength: strong.
- **Verdict: TEST.** It is the first item on the designer's own §四.10 fallback list, in its smallest form.
  - Smallest test: rerun X1c C2 with same-tick mean-subtracted drive.
  - Pass if there are no empty or saturated ticks, the count stays within ±50 % of the input count, and the thought still moves (stuck20 < 0.6).

**3. Choose loop strength for reproducibility, not for motion; silence already gates the loop**
- **Source:** 01 §0.1, §1.4, §3.2, §4.1; 05 §6.4; 00 §3 B5–B6.
- **Refines:** §二.4 (loop) and §二.5 (return strength). Answers §四.6.
- **Change, three points:**
  - **(a)** Chains are drivers; the loop and the predictions are modulators. While a line is being heard, internal L1 activity must push L2 less than the heard copies do. X1c's "internal activity at half strength" is the right direction. Then the state at the end of the question is set by the question, not by history.
  - **(b)** "Hold during the pause" needs no gate. Always-on and silence neurons have no chains (§二.3), so in a pause L1 gets almost no chain input and the loop takes over by itself. With item 2, even a weak loop keeps the count alive.
  - **(c)** A strong loop amplifies whatever is most stable across ticks, which is the part common to all questions, and so produces one generic state. Choose the gain where same-question similarity across different contexts is much higher than different-question similarity and the thought still moves. Do not choose the most-moving value.
- **Evidence:**
  - X1c loop tuning: gain 0 gives stuck20 0.59; gain 8 (chosen as most-moving) gives 0.29 but probe garble 1.00.
  - X1c §4: question-end states are frame-specific only (0.42–0.61 for the same question against 0.33–0.52 for another filler in the same frame).
  - Old model: an always-on loop gave a content-free state, decoding at chance. A loop that worked only when input was absent decoded 0.90 at 500 dialogues.
  - After always-on neurons are muted, silence sends about 3 chain neurons against about 718 for one character (05 §1.1).
  - Brain: the MD thalamus is needed during the delay, not during encoding, and amplifies existing cortical structure (Schmitt 2017; Bolkan 2017). Raising overall PFC excitability by 8 % made cells respond to both rules, losing specificity (Schmitt 2017). Weak distractors don't move an attractor; strong input replaces it (Brunel & Wang 2001).
  - Strength: strong for the principle; the value still has to be found.
- **Verdict: TEST.** This is the parameter scan §四.6 asks for, using the protocol in item 4.

**4. Measure hold, motion and reproducibility the way the brain encodes them**
- **Source:** 01 §4.4 and §8 #5; 00 §6 D0.7; 05 §9.4; STATE §4.5.
- **Refines:** §二.4 ("思考范围慢慢移动"), the §三 row "每走一圈经过的神经元都不一样", IDEAS #23, §四.6.
- **Change (test design only):**
  - **(i) Reproducibility:** present each of N questions after two different preceding lines, and decode which question it was from the top-layer trajectory (chance = 1/N). Do not use Jaccard on the same neurons: a moving thought changes neurons but keeps the information.
  - **(ii) Update:** A → pause → B must decode as B.
  - **(iii) Distractor:** 1–2 ticks of noise or own voice in the pause. Allow brief entry; require recovery. PFC encodes the distractor briefly, then recovers the target (Jacob & Nieder 2014).
  - **(iv) Loop period:** the L_last→L1→…→L_last loop spans several ticks, and synchronous updates split a thought into interleaved phases (old model: Jaccard 0.06 / 0.82 / 0.06 / 0.47). Compare windows one loop long, and keep top-layer lateral links so the phases mix.
- **Evidence:** dynamic code with stable information (Murray 2017; Spaak 2017; Stokes 2013); mPFC delay activity is a relay sequence of different neurons (Bolkan 2017). Strength: strong.
- **Verdict: ADOPT.**

**5. Convergent thresholds: a neuron fires only when several of its inputs are on together (deep sensory layers, PFC middle layers)**
- **Source:** 05 §1.1, §4.4, §4.5, §6.4; 00 §4.2.
- **Refines:** §二.1 (sensory sparsity via a fraction of the in-weight sum that rises with depth), §二.4 ("几个无关的神经元亮一下，不会改变整个范围"), §二.2.
- **Change:** from sensory layer 2 on, and in the PFC middle layers, give each neuron about 4–7 comparable birth in-weights from its neighbourhood, not one dominant line. The designer's fraction-of-in-weight threshold then becomes a coincidence test, which gives three things:
  - **(a)** Counts vary with how rich the signal is: a rich sound satisfies many input combinations, silence few. This is the designer's "信号强弱 = 激活多少".
  - **(b)** Different sounds overlap less.
  - **(c)** A few scattered irrelevant L1 units cannot recruit L2, while a coherent new line can. That gives a distractor-robust thinking range plus quick updates, with no gate.
- **Evidence:**
  - A numpy replica of our 3D net: one-input neurons keep overlap at 0.53–0.55 at any sparsity, while 7 equal inputs + top 2 % give 0.24. With a single dominant input and an input layer that is always half on, no threshold can change the counts (05 §4.5).
  - Brain: fruit-fly Kenyon cells take about 6 inputs and keep the top 5 % (Dasgupta 2017). Removing feedback inhibition makes the code denser and hurts fine discrimination (Lin 2014).
  - Strength: strong.
- **Verdict: TEST.** Cheap numpy check: counts per layer for silence, one character and a sentence, plus overlap, for 1, 4 and 7 inputs. Confirm that similar inputs stay similar (local3d measured 0.989).

**6. One-sided, slow threshold adjustment (only over-active neurons)**
- **Source:** 00 §2.1 (1c) and §3 B9; 05 §3.3 and §11 suggestion 2.
- **Refines:** §四.10, the designer's second fallback ("按长期活跃程度自我调整门槛"); the anti-hub backup for item 1.
- **Change:**
  - Each neuron's threshold rises only by its excess over the layer's typical rate, with τ ≈ 1,000–3,000 ticks. That is far longer than one thought, so it is not fatigue.
  - It never falls below the birth value.
- **Evidence:**
  - X1c problem #6: two-sided, unbounded adjustment let silent units fire on anything and collapsed all questions into one state (Jaccard 0.92). The one-sided form excludes that failure by construction.
  - Old-model replica: cross-sentence overlap fell from 0.226 to 0.040.
  - Brain: firing-rate set points (Hengen 2013; Turrigiano 1998); without adaptive thresholds a few units monopolise competitive learning (Diehl & Cook 2015).
  - Strength: strong.
- **Verdict: TEST**, and only if hubs appear after item 1.

**7. If inhibition is needed, use local patches, not one pool per layer**
- **Source:** 05 §4.3 point 6 and §2.2.
- **Refines:** §四.10, IDEAS #23.
- **Change:** use patches of a few dozen neurons, so a loud input in one patch cannot take the whole layer's share.
- **Evidence:** the inhibition radius is 100–200 µm (Packer & Yuste 2011; Fino & Yuste 2011). Strength: moderate.
- **Verdict: TEST**, together with item 2.

**8. "兴奋度 = 上下文长度" means the loop/return weight relative to the chain, not a global threshold**
- **Source:** 05 §4.3 point 2 and §12.7.
- **Refines:** §二.4, the §二.5 rational/instinct knob, IDEAS #15.
- **Change:**
  - In a PFC whose count is controlled, raising everyone's excitability does not change which neurons win.
  - So the context-length knob is the weight of internal drive against heard drive.
  - Higher keeps more history and makes paths less reproducible (the item 3 trade-off).
- **Evidence:** the logic of count control; attention multiplies goal-relevant input by about 2–3 before competition (Ni 2012). Strength: moderate.
- **Verdict: ADOPT.**

**9. Concept area: convergent, sparse and learned; PFC predictions teach "same context → similar code"**
- **Source:** 01 §6 and §9; 05 §4.4.
- **Refines:** §二.2. Answers §四.14.
- **Change:**
  - Concept neurons get local random fan-in from several areas (several inputs each, as in item 5), about 5–10 % active.
  - Unlike the PFC middle layers, they learn with the one rule.
  - Through the existing return chains, the PFC's prediction ("after 吃 …") lights concept neurons before the word arrives. The arriving word's concept neurons rise together with them and their mutual links strengthen. Words used in the same contexts then come to share neurons.
- **Evidence:**
  - v2's meaning layer raised accuracy on untaught items from 0.50 to 1.00.
  - Random mixing produces mixed selectivity, with an optimal coding level around 0.1 (Barak 2013; Rigotti 2013).
  - Sparse expansion helps only if the synapses learn the input's clusters (Babadi & Sompolinsky 2014).
  - Strength: moderate.
- **Verdict: TEST**, after §四.1 works. Use 4 frames × 8 fillers: compare concept-code overlap for fillers sharing a frame against those that don't, and check that held-out fillers are answered by class.

**10. The reward window for "刚才"**
- **Source:** 05 §8.1 and §8.5.
- **Refines:** §二.8 (reward as a tool).
- **Change:** amplify the changes from roughly the last 4–27 ticks (0.3–2 s). This is a parameter of the designer's existing tool, not a module. It also settles the documents' "conflict" over "多巴胺只是工具": the tool as now written is the three-factor gate those papers describe.
- **Evidence:** dopamine enlarges spines only 0.3–2 s after the glutamate input (Yagishita 2014); in cortical slices, pairing plus noradrenaline or serotonin works when given right after pairing, the effect halves at a 5 s delay and is gone at 10 s (He 2015). Strength: strong (slice data).
- **Verdict: ADOPT.**

**11. Adults learn things that fit existing knowledge quickly (a schema effect)**
- **Source:** 01 §6.
- **Refines:** §二.6 and the §二.8 two correction mechanisms. Answers §四.13.
- **Change:** test only. In the full-model test, compare how fast an adult learns a new pair that fits a known frame against an arbitrary one. Prediction: fitting pairs need far fewer exposures; arbitrary ones rely on repeated recall through the hippocampus.
- **Evidence:** with a schema, new associations learned in one trial stop depending on the hippocampus quickly (Tse 2007; Kumaran 2016). Strength: strong.
- **Verdict: TEST.**

**12. Inhibitory partners: test them last and watch for lock-up**
- **Source:** 05 §12.1.
- **Refines:** §四.3, the third fallback in §四.10.
- **Change:** keep the designer's form. When testing, also log how many neurons each partner ends up inhibiting over a life. The documents' warning: since almost every neuron is off almost all the time, partners learn to inhibit nearly everything, and the thought then blocks all new input.
- **Evidence:**
  - X1c: partners on or off made no accuracy difference; with partners off, the agent talked during the question 0.80 of probes against 0.42 with them.
  - Brain: no 1:1 partners exist; the inhibition that is learned targets what is predictable (Schneider 2018; Vogels 2011).
  - Strength: moderate.
- **Verdict: TEST (last).**

**13. Splitting must not make common inputs louder**
- **Source:** 00 §4.2 point 2 and §5 row 9.
- **Refines:** §二.9, §四.5.
- **Change:** after a split, the count for the parent's stimulus should rise and then return. If it keeps rising for background sounds, raise the children's threshold fraction.
- **Evidence:** a repeated irrelevant sound shrank the active share of layer 2/3 from 19.9 % to 6.1 % (Kato 2015); expansion is followed by a return (Reed 2011). Strength: moderate.
- **Verdict: TEST (low priority).**

**14. Vision: thermometer-code local contrast, not brightness**
- **Source:** 00 §4.2 point 1.
- **Refines:** the §二.1 example "白光多、纯黑少".
- **Change:**
  - When vision is added, the paired code carries centre–surround contrast.
  - Uniform white or black then becomes always-on, gets no chains, and is calm under the existing rule.
- **Evidence:** V1 output layers have more dark-preferring cells (Yeh, Xing & Shapley 2009). Strength: moderate to strong.
- **Verdict: TEST (later).**

**15. Already right; keep (evidence only) — ADOPT**
- **Two-layer weights:** these match heterosynaptic normalisation (Royer & Paré 2003). In X3, recall went from 0.0 to working with them.
- **Return weaker than forward:** feedback is modulatory (Sherman & Guillery 1998). In our replica, a return stronger than the "next" link left recall stuck at the sentence start.
- **Sensory count varies, PFC count stays:** both are supported (Schmitt 2017; Bolding & Franks 2017).
- **No chains for always-on shallow neurons:** this is the brain's "only changes are passed on" (silence sends about 3 chain neurons).
- **Dale's law in the motor area:** an inhibitory neuron's learned out-links must stay negative. The old code learned excitatory outputs for the inhibitory neurons (05 §1.6).
- **Fixed random PFC middle layers first (§四.11):** random mixing already gives mixed selectivity.

### Ideas in these documents that replaced the design (do not repeat)
1. The synthesis's "four mechanisms + one layout" as the organising core.
2. Short-term depression on chains and sensory outputs, plus "learning × reserve". This is fatigue (§四.6). The designer's "only just-changed neurons learn" plus "no chains for always-on neurons" already do its job.
3. Short-term facilitation as a "silent backup" in the PFC: a new synaptic clock.
4. A thalamus module (fixed groups, latch, multiplicative MD gain) and a "fresh input" unit or TRN-like gate to release it. It is a new module and an event detector, and unnecessary because silence sends no chains (item 3).
5. A GO start unit, a 3-tick beat ring, a hold loop for disinhibitory neurons, and release at boundaries. These are a timer plus a new module. The designer's competition and innate brake already cover this.
6. A direct 1:1 return from the hippocampus to the PFC, and a slowly drifting "context" fed from the PFC into the hippocampus. This conflicts with §二.6. X3 solved "which episode" with the hippocampus's own recurrence.
7. Learned negative-image inhibition at chain exits, and self-voice copy inhibition.
8. A "news" population (acetylcholine/noradrenaline-like) and an internal match signal for "说得像". These are separate value modules; the world praising near-misses (§二.11) covers this.
9. A variable "at most 8 %" count in the PFC (conflicts with §二.4); lowering chain strength to filter input; exact top-k everywhere.
10. A rule variant for one region only. Keep one rule; the needed change is the designer's own 11-01 correction (item 1).
11. Dropping the partners outright, and replacing splitting with splitting triggered by the threshold adjustment. Partners stay as the last test; splitting stays the slow structural process.

### summary_zh
1. 这三份调研原本针对旧前额叶（接收层、思考层、丘脑），能直接沿用到设计者分层前额叶的，主要是几条"原理和参数"，不是新模块。
2. 最有价值的一条：X1c 代码里，前额叶被保持的神经元在答案"灭"的时候被减弱，正好抵消了"加强"。按设计者 11-01 的更正，应该只在真的"后亮"时才减弱。这直接对着待定第 1 条。
3. 控制前额叶亮多少：只调权重参数做不到，因为每个神经元的门槛看不到全区总量。最省的做法是和 X3 海马体同一种增益控制，即同一拍里扣掉回路和预测输入的平均值。它相当于每片一个不学习的抑制神经元，而且要当拍生效，不要晚一拍（X1c 局部抑制晚了一拍，很可能就是爆发的原因）。
4. 同一问题走同一路：链是"驱动"，回路是"调制"，回路要弱于链；挑参数看的是"可重复"，不是"动得最多"。安静时没有链输入，回路自然接管，不需要闸门。
5. 测法：同一问题放在不同前文之后，用解码判断是哪个问题；再测"更新"和"干扰后恢复"（允许干扰短暂进来）。
6. 感觉区和前额叶中间层：每个神经元接几根差不多强的输入，用设计者的"入边权重和的比例"门槛，就成了组合检测。这样亮多少会随信号丰富程度变化，几个无关的神经元也带不动整个思考范围。
7. 其次是：只罚过度活跃的慢门槛（防"枢纽"）；抽象概念区用汇聚、稀疏、会学习的形式，靠前额叶预测加同时亮加强，形成"上下文相同，编码相近"；奖励作用于约 4–27 拍内的变化；抑制伙伴放到最后再测。
8. 不要再用：疲劳式短时抑制、丘脑模块、"新鲜"闸门、GO 和节拍、负像抑制、独立的评价或调质模块。它们换掉了设计者的设计，或者是在打补丁。