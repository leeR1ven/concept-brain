(Saved by the coordinator from the reader's final hand-back.)

## Harvest: motor area, babbling, speech routes, self/other voice, turn-taking, values, and the world

Documents read: research/04_speech_motor.md, review2/embodied, review2/develop, v3design/develop, v3design/review/develop. I also checked them against ARCHITECTURE.md (all of it), IDEAS #19–24 and v3/x1c/REPORT.md.

A caution before the list. 04's findings about the code come from the old `brain/` code (dis_thr 2.0, MotorArea D/I/M). In that code's rule, a source that stays on weakens its own links. So 04's main conclusion, "a held state learns nothing, so holding must be built as structure", does not carry over: ARCH §二.8 (11-01) says held PFC units still grow links. Most v3 items are written in v3-core terms (M4, P5b, TD). I translated the parts that fit the designer's architecture.

### Ranked ideas (by expected value)

**1. The learned stop targets phasic stop neurons, and inhibitory outputs stay negative.**
- Source: 04 §1.4.1, §7 P6, §9 R4, §10 prototype 3.
- Refines: §二.7 ("说完一句话就停"), §二.1, §二.3.
- Change: the motor area's inhibitory neurons come in two kinds.
  - Tonic blockers: always on, they are the default "no", and they have no chain pair (§二.3).
  - Phasic stop neurons: normally off, and they do have chain pairs. The learned "muscles first, then inhibitory neurons" stop goes onto these.
  - Learned outputs of inhibitory neurons keep a negative sign (Dale's law).
  - Detail worth testing: let the innate brake drive the phasic stop neurons. Then every hand-over gives the PFC an example of "my last syllable → stop".
- Evidence: links learned onto an always-on inhibitory neuron came out at 0.000, against 1.15–1.46 for onset links (04 prototype 3). Under any form of the rule, only neurons that rise can receive learned links. Biology: start/stop cells (Jin & Costa 2010). X1c already uses this split (its item 8: copies of the always-on blockers fed a constant signal into L1). Medium–strong.
- Verdict: **ADOPT**. Write it into §二.7.

**2. Learn the babbling map with existing parts, and drop X1c's birth-given auditory→motor stand-in.**
- Source: v3review/develop §10 (babble_toy); review2/develop §4.5; 04 §2.2 (DIVA), §3.1 (canonical babbling).
- Refines: §二.7 咿呀学语; §二.12 (first-generation hand wiring is allowed, but tests of new content must use learned links).
- The problem, worked out from the X1c rule form (my derivation):
  - A link into a muscle neuron forms only from units that were on just before that muscle rose.
  - X1c hears its own voice 1 tick after the muscles move. So a single babble teaches muscle→sound (a forward prediction).
  - It also weakens sound→muscle, because the sound is still on when the muscles fall.
  - So the direct link X1c needed cannot be learned from single babbles.
- Change: three cheap ways to get the sound→muscle direction:
  - (a) Hear the own voice in the same tick as the muscles. They then rise together, and the "同时亮" term links them both ways.
  - (b) Reduplicated canonical babbling (bababa), the infant stage at 5–10 months. The sound of one syllable comes just before the muscles of the next.
  - (c) Imitation games: the child babbles, the caregiver imitates, the child repeats. This also puts sound before the same muscles.
- Evidence:
  - babble_toy with lawful physics produced the right syllable for 0.84–0.91 of the lexicon after 300 babbles and 0.95–0.99 after 1,000, while random babbling alone bound only 0.15–0.21 of it by 1,000. That toy used a delta rule, not ours: it shows the information is there, not that our rule extracts it.
  - X1c §6 lists the stand-in as a deviation from the design.
  - Biology: Goldstein & Schwade 2008; Kuhl & Meltzoff 1996.
  - Medium.
- Verdict: **TEST**. Smallest test: the X1c body with lawful physics and the stand-in removed. Arms: same-tick hearing, reduplicated babbling, imitation game. Metric: the share of caregiver syllables the agent can then say.

**3. "0 errors after maturity" (§四.8): babbling must go through the same competition, silence must beat a wrong word, and errors must be counted inside exchanges.**
- Source: v3review/develop issues 1 and 4.
- Answers: §四.8; refines §二.11.
- Change:
  - (a) Babbling is weak random input to muscle neurons. It must beat the tonic blockers like any other drive and never bypass them. It then fades without a schedule, as the PFC learns (through upset values and the brake) to drive the stop neurons during exchanges.
  - (b) "0 wrong answers" comes from release by sharpness: a diffuse intention wins nothing, so the result is silence, not a wrong word.
  - (c) Measurement: count talking over others and wrong answers only inside exchanges with an addressed partner. Solitary vocal play is not an error (ask the designer). Define "maturity" by turns experienced (about 300 exchanges where the agent spoke and a person replied, capped by heard syllables), not by words heard.
- Evidence:
  - With a constant babble rate that bypasses the gate, about 75 extra turns come after maturity, and P(0) ≈ 2×10⁻³³ (PROVED). The rate would have to fall 37–74-fold. 64 % of random babbles are garble.
  - Learning alone cannot reach a hard 0; only structure (the brake) can.
  - X4: when merged parents conflict, the offspring mostly stay silent rather than answer wrongly.
  - Weir 1962 (crib speech is normal); Romeo 2018 (turns, not words, predict outcomes).
  - Strong (arithmetic).
- Verdict: **ADOPT** (a) and (c). **TEST** (b) and the fading. Smallest test: count spontaneous releases inside exchanges by age, from an X1c-type life log.

**4. The world package: what the world must provide.**
- Source: v3develop §2–3 (role_toy, cue_count, W1–W5); v3review/develop issues 3, 11, 12 (caregiver_toy); review2/develop §3.1(b) and §4.3.
- Refines: §二.11, §五.
- Change (world code only):
  - Self-answers (formats) fade per task as the child succeeds.
  - Corrections restate the question–answer pair; "不是X，是Y" only signs the value.
  - Caregivers move on mostly after correct answers.
  - Re-ask about 30 % and move on about 15 %, with jitter (as in X1c). Care never stops: judge stability with care continuing, and keep "untutored" only as a stress test.
  - Politeness 0.7–0.9, plus one sealed test partner who is impolite.
  - Contingent replies to babble early on; caregivers are pleased at near-misses.
  - Adults answer each other's greetings and questions within earshot.
  - Question and approval prosody, and faces.
  - Many fillers per frame (type frequency).
  - Lawful vocal physics.
  - Keep the rule free of any "predicted but absent → weaken" term, so moving on stays neutral.
- Evidence:
  - role_toy: a learner with no role rule scores 1.00 (and still 1.00 after 3,000 untutored exchanges) when the world has ≥ 30 % formats or replayed pairs. It scores 0.00 in a world with plain recasts only.
  - caregiver_toy (on the v3 learner, not ours): a realistic caregiver mix ends at 0.01 after 3,000 untutored exchanges, against 0.65 when care continues in half of them.
  - ARCH_v2's absent-prediction penalty raised silent replies from 0.17 to 0.38.
  - 87 % of questions contain a question syllable; greetings and corrections contain none.
  - Ninio & Bruner 1978; Chouinard & Clark 2003; Bybee 1995.
  - Limit: X1c, with formats, still learned nothing, so the world alone does not solve §四.1. Medium.
- Verdict: **ADOPT**.

**5. The innate brake is the content-free "when"; the PFC supplies the "what".**
- Source: 04 §4.3, §5.3, §8.2 (the learnable part of R1); review2/embodied §2.3; v3review issue 6.
- Refines: §二.7 (the competition; reply timing) and the §二.11 brake.
- Change:
  - (a) Drive the brake by the near voice's active-neuron count (signal strength, §二.1), not by any silence test. Distant crowd voices then barely brake.
  - (b) The PFC prepares the answer while the partner talks, the brake holds it back, and it is released when the partner stops ("plan early, fire late").
  - (c) Timing that does not depend on content: the partner's offset lights the same just-changed neurons after every line. So the one rule learns "partner just stopped → my motor-onset copy" for every content. This needs the offset-lit neurons to keep their chains; X1c cuts units that are on more than 60 % of the time.
  - No go neurons, no beat timer.
- Evidence:
  - Biology: a trigger unrelated to the movement in motor cortex (Kaufman 2016; Inagaki 2022); planning starts mid-question (Bögels 2015); the most common gap is 0–200 ms (Stivers 2009); infants take turns before they have content (Hilbrink 2015).
  - Ours: exact-silence triggers gave 0 heard lines at noise σ = 1. With 16–64 bodies the brake was on 33–53 % of the day.
  - X1c: failure to yield 0.00, but talking during the question 0.40–1.00 and 0.3–1.8 extra turns per exchange.
  - Medium.
- Verdict: (a) **ADOPT**. (b) and (c) **TEST**: in an X1c-type run, log the gap after the partner's end and check whether the timing transfers to a held-out frame; add a 16-voice crowd arm.

**6. Reward: contingency is already built into the designer's rule.**
- Source: 04 §3.1–3.2; v3develop S3, §2.1; v3review issue 2; review2/embodied §2.1.2.
- Refines: §二.8 奖励, §二.10.
- Change:
  - The rule stays as it is. "Amplify the changes on links that were just active together" credits only what happened right before the pleasure, so a smile that does not follow the act averages out.
  - Refinement: innate values are birth, unchangeable connections from coarse sensory features (a smile, an approval contour, crying) to the reward signal.
  - The world renders these cues imperfectly and based on truth: about 0.8 of correct answers approved, about 0.1 of wrong ones (rates ASSUMED).
- Evidence:
  - Biology: caregiver responses of the same amount only shape babble when they are contingent (Goldstein 2003, strong). Contingency speeds marmoset vocal development without changing total output (Takahashi 2017, strong).
  - Ours: a reward trace that never expired credited eating to old utterances (embodied review, PROVED).
- Verdict: **ADOPT**. Reject any "being answered = reward", including v3's (1−q̂) fix.

**7. Own and others' voices: shared content, distinct voice units.**
- Source: v3review issue 8; 04 R6 and §2.5; v3develop R3.
- Refines: §二.1, §二.7 (speech routes).
- Change:
  - Keep X1c's coding: content units shared by own and other voices, voice units distinct. Never make own-voice units disjoint, and test with multi-syllable answers.
  - Motor-return copies exist only for the agent's own actions. So a predicted partner line can reach the mouth only through inner hearing → the direct auditory→motor link. Watch that route with the route counter.
  - Optional: a learned motor→auditory prediction ending on auditory inhibitory neurons (efference suppression). It must stay partial, because own speech must enter the input (§五).
- Evidence: disjoint own-voice codes cannot predict the second syllable of the agent's own line (PROVED). X1c echo ≤ 0.02. Biology: Schneider 2018; Eliades & Wang 2008 (strong).
- Verdict: **ADOPT** the coding. **TEST** suppression only if echo or flooding by own speech appears.

**8. Whole-syllable competition in the motor area.**
- Source: 04 §0.3, §10 prototype 2, §2.3 (GODIVA).
- Refines: §二.7 (release by sharpness); applies §二.4's "rarely co-active units inhibit each other" in the motor area; this is also a §四.3 test.
- Change: babbling makes a syllable's muscles rise together, so they strengthen each other. Muscles that rarely fire together inhibit each other. Sharpness then means one whole syllable dominates, not muscles taken from several syllables.
- Evidence: gating each muscle separately caused crosstalk (我我我们们们, 04). X1c garble was 0.4–2.5 per exchange. Medium–weak.
- Verdict: **TEST**: garble per utterance with and without these learned links.

**9. Syllable pace from the force ramp; syllable hold from inner hearing.**
- Source: 04 §1.3, §2.2, §4.1.
- Refines: §二.7 (force neurons, 表征坍缩).
- Change:
  - Force-grade neurons rise in turn. The last grade's onset comes just before the next syllable, so it learns the strongest link and sets the pace. No timer.
  - The syllable is held by its 3-tick inner-hearing trajectory, which drives its muscles through the auditory→motor link, not by a hold loop.
  - Caveat: if "units already on strengthen links to the next one" applies here, the first grade (on for the whole syllable) also links to the next onset and fires it early. The motor area may need "only just-changed neurons participate".
- Evidence (04, numpy): the last grade got the strongest link to the next syllable (0.99), and the next syllable started exactly at tick 4. The first grade's link was 0.25–0.66. Medium.
- Verdict: **TEST**: syllable length in free running under the two rule forms.

**10. An innate harm brake.**
- Source: v3review issue 7; embodied §3 (change 3) and §5.
- Refines: §二.10, §二.7, IDEAS #17.
- Change: birth connections from "person upset, crying or hurt" features to the motor stop neurons. The ongoing action stops at once, without first having caused harm. The innate "upset" value then teaches the PFC. No veto module.
- Evidence: toddlers show concern without having harmed anyone (Vaish 2009); distress inhibits action directly (Blair 1995). A value learned purely from outcomes must cause harm before it can decline (PROVED). Medium.
- Verdict: **TEST** once world2 shows distress cues.

**11. Naming through the concept area (the sensory route of speech).**
- Source: review2/develop §4.2, D2, D3.
- Refines: §二.2, §四.14, §二.7 (routes).
- Change: a seen object lights the concept neurons that its spoken name also lights. These light the word's sound (inner hearing), which goes through the auditory→motor link to the mouth. No learned PFC question→answer link is needed.
- Evidence: in v2 the agent attended the object 1.00 of the time and recognised its name 0.73, but said it only 0.02, because there was no route. Medium.
- Verdict: **TEST** after §四.14: held-out "这是什么" with the concept area on and off.

**12. Two-layer weights for the motor area's learned links.**
- Source: 04 §1.1, R3 (Fiete 2010; Okubo 2015 cap both inputs and outputs).
- Refines: §二.4, §五 (X3).
- Change: use the same propagation, w/√(out·in), on learned links into and inside the motor area, so common syllables do not win every release.
- Evidence: X3 (recall was flooded without this); models that grow sequence chains need both caps. Medium.
- Verdict: **ADOPT**.

**13. Lineage: siblings in different families, and silence penalised too.**
- Source: v3review issues 4 and 9.
- Refines: §二.12, §四.12.
- Change:
  - Siblings keep the same birth wiring but live with different caregivers or curricula, so merging adds lessons instead of duplicating them.
  - Lives get longer each generation.
  - Exams penalise wrongful silence as well as errors; otherwise selection drifts toward quiet genes.
- Evidence: born-wired R18: two models that learned on non-overlapping connections merged with both skills (5/5 seeds); opposite answers to the same prompt merged to 0.000. X4: a skill only one parent had kept 0.42–0.73. One life ≈ 0.86 child-days of input.
- Verdict: **ADOPT**. **TEST** X4b with siblings taught disjoint lessons.

**14. A test for repairing misheard words.**
- Source: v3develop §1.2(c), W5, E4.
- Refines: §二.5.
- Change (test design): raise one agent with masking bursts (p 0.05 per line) and one in clean sound. Check that masked predictable syllables are restored, and that with a new object in view a new word is learned instead (Swingley 2016).
- Evidence: child data cited (strong); no run yet.
- Verdict: **ADOPT** as a test.

### Ideas that replaced the design (do not repeat)
- **04:**
  - dedicated "go" neurons onto all release neurons;
  - an innate 3-tick beat ring with a gated hold loop (a timer);
  - an HVC-like sparse sequence layer;
  - adaptation if sequences loop (refractoriness, §三);
  - "holding must be built as structure" (rests on the old rule).
- **review2/embodied:**
  - an event clock with line boundaries;
  - a goal slot with an innate gate and a 400-tick timer;
  - a separate value table;
  - a harm predictor with a veto;
  - a "simulate before acting" step;
  - a curiosity factor;
  - place cells.
- **review2/develop:**
  - the role veto (a trace plus an "answered" signal): J rose 0.70→0.93, but it is a role rule;
  - an "initiate" role key;
  - a basal-ganglia copy gate;
  - a trust learner;
  - curiosity urgency;
  - an imitation search that swaps one muscle at a time.
- **v3develop:**
  - a TD reward-error broadcast;
  - a precision broadcast;
  - "being responded to" as a reward;
  - a learning-progress curiosity reward;
  - attention gain;
  - a "who speaks next" learner;
  - time cells;
  - a fixed order of thought operations.
- **v3review/develop:**
  - its fixes to v3 parts: P5b (1−q̂), motor-precision babble scaling, TIME keys, expectancy-gated replay, discounting per event, an engaged-source selector, extra learning for addressed speech, and M4 PROGRAMS.
  - Keep only its world statistics and its arithmetic.

### summary_zh
1. 最值得马上采用：运动区抑制神经元分两种。常亮的阻断神经元管"默认不动"，没有链式神经元；平时不亮的"停止神经元"有链式神经元，"说完就停"学到它身上。抑制神经元学到的连接保持负号。
2. 咿呀学语要用现有部件学会，去掉 X1c 出生就接好的"听觉→运动"替代连接。按现在的规则，单次咿呀只学会"肌肉→声音"，反方向反而被减弱。可以试三种办法：自己的声音同一拍就听到、重复音节（bababa）、和大人互相模仿的发声游戏。
3. 长大后 0 错误：咿呀必须和别的信号一样，赢过抑制神经元才能动嘴，不能绕过。没把握时不说，而不是说错（X4 里父母答案冲突时后代大多不出声）。错误只在对话里统计；"长大"按经历过的对话轮次算。
4. 世界方面：自问自答随孩子进步逐渐减少；纠正时把问答重说一遍；大人重问、跳过，但一直在教；大人不总是礼貌；表情和语调要真实呈现，而且不完美。
5. 先天刹车按附近声音的强弱（亮了多少神经元）来刹，远处的人声几乎不刹。对方说完、刹车放开就是"什么时候说"，前额叶管"说什么"。不加计时器，也不加专门的起动神经元。
6. 设计者的奖励规则本身就只奖励"紧跟在行为后面"的好结果；文献证实回应多少不起作用，只有及时的回应才塑造发声。
7. 自己和别人的声音：内容共享、声音部分不同。不要把"自己的声音"做成完全分开的一套神经元。
8. 待测：整个音节一起竞争；音节节奏来自力度递增，靠内心的声音持续；先天"对方难过就停"的反射；经抽象概念区说出看见的东西的名字。
9. 不要再用：起动模块、节拍环、序列层、角色否决、单独的价值或好奇心模块、时间细胞、"被回应就奖励"。