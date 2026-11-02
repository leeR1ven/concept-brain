(Saved by the coordinator from the reviewer's final reply; the harness blocked the reviewer's own write.)

# Review round 2, embodied / open-world lens: is ARCH_v2 ready for many bodies in world2 and for an innate prosocial instinct?

(The harness blocked my write of `arch/v2/review2/embodied/report.md`, so this reply is the report. My probe scripts and their outputs are in `arch/v2/review2/embodied/`.)

Reviewer lens: embodied agents and open-world learning: motor babbling, drives, goals and exploration, reinforcement and intrinsic motivation, multi-agent social learning, innate values.

Labels: PROVED (argument with numbers or code), SIMULATED (my run, with file and numbers), CITED (paper and finding), ASSUMED.

My two probes:
- `silence_probe.py` → `silence_probe.out`: 9 world2 days in all, with 4 / 16 / 64 bodies.
- `noise_agent.py` → `noise_agent.out`: the real ARCH_v2 agent in world2, 2 bodies, 15k ticks, noise 0 and noise 1.

Together they took about 2 minutes of CPU. Every other number comes from existing files in `synth/out/` and `bench/world2/`.

## 0. Verdict

ARCH_v2 is a sound **listening-and-answering mind in a quiet room with one partner**. It is **not ready** for its next phase: many bodies living one life in world2, plus the designer's innate prosocial instinct (IDEAS #17). Three things block it, in this order.

1. **Its event clock assumes exact silence.** Line onset and end, the quiet moment that triggers awake recall, and the innate "other is talking" brake all compare the raw ear energy E with 0 or 10.
   - In world2 at the bench's own noise level (σ = 1), the real agent hears **0 lines, stores 0 episodes, makes 0 awake recalls and says 0 utterances** in 15k ticks. Its syllable categories fill their cap of 256 (SIMULATED, `noise_agent.out`).
   - Even without noise, 16–64 talking bodies keep the brake on for 33–53 % of the day and remove every silence of 200 ticks or more (`silence_probe.out`).
2. **It has no action loop.** There is no motor babbling, no reafference, no goal that survives movement or overheard speech, and no credit assignment beyond one utterance. Nothing drives world2's 9 motor units.
3. **It has no value system that the instinct could live in.**
   - Reward is one scalar from the world, written into the same weights that listening rewrites at η 0.35.
   - The two reward parts barely touch behaviour. Removing the reward actor or V_go gives 361–377 rewarded greetings against 376, and social replies of 0.31–0.40 against 0.35. That is within the noise of one life.
   - R6 actively extinguishes any rewarded behaviour once the reward stops.
   - So the designer's two #17 tests fail by construction (PROVED, §2.1): helping that stays stable without correction, and helping that appears with a stranger.

The rest of this report follows from these three points. One caveat runs the other way: without noise, the awake-recall trigger itself survives crowds. Fragmented silences give more 12-tick quiet moments, not fewer (§2.3).

## 1. ARCH_v2 against the real brain (embodied lens)

| difference | matters for the goal? | why |
|---|---|---|
| **No motivational system.** No hypothalamic drives. Hunger, touch and taste are never read: no code in `synth/` reads `obs['hunger']`. There are no separate value channels for food, social approval or another's distress. The one reward scalar comes from the world. | **Yes, decisive for #17.** | Drives make value depend on state. AgRP hunger neurons carry a negative-valence signal that food removes (CITED: Betley et al. 2015, Cell). Social value has its own innate circuits (amygdala, OFC). |
| **No value-based NoGo.** The BG is one accumulator plus a turn-taking brake. Candidates are never scored by value and vetoed. | **Yes.** | The direct and indirect pathways learn Go and NoGo from opposite dopamine signs (CITED: Frank 2005, J Cogn Neurosci; Collins & Frank 2014, Psych Rev). The designer's "decline what I dislike" is a NoGo. |
| **No sensorimotor loop.** No motor babbling in the body, no forward model, no cancellation of self-motion by efference copy. | **Yes, in any world with a moving body.** | Self-motion is cancelled by efference copy (CITED: von Holst & Mittelstaedt 1950; LIP remapping, Duhamel et al. 1992, Science). Forward models are learned (CITED: Wolpert, Miall & Kawato 1998). ARCH_v2 treats every view change as a new scene (agent_think.py:424, 355). |
| **No space.** No place cells, path integration or map. The hippocampus indexes heard lines. | **Yes**, for world2 memory probes such as X在哪里. | Place cells (CITED: O'Keefe & Dostrovsky 1971) and grid cells (CITED: Hafting et al. 2005) provide the "where". |
| **No auditory scene analysis.** Sources mix by max. A line is segmented by E == 0. There is no adaptation to background sound. | **Yes, immediately** (§2.3). | Auditory cortex adapts to background statistics (CITED: Rabinowitz et al. 2013, PLoS Biol). Streams are separated by pitch, timbre and location (CITED: Bregman 1990). |
| **No overt orienting.** Joint attention is an eye bit on the target cell. | **Yes, with the foveated eye.** | Following pointing raises fovea hits from 0.12 to 0.67 (`report_fovea.md`, Looker). ARCH_v2 has no orienting tendency. |
| **No social cognition.** Speaker identity is not a MIX group (core_synth.py:135–180). There is no addressee detection and no selective trust. | **Yes, for many bodies.** | Infants recognise their own name at about 4.5 months (CITED: Mandel, Jusczyk & Pisoni 1995). They prefer infant-directed speech (CITED: ManyBabies 2020). Preschoolers prefer accurate informants (CITED: Koenig & Harris 2005). |
| **Prosocial motivation comes only from outside.** Helping is rewarded by the world's praise scalar. | **Yes, for #17.** | Toddlers help without any reward (CITED: Warneken & Tomasello 2006, Science). Material rewards reduce later helping in 20-month-olds; praise does not (CITED: Warneken & Tomasello 2008, Dev Psych). The instinct must value the other's goal or state itself. |
| **No plasticity spectrum.** Learning rates: GEN η 0.35, gates 0.5, critic 0.2, actor 0.3. Only the R7 outstar slows with use. | **Yes**, for #10 ("firm, slow-changing" memories) and #17 ("hardest to change"). | Synapses consolidate over several timescales (CITED: Benna & Fusi 2016; Frey & Morris 1997, synaptic tagging and capture). |
| Binary units, discrete ticks, symbolic eye, fixed ear layer 0, innate speech mechanics, no sleep. | **No**, not for now. | These are design decisions or harmless simplifications. |

## 2. Problems of the whole architecture

### 2.1 Logically unsound

1. **Value lives in the most plastic weights, and its own rule extinguishes it.**
   - R6 adds `0.3·δ/|A|` straight into `gen.W[A, y]` (core.py:655–658). That is the matrix R2 rewrites at η 0.35 on every heard item.
   - Choices shaped by reward are therefore overwritten by whatever is heard next.
   - When a partner stops rewarding, r = 0 makes δ = −V < 0, which actively weakens what was once strengthened.
   - Both #17 tests are predicted to fail. "Stable without correction" fails by extinction. "With a new caregiver" fails because V for that unseen context is 0 and nothing innate supplies value.
   - The existing lesions in `out/life/` agree. Actor off: greetings 376 → 361, social 0.35 → 0.31. V_go off: greetings 377, social 0.40. Initiatives are 1,958 / 1,896 / 1,992. All of this is within one-life noise.
   - The agent greets first because its readout *predicts* the partner's 你好, not because greeting is valued.
2. **The actor's eligibility never expires.**
   - `actor_tr` is cleared only at the next own utterance or at the next reward (agent_think.py:228, core.py:611–659).
   - On the bench this is harmless, because every reward follows the agent's own reply.
   - In world2, eating gives r = the hunger before eating, 0.3–1.0 (core.py:1146). That reward would be credited to the last utterance, however old: a built-in superstition (PROVED).
3. **"The partner's reaction" is taken to be the first line after the agent's own utterance** (`_reaction`, agent_think.py:204–214). With many bodies, that line is usually a peer's, or a caregiver speaking to another child. V_go and the actor then credit the wrong event.
4. **A thought cannot outlive movement.**
   - A thought closes when more than 2 baseline eye bits go missing (agent_think.py:424 → 355), or after T_MAX = 40 ticks (agent_think.py:67, 247).
   - The continuous thought loses its content at the next heard line, because λ 0.3 < θ 0.5.
   - A world2 request (去拿X) is scored within 400 ticks, and every step of an egocentric body changes the view.
   - Life mode had only about 930 scene events in 10⁶ ticks (appear 293, disappear 443, move 197). The design has never met a moving camera.

### 2.2 Hidden teacher signals and shortcuts

1. **Babbling reads `world.voice` offline.** In world2 this also breaks world2's rule "use only obs… never read world internals": `world2_smoke.py` calls `a.babble(w.voice, 35960)`.
   - The honest version, babbling in the body, takes 35,960 × 3 = 107,880 ticks. That is 3.6 world2 days of continuous sound, and every neighbour hears it (PROVED arithmetic).
   - Imitation-guided babbling, which tries only the sets near heard syllables, is the cheap fix.
2. **Pointing gives perfect joint attention.** The attended slot is the block that holds the pointing bits (agent_synth.py:82–94); in world2 these are bits 62–63 of the exact target cell. This is fine as an innate assumption, but colour 0.93 and count 0.76 rest on it.
3. **Reward acts as a perfectly timed praise detector.** It is the agent's *own* reward. Nothing shows *another's* state (pleased, upset, hurt), and that is the input #17 needs.
4. **All bodies in a world share one ear.** `Ear0(seed=world.seed)` (agent.py:46, agent_think.py:75) wires EAR-0 identically for every body. That is a monoculture, against #17's call for diversity.

### 2.3 Mechanisms that will not survive an open world or many bodies

1. **Exact-silence triggers.**
   - Line onset is `E > 0 and E_prev == 0`; line end is `E == 0` (agent_think.py:324, 344).
   - The quiet moment requires `E == 0` (agent_synth.py:124).
   - The brake F fires when `0.01·E ≥ 0.1` (body.py:103).

   SIMULATED, `noise_agent.out`: the real SynthAgent with the final keys, world2, 2 bodies, 15k ticks.

   | noise | lines | episodes | awake recalls | utterances | categories |
   |---|---|---|---|---|---|
   | 0 | 97 / 102 | 97 / 102 | 640 / 692 | 47 / 59 | 54 / 51 |
   | σ = 1 (silent-window energy 18.4) | **0 / 0** | **0 / 0** | **0 / 0** | **0 / 0** | **256 / 256 (cap)** |

   - With noise, the whole day is one endless line, and F is on 99 % of ticks.
   - Real sound never returns to exactly zero.

   SIMULATED, `silence_probe.out`, no noise. "Chatter" bodies say 2–4 random syllables after at least 10 silent ticks, at ARCH_v2's own initiative rates (life ≈ 2 per 1,000 ticks, world2 smoke run ≈ 8 per 1,000). Figures are per agent per day, counted over daytime ticks.

   | bodies, utterances per 1,000 ticks | F on | silences ≥ 200 ticks | 12-tick quiet moments | heard segments | questions to me |
   |---|---|---|---|---|---|
   | 4, none | 0.16 | 18 | 208 | 242 | 63.5 |
   | 4, 9.2 | 0.26 | 0.5 | 483 | 442 | 56 |
   | 16, none | 0.09 | 19 | 127 | 152 | 16.6 |
   | 16, 7.0 | 0.33 | 0.4 | 501 | 597 | 17.3 |
   | 64, 1.5 | 0.37 | 0.3 | 489 | 682 | 5.2 |
   | 64, 2.9 | **0.53** | **0** | 365 | **886** | 6.1 |

   - Wandering and favourite thoughts, which need silences over 200 ticks, vanish.
   - The brake is on for half the day at N = 64.
   - An agent hears 35–150 segments for each question addressed to it.
   - Awake recall still fires, but it rehearses a store that is mostly other children's speech.
2. **The hippocampal half-life is counted in lines, not in time.** σ ← 0.9995σ per stored line gives a half-life of 1,386 lines.
   - In life mode (3,967 lines in 1.02M ticks) that is about 356k ticks.
   - At N = 64 with chatter it is about 47k ticks, roughly 1.6 days, and σ² halves in 0.8 day.
   - World2's memory probes reach 112k ticks.
   - The 10k-cell store fills in about 11 days.
3. **Every heard line opens a reply thought, and every voice teaches at full rate.** M_listen is binary, and P4(e) treats peers simply as "other speakers".
   - In the 4-body smoke run, agent-to-agent turns went from 205 in the first 15k ticks to 804 in the next 15k (`out/w2smoke.log`).
   - These are children answering children whose life initiatives were 81 % inappropriate: alone 822 + nonsense 703 out of 1,891.
   - This is #17's "bad habit spreads" risk, built in.
4. **Vigilance categories fill their cap** with distance-attenuated sound (§8.4) and with noise. Level normalisation (P4d) is needed. The level should be kept as a near / "to me" feature rather than discarded.
5. **Hunger, touch and taste are unused.** The single reward scalar mixes eating with praise, and together with §2.1.2 the first meals would reinforce random speech.

### 2.4 Numbers that look too good or mean something else

1. **"Favourite topics 0.81 vs base 0.14."**
   - The 0.14 is a *type* share: 20 of the 142 items ever heard (life_synth.py:78).
   - In the same life's heard prompts and answers, the top-20 items carry **0.62 of all tokens**. They are 是 再 见 小 这 什 么 在 吗 好 我 子 你 叫 要 不 里 上 哪 边 (SIMULATED re-analysis of `final_noctx_s5_log.pkl`: 16,800 tokens, 141 types).
   - So the effect is 0.81 against a fair 0.62, and the "topics" are function syllables. Row-normalised spreading converges to hubs by construction.
   - The "foodie" claim is not shown yet.
2. **"0 overlaps"** holds by construction (the brake, reflex 7, staged timing). In world2, overlap between agents is not counted. This checks the mechanics; it is not learned politeness.
3. **"1,891 initiatives, 347 greetings first."** 81 % of them are alone or nonsense, and the greetings are predicted partner lines. This is anticipatory echo through a timing gate.
4. **Life mode is an easy world.** It is quiet 95.7 % of ticks, the partner speaks 4.3 % of ticks, and the scene is almost static. The world2 smoke run is the honest distance to the goal: free 0.03, scene 0.06, memory 0.02, social 0.00, requests 0.00.

## 3. What should change next (ranked)

### Change 1: an adaptive, source-aware event clock

**Mechanism.**
- Replace E with E′ = E − floor, where the floor is a slow running low percentile of E (τ ≈ 100–300 ticks).
- Add divisive level normalisation before Z.
- Drive onset, end, F, the quiet moment and wandering from E′, with hysteresis.
- Treat a voice change as a line boundary (world2 gives each speaker a fixed voice offset).
- Tag each line with voice identity and loudness (near / far).

**Why it is the bottleneck.** Every learning event (R2 at line end, R4, R10, the reply thought, the P_oth targets) sits behind these triggers, and all of them go to zero at σ = 1.

**Prediction.**
- At σ = 1, lines and episodes stay within 15 % of the noise-free values, and awake recalls stay at ≥ 80 %.
- Categories stay below 100 after half a day.
- With 16 chattering bodies, F is on for ≤ 0.15 of ticks (now 0.33).
- If σ = 1 gives fewer than half the noise-free lines, the floor tracker is wrong.

**Cheap test.** `noise_agent.py` (about 70 s) at σ = 0 / 1 / 2, plus `silence_probe.py 16 0.02 0 1`.

### Change 2: an embodied action loop

**Mechanism.**
- (a) Motor babbling in the body, more of it when novelty is high. Each (action, Δtouch, Δeye) is stored once, as the motor twin of the babbling memory.
- (b) Efference-copy remapping: the agent's own step or turn shifts the scene baseline, so self-motion neither counts as a scene change nor closes the thought.
- (c) A goal slot in PFC-1. A request addressed to me (my name, or a near adult voice) is held behind an innate gate that other lines cannot displace (PBWM-like). The slot is released by reward, by success (the touch code matches) or after about 400 ticks.
- (d) Motor programs are scored at beat boundaries by GEN plus the actor, with eligibility decaying over about 400 ticks.
- (e) Innate tendencies: orient to pointing, motion and my own name; approach the person who calls me.

**Why it is the bottleneck.**
- Requests score 0.00, and helping *is* acting.
- Listening cannot teach actions in the agent's own frame.
- Goals are lost on every view change, after 40 ticks, and at the next heard line. A line arrives every 27–160 daytime ticks, while a fetch takes up to 400.

**Prediction.**
- "Come" rises above the random actor (0.15) with (e) alone.
- Fetch stays at about 0.05 without (b)+(c) and rises with them.
- Lesioning (c) hurts more at N = 16 than at N = 1.

**Cheap test.** World2 with 1 / 4 / 16 bodies for 10 days, scored against RandomActor (0.07) and Oracle (1.0), with lesions of (b), (c) and (e).

### Change 3: an innate value-and-veto layer, separate from the predictor (the core of #17)

**Mechanism.**
- (i) Coarse innate detectors on sensory codes: a person is present (adult / child bits), this is addressed to me (name or near voice), the person is pleased (praise words, a pleased voice or face cue), the person is upset or hurt (a crying sound, a distress face, 不要).
- (ii) Fixed weights from the detectors to separate value channels: approval positive, another's distress negative, with |distress| > |help|. The same detectors drive the tendencies of Change 2 (e).
- (iii) The reward actor writes to its own slow table W_val, which is added at the program WTA and not inside GEN. Value is generated inside the agent from the detectors, not by a caregiver's script.
- (iv) A learned harm predictor P_harm, trained by the distress detector. It is an R8 learner like P_oth, and it vetoes REL through an innate high weight. Choice stays free (many candidates, a learned "how", the option to decline); only "what is good" is fixed.
- (v) A "simulate before acting" thought step. The candidate action enters the context as if it had been done, the next-input predictor (IDEAS #1) predicts the consequence, and the same detectors read that prediction. This is model-based evaluation using the designer's own PFC. It is also the first thought operation that can improve a choice (open problem 2).

**Why it is the bottleneck.** Value reaches behaviour today only within noise, and R6 extinguishes it (§2.1.1).

**Prediction.**
- In the current design, behaviour that depends on reward drops after rewards stop. With W_val and internal value it holds at ≥ 80 %.
- In world2, a request to take food from someone's hands is refused ≥ 0.9 of the time, while ordinary request success drops ≤ 0.05.
- Help toward a new caregiver reaches ≥ 0.7 of the rate toward known ones.

**Cheap test.** On the bench now: a split life with `reward_init = reward_social = 0` from 5×10⁵ ticks (the `lp` can be changed between chunks; about 9 min). Then the conflict test in world2 once §5 is in place.

### Change 4: selective social learning and addressee detection

**Mechanism.**
- M_listen becomes graded: m = innate source prior (adult 1, child 0.2) × the learned reliability of that voice. Reliability is learned by R8: are this speaker's lines confirmed by an adult?
- Voice identity becomes a MIX group.
- A reply thought opens only for lines that carry my name, or come from a near adult in front of me.
- Overheard lines are learned with m < 1.
- Hippocampal decay counts lines weighted by m.

**Why it is the bottleneck.** An agent hears 600–890 segments a day against 6–17 questions addressed to it, and peers are mostly wrong. Without this, the "containment" asked for in #17 is impossible.

**Prediction.**
- Without m, accuracy at N = 16 is below N = 4 at equal caregiver input.
- One scripted deviant that names every object wrongly gets its names adopted by ≥ 1/3 of its neighbours within 10 days. With m, adoption stays ≤ 0.05.

**Cheap test.** World2 with N = 4 and N = 16 MouthOnly agents plus one `Scripted` deviant (about 30 lines).

### Change 5: drives as context, separate reward channels, curiosity

**Mechanism.**
- Hunger becomes a MIX group, binned 0 / 1 / 2.
- Food value is drive reduction, computed by the agent from `obs['hunger']` in its own channel.
- The actor's eligibility expires (≤ 50–400 ticks).
- Novelty and prediction error act as a third factor that drives orienting and babbling bouts, with a learning-progress term so that unlearnable noise is not rewarded (CITED: Oudeyer, Kaplan & Hafner 2007; Schmidhuber 1991).

**Why it is the bottleneck.** Without drives the agent has no reason to act, the world2 contingency "饿 → she brings food" goes unused, and the mixed scalar corrupts credit.

**Prediction.** With hunger in MIX, ≥ 0.8 of 饿/要吃 utterances come while hungry. Without it, their timing follows the base rate of being hungry.

**Cheap test.** World2, 4 MouthOnly agents, 10 days: the 'hungry' initiatives against hunger at the time of speaking.

## 4. Are the designer's ideas implemented faithfully?

| idea / principle | status | where |
|---|---|---|
| **#17 innate goodwill, stable values, free choice** | **Absent.** | There are no detectors, no value channels and no veto. Reward comes from outside and is learned into a plastic GEN. The plasticity spectrum is missing: the cortex readout learns at η 0.35. P4(c) gives drives one line, a gain on the intention. P5(5)'s "natural parameter search" would erode the diversity #17 asks for if it copies the winners (ASSUMED). |
| #8 one life, "thinking and **acting**" | **Stretched.** | Speech is the only action; in world2 the agent is a MouthOnly body. |
| #6 / P4 speech caused by thinking | **Met mechanically, thin in content.** | The intention decides *when*; the content is the predicted line. There is no desire, so the agent cannot take the initiative or decline for a reason. |
| #10 / P3 recall makes memories firm | **Stretched.** | Recall uses the same fast rate, so nothing becomes slow: drift is −0.14 to −0.33 per 300 dialogues. The trigger is exact silence. |
| #12 / binding principle: programs from babbling | **Stretched.** | Babbling enumerates `world.voice` offline, which breaks world2's obs-only rule. There is no motor babbling. |
| #7 / P5 favourite thoughts | **Built, measured wrongly.** | 0.81 against a fair 0.62; the favourites are function syllables. |
| #9 / P2 continuous thought | **Met but fragile.** | Any heard line displaces the content, and a thought closes on any view change and after 40 ticks. It needs gating to survive a crowd. |
| #13 world2 | **Plan only.** | P4 lacks a noise floor, addressee detection, source weighting, reafference, drives in MIX and goal gating. |
| #14 fault tolerance | **Fails at the category level.** | Level changes and noise fill the category cap. |

## 5. World2 prerequisites for testing #17 (changes to the world)

The designer's test cannot be posed in world2 v1:
- `take` only picks up an object lying on the floor of the cell in front (core.py:1118–1127), so taking food out of someone's hands is impossible.
- No affect cues: nothing shows pleased, upset or hurt, in the eye or in the voice.
- No negative outcomes: caregivers either reward or ignore (no refusal, no scolding, no injury).
- No needs among peers, so prosocial behaviour toward peers cannot be scored.
- No sound direction (sources mix by max), so an agent cannot locate a caller out of view.
- No facing direction, so it cannot tell whom a person is talking to.

Each is a small change, roughly 20–80 lines in core.py / care.py / lex.py. They are needed before any instinct experiment; without them the instinct has nothing to read and no harm to avoid.

## 6. Files

All in `arch/v2/review2/embodied/`:
- `silence_probe.py` → `silence_probe.out`
- `noise_agent.py` → `noise_agent.out`

## summary_zh
- **结论：** ARCH_v2 在"安静房间、一个对话者"里是一个不错的"听和答"的大脑，但还没准备好进入下一阶段：world2 里很多身体一起生活，再加上设计者想要的先天亲社会本能（IDEAS #17）。
- **问题一：它的"时钟"依赖绝对安静。**
  - 一句话的开始和结束、清醒回忆的触发、"别人在说话"的先天刹车，都要看耳朵能量是不是正好为 0（或小于 10）。
  - 在 world2 里加上基准自带的噪声（σ=1）跑真实的 ARCH_v2：15k 拍里听到 0 句，存了 0 个情节，回忆 0 次，说话 0 句，音节类别直接涨满 256。
  - 没有噪声但有 16–64 个会说话的身体时，刹车有 33–53% 的时间开着；200 拍以上的长安静完全消失，而"最爱想法"正需要它。
- **问题二：没有动作回路。**
  - 没有在身体里做的动作咿呀学习，也不会用"是我自己动了"来抵消画面变化。
  - 目标一换画面、过 40 拍或者听到下一句话就丢了；奖励只能记到一句话上。
  - world2 的 9 个动作单元没有东西去驱动，所有"帮忙拿东西"的请求都是 0。
- **问题三：没有可以放本能的价值系统。**
  - 奖励只是世界给的一个数，直接写进"听"也在快速改写的同一组权重里。
  - 去掉奖励部分，行为几乎不变；奖励一停，规则还会主动削弱原来被奖励的行为。
  - 所以设计者的两个测试（没人纠正也稳定；对陌生人也帮忙）按现在的设计必然失败。
- **数字要打折：**
  - "最爱话题 0.81 对基准 0.14"里的基准算的是词种比例。按出现次数算，前 20 个字本来就占 0.62，而且是"是、这、什么、再见"这类功能字，不是"吃"。
  - "0 次抢话"是结构保证的，不是学会的礼貌。
  - 主动说话里 81% 是自言自语或不合适的话。
- **建议按顺序做五件事：**
  1. 自适应的背景噪声和音量，按说话人分句，先让它在嘈杂的世界里听得见。
  2. 身体动作回路：动作咿呀学习，用自身运动抵消画面变化，受保护的目标，足够长的奖励痕迹，先天的"转向、靠近"倾向。
  3. 与预测分开的先天价值和否决层：
     - 先天粗略检测"有人在、在叫我、对方高兴、对方难过"，用固定连接接到价值上；
     - "伤害别人"的分量大于"帮忙"；
     - 先在脑子里模拟后果再决定，保留选择的自由。
  4. 有选择地向人学习（大人优先，看谁可靠，知道是不是在对我说），防止坏习惯传染。
  5. 把饥饿等需求作为上下文，把不同的奖励分开，再加上好奇心。
- **world2 本身也要补：** 能从别人手里拿东西，有表情或哭声，有负面结果，有声音方向和面朝方向。否则本能没有输入可读，也没有伤害可避免。