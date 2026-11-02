# Adversarial review of ARCH_v3_core: developmental psychology and developmental robotics

**Angle.** Language acquisition, turn-taking, caregiver input, babbling and vocal learning, multi-agent social worlds.

**Scope.** Review only. I ran three small local numpy toys, one process each, at most 110 MB, after `free -m`. I changed nothing in the repo or in other reviewers' folders.

**My files** (all in `docs/v3design/review/develop/`):
- `caregiver_toy.py` / `.out` / `caregiver_toy2.out` / `.json`. This reuses the synthesis' own `drift_toy.Agent` unchanged (imported read-only) and changes only the caregiver: she sometimes moves on quickly, sometimes re-asks, the give-up delay is jittered, there is a global timing scale, and in one arm her care continues after formats.
- `babble_toy.py` / `.out` / `.json`. This uses the bench's own `physics.Voice(physics='lawful')` and the real mouth rule. It asks whether online babbling plus the M2 inverse map can say the 141 lexicon characters.
- `garble_frac.py` / `.out`: the share of random babbles the mouth rule decodes as garble.
- `dev_checks.py` / `.out`: the arithmetic behind the PROVED numbers below.

Labels: PROVED (code, text or arithmetic), SIMULATED (my run), CITED, ASSUMED.

---

## Answer to the one question, from my angle

- **Buildable:** yes. Most of the world changes are realistic in kind.
- **Logically sound:** not yet.
  - Two world changes carry exactly the rules that the stability and maturity claims rest on: W1's clean answer slot and its timing ratio, and W4's policing of overlaps at probability 1.
  - The maturity rule as written cannot be met with the core's own spontaneous output (P1d babbling, ε).
  - The value prior P5b rewards contiguity rather than contingency. So talking always beats silence, corrections arrive as rewards, and crowds reward chatter.
- **Today's abilities:** likely for scene question-and-answer, naming, babbling (my toy shows lawful physics makes online babbling work) and greetings. At risk: multi-syllable replies (the self/other coding), mechanics after maturity, and helping in world2.
- **New abilities:** one life is about one child-day of input, and one lineage is about nine child-days in sequence. That is too little to show new kinds of ability.
- **Pure neurons:** outside my angle. The developmental cost is experience, not neurons: a realistic 10⁸-tick childhood is about 28 CPU-hours per agent at s = 1 (PROVED arithmetic).

---

## Issues, most severe first

### 1. FATAL as stated: "0 errors after maturity" cannot be met with constant-rate spontaneous babbling below M4

**Evidence.**
- **Babbling bypasses learning.** P1d is "babble at rate r_bab" (G26 = 0.01 per idle tick, a constant gene). Babble noise enters PROGRAMS directly (§1.6: "M4 output, babble noise"), so no learned NoGo can stop it. The bench body babbles whenever the body is idle, and that includes the gaps inside an exchange (`body.py` `idle`). The life monitor counts every garbled onset at any time, alone or at night (`life.py` lines 166–175). An extra turn is "a bout after a ≥ 5-tick silence".
- **Expected errors after maturity.** In a bench life about 947 answered exchanges come after maturity. With probes 28 % of exchanges and about 30 idle ticks before the next line, the extra-turn hazard is 0.079 per exchange. That gives about 75 extra turns after maturity, and P(0) ≈ 2 × 10⁻³³ (PROVED, `dev_checks.out`). With the lexicon-only mouth, 64 % of random babbles are garble (SIMULATED, `garble_frac.out`).
- **No constant rate works.**
  - L9 needs about 500–1,000 babbles in the first 20 % of life (my toy, issue 10), so r_bab ≥ 0.003–0.006.
  - P(0 extra turns after maturity) ≥ 0.5 needs r_bab ≤ 8 × 10⁻⁵.
  - The rate must therefore fall 37–74-fold between childhood and maturity (PROVED + SIMULATED).
- **Learning cannot reach a hard 0 either.** The design's own R-e argument bounds TD extinction: "NoGo grows by at most η_v·V₀ ≈ 0.3". Bounded extinction prevents mutism, but by the same token it cannot drive unwanted output to exactly 0. Only structure (the brake, refractoriness) can give a hard 0.
- **Exploration.** §6.3 says ε produces unprompted utterances (about 80 per life). Whether ε can start a release inside an exchange is unspecified.

**Smallest fix (generic).**
- Route P1d babbling and ε through M4 as candidates, so the brake, refractoriness and gates apply to them.
- Scale both by (1 − π_motor), the motor area's own precision, i.e. how accurate the forward/inverse maps are. Babbling then fades as vocal learning completes, with no schedule. That matches the decline of non-word vocalising (Oller 2000; CITED).
- Also scale both down while P4b "addressed to me" is on. Songbirds sing less variably when singing to a partner than alone (Kao, Doupe & Brainard 2005; CITED).
- ε re-ranks content only. It never creates a release.
- Count post-maturity mechanics only inside exchanges and addressed windows. Solitary vocal play ("crib speech", Weir 1962) is not a conversational error.

### 2. SERIOUS: P5b is contiguity, not contingency

**Evidence.**
- **The definition.** 5b is "a person-like onset while my own efference trace is high". With W4 caregivers, who "never start while it speaks" and answer the child, every vocalisation that does not overlap gets r ≥ w_resp, while silence gets r = 0 (PROVED from definitions).
- **§6.1's sign is wrong.**
  - A W3 replayed pair starts at stop + 6, so P5b fires: a wrong answer gets r ≈ 1 + 0.1 (W7), not 0. The text's "δ ≈ 0 − 0.5 = −0.5" omits P5b and becomes about +0.5 early in learning.
  - The right answer gets 1 + 0.8 with genes w_resp = w_plea = 1, not the "r = 1" in the text.
  - Right and wrong therefore differ only by 0.7·w_plea, and both beat silence.
- **The sign problem spreads.**
  - Garble, echo and extra turns are rewarded relative to silence.
  - Babble is never extinguished while W4 still answers 30 % of babbles.
  - Extra-turn suppression rests only on discounting the delayed reaction (γ⁸ ≈ 0.66), not on δ < 0 as §6.4 claims.
- **Crowds reward chatter.** In world2 with 16 random agents there are 19,919 agent-to-agent "answers" per day (CITED world2 refs). Peers' onsets reward any utterance.
- **What infants do.** Infants track contingency (a conditional probability), not mere temporal closeness (Watson 1985; Gergely & Watson 1999; Tarabulsy, Tessier & Kappas 1996; CITED). Toddlers' helping is even undermined by extrinsic rewards (Warneken & Tomasello 2008; CITED).

**Smallest fix.** r_5b = w_resp · onset · (1 − q̂). Here q̂ is GEN's own prediction, made in the context before my action, that a person starts by now. GEN already computes "another speaker is about to start" (§1.4). Holding q̂ costs one scalar trace.
- Routine reactions (a probe at 45, the reaction to any answer) and crowd background then stop rewarding talk as such.
- Approval (P5c) cleanly signs right against wrong.
- Asking (这是什么, "what's this") is still rewarded, because the reply is caused by the question.
- The 17-sub-item count is unchanged: this redefines 5b.

### 3. SERIOUS: W1 and W4 are timing rules moved into the world

**Evidence (SIMULATED, `caregiver_toy`, 5 seeds, the synthesis' own ST learner; untutored to 3,000 with 30 % silence).**

| caregiver behaviour after the child's silence | ST accuracy at end of formats → after 900 → after 3,000 untutored | share of answers that echo the question (end) |
|---|---|---|
| W1 as specified | 1.00 → 0.99 → 0.99 | 0.00 |
| moves on quickly (next question after 8–16 ticks) in 10 % | 0.99 → 0.71 → **0.10** | 0.90 |
| moves on quickly in 25 % | 0.87 → 0.25 → **0.01** | 0.99 |
| re-asks the question in 30 % | **0.71** → 0.86 → 0.86 | 0.29 at end of formats → 0.14 |
| re-asks in 60 % | 0.23 → 0.44 → 0.34 | 0.66 |
| give-up delay jittered ±50 % | → 0.96 | 0.04 |
| global timing ×0.5 / ×2 in the untutored world | → 0.95 / 0.99 | – |
| realistic mix (move-on 15 %, re-ask 30 %, jitter 50 %) | 0.63 → 0.19 → **0.01** | 0.99 |
| realistic mix, care continues in 50 % of later dialogues | 0.63 → 0.69 → 0.65 | 0.35 |
| move-on 10 %, care continues in 50 % | 0.99 → 1.00 → 1.00 | 0.00 |

- **What the anchor needs from the world.** It needs a clean answer slot, and a move-on delay that stays at least about 4× (two octaves) beyond the self-answer delay. A per-life scale preserves that ratio; within-life variety breaks it.
- **The synthesis toy's jitter claim is narrower than stated.** Its "±50 % jitter" never jittered the 45-tick give-up (PROVED from code: `now = qend + 45 - 15`).
- **Real caregivers do both things.** Re-asking and moving on are standard ways caregivers pursue a response (Pomerantz 1984; Stivers & Rossano 2010; Wood, Bruner & Ross 1976; CITED).
- **W4 contradicts §6.4.** §6.4's late-start lesson is "overlap → W4 does not answer → δ < 0". But a W4 caregiver "never starts while it speaks", so a late start is never overlapped and gets answered, which is a reward (PROVED from the text). "0 overlaps with caregivers" then comes from the polite world plus the brake, not from learning. It will fail with impolite partners: peers, or the sealed ×0.5 worlds.
- **W4 is unrealistic at probability 1.** Never answering overlapped speech is not how adults treat infants. They often yield to infants' interruptions; explicit "don't interrupt" teaching comes years later (ASSUMED from general literature).

**Smallest fix.**
- Make the development world use realistic caregiver statistics: pursuit, re-ask and move-on mixtures, and W4 politeness at p ≈ 0.7–0.9.
- Add one sealed world with an impolite partner who starts at the cut regardless.
- Judge L6 on that world with care continuing. Keep the untutored 10,000 test, but report its sensitivity to move-ons.
- Note that the echo cost of re-asks persists even with continuing care. It must be learned away through approval (actor and P5c), which needs fix 2.

### 4. SERIOUS: the maturity point is the wrong currency, too early for crowds, content-specific and gameable

**Evidence.**
- **Wrong currency.** Turn-taking is learned from contingent turns, not from heard words (Romeo et al. 2018; Gilkerson et al. 2018; Hilbrink, Gattis & Levinson 2015; CITED).
- **Too early for world2.** Questions put to the agent before maturity (PROVED, `dev_checks.out`):

  | world | questions before maturity |
  |---|---|
  | bench life | about 574 |
  | world2, 4 agents | about 168 |
  | world2, 16 agents | about 93 |

  The design's "about 600 questions in childhood" holds only in the bench life.
- **Timing is learned per content.** MIX's anchor is LATEST (item × age × source), and the M4 gates read MIX only (P2d). So learned NoGo and Go for timing transfer only through syllables the latest line shares with earlier lines. New question kinds first met after maturity (held-out, S1-B, the world growing with s) start their timing from scratch (PROVED by construction).
- **Gameable.**
  - If "heard" is counted by the agent's own SYL area, genes that hear less (G4, G6) delay maturity and lengthen the error-free-scoring childhood.
  - The lineage's −1 per post-maturity error selects for quiet genes (low r_bab, low u, high τ_ref), which drifts towards mutism in crowds.
- **Wrong test order.** L7 tests the post-maturity zero before L9 (online babbling) and L10 (perceived reward), and both of those change the signals L7 depends on.

**Smallest fix.**
- Maturity = the first tick at which the agent has had about 300 contingent exchanges (own utterance followed by a person's reply), counted by the world, capped at a world-counted 15,000 audible syllables.
- Let the M4 release channel's keys read TIME (source × age, with no content) directly, besides MIX. This gives content-general turn timing, the old P_oth generality, inside P2d.
- Run L9 and L10 before L7, or re-run L7 at L11.

### 5. SERIOUS: μ follows input presence only, so replay starts 12 silent ticks into an answer window

**Evidence.**
- μ < 0.05 after about 12 silent ticks switches the hippocampus to uniform spontaneous replay and gates live pathways down (§1.3 B2).
- §6.1's slow route answers at τ ≥ 16, and §6.6.3 relies on recall to bridge late answers. Both would then run on a randomly reinstated episode (PROVED from the definitions).
- One caveat: pointing keeps EYE above its set-point for pointed questions only. That makes the behaviour inconsistent across tasks.

**Smallest fix.** μ ← max(input presence, p̂_onset). An expected input keeps encoding on, matching ACh's role in attentional expectancy (Sarter, Hasselmo et al. 2005; CITED). Replay then happens only in unexpected quiet.

### 6. SERIOUS: crowds in world2

**Evidence.**
- **The brake makes agents near-mute.** P1c is non-selective and is on 16–52 % of the day (CITED hearing). Because every syllable is an onset in the bench mouth rule, it also cuts lines short at the next syllable.
- **The overlap metric blames the agent for others.** world2 counts my mouth ticks while others speak, whoever started. With 16–64 unpoliced child agents, post-maturity overlap cannot be 0 whatever the agent does.
- **Peers' errors become training data.** No prior gives addressed or adult speech more learning weight. Peers' childhood errors, in overheard peer Q–A pairs and babble, are learned like caregiver formats. Children weight directed speech (Weisleder & Fernald 2013) and trust reliable speakers (Koenig & Harris 2005; CITED).

**Smallest fix (redefinitions of 1c and 3a; no new sub-item).**
- P3 selects one engaged voice source: the one addressing me, or else the nearest.
- The brake reads that source only.
- P4b "addressed" multiplies η of prediction synapses on that stream, as π already scales representation learning.
- The post-maturity overlap metric becomes "over the engaged partner"; world2 already logs this.

### 7. SERIOUS: the IDEAS #17 values do not survive the TD form

**Evidence (PROVED, `dev_checks.out`).**
- **The time horizon is too short for world2.** G24 γ = 0.95 per 75 ms tick is a 1.5-second horizon. world2 scores requests within 400 ticks.

  | praise arrives after | value with γ = 0.95 | value with γ = 0.99 |
  |---|---|---|
  | 100 ticks | 0.006 | 0.37 |
  | 400 ticks | 1.2 × 10⁻⁹ | 0.018 |

  So fetching ("去拿X", fetch X), #17's central example, is not learnable from praise.
- **"Harm outweighs help" is fragile.** w_upset ≥ 1.5·w_plea holds only if the harm is perceived within ln 1.5 / −ln γ = 7.9 ticks (0.6 s) after the praise.
- **The agent must harm before it can decline.** Model-free TD learns to decline harm only after experiencing it, so gen 0 must cause harm. Toddlers show concern for others without having harmed (Vaish, Carpenter & Tomasello 2009), and distress cues inhibit action directly (Blair 1995; Pavlovian inhibition, Dayan et al. 2006; CITED).
- **Sycophancy.** P5c rewards anyone being pleased, including a requester whose request harms a third party.
- **Missing parts.** IDEAS #17 also asks for innate action tendencies and a harm brake; the design has value weights only.

**Smallest fix.**
- Discount per event onset rather than per tick (G24 keeps its number but changes its unit).
- Let P4c's "upset/hurt" also add an immediate, undiscounted NoGo to the currently eligible M4 channels: the brake form of 5c.
- The #17 test must include harm arriving after help, and a never-tried harmful request.

### 8. SERIOUS: self/other "disjoint units" against producing multi-syllable lines and imitating

**Evidence.**
- §6.5 R-a rests on self events being "disjoint units", and own voice never trains GEN.
- So GEN cannot predict the second syllable of the agent's own line (self-红 → 色, "red" → second syllable of 红色).
- The drift toy only ever says one token (PROVED: `ag.say([tok]…)`). Bench answers are 1–4 syllables (红色, 在左边, 不客气).
- The continuation must come from partly overlapping self/other codes, which contradicts R-a, or from CA1 completion, which is not walked through.
- This is the correspondence problem of imitation (Heyes 2001; role-reversal imitation, Carpenter, Tomasello & Striano 2005; CITED). If the disjointness is literal, it is fatal for multi-syllable replies.

**Smallest fix.**
- LATEST = shared item × age units plus a separate source population. Some conjunctions are then source-general (continuations transfer) and some source-specific (R-a protection where learned).
- X1 must use multi-syllable answers.

### 9. SERIOUS: experience per life and per lineage

**Evidence (PROVED arithmetic, ASSUMED 18,000 heard syllables per child-day).**

| amount | child-days of input |
|---|---|
| one bench life (15,457 syllables) | 0.86 |
| maturity point | 0.28 |
| a 10-generation lineage in sequence | 8.6 |

- A child at 12 months has heard about 427 bench lives' worth of input.
- Siblings live in the same S0 families, so merging mostly consolidates identical lessons rather than adding new ones.
- The develop member's phase plan (P0–P5, 0.6–1.2 × 10⁸ ticks) was dropped from the synthesis (grep finds no phases).
- One 10⁸-tick life costs about 28 CPU-hours, against about 109 CPU-hours for the whole lineage loop.

**Smallest fix.**
- Give each sibling a different family mixture, so that merges add disjoint lessons. That is born-wired R18's working case.
- Grow life length L along the generations.
- Make develop's world-only phases gen 0's world. Keep 10⁶-tick lives as the cheap screening tier.

### 10. MINOR: online babbling (W6, L9) is feasible, with gaps

**Evidence (SIMULATED, `babble_toy`, 3 seeds).** With lawful physics, the M2 delta-rule inverse map, trained on N random own-voice babbles, produces the correct character from hearing:

| babbles N | lexicon characters produced correctly (full 3,970-character mouth) |
|---|---|
| 100 | 0.38–0.58 |
| 300 | 0.84–0.91 |
| 1,000 | 0.95–0.99 |
| oracle (knows the muscle patterns) | 1.00 |

Random babbling alone binds only 0.15–0.21 of the lexicon by 1,000 babbles, and 0.55–0.62 by 20,000.

**What this shows.** Lawful physics is the lever. Caregiver imitation is not needed to bind sounds; it is motivation.

**Gaps.**
- How a new PROGRAMS unit is formed from the inverse map is unspecified. 2,048 fixed innate synergies would cover about 6 % of characters (2,048 of 35,960 four-muscle sets).
- The toy is a best case: a clean, aligned 8×6 grid, no noise, and voice offsets of only ±1.
- Babbling must fade (issue 1).

### 11. MINOR: plain recasts do teach scene content in the real core

**Evidence.**
- VR learns "attended object → heard word" from any recast. §6.6's finding 1 ("plain recasts teach only the value, not the answer") is an artefact of a toy that has no VR. It holds only for non-scene facts.
- That fits the child data, where recasts help (Saxton 1997; Farrar 1992; CITED).
- W3's "replayed pair" is really a re-ask followed by the answer, not Chouinard & Clark's reformulation. Its 0.3 / 0.4 / 0.3 rates are ASSUMED.

### 12. MINOR: the greeting walk (§6.3) is fragile

**Evidence.**
- With the stated numbers, release is at τ 19 and the first syllable at about 23 (PROVED, reproduces §6.3). That is with e = p, and the margin e is smaller.
- Under development jitter ×0.75 (the partner greets at 22) the partner always greets first. Bootstrapping then relies on ε.
- Social replies (你好 / 再见 / 晚安, "hello / goodbye / good night") are learnable only through the copy/echo source plus ε plus reward, because life.py never shows anyone else answering a greeting. §6 does not walk this route.
- In world2's egocentric view, P4a "arrives" fires whenever the agent turns its head, unless it reads the slow "present" integrator.

### 13. MINOR: internal inconsistencies

- `checks.py` uses I += G·e + u (it releases after 7 ticks even with e = 0), against M4's (G + u)·e.
- `checks.py` uses η_v 0.2 (10 events), against G11's 0.3 and §6.4's "7 events".
- P1c needs an innate corollary-discharge gate at birth, because the forward model that would tell it "self" from "non-self" is not learned yet. That is a hidden sub-item.

---

## Are W1–W7 realistic, or rules moved into the world?

| change | realistic in kind (CITED) | the rule it hides |
|---|---|---|
| W1 formats | yes (Ninio & Bruner 1978; Wood, Bruner & Ross 1976) | a clean answer slot, and a move-on ≥ 4× the self-answer delay (issue 3) |
| W2 overheard pairs | yes (Akhtar et al. 2001; Floor & Akhtar 2006) | none, though in world2 mostly peers' errors |
| W3 corrections | types yes, rates ASSUMED | the replayed pair is a format in disguise |
| W4 polite caregivers | contingency yes (Goldstein & Schwade 2008) | p = 1 policing guarantees 0 caregiver overlaps; it contradicts §6.4 |
| W5 jitter | yes | per-life scaling keeps the critical ratio fixed |
| W6 lawful physics | yes, and essential (issue 10) | none |
| W7 perceived approval | yes (Brown & Hanlon 1970) | the world still judges correctness (acceptable) |

## Hidden world-fitted structure in the "generic" parts

- G24 γ, G14 τ_e and G23 τ_ref are fitted to the bench's 6–45-tick speech exchanges, not to world2's actions of up to 400 ticks.
- The maturity count of 5,000 is fitted to the bench life's 600 questions.
- P4a "arrival" is a bench notion.
- The P1c brake assumes the bench's 2–3 voices.

## Migration order and first experiments

- Putting the world first (L0) is right, but it must use the realistic-caregiver arms.
- Move L9 and L10 before L7.
- Add my toy's move-on and re-ask arms, and multi-syllable answers, to X1. This is cheap: about 1 minute locally.
- Add X0, a post-maturity noise budget: babble and ε hazard counted per exchange from a v2.2 life log. It needs no training run.

---

## Verdict: fix, then build

The three fixes that matter most:
1. **Spontaneous variability through M4.** Route babbling and ε through M4, scale both by (1 − π_motor) and by "not addressed", and count post-maturity mechanics only in exchanges. Without this, the hard post-maturity zero is arithmetically impossible.
2. **P5b as contingency.** Redefine "being responded to" as a response not predicted before my act: onset × (1 − q̂). Talk itself is then not rewarded, approval signs answers cleanly, asking stays rewarded, and crowds stop rewarding chatter.
3. **A realistic world and a fairer maturity point.** Re-ask, move-on and politeness at p < 1 in the development world, with stability judged with care continuing. Maturity counted in world-counted contingent turns with a syllable cap. Content-free TIME keys for the release gates.

Next in importance: discount per event, and the "upset → NoGo" brake, for the #17 values (issue 7).

---

## summary_zh

1. **总评：先修再建。** 世界改造大多符合真实育儿，但有两处其实是把规则搬进了世界：
   - W1 要求问题后面那段时间里"只能出现答案"，而且大人放弃、改问下一题的时间必须是自答时间的 4 倍以上；
   - W4 要求大人百分之百不接被抢的话。
2. **我的小模拟（5 个种子）：**
   - 只要 10% 的沉默之后大人很快问下一题，3,000 段无人指导对话后正确率从 0.99 掉到 0.10；
   - 大人有 30% 重问时，孩子 29% 的回答是重复问题；
   - 真实混合情况下最后是 0.01；
   - 如果之后大人还继续教一半时间，"很快问下一题"造成的遗忘会被修复，但重复问题的毛病仍在。
3. **"长大后 0 错误"按现在的写法做不到：**
   - 咿呀学语是固定频率，而且绕过了基底节闸门。一生中长大后约 75 次多余发言，0 次的概率约 10⁻³³。
   - 学会咿呀需要早期频率高，0 错误需要后期低 37–74 倍，一个固定基因做不到。
   - 解决办法：咿呀和探索都走闸门，随运动模型变准、以及"有人冲我说话"而减少；只在对话里统计错误。
4. **"有人回应就奖励"只看时间挨得近，不看因果。** 所以说话永远比沉默好，纠正也成了奖励（§6.1 的符号算错了），人多时同伴的话也算回应。改为"只奖励我说话之前没料到的回应"。
5. **其他问题：**
   - 成熟点应按"有回应的轮次"计数，而不是按听到的音节数；
   - 价值的折扣按每拍算，只有约 1.5 秒，帮忙取东西学不会；"伤害大于帮忙"只在 0.6 秒内成立；
   - 等回答时，静默 12 拍后就会进入回放；
   - 人多时刹车让模型几乎不说话；
   - 一生的输入约等于孩子一天，一个血统约等于 9 天。
6. **好消息：** 在"有规律的发声物理"下，约 1,000 次咿呀就能把 95–99% 的词汇字说对。