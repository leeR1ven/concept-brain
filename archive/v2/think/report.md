(Saved by the coordinator from the reviewer's final reply; the harness blocked the reviewer's own write.)

# ARCH_v2-think: does "speech caused by thinking" help?

This tests the designer's principle on develop's prototype. The harness blocked `report.md` ("subagents should return findings as text"), so this reply is the full report. Code and outputs are in `/tmp/claude-0/-root/e92d3d7d-5823-4f5c-a0ec-4fe302bff9af/scratchpad/arch/v2/think/` (§12).

Labels: **PROVED** (argument with numbers), **SIMULATED** (numpy run; file and numbers), **CITED** (paper), **ASSUMED**.

Setup:
- **Bench:** the shared bench v1.2, copied unchanged into `think/bench_copy/`.
- **Life mode:** bench v2.0 `life.py`, copied in; the core modules are byte-identical.
- **Main regime:** as develop — `tutor='recast'`, 300 stage-A dialogues, then 2,000 each of B, C and D.
- **Baseline:** develop's final configuration re-run in my copy. It reproduces develop's report exactly: B/C/D 0.54 / 0.62 / 0.48 (`out/bench_base_s0.log`).

---

## 0. Verdict

The principle under test: speaking is caused by thinking, not directly by the external signal; with time to keep thinking the model would learn to think and then speak, and it could speak whenever it thinks.

What I built (§1):
- ARCH_v1's fixed GO timer is replaced by a speech intention.
- The thought is built one step per tick, and its readout confidence feeds an accumulator.
- A BG-like gate releases the accumulator into the body's GO. The innate F brake is kept.
- Two local prediction-error learners decide when to speak: one learns the timing of other speakers' turns, the other learns from the partner's reaction.

| question | result | label |
|---|---|---|
| (a) content | **No gain.** Seed 0: B/C/D 0.56 / 0.61 / 0.48 vs develop 0.54 / 0.62 / 0.48. Seed 1: 0.60 / 0.58 / 0.47 vs 0.55 / 0.62 / 0.47. The seed-1 C gap is a babbling artefact, not timing (§2). | SIMULATED |
| why no gain | develop's thought needs 3 steps (recall, comparator, recognition). Charged one tick each, they fit in the 4 silent ticks that v1 leaves before its first boundary, so the timer was never short of thinking time. | PROVED |
| (b) thinking time | **Cutting it hurts a lot; extending it adds almost nothing.** Frozen agent: B 0.05 / 0.06 / 0.31 / 0.35 with 0 / 1 / 2 / 3 steps; free = v1 = 3 steps. Five more steps of serial memory search: +0.01–0.03 at test time, but training with it broke rituals (0.93 → 0.64). | SIMULATED |
| (c) timing | **Yes: latency becomes a confidence signal.** Wrong replies are slower than correct ones in every task. Held-out items are slower. Hard tasks are slower: colour 16.8 and count 12.5 ticks vs ritual 6.7 (unbounded run). Release confidence predicts correctness: in C, 0.98 correct at confidence ≥ 0.6 vs 0.34–0.39 at < 0.1. | SIMULATED |
| (d) initiative | **Only where the partner waits.** Staged bench: 1 of 6,701 scene thoughts spoke, and it overlapped. Life: 639 initiatives in 11.7 h, including 106 rewarded greetings-first, a behaviour that is learned over the life. But only 3 namings, and many inappropriate utterances (237 nonsense, 293 talking alone). develop: 0 initiatives. | SIMULATED |
| (e) turn-taking | **Better.** Speaking after the partner's recast: 0.61 → 0.28 (B). Speaking after statements: 0.57 → 0.01 (D). The agent no longer "answers every line". | SIMULATED |
| (e) risk | **Real.** A GO caused by thought can fall anywhere; v1's GO always followed a line. Overlaps appeared in every version that let a reply start late: 804 → 875 → 13–29 ticks in the short runs, 17 exchanges in run 1, 3 in the unbounded final. A learned brake cannot remove them, because "the floor is free" becomes self-fulfilling. The GO-driven CD timer also deafens the agent after a refused GO. | SIMULATED + PROVED |
| monologues | **None.** One thought gives at most one utterance; 0 extra turns; the longest reply is 10 syllables, the same as develop. | SIMULATED |

**Minimal mechanism that keeps the body's 0-error mechanics (§9):**
1. The F brake on GO (kept from v1).
2. A first-syllable reflex on the current ear window.
3. Own voice tagged by the real mouth output, not by the GO-driven CD timer.
4. A hard turn-transition window: a reply may start at most 7 ticks after another speaker's line ends.
5. One utterance per thought.

Self-initiated speech passes only through the learned brake, whose prior is "the floor is taken". With these five, the final configuration on both seeds has 0 reply-path overlaps, 0 non-3-tick syllables, 0 garble and 0 extra turns. The only overlap left (seed 0, D, 1 exchange of 13 ticks) came from the self-initiated path.

**Bottom line.**
- "Speech from thinking" is the right architecture for honest timing and turn-taking: latency tells how sure the agent is, and it stops replying to every line. It is also the only route to initiative; in life the agent learned to greet first.
- On this bench it brings no content gain: develop already got that thinking for free.
- It adds a turn-taking risk that needs innate reflexes, not only learning.
- The gain the designer expects needs two things this prototype lacks:
  - thought operations that are longer than the timer's window **and** reliable (my serial memory search was longer but picked "decisive" rather than correct memories);
  - for initiative, a production target other than "predict the next heard line" — e.g. "say the name of what I attend to".

---

## 1. What I built

New files: `body_think.py`, `core_think.py`, `agent_think.py`, runners `run_think.py`, `run_life.py`, `eval_frozen.py`. develop's `agent.py`, `core.py` and `rules.py` are copied unchanged.

**Equivalence check (SIMULATED, `test_equiv.py`).** In v1 mode with develop's thought steps (`policy=v1 think_k=3 ior=0`), the new agent is identical to develop: 739 of 739 records (replies, latencies, after-recast speech) on A 100 + B 300.

### 1.1 Body (`body_think.py`): one change, one reflex
- **v1:** `GO(t) = [F2 − 2F − 2SPK − 2ΣCD ≥ 1]` — the offset of F, a fixed pulse 4 ticks after any line.
- **think:** `GO(t) = REL(t−1) ∧ ¬F(t−1) ∧ ¬SPK(t−1) ∧ ¬CD(t−1) ∧ ¬GO(t−1) ∧ ¬R1(t−1) ∧ mouth idle`. REL is the BG gate's release; the F brake is kept.
- **Reflex `brake_now`:** the first syllable also needs `0.01·E(t) < 0.1` on the current tick.
  - Reason: the boundary checks F(t−1) = [E(t−2) ≥ 10]. A partner who starts 0–1 ticks before the boundary is not seen: a 2-tick blind window in the GO → R1 → boundary pipeline.
  - v1 never meets it, because its GO only ever follows a line.
- **Unchanged:** CD, R1, R2, boundary, 3-tick beat, program WTA, END rule, breath counter, floor, D/I/M ramp.

### 1.2 Thoughts (`core_think.py`)
A thought opens:
- when another speaker's line ends — a **reply** thought about that line, i.e. develop's prompt context;
- at a visual change or pointing onset in silence, if no thought is open — a **scene** thought, with no prompt (the self-initiated path).

It closes:
- when the own utterance ends, so one thought gives at most one utterance;
- when another speaker starts;
- when the scene disappears or changes before release;
- after 40 ticks.

One step runs per tick while the mouth is idle and nobody speaks:

| step | operation | in develop |
|---|---|---|
| k = 0 | held prompt (PFC-1) + the tutor's pointing (keys P, L, N, CP, A, F) | zero time, at the boundary |
| k = 1 | hippocampal recall #1 with develop's key → NEXT / COMPLETION candidates, FAM | zero time |
| k = 2 | comparator (word primes vs scene per slot) → internal attention, match M | zero time |
| k = 3 | visual recognition (VR) at the attended slot | zero time |
| k = 4…8 (variant `ior=1`) | serial memory search: next-best episode, inhibition of return; its candidates are adopted if they raise the readout margin by > 0.05 | absent |

- **Scene thought:** attention follows pointing, or an innate scan of the slots; recall with the scene as the cue; VR; prompt-free `NOP` keys.
- **Readout confidence:** `e = clip(s1 − max(s2, 0), 0, 1)`, the margin of the best first-item program score over all routes. It is computed read-only, so thinking creates no MIX units.
- **Listening:** it uses the completed thought about the previous line, as develop did.

### 1.3 Speech intention (`agent_think.py`)
- **Accumulator:** `I(τ) = I(τ−1) + G·e(τ) + u`, where τ = ticks since the thought opened.
- **Gain:** `G = clip(g0 + β·(V_go − r̄), 0, 3)`, with g0 = 1 for reply thoughts (scene thoughts: 0.3; 1.0 in life) and β = 2.
- **Urgency u:** an innate drive.
  - Final configuration: u = 0.15 per tick for reply thoughts, so an unconfident question still gets a guess within the window. Scene thoughts: 0 (0.02 in life).
  - First runs: u = 0.05.
- **Release** happens when `I ≥ 1` and `p_oth < θ`:
  - θ = 0.25 inside the transition window τ ≤ 7;
  - outside it, or for scene thoughts, θ = 0.05 — but final replies may not start outside the window at all (`late_release=0`).
- **Own-voice tag (final):** own voice = SPK, or the mouth was active ≤ 2 ticks ago (the ear window). develop's tag was SPK or CD.

### 1.4 The two learned parts (local, prediction-error form; exact rules)
Features of a thought (binary; the held PFC line code):
- the thought kind;
- R = "I spoke between the previous heard line and this one";
- each prompt item, plus the first and last item;
- pointing;
- in life only, each visible eye bit.

**P_oth — "another speaker will start soon" (learned brake / NoGo).**
- Keys: each feature × τ (time cells, capped at 30), plus a τ bias key.
- Prediction: `p = clip(b(τ) + Σ wᵢ xᵢ, 0, 1)`. The innate prior is b = 0 for reply thoughts at τ ≤ 7, else 1 (the floor is assumed taken until learned free).
- Target: y = 1 if another speaker's line starts within the next L = 10 ticks, else 0. It is resolved when the window ends or an onset arrives; the presynaptic trace bridges up to 10 ticks.
- Rule: `Δwᵢ = 0.2 (y − p) / n`, where n = the number of active keys.

**V_go — "speaking here pays" (learned eagerness).**
- Prediction: `p = clip(0.5 + Σ wᵢ xᵢ, 0, 1)`.
- Eligibility: the features of the thought that produced the last utterance, held until the partner's reaction.
- Target: y = [reward > 0] on the first tick of the next line by another speaker, if it comes within 50 ticks. In life, no reaction within 50 ticks gives y = 0.
- Rule: `Δwᵢ = 0.2 (y − p) / n`; r̄ is the running mean of y.

Both are the PR rule of `arch/v2/predictive/pr.py`: signed; predicted → no change; predicted-but-absent → weaken. Each synapse sees only its own feature, its target's prediction, and one broadcast event (an auditory onset or the partner's reaction).

---

## 2. (a) B/C/D on the bench (SIMULATED; `out/bench_base_s*`, `out/think_final4_s*`)

| | develop s0 | think s0 | develop s1 | think s1 |
|---|---|---|---|---|
| B all (held) | 0.54 (0.09) | **0.56** (0.11) | 0.55 (0.11) | **0.60** (0.15) |
| B slot / what / where | 0.41 / 0.59 / 0.59 | 0.46 / 0.62 / 0.60 | 0.44 / 0.62 / 0.57 | 0.60 / 0.61 / 0.59 |
| C all (held) | 0.62 (0.20) | 0.61 (0.18) | 0.62 (0.17) | 0.58 (0.17) |
| C colour / count / offer / ritual / taste | 0.14 / 0.36 / 0.86 / 0.93 / 0.76 | 0.14 / 0.36 / 0.86 / 0.94 / 0.74 | 0.14 / 0.36 / 0.88 / 0.92 / 0.78 | 0.15 / 0.35 / 0.89 / 0.74 / 0.77 |
| D all (held) | 0.48 (0.48) | 0.48 (0.46) | 0.47 (0.38) | 0.47 (0.36) |
| D check / name / before / oneshot | 0.77 / 0.17 / 0.01 / 0.00 | 0.77 / 0.22 / 0.01 / 0.00 | 0.76 / 0.30 / 0.00 / 0.00 | 0.75 / 0.34 / 0.00 / 0.00 |
| latency median (mean) | 6 (6.0) | 6 (6.3–6.7) | 6 (6.0) | 6 (6.3–6.7) |
| overlap exchanges | 0 | 1 (D, 13 ticks, self-initiated) | 0 | **0** |
| non-3-tick syllables, garble, extra turns | 0 | 0 | 0 | 0 |
| speaks after the partner's recast (B / C / D) | 0.61 / 0.65 / 0.93 | 0.28 / 0.46 / 0.67 | 0.60 / 0.63 / 0.93 | 0.28 / 0.47 / 0.54 |
| speaks after statements (D) | 0.57 | 0.01 | 0.49 | 0.02 |

**The seed-1 ritual drop (0.92 → 0.74) is not about timing.** The agent says 谢谢 → 不客 and stops: it never produces 气 (0 replies containing it, vs 177 for develop). Its babbling bound only 66 of 67 categories. Because it is not deafened by the GO-driven CD, it heard more in stage A (1,702 vs 1,527 items) and formed an extra, unbound category.

**Seed-0 versions (iteration history, §6):**

| version | B | C | D | overlap exchanges |
|---|---|---|---|---|
| run 1: + serial memory search, urgency gated by V_go | 0.55 | 0.52 | 0.46 | 17 |
| unbounded: develop's steps, u 0.05, late replies allowed | 0.56 | 0.61 | 0.49 | 3 |
| hard window + "no urgency after a reaction to me" | 0.54 | 0.55 | 0.46 | 0 |
| hard window, u 0.15, CD own-voice tag (stopped after B) | 0.45 | – | – | 0 |
| **final:** + own voice from the real mouth tail | 0.56 | 0.61 | 0.48 | 1 |

---

## 3. (b) Accuracy against allowed thinking time

Protocol (SIMULATED; `eval_frozen.py`, `out/frozen_*.out`):
- frozen agents after D, develop's ablation protocol: probe-only worlds, learning off in B/C, the hippocampal index storing in D;
- `budget:k` releases as soon as k steps are done, with the thought frozen at k;
- frozen scores are below the online ones because of develop's known cross-stage interference.

| policy | steps used | B | C | D | latency |
|---|---|---|---|---|---|
| budget 0 (prompt only) | 0 | 0.05 | 0.33 | 0.09 | 6 |
| budget 1 (+ recall) | 1 | 0.06 | 0.33 | 0.18 | 6 |
| budget 2 (+ comparator) | 2 | 0.31 | 0.48 | 0.38 | 6 |
| budget 3 (+ VR) | 3 | 0.35 | 0.45 | 0.40 | 7 |
| free (intention) | 3 at the boundary | 0.35 | **0.37** | 0.39 | 6.2–6.4 |
| v1 (timer, thought completed) | 3 | 0.35 | 0.45 | 0.40 | 6 |

- The unbounded agent gives the same picture: B 0.04 / 0.04 / 0.32 / 0.36, C 0.27 / 0.41 / 0.45 / 0.49, D 0.08 / 0.18 / 0.42 / 0.40.
- **The free policy's C dip is a learned-brake misfire.** Taste questions were never released: the brake had learned "the partner speaks soon" for the items 好 and 吃, which the questions share with the recast 好吃. The hard window then turned that NoGo into silence. Online, taste was fine (0.74).

**Serial search at test time** (run-1 agent; 3 / 4 / 6 / 8 steps):

| stage | 3 | 4 | 6 | 8 |
|---|---|---|---|---|
| B | 0.33 | 0.34 | 0.35 | 0.35 |
| C | 0.45 | 0.46 | 0.47 | 0.47 |
| D | 0.39 | 0.39 | 0.41 | 0.40 |
| D: name | 0.38 | – | 0.46 | – |

- The forced 8-step policy overlapped 299 ticks in D: late starts run into scripted lines.
- **Reading:** thinking steps matter (0 → 3 steps is 7× in B). But develop's whole thought fits inside the timer's window, and more serial search adds 0.01–0.03.

---

## 4. (c) Latency grows on harder items (SIMULATED)

Unbounded run, seed 0 (`out/think_final_s0`). Mean latency in ticks for seen / held-out / correct / wrong replies:

| task | seen | held-out | correct | wrong |
|---|---|---|---|---|
| ritual | 6.7 | – | 6.6 | 8.9 |
| offer | 6.5 | 9.2 | 6.5 | 9.7 |
| taste | 7.3 | 8.5 | 7.2 | 9.5 |
| what | 6.7 | 6.9 | 6.7 | 6.8 |
| where | 6.6 | 11.9 | 6.4 | 8.5 |
| slot | 10.4 | 11.3 | 8.4 | 12.7 |
| count | 12.5 | 14.6 | 11.7 | 13.5 |
| colour | 16.8 | 12.3 | 16.3 | 16.5 |
| check (D) | 6.7 | 6.9 | 6.5 | 7.5 |
| before (D) | 11.7 | – | 15.0 | 11.6 |
| oneshot (D) | 6.5 | – | – | 6.5 (confidently wrong) |

- **Accuracy by latency:**
  - C: 0.96 at latency 6, 0.72 at 7–8, 0.42 at 9–12, 0.36 at ≥ 13.
  - B: 0.70 at latency 6 → 0.23 at ≥ 13.
  - D: 0.64 at latency 6 → 0.16 at ≥ 13.
- **Calibration** (release confidence → accuracy):
  - C: 0.98 at e ≥ 0.6 vs 0.34 at e < 0.1 (latency 6.3 vs 15.9).
  - B: 0.70 vs 0.38.
  - D is inverted at the top (0.52 at e ≥ 0.6): the one-shot probe echoes "这是什么" confidently.
- This is the speed–accuracy signature of evidence accumulation (CITED: Ratcliff & McKoon 2008, Neural Comput 20:873).
- The final configuration compresses latency into 6.1–8.4 ticks with the same ordering. Its calibration still holds in C: 0.98 vs 0.39, and 0.92 correct at latency 6 vs 0.29 at 9–12.

---

## 5. (d) Self-initiated speech

**Staged bench.** The partner always speaks 5 ticks after a scene appears and never waits in silence, so the brake (prior "floor taken") correctly blocks. Of 6,701 scene thoughts, 1 spoke (τ = 8, a D scene change).
- Cause: the brake had learned in stage-A play periods, where the partner waits for babbling, that the floor is free 8 ticks after a scene appears. Here the partner started at τ = 15, giving 13 overlap ticks.
- This is exactly the predicted risk.

**Life** (bench v2.0, seed 0, recast tutor, about 12 h; `out/life_*5_s0*`). Both agents get the same life adaptations:
- extra eye bits that newly appear in silence (the partner's "person" code, an object appearing) join the scene baseline;
- develop's offline babbling pass runs after one day of listening.

| | develop | think |
|---|---|---|
| hours / scored replies | 12.2 / 1,435 | 11.7 / 1,262 |
| free acc / scene (new) / held-out / memory / social | 0.16 / 0.17 / 0.13 / 0.00 / 0.08 | **0.23** / 0.17 / 0.06 / 0.00 / **0.28** |
| bye / goodnight / ritual | 0.04 / 0.52 / 0.41 | 0.37 / 0.75 / 0.46 |
| initiatives (per 1,000 quiet ticks) | 0 | 639 (1.19) |
| appropriate share, overall / partner present | – | 0.17 / 0.32 |
| greeted first (rewarded) / namings (rewarded) | 0 / 0 | **106** / 3 |
| nonsense / alone | – | 237 / 293 |
| reward from initiatives | 0 | 109 |
| overlap ticks / interruptions | 0 / 0 | 0 / 0 |
| longest monologue | 10 syllables | 10 syllables |

**Over life time** (fifths of the life):

| | 1st | 2nd | 3rd | 4th | 5th |
|---|---|---|---|---|---|
| initiatives | 12 | 51 | 98 | 250 | 228 |
| appropriate share | 0.00 | 0.04 | 0.15 | 0.19 | 0.19 |
| greetings first | 0 | 0 | 14 | 48 | 44 |

So self-initiated speech appears, gets rewarded, and is learned. Greeting first grows because:
- the partner's arrival opens a scene thought;
- a rewarded greeting raises V_go for the arrival features;
- the partner's "你好" right after it is learned in the scene context.

The agent does **not** learn to name what it sees:
- With no prompt, the only readout predicts the next heard line, so it says the partner's typical lines: "这是什么", "你要牛奶吗", "饼干放在哪里了".
- It also keeps saying "你好" alone, because V_go's baseline r̄ is low.

Reference: the bench's privileged RandomNamer makes 2.31 initiatives per 1,000 quiet ticks, 0.21 appropriate overall (about 0.8 with the partner present).

**Superseded runs.** My first life runs had an eye-rule bug: they absorbed the persisting pointing after a question. That cost the think agent its pointed answers (think 0.07 vs develop 0.18). Superseded by the fixed runs above.

---

## 6. (e) Overlaps and monologues

Iteration log (SIMULATED, seed 0). Short runs are A 100 + B 400; develop scores 0.17 with 0 overlap there.

| # | version | B | overlap | cause | fix |
|---|---|---|---|---|---|
| 1 | intention + brake; own speech censors the brake's window | 0.12 | 804 ticks | every reply at latency 14: the brake never learned "the partner waits after a question", because the agent's own speech censored those windows | resolve every window; fix the R feature |
| 2 | no censoring | 0.11 | 875 ticks | a thought outlived its dialogue and spoke into the next first prompt | close the thought on scene change or scene off |
| 3 | + transition-window prior | 0.15–0.16 | 13–29 ticks | releases at τ = 8–9 with the brake at 0.19–0.23 | stricter threshold (0.05) outside the window |
| 4 | + late threshold | 0.17 | 0 | – | – |
| 5 | full run 1 (+ serial search, gated urgency) | 0.55 / 0.52 / 0.46 | 17 exchanges | see below | see below |
| 6 | unbounded (no serial search, u 0.05, reflex) | 0.56 / 0.61 / 0.49 | 3 exchanges | late releases after a recast, past the world's 12-tick wait, ran into the next dialogue's scripted line | hard transition window |
| 7 | hard window + "no urgency when R = 1" | 0.54 / 0.55 / 0.46 | 0 | R = 1 also marks the partner's next *question* when the agent spoke after a recast, so it got no guess | rule dropped |
| 8 | hard window, u 0.15 | B 0.45 | 0 | **the CD trap:** a refused GO still runs CD for 4 ticks; develop treats CD as its own voice; the partner's next line starts inside that window, its onset is never detected, and the whole prompt is lost | own voice = the real mouth tail |
| 9 | **final** | 0.56 / 0.61 / 0.48 (s0), 0.60 / 0.58 / 0.47 (s1) | 1 / 0 | – | – |

Run 1's three problems:
- **Overlaps:** late releases after a recast where the brake had learned "free". When the agent speaks, the world waits for it, so "free" is self-fulfilling; when it does not, the partner starts at τ = 10, inside the blind window. Fix: the reflex.
- **Colour silence (spoke 29 %):** the urgency gate removed every guess. Fix: constant urgency.
- **Rituals 0.93 → 0.64:** the serial search adopted a more decisive but wrong episode — a stored recast 早上好 whose next line was a count question, giving "有几个牛奶". Steps after release also changed the plan to below the floor. Fix: serial search off in the main configuration.

**Lesson (PROVED by construction; seen in 5 and 6).** In a world that waits for whoever speaks, "the floor is free now" depends on the agent's own action. A learned turn-taking brake therefore cannot guarantee 0 overlaps; reflexes on the current input are needed. The v1 timer never needed them because its GO only ever followed a line. The predictive prototype found the same: "say the prediction" made its agent speak the partner's next prompt until a learned Go/NoGo existed (`arch/v2/predictive/report.md`).

**Ablations** (short runs, final configuration; `out/abl_*`; ±0.03 is noise):

| removed | B | answer rate | latency | speaks after recast | overlap |
|---|---|---|---|---|---|
| none | 0.16 | 0.75 | 6.6 | 0.51 | 0 |
| learned brake (η = 0) | 0.16 | 0.76 | 6.6 | 0.66 | 0 |
| learned eagerness V_go (η = 0) | 0.16 | 0.76 | 6.3 | 0.63 | 0 |
| urgency (innate drive) | 0.13 | 0.59 | 7.3 | 0.12 | 0 |
| hard window / + reflex | 0.16 | 0.76 | 6.6 | 0.51 | 0 |

- Both learned parts reach behaviour: turn-taking and latency. In life they are what produce, and then shape, the initiatives.
- Neither changes accuracy.
- The window and the reflex only matter where the long C/D runs met scripted silences: 3 and 17 overlap exchanges without them.

**Monologues:** none in any run. One thought gives at most one utterance; 0 extra turns; the longest reply is 10 syllables, the same as develop (C has a 10-character reply in both).

---

## 7. (f) Learning in use (seen items; first → last third; 8-bin curves in `out/collect_*.json`)

| | develop s0 | think s0 | develop s1 | think s1 |
|---|---|---|---|---|
| B | 0.43 → 0.71 | 0.47 → 0.74 | 0.42 → 0.73 | 0.49 → 0.80 |
| C | 0.61 → 0.73 | 0.60 → 0.73 | 0.63 → 0.73 | 0.59 → 0.68 |
| D | 0.43 → 0.49 | 0.44 → 0.49 | 0.42 → 0.52 | 0.41 → 0.52 |

8-bin curves for B:
- seed 0: develop 0.23 0.50 0.65 0.71 0.72 0.70 0.74 0.67; think 0.26 0.55 0.67 0.71 0.79 0.72 0.77 0.71.
- seed 1: develop 0.21 0.48 0.64 0.71 0.76 0.79 0.74 0.67; think 0.26 0.58 0.72 0.71 0.78 0.76 0.78 0.81.

The intention does not change what is learned. Listening learns from the completed thought whatever the agent says, so the curves track each other. B is slightly higher with the intention: the 0.49 vs 0.42 early value comes from fewer silent replies and better hearing (no CD deafness).

---

## 8. Honest answer to the designer's three claims

1. **"The old models only learned to reply fast, with no sustained thinking in the PFC."** Partly. develop's reply was fast but not thoughtless: its whole thought (prompt held in PFC-1, recall, comparator, recognition) ran before the fixed boundary, in zero time. With one tick per step it still fits inside the timer's 4 ticks. The limit was the absence of thought operations that would profit from more time, not missing time.
2. **"Given time to keep thinking, the model would learn to think and then speak."** Not in this architecture.
   - What it did learn is *when* to speak: confidence-gated, turn-aware, eager where speaking pays, silent after a partner's confirmation.
   - The thought operations themselves are fixed. More time only helps if more steps add reliable information; my serial memory search did not.
   - "Learning to think" needs a learned controller of thought operations: BG gating of which operation runs next, rewarded by the confidence gain. Not built here.
3. **"It could speak whenever it is thinking."** Yes, mechanically: scene thoughts speak in life, and greeting first was learned (106 times). But appropriate initiative needs:
   - a production target other than "predict the next heard line" — otherwise the agent says the partner's lines;
   - a partner who waits — on the staged bench initiative is unsafe;
   - innate reflexes plus a conservative learned brake.

---

## 9. The minimal mechanism that keeps the body's 0-error mechanics

| # | mechanism | why (evidence) |
|---|---|---|
| 1 | F brake on GO (kept from v1) | never talk while the other talks |
| 2 | first-syllable reflex: `0.01·E(t) < 0.1` | the GO → R1 → boundary pipeline has a 2-tick blind window; overlaps in run 1 |
| 3 | own voice = the real mouth output (≤ 2 ticks ago), not the GO-driven CD | a refused GO deafened the agent and the partner's next prompt was lost (B 0.56 → 0.45) |
| 4 | hard turn-transition window: a reply starts ≤ 7 ticks after another's line | the staged world waits ≥ 12 ticks after any line only if the agent starts inside that wait; late starts overlapped in both unbounded runs |
| 5 | one utterance per thought (closes at the utterance's end) | no monologues, 0 extra turns |
| 6 | self-initiated speech only through the learned brake with prior "floor taken" | initiative is safe only where the partner waits (life: 0 overlap; staged: 1 exchange) |

Everything else in the body is unchanged.

---

## 10. New fixed mechanisms (justification → plan to learn), and learned parts

**Fixed (innate):**
- **GO gate + the F brake.** Justification: v1's reflex, now gated. Plan: none needed.
- **First-syllable reflex.** Justification: a speaker yields when the other starts (CITED: Schegloff 2000, Lang Soc 29:1). Plan: tune its threshold by learning.
- **Own voice from the mouth output.** Justification: an efference copy of the produced syllable. Plan: none needed.
- **Thought lifecycle and one utterance per thought.** Justification: one selected action per decision (BG). Plan: a learned thought hold/close (assemblies).
- **Fixed order of thinking steps.** Justification: develop's operations, one per tick. Plan: a learned BG controller of operations.
- **Accumulator (θ = 1, urgency 0.15).** Justification: the urgency-gating model (CITED: Cisek, Puskas & El-Murr 2009, J Neurosci 29:11560). Plan: learn the urgency from how long partners wait (P_oth already learns it).
- **Transition window τ ≤ 7 (prior of P_oth and hard limit for replies).** Justification: transition-relevance places (CITED: Sacks, Schegloff & Jefferson 1974, Language 50:696); humans' gaps are about 200 ms (CITED: Stivers et al. 2009, PNAS 106:10587). Plan: in reactive worlds the brake learns it.
- **Time cells τ (one-hot ≤ 30).** CITED: MacDonald et al. 2011, Neuron 71:737; Mello, Soares & Paton 2015, Curr Biol 25:1113.
- **Life only:** absorb newly appearing eye bits; eye bits as turn features; "no reaction → V_go target 0".

**Learned:**
- **P_oth:** speech after recasts −0.15; 3× more blocked releases; in life it learned the arrival window, which enables greeting first.
- **V_go:** latency and post-recast speech; in life it raised initiative where it was rewarded (greetings).
- **Not learned here:** the content of thinking.

---

## 11. Issues for the coordinator (not silently forked)

1. **develop's babbling reads `world.voice`** (offline enumeration of all muscle sets). This is a privileged shortcut against the bench rules. I inherited it in both baseline and think, so comparisons are fair, but it should become bench `body.babble` + `keep='reward'`.
2. **Staged world traps for any agent whose GO is not tied to a line end. Please document them next to the CD trap.**
   - The boundary checks F(t−1), a 2-tick blind window.
   - The world does not wait in scripted silences: the 5-tick scene lead, D's change gaps, and after the 12-tick wait.
   - CD runs after a refused GO and deafens CD-as-own-voice agents.
3. **Life-mode traps.**
   - The partner's 8-unit "person" code appears in silence; parsers that learn pointing from "extra eye bits" (develop) take it for pointing.
   - Pointing persists into the silent answer window, so "new bits in silence" must use onsets, not presence (my own bug, fixed).
4. **Life, recast tutor.** Greetings and goodbyes are rewarded but never modelled, so listening learners cannot learn them except through reward; RefV1 learns them only in force mode. Suggest a modelled greeting in recast mode, or document this.
5. **Minor:** in `run_curriculum` logs, D-routine fillers keep B/C stage labels, so learning curves by `dlg` need the world's dialogue range.

---

## 12. Files (`arch/v2/think/`)

- **Code:**
  - `body_think.py` — GO gate, reflex;
  - `core_think.py` — staged thoughts, serial search, confidence;
  - `agent_think.py` — intention, P_oth, V_go, own-voice tail, life eye;
  - `world_probe.py` — my silent-scene probe; superseded by life mode, not used for results.
- **Runners:** `run_think.py`, `run_life.py`, `eval_frozen.py`, `run_bench.py` (develop baseline, logs saved).
- **Analysis:** `analyze.py`, `collect.py`, `life_summary.py`, `abl_table.py`, `diag_c.py`, `diag_rit.py`; `test_equiv.py` (739/739).
- **Batches:** `b5.sh`, `b6.sh`, plus the earlier `batch*.sh`.
- **Outputs (`out/`):**
  - main: `think_final4_s0.*`, `think_final4_s1.*`, `bench_base_s0.*`, `bench_base_s1.*`;
  - variants: `think_free_s0.*` (serial search), `think_final_s0.*` (unbounded), `think_final2_s0.*`, `final3_s0.stdout`;
  - frozen tests: `frozen_final4_s0.out`, `frozen_final_s0.out`, `frozen_ior_s0.out`;
  - life: `life_think5_s0.*`, `life_dev5_s0.*` (fixed; the earlier `life_think4` and `life_dev` runs are superseded);
  - ablations: `abl_*.stdout`.
- The final seed-0 agent is pickled as `think_final4_s0_agent.pkl`.
- **Drafts:** `draft_mech.md`, `draft_iter.md`, superseded by this report.
- **Compute:** local CPU only, one heavy process at a time; each run under 20 minutes. The GPU runner was not used.

---

## summary_zh

1. 按设计者的原则改了 develop 原型：说话由"想"触发，不再由固定的 GO 定时器触发。想法每个节拍走一步（回忆、比较、识别），读出的把握度不断累积，到达阈值后经基底节式的门控变成 GO；"别人在说话就不开口"的先天刹车保留。
2. 什么时候说话由两个局部的预测误差学习器决定：一个预测"对方马上要开口"（从听到的轮流说话中学），一个预测"在这里开口会有回报"（从对方的反应中学）。
3. 回答质量没有提高：种子0 B/C/D 0.56/0.61/0.48，develop 0.54/0.62/0.48；种子1 0.60/0.58/0.47，develop 0.55/0.62/0.47。原因：develop 的整个"想"只需3步，正好放进旧定时器留出的4个节拍里，旧模型其实并不缺思考时间。
4. 强行让它少想会明显变差（B：0步0.05，3步0.35）；让它多想（多做几步记忆搜索）几乎没有帮助，训练中还弄坏了礼貌用语。
5. 时间信息有用：难题、没学过的题、答错的题都说得更晚；它自己的把握度能预测对错（C 题把握高时98%对，把握低时只有34–39%）。它也不再"别人每说一句就接一句"：在对方复述之后再开口的比例从0.61降到0.28，在陈述句后几乎不再说话（0.57→0.01）。
6. 主动说话：在测试台上对方从不等待，主动说话被正确压住（6,701次里只说了1次，而且撞上了对方）。在"一生"模式里，它学会了对方到来时先打招呼（106次，有奖励，而且越来越多），但几乎不会给看到的东西命名（3次），常说对方的台词或自言自语。原因：它唯一的读出是在"预测下一句会听到什么"。
7. 风险是真实的：由想法触发的 GO 可能落在任何时刻。能保持0错误机制的最小机制是：保留 F 刹车；首音节要求"此刻耳朵安静"；用真实的嘴部输出而不是 CD 计时器来标记"自己的声音"；回答必须在别人说完后7个节拍内开始；一个想法最多说一句；主动说话只走学到的刹车。最终版本两个种子在回答路径上都是0重叠、0错误音节、0多余轮次。
8. 结论：这个原则在时机、轮流说话和主动性上是对的架构，但在这个测试台上不带来内容提升，还需要先天反射来保证安全。要真正受益，需要更长且可靠的思考操作，并为主动说话提供"说出我注意的东西的名字"这样的产出目标。
