# ML review of ARCH_v3_core (continual learning, credit assignment without backprop, scaling)

**Reviewer role:** ML researcher. I reviewed only. I modified nothing in the repo and launched no server jobs.

**Toys:** 6 local numpy scripts, one process at a time, peak RSS 250 MB or less, each run 3 min or less. They are in `docs/v3design/review/ml/`:
- `drift2.py` / `.out`: re-check and stress test of `drift_toy`
- `ca3_capacity.py` / `.out`: CA3 capacity against size
- `mix_arith.py` / `.out`: MIX activity arithmetic and the CA3 fix
- `merge_toy.py` / `.out`: unit claims and merge operators
- `meta_toy.py` / `.out`: the designer's metaplasticity idea
- `ga_toy.py` / `.out`: cost of the lineage gene search

**Labels:** PROVED = code or arithmetic; SIMULATED = my toy; CITED = a paper or report; ASSUMED = my judgement.

## What holds up

- **The drift_toy claim holds and is robust inside its own world** (SIMULATED, `drift2.out`, 5 seeds, formats 600 → untutored, 30 % forced silence). Untutored accuracy at 300 / 1,000 / 3,000 / 10,000 dialogues:

  | condition | 300 | 1,000 | 3,000 | 10,000 |
  |---|---|---|---|---|
  | ST, synth's setting | 0.99 | 0.99 | 0.99 | 0.99 |
  | S (self in stream, no age), synth's setting | 0.09 | 0.00 | 0.00 | 0.00 |
  | ST, per-event jitter ×[0.5, 2] on every world delay (also the give-up and stop gaps, which drift_toy did not jitter) | 0.96 | 0.96 | 0.96 | 0.96 |
  | ST, graded log-Gaussian time cells (σ 0.85 octave) | 0.98 | 0.98 | 0.98 | 0.98 |
  | ST, graded cells + ×[0.5, 2] jitter | 0.97 | 0.98 | 0.96 | **0.93** |

  The last row shows slow erosion starting. So the anchored context stops readout interference under realistic timing noise.
- **The sign-elected merge operator works on readouts over shared features** (SIMULATED, `merge_toy` part 2).
  - Pool-of-4 merge, lessons only one parent learned: plain mean 0.08; TIES 1.00; use-weighted 1.00; keep-larger 1.00.
  - Re-wiring 30 % of the features in one parent costs at most 0.06 (plain mean); TIES, use-weighted and keep-larger stay at 1.00.
  - Conflicting lessons come out a coin flip under every operator (0.45–0.65). This is as the doc predicts.
- **The world-first order (W1–W3) is right.**

## Issues, most severe first

### 1. FATAL (as stated): speech release has no working stop, so "0 extra turns / 0 garble after maturity" cannot be reached

**Evidence** (PROVED from the doc's equations and today's code):
- **The accumulator.** §1.2 M4 gives `I ← I + (G+u)·e` with `G = clip(Go − NoGo)`. Today's `agent_think.py` clips G at 0.
  - With G ≥ 0 and u = 0.15, NoGo can only stretch the time to release to 1/(u·e) ticks. It can never prevent it.
  - After my own answer, the refractory period τ_ref = 15 ends. Any partner line GEN predicts with margin e = 0.2–0.6 is then released after 11–33 ticks.
  - Partner lines arrive 10–40 ticks after my stop (corrections) or 0–60+ ticks (questions) (CITED `cue_test_base.txt`). Many are therefore pre-empted by the agent saying them: extra turns, or v2.2's "says partner lines".
  - §6.4's "about 7 events stop late starts" is false under the clip.
  - `checks.py` uses a different formula, `G·e + u`, under which a release happens after 7 ticks even at e = 0.
- **P_oth is deleted with no replacement.** §1.2 says P_oth "becomes part of GEN's time-specific prediction", but nothing in the M4 equation reads that prediction.
- **P5b rewards any utterance a partner onset follows.** That includes extra turns and role-confused lines. In untutored dialogues it pays late answers more: the efference trace at the next question is e^(−37/20) = 0.16 for an answer at τ 5, but e^(−13/20) = 0.52 for one at τ 30.
- **ε and babbling never fade.**
  - ε = 0.05 is release-time exploration (G22 replaces `veto_eps`). About 1,200 own answers after maturity × 0.05 ≈ 60 exploratory releases, each an extra-turn risk.
  - r_bab = 0.01 per idle tick never decays, which gives garble.
- **W4 removes the training signal.** Polite caregivers never start while the agent speaks, and the brake stops the agent starting while they speak. So the 7–10 overlap events §6.4 counts on cannot occur in W4 worlds.
- **NoGo would be per question.** MIX is anchored on item content, so a NoGo learned for one question does not carry over to others. About 600 childhood questions are spread over hundreds of contexts.

**Smallest general fix:**
- (a) `G = Go − NoGo` unclipped below, with I floored at 0.
- (b) An STN-like hold: GEN's predicted probability of a non-self onset in the next L ticks raises the release bound. This is P_oth rebuilt as GEN + M4; the doc's own circuit already lists the STN.
- (c) P5b as contingency: reward = the person onset minus the predicted onset given the non-self timeline. Watson's contingency is a conditional probability, not co-occurrence.
- (d) ε and r_bab scaled by the same use-dependence as η.
- (e) W4 politeness imperfect (about 0.8), so overlaps still teach.
- (f) A cheap local experiment X6 of M4 release on the symbolic or staged world, run before L7.

### 2. FATAL for §2.4 as stated: every generation's "new world seed" changes the world's physics, so inherited perception and motor weights are invalid

**Evidence** (PROVED from code):
- Bench `physics.Voice.__init__` draws every character's sound pattern and muscle set from `rng(seed)`. Lawful physics uses `seed + 2000`.
- `world.Curriculum(seed)` draws the eye codes.
- So the following are all keyed to sounds and sights that change each generation:
  - the inherited SYL templates;
  - the forward and inverse vocal maps;
  - VR and LOC;
  - every MIX/GEN/gate weight anchored on those items, including the "inherited turn timing".

**Fix:** split the seed.
- A lineage-fixed physics seed covers voices, mouth physics and eye codes: the language and the laws of nature.
- A per-life episode seed covers sampling, timing, noise and caregivers.
- The sealed opening keeps the physics. A physics change would be a different test, a new language.

### 3. SERIOUS: MIX is about 20× sparser than assumed, and the fixed fan-out GEN then reaches only part of the target code

**Evidence** (PROVED, `mix_arith.out`, using the `checks.py` sizes and RandMix's branch rule: a branch fires if at least 1 of its 3 lines is active, and a unit is eligible only if all 3 branches fire):
- A 6-syllable line activates **21–28** MIX units; a 1-syllable event activates **3–4**. `checks.py` assumes 480 (1 %); v2.2 had about 365.
- With GEN's fixed fan-out of 64 of 2,049 outputs, the share of output units reachable from the active set is 1 − (1 − 0.031)^|A|: **0.49** at |A| = 21 and **0.12** at |A| = 4.
- If k-WTA instead ranks units by summed branches, about 60 % of the 2-branch winners have no active anchor. That brings back ST0's unanchored drift.

**Fix:** a general sparsity rule. Each branch's fan-in ∝ 1/(the source area's active fraction), so P(branch) ≈ 0.4. That is about 42 lines on LATEST, 11 on THOUGHT and 13 on EYE. Then about 3,000 units are eligible and all 480 winners are anchored.

### 4. SERIOUS: pure-neuron CA3 holds 10–20× fewer episodes than today's store and does not grow with s under the scaling rule

**Evidence** (SIMULATED, `ca3_capacity.out`; Willshaw net, 50 % cue, best one-step threshold):
- At fan-in 1,000 and 1 % active, capacity (F1 ≥ 0.9) is about **1,000 / 500 / 500 / 500 patterns at N = 2.5k / 5k / 10k / 20k**. F1 at M = 2,000 is 0.78 / 0.72 / 0.70 / 0.68: flat in s.
- A dense net at N = 10k holds 16k or more.
- Fan-in 2,500 holds 4k at F1 0.99; fan-in 5,000 holds 8k at F1 1.00.
- The stability story leans on this store: own lines replayed with age, recall bridging late answers, and the 10,000-dialogue bar. v2.2's onset at 2,100–3,000 dialogues already matched turnover of a 10k store.

**Fix** (cheaper than the plan): make CA3 a dense bit matrix. That is 10⁸ bits = 12.5 MB at s = 1, against 80 MB for the planned sparse net, and recall is about 16k word operations. Alternatively scale CA3 fan-in ∝ s. State in §4 that CA3 connectivity is the one fan-in that scales.

### 5. SERIOUS: the merge prerequisites fail

- **"Order-independent unit claims" are order-dependent** (SIMULATED, `merge_toy` part 1). Sibling assembly Jaccard:
  - 0.66 at 25 % fill, 0.31 at 94 % fill, 0.20 with junk recruitment;
  - sound-alike syllables: 0.24–0.43;
  - even an order-free top-k code gives about 0.72 when the siblings' templates differ at cosine 0.99.
  - Compatibility check (ii) needs ≥ 0.9, so breeding is blocked.
- **SYL is oversubscribed** (PROVED): 141 lexicon characters × 16 units = 2,256 > 2,048 units.
- **The migration ladder is out of order** (PROVED from §8). L11 runs lineage generations 0–2 on the shortcut build. Its item IDs are numbered in hearing order (§2.4a-3), and the S7 assemblies only arrive at L12.

**Fix:**
- Put an explicit within-lineage alignment into the merge operator: activation matching on a shared probe stream (Git Re-Basin, Ainsworth et al. 2023; OT fusion, Singh & Jaggi 2020). The operator lies outside the agent, so #21 is untouched.
- Size SYL at 4 × lexicon × k or more.
- Run the lineage after S7, or with the alignment.

### 6. SERIOUS: generalisation costs, mostly in gating, not in the anchor itself

- **Same-timing transfer is not hurt by the anchor** (SIMULATED above). The costs are elsewhere.
- **(a) Source coding blocks self → other transfer.** Recasts do not teach content; the doc accepts this.
- **(b) Item-specific ages limit word-order paraphrases** (PROVED with the §6.0 bins). In a 6-syllable question at τ 5, the first and last items are 23 and 8 ticks old, so they share only bin 3. At τ ≤ 2 they share nothing.
- **(c) The M4 gates read only item-anchored MIX.** That contradicts the "indirection" claimed in P2d (Kriete et al. 2013). Gating learned for one question does not transfer to new fillers.
- **(d) The §6 numbers are about 2× too optimistic.** A prediction learned at τ 10 and read at τ 4–7 shares one of its two bins. So p ≈ 0.44, not 0.73–0.88. The greeting race (first syllable at τ 24–28 against the partner at 30) is lost at W5 timing ×0.75.

**Fix:** give the M4 gates (not GEN) a content-free key population: source × age bin × P4 × event kind. Gates do not predict content, so this does not bring back ST0's drift. Add held-out arms to X1: paraphrase, different answer latency, new fillers.

### 7. SERIOUS: "new kinds" with one three-factor rule plus random conjunctions

**Evidence** (ASSUMED, plus CITED v2.2):
- The function class is linear readouts and actor-critic over 3-way random conjunctions. MIX learns only by instar and pruning: the lazy / kernel regime (Chizat et al. 2019), with no task-driven intermediate codes.
- v2.2's class-dependent probe scored 0.25 / 0.50 / 0.75: "no class generalisation from any layer".
- Multi-step internal operations are trained by a scalar δ with τ_e = 20, with high variance and per-context learning.
- The M4 source list (GEN, copy/echo, CA1, VR, LOC) is today's routes renamed: the what / where / name task heads. A new kind that needs a new route needs a new mechanism.

**Fix:**
- Run X4 first; it is local and under 1 h.
- Pre-register the fallback: GEN's error fed to MIX through fixed random feedback as the instar's third factor (feedback alignment, Lillicrap et al. 2016). That is still M2.
- Make every area that projects to the programs a source.

### 8. SERIOUS: the scaling rule is nearly inert along s

**Evidence:**
- At fixed fractions and fan-ins, the MIX kernel (active-set overlap statistics) does not depend on N (PROVED). CA3 capacity is flat in N (SIMULATED).
- The data budget is 2 × 10⁻³ to 10⁻² syllables per learned weight (CITED learning member), against about 20 tokens per parameter (Hoffmann et al. 2022). In that data-limited regime, more parameters do not lower error (Kaplan et al. 2020; Bahri et al. 2021).
- Prediction: the (0.5, 1, 2) diagonal comes out flat or noisy. It cannot tell "doesn't scale" from "data-limited".

**Fix:**
- Scale CA3 connectivity (issue 4).
- Pre-register the expected exponent.
- Add a diagnostic arm that grows only experience at fixed s. The whole-model test stays as the designer decided.

### 9. SERIOUS: the stability claim is scoped to readout interference only

The toy's stability means "no gradient reaches the answer synapses". The following are untested:
- value drift under P5b (issue 1);
- drift of the MIX/SYL substrate, which is unrecoverable without a teacher except through replay, and replay is capacity-limited (issue 4);
- release behaviour.

**Fix:** L6 must measure the full core. X1 must keep the value loop and MIX plasticity on.

### 10. The designer's metaplasticity idea (coordinator's question)

**Is it enough to stop the collapse? No.**
- **PROVED:** with any positive per-synapse rate factor D(w), an error-driven rule has the same fixed point under a stationary stream (E[x(t − p)] = 0 for any D > 0). Metaplasticity rescales time; it cannot change where the weights end up.
- **SIMULATED** (`meta_toy`; one context; 30 % of presentations carry a wrong target, the v2.2 failure kind):

  | rule | survival (presentations) | correction latency |
  |---|---|---|
  | plain NLMS | 10 | 2 |
  | \|w\| form | 19–43 | 4–7 |
  | per-synapse use | 32 | 4 |
  | designer's \|w\| + use | **75** | **7** |
  | 2-variable Benna–Fusi | 1–3 | 1 |

  - Survival of 75 presentations is about 2,400 dialogues in a 32-question world, so it delays the collapse rather than preventing it.
  - Benna–Fusi does worst here because, with few training events, its visible weight is pulled back to a hidden weight that never consolidated.
  - A corrected habit never returned in any rule.
  - The trade-off is one dial: survival / latency ≈ 5–11 for every rate form.

**Interaction with NLMS:** if the cause is a cliff (pruning, store turnover; X2 is testing), metaplasticity on readouts does not touch it.
- **PROVED:** with ν = 1 normalisation (the one that avoided blocking in `v2/tune`), a context's prediction changes by η·err per event whatever the rates. That gives no behavioural protection against wrong targets in the same context.
- With ν = 0 there is protection, but the per-row form blocks new learning (the tune finding).
- |w| protection is not size-free: w ∝ 1/|A| ∝ 1/s.

**Stability vs plasticity:** "never forget in a lifetime" also means "never correctable". That contradicts #20, because childhood errors must be unlearned.

**Relation to the literature:**
- The rate forms are Adagrad-like preconditioning.
- EWC / SI add an anchor that moves the fixed point, but they need task boundaries, which one unbroken life does not have.
- Benna–Fusi is the moving-anchor version.
- Masse et al. 2018 found context gating plus mild stabilisation works best. Here the anchor is the gating.

**Recommended form:**
- `η_ij = η·max(1/(1 + c_ij/c0), floor 0.1–0.2)`.
- c_ij counts *confirmed* co-activity per synapse i→j: pre active with target present. For value synapses, count δ > 0 events.
- Symmetric, normalised by |A|.
- A new target in the same context then learns at full rate, while unconfirmed childhood errors stay plastic.
- Cost: +4 B per plastic synapse (about +12 MB for GEN).
- Treat it as insurance, not the cure. In X2, report retention together with n₅₀, correction latency and childhood-error unlearning.

### 11. MINOR (evolution cost and loose ends)

- **The GA itself is affordable** (SIMULATED, `ga_toy`). With P 12 × 10 generations = 120 lives, it recovers 44–65 % of the regret when a ±20 % gene change matters by at least 0.25 seed-SD. Fake genes wander about 8 %.
- **The gene audit is expensive** (PROVED). Detecting effects of 1 / 0.5 / 0.25 SD unpaired needs 1.4k / 5.6k / 22.6k lives. Use paired worlds with common random numbers.
- **Lamarckian confound.** Siblings share merged weights, so selection picks families and genes ride along. Add a Darwinian control lineage (genes only).
- **Scoring favours silence.** A post-maturity error scores −1, so add a speaking-rate term.
- **The G8 range is fitted to the bench.** Its upper bound 0.15 comes from the bench's 0.19 margin.
- **η_v is inconsistent:** 0.3 in the doc against 0.2 in `checks.py`, giving 7 against 10 events.

## Migration and experiments

- X1 and X2 first is right.
- The three biggest structural risks are tested late or never:
  - release control: L7;
  - CA3 capacity: L12;
  - merge prerequisites: L11.
- Add these cheap local checks before building: X6 (M4 release), a CA3 sizing check, an assembly-agreement check, and X4 run in parallel with X1.

## Verdict: fix, then build

The anchored-context idea is sound for readout stability. Release control, sizing and the lineage prerequisites are broken as written. The 3 fixes that matter most:
1. **Release control:** unclipped NoGo, plus an STN hold from GEN's predicted partner onset, plus contingency-based P5b, plus ε and babbling that fade with use. Test with X6.
2. **Sizing for the pure-neuron build:** branch fan-in ∝ 1/source sparsity; a dense-bit CA3 (or CA3 fan-in ∝ s) written into the scaling rule; SYL at 4 × lexicon × k or more.
3. **Lineage prerequisites:** a lineage-fixed physics seed; activation-matching alignment inside the merge in place of "order-independent claims"; lineage after S7; a Darwinian control arm.

## summary_zh

- **能保住的：** 我用自己的小模型复查了"最近一件事×过了多久"这个结构。所有延迟都打乱×[0.5,2]、时间细胞改成渐变调谐，无人指导1万段对话后仍有0.93–0.99。同一血统内"同向平均"的融合也确实有效。
- **致命一：说话的闸门停不下来。** 紧迫度u加上"NoGo不能低于0"，只要预测到对方要说的话，到时间就一定说出口。"被回应"奖励还会鼓励多说、抢对方的台词。探索和咿呀永不消退。"长大后0错误"按现在的公式做不到。
  - 修法：NoGo可以压到负值；GEN预测"对方快要开口"时提高放行门槛；"被回应"只奖励原本不会发生的回应；探索和咿呀随熟练度下降。
- **致命二：每代换世界种子，就把声音和眼睛编码一起换了。** 继承的听觉、发音和"何时说什么"全部作废。
  - 修法：一个血统固定"物理/语言"种子，每一生只换经历、节奏和噪声。
- **严重问题：**
  - 组合层实际每次只有约20个单元激活，单音节事件只有3–4个，不是设想的480个；
  - 纯神经元海马在固定扇入下只能存500–1,000段，而且不随规模变大。改成稠密比特矩阵反而更省内存，容量×16；
  - 音节单元的认领与先后顺序有关，兄弟之间一致度只有0.2–0.66，过不了0.9的融合门槛；
  - 闸门只读带具体内容的特征，学会的轮流说话不能迁移到新问题；
  - 按现在的规模规则，放大s几乎不改变模型能学什么。
- **设计者的"常用的强权重难变"：** 我证明了它只拖延遗忘，不改变最终结果。小模型里能多撑7.5倍，代价是纠错慢3.5倍。若"一辈子不忘"，也就一辈子改不了，和"童年可以犯错、长大要改掉"矛盾。
  - 建议：每个突触只记"被证实的使用次数"，学习率按它下降但留下限；当作保险，不当作解药。
- **结论：** 先修再建。最重要的三处：说话闸门、纯神经元的尺寸（组合层扇入、稠密海马、音节区大小）、血统前提（固定物理种子、融合前先对齐单元、加只遗传基因的对照组）。
