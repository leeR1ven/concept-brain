# Adversarial review of ARCH_v3_core: skeptical research engineer

**Scope.** I read ARCH_v3_core and its scratch scripts (checks, drift_toy, gene_map, knob_list). I also read the v2.2 report and logs, `world.py`, `life.py`, world2 `core.py`, `randlayer.py`, `gen_probe_v21.py`, the gpuport report and the eval member's `budget.py`. I ran six small local numpy checks, one process at a time, each under 400 MB.

All scratch files are in `docs/v3design/review/skeptic/`:
- `tick_toy.py` / `.out`
- `tick_variants.py` / `.out`
- `tauf_sweep.py` / `.out`
- `tick_bench.py` / `.out`
- `knob_audit.py` / `.out`
- `arith.py` / `.out`

**Verdict: fix, then build.** There is one FATAL-as-written problem, and it has a one-sentence fix. Five problems are SERIOUS.

**What I checked and found sound:**
- drift_toy reproduces the document's §0 table exactly.
- The checks.py arithmetic is right for its own inputs.
- The time-bin table, the maturity counts (5,000 / 15,457 = 0.32; about 575 questions) and the §6.2 restoration sums are correct.
- Every "starts from" class or function named in §1 exists in the repo (PROVED by grep).

---

## Ranked issues

### 1. FATAL as written: the meaning of O1 decides stability, and the document claims both meanings
**Evidence:**
- §1.4 promises "what comes next, **and when**" and "another speaker is about to start, which replaces P_oth".
- M2 says "t is the target area's **next activity** … predicted but absent → weaken".
- drift_toy, the only evidence for stability, learns once per heard onset: "over the context features F at the heard line's onset". That is event level, with no negatives on silent ticks.

**SIMULATED** (`tick_toy`, `tick_variants`; 5 seeds; ST context; 600 formats, then 3,000 untutored dialogues):

| O1 reading | decision rule | after formats | after 300 untutored | after 3,000 untutored |
|---|---|---|---|---|
| event level | threshold | 0.98 | 0.98 | 0.98 |
| event level | M4 accumulator | 1.00 | 1.00 | 1.00 |
| event level | M4 accumulator, η·max(1/n_use, 0.1) | 0.98 | 0.99 | 0.99 |
| tick level | threshold or M4 accumulator | 0.95–1.00 | **0.00–0.06** | **0.00–0.06** |

- The tick-level collapse happens with 30 % forced silence and also with none.
- In live untutored use the tick-level agent is right 0.01–0.08 of the time, silent 0.46–0.77, and says partner tokens 0.21–0.46. That is v2.2's failure signature.
- With the model tutor (answer at τ = 3), tick level survives only in one arm: threshold decision, release at τ = 1, never silent. That is not the design, which has a 4-tick CD and an accumulator.

**Why (PROVED by the stated rules):**
- In untutored use nobody else ever says the answer.
- Own voice is explained away (§6.1, t = 28).
- So on every tick before my own onset the answer is "predicted but absent" and is weakened. The earliest answer bin goes first, the release moves later, and in the end only the give-up bins remain.

Under event level, GEN gets no negatives on silent ticks. So it cannot learn "when", and the P_oth replacement claim and §6.4's "a new line starts at about 45" have no basis.

**Fix (generic):**
- Gate all prediction plasticity by the target area's onset transient, which is M1's existing event boundary. "Absent" then means predicted items that are not in the next heard event.
- Put timing only in TD (critic over MIX × TIME) and M4.
- Redo §6.1 and §6.4 under this rule.

### 2. SERIOUS: the value loop is untested, and P5b is triggered by the bench's own probe timing
**Evidence:**
- P5b gives reward for "a person-like onset while my own efference trace (τ_e) is high".
- `world.py` `_wait` starts the next untutored probe at max(Q_end + 45, stop + 13).
- My trace at that onset (PROVED, `arith`): 0.16 if I stop at Q_end + 8, 0.29 at + 20, 0.52 at ≥ + 32.

**Consequences:**
- For any threshold for "high" between 0.17 and 0.52, early (normal) answers go unrewarded and late ones are rewarded. The actor is pushed toward releasing in bins {4, 5}, where the anchor holds only give-up and partner content. This is a sixth drift route that the five-route table in §6.5 does not list.
- §6.5 R-e assumes "no reward means V_go → 0". In untutored use that premise is false: the probe partner keeps "responding".
- P5b also rewards echo (a copy source) exactly as much as a correct answer.
- Neither drift_toy nor the X1 harness has any value loop. `gen_probe_v21` speaks through `core.speak()` with its max/END/floor loop. The consequence is ASSUMED, not simulated.

**Fix:**
- World side: every partner, probe partners included, replies at a fixed short delay after the agent's turn, i.e. W4 everywhere.
- Declare the "trace high" threshold as a species constant.
- X1 must include the M4 accumulator, the critic, P5b, ε and the real probe timing.

### 3. SERIOUS: the core cannot meet the maturity rule as specified
§5.3 says: "From M on: 0 of each, in every tenth of life … hard rule". Four things conflict with it:
- **Exploration.** ε (G22, never below 0.01) never anneals. §6.3 itself counts "≈ 80 random utterances" per life, about 53 of them after M. Any ε release after my own answer is an extra turn.
- **Babble.** Babble (P1d, r_bab 0.01 per idle tick) feeds straight into the program layer ("PROGRAMS: M4 output, babble noise"), so learned NoGo cannot suppress it. That gives about 5,700 babble onsets after M (PROVED). Under W4, babble is still answered at 0.3, so P5b keeps rewarding it. Life and world2 count garble per onset.
- **Crowds.** world2 counts an overlap whenever another body is audible while my mouth sounds (`core.py` `overlap_cg` / `overlap_ag`), including bodies that start in the middle of my line. P1c only brakes onsets. The design cites "yielding at overlap (Schegloff 2000)" but no sub-item implements it. L7 runs world2 with 16 bodies, where "any post-maturity error" fails.
- **W4 makes the overlap clause empty.** In W4 worlds the caregiver "never start[s] while it speaks" and the brake stops my onsets, so overlaps cannot happen (ASSUMED tick order). §6.4's learning signal ("W4 caregivers do not answer it") then never fires.

**Fix:**
- Make babble and ε candidates inside M4's release competition. Exploration becomes noise in the Go/NoGo race, so it falls where NoGo has been learned, with no schedule.
- Add one clause to P1c: yield at the next syllable boundary when a non-self voice rises.
- Define the post-maturity criterion only on events the agent causes: talk-over starts, failure to yield within one beat, garble in released lines, extra turns.

### 4. SERIOUS: the compute numbers are floors, ignore L ∝ s, and the GPU figure contradicts the repo's own measurement
**SIMULATED** (`tick_bench`; the document's s = 1 sizes; CSR + bincount + argpartition; local core, A800 core ≈ 1.8× faster):

| tick type | local core | A800 core | document |
|---|---|---|---|
| silent | 2.45 ms | ≈ 1.4 ms | 0.2 ms |
| speech (MIX every tick) | 4.72 ms | ≈ 2.6 ms | about 2.1 ms |
| one replay reactivation | 7.85 ms | ≈ 4.4 ms | not costed |

- The silent cost is mostly dense per-unit work and value traces, not synaptic events. The document's events-only cost model misses it.
- Replay is 8 reactivations per quiet period (G20). In gpuport, awake recall was 70 % of the GPU time.
- **My estimates:**
  - s = 1: about 1,000–2,000 s per 10⁶ ticks.
  - s = 4: about 4,000–8,000 s per 10⁶ ticks.
  - §4 sets L ∝ s, so an s = 4 life is 4×10⁶ ticks: 16,000–32,000 s, or 3–6 chained 90-minute jobs. Even the document's own figures give 8,000–24,000 s.
- **GPU:**
  - The document assumes 0.3–0.6 ms per batched tick, i.e. 5–10 s per agent-life.
  - `v2/gpuport/report.md` measured 259 ms per tick at a batch of 64 on the A800 (250 agent-ticks/s) and 2.56 ms at a batch of 1. The assumption is 430–860× faster.
  - "A batched generation of 64 in 10–20 min" would really be about 72 h.

**Fix:**
- Measure first and treat GPU batching as unproven. Batch or vectorise the world too: 64 Python worlds per tick is a floor of its own.
- Budget s = 4 lives as chained jobs with exact checkpoint/resume.
- Keep the lineage on CPU at P ≈ 12.

### 5. SERIOUS: the representation swap that both merging and pure neurons need comes last
**Evidence:**
- §2.4(a) requires fixed arrays and claims that do not depend on hearing order ("Z categories … numbered in order of hearing, which breaks this").
- v2.2's GEN is 50,000 rows × 257 item-ID columns.
- L11 runs lineage generations 0–2 on the shortcut build, while S7 (IDs → assemblies) and S8 (LATEST list → units) come only at L12. So L11 merges arrays that cannot be merged, and every gene and inherited weight from L4–L11 lives in a representation the final build throws away.
- S6's check ends "or report the limit", so it cannot fail.

**Fix:**
- Bring in birth-match claims, S7 and S8 right after L2, and build L6's LATEST and anchored MIX in unit form once, not twice.
- Run a merge test in the symbolic harness before L11.
- Give S6 a real pass bar.

### 6. SERIOUS: hidden task- or world-specific structure (count: about 16 items)
- **M4 source list (4 items).** It is a "species constant", but 4 of its 5 sources (copy/echo, CA1 successor/completion, VR word, LOC word) are the v2.1 routes COPY_PROMPT/NEXT/COMP, `say_rec` and `say_loc`, renamed. L8 claims to delete exactly these.
- **Priors wired to bench signals (7 of the 17 sub-items):**
  - P1b: the 3-tick beat;
  - P1c: "or the last tick", a patch for reflex #3's blind window;
  - P2c: a separate "person" TIME pathway;
  - P3b: the pointing bits;
  - P4a: the person-code block;
  - P4b: "whatever cue the world offers", which has no bound, and "my name" cannot be innate;
  - P4c: affect cues co-designed with W7.
- **World rules standing in for deleted agent parts (4 items):**
  - W1: the self-answer fades "as the child succeeds", which means a tutor with a model of the learner;
  - W3: the replayed pair builds exactly the answering context the anchor needs;
  - W4: makes overlap impossible and creates δ by hand;
  - W7: approval at 0.8 / 0.1 of correctness is the reward number in perceptual dress.
- **Gene ranges (1 item).** G8's ceiling of 0.15 comes from the bench's measured margins.
- **Not supported:** I suspected the time bins were fitted to W1's 10 ticks. `tauf_sweep` shows the event-level anchor holds 0.95–1.00 for caregiver self-answer delays τ_f = 10–30 (SIMULATED).

**Fix:**
- Count these as per-world adapters and world rules, not as generic parts.
- Test "GEN as the only motor source", with VR/LOC/CA1 as inputs to MIX, against the gateR lesion (0.91 → 0.40, CITED audit).
- Run sealed worlds with W1/W3/W4/W7 partly off.

### 7. SERIOUS: the "PROVED" numbers in §6 do not use the M2/M4 rules as defined
- **Learning rate.** M2 sets η = η_level·max(1/n_use, floor), but §6 uses a constant η:
  - p after 5 exposures: 0.88 claimed, 0.60 with 1/n_use, 0.16 for a long-used synapse at floor 0.1;
  - the NoGo needs 7 events (§6.4), 10 in checks.py (which uses η_v 0.2), and 48–159 with the stated schedule.
- **Trace.** If prediction synapses keep a τ_e trace with m = 1 on every tick, the effective rate is η/(1 − λ) = 7.2, above 2, and the rule diverges (PROVED).
- **M4 formula.** M4 says I += (G + u)·e, but checks.py uses G·e + u. R-e's "urgency still releases" needs the latter.
- **Gene merges retune today's values without saying so:**
  - u 0.05 → 0.15;
  - eta_act 0.05 → 0.3;
  - act_T 60 → 20;
  - MEAN's set-point τ 20,000 → 800,000 (level 2 × 40²);
  - ct_assoc_eta 0.02 → 0.004.

**Fix:**
- Define n_use and the floors.
- Keep traces on m = δ synapses only.
- Re-run checks with the stated rules.
- Add a paired ladder step for the gene merges.

In the toy, 1/n_use did not break event-level stability.

### 8. MINOR (the brief asked): the knob arithmetic is a correct census with a misleading headline
- The 189 "deleted" knobs split as (PROVED, `knob_audit`):
  - 84 switches;
  - 30 numbers that are already dead in the final config;
  - 5 implementation caps;
  - 53 numbers of mechanisms deleted outright;
  - 17 numbers of mechanisms absorbed into others.
- The census lists types, not defaults, for the SynthAgent, AgentV21, AgentH and AgentV22 layers.
- v3 adds about 30 constants on the agent side that are declared nowhere:
  - the μ replay threshold of 0.05;
  - P5b's "high" threshold;
  - the bin width;
  - about 6 per-area active fractions;
  - about 9 fan-ins and fan-outs (64, 64, 32, 256, 64, GEN 64, CA3 1,000, CA1 64, programs 128);
  - 2 floors;
  - the π time constant;
  - the reserve size;
  - CA3 and μ gains;
  - P4 constants;
  - babble amplitude.
- It also adds about 11 lineage constants, the maturity count of 5,000, and about 12 world constants.
- The honest count is about 146 live numbers in v2.2 → about 100 in v3 (30 of them evolved): a cut of about 30 %, not "265 → 30 genes".

### 9. MINOR: the lineage selection has almost no statistical power
- P = 12, one life each, a fresh world seed per individual.
- Life metrics are "chaotic per seed": goodbye accuracy ranges 0.02–0.81, and two seeds cannot resolve differences under 0.05 (CITED v2.2).
- A ±20 % gene change is likely below that noise, so the gene audit will freeze most genes on noise.
- **Fix:** use the same world seeds for every individual within a generation (fresh seeds across generations), and run ≥ 2 short screening lives each.
- **Compute:** one job per generation holds only at v2.2 speed and without the zero-shot compatibility probe (check iv). With that probe or v3 tick costs it is 2–3 jobs, so 10 generations × 2 lineages take 30–45 h, not 15 h (PROVED).

### 10. MINOR: ladder criteria are measurable but miscalibrated
- **L6** passes on "whole-life free replies ≥ echo-or-constant reference". v2.2 fails that (0.47 / 0.52 vs 0.64 / 0.61): it is v2.2's open problem #2, unrelated to the anchor. L6's pass and fail conditions also leave a gap between them.
- **L8** passes at ritual ≥ 0.95 and fails below 0.9, but v2.2 final scores 0.88.
- **L2** requires empty-room greetings ≤ 15 per tenth; v2.2's mean is 16.4 / 17.2.
- **Drift to 10,000 untutored** costs about 7,700 s per run on v2.2-speed code (0.55 s per dialogue, from the finF log), above the 5,400 s job limit. That needs checkpoint/resume, which the plan does not include.
- **X2** is about 2.8 jobs, not 2.
- **New metrics needed:** a heard-syllable maturity clock, post-maturity mechanics per tenth with attribution to the agent, "lexicon chars bound", and the coverage package. All can be built from existing logs.
- **The anchor is a rewrite.** It means replacing RandMix's 54 pair-schema populations (`PAIRS`, `MIX_COUNT`), not changing a key.

### 11. MINOR: the agent loses the race against W1 formats
- In the §6.1 walk the agent's voice starts at τ = 10, the same as W1's τ_f.
- With CD = 4, the event-level agent answers before the caregiver in 0–1 % of format dialogues (SIMULATED, `tick_toy`).
- So W1's "fade as the child succeeds" never triggers, and δ never signs the child's own answers in formats.
- **Fix:** τ_f ≥ about 15. `tauf_sweep` shows no stability cost up to 30.

### 12. MINOR: four smaller corrections
- **Safety bound.** "Never overridden (PROVED)" holds only for clean windows, where w ≈ 0 anyway. At σ = 1 (w = 0.10), a correct window with a margin below 0.10 can be overridden by a confident prediction.
- **Replay of own lines.** Replay makes the successor the SYL target. For a successor I spoke myself, that teaches GEN my own answer, which live learning blocks. Either it is explained away in replay too, or replay reinforces my own errors. The design must say which.
- **Collapse attribution.** "That is v2.2's collapse" is overclaimed. The give-up route drifts from the start (toy S: 0.09 by 300). v2.2 holds 0.90–0.97 until 2,100 dialogues, then falls. That timing matches the prune age (R-d) or the turnover of the 10k hippocampal store (R-c). Label it ASSUMED and run X2 first.
- **LATEST size.** About 16 k units fits about 600 localist items. With 2,048 SYL assembly units it would be about 55 k.

---

## Do the first five experiments retire the biggest risks first?
**Partly.**
- X1 and X2 are rightly first. But X1 runs in a symbolic harness with no M4, no TD and no P5b, so it only re-tests what drift_toy already showed.
- X3 retires a low risk (L4).
- X5 diagnoses v2.2, not v3.
- Nothing tests pure-neuron feasibility, merging or compute.

**Proposed first five:**
1. X1+: the anchor with the full release/value loop, both O1 readings, and fixed vs stop-dependent partner timing (local).
2. X2 (about 3 jobs).
3. An S6 CA3 net (fan-in 1,000) on recorded v2.2 cue/episode streams (local).
4. A merge test: two siblings from one birth wiring, sign-elected merge, zero-shot probe (local).
5. A v3 tick benchmark at s = 1 and 4, on CPU and with torch at batches of 1, 16 and 64 (local + 1 job).

X4 comes next.

## The 3 fixes that matter most
1. **Event-gated O1.** Prediction synapses learn only at the target's onset events, and "when" lives in TD and M4. Test it in X1 with the whole release/value loop, and make every partner reply at a fixed delay after the agent's turn.
2. **A maturity rule the core can meet.** Babble and ε go through M4, P1c gets a yield clause, and the post-maturity rule counts only errors the agent causes.
3. **Reorder.** Assemblies, birth-match claims and LATEST units come before L6 and before any merge. The S6, merge and compute tests come first. Correct §7.5 for L ∝ s and for the gpuport measurement.

## summary_zh
- **结论：先修再建。** 有一个按原文就会失败的问题（改一句话即可），另有五个严重问题。
- **最关键：**"预测下一刻输入"如果真的每一拍都学（原文就是这么写的），那么在没人教的对话里，答案从来不是别人说出来的，每一拍都会被当成"该来没来"而削弱。我的小模拟（5 个种子）里，3,000 段对话之前、前 300 段内正确率就从 1.00 掉到 0.00，模型开始说对方的话，和 v2.2 的崩溃一模一样。只有"每次听到一句话时才学"才稳定（0.98–1.00），这也是作者小模型实际用的学法。但这样一来，模型学不会"什么时候说"，原文说能替代 P_oth 就没有依据了。改法：写明只在听到事件时学；"何时"交给奖励预测和基底节去学。
- **奖励回路没测过：**"被回应"奖励在测试台的提问节奏下，会奖励说晚了的答案，可能形成一条新的漂移路线。
- **"长大后 0 错误"现在做不到：** 探索和咿呀学语一辈子不停；人多时别人插进来也算重叠，而模型只会"别人说话时不开口"，不会"别人开口时停下"。
- **算力估计偏乐观：**
  - 我实测一拍的开销是原估计的 4–7 倍；
  - s=4 时一生要按比例变长，需要 3–6 个连续的 90 分钟任务；
  - GPU 批量的估计比仓库自己测的结果快了几百倍。
- **顺序要改：** 继承融合和纯神经元都需要的"分布式编码换装"被排在最后，应该提前做。
- **参数账：** 189 个"删除"里有 119 个只是开关、早已停用的或实现上的上限；新设计又多了约 30 个没有声明的常数。说"265→30"不诚实，实际大约是 146→100。
- **隐藏的专用结构：** 约 16 处，包括换了名字的复制路线、和测试台信号绑死的先天检测器，以及改写成世界规则的教学规则。
