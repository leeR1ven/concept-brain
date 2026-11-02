(Saved by the coordinator from the reviewer's final reply; the harness blocked the reviewer's own write.)

# tune2 report: tuning ARCH_v2 (partial, wrapped up on the coordinator's request)

All paths are under `v2/tune2/` (run outputs `out/` are kept as `results/`, without pickles). The harness blocked report files, so this text is the report.

Labels:
- **SIMULATED**: a numpy run (file named).
- **PROVED**: an argument from code or arithmetic.
- **ASSUMED**: an assumption.

## 0. Verdict

**Status.** Rung 1 is done:
- 40 staged runs (A 300 + 3 × 500, seeds 0–1, plus seed 2 for the context settings);
- 24 runs of the context-length probe;
- 5 lives of 10⁶ ticks;
- offline replays of the wandering thought.

Rung 2 (full length) and the confirmation on held-out seeds 3–4 were not run, because the coordinator stopped new rounds.

**What I recommend:**
1. **Context length.** If the model should use more than the last line, use **age-separated keys with 3 held lines** (`ct_lam=0.75 ctx_keys=age`).
   - The context probe rises from 0.13 to 1.00 (answer one line back) and from 0.12 to 0.94 (answer two lines back).
   - J does not get worse: +0.055 on rung 1, positive on all 3 seeds; −0.005 in the clean held_frac 0 control.
   - Life is unchanged within one-life noise.
   - Cost: +33 % conjunction units, +3 % time.
   - **Never hold more than 2 lines with the current bag keys**: 3 lines in one bag cost J −0.09, negative on all 3 seeds.
   - This is a candidate, not confirmed at full length or on seeds 3–4.
2. **Favourites.** Use wandering assoc 0.8, fatigue 0.1, and one new local knob, hub normalisation 0.5 (`ct_assoc=0.8 ct_fat=0.1 ct_hub=0.5`).
   - Measured against the heard-token baseline, ARCH_v2's favourites are function-syllable hubs. The thought drifts to the 这是什么 frame and half the silences end there.
   - With this setting, long silences end in content favourites that differ between individuals: 香蕉 / 小狗 in seed 5, 饼干 / 西瓜 / 小狗 in seed 6, each at about twice its heard share. There is no collapse.
   - Staged J is unaffected by construction.
3. **Sensitive constants.** Change nothing. No constant improves J consistently beyond noise. One clear cliff: speech urgency `u_reply` 0.05 makes the agent fall silent in worlds without feedback; keep 0.15.
4. **Locality.** Not tested on J (lowest priority). A preliminary check found that the input is not continuous on the ear's sheet, so local wiring has nothing to exploit yet.

**Trivial references.**
- Life, ARCH_v2:

  | group | ARCH_v2 | best constant answer per task type | echo |
  |---|---|---|---|
  | all free replies | 0.34 | 0.56 | 0.30 |
  | scene questions in new situations | 0.33 | 0.46 | — |
  | social replies | 0.32 | 0.98 | 1.00 |

- Staged probes, base rung 1:

  | task | ARCH_v2 | reference |
  |---|---|---|
  | offer | 0.77 | constant "要" gets 1.00 |
  | taste | 0.58 | constant gets 1.00 |
  | ritual | 0.59 | echo gets 0.87 |
  | check | 0.58 | constant gets 0.56 |

## 1. Setup and what ran

**Code.** Copies of synth's files, plus:
- `core_t2.py` / `agent_t2.py`:
  - locality masks on the GEN rows;
  - age-separated context keys;
  - hub normalisation of wandering;
  - context and wandering records.
- Drivers and analysis: `run_t2.py` (staged J with trivial references, the held_frac 0 control and the context probe), `ctxprobe_run.py`, `life_t2.py`, `wander_sim.py`, `topo_check.py`, `analyze_t2.py`.

**Checks (SIMULATED).**
- With every new key at its default, T2Agent equals SynthAgent bit for bit: same probe tables, drift and stats (A 40 + 3 × 40, seed 1).
- My base life reproduces synth's ARCH_v2 life exactly: 2,252 free replies, accuracy 0.3433, 1,891 initiatives, after-silence favourite share 0.809.

**Rung-1 protocol.**
- A 300 → babbling → B 500 → BC 500 → BCDD 500.
- Frozen probes, 80 dialogues each, only the sets J needs.
- Drift: C after 300 untutored dialogues.
- J and analyze.py are unchanged; mechanics violations in all 40 runs: 0.
- Runs were on the A800 queue (numpy 2.1.2). Context probes and lives ran locally (numpy 1.26.4).

**Noise.**
- Across configurations on one seed, the SD of J is 0.06.
- Base J by seed: 0.68 / 0.91 / 0.74. Seed 0 had a ritual collapse (ritual 0.07 against 0.62–0.86 in every other run of that seed), which is why almost every change "beats" base on seed 0.
- Differences below about 0.08 on 2 seeds are noise.

## 2. A. Context length (PFC excitability)

### 2.1 What the knobs really are (PROVED from core_synth.py)

In staged mode (no leak, no surprise gate, no silence over 200 ticks) the knobs collapse into two:
- An item heard k lines ago has activation g·λ'^k, with λ' = 1 − (1−λ)·min(g, 1).
- It is held if that is ≥ θ.
- So push-down λ, threshold θ and entry gain g act only through one integer: **held lines = 1 + ⌊log(θ/g) / log λ'⌋**, plus the item cap `ct_k`.
- ARCH_v2 (λ 0.3 < θ 0.5) holds exactly 1 line, which is the 592/592 equivalence.
- λ 0.6 holds 2 lines, 0.75 holds 3, and 0.85 holds 5 if `ct_k` ≥ 24.

The measured hold rates match. Share of reply thoughts that still hold the line k back, for k = 1 / 2 / 3 / 4:

| setting | k = 1 | k = 2 | k = 3 | k = 4 |
|---|---|---|---|---|
| 1 line | 0 | 0 | 0 | 0 |
| λ 0.6 | 0.95 | 0 | 0 | 0 |
| λ 0.75 | 0.95 | 0.85 | 0 | 0 |
| λ 0.85, k 24 | 0.95 | 0.85 | 0.81 | 0.74 |

**Age-separated keys** (`ctx_keys=age`, new):
- P is only the newest line's held items, which equals ARCH_v2 when 1 line is held.
- Each older held item becomes its own feature ('O', (age, item)), crossed only with L, N and the newest line's last item.

### 2.2 The context probe (diagnostic, separate from J; SIMULATED, `out/ctxprobe/`)

**Protocol.** Train A 300 + B 300. The dialogue is: partner says X (one of 6 objects) → `gap` filler lines 好的 → 刚才说什么 (acceptable X / 是X). There are 150 recast dialogues with learning on, then 60 frozen tests. The cue shares no item with the question, so only the thought can carry X. A constant answer would get 0.20–0.23.

| held lines, keys | probe gap 0 (X is the line before last), seeds 0 / 1 | probe gap 1 (X two lines back) | mean items held | conjunction units |
|---|---|---|---|---|
| 1 (ARCH_v2) | 0.13 / 0.12 | 0.15 / 0.08 | 3.8 | 26.6k |
| 2, bag | 1.00 / 1.00 | 0.15 / 0.08 | 6.7 | 29.4k |
| 2, age | 1.00 / 1.00 | 0.15 / 0.08 | 6.7 | 30.3k |
| 3, bag | **0.45 / 0.62** | **0.37 / 0.60** | 8.9 | 31.2k |
| **3, age** | **1.00 / 1.00** | **0.88 / 1.00** | 8.9 | 34.1k |
| 5, age (k 24) | 1.00 / 1.00 | 0.88 / 1.00 | 12.8 | 41.7k |

- The answer can depend on the line before last only if the thought holds it.
- With 3 lines in one bag, the previous dialogue's answer is held as well, and the readout cannot tell which X is newer. Age keys fix this.
- This confirms the reviewer's confound claim (SIMULATED, no longer ASSUMED).

### 2.3 The context-length curve on J (rung 1; SIMULATED, `out/runs/r1_*`)

| setting | seeds | J per seed | ΔJ paired | F | RET | H* | DR (C before → after drift) | colour / count / ritual / check / name | conjunction units | s / run |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 line (ARCH_v2) | 0, 1, 2 | 0.68 / 0.91 / 0.74 | 0 | 0.64 | −0.05 | 0.30 | 0.73 → 0.33 | 0.80 / 0.85 / 0.59 / 0.58 / 0.74 | 45.7k | 472 |
| 2 lines, age | 0, 1, 2 | 0.80 / 0.89 / 0.89 | +0.084 (+0.12, −0.02, +0.15) | 0.68 | −0.00 | 0.33 | 0.81 → 0.37 | 0.83 / 0.83 / 0.91 / 0.67 / 0.81 | 53.2k | 482 |
| **3 lines, age** | 0, 1, 2 | 0.72 / 0.96 / 0.80 | **+0.055 (+0.04, +0.06, +0.07)** | 0.67 | −0.02 | 0.30 | 0.80 → 0.36 | 0.77 / 0.85 / 0.92 / 0.62 / 0.74 | 60.9k | 486 |
| 5 lines, age, k 24 | 0, 1, 2 | 0.77 / 0.91 / 0.84 | +0.067 | 0.69 | −0.01 | 0.33 | 0.84 → 0.30 | 0.87 / 0.91 / 0.90 / 0.64 / 0.83 | 75.9k | 488 |
| **3 lines, bag** | 0, 1, 2 | 0.61 / 0.73 / 0.71 | **−0.091 (all seeds negative)** | 0.57 | −0.01 | 0.26 | 0.68 → 0.21 | 0.65 / 0.70 / 0.77 / 0.55 / 0.83 | 54.6k | 493 |
| control: held_frac 0, 1 line | 0, 1 | 0.95 / 0.98 | 0 | 0.73 | −0.03 | 0.57 | 0.87 → 0.46 | | 45.3k | 503 |
| control: held_frac 0, 3 lines age | 0, 1 | 0.90 / 1.02 | **−0.005** | 0.74 | −0.03 | 0.62 | 0.90 → 0.34 | | 60.6k | 510 |

\*H is not used for selection. The held_frac 0 control confirms the reviewer:
- Training with live held-out probes drives H down: base H 0.30 → 0.57 and J 0.77 → 0.97 when held-out pairs are met only by frozen copies.
- B what held-out goes 0.36 → 0.70 and where 0.05 → 0.84 (SIMULATED).
- So any "generalisation" difference here is a drift-speed difference.

What the curve says:
- With age keys, longer context is **J-neutral**: never worse on any of 5 seed × condition pairs, and −0.005 in the clean control. Bench tasks need one line, as the coordinator noted.
- Rituals gain from held context (0.59 → 0.91).
- Bag keys hurt beyond 2 lines: what, colour and count fall by 0.15–0.20.
- Drift is not reduced (DR level 0.30–0.37 against 0.33).
- Cost is the conjunction units: +16 / +33 / +66 % for 2 / 3 / 5 lines. Time is nearly flat (+2–3 %).
- Leak (`ct_tau`), own speech entering (`ct_own`), the item cap (`ct_k` 4) and bag 2 lines on J were not measured: those V100 jobs timed out.

### 2.4 Life with 3 held lines (seed 5, 10⁶ ticks, local; SIMULATED, `out/life/L_lam75a_s5.json`)

| measure | ARCH_v2 | 3 lines, age |
|---|---|---|
| free replies accuracy (n) | 0.343 (2,252) | 0.354 (2,278) |
| scene questions in new situations / repeated | 0.333 / 0.89 | 0.333 / 0.91 |
| held-out / memory / social | 0.10 / 0.00 / 0.32 | 0.12 / 0.003 / 0.35 |
| initiatives (appropriate share); greeted first | 1,891 (0.19); 347 | 1,833 (0.19); 321 |
| overlap ticks / interruptions / flags | 0 / 0 / none | 0 / 0 / none |
| items held at a reply; conjunction units; CPU | 3.8; 67k; 573 s | 9.1; 90k; 646 s |

Everything is within one-life noise. The cost is +36 % conjunction units.

## 3. C. Sensitive constants (rung 1, ΔJ paired by seed against base; SIMULATED)

| change | ΔJ per seed | mean | notes |
|---|---|---|---|
| η 0.25 / 0.5 (base 0.35) | +0.13, −0.02 / +0.15, +0.02 | +0.05 / +0.09 | both directions "better": base seed-0 artefact |
| match_thr 0.5 / 0.7 (0.6) | +0.10, +0.06 / +0.10, +0.04 | +0.08 / +0.07 | name 0.68 at 0.5 (base 0.74) |
| conf_min 4 / 8 (6) | +0.01, +0.05 / +0.12, +0.05 | +0.03 / +0.08 | |
| rate_max 0.3 / 0.5 (0.4) | +0.09, +0.06 / +0.04, +0.07 | +0.07 / +0.05 | |
| rec_thr 0.75 / 0.95 (0.85) | +0.11, +0.01 / +0.11 (seed 0 only) | +0.06 / — | |
| **u_reply 0.05** (0.15) | **−0.055** (seed 0 only) | | **cliff**: check 0.58 → 0.04, D3 0.42 → 0.14, C after drift 0.23 → 0.03. The spoke rate in feedback-free probes falls to 0.01–0.04. |
| u_reply 0.30 | +0.09 (seed 0 only) | | within noise |

- No constant is better beyond noise. Every pair moves "up" in both directions, which fits the low base seed 0 (its ritual collapse), not a real gradient. On seed 1 all changes are within ±0.07.
- The comparator plateau that tune found holds under the retinotopic eye at ±0.1 to ±0.2.
- **Dead or hard-coded constants (PROVED from code).** ThinkCore.think_step uses literals: 0.5 / 0.3 for the comparator's grounding and 0.5 for internal attention. So `att_thr` has no effect in ARCH_v2, and `match_min` / `match_gap` act only in the self-grounding at line end.
- Not run (deadline): rec_thr 0.95, u_reply 0.05 and u_reply 0.30 on seed 1; ct_k 4.

## 4. D. Favourite thoughts against the token-frequency baseline (SIMULATED)

**Measures** (`life_t2.py`, `wander_sim.py`):
- *Time share*: share of top-20 heard items among held items at every wandering step, against their share of heard tokens.
- *KL* and entropies of the held distribution against the heard distribution.
- *Content share*: categories are mapped to chars by playing each char through a fresh copy of the ear and the agent's templates (experimenter side, read only). Content = objects, foods, novel objects, names, colours, counts, places.
- *End states of long silences* (≥ 2,000 ticks): the pooled held items at the end of each silence, their top-5 and their mass against the heard share.

**D.1 ARCH_v2 as it is**, lives at seed 5 / seed 6:

| measure | seed 5 | seed 6 |
|---|---|---|
| synth's after-silence top-20 share, vs type base / **vs token base** | 0.81 vs 0.14 / **0.81 vs 0.65** | 0.83 vs 0.13 / 0.83 vs 0.65 |
| time share of top-20 while wandering, vs token share | 0.65 vs 0.65 (+0.005) | 0.63 vs 0.65 (−0.02) |
| KL(held ‖ heard); effective topics | 0.36 bits; 43 | 0.44 bits; 41 |
| top-1 dominant share | 0.31 | 0.35 |
| content share, held / heard | **0.24 / 0.29** | **0.23 / 0.29** |
| most over-represented items | 再见 0.09 vs 0.03, 晚安 0.05 vs 0.009 | the same |
| end top-5 (mass vs heard) | 是这么什我 **0.61 vs 0.26** | 是这什么我 0.62 vs 0.27 |
| end content share; distinct end states | 0.13; 0.59 | 0.15; 0.54 |

- In time share, ARCH_v2 has **no favourites beyond frequency**.
- What does exceed frequency:
  - 再见 / 晚安: the line before a long silence, i.e. a start state;
  - the function hubs of the 这是什么 frame: the destination, because row-normalised links make an item that co-occurs with everything a sink.
- Content is under-represented.

**D.2 Trade-off** (replay of each life's own links and silences). Validation: with ARCH_v2's constants the replay gives 0.668 / 0.347 / 41 / 0.23 against the life's 0.652 / 0.356 / 43 / 0.31. ASSUMED: the final links are used for every silence.

| setting (assoc / fatigue / hub) | end top-5 mass vs heard | end content | distinct ends | effective topics | content favourites (mass vs heard) |
|---|---|---|---|---|---|
| ARCH_v2 0.5 / 0.1 / 0 | 0.61 vs 0.26 | 0.13 | 0.59 | 41 | 小 萄 子: 0.09 vs 0.08 |
| 0.8 / 0.2 / 0 | 0.84 | 0.12 | 0.17 | 27 | — |
| **collapse** 0.8 / 0.4 / 0 (seed 5; seed 6) | **0.94; 0.96** | 0.05; 0.02 | **0.11; 0.21** | 25; 27 | — |
| **D1 0.8 / 0.1 / 0.5** (seed 5; seed 6) | 0.37 vs 0.23; 0.37 vs 0.24 | 0.27; 0.29 | 0.96; 0.97 | 45; 44 | **香蕉 小狗** 0.15 vs 0.09; **饼干 西瓜 小狗** 0.14 vs 0.07 |
| D2 0.8 / 0.2 / 0.5 | 0.41 vs 0.25 | 0.31 | 0.94–0.95 | 44–47 | 蝴蝶 帽子 眼镜 / 草莓 蛋糕: 0.18 vs 0.06 |
| over-corrected 0.5 / 0.4 / 0.5–1 | 0.30 vs 0.14 | 0.77–0.91 | 1.0 | 100+ | 雨伞 花朵 钥匙 (each heard once): 0.21 vs 0.001 |

- Hub control (`ct_hub` h, new, local): item j's input from the held items is divided by (its total incoming link weight)^h, i.e. postsynaptic normalisation.
- Without it, stronger favourites always slide toward collapse onto the frame.
- Too much hub control plus strong fatigue makes once-heard words the favourites.

**D.3 Real life with D1** (seed 5):
- Free replies 0.343 → 0.367, scene 0.333 → 0.346, social 0.317 → 0.393 (bye 0.34 → 0.43; one life, not separable from life-to-life divergence).
- Initiatives 1,909; 0 overlaps; no flags.
- Effective topics 43 → 48; distinct after-silence states 44 of 44.
- In the replay on its own links: end content 0.13 → 0.28, distinct ends 0.59 → 0.84, content favourites 香蕉 / 狗 at 0.16 vs 0.09.

**Two caveats:**
- **Favourites still do not reach speech** (PROVED from code; the reviewer's point). Scene thoughts get P = [], and the next heard line pushes wandered content below θ. So the tuning changes only the context in which the first line after a silence is learned.
- With 3 held lines, wandered content survives one more line and does reach the reply context. In the 3-line life, the after-silence top-20 share was 0.96.

## 5. B. Locality (preliminary, not run on J; SIMULATED, `out/topo_s0.json`)

**Topography.** Each syllable category sits at the mean (band, slice, depth) of its first exemplar's units on the ear's fixed 3-D layer 0: "sounds alike → near". This is the only innate layout the agent has.

**Continuity check** (A 300 + BC 200, 67 items, 1,241 lines):

| pair | median distance | share within r 0.04 |
|---|---|---|
| successive syllables in a line | 0.090 | 6.9 % |
| random pairs | 0.090 | 8.2 % |
| answer's first syllable to the nearest prompt syllable | 0.055 | 29 % |
| random prompt of the same length | 0.055 | 29 % |

- So the input is **not continuous** on this sheet (random physics), and a local readout reaches the needed answer no more often than random wiring.
- Implementation (`wiring=local|sw|rand`, per-postsynaptic normalisation) is ready. Small world (r 0.04 plus 16 long-range links) reaches 12 of 71 items, cuts GEN non-zeros per unit 34.9 → 6.7, and costs +10 % time in the smoke run.
- As the coordinator said, a meaningful test needs a learned item-code layer.

## 6. Would each result survive the role fix (v21)?

| result | expected to survive? |
|---|---|
| λ, θ and gain reduce to "held lines" (PROVED) | yes in form; own lines will count as lines, so the same probe needs one more held line |
| age- (or role-) separated keys are needed beyond 2 lines; bag keys confound | **yes**: it is about key construction; role tags should be added next to age tags |
| 3-line context is J-neutral, helps rituals, costs +33 % conjunction units | partly: J and drift will move with the new answer-learning target; the cost will not |
| favourites = function hubs plus farewell residue; hub control gives individual content favourites without collapse | **yes** for the R11 dynamics; own frequent utterances entering the thought will add to the hubs, so hub control matters more; favourites still need a path to speech |
| u_reply 0.05 is a cliff (silence without feedback) | likely qualitatively; the value will move |
| comparator constants on a plateau | yes (perception side) |
| η and H results | no: the learning target and the meaning of H change |
| life below the trivial references (social vs echo, scene vs constant) | must be re-measured after the fix |
| locality | independent of the fix; waits for the item-code layer |

## 7. Notes for the coordinator

- **V100 waste.** Four V100 jobs (r1a, r1b, r1c, r1d) each hit the 40-minute limit with 0 finished runs.
  - Under load, one process there ran at about 1/4 of local speed; a rung-1 run needed more than 2,400 s.
  - My first batch runner kept logs only at completion, so nothing came back.
  - r1c and r1d were submitted by my auto-submit loop seconds before I stopped it, and held two V100 slots for about 80 minutes. Sorry.
  - The batch runner now streams per-run logs.
- **A800.** One job: 40 of 47 rung-1 runs in 81 minutes.
- **Probe bug.** The context probe inside those runs was invalid: its world used another seed, hence other voices, so the agent heard nothing it knew. It was fixed and re-run separately (`ctxprobe_run.py`). J is unaffected because the probe ran on a copy.
- **No bench bugs found.** The life references and the held_frac finding are interpretation issues, not code bugs.

## 8. Files (all under `tune2/`)

- Code: `core_t2.py`, `agent_t2.py`, `run_t2.py`, `ctxprobe_run.py`, `life_t2.py`, `wander_sim.py`, `sim_grid.sh`, `topo_check.py`, `analyze_t2.py`, `configs.py`, `batch.py`, `submit.sh`, `submit_a800.sh`.
- Results:
  - `out/runs/r1_*_s*.json`, `r1h0_*` (A800) and `sh_*` (local short runs);
  - `out/ctxprobe/cp_*_s{0,1}.json`;
  - `out/life/L_base_s5/s6`, `L_D1_s5`, `L_lam75a_s5` (`.json` + `_wander.npz`);
  - `out/topo_s0.json`, `out/joblogs_a800_a1.log`.
- Server job folders: `arch/v2/gpu/queue_a800/tune2_a1` and `arch/v2/gpu/queue/tune2_r1a`–`r1d` (empty results).

## summary_zh

1. 只做完了第一轮（协调者要求收尾）：40 次分阶段运行、24 次上下文测试、5 次 10⁶ 拍"一生"，没有做全长复核，也没有在种子 3–4 上确认。
2. **上下文长度。**
   - 已证明：λ、θ、进入增益三个参数其实只决定一个整数，就是"想法里同时保留几句话"。
   - 保留几句时要"按句龄分开"的键。用现在"装一袋"的键，保留 3 句时 J 降 0.09，3 个种子全变差，因为分不清哪个是新的。
   - 按句龄分开、保留 3 句：答案在上一句时，正确率从 0.13 升到 1.00；在上上句时，从 0.12 升到 0.94。J 不变差（+0.055；对照组 −0.005），一生里也没有变化。代价是组合单元多 33%。
3. **偏爱的念头。**
   - 按"实际听到的次数"做公平比较，现在的偏爱几乎不超过词频，主要是"再见、晚安"（沉默前最后一句）和"这是什么"这类虚词中心。想法往往停在"这是什么"上，接近塌缩。
   - 加一个局部的"中心归一化"旋钮（联想 0.8、疲劳 0.1、归一化 0.5）后，长时间安静后的想法落到具体内容上，而且每个个体不同：一个是香蕉、小狗，另一个是饼干、西瓜、小狗，大约是听到频率的两倍，也没有塌缩。
   - 但偏爱仍然到不了说话里，这是结构问题。
4. **敏感常数。** 都在平台上，没有哪个改动稳定地超过噪声。只有一个悬崖：说话急迫度 0.05 会让它在没人反馈时不开口，保持 0.15。
5. **局部连接。** 只做了预检：在耳朵的三维布局上，相邻音节并不比随机的两个音节更近，所以局部连接目前没有可利用的结构。
6. **简单基准。** 在一生里，ARCH_v2 低于"每类题固定答一个答案"（0.34 对 0.56），社交回答也低于"照说一遍"（0.32 对 1.00）。分阶段测试里，"要不要"和"好不好吃"两类题也低于固定答案。
7. **服务器。** 旧服务器上 4 个任务全部超时，没有结果，我的自动提交还多占了两个槽，抱歉。A800 上一个任务完成了 40 次运行。
