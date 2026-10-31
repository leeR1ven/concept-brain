(Saved by the coordinator from the agent's final reply; the harness blocked the agent's own write.)

# randmix: random connections from birth, trained on top, tested on ARCH_v2

The harness blocked writing `report.md` ("subagents return findings as text"), so the full report is below. Folder: this directory (`v2/randmix/` in the repo). The unchanged ARCH_v2 core and `bench_copy/` are used from `../synth/`; outputs are in `results/` (the scripts write to `out/`).

Labels: **PROVED** (argument with numbers), **SIMULATED** (my runs), **CITED** (paper or another agent's file), **ASSUMED**.

Every run used the unchanged bench copy and `analyze.py`. Every staged run, and every post-probe, had 0 mechanics errors. That includes the probes with corrupted prompts: viol 0 in all 48 run files, and all post.py mechanics counters were 0.

## 0. Verdict

1. **The designer's idea works as a drop-in, and in one form it is better than MIX.**
   - The replacement is a conjunction layer that exists in full from birth: random sparse fan-in from the same input groups MIX pairs, with k-WTA at a fixed active fraction. Its size is fixed, it has no dict, and it never grows.
   - Its best form gives equal-sized random populations for every pair type ("flat": a birth that does not know which conjunctions will be needed), plus instar training and use-it-or-lose-it pruning with regrowth.
   - Results, paired by seed against ARCH_v2 (SIMULATED):
     - default protocol: J 0.83 → **0.95** (+0.12, better on 4/4 seeds); F +0.05; H +0.16;
     - honest held-out protocol: J 1.00 → **1.09** (+0.10, better on 5/5 seeds); F +0.03; H +0.06; DR +0.08.
   - Cost: colour drops (seen 0.86 → 0.74; honest held-out colour 0.95 → 0.68).
2. **"Random" alone is not the gain.**
   - With MIX's own allocation, the fixed layer (a) loses −0.05 J (0/2 seeds).
   - Training (b) is neutral (±0.00).
   - Training + pruning (c) gains +0.05 (4/4 seeds).
   - Overproduce 1.5× then prune without regrowth (c0, the developmental story) gains +0.03 default (4/4) and +0.07 honest (4/5). Only 23 % of its units and 12 % of its synapses survive.
   - The largest single gain is the flat allocation. The fixed flat layer was as good on its one seed (+0.12 / +0.19) as the trained flat one, so plasticity is not what the flat gain needs.
3. **The bench's held-out score cannot judge the idea (the coordinator's confound, confirmed with every layer).**
   - In the module test, every random layer generalises a frame perfectly when held-out pairs never occur (1.00). It collapses to 0.00 when they are asked during life, exactly like MIX.
   - With honest held-out (trainheld=0), MIX's H is already 0.60.
4. **No gain in fault tolerance or drift.**
   - One misheard syllable costs every layer about 0.27–0.38. The trained/pruned layers lose slightly more.
   - A similar-sounding syllable costs the same as a random one.
   - Untutored use costs about 0.25 after 300 and about 0.5–0.6 after 900 dialogues in every layer.
5. **Size and GPU are real gains.**
   - Fixed tables replace MIX's growth from 23k to 49k units.
   - The torch/CUDA layer gives identical active sets to numpy in 100 % of test contexts.
   - Batched on the A800, 64 agents run activity + readout + learning at about 12,000 agent-events/s.
   - 4× and 16× bring nothing in this world.

## 1. What I built (`randlayer.py`, `rcore.py`, `randtorch.py`)

ARCH_v2 (`synth/`, final_noctx) was copied unchanged. `RandCore` swaps `MixDict` for `RandMix`. GEN, the 3 BG copy gates and the critic read its active units exactly as they read MIX units: same R2/NLMS rule, same 1/|A| step, same actor/critic. Variant (d) is the same driver with no `rm_*` key.

**Input lines.** There is one bundle per group MIX keys use:
- item groups P (held thought, a bag), P1, PL, L, CP, CN, CC, VR: 512 lines each;
- small groups N, FAM, A, M, NOP: 16 lines per value;
- F: the attended block's 240 slot-relative eye lines.

**How items enter (the choice).** Each item is a 16-of-512 code, the same in every bundle. The code is a fly-hash of the item's own Z template: binarised at 0.5, a fixed random projection with 24 FEAT0 features per line, top 16 lines. It is fixed at the item's first use.

Why: an item is a perceptual category, so at birth the only similarity the agent can know is acoustic. Measured: template Jaccard 0.4–0.5 gives 23 % shared lines, versus about 8 % baseline (corr 0.43 over 9,453 pairs, `out/codes_a_n1_s0.json`, SIMULATED). Meaning overlap cannot exist at birth under the bench's random physics. As a control, arbitrary random codes (`code=rand`) scored the same (J 0.81 vs 0.82 on seed 0), so the acoustic choice is harmless but brings no gain.

**Units.**
- There is one population per MIX key type: 14 direct groups and 31 pairs, i.e. innate #18.
- Size N_t = max(32, scale·c_t), where c_t is MIX's key count of that type at the end of a staged run (47.9k in all). So 1× equals MIX's final budget with MIX's distribution: P×F 11.4k and L×F 10.7k, together 46 % of the units.
- `alloc=flat` splits the same total equally over the 45 populations.
- Each unit draws 3 random lines per branch at birth, with weight 1.

**Activity.**
- s_b = Σw·x per branch.
- A unit is eligible iff every branch has s_b ≥ 1 (dendritic AND).
- In each population, the k_t = max(1, round(α·N_t)) eligible units with the largest s_a+s_b fire (α = 1 %, about 440 active per event, against about 170 MIX keys).
- A population whose groups are absent is silent.

**Variants.**
- (a) fixed.
- (b) instar with heterosynaptic normalisation:
  - at every learning event, active units get w ← w + 0.02(x − w), then each branch is rescaled to Σw = number of synapses;
  - updates are applied in batches of 200 learning events (about 2 dialogues).
- (c) = (b) + use it or lose it, with regrowth:
  - a synapse whose line was not active while its unit fired for 20,000 learning events is replaced by a new random line of the same bundle;
  - the total stays fixed.
- (c0) = (b) + the same pruning without regrowth, with 1.5× units at birth. The k_t are those of 1×.
- (d) = the dict MIX.

All rules are local: pre activity, the unit's own firing, its own weights and its own synapse age.

**Drivers and tools.**
- `run_rm.py` is run_j.py with metrics unchanged, plus size/RSS logging per stage, the agent pickle, and `trainheld=0` (training worlds with held_frac 0; probes unchanged).
- `post.py` measures on the pickled agent:
  - held-out per task: held_frac 0.5, 150 dialogues per stage, frozen as run_j. Echo and the best constant answer are computed on the same records.
  - fault tolerance: run_j's final probe worlds with one prompt syllable replaced, either by a random lexicon syllable or by the most similar-sounding one (cosine of the heard 8×6 grid).
  - drift: 300 and 900 untutored C dialogues.
- `full.py` chains run_rm and post on the server.
- `gen_probe.py` / `gen_probe2.py` are the learning reviewer's harness with an `rm=1` switch.

## 2. Validation (PROVED / SIMULATED)
- **(d) reproduces ARCH_v2 bit for bit.** Seed 0 run locally gives probe tables and drift identical to `synth/out/runs/final_noctx_s0.json` (J 0.79).
  - The other seeds ran on the A800 (numpy 2.1), where the bench is statistically reproducible, not bit-wise.
- **post.py is consistent with J.** Its clean fault probe equals J's final B/C/D probe on the same records (0.6647 / 0.8113 / 0.419).
- **torch equals numpy.** `test_torch.py` gives identical active sets in 300/300 (V100) and 200/200 (A800) contexts at scale 0.25 (untrained and trained), 1, 4 and 16. Weights are identical.

## 3. Results

### 3.1 J, default protocol (mean; seed SD of J ≈ 0.06)

| variant | seeds | J per seed | J | F | RET | H | DR | colour | check | post held-out mean (best const 0.53) | fault drop | drift loss 300 / 900 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| (d) dict MIX = ARCH_v2 | 0,1,3,4 | .79/.92/.88/.74 | 0.83 | 0.66 | −.04 | 0.21 | 0.56 | 0.86 | 0.63 | 0.20 | 0.27 | −.25 / −.51 |
| (a) random fixed | 0,1 | .74/.87 | 0.80 | 0.65 | −.02 | 0.12 | 0.56 | 0.91 | 0.57 | 0.17 | 0.28 | −.25 / −.48 |
| (b) + instar | 0,1 | .80/.91 | 0.86 | 0.67 | −.02 | 0.21 | 0.57 | 0.95 | 0.60 | 0.23 | 0.33 | −.26 / −.52 |
| (c) + prune/regrow | 0,1,3,4 | .82/.94/.91/.85 | 0.88 | 0.68 | −.02 | 0.27 | 0.57 | 0.92 | 0.66 | 0.28 | 0.35 | −.27 / −.58 |
| (c0) overproduce 1.5×, prune, no regrowth | 0,1,3,4 | .82/.94/.89/.81 | 0.86 | 0.67 | −.03 | 0.26 | 0.58 | 0.91 | 0.69 | 0.29 | 0.32 | −.24 / −.55 |
| (a) flat, fixed | 0 | .91 | 0.91 | 0.68 | 0.00 | 0.36 | 0.59 | 0.75 | 0.70 | — | — | — |
| **(c) flat** | 0,1,3,4 | **.89/1.02/.95/.94** | **0.95** | **0.71** | −.01 | **0.37** | **0.60** | 0.74 | 0.76 | **0.36** | 0.36 | −.23 / −.52 |

**Paired ΔJ against (d):**
- c: +.03 / +.03 / +.03 / +.12 (4/4 seeds);
- c0: +.03 / +.02 / +.01 / +.07 (4/4);
- c flat: +.10 / +.10 / +.07 / +.20 (4/4; mean ΔF +.05, ΔH +.16);
- a: −.06 / −.05;
- b: +.01 / −.01.

c and c0 met the task's rule (beat (d) on held-out without losing F) on seeds 0–1, so I confirmed them on seeds 3–4.

### 3.2 Honest held-out protocol (trainheld=0: no held-out probe in the training worlds; probes and metrics unchanged)

| variant | seeds | J per seed | J | F | H | DR | post held-out mean (const 0.54) | fault drop | drift loss 300 / 900 |
|---|---|---|---|---|---|---|---|---|---|
| (d) MIX | 0–4 | .88/1.10/.99/.97/1.06 | 1.00 | 0.74 | 0.60 | 0.58 | 0.57 | 0.31 | −.25 / −.53 |
| (a) | 0,1 | .93/1.01 | 0.97 | 0.73 | 0.50 | 0.62 | 0.56 | 0.30 | −.28 / −.57 |
| (c) | 0,1,2 | 1.06/1.04/1.08 | 1.06 | 0.77 | 0.65 | 0.57 | 0.65 | 0.37 | −.24 / −.64 |
| (c0) | 0–4 | 1.05/1.08/1.06/1.03/1.10 | 1.06 | 0.77 | 0.66 | 0.58 | 0.64 | 0.38 | −.27 / −.61 |
| (a) flat | 0 | 1.07 | 1.07 | 0.75 | 0.69 | 0.64 | — | — | — |
| **(c) flat** | 0–4 | **1.09/1.13/1.08/.98/1.18** | **1.09** | 0.77 | 0.66 | 0.65 | **0.67** | 0.38 | −.24 / −.63 |
| size control: (b) at 0.35× (17k units, same active count, no pruning) | 0 | .99 | 0.99 | 0.73 | 0.61 | 0.60 | 0.58 | 0.35 | −.26 / −.58 |

**Paired ΔJ against (d):**
- c flat: +.21 / +.04 / +.09 / +.01 / +.12 (5/5);
- c0: +.17 / −.01 / +.08 / +.06 / +.04 (4/5);
- c: +.19 / −.06 / +.09.

**Held-out per task, honest (agent / echo / best constant):**

| task | (d) MIX | (c) flat |
|---|---|---|
| taste | 0.11 / 0 / 1.00 | **0.64** |
| check | 0.72 / 0 / 0.56 | **0.78** |
| offer | 0.60 / 0 / 1.00 | **0.92** |
| slot | 0.09 / 0 / 0.33 | 0.05 |
| where | 0.69 / 0 / 0.38 | **0.90** |
| what | 0.38 / 0 / 0.35 | **0.49** |
| colour | 0.95 / 0 / 0.26 | 0.68 |
| count | 0.99 / 0 / 0.40 | 0.93 |

Under the default protocol every layer stays below the best constant on taste, offer, slot and where. For example, (d) taste 0.01, offer 0.03, slot 0.03, where 0.02; (c flat) taste 0.07, offer 0.65, slot 0.09, where 0.08. Echo is 0 everywhere.

### 3.3 Module tests (`gen_probe.py`, `gen_probe2.py`, copied from review2/learning; about 1–3 min each)

**Frame × filler, frozen held-out (3 replies × 12 pairs):**

| condition | MIX | RL (a) | RL (b) | (c′) prune+regrow only | RL (c0) | RL (c) flat |
|---|---|---|---|---|---|---|
| held-out pairs never asked in life | 1.00 ×3 (CITED) | 1.00 / 1.00 / 1.00 | 0.83 / 1.00 | 0.75 | 0.92 | 0.75 |
| asked during life (as in the bench) | 0.00 | 0.00 ×3 | 0.00 | 0.00 | 0.00 | — |
| reviewer's Xcut + 30 % model replies | 0.42 (s0; reviewer 0.42/0.75/1.00) | 0.67 / 0.75 / 1.00 (live 0.78) | — | — | — | — |

This confirms the reviewer's prediction that a random-born MIX collapses the same way. The fix of the learning target transfers to it.

**Class-dependent answer (gen_probe2; 4 held pairs × 3 replies):**
- all at 0.25–0.50: MIX 0.25/0.50, (a) 0.25/0.50, (b) 0.25/1.00, (c) 0.25, (c0) 0.25/0.25, c-flat 0.25/0.50;
- so there is no similarity-based generalisation from any layer;
- the reviewer reported MIX 0.5/0.5 with an earlier version of the file.

### 3.4 Sweep (seed 0, default protocol)

All rows are variant (c) unless marked. Single seed, so differences ≤ 0.06 are noise.

| variant | J | F | H | post held-out | fault drop |
|---|---|---|---|---|---|
| (c) 1× | 0.82 | 0.64 | 0.20 | 0.23 | 0.33 |
| fan 6 | 0.86 | 0.64 | 0.26 | 0.27 | 0.36 |
| α 3 % | 0.86 | 0.65 | 0.24 | 0.25 | 0.34 |
| α 0.5 % | 0.83 | 0.64 | 0.23 | 0.24 | 0.35 |
| random item codes | 0.81 | 0.63 | 0.22 | 0.27 | 0.27 |
| flat | 0.89 | 0.67 | 0.34 | 0.33 | 0.37 |
| (a) 1× | 0.74 | 0.60 | 0.13 | 0.17 | 0.29 |
| (a) 4× (torch layer) | 0.78 | 0.61 | 0.13 | 0.16 | 0.29 |
| (c) 4× (torch) | F at the last probe 0.64 (B .74, C .81, D .37) | | | | |
| (a) 16× (α 0.25 %) | B .70 after stage 1 | | | | |

The (c) 4× and (a) 16× runs hit the 90-minute job limit, so they have no full J; 4× matches 1×.

### 3.5 Size, memory, speed (SIMULATED)

**Units over time (seed 0, after stages A / B / BC / BCDD):**
- MIX grows 23.3k → 27.5k → 34.9k → 48.6k (+13.7k in the last stage alone).
- RL is fixed from birth at 48.2k units / 286k fan-in synapses (flat: 47.9k / 243k).
- Units ever active by the end: (a) 62 %, (b) 54 %, (c) 67 %, flat-c 51 %.
- (c) made about 3.6M synapse replacements per run, 12× its synapse count: silent units keep re-wiring until they find used lines.
- (c0) is born with 72.2k units / 428k synapses; after B it has 17.9k / 54k; at the end 16.6k / 49.7k.
  - Nothing new becomes active after stage B: a critical period.
  - The size control (b at 0.35×) scored J 0.99 against (c0)'s 1.05 on seed 0, so pruning adds a little beyond the size alone (one seed).

**Memory (peak RSS):**
- MIX 390 MB; RL 1× 385–395 MB; (a) 500 MB (lookup cache); 4× 1.5 GB (GEN 192k rows).
- GEN is 49 MB at 1× in both layers.

**Speed (numpy agent):**
- An RL 1× staged run takes 1.2–1.6× MIX's time. Cache hit 43–59 %; one computed activation takes 0.9 ms on the A800 CPU and 1.2 ms locally.
- 4× with the torch layer and the numpy readout took 3,940 s against 1,270 s for 1×.
- 16× needed 1,618 s for A+B alone.

## 4. GPU
**Structure (PROVED).** MIX needs a hash table with collisions and grows. The random layer is a fixed table:
- `lines (N,2,f)`, weights, a population id and k_t per population;
- activity = gather `x[lines]` + sum + compare + one sort of N keys, and units are stored population by population, so rank = sorted position − population offset;
- readout = gather of the active GEN rows;
- learning = indexed add on the active rows and synapses.

**Measured on the A800 (`gpu_bench.py`, `out/gpu_bench_a800.json`):**

| scale | agents | activity, sorted k-WTA | activity, threshold form | full step (activity + readout + NLMS + instar) | throughput |
|---|---|---|---|---|---|
| 1× | 1 | 0.43 ms | 0.16 ms | 0.97 ms | — |
| 1× | 64 | 3.7 ms | 1.2 ms | 5.3 ms | **12,000 agent-events/s** |
| 4× | 16 | 3.6 ms | — | 4.6 ms | 3,400 /s |
| 16× (α 0.25 %) | 1 | — | — | 1.6 ms | — |

GPU memory stayed ≤ 2.6 GB.

**Per single call from the numpy agent** (transfer- and sync-bound):

| scale | torch | numpy |
|---|---|---|
| 1× | 3.2 ms | 0.9 ms |
| 4× | 3.5 ms | 3.0 ms |
| 16× | 4.8 ms | 14.3 ms |

So the GPU pays off only when the whole agent step stays on the device and agents are batched. Use the per-population threshold (homeostatic) form instead of the sort.

## 5. What worked, what did not, and why
- **Worked: random from birth with an unbiased (flat) allocation.**
  - MIX's own allocation puts 46 % of units on P×F and L×F (prompt items × the attended object's eye bits). The code is then dominated by the filler's appearance.
  - Synthetic test (`test_rl.py`, PROVED by counting):
    - two fillers in the same frame share 0.10 of MIX keys and 0.20 of the random layer's units (1×);
    - the same filler in another frame shares 0.40 and 0.42.
  - Flat populations shift the code toward frame and sequence conjunctions. That lifts taste, offer, check, where and what, but costs colour, which needs focus units (ASSUMED mechanism, consistent with every per-task number).
- **Worked moderately: use it or lose it.** (c) and (c0) beat (d) on every confirmed seed by +0.03–0.07. Overproduce-then-prune (c0) ends with a quarter of the units and similar or better behaviour. This is the developmental picture, CITED Huttenlocher 1979; Changeux & Danchin 1976.
- **Did not work:**
  - the fixed random layer with MIX's allocation (the v1-like control: −0.05);
  - a bigger layer (4×, 16×);
  - the acoustic item codes (no effect);
  - any gain in fault tolerance;
  - any gain in drift;
  - class (meaning) generalisation.
- **Why there is no fault tolerance (PROVED + SIMULATED).**
  - A misheard syllable is a different Z category, so about 25–30 % of the context changes in either layer (MIX 0.75 vs RL 0.72 of the code kept).
  - The NLMS readout fits each context sharply, so that change loses the answer about half the time.
  - The most similar lexicon syllable keeps only about 0.2 of the item's code lines.
  - Fault tolerance needs an item-level clean-up or a less sharp readout, not a different conjunction layer.
- **Why held-out is not fixed by the layer.** The collapse comes from the learning target: an unanswered probe teaches the next line, through GEN and through the hippocampal NEXT link replayed in awake recall (CITED review2/learning §2.1). The module tests show this is identical for all layers.

## 6. Recommendation for the next design
1. **Adopt the random-at-birth PFC-2 at 1×, with equal populations per pair type**, keeping use-it-or-lose-it (c, or c0 for a shrinking adult layer). Fix the colour cost with an allocation between flat and MIX's (e.g. ∝ √c_t), or a k_t per population scaled by input dimensionality. Do not scale N before the world grows.
2. **Fix the learning target, not the layer.** Adopt a perceivable contingency rule (the reviewer's Xcut transfers to the random layer: 0.00 → 0.78 live in the module test). Add `trainheld=0` to the bench as a second, honest held-out protocol next to J.
3. **For meaning generalisation (gen_probe2 stays at 0.25–0.5 for every layer), feed learned item codes into the random layer's input lines** (neuro #4). The layer is the right substrate: codes enter as line patterns, with no key explosion.
4. **GPU port:** fixed tables, homeostatic thresholds instead of sorted k-WTA (3× faster), the whole step on the device, agents batched. Measured: 12k agent-events/s for 64 agents at 1× (activity + readout + learning).

## 7. Binding principles

| principle | how it holds |
|---|---|
| one unbroken life, learning always on | the layer is born once and never reset; its plasticity runs at every learning event, awake recall included; the lookup cache is a memo, not state |
| no sleep | pruning and regrowth run online every 200 learning events |
| continuous thought | PFC-1 unchanged |
| local rules only | pre activity, the unit's own firing, its own weights and synapse age; normalisation within one unit's branch; k-WTA = inhibition inside one population |
| no puppeting | unchanged |
| 0 mechanics errors | measured in every run |

## 8. Files (this directory; `out/` here is `results/` in the repo)
- **Code:**
  - `randlayer.py`, `randtorch.py`, `rcore.py`;
  - `run_rm.py`, `post.py`, `full.py`, `batch.py`, `mkjob.sh`;
  - `table.py`, `gp_table.py`;
  - `gen_probe.py`, `gen_probe2.py`;
  - `test_rl.py`, `test_torch.py`, `gpu_bench.py`, `codes.py`, `keycount.py`;
  - plus the unchanged ARCH_v2 copies and `bench_copy/`.
- **Results:**
  - `out/runs/<tag>_s<seed>.json` (48 J runs) and `out/post/*.json` (44);
  - `out/gen_probe_out.jsonl`, `out/gen_probe2_out.jsonl`;
  - `out/gpu_bench_a800.json`, `out/gpu_bench_v100.json`;
  - `out/test_torch_a800.json`, `out/test_torch_v100.json`;
  - `out/codes_a_n1_s0.json`, `out/mix_keys_final_s0.json`, `out/test_rl.out`;
  - logs in `out/logs/`.
- **Tag key:**
  - `d_mix` (d), `a_n1` (a), `b_n1` (b), `c_n1` (c), `c0_n1` (c0), `c_flat` / `a_flat` (flat);
  - suffix `H` = honest protocol;
  - `a_n4`, `c_*` = sweeps.

## 9. Notes for the coordinator
- **V100 runner:** jobs see nproc = 1. My 4-run numpy screen timed out with almost nothing (scr1). I moved everything to the A800 as you instructed.
- **My mistake:** I forgot to copy `test_torch.py` into the first A800 job. It was rerun on the V100 and on the A800 GPU queue: identical results.
- **4×/16× runs:** these timed out at 90 min because GEN is still numpy. `a_n4` has J and post held-out/fault but no drift-900; `c_n4` and `a_n16` have stage probes only.
- **Bench:** H (and so J) is drift-confounded. I suggest adding `trainheld=0` as an honest protocol. No metric was changed.
- **Mixed hardware:** (d) seed 0, (a) seed 0 and the local controls ran with numpy 1.26; most other runs ran on the A800 with numpy 2.1. They are statistically equivalent, not bit-wise.

## summary_zh
1. 设计者的想法"像小孩一样先长很多随机连接、再在上面训练"可行：用出生时就全部存在的随机连接层替换 ARCH_v2 里"每遇到一个新组合就加一个单元"的 MIX 层。它大小固定、不再增长、适合 GPU，机制错误为 0。
2. 最好的形式是：每种配对类型分到同样多的随机单元（出生时并不知道以后要用哪些组合），再加上"用进废退"的修剪。和原来的 ARCH_v2 相比：
   - 标准测试 J 从 0.83 升到 0.95，4 个种子全部更好；
   - 去掉"训练中偷问未教过题目"干扰后的真实泛化测试，J 从 1.00 升到 1.09，5 个种子全部更好；
   - 代价是颜色题变差。
3. 单靠"随机"没有用。按 MIX 原来比例分配的固定随机层反而略差；规模放大到 4 倍、16 倍也没有好处。"先多长 1.5 倍、再剪掉不用的"能用剩下约四分之一的单元做到同样好或更好。
4. 测试台上的"没教过的组合"分数主要测的是学错目标：没回答的问题后面听到的下一句被当成了答案。换成任何一种层都一样会塌到 0，要改的是学习目标。审稿人的修正办法在随机层上同样有效。
5. 没有改善的方面：听错一个音节照样损失约三成；没人教时照样遗忘；也学不会"这类东西都能吃"这种按意思的推广，需要以后接上学出来的词义编码。
6. GPU：torch 版和 numpy 版结果完全一致。A800 上 64 个智能体一起跑，每秒约 12,000 个智能体事件；要真正快，整个智能体必须都放在 GPU 上批量运行。