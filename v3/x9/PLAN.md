# X9 PLAN: item 14, the abstract concept area as an offline module test

Written 2026-11-01 before any run. Sources: ARCHITECTURE §二.2, §二.8 (incl. the 11-01 held-neuron form), §四.14;
HARVEST #14; harvest h1 item 9, h4 item 9; v2/meaning report + design (RI = per-item index lines: NOT used).

## 0. Data (fixed before running)

- **Base stream:** the recorded bench Life stream `stream.json` (copied from x345; 2.58 M ticks, 12,087 lines, 11,292
  eye changes, 120 days). 3 ticks per char, as X3.
- **Why the base stream alone cannot carry the class test (PROVED from the frame table, `frames` in the log):** in the
  bench Life the only food-only frames are `X好不好吃` / `X好吃` (X is line-initial: no left context), and the only
  toy-subset frame is `我的X叫…` (4 animals; not 皮球/汽车). So a food has no class-specific context left once taste is
  held out, and the v2/meaning probe logic ("held out of one frame, present in another class frame") is impossible.
- **Main stream = base stream + the v2/meaning class dialogues** (`gen_probe_mean.py` w2 frames, the bench's own 12
  objects as fillers), inserted into silent daytime gaps (≥ 80 ticks, eye on, no eye change in the window; ≥ 1,500
  ticks apart; ≈ 1,200 exchanges spread over the whole life):
  offer `你要X吗→要X` (all), eat `你吃X吗→吃X` (foods), play `你玩X吗→玩X` (toys), taste `X好不好吃→好吃` (food) /
  `→不能吃` (toy). Prompt by `other`, reply by `other2` after 3 ticks. During the exchange the eye shows X (one object,
  random slot / colour / count, pointing at it, persons present) with the bench's own eye codes (Curriculum(0); checked:
  532/532 bench answer lines contain the named object's code).
- **Held out, per seed:** 2 foods (of 6) + 2 toys (of 皮球, 汽车, 鸭子: the toys with no shared char, so no sound
  overlap can leak class) are never heard in the taste frame (the inserted taste lines and the bench's own
  `X好不好吃` / `X好吃` lines with them are removed). They still occur everywhere else (offer, eat/play, all bench
  frames, vision), as in v2/meaning's probe.
- AUD = X3's char assemblies (2,048 units, 24 per char, physics seed 7, X3's char order; the 2 new chars 玩 能 drawn
  next from the same rng). EYE = the 1,024 world bits. Only feature units (ever on) have chain copies.

## 1. The concept area (one form; shared by all arms)

- Own population, N_C = 2,000 binary units, set point ≈ 6 % on (count control = X3's `Thr`: smoothed quantile,
  ceiling 2×; at test the instant quantile).
- Birth fan-in: local random, several comparable inputs from AUD and from EYE (counts set so the two areas' expected
  drive is comparable), plus silent potential synapses (birth weight 0) that can grow. Birth links never decay.
- Learning: the one rule (§二.8, 11-01 form) on every plastic link: B rises while A is on (rising, held, or just off)
  → A→B up (concave growth η·g(w)); A rises after B was on → A→B down; B falling changes nothing. One global decay
  of learned links (γ per 200 ticks, deletion below w_min).
- Two-layer weights: drive = w / √(out-total × in-total), recomputed from the current totals.
- Updated event-driven (state recomputed when the input changes: char onset, line end, eye change), as X3. ASSUMED
  shortcut, registered.
- **Context (the refinement):** the previous moment's concept activity drives the current moment through recurrent
  links (born random and/or learned by the temporal rule), acting at the update when the next input arrives.
  No PFC is needed (designer's order: PFC first). No per-item index lines, no batch PMI, no records.

## 2. Arms

| arm | context drive | role |
|---|---|---|
| **B0 base** | none (β = 0) | the area with feed-forward learning only |
| **Tb** | born random recurrence (fixed) | diagnostic form |
| **Tl** | learned temporal links only | diagnostic form |
| **Tbl** | born random + learned (the designer's literal "born random, learns with the rule") | diagnostic form |
| **T\*** = the one of Tb / Tl / Tbl chosen on dev seeds 100–101 only, frozen before seeds 0–9 | | **the refinement** |
| C-lean (T\* config) | as T\* | control: held fillers also removed from every class frame (eat/play, 我的X叫, X好吃) → class must be at chance (no leakage) |

Parameters (β, η, γ, fan-ins) are chosen on dev seeds 100–101 only. Test seeds 0–9 (10 seeds; held sets and network
seeds differ per seed). No change after seeing test-seed results (if I must, I will say so and why).

## 3. Metrics (frozen copy, learning off; probe = word heard in isolation after silence, eye empty)

- **code(X)** = union of the concept states at the two chars of word X.
- **within / between**: mean Jaccard of filler codes, same class vs other class, over pairs that share no char.
- **held class decode**: each held filler → class whose seen fillers' codes it overlaps more (mean Jaccard).
  40 decisions (4 × 10 seeds). Also seen fillers by leave-one-out.
- **cross-modal**: hear(X) vs see(X) = eye shows X alone (random slot/colour/count, no pointing, no persons), silence.
  Matched Jaccard − mismatched Jaccard (same eye draw for the other 11 objects); also 12-way identification top-1.
- **drift**: fraction of a code's units changed between 2/3 of the life and the end (1 − |A∩B| / max(|A|,|B|)),
  mean over the 12 fillers (also over the 80 most frequent chars).
- **collapse index**: mean pairwise Jaccard of the isolated codes of the 80 most frequent chars; also within-class
  filler overlap (identity kept if < 0.7).

## 4. Pass criteria (from the task; fixed)

| criterion | pass |
|---|---|
| held-out class decode | ≥ 0.9 (pooled 40 decisions) |
| within vs between | within > between on ≥ 9/10 seeds and mean ratio ≥ 1.5 |
| drift (fillers, last third) | ≤ 0.06 (mean over seeds) |
| cross-modal | matched > mismatched on ≥ 9/10 seeds |
| no collapse | collapse index ≤ 0.15 and within-class filler overlap < 0.7 |
| C-lean control | held decode ≤ 0.7 (else the test leaks) |

Verdict: PASS if T\* meets all five; PARTIAL if it meets the class criteria (decode, within/between) but not all;
FAIL otherwise. T\* must also beat B0 on held decode (paired). Report mean ± SD and paired T\* − B0.

---

## Amendment A (09:05, after the dev seeds 100–101 only; no test seed has been run)

**Why the form changed.** On dev seed 100 every single-layer form failed the class test before any test seed ran
(SIMULATED, `res/dev1`, `res/dev2`): within ≈ between (e.g. Tb 0.061/0.062, Tbl 0.063/0.064, Tl 0.069/0.074 at
β 0.1–0.2), held decode 0.25–0.5. Diagnosis (SIMULATED, `diag4.py`): in one population the char's own birth code
reinforces itself at every hearing and leaves ≤ 5 % of the state to the context; a context strong enough to enter
(Tb β ≥ 1, Tl β ≥ 0.2) makes codes history-dependent (drift 0.15–0.34) and loses identity (cross-modal top-1 0.03–0.3).
ARCH §四.14 leaves "要不要分几层" to experiment, so the area gets **two layers** (arm **L2**):
- **C1** (input layer) = the B0 layer: N 2,000, birth fan-in 40 AUD + 6 EYE comparable inputs, feed-forward learning.
- **C2** (deep layer) = N 2,000, ≈ 6 % on, no sensory input. Context: the previous moment's C1 activity through born
  random links K (40 per unit, fixed). The current item: C1 → C2 links F, born silent (all pairs potential, pF 1),
  learned by the one rule. d2 = β2 · K-push(C1(t−1)) + F-push(C1(t)), each two-layer-normalised; X3's count control.
  Items that follow the same context are learned onto the same randomly indexed C2 units; nothing is indexed per item.
- Decay of C2's learned links γ2 = 0.99995 per 200 ticks (deep layer decays slowest, ARCH §二.8); C1 keeps 0.9995.

**Rule detail (declared).** The "just off" case of §二.8 is not applied to feed-forward chain inputs (η_ff_off = 0).
Reason (PROVED from the rule): AUD(吃)→(food units) would grow 6× faster than any food's own link, so 吃 lights the
food units before the food arrives; the rule then *weakens* the arriving food's links to them (A rises after B was on).
SIMULATED check (seed 100, Tb): with it, drift 0.44 and cross-modal top-1 0.17 vs 0.42 without. In L2 the
"previous moment" relation is carried by the K route instead. Arm LETTER tests the rule's letter (η_ff_off 0.05 and the
temporal rule on K, η_K 0.05).

**Metric change (declared).** The class code is the deep layer's code of the filler's **first char**, one item per
first char (小狗 小猫 小鸟 → 小 once). Reasons: the second char's left context is always the first char of the same
word, so it cannot carry class (PROVED); 小 is shared by 3 toys and 8 names; v2/meaning also measured first syllables.
Whole-area word-union metrics are reported as secondary. Cross-modal = whole area (both layers), word code vs see code.
Drift is reported for both the whole-area word code and the class code; **both must be ≤ 0.06**. Collapse index on
the class layer's char codes and on the whole area.

**Frozen test arms (seeds 0–9 unless noted).** Common: g_E 0.07, cA 40, cE 6, f 0.06, η 0.05, n_ins 1,200.
| arm | what |
|---|---|
| **T6** = T\* | L2, β2 = 6 (chosen on dev: seed 101 held 4/4, within/between 0.224/0.063, drift class code 0.093) |
| T3 | L2, β2 = 3 (stability trade-off: seed 101 held 3/4, 0.201/0.068, drift 0.067) |
| **L20** = base | same two layers, K carries the CURRENT C1 state (no context), β2 = 6 |
| B0 | single layer, no context (secondary base) |
| LEAN | T6, held fillers removed from every class frame (control: must be ≤ 0.7) |
| BENCH (5 seeds) | T6 on the pure bench stream (no inserted dialogues) |
| LETTER (5 seeds) | T6 with η_ff_off 0.05, η_K 0.05 |
| Tb1, Tl03 (5 seeds) | best single-layer dev configs (cA 15, g_E 0.13; Tb β 1.0 slow decay; Tl β 0.3) |
Pass criteria unchanged (§4), applied to T6; paired difference vs L20 (and B0).
