# Step 4: a learned meaning layer — report

**Folder:** this one. The code runs on the repo's `../v21`, `../randmix` and `../synth`; the agent worked on unchanged flat copies of them, which are not committed.

`design.md` holds Phase 1 and its revision §6 (items 1–9). The code writes to `out/` (not committed); the results quoted here are copied into `results/`.

**Labels:** PROVED (argument, arithmetic or code), SIMULATED (my run; files named), CITED (another file or paper), ASSUMED.

## 0. Bottom line

1. **Class generalisation is solved in the module test, but the cause was not the code.** SIMULATED.
   - On the class probe, the design scores **held 1.00 on 10/10 seeds** (40/40 held fillers). Today's cores score 0.45–0.50.
   - **Seen pairs are unchanged** at 1.00.
   - **Swapping the codes of the held foods and toys** at test drops held to **0.00**: the answers flip.
2. **Two parts are needed.**
   - **(a) Learned meaning codes** fed into the random layer's item lines.
   - **(b) The same codes in the hippocampal DG cue.** The old cores' failure was mostly the hippocampal recall route, not the code. A held prompt (X好不好吃) recalls a taste episode of either class by identity overlap, and its stored answer is offered through COPY_next. That is a coin flip.
   - Proof of the recall effect:
     - meaning codes in the layer only: 0.70;
     - the same with the recall routes off at test: 1.00;
     - with the meaning cue in the DG: 1.00.
3. **Staged runs** (v2.1 + randmix c_flat, frozen protocol, canonical DG tie rule, A800, 5 paired seeds). SIMULATED.
   - Paired deltas:
     - **J 1.177 → 1.230** (+0.054 ± 0.051, up on 5/5 seeds);
     - J2 +0.111 (5/5);
     - F +0.039 (5/5);
     - H +0.077 (5/5);
     - drift unchanged.
   - **But the bench cannot credit meaning.** A control with sound + random second half, and the same DG cue, gets +0.030 J (3/4). Meaning over that control adds +0.032 ± 0.044 (mixed), which is within noise.
   - The reviewer predicted this: no bench held-out pair needs a class (PROVED from `world.py`).
4. **Fault tolerance is not delivered by meaning.**
   - Bench fault drop: 0.340 → 0.319 (mixed).
   - Probe: worse, 0.35 → 0.47.
   - A strong, unnormalised code cue in the DG lowers the bench fault drop by about 0.06 on every seed, but random codes do the same, and it costs J.
5. **Scaling: a 4× layer does not help once codes carry meaning.**
   - Short protocol (nS 500), 2 seeds: J −0.110 on both seeds.
   - Without meaning: +0.006 (mixed), matching randmix.
6. **Recommendation:** adopt the layer (§7). Add a class-dependent task to the bench. Do not scale. IDEAS #14 (fault tolerance) stays open.

## 1. Phase 1: design and attack (`design.md`)

### Four mechanisms, each attacked for drift, collapse, locality, one life and fault tolerance

- **A, RI (chosen): context-indexed Hebbian codes.**
  - Fixed random left/right index lines per item.
  - An instar synapse M[i,·] learns the context drive of item i's neighbours.
  - Code = top-16 of M/max(b, floor), where b is each line's own base rate (a positive-PMI code).
  - Sources: Kanerva 2000; Levy & Goldberg 2014 (CITED).
- **B, CL: competitive k-WTA over item × context.**
  - Rejected: two coupled plastic layers drift, and it collapses.
  - Module test: within-class Jaccard 0.78 vs between-class 0.76, purity 0.53, held 0.45.
- **C, TC: bootstrapped temporal context.**
  - The proved risk: power iteration of the co-occurrence matrix tends to one pattern, and similarity becomes syntagmatic.
  - It works when anchored: held 1.00. But its code drift in the last third of life is **0.082 vs 0.043** for RI.
  - Kept as an alternative, not chosen.
- **D, HIP: codes from hippocampal NEXT links in awake recall.**
  - Held 0.85. Coverage and timing depend on replay.
- **Top-down restoration of a misheard item: rejected by construction.** From context alone, a misheard item looks the same as a novel combination, so restoration would destroy the class test.

### Independent attack (one reviewer subagent; all points accepted, fixes in `design.md` §6)

1. The old K bits were already a perfect class code, so the code was never the bottleneck. This is what led me to find the recall cause.
2. My arithmetic in §2.A′ was wrong; corrected.
   - L要 and R吗 tie; R好 never enters a code.
   - The class-specific part is 6 of 32 lines.
3. Drift comes from rank flips of M/b, and M and b had mismatched time scales. Fixes:
   - floor on b and a slow b (τ 20,000);
   - η_min 0.002;
   - a random placeholder code from the first sight;
   - at most 2 line swaps per recompute, each only past a 1.2× hysteresis.
4. New, specific lesions:
   - held food ↔ toy code swap;
   - meaning off in both routes;
   - meaning off in the layer only;
   - recall routes off.
   - Also a realistic oracle ("oracle2": 6 class lines + 10 lines shared by all fillers).
5. The staged bench needs controls: a random second half (rand2) and a word-class oracle.

### A later fix after the first staged runs (`design.md` §6.9; PROVED arithmetic)

- An item is 1 of V identity lines in the DG bag but 16 of 512 meaning lines.
- So with equal weights the meaning cue's drive is 16V/512 times the identity cue's: 0.6× in the probe (V 20), but **8× on the bench** (V 256).
- At 8×, recall was dominated by meaning. Example: 中间是什么 recalled 刚才中间是什么. Staged J was −0.004.
- **Final rule (`hipnorm=1`):** meaning-group weight = D / (k_m · V), so the meaning cue weighs as much as the identity cue at any lexicon size.

## 2. What was built

- **`meanlayer.py`**
  - `MeanCodes`, with modes ri / tc / cl / hip / oracle / oracle2 / rand2 and lesions off / shuffle / swap.
  - `MeanMix(RandMix)`: every item bundle has 512 sound lines + 512 meaning lines, and an item enters as 16 + 16 lines. That is the same 1/32 density as randmix.
  - `code_stats` and `code_drift`.
- **`mcore.py`**
  - `MeanCore(RoleCore)`: v2.1 with the meaning-reading random layer.
  - Codes learn at the end of every heard line of another speaker. They learn under RAL too, because meaning does not depend on role. They never learn from own speech or from replay.
  - The meaning cue joins the DG as two groups (bag, first), fan 10, weight scaled as above.
- **`meantorch.py`**: the same with randmix's validated torch layer.
- **Drivers**
  - `gen_probe_mean.py`: the class probe on the v2.1 RAL harness. Options: 10 seeds, `w2=1` (two-syllable fillers as on the bench, 小 shared by 3 toys), `late=0.8` (new words never heard before 80 % of life).
  - `run_mean.py`: run_v21 with the canonical DG tie rule and per-stage code statistics.
  - `post_mean.py`: randmix post with held-out per task and references, the fault probe, and lesions.
  - `full_mean.py`, `paired_mean.py`, `gp_table.py`, `mkjob.sh`.
- **The smallest v2.1 + c_flat combination lives in my folder.** `mean=none` is that base. v22's folder was not touched.
- **Results**
  - `results/gp2/` (main 10-seed module sweep); `results/gp_local/`, `results/gp_hn/`, `results/gp_w2/`, `results/gp_late/`, `results/gp_ft/`;
  - `results/gp1_old/` (first sweep, earlier learner);
  - `results/gp_tables.md`;
  - `results/staged/{runs,post}`, `results/staged/paired_*.txt`.

## 3. Module test (symbolic class probe, 900 dialogues; 10 seeds unless stated; SIMULATED)

- **Measures:** held = 4 held taste fillers × 3 replies; seen = all seen pairs; fault drop = seen − one random prompt substitution.
- **Lesions** on frozen copies:
  - "no recall": the hippocampal copy routes switched off;
  - "off": the meaning half removed from both routes;
  - "layer off": the meaning half removed from the random layer only;
  - "swap": the meaning codes of the held foods and held toys exchanged.

| variant | held | food / toy | pooled fillers (p vs 0.5) | seen | fault drop | no recall | off | layer off | swap | codes: within / between Jaccard / NN purity |
|---|---|---|---|---|---|---|---|---|---|---|
| v2.1 MIX | 0.45 | 0.65 / 0.25 | 18/40 (0.79) | 1.00 | 0.41 | 0.53 | – | – | – | – |
| random layer, sound codes only | 0.50 | 0.70 / 0.30 | 20/40 (0.56) | 1.00 | 0.35 | 0.57 | – | – | – | – |
| + random second half + DG cue (control) | 0.57 | 0.65 / 0.50 | 23/40 (0.22) | 1.00 | 0.38 | 0.42 | 0.47 | 0.55 | 0.55 | 0.01 / 0.02 / 0.25 |
| + oracle2, layer only | 0.72 | 0.90 / 0.55 | 29/40 | 1.00 | 0.37 | 1.00 | 0.38 | 0.38 | 0.23 | 1.00 / 0.45 / 1.00 |
| + oracle2 + DG cue | 1.00 | 1/1 | 40/40 | 1.00 | 0.38 | 1.00 | 0.15 | 0.88 | 0.00 | 1.00 / 0.45 / 1.00 |
| + RI, layer only | 0.70 | 0.80 / 0.60 | 28/40 (0.008) | 1.00 | 0.38 | **1.00** | 0.42 | 0.42 | 0.25 | 0.90 / 0.41 / 1.00 |
| + RI + DG cue (weight 1) | **1.00** | 1/1 | 40/40 | 1.00 | 0.46 | 1.00 | 0.50 | 0.93 | **0.00** | 0.90 / 0.41 / 1.00 |
| **+ RI + normalised DG cue (final)** | **1.00** | 1/1 | 40/40 | 1.00 | 0.47 | 1.00 | 0.17 | 0.90 | **0.00** | 0.90 / 0.41 / 1.00 |
| RI + DG cue, without the drift fixes | 1.00 | 1/1 | 40/40 | 1.00 | 0.45 | 1.00 | 0.50 | 0.95 | 0.00 | 0.92 / 0.42 / 1.00 |
| TC + DG cue | 1.00 | 1/1 | 40/40 | 1.00 | 0.42 | 1.00 | 0.47 | 0.97 | 0.03 | 0.64 / 0.05 / 1.00 |
| HIP + DG cue | 0.85 | 0.70 / 1.00 | 34/40 | 1.00 | 0.42 | 0.72 | 0.47 | 0.72 | 0.17 | 0.11 / 0.01 / 0.98 |
| CL + DG cue | 0.45 | 0.60 / 0.30 | 18/40 | 0.99 | 0.45 | 0.53 | 0.28 | 0.47 | 0.45 | 0.78 / 0.76 / 0.53 |
| RI, k_x 10, + DG cue | 1.00 | 1/1 | 40/40 | 1.00 | 0.43 | 1.00 | 0.38 | 0.90 | 0.00 | 0.88 / 0.23 / 1.00 |

**Harder variants, final configuration (SIMULATED):**

| variant | sound codes only | RI, layer only | RI + DG cue |
|---|---|---|---|
| two-syllable fillers (`w2`), 5 seeds | 0.70 | 0.90 | 1.00 (weight 1); 0.95 (normalised) |
| **new words heard only in the last 20 % of life** (`late=0.8`), 5 seeds | 0.45 | – | **1.00 (20/20)** |

**Readout:**
- In the class frame, about 33 units are active for all seen foods and no toy (17 without meaning). A held food fires 85 % of them and 1 % of the toy-only units.
- Prediction "held ≥ 0.8": met.
- Code drift in the last third of life: 0.043 (RI) vs 0.082 (TC).
- Fault: substituting a frame word costs more with the DG cue (accuracy 0.81 → 0.65), so the predicted null holds and is even negative in the probe.

## 4. Staged runs (A800, numpy 2.1, canonical DG tie rule, frozen protocol, nS 1000; SIMULATED)

**Arms:**
- **vb**: v2.1 final + c_flat.
- **vn**: final design.
- **vrn**: control with a random second half and the same normalised DG cue.
- **vm**: meaning with the unnormalised DG cue.
- **vr**: random second half with the unnormalised DG cue.
- **vo**: oracle2 of the bench's word classes, taken from the agent's own categories, with the unnormalised DG cue.

**Cells:** arm mean (paired Δ vs vb, seeds up). P = `post_mean.py` held-out per task, held_frac 0.5, 150 dialogues per stage.

| arm (seeds) | J | J2 | F | H | DR | P held mean | P check | P colour | P what | P slot | P taste | fault drop (random) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| vb (0–4) | 1.177 | 1.716 | 0.768 | 0.719 | 0.956 | 0.782 | 0.526 | 0.754 | 0.727 | 0.480 | 0.815 | 0.340 |
| **vn (0–4)** | **1.230 (+0.054, 5/5)** | 1.827 (+0.111, 5/5) | 0.807 (+0.039, 5/5) | 0.796 (+0.077, 5/5) | +0.012 | +0.053 (4/5) | +0.212 (4/5) | **+0.154 (5/5)** | +0.078 | −0.042 | +0.042 | 0.319 (−0.021, mixed) |
| vrn (0–3) | +0.030 (3/4) | +0.064 | +0.022 | +0.046 | −0.019 | +0.045 | +0.131 | +0.087 | +0.028 | −0.030 | +0.152 | +0.005 |
| vn − vrn (0–3) | +0.032 ± 0.044 (mixed) | +0.064 | +0.024 | +0.044 | +0.029 | +0.007 | +0.134 | +0.053 | +0.042 | −0.022 | −0.099 | −0.023 |
| vm (0–4) | −0.004 (2/5) | +0.022 | −0.010 | +0.035 | +0.015 | +0.008 | +0.152 | +0.082 | −0.049 | −0.065 | −0.037 | **−0.067 (better on 5/5)** |
| vr (0–2) | −0.046 | −0.029 | −0.032 | +0.023 | −0.018 | −0.019 | +0.160 | +0.123 | −0.007 | −0.044 | −0.159 | −0.059 (better on 3/3) |
| vo (0–2) | −0.002 | +0.052 | −0.017 | +0.073 | +0.006 | +0.022 | +0.258 (3/3) | +0.010 | +0.082 | **−0.283 (worse on 3/3)** | +0.251 | +0.011 |

**Other measures for vn:**
- drift loss after 300 / 900 untutored dialogues: −0.003 / 0.001 (vb −0.012 / −0.004);
- role confusion 0.083 (vb 0.110);
- ritual 0.99;
- 0 mechanics violations in all 33 runs and every post probe;
- clean fault-probe accuracy +0.032 (5/5).

**Bench codes (vn, 5 seeds):**
- Over object / colour / count / slot / name syllables, by syllable position: within-class Jaccard 0.316, between 0.046, nearest-neighbour purity 0.89.
- Code drift between stages A→B / B→BC / BC→BCDD: 6.6 % / 17.5 % / 5.6 % of meaning lines (stage C brings new frames).

**Lesions at test (vn, 5 seeds):**

| lesion | probe all | held | taste held |
|---|---|---|---|
| intact | 0.807 | 0.796 | 0.90 |
| meaning off | 0.627 | 0.606 | 0.69 |
| layer off | 0.631 | 0.597 | – |
| shuffle | 0.533 | 0.509 | 0.26 |
| no recall | 0.779 | 0.809 | – |

- The layer reaches behaviour (principle 7).
- The random control vrn depends on its second half even more (off: 0.82 → 0.51; taste 0.97 → 0.17). So bench lesions show that the code is used, not that meaning is.

**Slot errors** in all arms are mostly silences, not wrong slots: the slot cost is in confidence, not in confusion between slot words. ASSUMED that this is the k-WTA shift; not isolated.

## 5. Scaling check (short protocol nS 500 so that 4× fits 90 min; seeds 0–1; SIMULATED)

- A full nS-1000 run at 4× timed out: about 1,900 s per stage against about 660 s at 1×.
- Profile at 4×: activation 5 ms per computed call against 1.1 ms; prune + CSR rebuild 0.62 s per batch against 0.12 s.
- torch per call (3.5 ms at 4×, CITED randmix) would not gain, so the runs stayed numpy.
- 4× used α/4: the same number of active units, 4× as many units.

| comparison | J | F | H | wall time |
|---|---|---|---|---|
| no meaning, 1× → 4× | 1.114 → 1.119 (mixed) | +0.004 | +0.010 | 1,233 → 3,016 s |
| **meaning, 1× → 4×** | **1.272 → 1.162 (−0.110, both seeds)** | −0.060 | −0.111 | 1,309 → 3,304 s |
| 1×, no meaning → meaning | +0.158 (both seeds) | +0.097 | +0.203 | |
| 4×, no meaning → meaning | +0.042 (mixed) | +0.032 | +0.082 | |

- The prediction "scaling helps only with this layer" **failed**: in this world a bigger layer does not help, and with meaning it hurts.
- ASSUMED cause: each unit is trained 4× less often in a short life.
- Meaning helps more in the shorter life (+0.16 at nS 500 vs +0.05 at nS 1000), consistent with generalisation speeding learning (ASSUMED).

## 6. Predictions checked

| prediction | result | verdict |
|---|---|---|
| class held ≥ 0.8 (SYNTHESIS Step 4) | 1.00 (10 seeds); new words 1.00; two-syllable 0.95–1.00 | met (needs the DG cue) |
| seen unchanged | 1.00 in the probe; bench F +0.039 | met |
| oracle ≥ 0.8 | layer only 0.72; with DG cue 1.00 | met only with the recall fix |
| lesion drops held to chance | meaning off 0.17; swap 0.00 | met |
| fault: no gain on random substitution | probe worse (+0.1 drop); bench −0.021 (mixed) | null, as predicted |
| staged held-out taste / offer up, F within the seed SD | taste +0.04, offer −0.01; F +0.04 (5/5) | taste / offer not met; F met |
| meaning credited beyond the random-code control | +0.032 J (mixed) | not shown on this bench |
| 4× no gain | −0.11 with meaning | "no gain" holds; it is a loss |

## 7. Recommendation and final configuration

**Final configuration (for v22):** v2.1 final + randmix c_flat + `mean=ri mean_hipcue=1 mean_hipnorm=1`.
- D 512 meaning lines; k_x 6; k_m 16;
- η_i = max(1/n_i, 0.002);
- τ_b 20,000; floor 2k_x/D;
- hysteresis 1.2; 2 swaps per recompute; recompute every 20 heard lines; random placeholder start; n_min 3;
- item bundles of 512 sound + 512 meaning lines;
- DG + bag and first-item meaning groups.

**Next steps:**
1. **Adopt it.** It is the first mechanism that passes the class test (0.50 → 1.00, 10 seeds), including words that are new late in life. On the bench it is +0.054 J (5/5) with no drift cost.
2. **Make the bench able to see Step 4.** Add a class-dependent task with toys, for example "X能不能吃" → 能吃 / 不能吃 with held toys and held foods. Without it the bench shows only unit statistics (vrn gets half the gain).
3. **Hippocampus:** any cue added to the DG must be scaled to the identity cue (§6.9). Unscaled, it buys fault tolerance (−0.06 drop) at a J cost.
4. **Do not scale the layer** before the world grows.
5. **Open:**
   - fault tolerance (IDEAS #14) needs an item-level clean-up or acoustic confidence, not meaning;
   - the slot held-out cost (−0.04, silences);
   - TC (more class-specific codes, about 2× drift) and k_x 10 as alternatives if the corpus needs stronger classes;
   - one more control seed (vrn s4) to complete 5-seed pairing.

**Binding principles:**

| principle | how it holds |
|---|---|
| one unbroken life, learning always on | codes learn from the first heard line; frozen copies learn nothing |
| no sleep | updates at line ends |
| continuous thought and v2.1's speech path | untouched |
| local rules | instar synapse M[i,j] (pre, post, own value); each line's own base rate; k-WTA; the DG reads the codes as a fixed random fan-in |
| no puppeting | untouched |
| 0 mechanics errors | measured in every run |
| every learned part reaches behaviour | lesioned (§4) |

**Size and speed:** the layer stays 47,925 units / 242,820 synapses. Codes add a 256 × 512 float table (0.5 MB). A staged run takes 2,611 s against 2,540 s for the base; post RSS is about 665 MB against 533 MB.

**Note for the coordinator:** no bench bug found, and metrics unchanged. My first sweep and runs S1–S2 used an earlier post-lesion version: their "off" lesion is layer-only and their "shuffle" was a no-op. They are not used for any lesion claim above. The paired tables (J, held-out, fault) are unaffected.

## summary_zh
1. 做了一个"词义层"：每个音节除了固定的声音编码，还有一个从上下文学出来的意思编码（经常跟在"吃"后面的词就会彼此相像）。这个编码同时送进随机连接层和海马体的检索线索。
2. 小测试里，"没在'好不好吃'里出现过的新食物/新玩具"答对率从 0.50（瞎猜）升到 1.00（10 个种子全对）；生命最后才学的新词也全对；把两个类别的编码对调，答案就全反了，说明确实是靠"意思"。
3. 关键发现：以前失败主要不是编码不好，而是海马体按"字面"回忆到别的类别的旧问题，把错答案搬了出来。让海马体也按意思回忆（权重要和字面线索相当）才解决。
4. 测试台上（5 个种子）J 从 1.18 升到 1.23，每个种子都升，也没有遗忘。但测试台里没有真正需要"按类别回答"的题，用随机编码的对照也能拿到约一半的提升，所以测试台证明不了"意思"本身的作用，建议加一类"能不能吃"的题。
5. 没做到的：听错一个音节的容错没有改善；把随机层放大 4 倍反而更差，所以先不要放大。
