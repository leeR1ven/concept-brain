(Saved by the coordinator from the agent's final hand-back. In the repo, `world.py` and `core3.py` come from `../x1c/` and `voice.py` from `world/`; they are not duplicated here.)

# X7 report: HARVEST items 13, 26, 28a, 28b tested on the designer's architecture

Item 13 failed both pass conditions. Item 26 failed in all four arms. Item 28a failed. Item 28b is partial: it passes only if the motor area learns with the older "only just-changed units" form of the rule, and fails with the 11-01 held form.

Labels: PROVED = exact run numbers or the code's logic; SIMULATED = a toy or harness run; ASSUMED = my judgement. Plans and pass criteria were fixed in `PLAN.md` before the runs; the amendments are listed in §5. All numbers are 5 seeds, mean ± SD; paired differences are against the base arm. Folder: `v3/x7/`.

## 1. Verdict table

| item | verdict | key numbers | paired difference |
|---|---|---|---|
| **13** convergent thresholds (eq4, eq7 vs dom1) | **FAIL** (both arms, both conditions) | L4 count for silence / 1 syllable / sentence: dom1 1922 / 1838 / 1633; eq7 1878 / 1841 / 1736. Silence lights **more** neurons than a sentence in every arm. Overlap of different syllables at L4 (J_diff): dom1 0.893, eq7 0.947. Spearman ρ at L4 (local3d had 0.989): dom1 0.42, eq7 0.28 | eq7 − dom1: J_diff **+0.054 ± 0.018** (worse in 5/5 seeds); ρ −0.145 ± 0.046; overlap of different sentences +0.168 ± 0.029. eq4 − dom1: J_diff −0.002 ± 0.022; ρ −0.070 ± 0.068 |
| **26** babbling teaches sound → the same muscles (rule exactly as in §二.8, 11-01 held form) | **FAIL** in all 4 arms (the base fails as predicted) | Share of the 34 syllables said correctly from inner hearing after 2000 babbles: base 0.01, same-tick 0.00, bababa 0.00, imitation 0.00. From hearing: 0.00 in every arm. Hand-wired stand-in: 0.86 ± 0.03. Garble from inner hearing: 0.69 / 0.01 / 0.26 / 0.58 | each arm − base: −0.006 ± 0.013 |
| **28a** whole-syllable competition | **FAIL** | Garble under noise and cross-talk: 0.53 without the links, 0.50 with them. Correct share on the clean probe falls from 0.88 to 0.81 | garble −0.027 ± 0.016 (−5 % relative; the pass needed −25 %); correct −0.004 ± 0.018 |
| **28b** syllable pace from force grades rising in turn | **PARTIAL** | Syllable length in free running (3 ticks taught), threshold φ = 0.6 of the input at the taught onset: onset form 3.00 ± 0.00 (100 % exactly 3); held form 1.85 ± 0.02 (11 % exactly 3) | held − onset = −1.15 ticks |

## 2. Part A, item 13 (SIMULATED; `sens.py`; `outA/AGG_main.txt`)

**Harness:**
- **Sound:** the project's `world/voice.py` (8 bands × 10 levels, 6 slices per syllable, 2 formant peaks per slice).
- **Ear sheet:** 24 slices (4 syllables).
- **Input layer:** thermometer on/off pairs, exactly half on (asserted in the code).
- **L1–L4:** the layout of local3d `model.py` (own position plus random neighbours, dx/dy ±1, dz ±2), with z = 40 / 60 / 120 / 240.
- **Threshold:** local3d's `threshold_activation` form: fire iff Σw·x > θ_l·Σw, with no bias and no top-k. Each θ_l is calibrated once so that one-syllable windows give 30 / 15 / 8 / 4 % active.
- **Arms:** dom1 = 1 dominant input + 3 weak inputs (the current form); eq4 and eq7 = 4 or 7 inputs of comparable weight.

**Results at L4:**

| | dom1 | eq4 | eq7 |
|---|---|---|---|
| count: silence / 1 syl / 2 syl / sentence / two speakers | 1922 / 1838 / 1759 / 1633 / 1770 | 1891 / 1835 / 1788 / 1710 / 1858 | 1878 / 1841 / 1803 / 1736 / 1755 |
| sentence ÷ silence | 0.85 ± 0.01 | 0.90 ± 0.02 | 0.93 ± 0.02 |
| J_diff, different syllables / different sentences | 0.893 / 0.643 | 0.891 / 0.652 | 0.947 / 0.811 |
| J_sim (same syllable, own vs other voice) | 0.980 | 0.977 | 0.983 |
| ρ, from L1 to L4 | 0.92 → 0.42 | 0.91 → 0.35 | 0.89 → 0.28 |
| changed L4 units per extra L1 unit: scattered / coherent patch | 1.54 / 1.68 | 1.79 / 2.05 | 0.86 / 0.92 |
| secondary, chain-bearing units only (on ≤ 60 % of the time): count for silence / 1 syl / sentence / two speakers | 437 / 391 / 297 / 568 | 266 / 259 / 261 / 571 | 23 / 32 / 61 / 258 (rises in 5/5 seeds) |
| chain units: J_diff / J_sim / ρ | 0.67 / 0.95 / 0.37 | 0.56 / 0.90 / 0.33 | 0.40 / 0.79 / 0.18 |

**Why it fails:**
1. **The on/off code is zero-sum** (PROVED from the code's form).
   - Every "up" line a sound turns on is paired with a "down" line it turns off.
   - Random wiring prefers neither polarity, so the number of neurons whose inputs coincide does not depend on the sound.
   - Silence is itself a fully consistent reading (all "down" lines on), so it satisfies coincidences as easily as speech.
2. **Overlap rises instead of falling.** A strict coincidence test passes mostly for neurons whose inputs are all quiet-cell lines, and every sound shares those.
3. **Similarity is lost faster with comparable inputs. This is not a harness bug.** On smooth image-like inputs at local3d-like sparsity, ρ at L4 is dom1 0.89, eq4 0.66, eq7 0.52. The base arm roughly reproduces local3d's high-ρ regime there (`outA/sanity_rho.txt`).
4. **The result holds in other settings:**
   - research 05's own setting, a window of exactly one syllable: L4 J_diff is dom1 0.53–0.60 against eq7 0.81–0.83, and silence lights at least as many neurons as a syllable;
   - less sparse targets (40 / 30 / 20 / 10 %): same conclusions;
   - top-k instead of the threshold: eq7 still does not decorrelate.

   I could not reproduce research 05's overlap of 0.24 inside the designer's local net.
5. **What does follow signal richness is the chain-bearing count.** At L1–L2 it grows with richness in every arm (sentence ÷ silence 2.6–3.3 at L1). At L4 it grows only in eq7, and there the counts are tiny (23–61 of 46,080). This comes from ARCH §二.3 (always-on units have no chain), not from convergent thresholds, and it is already in the design.
6. **Distractors.** eq7 halves the downstream effect of extra L1 units (0.86 against 1.54). It does this equally for scattered units and for a coherent patch, so it does not single out irrelevant units.

## 3. Part B, item 26 (SIMULATED; `babble.py`; `outB/AGG.txt`; A800 job `x7_babble2`)

**Harness:**
- **Body:** X1c's `world.py` Physics and render, unchanged (34 syllables, 24 muscles, one 4-muscle set per syllable, the clear / mumble / garble mouth rule).
- **Motor area:** a copy of X1c `core3._motor` (tonic blocking neurons, divisive gain, release at ≥ 0.6 × the maximum drive).
- **Learned link:** the stand-in map is removed. In its place is a learned matrix W from 180 auditory on-units to 24 muscles.
- **Rule** (§二.8 with the 11-01 held form):
  - when B rises while A is on (A rose with B, is held, or fell ≤ 2 ticks earlier), A→B grows;
  - when A rises while B is on (or B fell ≤ 2 ticks earlier), A→B weakens;
  - a fall changes nothing;
  - growth is concave, with the one global decay.
- **Babbling:** random drive on 3–5 muscles, held 3 ticks; it must beat the blocking neurons.
- **Probe:** each of the 34 syllables, by inner hearing (content units, return gain 1.0) or by hearing (the caregiver's rendered voice, gain 0.5, innate brake off during the probe).

**Main result (X1c physics, weakening rate η⁻ = strengthening rate η, 2000 babbles; 5000 babbles are no better):**

| arm | inner correct | hear correct | inner garble | true / other weight | top-4 muscles exactly right | strengthening ÷ weakening |
|---|---|---|---|---|---|---|
| (d) base | 0.01 ± 0.01 | 0.00 | 0.69 ± 0.12 | 0.52 (true pairs weakened) | 0.00 | 0.12 |
| (a) same tick | 0.00 | 0.00 | 0.01 ± 0.03 | 1.27 ± 0.03 | 0.02 (0.15 at 5000) | 2.43 |
| (b) bababa | 0.00 | 0.00 | 0.26 ± 0.31 | 0.69 | 0.00 | 0.65 |
| (c) imitation game | 0.00 | 0.00 | 0.58 ± 0.23 | 0.71 | 0.00 | 0.21 |
| hand-wired stand-in | 0.86 ± 0.03 | 0.27 ± 0.34 | 0.14 | — | — | — |

The weight ratio and the top-4 columns read the learned map directly and do not depend on the readout gain.

**Why each arm fails (PROVED from the rule's form; the measured counts match):**
- **Base:** every onset of the child's own sound comes 1 tick after a muscle onset, so the rule can only weaken sound→muscle. This is the known problem, confirmed.
- **Bababa:** each repetition gives one strengthening (the next muscle onset while the sound has just fallen) and one weakening (the sound rises again 1 tick later). The first repetition only weakens, so the net is 2η − 3η⁻, which is negative unless η⁻ < ⅔η.
- **Imitation game:** the child's first babble weakens the link, the caregiver's syllable strengthens it, and the child's own repeat weakens it again. The net is η − 2η⁻, which is ≤ 0 whenever η⁻ ≥ ½η. There were about 650 imitations per 5000 babbles.
- **Same tick:** it only strengthens, but 75 % of random babbles are garble. X1c's garble sound is random content, so it teaches random links that swamp the syllable signal (true / other only 1.27).

**Sensitivity runs (none passes):**
- **η⁻ = 0.5η:**
  - same-tick top-4 rises to 0.36 (0.58 at 5000);
  - bababa barely learns (weight ratio 1.17);
  - imitation and base still anti-learn.
- **"Lawful physics"** (the harvest's smallest test asked for it): a garbled gesture sounds like a blend of the syllables it partly forms.
  - Same-tick then learns the right map shape: top-4 exact 0.82–0.94, true / other 2.6–3.0.
  - Bababa learns it only at η⁻ = 0.5η (top-4 0.78).
  - Imitation still anti-learns.
- **Even the best learned map says only 0.02–0.05 of the syllables correctly**, against 0.86 for the stand-in. At the best gain in a sweep from ×0.25 to ×16 it reaches 0.19–0.23.
  - The reason is the margin: (weakest own-muscle drive) ÷ (strongest other drive) is 1.08 for the learned map against 1.50 for the stand-in.
  - X1c's release-by-sharpness needs about 1.5.

**Conclusion:** under the designer's rule, babbling cannot teach "heard sound → the same muscles". Two of the fixes do not change the sign at η⁻ = η (bababa and imitation); same-tick hearing gives a map too blunt for the motor competition. Generation 0 should hand-wire this map, which ARCH §二.12 allows.

**What might work (ASSUMED, untested):**
- The sound must be on before the muscle onset and must not rise again after it, for example:
  - an auditory trace held across the repetition by inner hearing;
  - or the caregiver's voice overlapping the child's own onset.
- The map would also need a larger margin: a sharper competition, or weakening of chance pairs.

## 4. Part C (SIMULATED)

**28a, whole-syllable competition** (`partc.py`, `outC/AGG.txt`):
- **Setup:** muscle→muscle links learn by the same rule during 2000 practice repetitions through the hand-wired map, with noise and 30 % cross-talk; inhibitory partners use the X1c form.
- **The links do become syllable-shaped:**
  - excitation between muscles of the same syllable: 0.81, against 0.40 for other pairs;
  - partner inhibition: 0.20, against 0.49.
- **One settling pass in the competition still removes little garble:**
  - with noise and cross-talk, garble goes 0.53 → 0.50;
  - on the clean probe, garble goes 0.12 → 0.18 and the correct share falls 0.88 → 0.81.
- **Partners alone do slightly better but still miss the −25 % pass:** cross-talk garble 0.70 → 0.62, correct 0.26 → 0.30.
- **Likely causes (ASSUMED):** syllables share up to 2 muscles, so a syllable's excitation also lifts its neighbours' muscles; and 38 % of the practice itself was garble.

**28b, pace from force grades** (`pace.py`). In this toy, a syllable's hold until the next one starts is ASSUMED to come from inner hearing.
- **Held form (11-01):**
  - every grade of syllable i (muscle, f1, f2, f3) links equally to syllable i+1 (2.00 each);
  - the next syllable's input therefore ramps 0.50 → 0.75 → 1.0 over the ticks and it starts early;
  - the length depends on the threshold: 1.26 / 1.85 / 2.66 ticks at φ = 0.4 / 0.6 / 0.8.
- **Onset form ("only just-changed units"):**
  - only the last grade links (f3 2.00, the others 0);
  - the input is a step at the taught onset;
  - the length is 3.00 at every φ.
- **Implication:** the force ramp sets the pace without a timer only if the motor area learns by the just-changed form. The held form, which the designer fixed for the PFC, breaks the pace in the motor area. This needs the designer's decision.

## 5. Harness changes, amendments and rule exceptions

- **No new mechanism.** No new module, rule or timer, and nothing listed in §三. 28b's onset form is the comparison the harvest asked for; it is the older wording of §二.8 ("只有刚变化的神经元参与").
- **Part A:**
  - The harness is new, built from local3d's net and threshold plus the project's Voice.
  - θ is calibrated per layer to the target sparsity. It rises with depth for dom1. For eq4 and eq7 it dips at L2–L3 (the layer below is already sparser) and rises again at L4.
  - The chain-bearing metrics were added after the first dev run and are reported as secondary only.
- **Part B amendment (made after seed-0 dev runs, before the final runs):**
  - The plan scaled the readout so the total learned weight equals the stand-in's. In arm (a) the own-voice rows took most of that weight.
  - So the main readout now matches the stand-in's weight on the content rows only.
  - Both forms are reported, plus the oracle gain sweep; the conclusion is the same under all three.
- **Part B additions:** the lawful-physics sensitivity, the 5000-babble extension, and the gain-free diagnostics (top-4 and margin).
- **Part B scope limits:**
  - No PFC was needed for this map.
  - The learned link is read out only in the probes; it does not drive the muscles during babbling (open loop).
  - The brake is off during probes.
  - The child's repeat in the imitation game is an ASSUMED behaviour.
- **Compute:**
  - Part A ran locally: one process, ≤ 225 MB, about 1 minute per 5-seed grid.
  - Part B ran on the A800 CPU queue: `x7_babble1` crashed when writing JSON; the rerun `x7_babble2` took 119 s.
  - Part C ran locally in under a minute, 35 MB.
- **Repo:** not touched; nothing pushed.

## 6. Files

Everything is in this folder (`v3/x7/`):
- **Plan:** `PLAN.md` (pre-registration and amendments).
- **Part A:** `sens.py` (harness), `aggA.py` (aggregator), side checks `sanity_rho.py`, `onechar_check.py`, `topk_check.py`.
- **Part B:** `babble.py` (harness), `aggB.py`, `margin_check.py`.
- **Part C:** `partc.py` (28a), `pace.py` (28b), `aggC.py`.
- **Copied inputs:** `world.py`, `core3.py` (from `v3/x1c`), `voice.py` (from `world/`).
- **Results:**
  - `outA/`: `AGG_main.txt`, `AGG_t40.txt`, `main.json`, `t40.json`, `sanity_rho.txt`, `onechar.txt`, `topk.txt`
  - `outB/`: `AGG.txt`, `{x1c,law}_r{10,05}.json` and `.log`, `margin.txt`, dev runs `dev*.json`
  - `outC/`: `AGG.txt`, `main.json`, `pace.json`

## summary_zh

1. 第 13 条（感觉区深层接几根差不多强的输入，用门槛做"几根同时亮才亮"）：**不行**。正反对编码里，声音每打开一根"高"线就关掉一根"低"线，所以亮多少和声音丰富不丰富无关，安静本身也是一个完整的读数。第 4 层亮的个数：安静 1878–1922，一句话 1633–1736，安静反而略多。
2. 第 13 条（续）：
   - 不同声音的重叠没有变小，反而变大（7 根输入 0.947，原来 0.893）；相似度保持也变差（相关系数 0.42 降到 0.28）。
   - 真正"信号强 = 亮得多"的，是去掉常亮神经元以后、有链式神经元的那部分：浅层每种接法都这样（一句话是安静的 2.6–3.3 倍）。这一点架构 §二.3 已经有了。
3. 第 26 条（只靠咿呀学会"听到的音 → 同样的肌肉"）：**不行**。四组（基础、同一拍听见、bababa、模仿游戏）说对的比例都是 0–0.01，手动接好的版本是 0.86。
4. 第 26 条原因：
   - 自己的声音晚一拍，按规则只会把"声音→肌肉"减弱（真实连接 / 其他连接 = 0.52，比随机还低）；
   - bababa 和模仿游戏每次都是"加强一次、减弱一两次"，净效果不是正的；
   - 同一拍听见能学到一点，但学到的对应太模糊，过不了运动区的竞争；
   - 所以第一代应该手动接好这条连接（§二.12 允许）。
5. 第 28a 条（整个音节一起竞争）：**不行**。肌肉之间确实学到了按音节分组的连接，但乱码只少了 5%（要求 25%），干净时说对率还从 0.88 降到 0.81。
6. 第 28b 条（力度神经元依次亮起定节奏，不用计时器）：**部分行**。运动区如果按"只有刚变化的神经元参与"来学，音节长度正好 3 拍（100%）；如果按 11-01 的"一直亮着的也长连接"来学，第一档也连到下一个音节，下一个音节提前开始，平均只有 1.85 拍。
7. 需要设计者定：前额叶要用"一直亮着的也长连接"，运动区的节奏却要"只有刚变化的才长连接"，两处是否用不同的形式？