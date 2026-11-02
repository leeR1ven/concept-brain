(Saved by the coordinator from the agent's final hand-back. In the repo: X1c/X1d files are imported from `../x1c` and `../x1d`; `core5.py` is X1f's copy of X1e's core; debug scripts and logs were not copied.)

# X1f report: a generation-0 brain with hand-added connections, and learning on top

Labels: **PROVED** means exact numbers from runs, 5 seeds (mean ± SD) unless a count is given. **SIMULATED** means 1 seed, a pilot or a screen. **ASSUMED** means my judgement, not tested.

Folder: `v3/x1f/`. I did not modify the repo or any other agent's folder. Locally I ran 1 process at a time, peak about 60 MB. Three A800 CPU jobs ran (`x1f_g0a`, `x1f_g0na`, `x1f_diag1`), at most 2 at a time.

I followed the coordinator's 11-01 update (ARCH §二.11):
- **Main arm G0:** hand-wired sound ↔ action map only, then an overhearing stage, then a speaking stage.
- **Diagnostic arms:** G0T (hand-wired thinking + responding on a base set) and G0P (paths for all seen pairs). Both are labelled DIAGNOSTIC.

## 0. Bottom line

1. **Speech basics work** (PROVED). With the hand-wired map A1, the syllable-shape blocking gate A2 and two existing motor knobs:
   - a forced whole-answer inner hearing is said exactly 1.00 (G0) or 0.94 ± 0.04 (G0T);
   - garble is 0, talk-over 0 and failure to yield 0;
   - the newborn never babbles.
   - X1c's stand-in map garbled 77–95 % of forced answers.
2. **Hand-wired thinking works for the wired pairs, learning off** (PROVED, G0T, 5 seeds).
   - Wired-pair accuracy: probes 0.95 ± 0.04; live world 0.96 ± 0.04 (format caregiver) and 0.98 ± 0.03 (realistic caregiver).
   - Talk-over 0, failure to yield 0, garble 0 per exchange.
   - Question decoding at lag 12 on wired pairs: 0.85 ± 0.12.
3. **Learning new pairs fails on every route** (PROVED).
   - **Main path (G0):** overhearing 100 → 10,000 dialogues (30,000 for seed 0), then speaking. Heard-pair accuracy 0.00–0.03 at every point; held-out 0.00–0.04.
   - **Diagnostic arm G0T:** never-wired pairs stay at 0.00–0.05.
   - **G0P, the decisive one:** every seen pair has a perfect, question-specific, held path (decoding at lag 12 = 0.94 on the new pairs) and the child is quiet. After 10,000 overheard dialogues the new pairs are still at 0.00–0.03.
4. **The chain breaks at "learned specific prediction"** (PROVED). The learned top → L1 drive into the answer's inner-hearing copies is 0.10–0.24. That is below the 0.35 prediction floor, and the best *wrong* syllable gets as much or more.
   - Raising the learned gain 4× lifts the drive above the floor (1.2), but the wrong syllable rises more (1.7). The wired pairs then collapse from 0.91 to 0.
5. **Learning in the top layer's internal links (L33, X1e's form) destroys the hand-wired thinking** (PROVED). Wired pairs fall 0.95 → 0.35 over the life. Wiping L33 restores 0.87. Without L33 learning the wired pairs stay at 0.85–1.00 through 3,000 overheard + 300 spoken + 1,000 untutored exchanges.
6. **Corrections of wired answers never switch** (PROVED). 0 of 20 changed wired pairs switched (20 = 10 in G0T + 10 in G0T_noL33); the old answer repeats in up to 21 asks. Nothing was learned, so there were no learned pairs to change.
7. **Merge (test 5) was skipped.** Its precondition, test 2 passing, failed.

## 1. What was built

The base is X1d's RNpm:
- no PFC inhibitory neurons;
- X3 patch-mean;
- the held-neuron onset rule everywhere, post side = the next input;
- L33 learning in X1e's form (RNpm_L2: onset rule, two-layer weights g33 = 2, dec_33 = 0.95).

`core6.py` is X1e's core5 plus default-off hooks. It reproduces core5 bit-exactly over 3,000 ticks (PROVED, `eqtest6.py`).

Parameter changes, written into PLAN.md before the 5-seed runs (PROVED, seed-0 screens):
- **At X1d's thresholds (f2 0.22, f3 0.3) the top layer barely sees many syllables.** 24 of 27 syllables light fewer than 8 reliable L3 units across voices; 色, 大 and 香 light about 1 unit at all.
- **Lower thresholds alone saturate.** With f2 0.18, f3 0.25, after 300 overheard dialogues mean L3 is 300 of 400 and P(saturated) 0.20, with or without L33.
- **Fix used.** The X3 patch-mean is applied to the whole top-layer drive (pm_ff = 1, X1e's option, ARCH §四.10), with f2 0.18 and f3 0.08.
  - Every syllable lights 32–60 L3 units (reliable within a voice: median 39, minimum 23).
  - L3 stays bounded: mean 50–110, maximum 167 of 400, P(saturated) 0 in all lives.

Speech-basics screen (SIMULATED, seed 0, learning off):

| map / motor | forced whole answer said exactly | garble | mixed inner hearing (2 syllables): said / garble | spontaneous speech |
|---|---|---|---|---|
| X1c stand-in, X1c knobs | 0.04–0.09 | 0.91–0.95 | 1.00 / 0.9 | 0 |
| A1 map, c_mb 0, g_dir 0.25 | 1.00 | 0 | 0.82–1.00 / 0.82–1.00 | 0 |
| A1 + A2 gate (final) | 1.00 | 0 | 0.14 / 0.00 | 0 |

## 2. Test 1: sanity, learning off (PROVED, 5 seeds)

| measure | G0 (speech basics only) | G0T wired pairs | pass? |
|---|---|---|---|
| forced whole-answer inner hearing said exactly | 1.00 ± 0.00 | 0.94 ± 0.04 | yes (≥ 0.9) |
| wired-pair accuracy: probe / live format / live real | – | 0.95 ± 0.04 / 0.96 ± 0.04 / 0.98 ± 0.03 | yes (≥ 0.8) |
| never-wired seen pairs: probe / live | 0 / 0 (silent) | 0.02 / 0.01–0.02 (silent 0.87) | – |
| talk-over / failure to yield per exchange | 0 / 0 | 0 / 0 | yes |
| garbled utterances per exchange | 0 | 0 | yes |
| question decoding at lag 12, mouth muted (wired / never-wired / held-out) | – | 0.85 ± 0.12 / 0.33 / 0.28 | yes (≥ 0.8) |
| phase-1 decoding over all 32 questions, lags 0 / 6 / 12 (chance 0.031) | 0.16 / 0 / 0 (pause dies at birth) | 0.77 / 0.52 / 0.53 | – |

In G0T the answer comes at lag 1.0–1.3: the plan forms early and fires late, once the partner's voice stops braking.

## 3. Test 2: learning

### 3.1 Main path G0: overhearing, then speaking

Accuracy curve against the number of overheard dialogues (PROVED, 5 seeds; one seed to 30,000):

| dialogues overheard | 100 | 300 | 1,000 | 3,000 | 10,000 | 30,000 (seed 0) | after speaking 300 | after untutored 1,000 |
|---|---|---|---|---|---|---|---|---|
| heard pairs, accuracy | 0.00 | 0.02 | 0.03 | 0.02 | 0.02 | 0.00 | 0.02 | 0.01 |
| held-out (never heard) | 0.00 | 0.00 | 0.02 | 0.04 | 0.04 | 0.00 | 0.02 | 0.00 |
| own − other prediction specificity | 0.02 | 0.02 | 0.02 | 0.03 | 0.02 | 0.01 | 0.03 | 0.01 |
| child utterances per dialogue while overhearing | 0.65 | 0.69 | 0.88 | 0.98 | 1.08 | 0.15 | – | – |
| of which garbled | 0.24 | 0.14 | 0.12 | 0.16 | 0.20 | – | – | – |

- The no-L33 variant (G0_noL33) is the same: accuracy ≤ 0.05, and 0.94–1.29 utterances per dialogue.
- Path after 10,000 dialogues: question decoding at lag 0 / 6 / 12 = 0.26 / 0.05 / 0.01. In seed 0, same ≈ cross ≈ 0.75 at lag 12: every question falls into one generic state.
- The predictions are dominated by a hub syllable (甜 or 蕉, share 0.5–0.93). When that hub prediction is sharp, the child says it, so the child is **not** quiet while overhearing. The criterion was ≤ 0.1 per dialogue; this fails.

### 3.2 Diagnostic G0T: wired base set, then overhearing and speaking for the never-wired pairs

| checkpoint | wired: L33 on | wired: L33 off | never-wired: L33 on / off | held-out: on / off |
|---|---|---|---|---|
| t1 | 0.95 | 0.95 | 0.02 / 0.02 | 0.02 / 0.02 |
| overheard 300 | 0.67 | 0.75 | 0.04 / 0.02 | 0.00 / 0.10 |
| overheard 1,000 | 0.80 | 0.87 | 0.05 / 0.04 | 0.02 / 0.04 |
| overheard 3,000 | 0.71 | 0.85 | 0.00 / 0.04 | 0.02 / 0.02 |
| speaking 300 (probe; live) | 0.40; 0.48 | 0.91; 0.86 | 0.00 / 0.04 | 0.00 / 0.02 |
| untutored 1,000 (probe; live) | 0.35; 0.36 | 0.85; 0.88 | 0.02 / 0.02 | 0.00 / 0.02 |
| after wiping L33 | 0.87 | 0.85 | 0.02 / 0.02 | – |

- Never-wired pairs reach 0.00–0.05 against the pass mark of 0.5: **fail**.
- Wired pairs stay ≥ 0.8 only without L33 learning.
- While overhearing, the G0T child speaks 0.9–1.3 times per dialogue. These are mostly correct answers to wired questions, given in the 2–6-tick gap before the adult answers. Garble is 0.02–0.25 per dialogue, and the child yields: it speaks during adult speech ≤ 0.01 per dialogue.

### 3.3 How the new pairs got their paths

- **G0T.** A never-wired pair whose filler has a hold keeps that filler hold through the pause. This is a filler-level path, not a question-specific one: decoding at lag 12 = 0.29–0.40.
- **L33 learning formed no new question-specific assemblies.** New-pair decoding did not rise (0.33 → 0.29–0.36), and specificity stayed at 0.01–0.04. L33 instead pushed spurious drive into the hand-wired working memory and broke it. Wiping L33 restores 0.87; L33 had 33–47 % of all pairs at positive weight.
- **G0P (DIAGNOSTIC: paths for every seen pair, inner-hearing links C for the base pairs only).**
  - The new pairs get perfect held, moving paths: decoding at lag 12 = 0.94 ± 0.09.
  - The child is quiet: 0 garble, 0.5 utterances per dialogue, all of them wired answers.
  - The wired pairs stay at 0.94–1.00.
  - After 10,000 overheard dialogues, new-pair accuracy is 0.00–0.03 and specificity 0.00–0.04.

### 3.4 Where the chain breaks: learned drive into the answer's L1 copies

This is the drive before the threshold, from the probe at lags 1–6. The prediction floor is 0.35.

| arm | into the answer's 1st-syllable copies | into the best wrong syllable | mean over content copies |
|---|---|---|---|
| G0T, overheard 1,000–3,000 | 0.22 | 0.28–0.30 | 0.06 |
| G0P_noL33, overheard 10,000 (3 seeds) | 0.12–0.19 | 0.12–0.23 | 0.02–0.04 |
| G0P_noL33 with g31 × 4 (seed 0) | 1.21 | 1.67 | 0.69 |

Diagnosis:
- **PROVED:** the numbers in the table.
- **ASSUMED, the mechanism:**
  - The ARCH two-layer weights are scale-free, so the drive a state can put on L1 grows with the number of its active units and shrinks with the in-total of the target copies.
  - A small, specific held state (4–5 units) cannot reach the floor.
  - The large, generic feed-forward codes (the last question syllable, shared across a frame) carry most of the learned drive. They predict every answer of the frame, so the prediction is a frame-level mixture plus a hub.
  - A2 then correctly keeps quiet, because the inner hearing is unsure.

## 4. Test 3: stability (PROVED)

| arm | untutored 1,000: wired | untutored 1,000: learned | saturation P(L3 ≥ 90 %) | amnesia (wipe L33) | pass? |
|---|---|---|---|---|---|
| G0T_noL33 | probe 0.85 ± 0.07, live 0.88 ± 0.05 (from 0.95) | none learned | 0 | 0.85 → 0.85 | yes for wired |
| G0T (L33 on) | 0.35 ± 0.12 | none | 0 | 0.35 → 0.87 | no |
| G0 | – | 0.01 | 0 | 0.01 → 0.01 | nothing to keep |

There is no hippocampus in this build, so there was no fast store to wipe. I wiped the fastest-decaying learned store (L33) instead, as written in PLAN.md.

## 5. Test 4: corrections (PROVED)

4 answers were changed per seed: 2 wired and 2 never-wired (colour or size pairs).

- **Wired pairs, 0 of 20 switched** (10 in G0T + 10 in G0T_noL33, 300 caregiver exchanges).
  - G0T_noL33: the old answer repeats in 8–20 asks (`OOOOOOOO…`).
  - G0T: the old answer, or silence once L33 has broken the path.
  - Hand-wired links are birth links. Under the held-neuron rule, Q-state → old syllable only ever strengthens, so a wired answer cannot be unlearned; it could only be outcompeted.
- **Never-wired changed pairs:** 3 of 10 gave the new answer once (after 0–6 re-teachings), never stably.
- **Unchanged wired pairs** (probe, about 9 pairs per seed, noisy): G0T_noL33 0.80 → 0.64.
- **G0:** there were no learned pairs to change; 0 of 20 changed pairs were ever answered.

## 6. Extra: X6b's innate "listening" signal on the PFC (SIMULATED, 3 seeds, 3,000 overheard dialogues)

`core7.py`, g_listen 0.8: while a partner voice is loud in the input layer (the same reading as the brake), L3 lateral drive and top → L1 drive are scaled by 0.2.

| | same@12 | cross@12 | cross same frame@12 | question decoding 0 / 6 / 12 | frame decoding@12 | filler given frame@12 (chance 0.125) | own / other specificity | accuracy |
|---|---|---|---|---|---|---|---|---|
| off (seeds 0, 1) | 0.76–0.81 | 0.76–0.81 | 0.76–0.81 | 0.13–0.23 / 0.09–0.11 / 0.02–0.06 | 0.20–0.23 | 0.13–0.15 | 0.19 / 0.18 | 0 |
| on, seed 0 | 0.23 | 0.08 | 0.22 | 0.11 / 0.08 / 0.05 | 0.37 | 0.04 | 0.15 / 0.08 | 0 |
| on, seeds 1 and 2 | 0.00 | 0.00 | 0.00 | 0.10–0.15 / 0–0.02 / 0–0.01 | 0.02 | 0–0.02 | 0.03–0.09 / 0.02–0.04 | 0–0.04 |

- The signal removes the generic attractor: in seed 0 the path becomes input-locked at frame level.
- In 2 of 3 seeds the pause dies.
- It does not carry the filler. The question-end code is the last syllable, which is shared by the whole frame.
- Not adopted.

## 7. Diagnosis summary: where the chain breaks

1. **Reliable codes**, fixed here:
   - At X1d's thresholds the top layer misses syllables. With pm_ff it sees them.
   - The codes are voice-dependent: of the units reliable within one voice, about 30 % stay reliable across voices.
2. **Question-specific path:**
   - The random top layer never gives one (G0: decoding at lag 12 ≤ 0.06).
   - The question-end code is frame-only (X1d: cross same frame 0.28 vs same 0.32).
   - Hand wiring gives one (G0T 0.85, G0P 0.94).
   - L33 learning does not create one, and it breaks the hand-wired one.
3. **Specific learned prediction: this is the failing link, even with a perfect path.**
   - Learned drive into the answer is 0.12–0.24, below the 0.35 floor, and no larger than into a wrong syllable.
   - More gain makes the wrong syllables win.
4. **Release:**
   - Works whenever the inner hearing is sharp (forced 0.94–1.00).
   - A2 turns mixtures into silence.
   - In G0, babble returns as the learned hub prediction becomes sharp (0.65–1.3 utterances per overheard dialogue).

## 8. Deviations and limits

**Diagnostic arm, B/C hand wiring** (labelled; the forms are my choice):
- The working-memory (WM) part of L3, 4–5 of the 12 patches (214 units in seed 0), loses its random L2 → L3 and birth L3 → L3 inputs.
- It has its own fixed threshold, no patch-mean, and hand-wired inhibitory links between assemblies ("rarely co-active neurons inhibit each other", ARCH §二.4). This is PFC inhibition inside the diagnostic arm only.
- Its sound input comes from the heard L1 content copies. The L2 and L3 codes are voice-dependent, and several syllables cannot be told apart there.
- The hand-wired top → L1 links are not masked by M31 locality.

**Other limits:**
- The X1c body has no order or force grades within a syllable, so A1 is a set map, not a sequence map.
- G0T and G0P overhearing ran to 3,000 and 10,000 dialogues. Only G0 seed 0 went to 30,000.
- G0P has 3 seeds (plus 1 seed with L33 on and 1 with g31 × 4).
- The listening test has 3 seeds.
- The probes use 11 pairs per set, so single checkpoints are noisy (± 0.1).

## 9. Registry (full table in `REGISTRY.md`)

| set | area | from → to | why |
|---|---|---|---|
| A1 | auditory → motor | L1 content copy of each feature → muscles; NNLS (ridge 0.3) from each syllable's 10-feature code to its 4-muscle set; 598 links; replaces Mmap on both routes | thinking a syllable's sound produces it, with no learning |
| A2 | motor | muscles about to fire this tick → two groups of blocking neurons; block if < 3 or > 5 would fire | release by sharpness: unsure means silent, not garble |
| knobs | motor | c_mb 1.2 → 0; g_dir 0.5 → 0.25 | removes the speech latch and echo interference at syllable transitions |
| B0 (diag.) | L3 | WM patches: random inputs removed, own threshold 1.0 | clean held states |
| B1 (diag.) | L1 → WM | heard copies of a syllable → assembly (5.5 for a filler hold, 0.8 for a path step) | the WM's sound input |
| B2 / B3 (diag.) | WM ↔ WM | self links 3.0 per assembly; −3.5 / 5 between units not wired together | an assembly holds itself; one held assembly at a time |
| B4 (diag.) | WM → WM | filler hold H_X & last question syllable → Q1; Qk & answer syllable k → Q(k+1); Qn & last answer syllable → D (answered); successor inhibits predecessor | a fixed question-specific path |
| C1 (diag.) | WM → L1 | Qk → the 10 content copies of answer syllable k | inner hearing of the answer, said through A1 |

The base set is 11 of the 22 seen pairs, fixed per seed. The other 11 seen pairs and all 10 held-out pairs are never wired. G0P wires paths (B) for all 22 seen pairs but C only for the base 11.

## 10. Files (agent's working folder; the repo copy holds the code, plans, registry, results and job lists)

- **Plans:** `PLAN.md` (pass criteria and parameter changes, written before the runs), `REGISTRY.md`.
- **Code:**
  - core and wiring: `core6.py` (hooks, A2 gate, working memory), `core7.py` (+ listening signal), `wire.py` (A1 and B/C construction);
  - stages and probes: `world6.py` (overhearing world, 4 adult voices), `common6.py` (birth, probes, learned-drive measure), `run6.py` (all stages and tests);
  - tools: `agg6.py`, `listen_test.py`, `screen_motor.py`, `explore_codes*.py`, `sat_test.py`, `eqtest6.py`, `dbg_*.py`;
  - copies of X1c/X1d/X1e code: `core3.py`, `core4.py`, `core5.py`, `world.py`, `run.py`, `phase1.py`, `run2.py`, `regimes.py`.
- **Results:**
  - `out6/` (local): `G0T_s0-4.json`, `G0T_noL33_s0-4.json`;
  - `out6_a800/`: `G0_s0-4.json`, `G0_long_s0.json`, `G0_noL33_s0-4.json`, `G0P_noL33_s0-2.json`, `G0P_s0.json`, `G0P_noL33_g31x4_s0.json`;
  - `out7/listen_*.json`.
- **Logs:** `logs/`, `logs_a800/`; job lists in `jobs/`.

## summary_zh

1. 按设计者 11-01 的系统化训练法，先手动接好"声音↔肌肉"：模型心里想到一个答案，就能本能地说对（5 次一生 1.00）。原来 X1c 那张替代图会说成乱码（77–95%）。
2. 加了"指令像一个完整音节才放行"的阻断连接：心里同时想到两个音（没把握）时就不说（乱码 0）。等待时安静，不抢话，不会不让。
3. 主路线（只接说话基本功，再旁听大量大人对话，然后被问）：旁听 100 到 1 万段（一次一生到 3 万段），答对率始终 0–0.03，没听过的也是 0–0.04。
4. 主路线里，前额叶停顿后的想法对所有问题都一样，预测被一个"万能字"占住。孩子旁听时还会说出这个字（每段对话约 1 次），不够安静。
5. 诊断组（手动接好一半问题的"思考+回应"）：接好的题在真实的连续世界里答对 0.96–0.98，不抢话、不乱码；问题之后 12 拍还能分出是哪一题（0.85）。
6. 但没接过的题，旁听 3,000 段再加教 300 次后，仍只有 0–0.05。
7. 关键诊断：即使给所有问题都手动接好一条完美的、各题不同的思路，孩子也安静，旁听 1 万段后新题仍答不出（0–0.03）。原因是学到的"预测答案"那股推动只有门槛的一半左右，而且推错字和推对字一样多；把它调大，错的字涨得更多。断点在"学出针对问题的预测"这一环。
8. 最后一层内部连接的学习（X1e 的形式）不但没形成新的思路，反而把手动接好的思路搅乱（答对率 0.95→0.35，删掉这些学到的连接就恢复到 0.87）；关掉它，接好的题一生都保持在 0.85 以上。
9. 改正：手动接好的答案教 8–20 次也改不过来，因为这条规则只会加强旧答案。试了 X6b 的"正在听时压低回路"：它去掉了"万能状态"，但带不住问题里的那个词，3 次里有 2 次停顿时想法熄灭了。
10. 结论：第一代手动接线能让模型"会想、会等、会答"。但在现有学习规则和两层权重下，学不出新的问答。下一步要先解决"小而具体的想法推不动、常见的大编码推一切"这个问题，再谈融合（第 5 项因此没做）。