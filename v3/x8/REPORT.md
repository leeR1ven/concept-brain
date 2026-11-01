(Saved by the coordinator from the agent's final hand-back. In the repo: `x4_merge.py` is imported from `../x345/`; `results/` holds the tables (the per-run JSON are not committed).)

# X8 report: the designer's two correction mechanisms (ARCH §二.8) in the X4/X4b harness

Labels: **PROVED** means code or arithmetic. **SIMULATED** means a run (5 seeds unless noted; mean ± SD; Δ is the paired per-seed difference). **ASSUMED** means my judgement.

## 0. Bottom line

No arm meets all the pre-registered criteria. **Items 3, 4 and 12 all FAIL as specified.** Each one gives half of what IDEAS #22 asks for:

- **Kept for life against time alone:** items 3 and 12 do this well.
- **Learning new things as an adult:** the fast store does this well (item 20).
- **Correcting answers:** none of the three makes this fast.
  - The only thing that helps much is the "upset" half of the reward tool, which reverses the just-made changes (ARCH §二.8, already adopted).
  - With it, corrections of freshly learned answers take 5.7 re-teachings (65 % switch within ≤ 5). The target was ≥ 90 % within ≤ 5.

Two other effects decide the outcome.

1. **Retention fails in every arm.** It fails because of interference from other experience, not because of decay.
   - The X4 PFC code after "q1 q2" contains 64 % of the code for "q2" heard alone (SIMULATED).
   - Unrelated lines and unanswered questions overwrite the answers. On seed 0, after 600 unanswered asks, 22 of 22 lost answers were replaced by the char that follows q2 in another question.
   - The best cortex-only retention after the 6,000-line stretch is 0.57 (item 3 with own voice). The target was 0.8.
   - This is ARCH §四.6 ("same question → same path, different questions → different paths"). The decay rule cannot fix it.
2. **Every rehearsal of the old answer re-teaches it.** Rehearsal comes from three sources: own voice, inner recall from the fast store, and recall during the correction episode itself.
   - Under concave growth, a link's strength is a use count (PROVED in §3).
   - So a correction only wins if the old answer gets fewer strengthenings than the new one. Rehearsal at the same rate makes the switch impossible.

## 1. Verdict table

Unless a row says otherwise: w_own = 0, post-stretch protocol as pre-registered, 5 seeds.

| item | verdict | key numbers | paired difference |
|---|---|---|---|
| **3** concave on the way down (weakening and decay ÷ (1+w)^pow, 1 % floor) | **FAIL** (P1, P2). PARTIAL on "kept for life with nothing else happening" | <ul><li>1M untouched ticks: retention **0.87 ± 0.06**, base 0.12 ± 0.04</li><li>after the 6,000-line stretch: 0.09, base 0.01</li><li>re-teachings to switch: 13.9, base 8.8 (≤ 5: 0.10 vs 0.50)</li><li>freshly learned answers: 13.6, base 11.8 (≤ 5: 0 vs 0)</li></ul> | <ul><li>life +0.75 ± 0.08</li><li>stretch +0.08 ± 0.01</li><li>re-teachings +5.0 ± 1.5 (fresh +1.75 ± 0.6): harder to correct</li></ul> |
| 3, the weakening half only (= X4's own code) | FAIL | life 0.14; re-teachings 11.0 | re-teachings +2.1 ± 0.7 |
| **4** fast store + pleased reward | **FAIL** for corrections and the amnesia test | <ul><li>cortex retention with the store wiped: 0.09</li><li>cortex retention with the store kept: 0.34 (recall re-teaches the cortex)</li><li>re-teachings 10.3 (≤ 5: 0.25)</li><li>freshly learned: 10.4 (0.25)</li></ul> | <ul><li>re-teachings vs base +1.4 ± 3.7 (post-stretch), −1.4 ± 2.0 (fresh): no gain</li><li>amnesia +0.08 ± 0.01</li></ul> |
| 3 + 4 together | **FAIL** | <ul><li>store kept: cortex retention **0.73** vs 0.09</li><li>store wiped: 0.11</li><li>re-teachings **19.6** (0 % ≤ 5); fresh 17.2</li></ul> | vs item 3: <ul><li>kept +0.65 ± 0.06</li><li>amnesia +0.03 ± 0.01</li><li>re-teachings +5.7 ± 3.3 (fresh +3.7 ± 0.8): worse</li></ul> |
| 3 + 4 + upset (reverse the last 24 ticks of changes when the answer is wrong) | **PARTIAL**, the best arm | <ul><li>re-teachings 8.6 post-stretch (≤ 5: 0.20)</li><li>freshly learned: **5.7 (≤ 5: 0.65)**</li><li>the same at every own-voice weight</li><li>unchanged lessons vs control: 0.96</li></ul> | <ul><li>vs 3 + 4: −11.0 ± 1.0 (post), −11.6 ± 2.2 (fresh)</li><li>vs item 3, fresh: −7.9 ± 1.6</li></ul> |
| 5, pleased reward | no effect on corrections; helps write new lessons into the cortex | <ul><li>re-teachings: item 4 vs store without reward, Δ +0.15 ± 0.20</li><li>adult's new lessons known by the cortex alone: 0.63 vs 0.23</li></ul> | <ul><li>+0.39 ± 0.07</li><li>3 + 4 vs 3 + store without reward: +0.23 ± 0.11</li></ul> |
| **12** fast part in each synapse (w = birth + F + S) | **FAIL** | <ul><li>life **0.96 ± 0.02**</li><li>stretch 0.10</li><li>re-teachings 7.7 (≤ 5: 0.50) post-stretch</li><li>freshly learned: 10.4 (0 %)</li><li>with upset + own voice 1.0: 8.3</li></ul> | <ul><li>life +0.84 ± 0.03</li><li>re-teachings −1.15 ± 0.6 (post), −1.4 ± 0.4 (fresh)</li></ul> |
| **20** adults learn fast through the store | **PASS** for behaviour; PARTIAL for the cortex | <ul><li>teachings before the first right answer: adult without store 9.4 (fresh 3.2); with store 4.2 (fresh 4.0)</li><li>3 + 4 + upset: 3.1 (fresh 1.6)</li><li>new lessons in the adult's cortex after 10 rounds: 0.29–0.63</li></ul> | <ul><li>store vs no store, adult: −5.2 ± 0.6</li><li>adult minus fresh: base +6.2 ± 0.4, store +0.2 ± 0.9</li></ul> |
| self-rehearsal (protocol step 5) | corrections beat it only with upset | <ul><li>base re-teachings at w_own 0 / 0.25 / 0.5 / 1: 8.8 / 10.8 / 17.5 / 20.1</li><li>base retention: 0.01 / 0.10 / 0.29 / 0.48</li><li>3 + 4 + upset: 8.6 / 8.0 / 8.2 / 8.2</li></ul> | – |

**Pass criteria (pre-registered).**
- **P1 (cortex-only retention ≥ 0.8):** fails in every arm; the maximum is 0.57.
- **P2 (≥ 90 % switch within ≤ 5):** fails in every arm.
  - Post-stretch maximum 0.50 (base and item 12, mostly re-learning answers already lost).
  - Freshly learned: 0.65 (3 + 4 + upset).
  - Sensitivity, 3 seeds: 0.83 for 3 + 4 + upset with store decay 0.95.
- **P3 (unchanged lessons):**
  - As end/start, the ratio is 0.5–0.99 in every arm, because the 400 filler lines of phase 3 erode them.
  - Against the no-correction control, the corrections themselves cost 0–7 % in most arms. Item 12 is the exception: 0.75–0.84.
  - Where few unchanged lessons were known at the start (base at w 0: 0–2 per seed), the ratio is noise.
- **P4 (self-rehearsal):** fails in every arm.

## 2. Sensitivity (3 + 4 and 3 + 4 + upset, seeds 0–2, w_own 0 and 1; `results/sens_table.txt`)

The store's decay speed is a real trade-off; no setting gives both retention and fast corrections. All rows below are 3 + 4 + upset at w_own 0.

| store decay per 200 ticks | cortex retention, store kept | re-teachings | switched ≤ 5 |
|---|---|---|---|
| 0.95 (fast) | 0.48 | **3.8** | **0.83** |
| 0.97 (main) | 0.73 | 8.2 | 0.08 |
| 0.99 (slow) | **0.93** | 17.8 | 0.00 |

- Return strength ×2: 6.2 re-teachings, 0.50 switched.
- Cortex-only retention with the store wiped is unchanged across all settings: 0.07–0.13 at w 0, 0.24–0.51 at w 1.

## 3. Mechanisms

**Concave growth makes strength a use count (PROVED).**
- With g = 1/(1 + w), each strengthening raises Φ(w) = w + w²/2 by η + η²/(2(1+w)²). Numerically, 10 events give Φ 3.16 vs 3.0, and 30 give 9.24 vs 9.0.
- So to overtake an answer taught n times with the 1.25× sharpness, a new answer needs about 1.5 n strengthenings, unless the old one weakens.
- Rehearsal of the old answer at the correction rate (own voice at weight 1, or recall) keeps Φ_old ahead forever.
- This is the X2 lock-in, reproduced: after 3,000 self-answered asks, base at w_own 1 never switched 15 of 20 changed answers in 20 re-teachings.

**Why the fast store does not speed corrections (SIMULATED, traces in `diag/d4.py`).**
- In each correction episode the adult first asks the question. The summed drive (cortex + store) picks the old answer, the store supports it, so recall re-lights the old answer in AUD. Both stores then store it again before the recast arrives.
- The store's old-answer link even grew: 2.65 → 3.98 θ over 5 episodes.
- The upset reversal undoes exactly that increment: the store's old link fell 2.65 → 1.19 θ after one episode.
- The pleased reward fires only once the answer is already right, so it cannot speed a switch.

**Recall does consolidate, but only while the store holds the lesson (SIMULATED).**
- With the store kept, the cortex's own retention after the stretch is 0.73 (3 + 4) vs 0.09 (item 3 alone).
- Wipe the store at the start of the stretch and it is 0.11. The store's lesson memory survives only because asks keep cueing it.

**The retention loss is interference, not decay (SIMULATED, `diag/d1.py`, base, seed 0).**

| what happened after learning | cortex right (0.88 before) |
|---|---|
| 600 lines' worth of untouched time | 0.88 |
| 600 filler lines | 0.71 |
| 3,000 filler lines | 0.11 |
| 600 unanswered asks | 0.40 |

- Most losses are "no answer": a competitor reaches the 1.25× sharpness limit.

## 4. What I changed in the harness, and rule exceptions

- `x4_merge.py` is copied unchanged and imported. `x8.py` adds:
  - **Rule variants:** base = plain weakening and decay; item 3 = concave weakening and decay; X4's code = concave weakening only.
  - **The fast store:** its own population of 4,096 units, 1 % active, fed by AUD chain copies (fan-in 32), with birth recurrence (1.0).
    - It is silent without chain input.
    - Its learned links into the same AUD use the same rule, with η 1.0 and decay 0.97 per 200 ticks.
  - **Recall:** in the silent answer moment, AUD lights the winner of the summed two-layer return drive (cortex + store, scale 1) when the store's drive to that char reaches θ.
  - **Reward tool:** changes from the last 24 ticks, fading; pleased adds them again ×2, upset reverses them (net ×−2).
  - **Judged asks with recasts.**
  - **Own voice:** a fixed subset of round(w × 16) of the char's content units.
  - **Item 12:** F decays 0.99 per 200 ticks, S never decays; F moves into S when F > 0.5 or at a pleased moment.
- **Plan changes, all logged in PLAN.md §9.** Items 1–7 came from seed-0 pilots before the main runs; item 8 was added after the first main job.
  1. An ask now has the same step structure as a heard line (two silent steps had shifted the live threshold).
  2. Tests use a fixed probe threshold.
  3. Store wiring and "silent without input": the PFC-like setting gave 40 % false recalls on never-heard questions.
  4. Recall-lighting rule: with "store alone wins", false recalls taught the cortex wrong answers.
  5. The planned r_hip calibration was dropped (r_hip = 1). The rule could not be met by scaling, and it ignored false recalls.
  6. A no-correction control was added for P3.
  7. Item 12 ran in the same batch; it is judged only because items 3 and 4 failed.
  8. Added after the first main results: corrections right after learning (job `x8_c1`, 165 runs), because the stretch had already erased most of the answers to be changed.
- **Rule exceptions (ASSUMED to be minor form choices, not new modules):**
  - The store's recurrence alone cannot fire it.
  - Recall lights AUD only when the store reaches threshold for the winning char.
  - The stop-neuron link of "upset" is not modelled (there is no motor area).
- **Known harness deviations:**
  - X4 does not let held PFC units learn (ARCH §二.8 says they should). Only 2.6 % of PFC units are held across an onset, so the effect is minor.
  - There are no thinking dynamics; speech is the readout competition.
- **Compute.** Local runs: pilots and diagnostics only, one process, peak 291 MB. Server: six A800 jobs (`x8_m1`–`m4`, `x8_c1`, `x8_msens`), never more than 2 of mine queued or running. All 388 runs returned rc = 0.

## 5. Files (agent's working folder; the repo copy holds the code, plan, diagnostics and tables)

- **Plan:** `PLAN.md` (the plan plus the change log).
- **Code:**
  - `x8.py`, `x4_merge.py` (the unchanged copy);
  - `make_jobs.py`, `analyze.py`;
  - `diag/d1.py`–`d5.py`: erosion sources, store specificity, store drive scales, correction traces, ask-interference proof.
- **Results:**
  - `results/main/` (175 JSONs) and `results/main_tables.md`;
  - `results/c/` (165 JSONs) and `results/c_tables.md`;
  - `results/sens/` (48 JSONs) and `results/sens_table.txt`.
- **Pilots:** `pilot_*.json`, `full_A34U.json`.
- **Server copies:** `.../arch/v2/gpu/queue_a800/x8_*/result/`.

## summary_zh
1. 第3条（越强越难改，也用在减弱和降权上）：部分行。一辈子什么都不发生时能记住（100万拍后 0.87，原规则 0.12）；但中间听了别的话照样忘（6000句后 0.09），改答案也更难了（13.9 次，原规则 8.8 次）。
2. 第4条（海马体快速记住 + 高兴时奖励）：不行。改答案仍要约 10 次，因为每次大人先问问题时，海马体先把旧答案又想起来、又存了一遍；删掉海马体后皮层只记得 0.09。
3. 第4条再加"难过时把刚才的变化反过来"（§二.8 里已有）：部分行，是所有组里最好的。刚学会的答案平均 5.7 次改过来，65% 在 5 次以内；自己说话的分量多大都不影响，但没到 90%。
4. 第5条奖励里的"高兴"：对改正没用，因为高兴总在答对之后才来；但能帮新学的东西写进皮层（皮层自己会的从 0.23 升到 0.63）。
5. 第20条（大人靠海马体仍学得快）：行。大人学新课从 9.4 次降到 4.2 次，和新生儿一样快；但真正写进皮层的只有三到六成。
6. 第12条（突触里加一个快的部分）：不行。什么都不发生时记得最牢（0.96），但听了别的话照样忘（0.10），改答案也没快多少（7.7 次）。
7. 自我复习：自己声音的分量从 0 加到 1，皮层记住的从 0.01 升到 0.48，但改答案从 8.8 次变成 20 次以上；只有"难过时反过来"能赢过自我复习。
8. 海马体忘得快慢是一个取舍：忘得快（0.95），改答案 3.8 次，但记得少（0.48）；忘得慢（0.99），记得多（0.93），但几乎改不过来（18 次）。
9. 最大的问题不在遗忘规则：这个小测试里，前额叶对"问题"的编码 64% 由最后一个字决定，别的句子一来就把答案冲掉。这就是待定第6条"同一个问题走同一条路、不同问题走不同的路"，要先解决它，"学会了一辈子不忘"才测得出来。
