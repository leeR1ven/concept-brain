(Saved by the coordinator from the agent's final hand-back. In the repo: the 25 MB of per-run JSON are not committed; `tables/item_*.txt` are the analyzer's tables from them. `x3_hippo.py` is imported from `../x345/`; `stream.json` must be regenerated there with `record_stream.py`. The debug scripts were not copied.)

# X6 report: harvest items 17, 18, 19, 21, 22 on the X3 hippocampus

**Labels:** SIMULATED = runs (files listed at the end); PROVED = arithmetic or code; ASSUMED = judgement.

**Setup.**
- I wrote the plan (`PLAN.md`) before any run. Every later change is logged there with its reason; all were made on calibration seed 9, before the result runs.
- Result runs: 5 seeds (items 21 and 21g: 3 seeds), all on the A800 CPU queue in 6 jobs (`x6_cal`, `x6_m1`, `x6_m2`, `x6_d1`, `x6_d2`, `x6_g1`).
- Every number is mean ± SD over seeds. Paired = same seed, so the same world, probes and birth wiring.
- Caveat: there is one recorded world stream. Seeds change the probe texts, the birth wiring and the test randomness, not the world, so the SDs are small.

## Verdict table

| item | verdict | key numbers (SIMULATED) | paired difference |
|---|---|---|---|
| **17** pause bridge | **FAIL** on the planned metric; **PARTIAL** overall | **Planned metric** (re-hear Q, then the hippocampus runs on its own; share where the right A comes out first): no inner voice (A0) 0.02±0.01; inner voice 25 % (A1) 0.14±0.04; inner voice 10 % (A1w) 0.03±0.02. X3 unstepped (no stepped pause) 0.03±0.02. <br>**Natural recall for the inner-voice arms** (Q held again, so the inner voice is on again in the recall pause): A1w **0.64±0.04**, wrong answer 0.02; A1 0.18±0.09; A0 0.02. <br>**Ideal re-hearing** (exact encoding-time state; storage only): A1w 0.80, A1 0.72, A0 0.01. <br>**Wide window, 15 ticks** (A2): 0.07 on the planned metric, 0.87 with ideal re-hearing. It ruins ordinary recall: i100 0.87 → 0.10, r50 0.98 → 0.36. | A1−A0 (planned): +0.12±0.04, 5/5 seeds. <br>A1w−A0 (natural): **+0.62±0.04, 5/5**. <br>Inner-voice side effects: i100 −0.01±0.04, r50 −0.01±0.02. <br>A2−A0: i100 −0.77±0.09 → FAIL. |
| **18** strength ratio instead of a switch | **FAIL** | X3 with the switch: fresh r50 0.99±0.00. <br>No switch, R (chain drive / learned drive) = 9.0 / 5.5 / 2.2 / 0.47 / 0.35: fresh r50 = 0.07 / 0.14 / 0.11 / 0.00 / 0.00. <br>X3's stale sums: 0.00. <br>"Dark" silence (no input) is no better. | vs the switch: −0.86 to −0.99, 0/5 seeds better at every R |
| **19** cued vs random kicks | **PARTIAL** (planned: FAIL, because the chimera criterion failed) | Random vs cued kicks: <br>real sequence 0.34 vs 0.55; <br>chimera 0.46 vs 0.43; <br>any heard pair 0.57 vs 0.91; <br>right continuation (cued only) 0.65; <br>wrong pairings per 1,000 quiet ticks **113±3 vs 50±18**: "thought → unrelated" 68 vs 8, inside the recall 45 vs 41. | cued − random: chimera −0.03±0.15 (target ≤ −0.20 ✗); wrong pairings −63±16, i.e. 44 % of random (target ≤ 50 % ✓) |
| **21** sizing | **Settled**; decay ∝ 1/s **PASS** | **A50** = age (in ticks) at which recall falls to 0.5. s = 1: 26.1 k. <br>**γ fixed, s = 4:** k ∝ s, C fixed (X3's scaling) 30.4 k (**1.17×**); k and C fixed 1.31×; k and C ∝ s 1.37×; k fixed, C ∝ s 1.91×. So the horizon is flat to sublinear, not ∝ s. <br>**γ ∝ 1/s** (400 k life): s = 4 with k and C ∝ s 62.9 k (2.57×); s = 4 with k fixed, C ∝ s **125.5 k (5.12×)**; fresh r50 1.00 in all. | γ ∝ 1/s vs γ fixed, same arm: s2c 1.50×, s2d 1.37×, s4c 2.51×, s4d 1.76× (all ≥ 1.3 ✓) |
| **22** one-shot fact (sensory level) | **FAIL** end to end; storage half PASS | **End to end** (hear "我的X叫" in the test-time scene; name recalled at AUD): fresh 0.09±0.05 (0.35 at 1 k, then 0); adult 0.01±0.03. Same scene: 0.20 / 0.04. <br>**Storage half, ideal re-hearing:** fresh 1.00 at every delay; adult 0.84 (1.00 at 1–2 k, 0.75 at 5 k, 0.60 at 10 k). With less scene weight in the hippocampal input (cE 20), adult 1.00. <br>Stored-code cue (50 % of the stored 叫-moment code): fresh 0.99, adult 0.74. | — |

## What the failures have in common, and what else was found

1. **Re-hearing a line does not bring the hippocampus back to that line's stored code.** (SIMULATED; this is the main finding.)
   - From the exact encoding-time state, replay is 1.00. From any other state, the overlap with the stored code falls to 0.2–0.35 by the 5th–6th char.
   - The causes are the chaotic birth recurrence (ρ = 1) and the drifting threshold tracker.
   - X3's 3-char cues still worked (i100 ≈ 0.9). The 5–7-char question cues of items 17 and 22 do not.
   - This breaks the designer's rule that similar signals give similar recall.
   - Lowering ρ to 0.5 (follow-up, not a verdict run) raises item 17's natural recall from 0.64 to 0.87±0.03 (wrong 0.00). But X3's separation of similar episodes drops from 1.00 to 0.67, and unique recall from 0.98 to 0.81. It did not rescue item 22 end to end (0.14).
2. **X3 as it is cannot link a question to its answer even across one silent moment.**
   - The unstepped reference gets 0.01–0.03 in every recall form, the ideal one included.
   - After a line ends, the eye-only input is below the speech-tuned threshold, so the hippocampal state collapses to 0–2 units. That measurement is SIMULATED at mid size (N_H 10 k, C 1 k).
   - The inner voice is what keeps it alive: 45–81 units in the pause. A weak inner voice (10 %) works better than 25 %, because its pause state is small and set by the input, so it can be replayed.
3. **The scene dominates the hippocampal code.** The expected active inputs per hippocampal unit in a speech moment are eye 3.14, AUD 2.34, voice 1.0 (PROVED from the wiring). So about half of each code is the scene, and in the adult that sharing is the main source of interference.
4. **Item 18, split by side** (follow-up diagnostics):
   - The learned drive must be ≈ 0 while listening: λ = 4 while listening, with X3's recall, already cuts r50 to 0.31.
   - It must be about 20× stronger in recall: listening as in X3, no-switch recall with λ_rec 20, gives r50 0.84 with the scene visible and 0.99 in the dark.
   - Whether sound is present changes the chain drive only about 1–2×, and the eye input stays on in silence. So a single ratio cannot serve both.
   - At high λ the network also locks onto its first learned pattern and stops learning (≈ 800 live links).
5. **Item 21.**
   - X3's "horizon ∝ s" came from the 0.9 threshold falling on a coarse bin. My data show it at s = 1: recall 0.92 in the 10–20 k bin, then 0.23 at 20–50 k.
   - The horizon is set mostly by the decay. In patterns, A50 is about 700 updates (0.026 updates per tick), well below the reviewers' 0.1/f² ≈ 2,400 (PROVED arithmetic).
   - **Recommended:** keep k fixed, scale C ∝ s, and decay ∝ 1/s. The cost is memory ∝ s², because N·C grows ∝ s²: 16× the synapses at s = 4. Run time is 4.2× (822 s vs 194 s per 400 k-tick life).
   - Decay ∝ 1/s was not tried on arms a and b, by the pre-registered rule.

## What I changed in the harness, and rule exceptions
- `x6_hippo.py` imports `x3_hippo.py` unchanged. With the new options at their defaults it is the X3 base. The X3 rule, the two-layer weights and the gain control are kept.
- **Stepped pause, item 17 Q–A probes only:** one update per 3 ticks. X3 updates only when the input changes, which makes any pause free.
- **Item 17 arms:** inner voice = a fixed 10 % or 25 % of Q's AUD units, entered through the copies. Learning window = an option.
- **Item 18:** the no-switch update rule (β, λ, λ_rec). Two-layer sums are kept current at every learning step when λ > 0 (HARVEST 一.6, adopted). X3's stale sums gave new links a 1/√10⁻⁶ boost; a stale-sum arm is reported.
- **Cue start:** cues now start from the hippocampus's own current state. X3's start (one eye-only update without gain control) turns on about 2k units; it is kept as `i100x`, which matches i100 (0.94 vs 0.92).
- **Diagnostics added after calibration** (logged in PLAN): ideal re-hearing, stored-code cues, and the item-17 "natural" recall.
- **ARCH §二.8 held-unit rule:** coded (`held`) but not used, because the brief says to keep the X3 rule. One mid-size try showed no effect.
- **Rule exceptions:**
  - Nothing from §三 was added and there is no new module.
  - Local runs were small checks only, peak 272 MB.
  - Never more than 2 of my jobs at once; the longest job was 72 min.
  - Item 19's "speech onsets driven by recall" could not be measured: there is no motor area in this harness.

## Files (agent's working folder; see the note at the top for what is in the repo)
- `PLAN.md`: plan, plan changes, follow-up list.
- Code:
  - `x6_hippo.py`: the harness.
  - `x6_run.py`: the runner.
  - `x6_specs.py`: all specs.
  - `x6_analyze.py`: tables. Usage: `python3 -I x6_analyze.py {17|18|19|21|22|sep} out [prefix]`.
  - `mkjob.py`: makes the queue jobs.
  - `x3_hippo.py`, `record_stream.py`, `x3_main.py`: copies; `stream.json`: same md5 as X3.
- `out/*.json` (25 MB): results.
  - Pre-registered: `q17_*`, `q18_*`, `q19_*`, `q21_*`, `q21g_*`, `q22_*`.
  - Follow-ups: `d18_*`, `dr0.5_*`, `dr0.25_*`, `dsep_*`, `de20_*`.
- `cal/*.json`: seed-9 calibration. `specs/*.json`: job specs.
- `smoke.py`, `dbg1–6.py`: debug scripts.

## summary_zh
1. **17 海马体跨停顿：部分行。**
   - 没有内心的声音，海马体连不起问题和答案：0.02。X3 原样也只有 0.03，因为一句话说完后的安静时刻，海马活动塌到几乎为零。
   - 弱的内心声音（问题的听觉神经元亮 10%）能在停顿里保住问题：重听问题时答对 0.64，几乎不答错（0.02）。
   - 按事先定的测法（重听后海马体自己往下走）只有 0.03–0.14，没过 0.8。
   - 加宽学习窗口（15 拍）能连上，但把普通回忆从 0.87 毁到 0.10，不行。
2. **18 不用开关、靠强度比例：不行。**
   - 学到的连接一直开着，任何比例下新鲜回忆都 ≤ 0.15，有开关时是 0.99。
   - 原因：听的时候学到的连接必须几乎为零，回忆时又要强约 20 倍。有没有声音，比例只变一两倍，而且安静时眼睛的输入还在。
3. **19 当前想法提示的回忆 vs 随机刺激：部分行。**
   - 每 1000 安静拍错学的配对从 113 降到 50，其中"想法 → 无关内容"从 68 降到 8。
   - 但回忆内部拼出没听过的组合一样多（0.43 对 0.46）。
4. **21 海马体怎么随整体放大：已定。**
   - 降权不变时，记忆期限几乎不随规模变：4 倍大只长 1.2–1.4 倍。X3 说的"正比于规模"是阈值落在粗分箱上的假象。
   - 降权放慢（∝ 1/s）后就长了：4 倍大时长 2.6–5.1 倍。
   - 建议：活跃数不变、入边 ∝ 规模、降权 ∝ 1/规模（内存按规模平方增长）。
5. **22 说一次就记住：端到端不行，存储部分行。**
   - "我的小狗叫豆豆"说一次，1k–10k 拍后问"我的小狗叫"，听觉区说出名字：新海马 0.09，大人 0.01。
   - 但名字确实存下了：如果重听能回到当时的海马状态，新海马 1.00，大人 0.84；降低场景权重后，大人也是 1.00。
6. **共同的卡点：重听同一句话，海马体回不到原来的编码。**
   - 原因是出生随机回路太强（ρ=1），编码强烈依赖听之前的内部状态。这违背了"相似的信号引起相似的回忆"。
   - ρ 降到 0.5，问答跨停顿升到 0.87，但相似经历的区分从 1.00 降到 0.67。这是下一步要解决的。
7. **场景（眼睛）占了海马输入的约一半，所以同一句话换个场景就是另一个编码。**
