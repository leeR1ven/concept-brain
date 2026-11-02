(Saved by the coordinator from the agent's final hand-back. In the repo: X1c/X1d files are imported from `../x1c` and `../x1d`; `res/` holds the tables and the AGG3 aggregates (the 17 MB of per-life JSON are not committed).)

# X1e report: does letting the PFC top layer's internal links (L3→L3) learn carve reproducible, question-specific paths, so that Q→A can form?

Labels: **PROVED** means counted or exact numbers from runs, with files named. **SIMULATED** means one seed or a pilot. **ASSUMED** means my interpretation, not tested.

## 0. Bottom line

**Every arm fails all three pass criteria** (PROVED; 5-seed means; criteria were fixed in `PLAN.md` before the batches).

| criterion (5-seed mean at end of teaching) | target | best arm value | status |
|---|---|---|---|
| question decoding at lag 12, with within-path J(t,t+1) < 0.8 | ≥ 0.30 | 0.064 (R1e_Lq; within 0.798). Chance is 0.031. The ×4 R1e arm reaches 0.061 but is near-stuck (within 0.89). | FAIL |
| Q→A specificity, own − other | ≥ 0.10 | **fair metric (one other question):** −0.03 to +0.013 in every non-oracle arm. Null is 0 ± 0.02. | FAIL |
| seen-pair first-syllable accuracy | > 0.20 | 0.00–0.01 in every arm; full-answer accuracy 0 | FAIL |

Screening covered 469 lives across 3 grids, plus ~30 local pilots. The best single life reached lag-12 decoding of 0.141, and only one life had ≥ 0.1 together with motion kept. **No life approached 0.3.**

What L33 learning actually does:
- L3 runs at 10–25 % activity, so the rule connects 70–100 % of all L3 pairs.
- The result is either **one generic attractor** or a **pause that dies**. It is never a question-specific path.

Main new findings beyond X1d:
1. **Quiet wait fixes the babble, not the path.** With the coordinator's quiet-wait fix, R1e response rate fell from 0.98 to 0.36–0.45 and garble from 3.06 to 1.0–1.3 per exchange. Path and specificity did not change.
2. **Overhear-first gives no Q→A** at 100, 300, 1,000 or 3,000 overheard dialogues.
3. **X1d's specificity metric (and the pass criterion as first written) is biased.** Pure noise gives about +0.07 to +0.10 on it. X1d's oracle "+0.11" is mostly this bias. The unbiased value is +0.03, or +0.06 with 4× sparser oracle codes.
4. **Hubs are not in-strength hubs.** Equalising in-strength exactly still leaves hub share at 0.91–0.99.
5. **×4 scale helps a little in R1e only:** lag-12 decoding 0.061 vs 0.036 for its reference, but the thought is near-stuck.

## 1. What I built (all in `x1e/`; X1d and X1c code copied, not edited)

**`core5.py`** is X1d's `core4.Brain` plus options. PROVED that the defaults reproduce core4 exactly: L3 states, mouth, L31 and P33 identical over 3,000 ticks for both bases (`eqtest.py`).
- `rule33='onset'`: L3→L3 learns by the same onset/held rule as top→L1.
  - B rises while A is on (new or held) or just fell (≤ 2 ticks): A→B grows.
  - A and B co-rise: both directions grow.
  - A rises while B was already on: A→B weakens.
  - Nothing happens at a fall.
  - Concave growth with a 1 % floor; the learned part can cancel the birth part, but the total stays ≥ 0.
  - The one global decay (`dec_33`) acts on learned links only; birth links B33 are exempt.
- `nrm33='arch'`: ARCH two-layer weights, P33 = g33·W/(out^a·in^(1−a)), with no total rescale (scale-free).
- `a31`/`a33`: the exponent split, used for the hub test.
- Diagnostic-only options: `pm_ff`, `homeo3`, `held33`.

**`run3.py`** is X1d's run2 plus:
- path metrics at every teach mark ≥ 300;
- hub and in-strength statistics;
- **`OverhearWorld`** and the overhear-first stage: adult C1↔C2 dialogues, never the held-out pairs, mouth blocked by `theta_m`;
- an **unbiased specificity control** (`other1_max`, `lag_diff_max`);
- the option to skip the correction and untutored phases.

**Tools:**
- `regimes5.py`: arms; adds the oracle arms;
- `pilot.py`: pilots;
- `motor_screen.py`: motor-knob screen;
- `spec_null.py`: permutation null plus a positive control for specificity;
- `agg3.py`, `qsum.py`: aggregation;
- `mkbatch.py`, `submit.sh`, `sequencer.sh`, `runlist.sh`: job tools.

## 2. Screening: L33 learning alone (PROVED, seed 0 or 0–1, teach 600; `res/grid1..3_table.txt`)

| screen | lives | what varied | lag-12 decoding (learned arms) | fates |
|---|---|---|---|---|
| grid1, default motor | 281 | g33 1–8 / 4–16, eta33 .005/.02, decay .999/.99/.95, threshold, a33 .5/0, patch-mean on FF, homeostasis | max 0.086, mean 0.028; none ≥ 0.1 | 33 % stuck (within ≥ 0.8), 48 % dead (P(empty) ≥ 0.3), 19 % alive and moving but not question-specific |
| grid2, weak lateral, fast decay | 84 | g33 0.5–2, eta .01–.05, decay .9–.98, k3 25, teach 1,500 | max 0.141 (one life), mean 0.045; teach 1,500 gives 0.028 | 56 % stuck |
| grid3, quiet wait | 104 | as above, quiet-wait knobs | max 0.102, mean 0.027; L33 frozen references equal | 56 % dead |

Typical failure 1, **generic attractor** (pilot, RNpm g33 8, eta 0.02, decay 0.999):
- same 0.73 vs cross 0.73 at lag 12;
- long-silence turnover 0.024 per tick, stuck20 0.95;
- 89 % of pairs learned, in-weight median 208;
- in R1e the attractor overrides even the question: same 0.81 vs cross 0.77 at lag 0.

Typical failure 2, **dying** (g33 ≤ 2 or lower eta): P(L3 empty) at lag 12 = 0.6–0.98.

Diagnostic ablation (SIMULATED): turning off held-unit learning does not prevent the generic attractor (86 % of pairs learned, turnover 0.003). The held clause is not the cause.

## 3. 5-seed batches: full X1d protocol (birth 200, teach 600, correction 300, untutored 1,000; PROVED; `res/M1_table.md`, `res/M2_table.md`, `res/M3_table.md`)

Arm settings:
- **RNpm L33 arm:** g33 2, eta33 0.02, decay 0.95, f3 0.2667.
- **R1e L33 arm:** g33 1, decay 0.95.
- **"q" (quiet wait):** g_ret_mot 0.15, c_mb 2.0, theta_m 0.26.
- **"OH":** 3,000 overheard dialogues before teaching.

### RNpm (main base)

| metric | L (default motor) | ref_q | Lq | ref_q_OH | Lq_OH | Lq_OH long gap | Lq_OH + shaping | Lq_OH, teach 1,500 |
|---|---|---|---|---|---|---|---|---|
| decoding lag 0 / 6 / 12 | .15/.04/.03 | .14/.05/.02 | .15/.05/.03 | .14/.05/.02 | .12/.02/.03 | .14/.04/.02 | .14/.04/.03 | .14/.04/.03 |
| same − cross at lag 12 | .023 | .021 | .020 | .013 | .018 | .007 | .028 | .012 |
| within J(t,t+1) / damage 10 % at lag 12 | .26/.36 | .32/.39 | .31/.27 | .25/.25 | .35/.44 | .27/.27 | .32/.35 | .40/.44 |
| P(L3 empty) at lag 12 | .53 | .44 | .48 | .44 | .38 | .58 | .40 | .31 |
| specificity: X1d (biased) / per-lag / fair | .020/.011/– | .042/.018/– | .030/.014/– | .029/.017/.013 | .033/.019/−.003 | .041/.019/– | .014/.013/– | .034/.014/– |
| hub share | .61 | .67 | .65 | .75 | .78 | .66 | .75 | .59 |
| seen accuracy / first syllable | 0/.01 | 0/0 | 0/.01 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 |
| response rate / garble per exchange | .75/2.26 | .57/1.75 | .51/1.64 | .49/1.55 | .48/1.54 | .51/1.62 | .49/1.55 | .55/1.72 |
| talk-over / failure to yield per exchange | .019/0 | .024/0 | .024/0 | .026/0 | .019/0 | .011/0 | .009/0 | .019/0 |
| FORCED at lag 6: first syllable / said | .12/.78 | .04/.66 | .06/.70 | .03/.55 | .05/.60 | .03/.65 | .03/.71 | .02/.64 |
| L33 learned pair fraction | .97 | 0 | .99 | 0 | .97 | .99 | .97 | .98 |

### R1e (secondary base) and oracle diagnostics

| metric | R1e_L | R1e_Lq | ref_q_OH | Lq_OH | ORACLE q (k 40) | ORACLE q (k 10) | ORACLE RNpm q |
|---|---|---|---|---|---|---|---|
| decoding lag 0 / 6 / 12 | .11/.05/.04 | .13/.07/**.064** | .06/.04/.04 | .13/.04/.04 | (oracle off in measure) | – | – |
| within J(t,t+1) | .79 | .80 | .99 | .82 | .99 | .97 | .29 |
| specificity: biased / per-lag / **fair** | .014/.010/.008 | .016/.010/.011 | .005/.006/.001 | .012/.015/−.011 | .122/.042/**.029** | .154/.083/**.061** | .008/.007/−.007 |
| hub share; answer is the top-predicted syllable (max over lags) | .67 | .72 | .78 | .58 | .15; .11 | .25; .26 | 1.00; .01 |
| response rate / garble per exchange | .98/3.06 | .45/1.26 | .33/.91 | .36/.97 | .65/1.42 | .39/1.07 | .25/.74 |
| FORCED at lag 6: first syllable / said | .00/.82 | .36/.85 | .58/.83 | .23/.81 | .11/.76 | .50/.91 | .33/.94 |
| seen accuracy / first syllable; correction new/old; untutored accuracy | all 0 (one 0.01) | 0 | 0 | 0 | 0 | 0 | 0 |

### ×4 scale

Settings and deviations:
- n2 = n3 = 1,600.
- Overhear 1,000, teach 600, no correction or untutored phase (a compute deviation).
- RNpm needs g31 0.5 (at 1.0 it saturates: mean 1,031 of 1,600 active, P(sat) 0.17).
- theta_m 0.35, because at 0.26 the ×4 child responds 0.94 of the time.

| metric | RNpm ref | RNpm L33 | R1e ref | R1e L33 |
|---|---|---|---|---|
| decoding lag 0 / 6 / 12 | .20/.05/.03 | .20/.05/.02 | .09/.05/.04 | .17/**.10**/**.061** |
| within J / P(empty) at lag 12 | .31/.52 | .38/.45 | .94/.005 | .89/.03 |
| fair specificity / hub share | −.004/.85 | −.028/.98 | .003/.83 | −.002/.74 |
| response rate / garble per exchange | .35/1.00 | .33/.96 | .18/.45 | .18/.46 |
| L33 learned pair fraction | – | .97 | – | .55 |

During overhearing, R1e ×4 with L33 reached lag-12 decoding of 0.083 ± 0.04 at 300 and at 1,000 dialogues. The direction is the one the designer expects (bigger is better), but it is small and the thought is near-stuck.

### Overhear-first curve (5 seeds, measured before any teaching; PROVED)

| arm | N = 100 / 300 / 1,000 / 3,000: decoding at lag 12 | fair specificity | hub share | first syllable |
|---|---|---|---|---|
| RNpm Lq_OH | .034 / .028 / .041 / .037 | −.046 / −.008 / .001 / −.010 | .68 / .64 / .46 / .44 | 0 / 0 / 0 / .01 |
| RNpm ref_q_OH | .009 / .044 / .028 / .028 | −.019 / −.039 / −.003 / −.004 | .69 / .62 / .42 / .58 | 0 / 0 / 0 / .01 |
| R1e Lq_OH | .034 / .053 / .040 / .033 | .001 / −.009 / .007 / .004 | .70 / .69 / .57 / .76 | 0 at every N |
| R1e ref_q_OH | .033 / .048 / .050 / .033 | ≈ 0 at every N | .75 / .72 / .65 / .80 | 0 at every N |
| RNpm long gap (10–14 ticks) | .030 / .041 / .039 / .036 | – | .72 / .72 / .47 / .51 | 0 at every N |

Overhearing lowers the hub share (0.8 → 0.44–0.6) but creates no question-specific prediction. L33 on vs off makes no difference.

## 4. The specificity metric (PROVED)

- X1d's metric is mean over questions of the max over lags of `own`, minus the same for `other`. Here `own` is a single question, while `other` is an average over ~20 questions, so its max is less inflated by noise.
- On uniform random scores it gives **+0.105**.
- On real brains, with the answer→question mapping permuted 200 times, the null is:
  - teach 300: +0.016 ± 0.014;
  - after 300 overheard dialogues: **+0.068 ± 0.021**;
  - after 3,000 overheard dialogues: +0.081 ± 0.025.
- The observed values were +0.083 and +0.042, i.e. at or below the null.
- **Fair metric** (the same statistic on one other question): null 0 ± 0.02. It reads −0.018, −0.060 and −0.028 (long gap) on those brains.
- **Positive control:** forcing the answer's inner hearing gives fair **+0.81**, so the metric does detect real specificity.

The pass criterion should be judged on the fair metric. Files: `spec_null.py`, `specnull_*.log`.

## 5. Hub check (item 4)

- The ARCH √(out·in) weights do act (PROVED).
  - P33 column sums stay ≈ g33 however much L33 grows.
  - The √ form compresses the raw in-strength spread; with a31 = 0, every L1 unit's propagation in-strength is exactly 1.00.
- **The hub syllable is not an in-strength hub** (PROVED, all batches): its in-strength is 1.0–1.45× the median syllable for connection strength and 1.0–1.17× for propagation.
- With full in-normalisation (a31 = 0, SIMULATED, 1 seed):
  - hub share is still 0.91 (R1e) and 0.99 (RNpm);
  - RNpm's pause dies;
  - R1e's path collapses to chance.
- Interpretation (ASSUMED): the hub is whatever most often follows the generic pause state, mostly the first syllable of the next question (苹, 给) or a frame syllable. It is a symptom of the missing path, not of the normalisation.

## 6. Quiet wait (coordinator fix; PROVED unless marked)

- Knobs were chosen by a probe-time screen on a trained R1e (`motor_R1e*.log`, SIMULATED):
  - default knobs: probe silent 0.00, forced first syllable 0;
  - (0.15, 2.0, 0.26): silent 0.91, forced first syllable 0.68.
- In training (5 seeds):
  - R1e response rate 0.98 → 0.36–0.45, garble 3.06 → 0.97–1.26 per exchange;
  - RNpm 0.75 → 0.48–0.51, and 2.26 → 1.54–1.64.
  - **Not robust across seeds:** response-rate SD is about 0.3–0.4, and one R1e oracle seed stayed at 0.98.
- The remaining speech is 95 % driven by the **inner hearing** (auditory return → speech link). Before the fix, 60 % came from PFC motor intent.
- In RNpm, lowering g_ret_mot alone made the child *more* talkative at probe time (silent 0.50 → 0.00), because it also weakens the PFC→stop path.
- Talk-over stays 0.01–0.03 per exchange and failure to yield is 0 everywhere.
- Shaping with quiet wait and overhearing: pleased 0.04–0.06 per exchange, answer-syllable hits 0.02–0.05, first syllable 0.

## 7. Where the chain breaks (numbers PROVED, interpretation ASSUMED)

1. **Question end: weak.** Question decoding at lag 0 is 0.06–0.20 (2–6× chance). The state is mostly the frame (last syllable): frame decoding at lag 0 is about 0.8.
2. **Pause: lost by lag 6 in every arm** (decoding at lag 6: 0.02–0.10).
   - Strong lateral: a generic attractor (stuck, same ≈ cross).
   - Weak lateral: activity dies, especially in the quiet child, whose own babble had been keeping RNpm alive.
   - The middle band moves but forgets.
3. **Prediction: weak even given a perfect path.** With the oracle and quiet wait, fair specificity is only 0.029. With 4× sparser oracle codes (10 of 400 units) it rises to 0.061 and the answer becomes the top prediction 0.26 of the time.
   - So the Hebbian top→L1 readout suffers **crosstalk**: with 512 codes of 40 units in 400, each unit sits in ~51 codes.
   - RNpm's top→L1 form (scale-free, absolute floor) cannot read out even a perfect path (hub 1.00).
4. **Release: works when forced.** The forced answer is said 0.55–0.94 of the time. With quiet wait, the correct first syllable comes out 0.23–0.58 of the time in R1e, versus 0 before. Link 3 is no longer the main block.

**Root cause (ASSUMED, supported by the density numbers and the sparse-oracle gain):** associative capacity.
- L3 at 10–25 % activity in 400 units saturates the learned matrix (70–100 % of pairs), so learned links encode the average transition, not 32 question-specific sequences.
- Willshaw estimate: about 0.69·N²/k² ≈ 70 storable transitions for N = 400, k = 40. Needed: ≥ 32 questions × 12 lags = 384.
- At N = 1,600 with k = 40 (2.5 %), capacity is about 1,100.
- ARCH says the thinking range scales proportionally with size, which keeps capacity per unit constant. **Bigger helps capacity only if the active fraction falls.** That is a designer decision: test ×4/×16 with a thinking range of an absolute ~20–40 units.

## 8. Deviations and limits

- **Minor parameters re-chosen per arm:**
  - RNpm f3 was re-set to keep the threshold level; 4.8 instead of 5.4, because 5.4 died in the quiet child.
  - dec_33 = 0.95 per 100 ticks is faster than dec_31 = 0.998. That arguably runs against ARCH's "deeper decays slower"; 0.999 was not better in the screens.
- ×4: overhear 1,000, no correction or untutored phase; g31 0.5 and theta_m 0.35 re-tuned.
- The fair specificity metric was added mid-way. M1 lacks it, but the two main overhear arms were re-run (deterministic, identical numbers) with it, as the `_v2` arms.
- Server numpy 2.1 and local numpy 1.26 give different trajectories. A local "best" seed (gap 0.12) did not replicate on the server.
- Held-out accuracy needs the concept area and is 0 everywhere, as expected.
- Compute:
  - A800 jobs `x1e_grid1..3`, `x1e_M1..M3`, at most 2 at a time.
  - Local runs were one process under 150 MB. Twice, two of my processes overlapped for about 10–60 s.
  - The repo and other agents' folders were not touched.

## 9. Files

In the agent's working folder (the repo copy holds the code, plan, job lists and result tables):
- **Plan:** `PLAN.md` (criteria fixed before the batches, plus the addendum).
- **Core code:** `core5.py`, `run3.py`, `regimes5.py`.
- **Tools:** `pilot.py`, `motor_screen.py`, `spec_null.py`, `agg3.py`, `qsum.py`, `eqtest.py`, `mkbatch.py`, `submit.sh`, `sequencer.sh`, `runlist.sh`, `jobs/*.txt`, `jobs/job.sh`.
- **Copies of X1c/X1d code:** `core3.py`, `core4.py`, `world.py`, `run.py`, `phase1.py`, `run2.py`, `regimes.py`.
- **Results:**
  - `res/grid1..3/*.json` and `res/grid*_table.txt` (469 screening lives);
  - `res/M1`, `res/M2`, `res/M3` (`*.json`, `AGG3_*.json`) and `res/M*_table.md`;
  - `out3/` (local pilots);
  - logs: `pilot_p1.log`, `loc*.log`, `motor_*.log`, `specnull_*.log`.
- **Raw job folders:** `../gpu/queue_a800/x1e_*/result/`.

## summary_zh

1. 这次让前额叶最后一层内部的连接也按同一条规则学习（"有关的互相激活"），用了 X1d 的两个底座，正式跑了 22 组、每组 5 次一生，前面还筛了约 470 次。
2. 结果：三条及格线都没过。停顿 12 拍后认出是哪个问题，最好只有 0.06（及格 0.3，乱猜 0.03）；答案预测不针对问题；第一个字答对率 0–0.01。
3. 原因：最后一层同时亮的神经元太多（10–25%），学习后 70–100% 的神经元两两都连上了，于是要么所有问题掉进同一个想法、卡住不动，要么停顿时熄灭。
4. 按您说的让孩子等待时安静（只调运动区原有的三个参数）：抢着乱说从 98% 降到约 40%，乱码少了一半多，但路线和答案没变。
5. 按新方法先旁听大人问答 100 到 3000 次、孩子不出声：旁听越多，"万能字"越少，但答案仍不针对问题，答对率仍是 0。
6. 发现以前的"针对性"算法有偏差：纯随机也能算出约 0.1。换成公平算法后，X1d 手接完美路线的结果只有 0.03；把那条路线的神经元用得更稀疏（每次只亮 10 个），提高到 0.06。
7. "万能字"不是因为它的入边特别强（只比平均高 1.0–1.45 倍），把入边完全拉平后它仍然独占 0.9 以上。它是停顿状态不分问题造成的。
8. 规模放大 4 倍：一组（全局门槛）有一点改善（0.04 升到 0.06），但想法几乎卡住。另一组（不加抑制）没有改善。
9. 我的判断：400 个神经元、每次亮 10–25%，能记住的"先后顺序"大约只有 70 条，而 32 个问题 × 12 拍就需要约 380 条。规模变大只有在同时亮的比例下降时才有用。
10. 需要您定：放大时"思考范围"是按比例增加，还是保持在约 20–40 个神经元（更稀疏）？后者是下一步最值得测的。
