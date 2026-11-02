(In the repo: `x6_hippo.py` is imported from `../x6/` and `x3_hippo.py` from `../x345/`; `stream.json` must be regenerated there with `record_stream.py`. The 32 MB of per-life JSON are not committed; `tables/` holds the analyzer's tables. The debug scripts (`dbg1–3.py`, `smoke.py`, `test_an.py`) were not copied.)

# X6b report: the hippocampus without birth recurrence (fast weights only), and an innate listening signal (ARCH §二.6, §四.19)

Labels: **SIMULATED** means runs: 5 seeds (0–4), mean ± SD, files in `tables/`. **PROVED** means code, arithmetic or an exact match. **ASSUMED** means my judgement. Paired means the same seed, so the same world stream, probes and birth wiring. As in X6 there is one recorded world stream; seeds change the probes, the wiring and the test randomness, so the SDs are small.

## 结果（简明）

1. **设计者说"海马体记得快，靠的是权重变得快，和内部随机连接无关"：基本成立。** 去掉出生随机回路（ρ = 0），只靠快速变化的权重：
   - 同一场景里重听一句话，海马体回到当初存下的编码：重叠 0.85（X3 只有 0.33）。把门槛改成"每次更新都定成固定亮数"后是 1.00。
   - 普通句子听一次就记住：重听前三个字，能接出后面的字 0.84–0.90（X3 0.88）。
   - 问题和答案隔着停顿也能连上：重听问题后说对答案 0.94–1.00，几乎不说错（X3 0.64）。这一条及格。
2. **做不到：区分"开头一样"的两段经历。**
   - 同场景、同一个人说：0.45–0.56，等于掷硬币（X3 1.00）。
   - 换一个人说：0.77–0.84。
   - 换场景：0.47–0.60（X3 0.95–0.97）。
   - 及格线 0.9，不及格。原因由结构决定：没有回路时，一个字的编码只取决于此刻听到的字、说话人和场景，带不上"前面听过什么"。
3. **"说一次就记住"的事实**（"我的小狗叫豆豆"说一次，1k–10k 拍后听"我的小狗叫"）：比 X3 好，但没过 0.6。
   - ρ = 0 时 0.25–0.29；门槛每次都定成固定亮数时 0.45–0.54。
   - 说错时多半说成另一只宠物的名字（0.38–0.46）。四句话都是"我的X叫…"，"叫"字的编码在四句里一样，所以分不开。
   - 真实提示已经和理想提示一样好（0.45 对 0.45）。差在存储时没有把四句分开。
4. **普通回忆略降**：用半个编码当提示（r50）0.77–0.88，X3 0.99；记忆期限 16–19k 拍，X3 24.5k。"普通回忆不能下降"这条不及格。
5. **换场景时编码不同**（重叠 0.32）。设计者说这正常，只作记录。
6. **"正在听"信号**（只看输入层有没有人说话，正在听时把回路压低到 1%–50%）：
   - 有出生回路时，它确实让重听回到原编码（0.78–0.98）。但记忆坏了：r50 ≤ 0.50，记忆期限只有 133–4,280 拍，问答跨停顿 ≤ 0.40。不及格。
   - 没有回路时（ZL 对 Z10a1，两者只差这个信号）：大人记事实 +0.12，但问答跨停顿从 1.00 掉到 0.35，r50 −0.15，期限少约 4,000 拍。不采用。
7. **按 PLAN 的判定**：去掉回路"部分通过"（重听回到原编码，问答跨停顿）；"正在听"信号"不通过"。
8. **下一步要解决**：怎样让编码带上"前面听过什么"，同时不破坏"同一输入同一编码"。

## 0. Verdict table (criteria fixed in `PLAN.md` before any full-size run)

| test | pass mark | X3 (reference) | ρ = 0 arms | listening-signal arms (ρ 1 or 0.5) | passed by |
|---|---|---|---|---|---|
| 1. fresh r50 without a store/recall switch (age < 10 k) | ≥ 0.9 | 0.99 (with its switch) | ZL (no switch) 0.72 ± 0.05; Z1 / Z4 / Z10 (with X3's switch) 0.77 / 0.85 / 0.88 | best L001 0.50 ± 0.03 | none |
| 2. re-hearing returns to the stored code, same scene: mean over chars 1–6 / char 6 / other-line control | ≥ 0.7 / ≥ 0.7 / ≤ 0.3 | 0.33 / 0.23 / 0.04 | Z1–Z10 0.85 / 0.81 / 0.08; ZL 0.99 / 0.99 / 0.07 | L001 0.98 / 0.99 / 0.07; L1 fails (0.45) | Z1, Z4, Z10, ZL; L001, L03, L3, L5, L1r, L3r, L5r |
| 3. one-shot fact, own-scene cue (R1s), fresh / adult | ≥ 0.6 both | 0.20 / 0.04 | best ZL 0.38 / 0.66; Z10 0.25 / 0.29 | best L001 0.17 / 0.03 | none |
| 4. Q → A across the pause, 10 % inner voice: right / wrong | ≥ 0.8 / ≤ 0.05 | 0.64 / 0.02 | Z1 0.94 / 0.00; Z4 and Z10 1.00 / 0.00; ZL 0.35 / 0.02 | best L001 0.40 / 0.07 | Z1, Z4, Z10 |
| 5. separation, age < 20 k: other voice / other scene | ≥ 0.9 both | 0.97 / 0.95 | best Z4 0.84 / 0.60 | best L001 0.53 / 0.25 | X3 only |
| 6. ordinary recall must not fall: i100 and r50 ≥ X3 − 0.05 | – | i100 0.88, r50 0.99 | Z10: i100 0.90 (ok), r50 0.88 (fails by 0.06) | best L001 0.61 / 0.50 | none |

**Verdict per proposal** (PLAN rule: PASS = some arm passes tests 1–6; PARTIAL = it passes test 2 and at least one of tests 3 or 4; FAIL otherwise):
- **ρ = 0 with fast weights (Z1, Z4, Z10, ZL): PARTIAL.** Z1, Z4 and Z10 pass tests 2 and 4.
- **Listening signal (L001–L5r): FAIL.** 7 of the 8 arms pass test 2; none passes test 3 or 4.
- The follow-up Z10a1 (not pre-registered, not used for the verdicts) passes tests 2 and 4.

## 1. What was built

- **Harness.** `x6b_hippo.py` subclasses X6's harness; `x6_hippo.py` and `x3_hippo.py` are unchanged. With `ls = None` it is X6's harness.
  - Regression (PROVED, `regress.py`, `tables/regress.txt`): the X3 arm reproduces X6's own runs exactly. Every test row and every live-link count is identical in the uni, qa, fresh and adult lives (5 seeds) and in the sep life (the 3 seeds X6 ran).
  - Full size as in X6: N_H 10,000, k 64, C 2,000, the X3 rule, two-layer weights, decay γ 0.99 per 200 ticks.
- **ρ = 0 arms (Z1, Z4, Z10).**
  - No birth recurrence. X3's store/recall switch is kept: learned links are off while listening, and recall is X3's.
  - η × 1 / × 4 / × 10 on all hippocampal learning rates.
  - For × 4 / × 10 the decay is 0.98740 / 0.98569 per 200 ticks, so that a one-exposure link falls below w_min at the same age.
- **Listening signal (LS).**
  - 64 binary units, birth-wired, never learning. Each reads 300 random units of the auditory input layer (the heard char + the voice), not the inner voice.
  - L = the fraction of units on.
  - The hippocampus's recurrent drive (birth recurrence and learned links) is multiplied by s = 1 − (1 − f)·L.
  - One update for listening, silence and recall, with no switch: d = birth(input) + s·[ρ·(brec − mean) + λ·learned].
  - Arms:
    - L001, L03, L1, L3, L5: ρ 1, f 0.01–0.5;
    - L1r, L3r, L5r: ρ 0.5;
    - ZL: ρ 0 + LS.
  - Calibrated on seed 9 only, by the pre-registered rule (`tables/cal.txt`, `cal_choice.json`):

    | arm | λ_full | α | other |
    |---|---|---|---|
    | L001 | 24 | 1 | |
    | L03 | 24 | 1 | |
    | L1 | 12 | 0.3 | |
    | L3 | 6 | 1 | |
    | L5 | 3 | 1 | |
    | L1r | 12 | 1 | |
    | L3r | 3 | 0.3 | |
    | L5r | 3 | 0.3 | |
    | ZL | 12 | 1 | f 0.01, η × 10 |
- **References.** X3 (ρ 1 with its switch) and NS (X6's no-switch arm, λ 2.5).
- **Follow-up Z10a1** (job `x6b_d1`; not pre-registered, not used for verdicts).
  - It is Z10 with α = 1 (threshold set every update, X3's k-WTA limit) and **no listening signal**.
  - ZL differs from Z10 in two things: the signal and α. Z10a1 separates them:
    - ZL vs Z10a1 isolates the signal (`tables/pair_ZL_vs_Z10a1.txt`);
    - Z10a1 vs Z10 isolates α (`tables/pair_Z10a1_vs_Z10.txt`).
- **Lives** (as in X6):
  - uni: 100 k ticks, tests every 5 k;
  - qa: u + qa lines, 10 % inner voice;
  - sep: X3's mix;
  - fresh / adult: 4 one-shot facts at 3 k / 300 k ticks, tested 1–10 k later.
  - In all, 15 arms × 5 lives × 5 seeds = 375 lives.
- **Changes after the plan** (all logged in `PLAN.md`, all before any result run):
  - L001 (f 0.01) added;
  - a decay-formula bug fixed, and the 24 affected calibration lives rerun;
  - calibration outcome recorded;
  - Z10a1 added as a follow-up.

## 2. What the no-recurrence hippocampus can and cannot do (SIMULATED, 5 seeds; `tables/res.txt`)

### 2.1 Same input → same code: yes, in the same scene

| arm | overlap, chars 1–6 | char 1 | char 6 | other-line control | other scene (reported only) | fact line (7 chars) | Q line |
|---|---|---|---|---|---|---|---|
| X3 | 0.33 | 0.48 | 0.23 | 0.04 | 0.13 | 0.37 | 0.33 |
| Z1 / Z4 / Z10 | 0.85 | 0.93 | 0.81 | 0.08 | 0.32 | 0.89 | 0.88 |
| Z10a1 (α 1) | 1.00 | 1.00 | 1.00 | 0.07 | 0.32 | 1.00 | 1.00 |
| ZL | 0.99 | 0.97 | 0.99 | 0.07 | 0.33 | 0.97 | 0.99 |

- PROVED by construction: with ρ = 0 and the learned links off while listening, a code depends only on the current input (char + voice + scene) and the threshold state.
- The remaining 0.15 in Z1–Z10 comes from the slowly tracking threshold (α 0.3). Setting the threshold every update gives 1.00 (Z10a1 − Z10: +0.15 ± 0.00, 5/5 seeds).
- In another scene the code differs (overlap 0.32). The designer calls this normal, so it is reported, not judged.

### 2.2 One-shot storage and recall from a real cue: yes for ordinary lines and for Q → A; partly for facts

| arm | i100 (re-hear the first 3 chars) | r50 (half the stored code) | Q → A right / wrong | horizon A50 (ticks) |
|---|---|---|---|---|
| X3 | 0.88 | 0.99 | 0.64 / 0.02 | 24,523 |
| Z1 | 0.84 | 0.77 | 0.94 / 0.00 | 16,329 |
| Z4 | 0.89 | 0.85 | 1.00 / 0.00 | 19,021 |
| Z10 | 0.90 | 0.88 | 1.00 / 0.00 | 18,697 |
| Z10a1 | 0.90 | 0.88 | 1.00 / 0.00 | 16,030 |

- Paired vs X3:
  - i100: Z1 −0.04 (0/5 seeds better), Z4 +0.01 (3/5), Z10 +0.02 (4/5);
  - r50: −0.12 to −0.22 (0/5 for every Z arm).
- Faster weights help: from Z1 to Z10, r50 rises 0.77 → 0.88, the horizon 16.3 k → 18.7 k, and adult facts 0.01 → 0.29.

**One-shot facts** (`tables/facts.txt`; 80 tests per life):

| arm | R1s fresh / adult | ideal re-hearing I1, fresh / adult | wrong answer = another fact's name (fresh) | R1s fresh at 1 / 2 / 5 / 10 k |
|---|---|---|---|---|
| X3 | 0.20 / 0.04 | 1.00 / 0.84 | 0.06 | 0.30 / 0.20 / 0.05 / 0.25 |
| Z1 | 0.14 / 0.01 | 0.17 / 0.03 | 0.15 | 0.45 / 0.10 / 0.00 / 0.00 |
| Z4 | 0.26 / 0.16 | 0.35 / 0.33 | 0.38 | 0.50 / 0.20 / 0.15 / 0.20 |
| Z10 | 0.25 / 0.29 | 0.41 / 0.50 | 0.41 | 0.50 / 0.15 / 0.15 / 0.20 |
| Z10a1 | 0.45 / 0.54 | 0.45 / 0.54 | 0.46 | 0.50 / 0.45 / 0.40 / 0.45 |
| ZL | 0.38 / 0.66 | 0.39 / 0.66 | 0.28 | 0.40 / 0.40 / 0.45 / 0.25 |

- In X3 the ideal cue works (1.00) but the real cue does not (0.20). This is X6's problem.
- Without recurrence the real cue is as good as the ideal one (identical in Z10a1), so the cue problem is solved. What is missing is storage that tells the four facts apart.
- All four facts have the form "我的X叫N" and are said by the same voice within about 450 ticks.
  - With ρ = 0 the code of "叫" depends only on the char, the voice and the scene (PROVED by construction). So whenever the facts share a scene, "叫" has one code, and its learned link points to every name.
  - The most common wrong answer is another fact's name: 0.38–0.46 in fresh Z4, Z10 and Z10a1.
  - In Z10a1 the rate is flat over the delays (0.40–0.50), which suggests confusion, not forgetting (ASSUMED).
- In the adult, Z1–Z10 mostly give some unrelated char (0.50–0.99). With α 1 this drops to 0, and adult R1s rises to 0.54–0.66.

### 2.3 Separation of similar episodes: no

The twins share their first 3 chars. Cue: 50 % of A's stored char-3 code. Right: A's 4th char comes out. Age < 20 k.

| arm | same scene + voice (pS, reported only) | other voice (pV) | other scene (pD) | pS gives the twin's char |
|---|---|---|---|---|
| X3 | 1.00 | 0.97 | 0.95 | 0.00 |
| Z1 | 0.45 | 0.77 | 0.47 | 0.29 |
| Z4 | 0.56 | 0.84 | 0.60 | 0.38 |
| Z10 | 0.50 | 0.83 | 0.58 | 0.45 |
| Z10a1 | 0.49 | 0.81 | 0.48 | 0.39 |
| ZL | 0.33 | 0.66 | 0.42 | 0.39 |

- In the same context the result is a coin flip between the twins. This is expected: their char-3 codes come from the same input.
- Context separates only partly:
  - other voice 0.77–0.84, paired vs X3 −0.13 to −0.20;
  - other scene 0.47–0.60, paired vs X3 −0.35 to −0.48;
  - 0/5 seeds better in every arm.
- ASSUMED: X3's separation came from the birth recurrence carrying the start of the line into the code. Without it, only the context inputs of that moment differ.

## 3. The listening signal (SIMULATED, 5 seeds)

| arm | f | λ_full / α | re-hear overlap (char 6) | r50 | i100 | Q → A right | separation other voice / other scene | A50 (ticks) |
|---|---|---|---|---|---|---|---|---|
| NS (no signal, λ 2.5) | – | 2.5 / 0.3 | 0.47 (0.52) | 0.14 | 0.11 | 0.00 | 0.36 / 0.28 | 317 |
| L001 | 0.01 | 24 / 1 | 0.98 (0.99) | 0.50 | 0.61 | 0.40 | 0.53 / 0.25 | 4,280 |
| L03 | 0.03 | 24 / 1 | 0.96 (0.98) | 0.44 | 0.55 | 0.31 | 0.44 / 0.30 | 2,566 |
| L1 | 0.1 | 12 / 0.3 | 0.45 (0.71) | 0.22 | 0.09 | 0.00 | 0.31 / 0.22 | 451 |
| L3 | 0.3 | 6 / 1 | 0.83 (0.89) | 0.15 | 0.22 | 0.09 | 0.11 / 0.15 | 333 |
| L5 | 0.5 | 3 / 1 | 0.82 (0.93) | 0.05 | 0.16 | 0.13 | 0.16 / 0.05 | 133 |
| L1r (ρ 0.5) | 0.1 | 12 / 1 | 0.93 (0.96) | 0.30 | 0.40 | 0.16 | 0.21 / 0.30 | 920 |
| L3r (ρ 0.5) | 0.3 | 3 / 0.3 | 0.82 (0.88) | 0.05 | 0.14 | 0.01 | 0.20 / 0.07 | 217 |
| L5r (ρ 0.5) | 0.5 | 3 / 0.3 | 0.78 (0.86) | 0.05 | 0.13 | 0.00 | 0.17 / 0.02 | 133 |

- **With birth recurrence, the signal does make re-hearing reproducible:** 7 of 8 arms pass test 2.
- **But storage breaks:**
  - r50 ≤ 0.50, i100 ≤ 0.61;
  - horizon ≤ 4,280 ticks (X3 24,523);
  - Q → A ≤ 0.40, separation ≤ 0.53.
  - Diagnostic: the same learned links recalled with X3's recall dynamics give r50 0.18–0.57 (NS 0.89). So the links stored while listening are already poor.
  - ASSUMED: this repeats X6 item 18. The learned links must be about 0 while listening, and even f = 0.01 (λ ≈ 0.24 while listening) is too much.
- **At ρ = 0** (ZL vs Z10a1, paired, `tables/pair_ZL_vs_Z10a1.txt`):
  - adult facts: +0.12 ± 0.14 (4/5 seeds better);
  - Q → A: −0.65 ± 0.03;
  - r50 −0.15, i100 −0.09;
  - separation: same context −0.16, other voice −0.16, other scene −0.07;
  - horizon −4,050 ticks.
  - All of these are 0/5 seeds better.
  - ASSUMED: the signal does not see the inner voice. So in the Q → A pause the learned links run at full strength, and the stored pause state is driven by recall, not by the input.
- **Not adopted.**

## 4. Other measurements (`tables/res.txt`)

- **Units on:**
  - Z arms: 117 in speech, 7 in silence (X3: 109 / 10);
  - α-1 arms (Z10a1, ZL, most L arms): 64 at all times.
- **Live links:** Z arms 1.8–1.9 M (X3 2.9 M); ZL 1.08 M; Z10a1 0.91 M.
- **Readout of stored codes:** Z4–Z10 0.99–1.00 (X3 0.95).

## 5. Files (all in `v3/x6b/`)

- **Plan:** `PLAN.md` (plan and the logged changes after it).
- **Code:**
  - `x6b_hippo.py`: the harness.
  - `x6b_run.py`: the runner. Usage: `python3 -I x6b_run.py specs/<file>.json`; it writes to `out/`, which is git-ignored.
  - `x6b_specs.py`: builds the specs (`cal`, or `res cal_choice.json`).
  - `x6b_mkjob.py`: makes the queue jobs.
  - `x6b_analyze.py`: the tables (`res <dir> [arms]`, `cal <dir>`).
  - `x6b_extra.py`: fact errors and paired arm comparisons.
  - `regress.py`: the X3-vs-X6 regression.
  - `grid.py` with `grid1–3.txt`: the small-model smoke grids (one JSON config per line on stdin).
- **Inputs:** `cal_choice.json`, `specs/*.json` (calibration `cal_p0–7`, `cal3_p0–3`; results `r_p0–15`; follow-up `d_p0–3`; `arms.json`).
- **Tables:** `tables/res.txt`, `cal.txt`, `regress.txt`, `facts.txt`, `pair_ZL_vs_Z10a1.txt`, `pair_Z10a1_vs_Z10.txt`.
- **Compute:**
  - A800 CPU jobs `x6b_cal1–3` (5–6 min each), `x6b_r0–r3` (23–25 min each) and `x6b_d1` (4 min), 4 processes each.
  - Locally only small checks, one process at a time.
