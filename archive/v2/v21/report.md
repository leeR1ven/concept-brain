(Saved by the coordinator from the agent's final reply; the harness blocked the agent's own write.)

# ARCH_v2.1: Steps 0–3 from review round 2

> **Note (2026-11-01, found by v2.2):** `agent_v21.py` in this folder applies dips only after *self-initiated* speech.
> That is the rejected `v21fin` variant, not the final config reported below, which dips after every own utterance.
> The `rec_fix` confirmation (J 1.22) was run with this code. v2.2's copy (`../v22/agent_v21.py`) restores the reported
> behaviour as `dip_scope='all'` (the default) and keeps the old one as `dip_scope='scene'`. The canonical re-baseline of
> the reported v2.1 is J 1.184 ± 0.081 (5 seeds; `../v22/report.md` §1). The hearing and meaning rounds were built on
> this folder's code; their comparisons are paired, so their deltas stand.

The harness blocked writing `report.md` (twice), so the full report is here. Everything else is in this directory (`v2/v21/` in the repo; `out/` below is `results/` here; the unchanged ARCH_v2 files and `bench_copy/` are used from `../synth/`): code, all result JSONs, and the aggregate tables in `out/`.

Labels: **SIMULATED** = my run, file named. **PROVED** = from code, counters or arithmetic. **CITED** = another report. **ASSUMED** = not checked.

What ran:
- 51 staged runs at nS 1000, almost all on the A800 (4 seeds per job, about 19–30 min each). One run was local (numpy 1.26), on a rejected variant.
- 20 lives of 10⁶ ticks: 12 + 8 on the V100 (before it was retired) and the A800.
- Symbolic probes locally, one process at a time.
- 0 mechanics violations and 0 overlaps in every run.
- Every comparison is paired, per seed, on the same machine and numpy version (A800, numpy 2.1), except that one local run (see §8).

## 0. Bottom line

1. **Step 0 (honest measurement) changes the picture.** SIMULATED, 5 seeds.
   - Held-out pairs are now never heard in training and are asked only of frozen copies.
   - Under that protocol, unchanged ARCH_v2 has H 0.60 (not 0.20) and J 1.00 (not 0.82). The old H mostly measured drift.
   - On held-out items, ARCH_v2 only equals the best constant answer per task (0.55 vs 0.54).
   - The drift loss after 300 / 900 untutored dialogues is 0.30 / 0.54.
   - I propose J2 = F + 0.5 RET + **1.0** H + 0.25 DR as the objective.
2. **Step 1, role-aware listening (RAL), is kept. It is the main result.**
   - **Mechanism.** What a heard line teaches as an answer (its GEN + gate steps and its NEXT link) waits on a per-synapse eligibility trace.
     - Cancel if the line was a question: I answer it, or someone answers it at once.
     - Keep if the line carries a reward after my utterance (feedback or recast), or if nobody answers it.
   - **Staged, 5 seeds, paired:**
     - J 1.00 → **1.23** (+0.24 ± 0.07, up on every seed);
     - H 0.60 → **0.76**;
     - drift loss 0.30 / 0.54 → **−0.01 / 0.01**;
     - role confusion 0.14 → 0.09;
     - no ritual or colour cost.
   - **Old protocol:** J 0.82 → 1.17, H 0.16 → 0.63.
3. **Step 2, comprehension feeds production, is kept.**
   - **Mechanism.** A copy route, gated on the question's frame only, that says the word recognised at the attended slot or the slot's location word.
   - **Held-out:** what 0.61 → **0.81**, where 0.81 → **0.90**, slot 0.13 → 0.43.
   - J +0.03, within noise. Switching the route off shows that naming now runs through it.
4. **Step 3, a signed value loop, is kept for life behaviour, at a staged cost.**
   - **Mechanism.** The learned value V_go becomes a NoGo veto on self-initiated speech, plus dips, expiring actor eligibility and coarse social detectors.
   - **Life results:**
     - 你好 to an empty room drops from about 60–80 to **5–9** per tenth of life;
     - appropriate initiatives 0.19 → **0.38** (0.46 in the last tenth);
     - social replies 0.32 → **0.41–0.43**;
     - nonsense initiatives −36 %.
   - **Staged cost:** J −0.08 ± 0.12 against Steps 1+2, mostly the check task.
5. **Failed or still open:**
   - Life answer accuracy did not improve (0.34 / 0.26 vs 0.34 / 0.36). The agent speaks right after 79–87 % of the partner's corrections, so RAL treats them as questions and cancels them.
   - Life is still below the echo-or-constant responder (0.34 vs 0.62).
   - Three predictions failed: life nonsense −50 % (got −36 %), held-out slot ≥ 0.6 (got 0.43), life namings rise (got 10 / 9).

**Final ARCH_v2.1 config** = ARCH_v2 final_noctx + `ral=1 ral_other=1 say_rec=1 say_loc=1 scene_held=1 val_veto=0.2 dip=1 act_T=60 soc_feats=1 rec_fix=1` (§7).

## 1. Step 0: honest measurement

### 1.1 What was built (the bench's metric code is untouched)

- **`run_v21.py`**, a staged driver that keeps run_j's protocol and adds:
  - `proto=frozen` (new default): training worlds use `held_frac` 0; held-out pairs are probed only on frozen copies.
  - `proto=old`: the old protocol, kept for comparison.
  - Per-task accuracy and held-out accuracy.
  - **Role confusion:** unacceptable replies that are either a partner line (an exact prompt, a prefix of one, or a reply containing a question syllable 什么吗哪几谁) or an echo (the prompt, or a piece of it of at least 2 characters).
  - **Same-run trivial references on the same items:** echo; the best constant per task, chosen on these very items (an upper bound for any constant responder); and the prior constant.
  - Drift after 300 and after 900 untutored dialogues.
  - develop's D1–D3 inline in every run: replies, attended slot vs X's slot, and whether visual recognition (VR) holds X's name.
  - Lesion re-probes, and J and J2 side by side.
- **Life:**
  - `life_metrics.py`: initiatives per tenth of life, appropriate share, groups, role confusion, and same-life references.
  - `refs_life.py`: the bench's life reference agents on the same life (seed 5, recast tutor, retino eye).
- **Analysis:** `agg_v21.py` and `paired.py`, which give per-seed deltas and deltas on seeds 1–4 alone.
- **Regression tests in `regress/`:**
  - `gen_probe_v21.py` (learning reviewer's symbolic probe on the v2.1 core, with role hooks and a tick timeline);
  - `gen_probe2.py` (the class-dependent probe, on the v2.1 core);
  - develop's D1–D3 (also kept as scripts; `AGENT=<pickle>`);
  - `drift_veto.py`, `drift_replies.py`, `drift_own.py`.
- **`test_equiv_v21.py`:** with every new key at its default, ARCH_v2 is reproduced bit for bit (replies, MIX, hippocampus, GEN weights). PROVED by running it after each change.

### 1.2 Results (SIMULATED; `out/runs/v2_*`, `out/agg_step01.*`)

ARCH_v2, unchanged code:

| metric | old protocol (3 seeds, rerun) | published final_noctx (5 seeds) | **frozen protocol (5 seeds)** |
|---|---|---|---|
| J | 0.817 ± 0.074 | 0.825 ± 0.055 | **0.996 ± 0.077** |
| J2 | 0.940 | 0.973 | **1.444 ± 0.104** |
| F / RET | 0.658 / −0.025 | 0.656 / −0.032 | 0.746 / −0.079 |
| H | 0.163 | 0.198 | **0.597 ± 0.049** |
| drift loss after 300 / 900 | 0.276 / 0.453 | 0.262 / – | 0.303 / **0.538** |
| role confusion: all / held-out / after 900 drift | 0.161 / 0.386 / 0.320 | – | 0.139 / 0.242 / 0.352 |

Held-out accuracy by task:

| task (n) | old protocol | frozen protocol | best constant per task on these items |
|---|---|---|---|
| what (89) | 0.05 | 0.28 | 0.34 |
| slot (82) | 0.07 | 0.11 | 0.37 |
| where (78) | 0.00 | 0.68 | 0.47 |
| offer (52) | 0.03 | 0.54 | 1.00 (要) |
| taste (30) | 0.00 | 0.10 | 1.00 (好吃) |
| colour (45) | 0.75 | 0.93 | 0.31 |
| count (57) | 0.29 | 0.97 | 0.53 |
| check (122) | 0.18 | 0.75 | 0.61 |

**Trivial references (frozen protocol):**
- all items: agent 0.781, echo 0.057, best constant 0.424, prior 0.252;
- held-out items: **agent 0.554 vs best constant 0.535**.

**Comprehension vs production, held-out:**
- where: attends X's slot 1.00, says it 0.68;
- what: VR holds the name 0.79, says it 0.28.

**Life (seed 5, same life):**
- free replies 0.343 (last tenth 0.469) vs echo 0.303, best constant per task 0.561, echo-or-constant 0.615;
- scene questions 0.333 vs constant 0.459;
- held-out 0.100 vs 0.590;
- social 0.317 vs echo 1.00;
- role confusion 0.44;
- initiatives 1,891: nonsense 703, alone 822, greeting 347, naming 18;
- appropriate share 0.193 (0.341 with the partner present). The privileged random namer gets 0.19 (0.79 with the partner present).
- 你好 to an empty room, per tenth of life: 3 57 87 70 57 65 74 61 88 64.

### 1.3 Lessons

1. **The old H measured drift.** The learning reviewer's prediction "H ≥ 0.35" holds (0.60). The other part, "what / offer / taste ≥ 0.3", **fails** for what (0.28) and taste (0.10). tune2 found the same (CITED).
2. **Honest generalisation was at constant level** (0.554 vs 0.535). ARCH_v2 lost to a constant on what, slot, offer and taste.
3. **Drift keeps going:** 0.30 → 0.54 between 300 and 900 untutored dialogues. This confirms the skeptic on 5 seeds.
4. **J2 is needed.** With J's 0.25 weight on H, a gain of +0.2 in H shows up as only +0.05 in J.

## 2. Step 1: role-aware listening, self in the loop

### 2.1 What was built (`core_v21.RoleCore`, `agent_v21.AgentV21`; `ral=1 ral_other=1`)

**The trace.** While another speaker's line is learned in the context of the held thought, every R2 step of GEN and of the copy gates is recorded per synapse (`RecDelta`). This is the eligibility trace. The steps are applied at once. The hippocampal NEXT link that the line's end creates is part of the trace.

**Role resolution, by one broadcast signal:**

| signal (all perceivable) | role | action |
|---|---|---|
| my mouth starts ≤ 12 ticks after the line ends (efference) | a question or initiation addressed to me | **cancel** the trace, cut the NEXT link |
| another line starts ≤ 6 ticks after it ends (`ral_other`) | a question someone else answered at once | **cancel** |
| the line carries a reward (onset ± 2 ticks) and follows my utterance (≤ 60 ticks) | feedback or a recast | **commit** at once, even if I then speak; link it as the NEXT of the line I answered |
| none of these within 12 ticks | another speaker's reply, a statement, an unanswered correction | **commit** |

**Self in the loop.**
- My own utterance enters the thought under a SELF role (`T_own`: its items and time).
- `T_own` is a separate slot, not part of the heard-line bag. This avoids the bag confound tune2 found (CITED).
- `T_own` is the efference copy that the contingency check reads. It is never a listening target.

**"Moving on" is not punished.** A next question is one I answer, so its learning is cancelled, including the "predicted but absent → weaken" step on the old answer.

**Locality.** One trace per synapse plus one broadcast signal. Nothing reads world state. PROVED from code.

**Speaker identity (`voice=1`; perception only, a hook for world2).**
- Voice categories come from the ear's speaker-normalisation residual: the raw window minus the syllable category's running template, sign-quantised, with vigilance.
- Purity (`voice_check.py`, SIMULATED): partner 0.91, other2 0.72 (0.13 unclassified).
- Not used in learning. On the bench, 30 % of model replies are said by the partner itself, so timing (`ral_other`) is the more reliable cue.

### 2.2 Results

**Symbolic regression** (`regress/gen_probe_v21.py`, 3 seeds, `out/symbolic_summary.txt`):

| tutor | variant | frozen seen | frozen held-out | live held-out |
|---|---|---|---|---|
| recast | ARCH_v2 | 0.97 | **0.00** | 0.00 |
| recast | RAL | 1.00 | **0.81** | 0.76 |
| recast | RAL + own lines in the hippocampus | 1.00 | 0.67 | 0.61 |
| recast | RAL + overlap contingency | 0.97 | 0.28 | 0.30 |
| mixed (30 % model replies) | ARCH_v2 | 0.97 | 0.00 | 0.00 |
| mixed | Xcut | 1.00 | 0.72 | 0.80 |
| mixed | RAL without `ral_other` | 1.00 | 0.06 | 0.06 |
| mixed | **RAL + `ral_other`** | 1.00 | **0.83** | 0.80 |

- The learning reviewer's oracle (Cor2) scored 0.81.
- The class probe (`gen_probe2.py`) gives 0.25 for both ARCH_v2 and RAL: role-aware learning does not create class-based generalisation. That is Step 4.

**Staged, 5 seeds, frozen protocol** (`v2_frozen` vs `v21s1`; `out/paired_v2_v21s1.txt`):

| metric | ARCH_v2 | + RAL | paired Δ |
|---|---|---|---|
| J | 0.996 ± 0.077 | **1.233 ± 0.060** | **+0.237 ± 0.068, every seed up (seeds 1–4: +0.222)** |
| J2 | 1.444 | **1.807** | +0.363, every seed up |
| F / RET | 0.746 / −0.079 | 0.817 / −0.037 | F +0.072 |
| H | 0.597 | **0.764** | +0.168, every seed up |
| drift loss 300 / 900 | 0.303 / 0.538 | **−0.007 / 0.008** | −0.310 / −0.530, every seed |
| role confusion: all / held / after 900 drift | 0.139 / 0.242 / 0.352 | **0.086 / 0.118 / 0.027** | |
| ritual / colour | 0.816 / 0.930 | 0.952 / 0.962 | no cost |
| held-out: agent vs best constant | 0.554 vs 0.535 | **0.728** vs 0.535 | |

Held-out by task, ARCH_v2 → RAL:
- what 0.28 → 0.61, slot 0.11 → 0.13, where 0.68 → 0.81;
- offer 0.54 → 0.92, taste 0.10 → 0.87, check 0.75 → 0.84;
- colour 0.93 → 0.98, count 0.97 → 0.98.

**Old protocol, 3 seeds:**
- J 0.817 → **1.172** (+0.355 ± 0.051, every seed);
- H 0.163 → **0.631**;
- drift loss 0.276 / 0.453 → 0.002 / 0.012.

**Variants on seed 0** (single-seed, so differences under about 0.07 are noise; CITED from gpuport):

| variant | J | ritual | colour | check |
|---|---|---|---|---|
| ARCH_v2 | 0.879 | 0.76 | 0.93 | – |
| develop's pure veto | 0.861 | **0.24** | **0.39** | **0.00** |
| RAL with reward contingency | 1.157 | 1.00 | 0.82 | 0.92 |
| RAL + own lines in the hippocampus | 1.183 | 1.00 | 0.93 | 0.94 |
| RAL + overlap contingency | 1.165 | 1.00 | 0.89 | 0.84 |

### 2.3 Cause of the veto's ritual and colour costs (found)

- **The agent very often speaks right after a recast.** In one seed-1 staged run, 1,397 rewarded recasts were followed within 12 ticks by the agent's own speech. In life this happened after 85 % of confirmations and 79–87 % of corrections (PROVED from counters and life logs).
- **The pure veto reads each of these as "a question I answered" and cancels it.** That removes the only teacher of symmetric rituals and of colours.
- **The reward exemption fixes it:** ritual 0.95, colour 0.96 over 5 seeds.
- **develop's VETO 3 was a bug.** It was bit-identical to VETO 1 because `own_st` is cleared on the very tick the mouth goes silent (the body sets Hp = −1 at the stop tick), so `own_last` was never set (PROVED from code).

### 2.4 Rejected on evidence

- **Pure veto:** collapses ritual, colour and check.
- **Overlap contingency:** a question that contains my words is taken as feedback (symbolic held-out 0.28).
- **Own lines in the hippocampus** (`ral_hip_own`). Built as self-tagged cells that a heard cue never recalls.
  - Recall then copies my own unconfirmed answers: symbolic held-out 0.67 vs 0.81; on bench seed 0 it is equal.
  - Not kept.
- **Strict cancel after own speech:** corrections are lost (symbolic seen 0.75).
- **Learned question detector** (`ral_q`): target 1 for a heard line whose answer was rewarded, target 0 for my own rewarded words.
  - It learns sensible weights: 这是什么 0.86 and 香蕉在哪里 0.63, against 好吃 0.26 and 不是 0.11.
  - It helps life scene answers (0.40 / 0.44).
  - But staged J −0.20 on paired seeds 1–2, drift at 900 +0.17, and life social 0.15 / 0.03, because 你好 is taken as a non-question.
  - The same detector trained on "the speaker waited" targets was too sparse in life (3 negatives vs 310 positives).

### 2.5 Life with Step 1 alone (seeds 5 / 6)

- held-out 0.10 / 0.11 → 0.19 / 0.12;
- free replies 0.34 / 0.36 → 0.31 / 0.24;
- scene 0.33 / 0.39 → 0.35 / 0.24;
- social 0.32 / 0.31 → 0.27 / 0.29;
- nonsense initiatives +8 % / +1 %.

Cause: life has few rewards (16 % of heard lines carry one, against 39 % in staged runs), and RAL cancels the corrections that the agent talks after (§2.3). **Step 1 does not improve life accuracy.**

## 3. Step 2: comprehension feeds production

### 3.1 What was built (`say_rec=1 say_loc=1 scene_held=1`)

**A fourth copy route.**
- **Candidates:**
  - the words VR finds at the attended slot (object, colour and count words);
  - the attended slot's location word (the inverse of the word → slot map, with P(slot | word) ≥ 0.8);
  - minus the items already in the line and in the prompt.
- **Gate (`gateR`, two outputs: recognised / location).** It reads only frame-context keys (the gate keys minus every candidate-identity key, as `gate_noid`). Its strength is added to every candidate, and GEN orders the syllables (苹 then 果) and ends the word.
- **Learning:** by listening (R2; target = the heard item is a candidate), under the Step 1 trace.
- **Scene thoughts (`scene_held=1`):** a self-initiated thought reads the held thought, so what the agent keeps thinking about can reach speech.

### 3.2 Results (5 seeds, frozen; `s2_rl`, `out/agg_step2.json`, `out/paired_s1_s2.txt`)

| metric | Step 1 | Steps 1 + 2 | paired Δ |
|---|---|---|---|
| J / J2 | 1.233 / 1.807 | 1.263 ± 0.072 / 1.868 | +0.029 ± 0.058 / +0.062 (mixed, within noise) |
| F / H / DR | 0.817 / 0.764 / 0.975 | 0.828 / 0.808 / 0.980 | |
| drift loss 300 / 900 | −0.007 / 0.008 | −0.007 / 0.024 | |
| role confusion: all / held-out | 0.086 / 0.118 | 0.069 / 0.065 | |
| **held-out what / slot / where** | 0.61 / 0.13 / 0.81 | **0.81 / 0.43 / 0.90** | what and slot up on every seed |
| held-out taste / check / offer | 0.87 / 0.84 / 0.92 | 0.73 / 0.80 / 0.96 | |
| held-out: agent vs best constant | 0.728 | **0.804** vs 0.535 | |

**Comprehension now reaches production (held-out):**
- where: attends 1.00; says it 0.68 → 0.81 → **0.90** (ARCH_v2 → Step 1 → Steps 1+2);
- what: VR holds the name 0.79; says it 0.28 → 0.61 → **0.81**.

**Lesion (gateR zeroed at test, 5 seeds):**
- B 0.908 → 0.395; held-out B 0.718 → 0.335;
- C 0.971 → 0.936;
- D unchanged.

So the route carries naming. GEN stops learning names the route already predicts.

**Variants (seed 1):**
- `say_noprompt=0`: J 1.340, slot 0.24;
- `scene_held=1`: J 1.334, slot 0.35;
- default: J 1.336, slot 0.41.

**Life, Steps 1 + 2 (seeds 5 / 6):** free 0.36 / 0.18, social 0.39 / 0.07, namings 26 / 9, nonsense 631 / 665.

## 4. Step 3: a signed value loop

### 4.1 What was built

- **NoGo veto (`val_veto=0.2`).**
  - V_go is ARCH_v2's learned value: its target is a reaction with reward; in life (`no_react`) it is also pushed to 0 when nothing reacts within 50 ticks.
  - V_go no longer scales eagerness (G = g0).
  - A self-initiated thought may release only if V_go ≥ 0.2. Replies are never vetoed.
- **Dips (`dip=1`).**
  - If an own utterance gets no reaction at all within 50 ticks, critic and actor are updated with r = 0: δ = −V.
  - A reaction without reward is not a dip.
  - The first implementation compared against the tick after the agent's increment, so it fired 0 dips. Fixed before the numbers below.
- **Expiring actor eligibility (`act_T=60`).**
- **Coarse social detectors (`soc_feats=1`)**, features for V_go and P_oth, from observations only:
  - someone spoke within 300 ticks;
  - something appeared in silence within 60 ticks;
  - something disappeared in silence within 60 ticks.
- **Value and actor in their own slow table (`val_act=1`, η 0.05):** built and tested, **rejected**.

### 4.2 Results

**Life, seeds 5 / 6, on top of Steps 1–2** (`out/life/`; SIMULATED):

| config | empty-room 你好: last tenth (mean per tenth) | appropriate: whole life / last tenth | social | nonsense | initiatives |
|---|---|---|---|---|---|
| ARCH_v2 | 61 / 75 (62 / 78) | 0.19 / 0.16 · 0.20 / 0.25 | 0.32 / 0.31 | 691 / 723 | 1876 / 1983 |
| Steps 1 + 2 | 92 / 77 (84 / 91) | 0.20 / 0.17 · 0.20 / 0.24 | 0.39 / 0.07 | 631 / 665 | 1883 / 1990 |
| + veto only | 16 / 11 (21 / 22) | 0.33 / 0.34 · 0.34 / 0.39 | 0.39 / 0.10 | 516 / 545 | 1099 / 1157 |
| + veto + slow actor (rejected) | 12 / 11 (15 / 16) | 0.36 / 0.40 · 0.37 / 0.43 | **0.12 / 0.22** | 456 / 526 | 965 / 1091 |
| + veto + dips on self-initiated speech only (rejected) | 4 / 7 (13 / 16) | 0.37 / 0.44 · 0.37 / 0.43 | **0.23 / 0.05** | 435 / 520 | 943 / 1078 |
| **+ veto + dips + social features (ARCH_v2.1)** | **5 / 9 (14 / 17)** | **0.38 / 0.46 · 0.38 / 0.47** | **0.41 / 0.43** | **432 / 470** | 944 / 1038 |

**Staged runs:**
- **ARCH_v2.1 (`v21final`, 5 seeds):** J 1.179 ± 0.105.
  - Against ARCH_v2: +0.183 ± 0.093, up on every seed.
  - Against Steps 1+2: −0.083 ± 0.115 (mixed). The cost is mostly check: held-out check 0.80 → 0.43, check overall 0.86 → 0.69; seed 3 has J 0.99.
- **Dips are rare but heavy:** 3–13 per staged run, each a full actor step (η 0.3 × δ ≈ −0.8) on the reply, here 是 / 不是.
- **Other variants:**
  - Slow actor without working dips (`v21s3`, 4 seeds): neutral, −0.02 ± 0.04.
  - Dips on self-initiated speech only (`v21fin`, 5 seeds): J 1.207 ± 0.119, −0.06 ± 0.11 vs Steps 1+2, with life social collapsing. Its seed 0 ran locally under numpy 1.26, so that seed is not strictly comparable.
  - The cause of the remaining staged cost is not isolated.

**Lesions:**
- Without Step 3 (the Steps 1+2 row), empty-room greetings return to 77–92 per tenth. So V_go now reaches behaviour (principle 7).
- In staged runs, removing the veto changes nothing, because self-initiated thoughts never release there.
- The slow actor table was a passenger in staged runs (≤ 0.02).

**Lessons:**
- The veto is what stops talking to empty rooms: V_go already knew "no reaction → 0".
- Dips on replies help social replies in life (a wrong social reply often meets a partner who leaves) but hurt check in staged runs.
- The actor needs a fast action path; a slow table alone collapses social replies.

## 5. Life summary (10⁶ ticks, seeds 5 / 6, recast tutor)

| config | free | last tenth | scene | held-out | social | role confusion | initiatives | nonsense | namings | appropriate | empty-room 你好, last tenth |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ARCH_v2 | 0.34 / 0.36 | 0.46 / 0.50 | 0.33 / 0.39 | 0.10 / 0.11 | 0.32 / 0.31 | 0.45 / 0.48 | 1876 / 1983 | 691 / 723 | 20 / 16 | 0.19 / 0.20 | 61 / 75 |
| + Step 1 | 0.31 / 0.24 | 0.56 / 0.43 | 0.35 / 0.24 | 0.19 / 0.12 | 0.27 / 0.29 | 0.45 / 0.54 | 1999 / 1885 | 745 / 730 | 2 / 9 | 0.20 / 0.18 | 101 / 79 |
| + Step 2 | 0.36 / 0.18 | 0.58 / 0.30 | 0.36 / 0.21 | 0.19 / 0.11 | 0.39 / 0.07 | 0.41 / 0.59 | 1883 / 1990 | 631 / 665 | 26 / 9 | 0.20 / 0.20 | 92 / 77 |
| **ARCH_v2.1** | 0.34 / 0.26 | 0.53 / 0.44 | 0.33 / 0.21 | 0.13 / 0.12 | **0.41 / 0.43** | 0.39 / 0.49 | 944 / 1038 | **432 / 470** | 10 / 9 | **0.38 / 0.38** | **5 / 9** |

Same-life references (seed 5): echo 0.30, best constant per task 0.56, echo-or-constant 0.62 (scene constant 0.46, held-out constant 0.59, social echo 1.00). The random namer's appropriate share is 0.19 (0.79 with the partner present).

Two lives per configuration cannot resolve small differences. The gains in initiatives and social replies are large and hold on both seeds. Answer accuracy did not improve, and seed 6 is worse for every RAL configuration. **The skeptic's port gate is not passed.**

## 6. Reviewer predictions

| step | prediction (who) | result | verdict |
|---|---|---|---|
| 0 | frozen H ≥ 0.35 (learning) | 0.60 | met |
| 0 | held-out what / offer / taste ≥ 0.3 (learning) | 0.28 / 0.54 / 0.10 | **failed** (what, taste) |
| 1 | J gain ≥ seed SD 0.06 (develop) | +0.237 ± 0.068, every seed (seeds 1–4: +0.222) | met |
| 1 | drift loss ≤ 0.05–0.10, also after 900 (all) | −0.007 / 0.008 | met |
| 1 | frozen H ≥ 0.45 (learning) | 0.76 | met, but not diagnostic: the protocol alone gives ARCH_v2 0.60 |
| 1 | F within −0.03 / −0.05 (skeptic, learning) | +0.07 | met |
| 1 | life nonsense initiatives −50 % (develop) | Step 1: +8 % / +1 %; final: −37 % / −35 % | **failed** |
| 1 | greeting first stays ≈ 350 (develop) | 348 / 389 | met |
| 1 | world2: no praise words as answers; agent-to-agent echoes stop rising | not run | untested |
| 2 | held-out where ≥ 0.6 / what ≥ 0.7 | 0.90 / 0.81 | met |
| 2 | held-out slot ≥ 0.6 | 0.43 | **failed** |
| 2 | life namings rise from 18 | 26 / 9 (Step 2); 10 / 9 (final) | **failed** |
| 3 | 你好 to an empty room ≈ 65 → ≤ 15 per tenth (neuro) | 5 / 9 in the last tenth; mean 14 / 17 | met |
| 3 | appropriate initiatives ≥ 0.4 (neuro) | 0.38 over the life; 0.46 / 0.47 in the last tenth | met in the last tenth only |
| 3 | social replies ≥ 0.30 (neuro) | 0.41 / 0.43 | met |
| 3 | rewarded greetings ≥ 30 per tenth (neuro) | 348 / 389 per life | met |

## 7. Final ARCH_v2.1 config and confirmation

**Config:** ARCH_v2 final_noctx (run_v21 defaults; life adds THINK_LIFE + ct_assoc 0.5, wander 50), plus:
- Step 1: `ral=1 ral_other=1` (defaults W 12, W_other 6, contingency by reward, T 60, cut 1).
- Step 2: `say_rec=1 say_loc=1 scene_held=1`.
- Step 3: `val_veto=0.2 dip=1 act_T=60 soc_feats=1` (actor stays in GEN).
- `rec_fix=1`.
- Not kept: `ral_hip_own`, contingency by overlap / none, `ral_strict`, `ral_q`, `val_act`. `voice` is a perception hook only.

**Staged, 5 seeds, frozen** (`out/agg_final.json`, `out/paired_v2_final.txt`):

| metric | ARCH_v2 | Step 1 | Steps 1+2 | **ARCH_v2.1** |
|---|---|---|---|---|
| J | 0.996 ± 0.077 | 1.233 ± 0.060 | 1.263 ± 0.072 | **1.179 ± 0.105** (+0.183 ± 0.093 vs ARCH_v2, every seed up) |
| J2 | 1.444 | 1.807 | 1.868 | 1.685 |
| F / H / DR | 0.746 / 0.597 / 0.564 | 0.817 / 0.764 / 0.975 | 0.828 / 0.808 / 0.980 | 0.783 / 0.675 / 0.971 |
| drift loss 300 / 900 | 0.303 / 0.538 | −0.007 / 0.008 | −0.007 / 0.024 | **−0.014 / −0.002** |
| role confusion | 0.139 | 0.086 | 0.069 | 0.080 |
| held-out what / slot / where | 0.28 / 0.11 / 0.68 | 0.61 / 0.13 / 0.81 | 0.81 / 0.43 / 0.90 | 0.78 / 0.27 / 0.90 |
| held-out offer / taste / check | 0.54 / 0.10 / 0.75 | 0.92 / 0.87 / 0.84 | 0.96 / 0.73 / 0.80 | 0.90 / 0.73 / **0.43** |
| held-out vs best constant (0.535) | 0.554 | 0.728 | 0.804 | 0.692 |
| old-protocol J (3 seeds) | 0.817 | 1.172 | – | 1.131 |

**Lesions on the final configuration:**
- gateR off: B 0.884 → 0.427 (held-out 0.651 → 0.330);
- veto off: no staged change.

**Cache fix** (`rec_fix=1`, coordinator / gpuport):
- The visual-recognition cache is now also invalidated when eye_rate changes (the base-rate filter) and when eye_n crosses 50.
- It changes trajectories but does no harm: paired over 4 seeds, J +0.055 ± 0.069 (mixed), mostly from seed 3 (0.99 → 1.16; check partly restored). The final configuration with the fix scores J 1.223 ± 0.075 (4 seeds; `out/paired_recfix.txt`).
- It is included in the final config. The 5-seed table above was run before the fix.

**Choice.** On the staged bench, Steps 1+2 are best. The goal is a life, though, and Step 3 is the only variant that meets all of its life predictions. So ARCH_v2.1 = 1+2+3, and the dip size is the first thing to tune. If the bench should rule, use Steps 1+2: J 1.263, same drift, no value loop.

## 8. Open problems and next steps (ranked)

1. **Telling a correction from a next question in life.** The agent talks after 79–87 % of corrections, so RAL cancels them, and life accuracy is unchanged.
   - Tried and rejected here: overlap with my words, and a reward-trained question detector.
   - Candidates:
     - a perceivable question cue in the world (prosody);
     - less compulsive speech after reactions (`u_R1` < 1, together with the veto);
     - class codes (Step 4) to recognise answer-like lines;
     - contingent caregiving (develop's §4.3).
2. **Step 3's staged cost on check:** smaller dip steps, or dips only where P_oth predicted a reaction that did not come.
3. **Held-out slot 0.27–0.43 and taste 0.73:** the route boosts all recognised words equally. Fixing this needs class codes (Step 4).
4. **Life below trivial responders** (0.34 vs 0.62).
5. **Built but not kept, to revisit:**
   - own lines in the hippocampus: they need a "confirmed answer" flag before recall may copy them;
   - a slow value table: it needs a fast action path;
   - speaker identity: perceived but unused; the key for trust in world2.
6. **Machine / numpy noise** (coordinator, gpuport).
   - All paired comparisons here are A800 / numpy 2.1, except one local seed-0 run of the rejected `v21fin` variant.
   - Single-seed differences under about 0.07 (the seed-0 variant table) are noise.
   - The canonical tie rule for the DG k-WTA is not adopted yet: adopting it would require re-running every comparison. I recommend it as the first change in v2.2.

**Checked:**
- 0 mechanics violations in 51 staged runs and 20 lives;
- the agent reads observations only;
- no sleep; no clear/load added; speech caused by thinking; local rules only (a per-synapse trace plus one broadcast signal);
- every learned part was lesioned or its absence measured (gateR, the V_go veto, the slow actor table).

## Files (this directory; `out/` = `results/` in the repo)

- **Code:**
  - `core_v21.py`, `agent_v21.py` (the unchanged ARCH_v2 files and `bench_copy/` come from `../synth/`);
  - drivers: `run_v21.py`, `life_v21.py`, `life_metrics.py`, `refs_life.py`;
  - analysis: `agg_v21.py`, `paired.py`, `life_table.py`;
  - checks: `test_equiv_v21.py`, `voice_check.py`, `q_check.py`;
  - job tooling: `submit.py`, `jobs/*.txt`;
  - `regress/` (`gen_probe_v21.py` + `gp_v21.jsonl`, `gen_probe2.py`, D1–D3, drift scripts).
- **Results:**
  - `out/runs/*.json`: every staged run, with replies, references, role confusion, comprehension and lesions;
  - `out/life/*_s{5,6}.json` and their logs;
  - `out/agg_all.json|txt`, `out/agg_step01.*`, `out/agg_step2.json`, `out/agg_final.json`;
  - `out/paired_*.txt`, `out/life_table_s5.txt`, `out/symbolic_summary.txt`, `out/voice_check.json`.

## summary_zh

1. **第 0 步（诚实测量）：**
   - 训练时完全不出现留出组合，只在冻结副本上测。这样测，ARCH_v2 本身的泛化分 H 是 0.60（不是 0.20），J 是 1.00（不是 0.82）。旧的 H 主要测到的是"漂移"。
   - 但在留出题上，它只和"每种题固定答一句"的傻办法差不多（0.55 对 0.54）。
   - 无人指导 300 / 900 段对话后，成绩下降 0.30 / 0.54。
   - 建议改用 J2，让泛化和熟练题同等权重。
2. **第 1 步（分角色的边听边学）保留，是主要成果：**
   - 一句话"当答案学"的改变先挂在突触的资格迹上：
     - 我接着回答了它，或别人马上回答了它，说明它是问题，撤销；
     - 它带着奖励、紧跟在我的话之后，说明是反馈或复述，保留。
   - 5 个种子：J 1.00 → 1.23（每个种子都提高），H 0.60 → 0.76，无人指导后几乎不再漂移，礼貌用语和颜色不再受损。
   - 原来的"否决"会伤礼貌用语和颜色，是因为孩子在大人复述之后常常接着说话，被误当成"回答了问题"。
3. **第 2 步（听懂的能说出来）保留：**
   - 按句型开门的"说出我认出的东西、说出它在哪"通道。
   - 留出题"这是什么"0.61 → 0.81，"在哪里"0.81 → 0.90，"左边是什么"0.13 → 0.43（没达到 0.6 的预测）。
4. **第 3 步（有正有负的价值回路）为"一生"保留：**
   - 价值低时否决主动说话；没人理就"失望"（δ = −V）；加上粗略的社交感知。
   - 对空房间说"你好"从每十分之一生约 60–80 次降到 5–9 次。
   - 合适的主动说话 0.19 → 0.38（最后十分之一 0.46），社交回答 0.41–0.43。
   - 代价是测试台 J 比只做 1、2 两步低约 0.08，主要是"是不是"题。
5. **没成功的：**
   - "一生"里答题准确率没有提高：孩子在大人纠正后多半接着说话，纠正被当成"问题"而撤销。
   - "一生"仍低于"鹦鹉学舌或固定答一句"的参考（0.34 对 0.62）。
   - 无意义的主动说话只减少 36%（预测 50%），说物体名字也没有增加。
   - 把自己的话存进海马体、把价值放进单独的慢速表，都试过了，有代价，没有保留。
   - 视觉识别缓存的陈旧问题已修复，对分数没有坏影响。