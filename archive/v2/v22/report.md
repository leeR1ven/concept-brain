# ARCH_v2.2 report: merge the validated pieces, then fix learning from corrections in life

**Repo layout.** This folder holds v2.2's own code plus four files that replace the repo copies: `agent_v21.py` (dip
scope, core factory), `rules.py` (canonical DG tie rule), `randlayer.py` (`pow:p` allocation) and `submit.py`. Everything
else comes unchanged from `../v21` and `../synth`; `_paths.py` loads the three replaced modules first (`submit.py` is only job tooling). A short staged run
from this layout reproduces the working copy bit for bit (replies, GEN, scores and statistics), and `test_equiv_v22.py`
prints EQUIVALENT. The code writes to `out/` (not committed); the results quoted here are copied into `results/`, without
the life logs (`*_log.pkl`, 39 MB).

Labels: **PROVED** = from code, counters or exact equality; **SIMULATED** = my runs (file named); **CITED** = another report; **ASSUMED** = not checked.

**What ran**
- **Server:** 15 A800 CPU jobs, never more than 2 of mine at a time.
  - 42 staged runs at nS 1000, frozen protocol.
  - 30 lives of 10⁶ ticks.
  - All paired comparisons below are A800 / numpy 2.1, same machine.
- **Local (numpy 1.26, one process at a time):**
  - 27 symbolic probes;
  - 11 short life screens (300k–600k ticks);
  - 10 staged screens (nS 500, seed 0).
  - Local runs are paired only with each other and are marked "local".
- **Mechanics:** 0 violations in every staged run, and 0 overlap and 0 interruptions in every life (PROVED from counters).
- **Determinism:** server runs repeat bit for bit. `base3k` s0 reproduces `base` s0 exactly (PROVED).

## 0. Bottom line

1. **Step 1 is adopted.** The DG k-WTA now breaks exact ties by unit index, using a stable sort, in `rules.py`.
   - New reference, canonical ARCH_v2.1 (5 seeds): **J 1.184 ± 0.081, J2 1.701 ± 0.147**.
   - Life free replies: 0.34 / 0.28 (seeds 5 / 6).
   - The symbolic regression is reproduced exactly: held-out 0.81 (recast tutor) and 0.83 (mixed tutor).
   - **The v21 folder's code was not the reported final config.** Its `agent_v21.py` applied dips to self-initiated speech only, which is v21's rejected `v21fin` variant. v21's rec_fix confirmation (J 1.223) was run with that code. I restored the reported behaviour (dips after every own utterance) as the default, `dip_scope='all'` (PROVED by diffing the job copies).
2. **Step 2 is kept: randmix's `c_flat` inside the v2.1 core.**
   - Staged: J +0.006 ± 0.064 and J2 +0.037 ± 0.090. That is within noise.
   - Generalisation is better on every seed: held-out vs best constant +0.048, RET +0.033.
   - Life: free replies +0.04 / +0.12.
   - Symbolic held-out: 0.81 → 0.92 (recast), 0.83 → 0.89 (mixed).
   - Cost: colour −0.10. The √c_t allocation partly restored colour but gave back the generalisation gain (J2 −0.043 vs flat), so I rejected it.
   - In the final configuration colour is back at the reference level (0.91 vs 0.91).
3. **Step 3, the main result, is a reaction-window reply veto.**
   - **The problem, measured:** canonical v2.1 learns only 0.23 / 0.13 of the partner's corrections in life, because it speaks after 0.77 / 0.87 of them.
   - **The mechanism, `veto_R1`:**
     - a reply to a line that started within 40 ticks (3 s) of my own utterance's end is held back if the learned value V_go of that context is below 0.2;
     - 10 % of held-back replies escape (exploration);
     - the veto is active only while a fast running reward rate is at least 0.15.
   - **Final ARCH_v2.2 life:**
     - free replies **0.47 / 0.52**, against 0.34 / 0.28 for v2.1 and 0.38 / 0.40 for Step 2;
     - corrections learned **0.81 / 0.87**;
     - last tenth of life **0.65 / 0.64**, against the same-life echo-or-constant reference 0.64 / 0.61. That is above the reference in the last tenth on both seeds, but **not over the whole life**.
   - **Life gains kept:** social 0.67 / 0.77, appropriate initiatives 0.48 / 0.49, role confusion 0.39 / 0.48 → 0.26 / 0.24, nonsense initiatives −44 %.
   - **Staged:** **J 1.205 ± 0.015, J2 1.787 ± 0.040**; vs v2.1 +0.020 ± 0.090 (J), vs Step 2 +0.014 ± 0.120.
4. **Step 4: the staged cost of v2.1's value loop is confirmed, and its source is not the dips.**
   - Canonical v2.1 vs Steps 1+2 only (4 seeds): J −0.065 ± 0.058, J2 −0.148 ± 0.101 (lower on every seed), held-out check 0.92 → 0.55.
   - Local single-seed isolation: removing dips does not help. A 4× smaller dip is neutral (+0.015). Restoring learned reply eagerness recovers about half (+0.094).
   - In the final configuration held-out check is back to 0.82. The residual cost against Steps 1+2 is J −0.040 ± 0.043 (mixed), with F −0.042 on every seed.
   - Learned reply eagerness on the final configuration changed nothing (J −0.006 ± 0.057), so it is not adopted.
5. **Step 5: retention collapses between 2,100 and 3,000 untutored dialogues.**
   - Final config (5 seeds): accuracy 0.97 → 0.92 (1,500) → 0.90 (2,100) → **0.16 (3,000)**. The agent then says partner lines (0.48) or stays silent (0.25).
   - Canonical v2.1 collapses too, less on the server: 0.89 (2,100) → **0.54** (3,000, 4 seeds).
   - Locally, both layers collapse: MIX → 0.14, `c_flat` without the veto → 0.00. So the veto does not cause it.
   - Reported only; no consolidation mechanism was built.

**Final ARCH_v2.2 config**
- ARCH_v2.1 final: `ral=1 ral_other=1 say_rec=1 say_loc=1 scene_held=1 val_veto=0.2 dip=1 act_T=60 soc_feats=1 rec_fix=1`, with `dip_scope=all`.
- The canonical DG tie rule (in code, always on).
- `rm=1 rm_train=1 rm_prune=1 rm_age=20000 rm_regrow=1 rm_alloc=flat`.
- `veto_R1=1 veto_W=40 veto_eps=0.1 veto_gate=0.15 veto_fast=0.04`.
- Life adds v2.1's THINK_LIFE, `ct_assoc` 0.5 and `wander` 50.

## 1. Step 1: canonical tie rule and new reference

**Built.** In `rules.Hippo.dg`, `np.argpartition` was replaced by `np.argsort(-d, kind='stable')[:k]`, as gpuport's `run_j_canon.py` does. `test_equiv_v22.py` shows that AgentV22 with every new key at its default equals AgentV21 bit for bit on the v2.1 final config (replies, GEN, hippocampus, unit counts). PROVED; re-run after every change.

**Re-baseline** (`base`, seeds 0–4, A800; SIMULATED, `out/agg_all.txt`):

| run | J per seed | J (mean ± SD) | J2 | F / RET / H / DR |
|---|---|---|---|---|
| v21final (old tie rule, no rec_fix; CITED) | 1.145 / 1.282 / 1.258 / 0.990 / 1.222 | 1.179 ± 0.105 | 1.685 | 0.783 / – / 0.675 / – |
| v21final_rf (old tie rule, rec_fix, self-initiated-only dips, 4 seeds; CITED) | 1.137 / 1.296 / 1.300 / 1.160 | 1.223 ± 0.075 | 1.788 | – |
| **v2.2 reference: canonical, rec_fix, dips after all own speech** | 1.137 / 1.315 / 1.213 / 1.073 / 1.185 | **1.184 ± 0.081** | **1.701** | 0.795 / −0.049 / 0.689 / 0.967 |

Other reference values (SIMULATED):
- drift loss after 300 / 900 untutored dialogues: −0.034 / −0.019;
- role confusion 0.070;
- held-out: agent 0.700 vs best constant 0.535;
- per-task held-out: what 0.80, slot 0.30, where 0.82, offer 0.91, taste 0.55, colour 0.93, count 0.98, check 0.54;
- lesion gateR: B/C/D mean 0.628 vs 0.795 intact.

Reference life (`L_base`, SIMULATED):
- free replies 0.34 / 0.28; last tenth 0.55 / 0.42;
- echo-or-constant reference 0.61 / 0.62;
- social 0.44 / 0.52; role confusion 0.39 / 0.48; appropriate 0.38 / 0.38.

## 2. Step 2: the random-at-birth layer in the v2.1 core

**Built.**
- `core_v22.V22Core(rm=…)` swaps MixDict for randmix's RandMix (`randlayer.py`, copied).
- GEN, the three copy gates, gateR and the critic read its units unchanged.
- The layer's plasticity (instar, use-it-or-lose-it with regrowth) sees the GEN context of each learning event. The say-route lookup is saved and restored around, so it does not change what the layer learns from.
- Item codes come from the agent's own perceptual (Z) templates.
- The RAL trace undoes readout steps only, not the layer's own learning.
- New allocation option `alloc='pow:p'`: population size ∝ c_t^p at the same total.

**Paired, 5 seeds: base → `cf`** (SIMULATED, `out/paired_base_cf.txt`):

| metric | base | c_flat | Δ (seeds) |
|---|---|---|---|
| J / J2 | 1.184 / 1.701 | 1.191 / 1.739 | +0.006 ± 0.064 / +0.037 ± 0.090 (mixed) |
| F / RET / H / DR | 0.795 / −0.049 / 0.689 / 0.967 | 0.780 / −0.016 / 0.731 / 0.944 | RET +0.033 (all up), H +0.042 |
| held-out vs best constant 0.535 | 0.700 | 0.748 | **+0.048, all seeds up** |
| drift loss 300 / 900 | −0.034 / −0.019 | −0.003 / 0.007 | +0.03 (still about 0) |
| held-out where / taste / slot | 0.82 / 0.55 / 0.30 | 0.97 / 0.87 / 0.46 | |
| colour seen / held-out | 0.91 / 0.93 | **0.81 / 0.84** | −0.10 (seed 3: 0.85 → 0.50) |
| check seen | 0.78 | 0.67 | |
| life free replies, s5 / s6 | 0.34 / 0.28 | **0.38 / 0.40** | held-out in life 0.12 / 0.11 → 0.29 / 0.22 |

**Colour fix, √c_t allocation (`sq`, 4 seeds):**
- Against base: J −0.007 ± 0.040, J2 +0.006.
- Colour 0.865, between flat's 0.81 and base's 0.91.
- Against flat it is worse on every seed: H −0.024, held-out vs constant −0.026; J2 −0.043 ± 0.036.
- Life 0.38 / 0.22; seed 6's social replies collapsed to 0.05, though life social is chaotic.
- **Rejected.** The other suggested fix, k_t scaled by input dimensionality, was not tried.
- In the final configuration colour is 0.91 seen (base 0.91) and 0.87 held-out (base 0.93).

**Regression tests** (local; `regress/sym_summary_v22.txt`, SIMULATED):

| probe | tutor | MIX (canonical) | c_flat | √c_t |
|---|---|---|---|---|
| gen_probe_v22, frozen held-out, 3 seeds | recast | 0.81 [0.75 / 0.67 / 1.0], identical to v21 | **0.92** | 0.89 |
| gen_probe_v22, frozen held-out, 3 seeds | mixed | 0.83, identical to v21 | **0.89** | 0.92 |
| gen_probe2 (class-dependent) | – | 0.25 / 0.50 / 0.75 | 0.25 / 0.50 / 0.75 | 0.25 / 0.50 / 0.75 |

No class generalisation from any layer. develop's D1–D3 run inline in every staged run, for example held-out where: attends 1.00, says it 0.83 (base) vs 0.97 (c_flat).

**Units, memory, speed** (SIMULATED):
- 47,925 units and 242,820 fan-in synapses, fixed from birth. MIX grows to about 45.3k units.
- 51–53 % of units are ever active; about 3.0M synapse replacements per staged run.
- About 365 active units per event; lookup cache hit rate 0.44.
- GEN is 50,000 rows × 257 (49 MB) for both layers.
- Peak RSS 701 vs 705 MB.
- Time per staged run 2,633 s vs 1,400 s (1.9×); per life 1,000 s vs 500 s.
- Lesions: undoing the layer's learned fan-in weights at test (`rm_w:1`) drops B/C/D from 0.78 to 0.67, so the layer's learning reaches behaviour.

## 3. Step 3: learning from corrections in life

### 3.1 Diagnosis

Canonical v2.1 lives; `life_diag.py` matches each of the agent's own role resolutions to the world's line types (PROVED from logs).

| line type (seed 5 / seed 6) | learned (committed) | agent speaks within 12 ticks after it |
|---|---|---|
| correction (unrewarded recast after my answer) | **0.23 / 0.13** | 0.77 / 0.87 |
| answer told after my silence | 0.24 / 0.13 | 0.76 / 0.87 |
| statement (facts, 这是X) | 0.34 / 0.23 | 0.66 / 0.77 |
| question | 0.21 / 0.20 (left unanswered) | 0.79 / 0.80 |

- After a correction the agent mostly says the partner's next line (这是什么, 晚安, 这是, 再见…). It repeats the correction exactly in only 0.20–0.23 of cases.

### 3.2 Cues rejected on evidence, before building

All measured with `an/cue_test.py` (`out/cue_test_base.txt`) or read from code.
- **Timing as a role rule.** "Commit if the line starts within 40 ticks of my utterance" catches 1.00 of corrections but also 0.33–0.35 of questions: 266 corrections against about 347 wrongly learned questions on seed 5.
  - The world times both alike: a recast starts at max(question end + 45, my stop + 6), the next probe at max(question end + 45, my stop + 13) (`world.py`; PROVED).
- **Take-up (I repeat the line).** Exact repeat after corrections 0.20–0.23 vs 0.08–0.09 after questions; repeating a piece of it 0.24–0.26 vs 0.21–0.24. It catches a quarter of corrections at about a 1:1 error ratio.
- **Initiation–response–feedback parity** (the line right after my reply is feedback). Rejected from counts alone: 384 / 1,731 questions follow my reply to a probe directly, and 70 % of goodbyes (再见) follow my last answer.
- v21 had already rejected overlap contingency and a reward-trained question detector (CITED).

### 3.3 Behavioural variants: life, MIX core, seeds 5 / 6

SIMULATED (`out/life_table.txt`). Variants are listed in the order tried; the last row is the final ARCH_v2.2.

| variant | free | last tenth | social | corrections learned | role confusion | nonsense | staged J (5 seeds) |
|---|---|---|---|---|---|---|---|
| base | 0.34 / 0.28 | 0.55 / 0.42 | 0.44 / 0.52 | 0.23 / 0.13 | 0.39 / 0.48 | 429 / 477 | 1.184 |
| `u_R1=0` (no urgency after a reaction) | 0.34 / 0.34 | 0.51 / 0.53 | 0.41 / 0.52 | 0.39 / 0.27 | 0.35 / 0.35 | 381 / 456 | – |
| `urg_gate` (learned urgency) | 0.44 / 0.29 | 0.60 / 0.55 | 0.67 / **0.20** | 0.44 / 0.24 | | | – |
| veto on all replies | **0.00 / 0.00** (absorbing silence) | | | | | | – |
| veto on R=1 replies, no window | 0.28 / 0.26 | | **0.25 / 0.17** | 0.97 / 0.99 | 0.23 / 0.27 | 174 / 193 | – |
| + 10 % escape | 0.33 / 0.28 | | 0.28 / 0.24 | 0.89 / 0.88 | | | – |
| **+ 40-tick reaction window (vW)** | **0.50 / 0.48** | 0.67 / 0.59 | 0.71 / 0.70 | 0.87 / 0.89 | 0.25 / 0.28 | 233 / 236 | **0.950 (−0.234, every seed)** |
| vW + fast reward-rate gate (vWf) | 0.51 / 0.46 | 0.64 / 0.58 | 0.74 / 0.70 | 0.85 / 0.78 | 0.23 / 0.30 | 244 / 240 | seed 0: 1.048 vs 1.137 |

**Why the window matters.** Most goodbyes start later than 40 ticks after my last utterance (89–93 %) and so escape the veto. Every correction, confirmation and returned greeting starts within it (PROVED from logs).

**Why the gate matters.** The absolute veto (vW) collapses staged drift: DR 0.97 → 0.55, loss after 900 untutored dialogues 0.44. With `c_flat` (`fin`), accuracy after 3,000 untutored dialogues falls to 0.01. In a world with no rewards V_go goes to 0 and the veto silences every reply that comes soon after the agent's own speech.

Local paired attempts at a fix (nS 500, seed 0; drift after 300 / 900 untutored dialogues; SIMULATED):

| variant | drift 300 / 900 | J |
|---|---|---|
| base (local) | 0.86 / 0.86 | 1.052 |
| threshold relative to the slow reward rate (`veto_rel`) | 0.45 / 0.42 | 0.898 |
| gate on the slow reward rate | 0.75 / 0.77 | 1.002 |
| **gate on a fast reward rate (time constant 25 reactions)** | **0.97 / 0.92** | **1.085** |

**The mechanism kept** (`agent_v22.py`; all inputs perceivable and local):
- **Which replies:** a reply thought to a line that started within 40 ticks after my own utterance ended (efference timing).
- **Rule:** withheld while V_go(thought features) < 0.2. V_go is the existing learned value: pre = the thought's features, post = the partner's rewarded reaction.
- **Escape:** 10 % exploration.
- **Gate:** used only while a fast running reward rate rr ≥ 0.15. rr is updated from the same reaction samples as ThinkAgent's rbar.
- **Effect on listening:** a correction the agent stays silent after is then learned by the unchanged RAL rule ("nobody answered → commit").

### 3.4 Final ARCH_v2.2 (c_flat + gated veto): lives, seeds 5 / 6

SIMULATED (`L_finF`); same-life references from the v2.1 metrics.

| metric | v2.1 (canonical) | Step 2 (c_flat) | **ARCH_v2.2** | same-life reference |
|---|---|---|---|---|
| free replies, whole life | 0.34 / 0.28 | 0.38 / 0.40 | **0.47 / 0.52** | echo-or-constant 0.64 / 0.61 |
| free replies, last tenth | 0.55 / 0.42 | 0.59 / 0.61 | **0.65 / 0.64** | 0.68 / 0.63 for that tenth |
| scene questions | 0.33 / 0.21 | 0.41 / 0.35 | 0.41 / 0.48 | best constant 0.47 / 0.46 |
| held-out | 0.12 / 0.11 | 0.29 / 0.22 | 0.24 / 0.34 | best constant 0.63 / 0.54 |
| social (≥ 0.30 required) | 0.44 / 0.52 | 0.35 / 0.65 | **0.67 / 0.77** | echo 1.00 |
| appropriate initiatives (≥ 0.38 required) | 0.38 / 0.38 | 0.42 / 0.39 | **0.48 / 0.49** (last tenth 0.57 / 0.64) | |
| empty-room 你好 per tenth, last tenth (mean) (≤ 15 required) | 4 / 8 (13.2 / 17.0) | 14 / 10 | 15 / 7 (16.4 / 17.2) | |
| corrections learned | 0.23 / 0.13 | 0.32 / 0.18 | **0.81 / 0.87** | |
| role confusion | 0.39 / 0.48 | 0.38 / 0.37 | **0.26 / 0.24** | |
| nonsense initiatives | 429 / 477 | 333 / 488 | 242 / 267 | |
| memory questions | 0.01 / 0.00 | 0.01 / 0.01 | 0.01 / 0.01 | constant 0.39 |

Learning curve, ARCH_v2.2, free replies per tenth of life (`out/life_curves.txt`):
- seed 5: 0.10, 0.27, …, 0.58, 0.65;
- seed 6: 0.17, 0.40, …, 0.57, 0.64.

**Functional target:**
- **Met in the last tenth of life on both seeds** (0.65 vs 0.64 for that tenth's reference on seed 5, 0.64 vs 0.63 on seed 6) and for scene questions on seed 6.
- **Not met over the whole life** (0.47 / 0.52 vs 0.64 / 0.61), because the first tenths are near 0.1–0.4.
- Social and appropriate are met. Empty-room greetings are met in the last tenth (15 / 7), not as a mean (16.4 / 17.2). The reference's own mean is 13.2 / 17.0, and the first three tenths are high in every configuration.

**Staged, 5 seeds** (SIMULATED, `out/paired_base_finF.txt`, `out/paired_cf_finF.txt`):

| metric | v2.1 base | Step 2 (cf) | **ARCH_v2.2 (finF)** | Δ vs base / vs cf |
|---|---|---|---|---|
| J | 1.184 ± 0.081 | 1.191 ± 0.117 | **1.205 ± 0.015** | +0.020 ± 0.090 / +0.014 ± 0.120 (mixed) |
| J2 | 1.701 | 1.739 | **1.787 ± 0.040** | +0.086 / +0.048 |
| F / RET / H / DR | 0.795 / −0.049 / 0.689 / 0.967 | 0.780 / −0.016 / 0.731 / 0.944 | 0.788 / −0.041 / **0.776** / 0.970 | |
| drift loss 300 / 900 | −0.034 / −0.019 | −0.003 / 0.007 | −0.062 / −0.042 | |
| role confusion | 0.070 | 0.090 | 0.065 | |
| held-out vs best constant 0.535 | 0.700 | 0.748 | 0.759 | |
| held-out check / colour | 0.54 / 0.93 | 0.56 / 0.84 | **0.82** / 0.87 | |
| ritual / colour seen | 0.98 / 0.91 | 1.00 / 0.81 | **0.88** / 0.91 | ritual drops on seed 1 (0.62) |

**Lesions at test** (B/C/D mean; intact 0.788):
- gateR off: 0.589;
- random layer's learned weights undone (`rm_w:1`): 0.676;
- veto off: 0.815. The veto still withholds some answers in frozen probes (a cost of +0.027 at test), and it carries the life gain.

## 4. Step 4: Step 3's staged cost

| comparison | evidence | result |
|---|---|---|
| Steps 1+2 only (`s12`) vs canonical v2.1 | 4 seeds, MIX, SIMULATED | the value loop costs J −0.065 ± 0.058 and J2 −0.148 ± 0.101 (every seed); held-out check 0.92 → 0.55 |
| v2.1 without dips | local, seed 0, nS 500 | J 0.986 vs 1.052: the dips are not the cost |
| dips scaled ×0.25 (`dip_g`) on the gated veto | local, seed 0 | +0.015, noise |
| learned reply eagerness restored (`gain_reply`; v2.1 had fixed G = g0 once the veto was on) | local, seed 0 | +0.094 |
| `gain_reply` on the final config | 4 seeds, server | J −0.006 ± 0.057, lives 0.45 / 0.52 vs 0.47 / 0.52: not adopted |
| final config vs Steps 1+2 | 4 seeds, server | J −0.040 ± 0.043 (mixed), F −0.042 (every seed), held-out check 0.82 vs 0.92 |

- Dips fire only 2–14 times per staged run (PROVED from counters).
- **Verdict:** in the final configuration the cost is within noise on J and J2 and the life gains are kept. A small cost on F remains.
- `dip_scope='pred'` (a dip only where a learned reaction predictor expected a reaction) was built but **not tested**.

## 5. Step 5: long untutored use, on the final config

5 seeds on the server; base3k is canonical v2.1 on 4 seeds (SIMULATED).

| dialogues untutored | 0 | 300 | 900 | 1,500 | 2,100 | 3,000 |
|---|---|---|---|---|---|---|
| ARCH_v2.2, C accuracy | 0.908 | 0.970 | 0.950 | 0.921 | 0.898 | **0.162 ± 0.11** |
| ARCH_v2.1 (base3k) | 0.921 | 0.967 | 0.954 | 0.927 | 0.895 | **0.543** (0.87 / 0.55 / 0.24 / 0.52) |
| v2.2 with the absolute veto (`fin`) | 0.91 | 0.57 | 0.47 | 0.28 | 0.12 | 0.01 |

- **Shape of the failure:** at 3,000 the agent says partner lines (0.39–0.68) or stays silent (0.20–0.34).
- **Local single-seed attribution** (nS 500): MIX 0.86 → 0.76 (2,100) → 0.14 (3,000); `c_flat` without the veto 0.67 → 0.51 → 0.00.
- **So:**
  - a late collapse is already in ARCH_v2.1;
  - it is worse with the random layer;
  - it is not caused by the veto.
- **Possible causes** (ASSUMED, not tested):
  - the 10k-cell hippocampus fills with untutored episodes, and awake recall then replays partner → partner transitions;
  - unanswered questions accumulate as learned continuations;
  - the layer's pruning age (20,000 learning events, about 2,000 dialogues) coincides with the onset.
- As instructed, no consolidation mechanism was built.

## 6. Final ARCH_v2.2 config

- ARCH_v2.1 final, with `dip_scope='all'`.
- The canonical DG tie rule (code).
- **Step 2:** `rm=1 rm_train=1 rm_prune=1 rm_age=20000 rm_regrow=1 rm_alloc=flat`.
- **Step 3:** `veto_R1=1 veto_W=40 veto_eps=0.1 veto_gate=0.15 veto_fast=0.04`.
- **Not kept:**
  - √c_t allocation (`pow:0.5`);
  - `u_R1=0` and `urg_gate`;
  - the veto on all replies, without the window, with the absolute threshold, with the relative threshold, or with the slow gate;
  - `gain_reply` and `dip_g=0.25`.
- **Principles:**
  - one life with learning always on; no sleep;
  - speech still caused by thoughts (the veto only withholds);
  - local rules only (efference timing, a learned value, one reward broadcast, the agent's own noise);
  - no world state read: the diagnostics read the world log only after the run;
  - 0 mechanics errors;
  - every new part lesioned: the veto, the layer's learned weights, gateR.

## 7. Open problems, ranked

1. **Collapse after about 2,100–3,000 untutored dialogues** into partner lines, in v2.1 and worse in v2.2. This is the next mechanism to design (consolidation or protection).
2. **Whole-life accuracy is still below the reference.** Life free replies are 0.47 / 0.52 against 0.64 / 0.61, reached only late in life.
   - Memory questions stay at 0.01: statements are still cancelled in 0.57–0.64 of cases because the agent talks after them.
   - Held-out in life is 0.24 / 0.34 vs constants 0.63 / 0.54.
3. **The role rule still has no content cue.** The veto fixes the behaviour (the agent stays quiet after reactions), but RAL still cannot tell a correction from a question by content. The class codes from the other agent's Step 4 are the content route.
4. **Small residual staged costs:** F −0.04 vs Steps 1+2; the veto withholds some frozen-probe answers (+0.027 when lesioned at test); ritual 0.88.
5. **Life social replies are chaotic per seed.** Goodbye accuracy alone ranges 0.02–0.81 across configurations, and goodbyes are about 18 % of free items. Two seeds cannot resolve differences under about 0.05.
6. **Colour** with `c_flat` varies by seed: k_t by input dimensionality was not tried.
7. **Speed:** the random layer is 1.9× slower in numpy, so the GPU port matters.
8. **Step 4 dips:** the 5-seed server test of the dip size on the final config was not run (local evidence only), and `dip_scope='pred'` is untested.

## 8. Files (this folder; results in `results/`)

- **Code:**
  - `core_v22.py`, `agent_v22.py`, `randlayer.py` (with `pow:p` allocation), `rules.py` (canonical tie rule);
  - patched copy of `agent_v21.py` (core factory, `dip_scope`);
  - `run_v22.py` (`rm_*` keys, `drift3`, `rm_w` lesion, timings, RSS), `life_v22.py` + `life_diag.py`;
  - `test_equiv_v22.py`, `submit.py`, `paired22.py`, `agg22.py`, `life_cmp.py`;
  - `an/timeline.py`, `an/cue_test.py`, `an/curve.py`;
  - `regress/gen_probe_v22.py`, `regress/gen_probe2_v22.py`, `regress/sym_v22.sh`, `regress/sym_sq.sh`.
- **Results:**
  - staged run files `out/runs/*.json` (local ones are prefixed `loc_`);
  - lives `out/life/*.json` and their logs `*_log.pkl`;
  - tables `out/agg_all.txt`, `out/paired_*.txt`, `out/life_table.txt`, `out/life_table_local.txt`, `out/life_curves.txt`, `out/cue_test_base.txt`;
  - `regress/sym_summary_v22.txt`, `regress/*_out.jsonl`;
  - job logs `out/logs/*`;
  - job folders `arch/v2/gpu/queue_a800/v22_*`.

## summary_zh

1. **第1步**：海马齿状回的k-WTA平局改为按编号稳定排序。重新跑了ARCH_v2.1（5个种子）：J 1.184，J2 1.701，作为新基准。
   - 发现v21文件夹里的代码其实是被否决的"只对主动说话做失望"版本，已改回报告中的版本。
2. **第2步**：把"出生即随机连接的组合层"（等量分配）放进v2.1，保留。
   - 测试台J不变（在噪声内），但泛化在每个种子上都更好。
   - 一生回答正确率从0.34/0.28提高到0.38/0.40。
   - 颜色题有损失。按√c分配能补回一部分颜色，但丢掉了泛化收益，所以没采用。
   - 最终配置里颜色已回到基准水平。
3. **第3步（主要成果）**：
   - 原来孩子在大人纠正后77–87%会马上说话，于是纠正被当成"问题"撤销，只学到13–23%。
   - 计时、重复对方的话、话轮奇偶这几个线索都用数据排除了。
   - 有效的办法是：对方在我说完后3秒内说的话，如果过去在这种情况下接话从没受欢迎（学到的价值低），就不接话。另加10%探索。
   - 只在近期有奖励的环境里才启用，避免在没人反馈时学成沉默。
   - 结果：纠正被学到81–87%，一生回答0.47/0.52，最后十分之一0.65/0.64，超过同一生"鹦鹉或固定答案"参考（0.64/0.63）。整个一生平均仍低于参考。
   - 社交0.67/0.77，合适的主动说话0.48/0.49，角色混淆降到0.25左右。
   - 测试台J 1.205，没有代价。
4. **第4步**：v2.1价值回路的测试台代价确认存在（"是不是"留出题0.92→0.55），但原因不是"失望"步长。最终配置里这个代价基本消失（留出0.82）。
5. **第5步**：无人指导使用到2,100段对话时都还好；到3,000段时最终配置跌到0.16（开始说对方的话）。v2.1本身也会跌（到0.54）。按要求只报告，未做巩固机制。
