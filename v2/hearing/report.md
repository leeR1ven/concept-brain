# Open-world hearing for ARCH_v2.1 (Step 6, first part): report

Everything is in this folder: `design.md` (written before building, with a §5 listing the revisions that the tests forced), the code, and the results under `results/`. The code runs on the repo's `../v21` (ARCH_v2.1) and `../synth` (the unchanged ARCH_v2 core and its bench copy); world2 is `../bench/world2`.

Labels: **PROVED** = argument, enumeration or exact identity test. **SIMULATED** = my run, file named. **CITED** = a paper or another report. **ASSUMED** = not checked.

## 0. Bottom line

1. **The agent now hears in noise, and the bench is untouched.**
   - **Bench, clean (SIMULATED, A800, 5 seeds, frozen protocol, canonical DG tie rule).** v2.1 + hearing reproduces v2.1 bit for bit: same replies and same MIX on every seed, ΔJ = ΔJ2 = 0.0000 on all 5 seeds, J 1.2291 in both arms, 0 mechanics violations.
   - **Bench at its own noise level σ = 1 (SIMULATED, seeds 0–2).**
     - v2.1 scores J 0.000: it hears nothing (0 MIX units, 0 episodes).
     - v2.1 + hearing scores J 1.200 / 1.302 / 1.321, against 1.176 / 1.340 / 1.283 clean (mean 1.274 vs 1.266), with 0 violations.
     - Quiet moments 8,682 vs 8,681 clean; awake recalls 69,435 vs 69,427.
2. **World2 noise is solved (SIMULATED).**
   - Ear alone, 2 bodies, σ 0.5 / 1: v2.1 finds 6 / 1 of 130 lines, fills 256 categories (218 / 250 junk) and gets 0 quiet moments.
   - Hearing finds 130 / 129 of 130 lines with near-onset recall 0.99. It forms 66 categories, the same as at σ 0, with 0 junk and purity 1.00. Syllable coverage is 1.00 and quiet moments are 123 per day (122.5 at σ 0).
   - Full v2.1 agent at σ 1 (1 day, 2 agents): v2.1 has 0 lines, 0 episodes, 0 recalls and 0 utterances. With hearing: 330 lines, 327 episodes, 2,257 recalls and 193 utterances, against 304 / 304 / 2,177 / 201 noise-free. Free accuracy is 0.16 vs 0.185, with 0 talk-over-partner.
3. **Crowds are partly solved (SIMULATED).** v2.1 never segments a line that starts while someone else is audible (recall 0.00 for those lines).
   - The new syllable clock follows recognition, and a change of syllable rhythm counts as a new speaker.
   - Near-line syllable coverage, v2.1 → hearing:
     - 16 bodies, light chatter: 0.85 → 0.995;
     - 16 bodies, heavy chatter: 0.79 → 0.97;
     - 64 bodies: 0.65 → 0.86.
   - For near lines with overlapped starts: 0.22 / 0.06 / 0.14 → 0.97 / 0.77 / 0.74.
   - Junk categories −31 to −57 %, duplicate categories −35 to −80 %.
4. **One unbroken life (SIMULATED).** The noise level changed with no reset (σ 0 → 1 → 0.5 → 1).
   - Ear alone over 10 days: the floor followed (μ 0 → 6.1 → 2.6 → 6.1), near recall was 0.88–0.98 every day and near coverage 0.92–1.00.
   - Full agent over 4 days: it heard and spoke every day, with 0 talk-over.
   - v2.1 went deaf on day 3 and stayed deaf.
5. **Not solved:**
   - the brake F in crowds is unchanged (on 16 / 31 / 52 % of the day), and agents barely speak in heavy crowds, with v2.1 or with hearing;
   - far speech in noisy crowds fragments (line precision 0.62–0.74);
   - σ = 2 works only in part (near recall 0.84, coverage 0.66);
   - with a 207-character lexicon, crowds still push the category cap of 256;
   - trust-weighted listening is not built.

## 1. What was built

Everything sits behind `h_*` keys. Every key off = v2.1, which `test_equiv_h.py` and the 5-seed A800 runs confirm. The code is a wrapper class `AgentH(AgentV21)` (`agent_h.py`) plus one module (`hear.py`). Only the front end and its triggers change; `core_v21` is not touched, so it merges as a wrapper over v22's AgentV21.

| key | mechanism | basis |
|---|---|---|
| `h_gate` | **Adaptive noise floor and a sound detector with a fixed false-alarm rate.** One unit tracks μ (mean) and ν (mean absolute deviation) of e2, the energy of the 2 newest slices. It learns over background ticks only (τ 500 ticks), and only from pairs ≤ μ + 4ν. It is calibrated once at birth from the warm-up stream. A pair holds sound if e2 > μ + 7ν, or > μ + 4ν while the previous pair was sound. The agent reads the window only while it holds sound, and zeros otherwise. **Every v2.1 trigger becomes floor-relative with no trigger code changed**: onset, syllable clock, line end, quiet moment, wandering, the brake F, and the eye parser's silence test. A "stuck" rule handles a background that rises after birth: if sound seems to last more than 600 ticks, the floor also adapts during sound. | background adaptation in the auditory system (CITED: Dean, Harper & McAlpine 2005; Rabinowitz et al. 2011, 2013) |
| `h_relock` | **The syllable clock follows recognition.** At every tick inside a line the window is matched to the categories. If the sample window is not recognised (< 0.75) but one of the other two phases was (≥ 0.85), the clock moves to that phase and that window becomes the item. | theta entrainment where syllables are parsed (CITED: Giraud & Poeppel 2012) |
| `h_vbound` | **Voice-change boundary.** A relock ends the line and starts a new one. In world2 every speaker talks at the same rate, so a second speaker shows up as a second phase. | onset asynchrony separates streams (CITED: Bregman 1990) |
| `h_nocreate` | No new categories in a line that showed a second rhythm. | – |
| `h_conf` | **New categories only from clearly heard windows:** all 3 fresh pairs above the onset threshold. | – |
| `h_level` | **Level-invariant recognition.** A window that matches nothing is tried again 1–3 levels louder (signal cells only); a match ≥ 0.85 counts as that category. World2's distance subtracts levels, so this undoes distance. | level-invariant coding in auditory cortex (CITED: Sadagopan & Wang 2008; Barbour 2011) |
| `h_denoise` | **Syllable patterns from the denoised window.** Cells at background level are silenced before ear layer 0: cells ≤ ceil(5 × mean background per cell), which is 0 on the bench and 2 at σ 1. The triggers still read the gated raw window. The babbling inverse map compares like with like: heard syllables are matched against own-voice memories denoised the same way. Without this, utterances at σ 1 fell from 201 to 92. | the same per-channel adaptation |
| `h_back` | onset back-tracking (off) | **rejected**: noise outliers misaligned near lines |

All of it is local: one adapting unit (μ, ν), template matching the ear already does, and one clock counter. Nothing reads world state; the warm-up windows are the allowed birth input. No sleep, no reset, no puppeting.

## 2. Ear module alone, world2

Setup (SIMULATED; `record.py`, `ear_eval.py`, `ear_table.py`): seed 0, 1 day, frozen PFC, listeners silent. "Chatter" bodies say 2–4 random lexicon characters after 10 quiet ticks; light = p 0.004, heavy = p 0.02 (≈ 7 utterances per 1,000 ticks).

Hearing = `h_gate h_conf h_relock h_vbound h_nocreate h_level h_denoise`. Coverage = the share of a near line's syllables sampled aligned and recognised.

| N, chatter, σ | front end | lines found / true | precision | near recall | near coverage (overlapped starts) | categories (junk / dup) | quiet moments/day | F on |
|---|---|---|---|---|---|---|---|---|
| 2, –, 0 | v2.1 | 128 / 130 | 1.00 | 0.99 | 1.00 (1.00) | 66 (0 / 0) | 122.5 | 0.08 |
| 2, –, 0 | hearing | 128 / 130 | 1.00 | 0.99 | 1.00 (1.00) | 66 (0 / 0) | 122.5 | 0.08 |
| 2, –, 0.5 | v2.1 | 6 / 130 | 0.00 | 0.00 | 0.28 | 256 (218 / 0) | 0 | 0.29 |
| 2, –, 0.5 | hearing | 130 / 130 | 0.99 | 0.99 | 1.00 (1.00) | 66 (0 / 0) | 123.5 | 0.08 |
| 2, –, 1 | v2.1 | 1 / 130 | 0.00 | 0.00 | 0.07 | 256 (250 / 0) | 0 | 0.99 |
| 2, –, 1 | hearing | 129 / 130 | 1.00 | 0.99 | 1.00 (1.00) | 66 (0 / 0) | 123 | 0.08 |
| 2, –, 2 | hearing | 180 / 130 | 0.61 | 0.84 | 0.66 | 44 (6 / 0) | 123 | 0.07 |
| 16, light, 0 | v2.1 | 392 / 434 | 1.00 | 0.83 | 0.85 (0.22) | 256 (42 / 37) | 262 | 0.17 |
| 16, light, 0 | hearing | 414 / 434 | 0.98 | 0.96 | 0.995 (0.97) | 225 (18 / 7) | 262 | 0.17 |
| 16, light, 1 | v2.1 | 1 / 434 | 0 | 0 | 0.01 | 256 (246 / 0) | 0 | 0.99 |
| 16, light, 1 | hearing | 444 / 434 | 0.70 | 0.95 | 0.99 (0.93) | 204 (38 / 4) | 264 | 0.16 |
| 16, heavy, 0 | v2.1 | 865 / 1036 | 0.98 | 0.77 | 0.79 (0.06) | 256 (58 / 22) | 418 | 0.38 |
| 16, heavy, 0 | hearing | 979 / 1036 | 0.91 | 0.91 | 0.97 (0.77) | 248 (40 / 5) | 418 | 0.38 |
| 16, heavy, 0.5 | hearing | 990 / 1036 | 0.89 | 0.93 | 0.97 (0.83) | 249 (41 / 5) | 419 | 0.38 |
| 16, heavy, 1 | v2.1 | 1 / 1036 | 0 | 0 | 0.03 | 256 (237 / 0) | 0 | 0.99 |
| 16, heavy, 1 | hearing | 920 / 1036 | 0.62 | 0.86 | 0.86 (0.74) | 256 (70 / 5) | 398 | 0.31 |
| 64, heavy, 0 | v2.1 | 1256 / 1870 | 0.97 | 0.88 | 0.65 (0.14) | 256 (81 / 24) | 269 | 0.60 |
| 64, heavy, 0 | hearing | 1406 / 1870 | 0.91 | 0.93 | 0.86 (0.74) | 256 (56 / 16) | 269 | 0.60 |
| 64, heavy, 1 | v2.1 | 1 / 1870 | 0 | 0 | 0.03 | 256 (210 / 3) | 0 | 0.99 |
| 64, heavy, 1 | hearing | 1503 / 1870 | 0.74 | 0.93 | 0.78 (0.74) | 256 (69 / 8) | 295 | 0.58 |

- **Silences ≥ 200 ticks are unchanged** from the noise-free value (33 / 16 / 0.25 / 0 per day). In crowds they do not exist, and nothing here creates them.
- **Seed-1 replicate (SIMULATED).** 2 bodies at σ 1: near recall 0.99, coverage 0.995. 16 bodies, light chatter, σ 1: 0.97 / 0.99. 16 bodies, heavy chatter, σ 1: 0.95 / 0.92. v2.1 is deaf in all of them.
- **Why each piece is there (SIMULATED, ablation sequence in `results/eval_*.log`):**
  - **gate alone:** fixes noise for 2 bodies, but in crowds + noise the untrimmed floor ran away (μ 10–20) and the agent went deaf again (near recall 0.01). The trim fixed this.
  - **relock:** near coverage of overlapped lines 0.06 → 0.85 (16 bodies, heavy).
  - **conf:** at 2 bodies, σ 1, junk 12 → 0.
  - **level:** duplicates 34 → 5 (16 bodies, heavy).
  - **denoise:** in noisy crowds, aligned windows matched their own templates at a median of only 0.78 (0.96 clean). With denoising, coverage rose 0.71 → 0.86 (heavy) and 0.88 → 0.99 (light), and duplicates fell 31–46 → 4–5.

## 3. The full v2.1 agent in world2

Setup (SIMULATED; `w2_agent.py`, `results/w2/`): world2 seed 0, 1 day, 2 full agents (v2.1 final life config) plus chatter bodies. Values are summed over the 2 agents; "true near cg" = caregiver lines that reached the agents at attenuation 0.

| world | front end | true near cg | lines | categories | quiet moments | awake recalls | episodes | utterances | F on | talk-over partner | free acc (n) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2, σ 0 | v2.1 | 312 | 303 | 60 / 69 | 285 | 2177 | 303 | 202 | 0.09 | 0 | 0.185 (108) |
| 2, σ 0 | hearing | 312 | 304 | 60 / 63 | 285 | 2177 | 304 | 201 | 0.09 | 0 | 0.185 (108) |
| 2, σ 1 | v2.1 | 259 | **0** | 256 / 256 | 0 | 0 | 0 | **0** | 0.99 | 0 | 0.00 (82) |
| 2, σ 1 | hearing | 332 | 330 | 79 / 85 | 295 | 2257 | 327 | 193 | 0.10 | 0 | 0.161 (100) |
| 16 light, σ 0 | v2.1 | 67 | 666 | 256 / 256 | 561 | 4412 | 660 | 50 | 0.15 | 0 | 0.04 (22) |
| 16 light, σ 0 | hearing | 47 | 753 | 204 / 248 | 588 | 4628 | 753 | 71 | 0.16 | 0 | 0.19 (16) |
| 16 light, σ 1 | v2.1 | 52 | 0 | 256 / 256 | 0 | 0 | 0 | 0 | 0.99 | 0 | 0 (24) |
| 16 light, σ 1 | hearing | 44 | 714 | 168 / 248 | 568 | 4446 | 625 | 115 | 0.14 | 0 | 0.00 (21) |
| 16 heavy, σ 0 | v2.1 | 58 | 1505 | 256 / 256 | 1089 | 8630 | 1446 | 20 | 0.31 | 0 | 0.11 (19) |
| 16 heavy, σ 0 | hearing | 46 | 1656 | 216 / 256 | 1077 | 8534 | 1656 | 18 | 0.31 | 0 | 0.11 (17) |
| 16 heavy, σ 1 | v2.1 | 42 | 0 | 256 / 256 | 0 | 0 | 0 | 0 | 0.99 | 0 | 0 (16) |
| 16 heavy, σ 1 | hearing | 61 | 1611 | 241 / 256 | 1070 | 8461 | 1403 | 159 | 0.28 | 0 | 0.04 (22) |
| 64 heavy, σ 0 | v2.1 | 18 | 2458 | 256 / 256 | 878 | 6979 | 2160 | 0 | 0.52 | 0 | 0 (9) |
| 64 heavy, σ 0 | hearing | 18 | 2663 | 256 / 256 | 878 | 6979 | 2659 | 0 | 0.52 | 0 | 0 (9) |

- **Seed-1 replicate (hearing, 2 bodies), σ 1 vs σ 0:** lines 316 / 317, episodes 301 / 317, recalls 2,015 / 2,121, utterances 166 / 159, talk-over 0.
- **Lives diverge.** At σ 0 the hearing keys change behaviour slightly in crowds (relock and level matching fire), so the lives differ.
- **Free accuracy after 1 day rests on 9–108 items.** It is not a learning result.
- **Inverse map under denoising.** At σ 1, 85–100 % of categories get a speech program (e.g. 139 / 146), against about 15–25 % in clean crowds for both v2.1 and hearing (e.g. 28 / 35, 54 / 46).

## 4. Bench regression

Setup (SIMULATED; jobs `hearing_benchA/B/N` on the A800, numpy 2.1, Python 3.11; copies in `results/bench/`): v2.1 final config, frozen protocol, `run_h_canon.py` = run_v21 with the canonical tie rule from gpuport.

| seed | J, v2.1 | J, + hearing | ΔJ | ΔJ2 | replies / MIX | hearing counters | J at σ 1, + hearing | J at σ 1, v2.1 |
|---|---|---|---|---|---|---|---|---|
| 0 | 1.1764 | 1.1764 | 0 | 0 | identical / 45,540 | all 0; max sound run 29 | 1.200 | **0.000** |
| 1 | 1.3397 | 1.3397 | 0 | 0 | identical / 45,269 | all 0; 26 | 1.302 | – |
| 2 | 1.2832 | 1.2832 | 0 | 0 | identical / 45,282 | all 0; 23 | 1.321 | – |
| 3 | 1.1059 | 1.1059 | 0 | 0 | identical / 45,345 | all 0; 23 | – | – |
| 4 | 1.2401 | 1.2401 | 0 | 0 | identical / 46,232 | all 0; 29 | – | – |

- **J2, clean.** 1.744 / 1.984 / 1.911 / 1.551 / 1.803 in both arms.
- **Noisy bench, seeds 0–2.**
  - J2 1.790 / 1.921 / 1.976.
  - H 0.786 / 0.824 / 0.873, against 0.757 / 0.859 / 0.837 clean.
  - Drift after 900 untutored dialogues is unchanged.
  - The noisy-bench arm ran only seeds 0–2, to fit one job wave.
- **Local identity test** (`test_equiv_h.py`, all keys on, including `h_back`): EQUIVALENT.
  - 258 replies identical; MIX 23,036; hippocampus 564; 67 categories; GEN weights identical.
  - Every hearing counter is 0.

## 5. One unbroken life with changing noise

**Ear alone, 10 days** (SIMULATED; 16 bodies, light chatter, σ by day 0 / 1 / 0.5 / 1 from days 0 / 3 / 6 / 8; `life_eval.py`, `results/life_*.log`):
- **Floor:** μ = 0 → 6.06–6.15 → 2.57–2.62 → 6.10–6.15, with no reset.
- **Near lines:** recall 0.88–0.98 and coverage 0.92–1.00 on every day.
- **Quiet moments:** 242–273 per day.
- **Categories:** new junk per day 7.5 / 2 / 7.5 / 16.5 (the first noisy day) / 6 / 3.5, then 0. The cap of 256 was reached on day 5.
- **v2.1:** at the cap on day 0 (32 junk categories); from day 3 on, 0 lines and 0 quiet moments.

**Full agent, 4 days** (SIMULATED; same world, σ 0 / 1 / 0.5 / 1 by day):
- **Floor:** followed 0 → 6.07 → 2.6 → 6.1. At the 0 → 1 switch the stuck rule fired (36 / 39 updates, longest sound run 636 / 639 ticks) and the agent recovered by itself.
- **Lines per day (agent 1 / agent 2):** 253 / 500, 339 / 567, 268 / 464, 288 / 466.
- **Utterances per day:** 27 / 44, 23 / 33, 47 / 57, 42 / 69.
- **Mechanics:** 0 talk-over, 0 garble.
- **Categories:** at the cap by days 2–3.

## 6. The three attacks

1. **Is the bench still segmented exactly as now?** Yes.
   - **PROVED.** Bench warm-up silence is exactly 0, so μ = ν = 0 and stay 0. With a threshold of 0, "pair holds sound" ⇔ e2 > 0, and every bench speech pair has e2 ≥ 21 (enumeration over 207 characters × 7 voices, `results/levels.json`). So the gated window equals the raw window on every tick.
   - **Not provable, but measured.** The relock, level-matching and creation-gate paths cannot be proved silent. Over 5 full bench lives they fired 0 times, and the replies are identical.
   - **Why relock is safe.** Misaligned windows never reached a match of 0.85 in any world2 recording (2,300–2,500 per condition).
2. **Do categories still form?** Yes.
   - In noise: exactly the noise-free number (66) with purity 1.00.
   - In crowds: fewer junk and duplicate categories than v2.1.
   - Remaining limit: with chatter using all 207 lexicon characters, 207 legitimate categories plus junk still reach the cap of 256. Junk is never removed.
3. **Is it stable over one life?**
   - The floor follows noise up and down without a reset.
   - The trimmed estimator converges from a calibrated start and from a zero start at σ 0.5 / 1 / 2 (SIMULATED, pure-noise test).
   - The untrimmed and 2.5ν-trimmed versions were unstable (runaway / collapse) and were rejected.

## 7. What failed or stays open

1. **The brake F in crowds is unchanged:** on 0.16 / 0.31 / 0.52 of the day for 16-light / 16-heavy / 64 bodies, against the embodied reviewer's target ≤ 0.15.
   - In heavy crowds both front ends speak about 18–20 times a day for 2 agents, and 0 times at 64 bodies.
   - No fixed level separates a = 0 from a = 1 (char energy 85–123 vs 57–91, PROVED), and the onset tick of a near line must brake.
   - A near-only brake needs a learned or relative loudness reference. Not built.
2. **Far speech in noisy crowds** fragments: line precision 0.62–0.74, far recall 0.41–0.51, and 38–70 junk categories.
3. **σ = 2**: near recall 0.84, coverage 0.66.
4. **Category cap (V = 256) and no recycling.** This is a core-side dimension.
5. **Binding in clean crowds is low (15–25 %)** with v2.1 too. Denoised (peaks-only) binding binds 85–100 %, so a level-normalised binding is probably the fix. It changes the bench and needs its own J test.
6. **Trust-weighted listening is not built.** Measured: nearness is a perfect adult cue at 16 bodies (100 % of near lines are caregivers), but fails at 64 (87 % of near lines are peers). Trust needs voice identity (v21's `voice` hook, untested in world2).
7. **`h_back` rejected** (noise outliers just before onsets).
8. **Not run:** full 10⁶-tick bench life mode.

**Bench bugs:** none found. One note: world2's `warmup_windows()` includes the world's noise, which the calibration relies on.

## 8. Predictions (design.md §4) and the embodied reviewer's

| prediction | result |
|---|---|
| 2 bodies, σ 0.5 / 1: lines and quiet moments within 15 %, near recall ≥ 0.95, categories ≤ 1.15 ×, purity ≥ 0.98 | **met:** 130 / 129 lines, 123 quiet moments, recall 0.99, 66 = 66 categories, purity 1.00 |
| σ 2 is a partial failure | as predicted (0.84 / 0.66) |
| Crowd + σ 1 within 0.1 of the same crowd at σ 0 | **near lines met** (recall 0.95 / 0.86 / 0.93 vs 0.96 / 0.91 / 0.93); coverage 0.11 short at 16 heavy; **precision and far recall failed** |
| Relock + boundaries + no-create cut junk by ≥ 30 %, overlapped coverage ≥ 0.5 | **met:** −31 to −57 %; 0.74–0.97 |
| Full agent at σ 1: lines/episodes within 15 %, recalls ≥ 80 %, utterances ≥ 50 %, categories < 100 after half a day, 0 talk-over | **met:** +9 % / +8 %, 104 %, 96 %, 79 / 85 after a full day, 0 |
| Bench ΔJ = 0 if no relock fires | **met:** 0 relocks, ΔJ = ΔJ2 = 0 on 5 seeds |
| Reviewer: F ≤ 0.15 with 16 chattering bodies | **failed:** 0.31 with heavy chatter (0.16 light) |
| Reviewer: if σ 1 gives fewer than half the noise-free lines, the floor tracker is wrong | not the case |

## 9. Config to merge

`h_gate=1 h_conf=1 h_relock=1 h_vbound=1 h_nocreate=1 h_level=1 h_denoise=1`, with the defaults `h_kon 7 h_koff 4 h_tau 500 h_trim 4 h_stuck 600 h_lock 0.85 h_lvth 0.85`; `h_back` stays 0.
- **Merge path:** subclass v22's AgentV21 with `AgentH`, or copy `hear.py` plus the `step`, `_detect`, `_bind_from_memory` and `babble` hooks.
- **Driver:** `run_h_canon.py` accepts the keys.

## 10. Files

All in this folder:
- **Code:** `hear.py`, `agent_h.py`, `run_h_canon.py`, `test_equiv_h.py`, `w2_agent.py`, `record.py`, `ear_eval.py`, `life_eval.py`, `levels.py`.
- **Diagnostics:** `diag*.py`, `onset_cues.py`.
- **Tables:** `ear_table.py`, `w2_table.py`, `bench_table.py`.
- **Job tooling:** `submit_h.py`, `jobs/`.
- **v2.1 code:** the agent worked on an unchanged copy of `../v21` + `../synth` (`v21copy/`, not committed; the repo files point at `../v21` and `../synth`).
- **Results:** `results/levels.json`; `results/ear/*.json` and `results/eval_*.log` (every variant); `results/w2/*.json`; `results/life*`; `results/bench/*.json`.
- **Recordings:** `rec/` (51 MB of world2 pickles, not committed; `record.py` regenerates them).

## summary_zh

- **问题**：v2.1 的耳朵只认"完全安静"。world2 一有噪声（σ=1）就一句也听不到，基准测试在 σ=1 时 J=0。很多人同时说话时，别人说话中间开始的句子一句也分不出来。
- **做法**（全部是局部、类脑的机制，默认关闭，开关在 `h_*` 键里）：
  - 自适应噪声底：一个单元记住背景的平均值和波动，"有声"等于明显高于背景；
  - 只从听清楚的声音里学新音节；
  - 音节节拍跟着"认出来的音节"重新对齐，节拍一变就当作换了人说话；
  - 远处变小的音节放大后再认；
  - 先去掉背景噪声再认音节，咿呀学语的记忆也用同样方法比较。
- **结果**：
  - 安静的基准测试 5 个种子与原来逐位相同，J 和 J2 完全不变。
  - 基准测试在 σ=1 下 J 从 0 恢复到 1.27（安静时 1.27）。
  - world2 噪声下听到的句子、回忆、说话次数都和安静时差不多，类别数相同、零垃圾，从不抢话。
  - 人多时，句子开头被盖住的近处说话从 0 变成大部分能听清。
  - 一生中噪声变来变去，不用重置也能跟上。
- **没解决**：
  - 人多时"别人在说话"的刹车照旧，孩子在人群里几乎不说话；
  - 远处的话在噪声里会碎；
  - σ=2 只能部分听清；
  - 类别上限 256 在人多时仍会被占满；
  - 按信任程度加权的倾听还没做（它需要识别说话人的声音）。
