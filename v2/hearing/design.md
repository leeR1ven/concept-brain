# Open-world hearing for ARCH_v2.1: design (Step 6, first part)

Folder: `arch/v2/hearing/`. Labels: PROVED (argument or enumeration with numbers), SIMULATED (my run; file named),
CITED (paper or another report), ASSUMED.

## 0. The problem in one paragraph

Every event of the v2.1 front end is defined by **exact silence of the raw ear window**: line onset `E > 0 after
E == 0`, the syllable clock (sample every 3rd tick from the onset), line end `E == 0`, the quiet moment for awake
recall (12 ticks `E == 0`), wandering (200 ticks), and the body's brake F (`E >= 10`) (agent_think.py, agent_synth.py,
body_think.py; PROVED from code). World2's noise is `max(signal, clip(rint(N(0, σ)), 0, 10))` per cell, so with σ > 0
the window is never 0: one endless line, a free-running syllable clock, the category cap filled with noise, no quiet
moment, F on all day (embodied reviewer, SIMULATED; reproduced below). Crowds remove silence differently: other
bodies' speech overlaps, so lines that start while someone else is audible are never segmented.

## 1. Facts the design rests on

**Exact level facts (PROVED by enumeration, `levels.py` → `results/levels.json`; 207 lexicon chars × 7 voices).**

| quantity | value |
|---|---|
| fresh-pair energy e2 (2 newest slices, 16 cells) of any char, near speaker (a = 0) | min 21, mean 34 |
| e2 at attenuation a = 1 / 2 / 3 / 4 | min 15 / 11 / 7 / 3; mean 24 / 20 / 16 / 12 |
| aligned full-char window energy E, a = 0 / 1 / 2 | 85–123 / 57–91 / 43–79 |
| noise e2, σ = 0.5 / 1 / 2: mean (MAD) | 2.6 (1.2) / 6.1 (2.0) / 12.6 (3.8) |
| P(noise e2 ≥ 20), σ = 1 | 6.5·10⁻⁶ per tick |
| P(noise window E == 0), σ = 1 | 2·10⁻⁸ (why v2.1 hears nothing) |

Consequences:
- On the bench (no noise) every sounding fresh pair has e2 ≥ 21 and every silent one e2 = 0. A detector with any
  threshold in [0, 21) gives exactly the same events as `E > 0`. **This is what makes bench identity provable.**
- At σ = 1 a threshold of mean + 7·MAD ≈ 20 has a false-alarm rate ≈ 10⁻⁵ per tick and still passes every near char
  (e2 ≥ 21, and max-mixing noise only adds energy).
- Near (a = 0) and far speech overlap in level (a = 1 reaches E 91 > near minimum 85), so no fixed level threshold
  separates "my partner" from "someone 4 cells away" tick by tick.

**Baseline ear-alone measurements (SIMULATED, `record.py` + `ear_eval.py`, world2 seed 0, 1 day, listeners silent;
`results/eval_base.log`).** v2.1 front end:
- N = 2, σ = 0: 128/130 lines, near-line onset recall 0.99, 66 categories, 0 junk, purity 1.00, 122 quiet moments/day.
- N = 2, σ = 0.5 / 1: **6 / 1 lines, 256 categories (cap), 218 / 250 of them junk, 0 quiet moments**.
- N = 16 with heavy chatter (p 0.02, ≈ 7 utterances per 1,000 ticks per body), σ = 0: lines whose start is clean
  (nobody else audible in the 3 ticks before) are found with recall 1.00; lines that start while another source is
  audible: **0.00** (24 of 105 near lines). Categories hit the cap 256 (58 junk), 30 % of samples undetected, F on 38 %
  of the day, 0.25 silences ≥ 200 ticks per day.
- N = 64 heavy chatter, σ = 0: near overlapped starts 49 of 408 (recall 0), 81 junk categories, F on 60 %.

So there are two separate problems: **noise** (solvable exactly) and **overlap** (cocktail party; partially solvable).

## 2. Mechanisms (each behind a config key; all off = v2.1 bit for bit)

### H1 `h_gate`: adaptive noise floor + constant-false-alarm sound detector (the core fix)
- Signal: e2(t), the energy of the 2 fresh slices (what changed this tick).
- Floor: μ, ν = leaky running mean and mean absolute deviation of e2 (τ 500 ticks) over **background** ticks (no
  detected sound in the window) and only pairs with e2 ≤ μ + 4ν (trimmed: faint far speech below threshold must not
  inflate the floor; SIMULATED: without trimming, crowds + noise ran away to μ ≈ 15–20 and the agent went deaf again).
- Birth calibration from the warm-up stream (`warmup_windows()`, the one allowed non-obs input): μ, ν of the low half
  of its fresh pairs (silent or noise-only). A rising background is followed by a "stuck" rule: if sound seems to last
  > 600 ticks without a break (no line is that long), the floor also adapts during sound.
- Detector: s(t) = e2(t) > μ + 7ν (hysteresis: μ + 4ν while the previous pair was sound); the window holds sound
  S(t) = s(t) ∨ s(t−1) ∨ s(t−2).
- **Gating**: the rest of the agent reads win_eff = win if S else 0. No trigger code changes: onset, clock, line end,
  quiet moment, wandering, the brake F (= E_eff ≥ 10), the eye parser's silence test all become floor-relative.
- Local and brain-like: per-channel adaptation of auditory neurons to the background's mean and variance (CITED:
  Dean, Harper & McAlpine 2005, Nat Neurosci, level adaptation in the inferior colliculus; Rabinowitz et al. 2011
  Neuron / 2013 PLoS Biol, contrast gain control and noise-invariant cortical responses). One unit holds μ, ν; it sees
  only its own input and its own output (sound/no sound). CFAR = the detector keeps a fixed false-alarm rate.

### H1b `h_back`: onset back-tracking
If a run starts at t and the previous pair already exceeded the low threshold μ + 4ν, the syllable clock is reset at
t − 1. Faint onsets (a ≥ 1 in noise) otherwise start one tick late and every sample of the line is misaligned. Never
fires in a silent background (threshold 0 and e2(t − 1) = 0 at every onset).

### H2 `h_relock`: the syllable clock follows recognition (phase relocking)
At every tick inside a line the window is matched to the categories (Z, the same match as detection). At a sample tick
whose window is not recognised (m < 0.75), if one of the two previous windows (the other two phases) was recognised
with m ≥ θ_lock = 0.85, the clock relocks to that phase and that window is the item. Brain analogue: theta phase
resets to acoustic edges and stays entrained where syllables are parsed (CITED: Giraud & Poeppel 2012, Nat Neurosci).
It fixes lines that start while another source sounds, provided their syllables are known.

### H3 `h_vbound`: voice-change boundary = rhythm change
In world2 every speaker talks at the same syllable rate; two speakers differ in phase. A relock therefore means
"another speaker's syllables now dominate": the current line ends and a new one begins at the relocked phase (the
reply thought of the old line opens and is closed at once by the new onset, as for any next speaker).

### H4 `h_nocreate`: no new categories from a run that showed a second rhythm
After a relock in a run, unrecognised windows are only matched, not recruited (overlapped speech is a poor teacher;
junk categories are never removed and the cap is the PFC's item count V = 256).

### Considered, not built first
- **Per-cell noise gate before the ear** (subtract a per-band floor): not needed if noisy patterns still match their
  templates (the match ignores extra bits; to be checked by purity) and it would make categories depend on the floor.
- **Near-only brake** (F only for loud speech, to cut F-on in crowds): no fixed level separates a = 0 from a = 1
  (above), and the onset tick of a near line must brake (bench mechanics). Decide only if the full agent in crowds
  shows replies blocked by far speech.
- **Trust-weighted listening** (adult prior × learned reliability) and voice categories as speaker identity: this is
  learning-rate modulation in the core (v22's side). The front end can deliver per-line tags (voice category,
  loudness) as a hook; I measure how well they separate sources but do not change learning.

## 3. Attacks

**A1. Does the bench still segment exactly as now?**
- H1: on the bench the warm-up silence is exactly 0, so μ = ν = 0; `_update` with e2 = 0 keeps them exactly 0
  (0 + α·(0 − 0) = 0). Threshold 0 ⇒ s(t) ⇔ e2(t) > 0 ⇔ (speech pair present, e2 ≥ 21) ⇒ S(t) ⇔ E(t) > 0 ⇒
  win_eff = win, E_eff = E on every tick. The stuck rule needs a sound run > 600 ticks; a bench line plus a reply is
  < 100 ticks (to be logged). **PROVED, and checked bit for bit by `test_equiv_h.py`.**
- H1b: needs e2(t−1) > 0 at an onset; at every bench onset e2(t−1) = 0. PROVED.
- H2–H4: fire only if a misaligned window (another phase) matches a category at m ≥ 0.85 while the aligned one
  matches < 0.75. A misaligned window is the template's pattern shifted by 2 or 4 slices; the FEAT0 bits are
  position specific, so it should match at chance (≈ 0.4–0.6). Not provable; **measured**: count of relocks on bench
  runs (the equivalence test) and the distribution of misaligned-window matches in world2. If any relock fires on
  the bench, H2–H4 are restricted or dropped.

**A2. Will categories still form?**
- Noise adds bits only at levels 1–3 in cells where speech is below the noise; the match m = |f ∧ T|/|T| ignores
  extra bits, so a noisy sample still matches its clean template; a template learned in noise carries a little mass
  on noise bits (rate 0.1 running average) and still matches new noisy samples. Prediction: purity ≥ 0.98 and the
  number of categories at σ = 1 within 15 % of σ = 0 (N = 2).
- Crowds: junk comes from misaligned (overlapped-start) and overlapped windows. Prediction: H2–H4 cut junk
  categories by ≥ 30 % and raise near-line syllable coverage of overlapped lines from ≈ 0 to ≥ 0.5; overlapped
  speech of two equally loud voices stays unparsed (no pitch cue exists in this world).

**A3. Is it stable over one unbroken life?**
- Runaway floor: a floor that learns from what it does not detect can climb (faint speech counted as background →
  higher threshold → more speech counted). Trimming at μ + 4ν and the large gap to the detection threshold (7ν) are
  the defence; SIMULATED check on pure noise: σ 0.5 / 1 / 2 converge to the true mean/MAD from both a calibrated and a
  zero start (`results/` session log: thresholds 10.7 / 20.2 / 38.5).
- Changing noise within one life (σ 0 → 1 → 0.5 → 1 over 10 days): the floor must follow without a reset, and the
  categories must not be duplicated (a category formed in silence must keep matching its noisy samples). Test: an
  online 10-day life of the ear module with this noise schedule, per-day lines, categories, junk and floor.
- Relocking can also mis-fire on noise; it only acts on recognised windows (m ≥ 0.85), so it cannot create anything.

## 4. Predictions (to be checked in report.md)
1. Ear alone, N = 2: at σ = 0.5 and 1, lines and quiet moments within 15 % of σ = 0, near onset recall ≥ 0.95,
   categories ≤ 1.15 × the σ = 0 count, purity ≥ 0.98. σ = 2 is a stress point (near speech e2 34 vs a noise
   threshold ≈ 38): expect partial failure.
2. Crowds + noise (N = 16 / 64, σ = 1): line recall and coverage within 0.1 of the same crowd at σ = 0.
3. Full v2.1 agent in world2 at σ = 1 (2 bodies): lines, episodes within 15 % of σ = 0, awake recalls ≥ 80 %,
   utterances > 0 and ≥ 50 % of σ = 0, categories < 100 after half a day, 0 talk-over-partner.
4. Bench, seeds 0–4, frozen protocol, canonical tie rule: J and J2 identical to v2.1 (Δ = 0 exactly) if no relock
   fires; otherwise |ΔJ| within the seed noise (≤ 0.07 per seed, mean within ± 0.03).
5. Not predicted to improve: F-on fraction in crowds without noise (the gate changes nothing there), long silences
   in crowds (they do not exist), the learning content of overheard far chatter (trust is not built).

## 5. Revisions made during Phase 2 (each one forced by a measurement; details in report.md)

1. **Trimmed floor (H1).** The first floor learned from every non-sound tick. In crowds + noise faint far speech
   below threshold leaked into it and the floor ran away (μ 10–20, threshold 50–100; the agent went deaf again:
   near recall 0.01–0.10). Fix: background updates ignore pairs above μ + 4ν. A trim at 2.5ν collapsed the floor
   (selection bias); 4ν is stable from both a calibrated and a zero start (σ 0.5 / 1 / 2).
2. **Back-tracking (H1b) rejected.** A noise outlier just before a near onset (P ≈ 5·10⁻³ per onset at 4ν) shifted
   the clock wrongly; with 2 bodies it only did harm. A stricter 5.5ν version helped far lines in noisy crowds
   slightly but is not needed for anything near. Kept as key `h_back` (default off).
3. **New: H5 `h_conf`, recruit only from clearly heard windows.** A new category needs all 3 fresh pairs of its window
   above the onset threshold. On the bench every sample window qualifies (threshold 0, speech pairs e2 ≥ 21). It
   stops junk from faint far speech and from windows that straddle a gap.
4. **New: H6 `h_level`, level-invariant recognition.** In crowds the same syllable heard from 1–3 levels farther away
   did not match its template (FEAT0 is a level thermometer; world2's distance subtracts levels) and got a duplicate
   category (34 duplicates at N 16). An unmatched window is compared again 1–3 levels louder; ≥ 0.85 = that
   category. Brain analogue: level-invariant coding in auditory cortex (CITED: Sadagopan & Wang 2008, J Neurosci;
   Barbour 2011, Neurosci Biobehav Rev). Never matched on the bench (counter 0).
5. **New: H7 `h_denoise`, syllable patterns from the denoised window.** In noisy crowds aligned windows matched their
   own templates at a median 0.78 only (clean 0.96), so categories duplicated. Silencing cells at background level
   (≤ ceil(5 × mean background per cell): 0 on the bench, 2 at σ 1) before layer 0 makes near patterns nearly noise
   free. The inverse map then has to compare like with like: heard syllables are matched against own-voice
   memories denoised the same way (without this, utterances in noise halved: 92 vs 201 per day).
6. **Relock threshold confirmed.** Misaligned windows never reached m ≥ 0.85 in any recording (P = 0.00 over
   2,300–2,500 misaligned windows per condition), so θ_lock = 0.85 cannot relock onto a wrong phase from a single
   source.
