# X1f plan: generation 0 with hand-added connections; then learning on top (written 2026-11-01, before the runs)

Authority: docs/ARCHITECTURE.md (11-01), incl. sec. 2.11 "systematic training" (coordinator update during the task):
the main path is (a) hand-wired sound <-> action map + an overhearing stage + a speaking stage; hand-wired thinking
(b) and responding (c) is a secondary, DIAGNOSTIC arm.

## Base (both arms)
X1d RNpm (no PFC inhibitory neurons, X3 patch-mean, held-neuron onset rule, post = next input) with L3->L3 learning in
X1e's form (RNpm_L2: onset rule, ARCH two-layer weights g33 = 2, dec_33 = 0.95). Parameter changes, fixed before the
5-seed runs:
- at X1d's thresholds (f2 0.22, f3 0.3) 24 of 27 syllables light fewer than 8 reliable top-layer units across voices
  (several light ~1 unit at all: 色, 大, 香), i.e. the PFC top layer does not see them;
- lowering the thresholds alone (f2 0.18, f3 0.25) gives reliable codes but the top layer saturates once the learned
  links exist (overhearing 300 dialogues: mean L3 300 of 400, P(saturated) 0.20, with or without L33 learning);
- so the X3 patch-mean (ARCH 4.10, "subtract the patch's mean drive in the same tick") is applied to the whole
  top-layer drive (pm_ff = 1, X1e's option), with f2 0.18, f3 0.08: every syllable lights 32-60 top units (reliable
  within a voice: median 39, min 23), and over 600 overheard dialogues L3 stays at mean 87, max 134 of 400, never empty.

## Generation-0 hand-added sets (REGISTRY.md)
- A1 sound -> muscle map (non-negative least squares from each syllable's content code to its muscle set), replaces
  X1c's stand-in (whose 4th/5th muscle margin is only 1.5x and garbles 77-95 % of forced answers).
- A2 blocking links: release only a syllable-shaped command (3-5 muscles), so a mixed inner hearing is silent.
- Motor knobs (existing): c_mb 1.2 -> 0 (no muscle->blocker latch), g_dir 0.5 -> 0.25 (direct echo link).
  Screen (seed 0, learning off): forced whole-answer inner hearing said exactly 1.00, mixed inner hearing said 0.14 with
  0 garble, spontaneous speech 0.
- Diagnostic arm only (B, C): filler-hold assemblies, base-pair question / answer-step assemblies (L3<->L3, H33) and
  answer-step -> inner hearing links (H31), for a base set = half of the seen pairs (seed-fixed).

## Arms
- **G0** (main): A1 + A2 only.  Stage O: overhear N dialogues (4 adult voices, all seen pairs, narration, confirmations,
  empty room), learning on, nobody asks the child; checkpoints N = 100, 300, 1000, 3000, 10000.  Stage S: speaking
  (X1d run2 format caregiver), 300 exchanges.  Then untutored 1,000 (3,000 for one seed), amnesia check, corrections.
- **G0T** (diagnostic): A1 + A2 + B + C (base pairs).  Test 1 (learning off), then the same stages O / S for the other
  (never-wired) seen pairs, then stability and corrections (2 wired + 2 learned changed answers).
5 seeds each (0-4).  Probes are frozen (learning off), question in voice C1, 16 silent ticks.

## Pass criteria (fixed now)
Test 1, sanity, learning off (G0T on wired pairs; G0 for the speech basics):
- forced whole-answer inner hearing said exactly >= 0.9 (G0); wired-pair accuracy >= 0.8 in the live world and probes (G0T);
- talk-over per exchange <= 0.02 and failure to yield = 0; garbled utterances per exchange <= 0.1;
- path: question decoding at lag 12 (wired pairs, muted mouth) >= 0.8.
Test 2, learning:
- G0: heard-pair accuracy after overhearing (best checkpoint) and after the speaking stage >= 0.5; held-out reported;
  child utterances while overhearing <= 0.1 per dialogue.
- G0T: never-wired pairs answered >= 0.5 after overhearing + speaking; wired pairs stay >= 0.8.
- report how new pairs got their paths: question decoding at lag 12 on never-wired pairs, own - other prediction
  specificity, L33 learned-weight statistics.
Test 3, stability: after 1,000 untutored exchanges, wired and learned pairs keep >= 0.8 of their accuracy; no
  saturation (P(L3 >= 90 %) < 0.01); amnesia: no hippocampus in this build -> wipe the fastest learned store (L33)
  and re-probe; pass if accuracy keeps >= 0.8 of its value.
Test 4, corrections: 4 changed answers (G0T: 2 wired + 2 learned; G0: 4 learned if any are learned); report re-teachings
  until the first correct new answer; unchanged pairs must keep >= 0.9 of their accuracy.
Test 5 (optional merge): only if test 2 passes for G0T.
If anything fails: report where the chain breaks (reliable codes -> question-specific path -> specific prediction ->
release), with numbers.
