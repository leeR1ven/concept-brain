# Adversarial verification, DYNAMICS lens: "Listen → Latch → Launch" (arch/brain/proposal.md)
(Saved by the coordinator from the verifier's final reply; the harness blocked the verifier's own write.)

Folder: this one. Labels: PROVED (arithmetic/code reading), SIMULATED (numpy runs), ASSUMED (argued, not run).
Toys at real sizes (ear 27.6k slots / 20.2k cortex, thinking 10k, MD latch 2k, hippocampus/PFC mirrors).
`circuit.py` = byte-identical copy of the proposal's; reproduces T1 16/20, overlap 0.47 vs 0.10, GO {5:14, 6:6}.
Box shared (4 CPUs, 4 GB): some runs cut short (§7).

## 0. Verdict: 1 FATAL (verified fix), 2 SERIOUS, 7 MINOR
Turn machinery (W, MD plan latch, vetoed GO, 3-tick beat, 1-winner selection, final marker, 12-syllable breath limit,
CD) is arithmetically sound under synchronous updates and survived every protocol stress. Single point of failure:
N's input. In the spec N counts ear-cortex neurons; returns and learned recurrence (kept by the proposal, omitted by
the toy) generate that activity in silence. Move N before the cortex → B1, B3, B5, B6 hold in the realistic config.
B2/B4 above chance, but headline magnitudes depend on (a) hippocampus learning during the test, (b) frozen PFC.

## 1. FATAL F1: N counts internally generated ear-cortex activity → no launch (B3), restarts (B6)
Claims attacked: §0 "No learned weight can open the mouth or keep it open"; §5 "B1,B3,B5,B6,B7 hold for any learned
weights"; §4.1 N margins (speech 0.45–0.8, silence 0.015–0.02); §2.1 "relays/returns as now"; §2.4 "ear internal,
ear↔mctx unchanged".
Mechanism (PROVED): N(t) = [0.001·#{non-muted ear-cortex on at t−1} − 10·ΣCD ≥ 0.3]. Ear cortex is top-k;
SenseArea.fire adds every extra current into the top-k scores, incl. learned ear→ear (k 96, eta 0.03, budget 1.0) and
relay returns (hip/PFC mirror → relay fires if ≥3 of 7 z-neighbours in its column on → +0.5 each). In silence these
push replayed speech-feature neurons into the winners ("hearing in the mind" working as intended). Toy has none.
Evidence (SIMULATED, circuit2.py replicates relay rule span 3/need 3/+0.5, relay t from mirrors t−1, ear t from relays
t−1, and the learned ear→ear path; expA/A2, expC, expD_returns.py):
| configuration | silence count after a line (median/max) | N falls at | test GO tick | restarts after own reply | training |
|---|---|---|---|---|---|
| toy (proposal) | 0.05 / 0.056–0.075 | rel 2–3 | 5–6 | 0 | GO 5–6, 0 fallbacks |
| + learned ear→ear only, 350 dial. (expC) | 0.10 / 0.23 plateau | rel 3 | – | – | – |
| + returns only, untrained (expD) | 0.25 / 0.38 (23/120 ticks ≥0.3) | rel 4–6 | 7,8,9,18 | 14 in 10 replies | – |
| + returns + learned ear→ear = AS SPECIFIED, 45 dial. | 0.75 / 0.97 (120/120 ≥0.3) | never | none (0/10) | – | 2/45 launches |
| hippocampus-only returns + learned ear→ear, 45 dial. | 0.19 / 0.45 (10/120 ≥0.3) | rel 4 | 7 | 0 | GO 6–7, margin shrinking |
As specified: teacher waited the full 200 ticks in 43/45 dialogues; the +100 fallback push (8301 ticks) never beat the
−1000·N veto; only 47 channels recruited vs 188 healthy. Reproduces STATE #6/#10. No threshold rescues a count-based N:
silence (0.75–0.97) overlaps window-full speech (p1 0.83, median 0.96 with learned recurrence).
FIX (SIMULATED, verified): N reads the summed loudness of the ear's INPUT WINDOW at t−1 (weight 0.01, thr 0.1 = any
sound), same CD veto. World facts (expH): silent window energy exactly 0; ≥2 slices of sound ≥22; window-full speech
≥52 (3000 lines). Pre-cortical → returns/recurrence/learning/restructure cannot reach it. With returns + learned
ear→ear (expD_fix_energy.txt, expF_fix_energy.txt): silence input 0.000 on 120/120; N falls rel 3 on 10/10; training
GO tick 6 on 45/45 (two rounds), 0 fallbacks; 40/40 corpus prompts + 20 one-char + 15 24-char + 15 ×8-repeat all
launch at tick 6, 0 moves during prompts, 0 restarts, 0 "~"; multi-turn 10/10; partner inside CD tail 10/10; quick
partner 15/15; 260 silent ticks after 5 replies → exactly 5 GOs. Costs: launch 1 tick later; two slices of own voice
exceed the new threshold so the zero CD start margin (M2) becomes load-bearing; back-to-back collisions slightly worse
(M6). Real code: one line in Brain.step (or count non-silence units of ear layer 0, outside cortex, no extra current).

## 2. SERIOUS
### S1: B4/B2 magnitudes depend on the hippocampus learning during the test
Claims: §6.3/§4.2 first syllable 16/20 (14/20), overlap 0.47 vs 0.10, plan decoding 0.80.
Toy shortcut (PROVED): circuit.py lets the hippocampus learn whenever `hip` is set regardless of b.learning; e5_recall.py
runs a teacher-cued guided pass over probes 0–9 right before recall; proposal's own test (§2.6 learning off) and
talk.py check() turn off every path incl. hippocampus.
Evidence (SIMULATED, expE_main.txt; E5 schedule 8 rounds × 25 new pairs + 20 probes, spec GO, PFC frozen):
| test condition (same weights) | first syllable | probes 0–9 / 10–19 | overlap matched/mismatched |
|---|---|---|---|
| T1 toy protocol (guided pass + hip learning) | 16/20 | 10/6 | 0.47/0.10 |
| T2 hip learning, no guided pass | 12/20 | 9/3 | 0.41/0.10 |
| T3 spec test, all learning off | 9/20 | 7/2 | 0.36/0.11 |
| T4 guided pass, then all off | 7/20 | 5/2 | 0.26/0.14 |
Plan at GO−1 vs last training round: same-probe Jaccard 0.164 (hip learning at test) / 0.124 (without) vs 0.05 other
probes; nearest-template decoding 1.00 / 0.75. Same probe's plan hip-learning on vs off overlaps only 0.113 — most of
the latched plan comes from what the hippocampus learns WHILE hearing the line (eta 0.5). Consecutive training rounds
0.161 (others 0.053). n=20 noise ±2–3.
Fix: keep hippocampus plastic during tests (fits B7) → ~12/20; or quote ~9/20 for learning-off check. 9/20 still ≫ 2/20
without the hip→think projection.

### S2: with PFC learning on, the plan loses specificity and period-2 returns ("B2 ≥0.5" [A] fails)
Evidence (SIMULATED, expE_pfc.txt; PFC receiving internal, pfc→think, think internal learn with real BrainCfg params +
set points sp_beta 0.05 tau 1000): training GO unchanged {5:218, 6:142}, 0 fallbacks. First syllable T1 9/20 (overlap
0.34 vs 0.17), T2 8/20, T3 9/20, T4 9/20. Train-vs-test plan decoding 0.70 (hip learning at test) / 0.40 without
(frozen PFC: 1.00/0.75). Same-probe plan Jaccard 0.063/0.049 vs 0.019; across rounds 0.072 (frozen 0.161).
Period-2: after training, thinking layer in silence J(t,t+1) 0.107 vs J(t,t+2) 0.190. Frozen untrained PFC (expG):
thinking layer flat 0.200/0.200 but latch content (thal groups ≥2 on) alternates 0.265/0.419, driven by the 2-tick
think→PFC→think loop (PFC receiving 0.104/0.136). Hippocampus is a travelling sequence (0.287/0.027), not period-2.
Latch is period-1 but samples a period-2 source once at a history-dependent phase → low cross-exposure overlap.
Fixes (ASSUMED, untested): GO reads the latch's 2-tick union (+1 tick latency); or drive latch/channels from the
hippocampus directly; or keep pfc→think plasticity slow vs the fixed hip→think projection.

## 3. MINOR
- M1 (PROVED): teacher's REST cue caps ρ < 3.7 (64 > ρ·I_c*, I_c* ≤ 16+1.0+0.1). ρ≈3 leaves 12.7 margin; raising ρ
  past 3.74 silently breaks training stops (every reply to the breath limit, marker never learns). Fix: ρ only on
  learned currents, or REST +128.
- M2 (PROVED): CD start margin exactly 0 ticks. N(t) vetoed iff a channel on during [t−5,t−2], i.e. t ∈ [b+2, L+5];
  own voice first reaches N at b+2. Any extra tick between channel and CD1 unmasks the first syllable (count-N ~35% of
  replies; energy-N 100%) → W reset and a restart after every reply. End margin 1 tick (0/1111 own-voice ticks ≥0.3
  outside window; latest L+4). Fix: also drive CD1 from GO/the boundary and/or add a 5th CD unit.
- M3: restructure drift (SIMULATED expB, real split/delete ×3): silence count ≤0.035 (claim holds) but window-full
  speech p1 drifts 0.637→0.589→0.563→0.541 (~−0.03/1000 dial.; deletions shrink each layer's top-k) → dropouts at thr
  0.3 after ~8–10k dial. Latent code bug (PROVED): talk.py calls mute_featureless() after restructure(), but
  SenseArea.restructure() just cleared usage → re-mute mutes nothing; split children of muted parents are born unmuted
  (53/52/49 children, nearly all from muted parents, 16–23% fire in silence). Moot under the F1 fix.
- M4 (PROVED): toy GO (0.1·Σthal − 5N − 5·SPK − 10·W) ≠ spec (−1000 vetoes): a latch ≥55 units overrides the N brake.
  Rerun with spec GO: same timing, 0 moves, 0 restarts, T1 16/20.
- M5: a stuck N fails silently (−1000 vetoes vs +100 fallback). Log/alarm on fallback attempts.
- M6 (SIMULATED expF item 5): back-to-back partner lines (3 lines, PAUSE 4, 10 episodes): toy 170 mouth ticks over
  partner sound, 3/4/5 GOs in 6/3/1 episodes; with fix 185 ticks, 1/3/6. If partner resumes within a tick of GO,
  W(t0+1)=0 → a second unprompted reply after the CD tail. talk.py check part 2 (context, PAUSE 4, probe) has this
  structure and must be rewritten. "0 restarts" holds only if the partner never resumes within ~5 ticks.
- M7 (PROVED): repeated syllables: re-selecting the same channel is not a rising edge → X→X gets no forward learning;
  Stream does not restart the self-voice source on a repeated char (phase continues, last slice dragged) → the own-voice
  transient for ear→channel learning is missing after a repeat. 13.4% of replies, 10.5% of prompts have adjacent
  repeats. Content only.

## 4. Claims tried and NOT broken
1. Every fixed inequality in §2.2–2.3 (PROVED, proof_thresholds_out.txt): CD veto −10/unit vs max N input 3.168;
   latch off under any single inhibitory term (5+2−10<2), holds at 2≥2; GO veto ≤200 vs −1000; W set 3≥2, hold 2≥2,
   reset ≤0; muscle release 1.2−2<1; latch size 45±6.6 (P(<5)≈4e−15); exact-cue recognition 16 vs ≤13.2.
2. N inside partner lines, toy config (expA/A2): window-full speech min 0.600 (5148 ticks, 300 lines; p1 0.636);
   silence max 0.056; 0 dropouts at thr 0.3/0.2/0.15; 24-char and ×8 repeats 0 dropouts; 1-char lines keep N on 3–5
   ticks.
3. CD covers own voice incl. sliding-window tail: 0/1111 own-voice ticks ≥0.3 outside [b+2, L+5].
4. Training and test launch on the same structural path: training GO {5:218, 6:142}, 0 fallbacks in 360 dial.; test
   {5:14, 6:6}; teacher never needs to push GO (toy config and under F1 fix).
5. Protocol stress, toy config (expF): 40 prompts all launched, 0 moves, 40/40 STOP, 0 restarts, 0 "~", every syllable
   3 ticks; 1-char / 24-char / ×8 all fine; multi-turn 10/10; partner in CD tail 10/10; quick partner 15/15; 260 silent
   ticks after 5 replies → 5 GOs.
6. No synchronous-update artefact in the turn circuit (every latch period-1; GO one-shot under SPK veto; no loop
   through N except via the ear, masked by CD).
7. Plan latch timing (PROVED): fills one tick after N falls, GO reads it once, GO/SPK/W kill it at GO+1 → clean +eta
   thal→channel event.
8. Restructure does not raise N's silence count (expB).
9. Learned ear→ear alone does not break N (plateau 0.23 < 0.3; costs margin and a tick).
10. Scaling: toy already at real sizes; dense thal→channel 2000×4097 ≈130 MB on GPU.
11. B1 survived every configuration (broken N errs toward silence).

## 5. Toy fidelity
Faithful: update order, rule arithmetic, world code, ear. Shortcuts that matter: (a) no returns / no learned ear
recurrence → F1; (b) hippocampus learns during tests + guided pass before recall → S1; (c) PFC frozen → S2; (d) GO
formula ≠ spec → M4 (no effect). Tests use familiar lines only (acknowledged).

## 6. Files
circuit.py (copy), circuit2.py (switches: spec GO, hip learning at test, PFC learning, relay returns, learned ear→ear,
energy-N, hip-only returns), earlib.py, proof_thresholds.py→_out.txt, expA_N_margins.py→expA_out.txt,
expA2_tail.py→expA2_out.txt, expB_restructure.py→expB_out.txt, expC_ear_learned.py→expC_out.txt,
expD_returns.py→expD_ret_earl.txt / expD_hiponly.txt / expD_fix_energy.txt, expE_train_test.py→expE_main.txt /
expE_pfc.txt, expF_protocol.py→expF_toycfg.txt / expF_fix_energy.txt, expG_period2.py→expG_out.txt,
expH_energy.py→expH_out.txt.

## 7. Not done
No torch/GPU run of real code (PFC learning, set points, mctx returns approximated in circuit2.py). A2 seed 0 only;
D-main stopped after its first round (memory). S2 fixes untested.
