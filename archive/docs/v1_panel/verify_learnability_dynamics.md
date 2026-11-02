# Dynamics verification: "Hold by structure, learn from one-tick transients" (learnability proposal)
(Saved by the coordinator from the verifier's final reply; the verifier could not write this file itself.)

**FATAL 0 · SERIOUS 1 · MINOR 8.** Timing mechanics work in this world. One real defect: the breath reserve, which the
proposal's toy never implemented; a small fix is checked.

Labels: PROVED = tick algebra or code reading; SIMULATED = own simulation (file + numbers); ASSUMED = not checked.
All scripts/outputs: this folder.

## 1. What was built
- `ear0.py`: exact replica of ear layer 0 (same Net3D constructor + seed). `validate_ear0.py`: 0 mismatches vs the
  proposal's `ear_np.World` over 1,356 ticks. FEAT0 = 1,153. Layer-0-only simulation is exact (F, T read only FEAT0;
  SenseArea.fire masks external currents to layers ≥1; layer 0 never split/deleted; vary only layers ≥1).
- `circ.py`: circuit written from the TEXT (F with −10ΣV, F2, S1–S8, V1–V4, GO, T top-K/hold 3.0/set point/ε, P, ring
  R2/R0/R1 with breath, WTA over Y with cue and END push, D/I/M same tick, real Voice.mouth). Exact Path.learn/decay.
  Every toy-vs-text difference is a switch.
- `run_train.py`: real corpus, §2.4 protocol, tick-level audit of every training turn; TEST protocol (pause 0, 60 free
  ticks) at checkpoints + stress scenarios.
- Variants (1,000 dialogues): A = text as written (`run_A.out`); B = toy rules (`run_B.out`); C = text + fix
  (`run_C.out`); D = per-tick breath (`run_D_200.out`).
- Other: `trace.out`, `sim_F.out`, `sim_F_worst.out`, `sim_T_drift.out` (4k/200, 2,000 dial.),
  `sim_T_drift2_gpu.out` (10k/500, 1,500 dial.), `stress_pause.out`, `stress_misalign.out`, `sim_free.out`.
- Corpus facts: lines 2–12 chars, no empty lines/whitespace/punctuation (no silence inside a line); 16% of dialogues
  have odd length; 3.5% of the other's lines end in a repeated char.

## 2. Issues

### S1 (SERIOUS): breath reserve breaks END learning and caps replies
Claims attacked: §2.2 R1 "x_R stops the ring after ≤15 syllables [P]"; §2.4 END push; §8.2 "every reply stops".
- PROVED: toy_circuit.py has no breath (l.118 R1 = R0∧Y; l.132 hard counter n_syl < 16). All toy B6 numbers come
  from a different mechanism.
- PROVED: τ = 100 ticks vs ~49-tick turn period → breath never refills. Back-to-back replies (per-syllable depletion):
  reply len 2/4/6/8/10/12 → breath at turn start 0.78/0.68/0.63/0.60/0.58/0.56 → ring dies after syllable 11/8/6/5/4/3.
- SIMULATED (A): R1 died before the last boundary in 642/1,455 teacher replies (44%) → no R0 at END push → END not
  learned there. END events 969 (A) vs 1,657 (fixed). Guided speech hides it (cue forces R0). Free replies longer:
  2.95 vs 2.40 syl at 500 dial., 2.25 vs 1.35 at 1,000.
- SIMULATED (D, existing Region.deplete per tick): END missed 291/293; from full breath the ring dies after 5 syl;
  x_R = 0.32 at test; 15 free replies cut.
- ASSUMED: read literally, "nothing chosen if breath has run out" would also truncate teacher-forced replies.
- FIX (SIMULATED, C): refill x_R to 1 when GO fires; END push forces R0 like the cue. 0 misses; ≤15 becomes true;
  outputs identical to B at 500 and 1,000. Equivalent: a syllable counter reset by GO.

### M1: toy differs from text in four rules (harmless once S1 fixed; B and C identical)
1. Own-voice veto timing: toy V1(t)=D(t); text ΣD(t−1). Own voice drives F on during [r0+2, m_e+4]; at m_e+4 current
   ≥0.022 (all chars), at m_e+5 ≤0.0007. Toy veto [r0+1, m_e+4] (0 spare at end); text [r0+2, m_e+5] (0 spare at
   start). Suggest V from D(t) and V4 held one tick longer → 1 tick slack each side.
2. R2: toy (GO∧¬F)∨R1 (F cannot stop a running ring); text GO−2F+2R1 (F can).
3. Cue forcing R0: text forces; toy drops cue if R0 off. Never disagree in protocol (0 unexecuted cues / 1,455).
4. Toy counter n_syl<16 acts in training too; never triggers (max reply 12).

### M2: F margin is 2.0×, not 2.75× (detection never failed)
Lowest current (thr 0.008): first tick after silence 0.0365; first tick right after own voice of same char 0.0270;
F(L+2) repeated char (盒×6) 0.0158; F(L+2) corpus (…呵呵) 0.0168; inside 40-char line 0.042; 40 repeats 0.045.
F(L+3) ≤0.001 always. Corpus audit: 0/2,404 lines offset away from L+3; 0 with first tick off.

### M3: training needs tick-exact agreement of teacher GO and model GO
If the teacher push lands one tick off the learned GO (L+5), GO and P fire twice, the ring carries two pulses.
GO at L+4: 138 short boundaries, 78 short syllables, guided 7/30; L+5: 0/0/30/30; L+6: 120/60/23/30.
Never happens in this deterministic world, but any timing change triggers it every turn.
Mitigation (ASSUMED): refractory GO + cue resets ring phase (like go_reset).

### M4: overlap outside protocol gaps (stress_pause.out; other says A, waits g, says B)
g 0–4: 0/30 overlap, B answered afterwards. g 5–7: 30/30, exactly one syllable over B (R2 kills ring), B answered
29–30/30. g 8–9: 30/30, whole reply over B (cannot hear it), 6/30 B never answered. g 10–12: 13/30.
PROVED: a line starting and ending inside the reply + 4-tick veto is never detected.

### M5: idle decay can silence the model permanently (PROVED)
S1→GO is GO's only learned input; at budget 2.0, γ 0.999/100 ticks, no GO events → <1 after 693 decay steps =
69,280 idle ticks (~87 min at 75 ms/tick); then GO never fires and nothing refreshes it. Real code decays regardless
of activity (areas.py l.281–283). Fix: fixed S1→GO 2.0, or decay gated on target firing, or floor w0 = 1.

### M6: set point shifts T codes once, then jitters (SIMULATED)
Hold: 0 changes of T between L+3 and readout in 2,896 (4k) and 2,171 (10k) turns. Max penalty 1.09–1.35 (<2.7
needed to break hold). Code shift: probe codes Jaccard 0.60–0.61 vs dialogue 0 (1.00 with penalty off) → settles in
~100 dialogues. At 10k/500 checkpoint-to-checkpoint Jaccard 0.849–0.857 (stationary jitter, not random walk).
Decoding stays 20/20. P→Y links learned in the first ~100 dialogues ~39% stale; afterwards ~8% of each code changes.
Context carry-over into readout 8.5–10.2% of K (lines 2–8 chars). Own voice: mean 9.9% (max 72.5%) of the code
survives the own reply (after P read it; no effect on first syllable).

### M7: harness / real-code integration (PROVED from code unless marked)
1. talk.py check() parts 1–2 must wait for the reply; the model now answers every line followed by ≥7 silent ticks.
2. Speed: on GPU Path.learn processes the whole table every tick (39.7M P→Y entries); ASSUMED ~1.5 GB temporaries,
   +1–2 min / 500 dialogues.
3. Region.homeostasis overwrites `base` each tick → the fixed ε needs its own constant current.
4. Same-tick dependency: Y selection needs R0(t) and cue(t) in the same tick; fire R0 before the motor part or compute
   it inside from R2(t−1).
5. FEAT0 needs a layer-0 usage count (featureless() returns only cortex neurons).
6. Building breath with Region.deplete gives variant D.

### M8: tick errors in §3 table (no behavioural effect; trace.out)
T frozen from L+3, not L+2 (53 units still change at L+2). F(r0+1) not vetoed (V(r0)=0); zero only because own voice
enters the ear at r0+1. First vetoed tick and first tick own voice changes T: r0+2.

## 3. Claims tried and NOT broken (more trusted)
- B1: 0 mouth moves while the other speaks in every training turn, test, and free conversation (1,600 turns, learning
  on/off). At odd-dialogue boundaries GO fires at L+5 while the next line has begun (144/152) but P and R2 blocked by
  F(L+5) ≥0.027: 0 P/R2 firings.
- F timing: off exactly at L+3 in every turn.
- Own-voice veto / no self-reply: exactly one GO per probe; 0 moves after reply end.
- B3: one GO at L+5, one P at L+6 (K units), first boundary L+7, train and test alike; latency 7 always.
- B5: 0 double rings, 0 garble. Guided exact except the 2 probes with repeated chars (哈哈哈, 哼哼) — world limitation.
- B6: every free reply stops by END or the 0.02 floor once S1 fixed.
- No period-2 cycle (T loop is one tick); one ring pulse and one GO per turn.
- Cue margin (PROVED): cued syllable scores 20; any other ≤ 15 + 2.0 + 0.2 = 17.2.
- 0 back-term hits on P→Y (PROVED): P on only at L+6; every Y silent at L+5.
- Free use with learning on (600 trained, then 1,600 turns): no collapse; same 7 distinct first syllables every
  400-turn window; S1→GO stays 2.0; ~2.0 syllables; replies every window. Caveat: repertoire already = prior.
- GPU-size T (10,000/500): same hold, penalties, drift as 4,000/200.
