# Full report: dynamics verification of arch/minimal/proposal.md
(Saved by the coordinator from the verifier's final reply; the harness blocked the verifier's own write.)

Counts: 0 FATAL, 1 SERIOUS, 7 MINOR. Verdict: mechanics sound.
Labels: PROVED = argument from spec/code with numbers; SIMULATED = numpy sims; ASSUMED = reasoned, untested.
Folder: this one. ear.py, brainsim.py, brainsim_y.py copied unchanged. turnsim.py = independent re-implementation of
spec §2.2/§2.3 (F, F1, O, GO, boundary, H/I/M, mouth) on the real ear replica with scripted output (no learned inputs).

## SERIOUS
### S1 (scope): any pause ≥3 ticks counts as end of turn; a turn cannot be aborted; a resumed line is never answered
Claims: B1 "[P] GO needs F(t−1)=0, and F=1 from the line's 2nd tick to e+3"; B3 "never a second turn"; stated residual
only "an other who starts while the model is still speaking is talked over".
Evidence (SIMULATED t_pauses.py, log_pauses.txt; trace S4 in trace.txt), two lines of the other separated by p silent
ticks, 40 pairs per p:
- p 0–2: lines merge, model answers after line 2 (correct).
- p 3 and 4: GO fires at e1+5 = tick 1 (p=3) or tick 0 (p=4) of the new line; F(t−1) still 0 there (F lags sound one
  tick) → model STARTS SPEAKING DURING the other's line, which the [P] statement excludes.
- p 3–8: mouth ticks during line 2 = 329, 339, 303, 267, 231, 191.
- every p ≥3: exactly one turn per pair; line 2's rising edge falls while ΣH = 4 → O cleared → line 2 NEVER answered.
Consequences now: check() phase 1 (three lines with listen(…, PAUSE=4)) will count ~9 mouth ticks per probe line;
phase 2 (decoding after context lines, pause 4) will contain the model's own replies; phase 3 still uses old say()
(go push + D cues); 3-utterance dialogues (16%) cannot be played as heard (why C10 drops the unanswered line).
Inside the protocol (C10 pairs, END + 3 silent ticks, reply() with 60 free ticks): 3,188/3,188 clean.
Fix: harness — measure B1 inside reply(); separate two lines of the other by ≥45 ticks (5+36+4). Open conversation
(ASSUMED): "other is talking" detector with own voice cancelled by efference (motor stage knows the muscle set → H →
own-voice-feature inhibition into F) + abort-on-onset → END. Cheap partial rule "abort if F=1 at GO+1" catches only
p = 3, 4.

## MINOR
### M1: END-cue stop inequality not proved; a failure would feed itself
- "cue 50 > Σ caps 46" covers program competition: every learned ŷ_c ≤ 46 < 50 → continuation cue always wins and never
  stops (PROVED). At an END-cued boundary the stop needs 50 + ŷ_END ≥ (1−U_b)^n · Σ_c ŷ_c; the sum has no cap.
- SIMULATED (t_equil.py, log_equil_600_10.txt; toy accelerated η×10, (1−γ)×10, 600 dial ≈ 4.4 τ; spec caps 30/10/3, cue
  50, U_b 0.25, gate on, stop rule): worst breath·Σŷ − ŷ_END = 11.4 vs 50 needed (~4.4× margin); END failures 0; cued
  program lost 0. Proposal's 500-dialogue logs at ~0.35 τ: Σŷ = 6 and rising.
- Latent loop (ASSUMED): if END cue fails, model releases its own program at a cued boundary; C11 gate open → learns
  "continue", never sees END; Σŷ rises; overrun into next line; O never set; 10-tick fallback starts next reply with
  different timing.
- Fix (logically trivial): END cue forces the stop (stop := END cued, or [no cue and ratio test]); gate learns only when
  the released program equals the cued one.
### M2: zero tolerance on the pause after END in training (PROVED by trace; SIMULATED trace S3)
Next line's rising edge needs F(B+4) = 0 (B = END tick) → ≥3 silent ticks after B. With 2, F stays on through the switch,
O never set, own GO never fires, teacher fallback starts the reply 11 ticks after the line instead of 5 → trains a timing
and GO−1 source the test never produces. PAUSE 4 (END + 3) is exactly the minimum. Fix: PAUSE 5; assert zero fallbacks
in say().
### M3: thin but harmless F margins; one-syllable reply safe by exactly one tick
F inside a line ≥0.0295 (thr 0.01); at e+3 min 0.0170 (proposal quotes 0.022); own-voice tail at L+4 ≥0.012; silence
≤0.0015 (PROVED: deterministic input → fixed layer-0 set with one feature neuron). Thin spots harmless: F off one tick
earlier → GO at e+4, deterministic, same in train and test; no rising edge possible. One-syllable reply (PROVED trace S2;
SIMULATED 423/423): F rises from own voice at e+7 while ΣH = 4 → edge blocked; END at e+8. Depends on own-voice lag
= 1 tick; with lag ≥2 every one-syllable reply would latch O after END and start a second turn. Re-check if
world/voice.py changes.
### M4: toys differ from spec in four places (no dynamics conclusion changes)
(a) brainsim_y.decide multiplies the cue by breath (cont = breath·Σ(ŷ+cue)); spec adds undiscounted. Harmless:
ŷ_END ≤0.8 < 0.75¹¹·100. (b) runs y1, y2, gate used no caps (B_*=0) and cue 100 vs caps 30/10/3 and cue 50.
(c) warm-up runs full step; before setup_y the muscle-level fallback releases 4 arbitrary muscles → toy answers each
warm-up char, babble enters muting/layer-0 stats; real code silent. Clean warm-up: 1,153 layer-0 feature neurons, same
F results. (d) no Y/END mirrors in the hippocampus though spec lists chains from them. No leaks: test has no cue/teacher;
update order matches Circuit.step; only same-tick reads (motor reads GO(t); I, M read H(t)) are in spec and current code.
### M5: hazards when porting to real code
(a) Y and END must stay structurally gated to boundaries: as ordinary threshold units with chains, the first-syllable
cue (present e+1..e+5) would fire Y and its hippocampal mirror before GO → leaks the teacher's answer into hip(e+4), the
source the first syllable is learned from. (b) H→O/GO inhibition should use use_x=False (motor std_U 0.5; default
use_x=True scales inhibition by H's reserve; still works — O's edge term exactly at threshold, x_H = 1 at GO+1 — but
margins shrink). (c) Decision paths need a dense GEMV: table + index_add_ with K = 4,097 targets per source ≈ 2·10⁸
colliding atomic adds per tick; evaluate only at boundary ticks; learn() only when gate open. (d) layer-0 usage pass is
new code (featureless() covers only cortex). (e) Layer 0 must stay fixed: vary_from ≥1 (ASSUMED); restructure never
touches layer 0 (split_frac[0]=0); SenseArea.fire adds extra currents only to _cx → F immune to imagery/learning (PROVED).
### M6: a repeated own syllable is heard as one held sound (content trap)
Stream never restarts the own-voice phase for an unchanged char; motor keeps the same H set across the boundary → "我我我"
heard as 我 with last slice held. SIMULATED (t_heldctx.py), Jaccard of decision sources at B−1 between consecutive
boundaries — inside a run of one repeated syllable vs distinct: ear cortex 1.0 from the 3rd syllable vs 0.22–0.29; mctx
1.0 vs 0.04–0.17; hippocampus ≈0.4 vs 0.17–0.37. Runs nearly absorbing ("我我我我我我我我我" in run_gate.log); cannot learn
how many repeats; only breath ends a run. Turn-taking safe (t_rep.py, t_proto.py rep/rep12/mix: F ≥0.030; 0 second turns
over 300 chars × 8 repeats + 1,246 training pairs). Fix (world, untested): restart own-voice phase at each syllable onset
(Voice.mouth signals re-articulation when the force ramp restarts, or the boundary sets new_self).
### M7 (ASSUMED): optional think→Y would read an unlatched, possibly period-2 thinking layer
First decision reads think(e+4) while the thalamus is still suppressed (F(e+3)=1 turns it off at e+4) — the window of
STATE §4.5's period-2 → source phase could depend on line-length parity. Measure before enabling.

## Equilibrium magnitudes (accelerated; spec caps/cue/U_b; τ ≈ 140 dialogues)
| Dialogues | Σ_c ŷ_c median/max | ŷ_END median/max | worst breath·Σŷ − ŷ_END (needs <50) |
|---|---|---|---|
| 50 | 4.4/8.4 | 0.39/0.76 | 2.6 |
| 100 | 10.5/13.5 | 0.88/1.34 | 6.1 |
| 200 | 17.1/19.5 | 1.63/2.15 | 8.1 |
| 400 | 22.3/25.0 | 1.98/2.59 | 11.4 |
| 600 | 22.0/24.4 | 2.08/2.81 | 11.2 |
END-cue failures 0; cued program never beaten. Learned totals per target at 600: hip → END 4.0, largest program 1.13,
cap 30; ear 3.1 / 0.57 / 10; mctx 1.2 / 0.20 / 3. No cap binds; proposal's ~21 for hip→END ~5× too high (safe side).

## Claims tried and NOT broken
| Claim | Result | Label |
|---|---|---|
| F on every tick inside any line from tick 1 (2–12 chars, repeated chars) | min 0.0295 over 3,188 pairs | SIMULATED; silence PROVED |
| F off exactly e+4, GO exactly e+5, same train and test | 3,188/3,188 (train 1,829 incl. repeated-char replies; test 1,359 incl. one-syllable and 12 random chars) | SIMULATED + PROVED |
| No turn without a heard line | 0 extra turn starts in 3,188 pairs + 300 held runs; own rising edge always falls while ΣH = 4 | SIMULATED + PROVED |
| Mouth still while the other speaks, within protocol | 0 ticks | SIMULATED |
| Training never deadlocks or skips | 0 fallbacks; continuation cue always wins; END margin ~4.4×; no reply cut | PROVED + SIMULATED |
| No latch/pulse artefacts | GO resets O, ΣH clears stale O; no 2-tick oscillation (O needs a new F edge); double GO only when no program exists (harmless); n resets every turn → no breath-refill defect | PROVED |
| B5 one clean char per 3 ticks | H exact set, I = ¬H (−10), M = H ∧ level ≤ φ; only other inputs to M are hip/PFC returns (0.3+0.3 < 1); C4/C6/C8 remove every other path into motor area or GO (enumerated in brain.py) | PROVED |
| Adding PFC, returns, ear learning back cannot break B1/B3/B5 | they reach only cortex ≥1 (masked from F), Y/END (read only at boundaries) or M (harmless) | PROVED |
| ≤12 syllables per heard line | syllable cap + no second turn | PROVED + SIMULATED |

## Files
turnsim.py; t_proto.py + log_proto_*.txt (7 runs); t_rep.py; t_pauses.py + log_pauses.txt; t_trace.py + trace.txt;
t_equil.py + log_equil_600_10.txt; t_heldctx.py; corpus_dyn.py (lines/replies 2–12 chars; 13.4% of replies contain a
repeated char, 4.7% end in one); an_logs.py.
