# Minimal-change architecture: structural turn gate + syllable-program release + content learned only into the release

Panel angle: **minimal change** from commit 004031a (+ the uncommitted think→motor path). Keep every area, size and the
learning rule; keep the hippocampus, the budgets and the quiet mouth while listening; change only what the logic requires.

Labels: **[P]** proved by an explicit argument with numbers · **[S]** simulated by me (numpy toy in this folder, same
update order, same rule, same timing; result quoted) · **[A]** assumed. One tick = all currents from tick t−1 states →
all parts fire → learning → reserve update (`Circuit.step`).

---

## 0. Summary

1. **When to speak is structure.** The fresh unit F is rewired to the feed-forward first ear layer (θ 0.01); with a 1-tick
   delay F1 and a latch O ("the other has the floor"), the existing GO fires exactly once, 5 ticks after every heard line,
   never during a line and never after the model's own line [P, S: 1,600/1,600 test lines].
2. **How to speak is structure.** A boundary releases exactly one syllable program (4 muscles), H holds it 3 ticks,
   inhibition is absolute ⇒ one clean char per 3 ticks [P, S: 0 garbled ticks]; ≤ 12 syllables per turn [P].
3. **What to say is learned with the unchanged rule, but only into the release decision**: hippocampus(GO−1) → first
   syllable, ear (own voice) → next syllable, both → END. Release units fire 1-tick pulses, so the designer's back term
   keeps 96 % (first) / 67 % (next) of the increments, vs 55 % / 37 % for a 3-tick held target [S].
4. **Two logically required content fixes**: (a) the decision is made over *syllable programs* (one unit per learned
   muscle set), not over 32 independent muscles: with perfect knowledge of the first-char distribution, top-4 muscles say
   "根" (starts 0.0 % of replies) while the program argmax says "你" (9.1 %) [S]; (b) the decision paths learn only from
   teacher-confirmed releases (one global third factor); with ungated real-time learning the model's own babble rewrote
   its first syllable from 你 to 的/了 within 200 lines; gated, the trained choice survived 145/150 free lines [S].
5. **Honest limit.** B1, B3, B5, the termination half of B6, and B7 hold by construction. B2/B4 are learnable but weak:
   at GO−1 the hippocampus keeps only ≈12 % line-specific neurons under real-time learning, and the corpus offers little
   (best bag-of-heard-chars predictor of the first reply char: 9.5 % vs 7.8 % shuffled at 30k dialogues). After 500 toy
   dialogues the model says the corpus mode (你 / 我) and then near-marginal chars (next-syllable choice 4.4 % vs 3.6 %
   context-free); END has not yet learned where replies end, so length is set by a breath constant; 14 % of said chars
   belong to the true reply, but no more than for shuffled replies [S].

---

## 1. Changes relative to the current code (complete list)

| # | Change | Where | Why (evidence) | Replaces / removes |
|---|---|---|---|---|
| C1 | **F reads the first ear layer** (layer 0: no learned or top-down input) — only neurons not in the silence pattern (warm-up usage ≤ 0.6, ≈1,150); weight 0.001 × reserve; θ_F = **0.01** | ear⇒fresh, `fresh_thr` | current θ 0.4 sits at the median of the in-speech current (0.435) ⇒ ≥ 600 of 5,018 in-line ticks off (235 one-tick + 186 ≥ 2-tick dropouts in 300 lines) [S]; new F ≥ 0.023 during any speech [P], ≤ 0.0015 in silence [S]; immune to imagery | cortex sources, θ 0.4 |
| C2 | **F1** (1 neuron) = F delayed 1 tick | new `Unit` | rising edge of "sound starts" | — |
| C3 | **O** (1 neuron, latch) = [2F − 2F1 + 4O − 6GO − 1.5ΣH ≥ 2] | new `Unit` | "the other spoke; my turn not yet taken" | — |
| C4 | **GO fixed**: [2O − 2F − 1.5ΣH ≥ 2] | `MotorArea.go` | fires 1 tick after sound stops, only if O | learned think→go, ear→go; `go_reset` |
| C5 | **Release stage**: Y = syllable-program units (one per muscle set the teacher has cued; ≤ 4,096 slots, recruited on first cue, fixed Y→D wiring = that set) + END; one winner per boundary; D = 1-tick per-muscle release pulse driven by the winner; new H[32] holds the set for the beat | `MotorArea` | B4 needs a char-level decision (readout result above); a pulse target keeps its sources' increments | top-4 over D; D held 3 ticks; `dis_thr`, `d_floor` |
| C6 | **I, M have no learned inputs**: I = [1.2 − 2.0·H ≥ 1] = ¬H; M_{m,l} = H_m ∧ (l ≤ φ); inh_w = 10 | `MotorArea` | articulation cannot be disturbed by learning (guided speech fell 100 % → 61–83 %) | learned →I, →M, →D |
| C7 | **Stop rule** at a beat boundary: END iff y_END ≥ (1−U_b)^n · Σ_c y_c, or n = 12 (U_b = 0.25); first boundary after GO always releases | `MotorArea` | END is learned like a syllable (content stop, not yet visible at 500 dialogues [S]); the breath sets the typical length; the cap guarantees termination | `stop()`, d_floor |
| C8 | **Learned decision paths** (rule unchanged, back = True, birth weight 0): hip→Y, ear→Y (× reserve), mctx→Y (× reserve), think→Y (optional); η 1e-5; γ 0.999/100 ticks; per-target budgets as **caps above equilibrium** (30, 10, 3, 3) | `brain.py` | content orders the programs; caps must not bind (a binding cap normalises away the prior, [S] in the muscle variant) | think→motor (back=False), ear→motor (k 8), mctx→motor, M→motor |
| C9 | **Featureless muting extended to the motor area**: tonic I neurons get no chain/mirror | `mute_featureless` | their mirrors switch off exactly when their muscle is used ⇒ the rule learned a context-free usage prior that fixed every first syllable to "再" (193/200) [S]; also removes the 32 always-winning PFC mirrors (synthesis F2) | chains of I |
| C10 | **Teacher protocol**: `listen(line, pause=0)`; `say(reply)`: cue the first program every tick until the model's own GO fires (fallback teacher GO after 10 ticks), cue program k at its boundary, cue END at the next; PAUSE 4; **only (other, reply) pairs**; cue = 50 > Σ caps 46 | `talk.py` | train and test share the GO−1 state; unanswered lines made the model take an uncued turn colliding with the next dialogue (69 fallbacks / 583 replies) [S]; after the fix: 0 fallbacks / 2,169 replies [S] | REPLY_GAP push of go; `stop()` |
| C11 | **Third factor on the decision paths only**: η of C8 paths × [teacher cue present at that boundary] | `Path.learn` call site | ungated, self-produced babble is learned as truth: first syllable drifted 你 → 的/了 in 200 free lines; gated: the trained choice is kept in 145/150 free lines with learning on [S] | — |

Unchanged: ear, motor cortex, hippocampus (except C9), PFC receiving / thinking / thalamus, relays and returns, ear↔mctx
paths, budgets of all other paths, decay, reserve, set point, warm-up, restructuring.

---

## 2. Complete specification

### 2.1 Areas (sizes from the code; muting counts from my replica [S])

| Area | Neurons (alive) | Active / tick | Rule | Change |
|---|---|---|---|---|
| Ear input | 8×6 cells × 20 lines | exactly ½ | thermometer pairs | — |
| Ear layer 0 (not cortex) | 1,920 | 768 (40 %) | top-k, feed-forward only | feeds F (C1) |
| Ear cortex, layers 1–3 | 20,160 − 3,153 muted = 17,007 | 3,168 (≈718 unmuted per char, ≈3 in silence) | top-k 30/20/10 % | — |
| Motor cortex, layers 1–2 | 3,840 − 896 muted | 30 / 20 % | top-k on the force map of t−1 | — |
| Hippocampus (mirrors) | ≈20,100 | 5 % ≈ 1,000 | top-k | no I mirrors (C9) |
| PFC receiving / thinking / thalamus | ≈20,100 / 10,000 / 2,000 | 8 % / 5 % / threshold | as now | F is the release signal (now robust) |
| Relays | as now | threshold 3 | as now | — |
| Motor area | 32 × (3 M + I + H + D) + END = 193 | ≤ 4 H, ≤ 12 M, 28–32 I | structural (§2.3) | C5–C7 |
| Syllable programs Y | ≤ 4,096 slots (≥ 3,970 chars); 987 recruited after 500 toy dialogues | 1 pulse per boundary | winner-take-all | C5 |
| Turn units | F, F1, O, GO | 0/1 | threshold | C1–C4 |

### 2.2 Connections

Fixed:

| From → to | Weight | Threshold / note |
|---|---|---|
| ear layer-0 feature neurons → F | 0.001 × reserve | θ_F = 0.01 |
| F → F1 | 1 | θ = 1 |
| F, F1, O, GO → O; H_m → O | +2, −2, +4, −6; −1.5 each | θ_O = 2 |
| O, F → GO; H_m → GO | +2, −2; −1.5 each | θ_GO = 2 (teacher fallback +9) |
| Y_c → D_m, m ∈ set(c) | set at recruitment, fixed | winner only |
| D_m → H_m (set), beat → H (hold 3 ticks) | structural | as the current hold |
| H_m → I_m | −2.0 (I tonic +1.2) | θ_I = 1 ⇒ I = ¬H |
| I_m → M_{m,l} | −10 | absolute |
| H_m, phase φ → M_{m,l} | ramp: on iff H_m ∧ l ≤ φ | as now |
| chains ear/mctx cortex, M/H/D/Y/END → hip (3.0), → PFC (0.3 × x) | 1:1 | I excluded (C9) |
| F → thalamus | −10 | existing |
| PFC/hip motor mirrors → M | 0.3 | existing; harmless (M gated by I) |

Learned (rule §2.4, birth weight 0, γ = 0.999 per 100 ticks, gated by C11):

| Path | Sources (active/tick) | Targets | η | Cap per target |
|---|---|---|---|---|
| hip→Y | hippocampal mirrors (≈1,000) | all Y + END | 1e-5 | 30 |
| ear→Y | ear cortex unmuted (≈718 in speech, ≈3 in silence), × reserve | all Y + END | 1e-5 | 10 |
| mctx→Y | motor cortex unmuted, × reserve | all Y + END | 1e-5 | 3 |
| think→Y (optional) | thinking layer (500) | all Y + END | 1e-5 | 3 |

Equilibrium totals (η × active × events per 100 ticks /(1−γ)): END from hip ≈ 0.96·1,000·1e-5·2.2/0.001 ≈ 21 < 30;
END from ear ≈ 0.67·718·0.5·1e-5·2.2/0.001 ≈ 5.3 < 10; every Y is rarer than END ⇒ caps do not bind [P]. Teacher cue
50 > Σ caps 46 ⇒ the cued program always wins [P]. All other paths: unchanged weights and budgets.

### 2.3 Motor stage per tick (S(t−1) = any H on at t−1; n = syllables in this turn)

```
GO(t)    = [2·O(t−1) − 2·F(t−1) − 1.5·ΣH(t−1) ≥ 2]                         (teacher fallback +9)
boundary = (GO(t) ∧ ¬S(t−1)) ∨ (S(t−1) ∧ φ(t−1) = 2)
y        = learned currents into Y and END from t−1 states (+ cue 50 in training)
if boundary:
   c* = argmax over recruited programs of y_c
   first = ¬S(t−1)
   stop  = ¬first ∧ ( y_END ≥ (1−U_b)^n · Σ_c ŷ_c + Σ_c cue_c   ∨   n = 12 )     (ŷ = learned part; the breath
                                                                                  discounts learned drive only)
   stop:  END pulse; H = 0; n = 0
   else:  Y_c* pulse; D pulse on set(c*); H = 1 on set(c*), 0 elsewhere; φ = 0; n += 1
else:     Y = D = END = 0; H(t) = H(t−1); φ += 1 if S(t−1)
I = ¬H ;  M_{m,l} = H_m ∧ (l ≤ φ) ;  mouth = Voice.mouth(muscles with any M on)
```
Recruitment: a cue for a set with no program assigns the next free slot (fixed Y→D = that set, learned inputs 0) —
the designer's "neuron generation", driven by use. Before any program exists the first boundary releases nothing.

### 2.4 Learning rule (unchanged, `Path.learn`)

i on at t−1 and j rising at t ⇒ w_ij += η·x_i/(1 + (w_ij − w0_ij)/w_ref), w_ref = 0.2;
i on at t and j on at t−1 ⇒ w_ij ← max(w_ij − η, w0_ij);
every 100 ticks: learned part × γ, then per-target budget scaling. All `back=False` exceptions are removed.
C11 multiplies η of the four decision paths by g(t) ∈ {0,1}: g = 1 at a boundary that carried a teacher cue (forward
term) and at the tick after it (back term); 0 otherwise. Every other path learns at every tick.

### 2.5 Timing protocol

Training: per (other line, reply) pair: `listen(line, pause=0)` → `say(reply)` (C10) → the END boundary is the first of
4 pause ticks → next pair. Test: `listen(prompt, pause=0)` + 60 free ticks (learning on; C11 keeps the decision paths
still without a teacher).

---

## 3. One dialogue tick by tick (other: 这是四川吗, ticks 0–14; reply: 对啊)

Ear window = 6 slices = 3 ticks, 2 new slices per tick; the other's voice enters the window in the same tick, the own
voice one tick after the mouth. F-currents: min–max over 200 lines / 200 replies [S, `t_fresh3.py`]; in-speech bound [P,
`t_fresh4.py`]: every one of the 3,970 chars activates ≥ 81 layer-0 feature neurons when ≥ 4 of its slices are in the
window (≥ 36 with 2 slices); with the worst possible reserve x* = 1/(1 + U(τ−1)) = 0.286 the current is
≥ 0.001·81·0.286 = 0.023 > 0.01; silence activates 1 feature neuron.

| t | window | F | F1 | O | GO | ΣH | motor stage | mouth | learning into Y/END |
|---|---|---|---|---|---|---|---|---|---|
| 0 | ·· ·· 这₀₁ | 0 | 0 | 0 | 0 | 0 | — | — | — |
| 1 | ·· 这₀₁ 这₂₃ | **1** (≥0.036) | 0 | 0 | 0 | 0 | — | — | — |
| 2 | 这 | 1 | 1 | **1** (2−0) | 0 (2−2) | 0 | — | — | — |
| 3–14 (e) | 是四川吗 | 1 (≥0.023 [P]) | 1 | 1 (0+4) | 0 (2−2) | 0 | no boundary ⇒ H=0, I=1, M=0 | — | none (no target fires) |
| 15 (e+1) | 吗₂₋₅ ·· | 1 (0.074–0.124) | 1 | 1 | 0 | 0 | — | — | — |
| 16 (e+2) | 吗₄₅ ···· | 1 (0.048–0.084) | 1 | 1 | 0 | 0 | — | — | — |
| 17 (e+3) | silence | 1 (0.022–0.045) | 1 | 1 | 0 | 0 | — | — | — |
| 18 (e+4) | silence | **0** (≤0.0015) | 1 | 1 (0+4) | 0 (2−2) | 0 | — | — | hip(18) = source of the first syllable |
| 19 (e+5) | silence | 0 | 0 | 1 (−2+4) | **1** (2−0) | 0 | **first boundary**: Y_对 wins (cue in training) → D pulse, H on its 4 muscles, M level 0 | **对** | fwd: hip(18), ear(18), mctx(18) → Y_对 |
| 20 | ·· ·· 对₀₁ | 0 | 0 | 0 (4−6−6) | 0 | 4 | hold, φ=1 | 对 | back: sources on at 20 lose η (4 % of hip(18) [S]) |
| 21 | ·· 对₀₋₃ | 1 (own) | 0 | 0 (2−6) | 0 | 4 | hold, φ=2 | 对 | — |
| 22 | 对 | 1 | 1 | 0 | 0 | 4 | **boundary**: END vs (1−U_b)¹Σy; Y_啊 wins | **啊** | fwd: ear(21)=[··│对₀₋₃], hip(21), mctx(21) → Y_啊 |
| 23–24 (L) | 对₂₋₅ 啊₀₁ … | 1 | 1 | 0 | 0 | 4 | hold | 啊 | back at 23: 33 % of ear(21) [S] |
| 25 (L+1) | 啊 | 1 | 1 | 0 | 0 | 4 | **boundary**: END wins (cue in training) → H=0 | — | fwd: ear(24), hip(24), mctx(24) → END |
| 26–28 | own tail | 1 (L+4 ≥ 0.013) | 1 | 0 (2−2) | 0 | 0 | idle | — | — |
| 29 (L+5) | silence | 0 (≤0.0015) | 1 | 0 | 0 | 0 | idle | — | — |
| 30 | next line ·· X₀₁ | 1 at 30 | 0 | **1** at 31 (rising edge, ΣH=0) | 0 | 0 | — | — | — |

The protocol's 4-tick pause leaves exactly one F-off tick (L+5) before the next line: enough for the rising edge [P].

---

## 4. Behaviours B1–B7

**B1 Listening (mouth still) — [P], confirmed [S].** (i) M fires only under H (I on ⇒ input − 10 < 1; M's only other
input is the 0.3 + 0.3 return). (ii) H changes only at a boundary; a boundary needs GO (when silent) or S (when already
speaking). (iii) GO needs F(t−1) = 0, and F = 1 from the line's 2nd tick to e+3 for any char and any repetition (bound
above; 30-tick repetition of one char ≥ 0.060 [S]). (iv) Learned currents reach only Y/END, which act only at
boundaries. Toy: 0 mouth ticks while listening in every training and test line of runs b, y1, y2 (≈ 3,700 lines) [S]; in run a the unanswered 3rd line of 3-utterance dialogues made the model take an uncued turn that overlapped the next line — the reason for C10.
Residual [A]: an other who starts while the model is still speaking is talked over (cannot happen in the protocol:
≤ 12 syllables = 36 ticks < 60 free ticks; training ends every turn with END).

**B2 Holding what was heard — [S], weak.** The hippocampus is unchanged: chain imprint at e+1…e+3, then fast-weight replay
that cycles back to the line's start (75 % of e+5 states best-match the first 30 % of the line [S, `t_hip3.py`]).
Decoding 20 lines at GO−1 = e+4 under real-time learning: 1.00 when the model has been silent, 0.70 when it speaks after
every line (its own replies interfere); Jaccard same line 0.27–0.31 vs different lines 0.16–0.19 [S]. So ≈ 12 % of the
≈1,000 active neurons are line-specific and reproducible. **Correction to STATE §4.2**: the 0.95–1.00 measured by
`check()` comes from learning being off — with frozen weights the after-pause state is a deterministic function of the
line (same-line Jaccard 0.98 at every tick e+1…e+6) [S, `t_hip2.py`]; it is not evidence of a strong held memory.
The source is needed only until GO−1; afterwards the own voice displaces it, which is exactly what keeps the link (§5).

**B3 Turn-taking — [P], confirmed [S].** §3: F(e+3) = 1 (≥ 0.022) and F(e+4) = 0 (≤ 0.0015) for every line [S over 200
lines; e+4 is [P]: silence drives 1 feature neuron]. O was set at the line's rising edge (ΣH = 0) and holds through e+5
(−2 + 4 = 2 ≥ 2). GO(e+5) = 2 − 0 − 0 = 2 ⇒ the first boundary is at e+5 for every line, and it always releases a
program (no threshold). Toy: latency exactly 5 in 60/60 untrained and 1,600/1,600 trained test lines; in training GO fired by
itself at e+5 before all 2,169 replies of runs b, y1, y2, 0 fallbacks [S]. No second turn: GO resets O (4 − 6 < 2); while speaking
ΣH = 4 blocks O (2 − 6); after speaking F stays on through L+4 (≥ 0.013) so F − F1 = 0 and O stays 0 [P with the
measured tail]; toy: 0 re-starts [S].

**B4 Content — mechanism [P], size [S] (§5, §7).** The first syllable is argmax_c Σ_{i∈hip(e+4)} w_ic (+ think): a
deterministic, learned function of the held state; the next ones are argmax over ear/hip/mctx context.

**B5 Articulation — [P], confirmed [S].** A release turns on exactly the 4 muscles of one recruited program, holds them
exactly 3 ticks (φ = 0,1,2), and switches to the next set in the same tick (I and M computed from the new H). The mouth
sees an exact char set ⇒ `Voice.mouth` returns that char (3,970/3,970 [S]); "~" is impossible. (With a muscle-level
top-4 instead, a random 4-set would still decode with P(~) = 5.5·10⁻⁵ [S, `t_mouth.py`].) Toy: 0 garbled ticks [S].

**B6 Stopping — termination [P], content stop [S, weak].** n ≤ 12 per turn and no second turn (B3) ⇒ ≤ 12 syllables per
heard line [P]. END is learned like a program: each continuation event raises one Y, each reply end raises END, so
y_END / Σ_c y_c would track P(end | context)/P(continue | context) for a count learner with non-binding caps [A]. Toy (500
dialogues): **END learned no content yet** — at matched positions the raw ratio y_END/Σy is 0.098–0.102 at true ends vs
0.097–0.099 mid-reply (n = 2…8) [S, `log_y1.pkl`]; corpus ceiling: P(end | last char) 0.33 vs 0.12 [S]. So, for now,
length is set by the breath: stop when 0.1 ≥ (1−U_b)^n ⇒ n ≈ ln 0.1 / ln(1−U_b): 14 → capped at 12 for U_b 0.15 (seen:
all replies 12), 8 for 0.25, 6 (= corpus median) for U_b ≈ 0.32 [P from the measured 0.1; S].

**B7 Stability and real-time learning — [P].** Every path learns at every tick except the decision paths, which learn at
every teacher-confirmed boundary (C11). Bounds: decision-path totals converge to ≤ η·n_active·rate/(1−γ) (≤ 21, below
the caps); all other paths keep their budgets; cue 50 > Σ caps 46 ⇒ guided speech is exact at every stage. I, M, H, GO,
O, F, F1 have no learned input ⇒ silence while listening, turn-taking, articulation and the 12-syllable bound cannot be
unlearned. Activity: all big areas are top-k; the motor stage is ≤ 4 muscles and ≤ 12 syllables per heard line.

---

## 5. Learned links: what is on at t−1 when the target rises at t; how many dialogues

| Link | t−1 (sources on) | t (target rises) | back-term check (target is a 1-tick pulse) | kept [S] |
|---|---|---|---|---|
| hip→Y (first) | hip(e+4): ≈1,000 = tail imprint of e+1…e+3 + replay; line-specific ≈ 12 % | Y_c1 at e+5 (cue c1) | hip(e+6): own-voice imprint + motor mirrors | **96 %** (55 % if held 3 ticks) |
| ear→Y (next) | ear(B−1) = own voice [c_{k−2} slices 4–5 │ c_{k−1} slices 0–3], ≈718 unmuted, × reserve | Y_ck at B | ear(B+1) = [c_{k−1} 2–5 │ c_k 0–1]: same slices, other positions ⇒ other neurons | **67 %** (37 % if held) |
| hip→Y, mctx→Y (next) | hip(B−1): mirrors of ear/mctx/motor at B−2 + replay; mctx(B−1): force map of c_{k−1} at levels 0–1 | Y_ck at B | hip(B+1), mctx(B+1) | (not separated) |
| ear/hip/mctx→END | the same contexts at the last boundary | END at B_end (cue) | contexts at B_end+1 | as above |
| Y recruitment | — | first cue of a new set | — | one-shot, fixed Y→D |

Per event, Σ_i Δw ≈ (kept fraction) × (active sources) × η × x̄: first ≈ 0.96·1,000·1e-5 = 0.0096; next ≈ 0.67·718·0.5·1e-5
= 0.0024. Push at a test context ℓ: y_c(ℓ) = Σ_events (Δ per event) × overlap(ℓ, event context). Between different lines the
hip overlap at e+4 is ≈ 0.16–0.19 Jaccard (≈300 shared neurons), so the generic part of y_c grows ∝ N_c, the number of
replies that started with c: the **corpus mode wins first** (你, 9.4 % of replies). A line-specific push beats it only if
spec_c > 0.19·(N_你 − N_c) — rarely, with this corpus.

Dialogues needed (toy, 500 dialogues, 723 replies; own decision before the cue vs the teacher's char) [S, `run_y1.log`]:

| dialogues | first syllable = teacher's | next syllable = teacher's | y_END/(breath·Σy) at true end / mid (position effect only, see B6) |
|---|---|---|---|
| 1–100 | 0.044 | 0.015 | 0.24 / 0.16 |
| 101–200 | 0.120 | 0.033 | 0.25 / 0.16 |
| 201–300 | 0.088 | 0.036 | 0.26 / 0.17 |
| 301–400 | 0.094 | 0.051 | 0.26 / 0.17 |
| 401–500 | 0.079 | 0.045 | 0.24 / 0.17 |
| ceiling | 0.095 (online "most frequent so far") | 0.112 (ideal bigram counts on 500 dialogues), 0.193 (30k) | 0.33 / 0.12 (P(end), 30k) |

So: the prior is learned in ≈ 100–200 dialogues. The next syllable is barely above context-free: dialogues 201–500, own
choice 0.044 vs most-frequent-so-far 0.036 vs an online count-bigram learner 0.115 (N = 2,291, SE ≈ 0.004) [S]: the
shared features of the ear code (char-to-char Jaccard ≈ 0.19 after muting) give every context a large generic push ∝ the
marginal, which the specific push (∝ bigram count) beats only for frequent pairs; line-specific first syllables (beating shuffled pairs) are not measurable at 500 and, by the corpus ceiling
(+1.7 points at 30k with an ideal bag-of-chars learner), need ≥ 10⁴ dialogues and ≥ 2,000 test lines to detect [A,
extrapolated from `corpus_stats2.py`].

---

## 6. The failures of STATE §4, one by one

| # | Failure | In this design |
|---|---|---|
| 1 | Ear afterimage ≈4 ticks | Not relied upon; F now reads layer 0, which no imagery or learned current can reach [P]. |
| 2 | Hip decodes 0.95–1.00 | Kept and used as the first-syllable source; but it is a frozen-weights effect — under learning it is 0.70–1.00 at e+4 with ≈12 % specific neurons [S]; this is the main limit on B4 (§8). |
| 3 | PFC receiving decodes 0.15–0.60 | No proof depends on it; C9 removes its 32 always-winning I mirrors. |
| 4 | Thinking layer ≈ chance untrained, 0.35–0.90 trained | No proof depends on it; think→Y is optional; if it becomes content-specific it adds to the first syllable through the same gated rule. |
| 5 | 2-tick loops alternate (period 2) | Irrelevant: no behaviour needs a held loop; the content source only has to exist at GO−1. |
| 6 | Fresh on ≈2 ticks after the end; teacher go braked; thalamus late; thinking drifts | F now has a deterministic offset at e+4 for every line [P/S]; GO is structural at e+5; the teacher waits for the model's GO, so there is no brake conflict (0 fallbacks / 2,169 replies [S]). |
| 7 | Rule cannot teach held thought → target | Not needed: GO is structural; content links run from sources that turn over when speech starts into 1-tick pulse targets (96 % / 67 % kept [S]). No `back=False` exception remains. |
| 8 | First-syllable content ≤ 0.6 < 0.8 needed | There is no threshold at the first boundary: content only has to order the programs (argmax). |
| 9 | Guided speech 100 % → 61–83 % | Learned input reaches only Y/END; cue 50 > Σ caps 46; I, M fixed ⇒ guided speech exact at every stage [P]. |
| 10 | Never opened its mouth | Opens at e+5 after every line [P]; 1,600/1,600 trained test lines [S]. |
| 11 | Budgets stable; featureless muted | Kept; muting extended to tonic I (C9); decision-path caps above equilibrium (§2.2). |

New findings from the simulations that any design must handle: (a) fresh θ 0.4 = in-speech median ⇒ dropouts;
(b) STATE's hippocampal decoding is a frozen-weights artifact; (c) mirrors of tonic neurons teach a context-free prior;
(d) a 32-muscle top-4 readout cannot express a char distribution; (e) unanswered lines trigger uncued turns; (f) ungated
real-time learning of one's own speech overwrites the learned prior.

---

## 7. Simulations (this folder; numpy 1.26, one core; ear and hippocampus at full size)

Replica: `ear.py` (Voice/Stream copied from world/voice.py; Net3D.respond ported; warm-up muting; reserve; fresh),
`brainsim.py` (motor cortex Net3D 8×4; hippocampus mirrors with chain 3.0, k 96, η 0.5, γ 0.7, budget 2.0, top-k 5 %;
proposal motor stage with muscle-level decision; turn circuit; learned decision paths; `Circuit.step` order),
`brainsim_y.py` (syllable-program decision, C5; gate C11). Omitted: PFC/thinking/thalamus, ear internal learning,
relays/returns (not used by any proof; they could change the hippocampal replay — §8).

| File | Question | Result |
|---|---|---|
| `t_fresh.py`, `t_fresh2.py` | Current fresh unit (cortex, θ 0.4) | in-line median 0.435; 235 one-tick + 186 ≥2-tick dropouts in 300 lines; off at e+1 (67) / e+2 (233) |
| `t_fresh3.py` | Layer-0 fresh unit | in-line min 0.069; e+3 0.022–0.045; e+4 ≤ 0.0015; own L+4 ≥ 0.013, L+5 ≤ 0.0015; silence ≤ 0.0005 |
| `t_fresh4.py` | Worst case over all 3,970 chars | ≥ 81 / ≥ 36 active feature neurons (4 / 2 slices); 30-tick repetition ≥ 0.060 |
| `t_turn.py` | B1/B3/B5 untrained | 0 mouth ticks listening; latency 5 for 60/60; 0 re-starts; 0 garbled |
| `t_hip.py`, `t_hip2.py`, `t_hip3.py` | Hippocampus at the turn | frozen weights: 0.95 at every tick, same-line J 0.98; learning on: 1.00 (silent model) / 0.70 (speaking) at e+4, J 0.27–0.31 vs 0.16–0.19; replay returns to the line start |
| `t_back.py` | Back-term survival | pulse 0.96 (hip→first), 0.67 (ear→next); held 0.55, 0.37 |
| `t_mouth.py` | Garble with 4 released muscles | 5.5·10⁻⁵ (random sets); exact sets 3,970/3,970 |
| `t_readout.py` | Muscle vs program readout with ideal counts | first: 根 0.000 vs 你 0.091; next: 0.141 vs 0.193 |
| `corpus_stats*.py` | Corpus ceilings | first char vs heard line: 9.5 % vs 7.8 % shuffled (30k); copying 15.8 % vs 5.7 %; bigram top-1 0.112 (500) / 0.19 (30k); P(end) 0.33 vs 0.12 |
| `t_learn.py` (run a) | Muscle readout, with I mirrors, 3-utterance dialogues | every reply "再" (193/200); 69 fallbacks / 583 replies |
| `t_learn2.py` (run b) | Muscle readout, C9 + C10 | 0 fallbacks; own top-4 muscle hits ≈ 0.70 (first) / 0.65–0.68 (next) vs context-free most-used baseline 0.65–0.76 / 0.63–0.65 (chance 0.50); test: one fixed 12-syllable string "再再儿儿c叟…" for all lines |
| `t_learn_y.py` (run y1) | Program readout, ungated, U_b 0.15 | first syllables 你 183 / 我 17; first = true 0.070 (shuffled 0.070); said chars in true reply 0.142 (shuffled 0.142); "你的了的了…" to the cap; with learning on in the test: first syllable drifts to 的 114 / 了 84 |
| `t_learn_y.py` (run y2) | Program readout, gated (C11), U_b 0.25 | training identical to y1 (all training releases are cued, so the gate is open); held-out test (learning off): latency 5 (200/200), 0 garbled, 0 mouth ticks listening, **length 8 for 200/200** (= ln 0.1 / ln 0.75, as predicted from the raw END ratio), first syllables 你 184 / 我 16, first = true 0.075 (shuffled 0.070), chars in true reply 0.144 (0.144); trained lines 0.135 (0.125), 0.163 (0.160) |
| `t_gate.py` (300 dialogues, gated, U_b 0.25; training identical to y1's first 300) | Does the gate stop the drift under real-time learning? | the model's own offline first choice at dialogues 251–300 was 我 (71/72) [`log_y1.pkl`]; free speech with learning on: 我 48/50, 50/50, 47/50 in three successive blocks of 50 lines, latency 5, 0 mouth ticks listening → **no drift**; ungated (y1): 你 → 的 114 / 了 84 within 200 lines. Side effect visible here: next syllables are the generic favourite ("我我我…": own next choice at 251–300 was 我 in 239 of ≈460) |

Commands: `python3 t_learn_y.py 500 y1 eta_hip=1e-5 eta_ear=1e-5 eta_ctx=1e-5 B_hip=0.0 B_ear=0.0 B_ctx=0.0 cue=100.0 U_b=0.15`;
y2 = same with `U_b=0.25 gate_decision=True`; run b = `t_learn2.py 500 b`; run a used an earlier stop rule
(END ≥ mean of the top-4 D) and `mute_motor=False` (qualitatively reproduced by `t_learn.py 400 a mute_motor=False`).
Each 500-dialogue run takes 6–10 min on one core.

---

## 8. Weakest points and what to measure first on the GPU

1. **Content is weak (B2/B4).** The hippocampus keeps ≈12 % specific neurons at GO−1, and the corpus gives ≈ +1.7 points of
   conditional information at 30k dialogues. Expect the corpus mode first (你/我), bigram chains next; "above shuffled"
   will take ≥ 10⁴ dialogues and ≥ 2,000 test lines to see. Measure: first-syllable distribution; "first syllable in the
   true reply" vs shuffled; own-decision match during training (as in §5). If it stays flat, the next lever is the
   hippocampal context input or a sparse own layer (synthesis step 7) — not part of this minimal set.
2. **Generic features dominate the learned evidence; greedy chains loop** ("的了的了…"). The marginal push is ∝ the shared
   features, the specific push ∝ the bigram count. Cheapest lever inside the framework [A, untested]: the existing set
   point (mechanism 1c, `Region.base`, already used in the PFC) on Y — programs that win more often than their target
   rate get a slow penalty, which turns the readout from "most frequent" toward "most specific" (PMI-like) and breaks
   2-cycles. The real fix is sparser sensory codes (synthesis steps 6–8).
3. **Stop quality.** END shows no content yet (0.100 vs 0.098 raw); length is set by U_b (≈ 0.32 for the corpus median 6).
   Measure the length distribution and the raw END ratio at matched positions on the GPU run.
4. **Toy omissions.** No PFC, no ear internal learning, no relays/returns: the real hippocampal replay may differ; F is
   immune by construction (layer 0), the turn timing is not. Re-measure §3's F values and the hip overlaps in the real
   model before training.
5. **Protocol margin.** PAUSE 4 leaves exactly one F-off tick before the next line; PAUSE 5 gives a margin.
6. **New mechanisms the designer must accept**: program recruitment (one-shot fixed wiring on first use) and a third factor
   on the decision paths (teacher confirmation). Both are "tools equivalent to hand-wiring" in the designer's terms, but
   they are new. Y is a localist motor code (one unit per program), in tension with designer item 8 ("not one neuron per
   char") — defensible for an action repertoire (syllable-specific premotor/HVC cells), and a sparse distributed version
   (8 of 4,096 neurons per program, crosstalk ≈ 56·mean evidence vs 8·own evidence) behaves almost identically [A].
7. **GPU cost**: dense decision paths ≈ (20k + 17k + 3k + 10k) × 4,096 ≈ 200M weights (0.8 GB fp32 without w0, since w0 = 0)
   — fits a V100, but needs a dense path class instead of the (src × K) index table.

---

## 9. Claims ledger

| Claim | Label |
|---|---|
| F (layer 0, θ 0.01) is on during every tick of any line from its 2nd tick, for all 3,970 chars | P (enumeration + reserve bound) |
| F turns off at exactly e+4 after every line; own voice keeps it on to L+4 | S (200 + 200 lines); e+4 bound P |
| No mouth movement while the other speaks | P (given F) + S |
| Mouth opens at e+5 after every line; never a second turn | P + S |
| Exactly one clean char per 3 ticks | P + S |
| ≤ 12 syllables per heard line | P |
| Decision links form under the unchanged rule; 96 % / 67 % of increments survive with pulse targets | S |
| Program readout is required for a char-level first syllable | S (ideal-count readout) |
| Self-produced speech must not train the decision paths; the gate suffices | S (ungated drift 你 → 的/了; gated 145/150 kept) |
| Learned weights bounded; guided speech exact at every stage | P |
| Prior learned in 100–200 dialogues; next syllable only +0.8 points over context-free at 500; END no content at 500 | S (toy) |
| While END has no content, length = ln(raw END ratio ≈ 0.1)/ln(1−U_b) (8 for U_b 0.25) | P from the measured ratio; S (200/200 replies of length 8) |
| Current fresh unit (θ 0.4) drops out inside lines | S (≥ 600 of 5,018 in-line ticks) |
| STATE's hippocampal 0.95–1.00 decoding reflects frozen weights | S (frozen 0.95 / J 0.98; learning on 0.70–1.00 / J 0.27–0.31) |
| Line-specific content beats shuffled only after ≥ 10⁴ dialogues | A (corpus extrapolation) |
| The real model's hippocampus behaves like the replica at the turn | A (replica omits returns/PFC) |
