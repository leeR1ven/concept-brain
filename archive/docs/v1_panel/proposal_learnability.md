# Hold by structure, learn from one-tick transients
## An order-learnable listen → hold → speak circuit (learnability-first proposal)

Panel member: learnability angle. Claims are marked **[P]** proved (explicit argument with numbers), **[S]** simulated
(toy run in this folder; result given), **[A]** assumed. All toys use numpy, the exact `Path.learn/decay` semantics of
`brain/net.py`, the update order "all currents from t−1 states → all parts fire → learning → reserve/set point", the real
`world/voice.py` (copied as `voice_copy.py`), a line-by-line numpy replica of `net3d.Net3D` (`ear_np.py`), and the real
corpus. Files: `/tmp/claude-0/-root/e92d3d7d-5823-4f5c-a0ec-4fe302bff9af/scratchpad/arch/learnability/`.

---

## 0. Summary

1. The rule stores i→j only if i is on at t−1 **and off during the whole time j stays on** (Lemma 1). So every learned
   link in this design has a one-tick source or a source that ends exactly when its target starts; nothing that is held
   is ever asked to teach. Holding, beat, start and stop timing are structure.
2. Thought **T** = fixed, reserve-weighted random expansion of the ear's input layer (layer 0) + a one-tick self-hold
   (h = 3). Untrained, it identifies 20/20 probe lines at the readout tick, holds exactly (consecutive-tick Jaccard 1.00
   from the 3rd silent tick), has no shared core (across-line Jaccard 0.04), and is a recency-weighted bag of the
   line's last ~8 characters **[S]**.
3. **When**: F (someone else is talking) → offset cell S1 → GO; S1→GO is learned in ≈ 50 dialogues **[P][S]**.
   **What first**: a one-tick output-gated copy **P = T ∧ GO** → learned P→Y. **What next**: learned Y→Y.
   **Stop**: learned Y→END. In training, none of these sources is on while its target is on, so the back term never
   cancels them (0 back-term hits on P→Y in 2,000 simulated dialogues) **[P][S]**.
4. The model selects among **syllable programs Y** (one unit per executed 4-muscle combination), and the D/I/M neurons
   are slaved to the winner. Selecting per muscle was simulated to turn every learned mixture into an unrelated "clean"
   character (根, 箍), and content dropped to chance **[S]**.
5. End-to-end toy (real ear, real protocol, 2,000 dialogues): 0 mouth moves while the other speaks; a reply to 20/20
   probe lines exactly 7 ticks after the line; 0 garble; every reply stops (by END or the 0.02 floor, never the
   backstop); no second start after its own reply; teacher-cued speech exact **[S]**. On the corpus, content is above the shuffled-pair baseline (first syllable +2 to
   +4 points) but weak **[S]**. The weak content is the main open problem, not the mechanics.

---

## 1. What the rule can and cannot learn

The rule as coded (`Path.learn`, `Path.decay`). For a link i→j, after all parts fire at tick t:
- forward: if s_i(t−1)=1, s_j(t)=1 and s_j(t−1)=0, then w ← w + η·x_i(t−1)/(1+(w−w0)/w_ref);
- back: if s_i(t)=1 and s_j(t−1)=1, then w ← max(w−η, w0);
- every 100 ticks: L ← γL; then for each target, Σ_i L_ij ≤ B (proportional rescale).

**Lemma 1 (cancellation) [P].** Let j rise at t and stay on during [t, t+d−1]. Then over that episode
Δw_ij = η·x_i(t−1)/(1+L/w_ref)·s_i(t−1) − η·#{t' ∈ [t+1, t+d] : s_i(t') = 1}, floored at w0.
Reason: the back term fires at t' iff s_j(t'−1) = 1, i.e. exactly for t' ∈ [t+1, t+d].
Consequences:
- (a) The link is learnable iff s_i(t−1) = 1 and s_i = 0 on [t+1, t+d]. Being on at t itself is harmless.
- (b) A source held across the onset (on at t−1 and at t+1) nets ≤ η(x−1) ≤ 0, so w stays exactly at w0. This is
  evidence #7 (think→go ≈ 0).
- (c) Each extra tick on which i is on while j is still on costs another −η.

**Lemma 2 (read once) [P].** Suppose a gated copy P of a held state fires one tick before every boundary of a
multi-syllable reply. Its k-th readout falls on the last tick of syllable k's 3-tick hold. Units present in two
consecutive readouts then get +η at boundary k and −η at readout k+1, so they net 0. A held state can therefore teach
through its gated copy **once per reply**: the first syllable.

**Lemma 3 (growth) [P].** With one event per step, x = 1 and w0 = 0: L + L²/(2w_ref) ≈ η·n.
With η = 0.05 and w_ref = 0.2, L = 1.0 after 70 events and L = 2.0 after 240.

**Lemma 4 (decay) [P].**
- Turn pair with REPLY_GAP 7: 3·6.4 + 7 + 3·6.2 + 4 ≈ 49 ticks (corpus means: other 6.4 chars, reply 6.2). That is one
  decay step per ≈ 2 turn pairs.
- Time constant: 1/(1−γ) = 1000 decay steps = 10⁵ ticks ≈ 2,050 turn pairs ≈ 1,400 dialogues (γ = 0.999,
  1.45 turns per dialogue).
- Equilibrium of a link refreshed r times per decay period: L* + L*²/(2w_ref) = η·r/(1−γ).

**Lemma 5 (per-target budget removes the prior) [P, S].** If target y's learned input saturates at B, then
Σ_{p∈P} L_py ≈ B·Pr(pattern | y), so y's frequency stops counting; an unsaturated target keeps its raw counts η·n.
Simulated on the corpus (`toy_content2.py`, `toy_reply.py`):
| per-target budget on P→Y | Result |
|---|---|
| B = 1, η = 0.01 (binds even for single events) | rare targets win; output at chance |
| B = 20 (never binds) | the prior mode for every line ("你是啊" for all 2,000 test lines; difference from shuffled 0.000) |
| B = 2 with η·K_T = 0.1 per event | the most frequent first syllables saturate, rare ones keep counts; best content (Table 8) |

**Lemma 6 (selecting per muscle is lossy) [S].**
- Every char needs a unique 4-of-32 muscle set; 4,782 of the 4,960 possible muscle triples occur in some char
  (3.3 chars per triple on average).
- So *any* 4-muscle set is decoded by `Voice.mouth` as some char (20,000 random sets: 100%), and only 10.8% are
  exactly a char.
- A readout that sums evidence per muscle and releases the top 4 therefore turns a mixture of plausible syllables into
  an unrelated, clean-sounding char. Simulated outputs: 根, 箍, 佳, with in-reply minus shuffled ≤ +1.7 points; the
  syllable-level readout reaches +3.3 to +4.7 points with the same data.

**Design principles that follow.**
- P1: hold by structure (self-hold, held syllable), never by learning.
- P2: every learned source is a one-tick pulse or ends exactly at the boundary.
- P3: a held state teaches only through a one-tick, content-free-gated copy, read once per reply.
- P4: select and learn at the level whose identity must survive (syllable programs); slave the muscles to it.
- P5: "when" (GO, beat, END) and "what" (P→Y, Y→Y) are separate inputs, both learned by the same rule.
- P6: size budgets so that frequent targets saturate and rare ones do not.

---

## 2. Architecture (complete)

### 2.1 Areas
| Area | Neurons (GPU / toy) | Activity | vs current code |
|---|---|---|---|
| Ear (3D net, 4 layers) | 1920 / 2880 / 5760 / 11520 alive (+25% spare); cortex (layers ≥1) 20,160 | fixed fractions 0.4/0.3/0.2/0.1 (top-k) | unchanged (internal plastic path, reserve U 0.5 τ 6, muting, split/delete) |
| FEAT0 = ear layer 0 minus featureless (usage > 0.6) | 1153 (measured; 767 of 1920 muted) | — | new *readout set*: the only ear layer that no learned or return current reaches |
| Motor cortex mctx | 640 / 1280 / 2560 | fixed fractions | kept as proprioception → hippocampus only; **no output to the motor area** |
| Hippocampus | ≈ 24k mirrors, 5% | top-k; chain 3.0; internal η 0.5, γ 0.7, B 2 | kept, **off the B1–B7 path** (no ⇒motor returns); its relays to the ear may stay |
| F (other is talking) | 1 | threshold | input, threshold and veto changed |
| F2 | 1 | F delayed one tick | new |
| S1…S8 (offset + time cells) | 8 | threshold | new |
| V1…V4 (own-voice veto) | 4 | delay line | new |
| GO | 1 | threshold; one tick | inputs changed |
| T (PFC thought) | 10,000 / 4,000 | top-K = 5% (500 / 200) | rewired (replaces think layer + thalamus) |
| P (readout, output gate) | 10,000 / 4,000 | threshold 2 | new |
| Beat ring R2→R0→R1 | 3 | threshold | `MotorArea.phase` made explicit |
| Y (syllable programs) | ≤ 3,970, recruited | winner-take-all at R0, held 3 ticks | new |
| END | 1 | competes in the same winner-take-all | new |
| D, I, M | 32, 32, 96 | same tick as Y | slaved to Y; no learned or return input |

Removed:
- PFC receiving layer (24k) and its relays (its job, the chain into PFC, is T's input);
- thalamus (2000) with fresh⇒thal;
- think→think, pfc→think, think→pfc;
- the learned paths think→go, ear→go, think→motor, ear→motor, mctx→motor and motor-internal;
- pfc/hip⇒motor returns;
- the free D firing outside the beat.

Why removed: Lemma 1 says these paths learn ≈ 0 or interfere (evidence #3, #4, #7, #9).

### 2.2 Fixed (born) connections and unit rules
All inputs are read from states at t−1 unless marked "same tick" (the motor same-tick comparison, as today).

| Unit | Rule (fires iff …) | Notes |
|---|---|---|
| F | 0.001·Σ_{i∈FEAT0} s_i(t−1)·x_i(t−1) − 10·Σ_k V_k(t−1) ≥ 0.008 | measured ranges below |
| F2 | F(t−1) | |
| S1 | F2(t−1) − 2F(t−1) − 2ΣV(t−1) ≥ 1, i.e. F(t−2) ∧ ¬F(t−1) | the offset of the other's voice |
| S_{k+1} | S_k(t−1) − 2F(t−1) ≥ 1 | one pulse travels S1 (L+4) … S8 (L+11) |
| V1 | Σ_m D_m(t−1) ≥ 1 | V_{k+1}(t) = V_k(t−1) |
| GO | Σ_k w_k S_k(t−1) − 5F(t−1) − 5·[any D](t−1) + ext ≥ 1 | no self-input: always a one-tick pulse |
| T_j | top-K of: Σ_{i∈in(j)} s_i(t−1)x_i(t−1) + 3.0·T_j(t−1) − 0.3·max(0, u_j/0.05 − 1) + ε_j | in(j) = 30 random FEAT0 neurons, weight 1; ε_j ~ U(0, 0.01) fixed; set point u_j += (T_j − u_j)/1000 |
| P_j | T_j(t−1) + GO(t−1) − 2F(t−1) ≥ 2 | a copy of T on exactly the tick after GO |
| R2 | GO(t−1) − 2F(t−1) + 2R1(t−1) ≥ 1 | |
| R0 | R2(t−1) + cue(t) ≥ 1 | the boundary; the teacher's cue also forces it |
| R1 | R0(t−1) + x_R(t−1)·Σ_y Y_y(t−1) ≥ 1.5 | continues only if a syllable was chosen; x_R = breath reserve (U 0.06, τ 100 ticks), stops the ring after ≤ 15 syllables **[P]** |
| D_m (same tick) | 2.5·Σ_{y: m∈set(y)} Y_y(t) ≥ 2.0 | slaved to Y; the teacher's cue enters only through the Y score (below), which it always wins |
| I_m (same tick) | 1.2 − 2·D_m(t) ≥ 1 | |
| M_{m,l} (same tick) | D_m(t) ∧ l ≤ phase(t) | force ramp 1→2→3; phase = ticks since R0 |

F's measured current (300 corpus lines, real ear replica, `toy_layer0.py`) **[S]**:
| Moment | Current |
|---|---|
| any tick of the other's line | ≥ 0.069 (p5 0.084) |
| silent ticks 0 / 1 / 2 after the line | ≥ 0.074 / 0.047 / 0.022 |
| silent tick ≥ 3 | ≤ 0.001 |
- So with threshold 0.008, F is on from the 2nd tick of the line through L+2 and off from L+3 (L = first silent tick).
  The margins are 2.75× and 8×.
- The model's own voice has the same profile (own-voice silent tick 2 ≥ 0.013); V vetoes it.
- The current code's F (cortex, threshold 0.4) was off on 36% of speaking ticks, because speaking currents range
  0.28–0.65 **[S]**.

**Syllable programs Y.**
- Creation: when a cued boundary executes a muscle set with no Y unit yet, a Y unit is allocated. Its born links go to
  the 4 D's of that set (2.5 each) and its learned inputs start at 0.
- Deletion: units unused for a long time are deleted (the framework's split/delete mechanism).
- Hold: once chosen at boundary b, a Y unit stays on during b, b+1, b+2 and is released at b+3.

**Boundary rule (at every R0 tick).**
- Each recruited Y gets the score
  sc_y = Σ_p L^{PY}_{py}·P_p(t−1) + Σ_{y'} L^{YY}_{y'y}·Y_{y'}(t−1) + 5·|set(y) ∩ cue(t)|.
- END gets sc_END = Σ_{y'} L^{YE}_{y'}·Y_{y'}(t−1) + 10·endpush(t).
- The winner is the argmax.
- If the winner is END, or its score is below θ_rel = 0.02, or the breath has run out, nothing is chosen: the ring
  stops and the mouth is silent. Otherwise the winner's 4 D's open for the beat.

### 2.3 Learned connections (all use the one rule; back term on)
| Path | η | w_ref | γ per 100 ticks | Budget per target | Fan |
|---|---|---|---|---|---|
| S_k → GO | 0.05 | 0.2 | 0.999 | 2.0 | 8 → 1 |
| P → Y | 0.0002 GPU (K_T 500) / 0.0005 toy (K_T 200), so that η·K_T = 0.1 per event | 0.2 | 0.999 | 2.0 per Y | dense 10k × ≤3970 (160 MB) |
| Y → Y | 0.01 | 0.2 | 0.999 | 2.0 per Y | dense ≤3970² (63 MB) |
| Y → END | 0.01 | 0.2 | 0.999 | **none**: bounded by decay ≤ 2.73 **[P, Lemma 4]** | ≤3970 → 1 |
| Ear internal, hippocampus internal, ear↔mctx | unchanged | | | | not on the B1–B7 path |

Why END has no budget: END is by far the most frequent target (once per reply), so a budget would always bind on it.
It would then be scored on the likelihood scale while syllables are scored on the count scale (Lemma 5). Simulated
with END budgeted:
- at γ = 0.999, END stopped 68% of replies;
- at γ = 0.9999, END stopped 1/2000 and 271/2000 replies ran into the backstop.

Without a budget, END stops 2000/2000 at γ = 0.999 and 1958/2000 at γ = 0.9999 **[S]**.

### 2.4 Protocol (training and test)
- Other's line: unchanged.
- **REPLY_GAP 5 → 7.**
- **Teacher GO push at L+5** (the 6th silent tick), not at the first reply tick.
- Cue at r0 + 3k: +5 to the D's of the syllable (was 2.5), which also forces R0.
- **END push (+10) at the boundary after the last syllable.**
- PAUSE 4; learning always on.
- Test unchanged: listen with pause 0, then 60 free ticks.

Why 7: L+7 is the earliest tick at which the model itself can start. F is off at L+3, S1 fires at L+4, GO at L+5,
P at L+6 and the boundary falls at L+7. With REPLY_GAP 7, training and test share one timeline. S2…S8 let any gap
from 7 to 14 be learned instead.

---

## 3. Tick by tick: B1–B6 with numbers

The example is a probe line, "这是四川吗" (5 chars, other's voice on ticks 0–14, L = 15), with the reply "对啊".
The model has had 200 or more training dialogues. Training differences are in brackets.

| t | Ear input | F current → F | S / GO | T | P | Ring | Y / D / mouth | Learning [training] |
|---|---|---|---|---|---|---|---|---|
| 0 | 这, 2 slices | from silence: ≈0 → 0 | S off; GO current 0 | starts replacing part of its units each tick | 0 | idle | none; I on; mouth silent | — |
| 1–14 | 这是四川吗 | ≥ 0.069 → 1 | S killed by F; GO current ≤ −5 | integrates | 0 | idle (R2 needs GO) | silent (**B1**) | no target rises, so no learning on these paths |
| 15 (L) | 吗 4 slices + 2 silent | ≈ 0.098 → 1 | | fade units enter | 0 | | silent | |
| 16 | 2 slices | ≈ 0.064 → 1 | | | 0 | | silent | |
| 17 | silent | ≈ 0.034 → 1 | | Jaccard with t+1: 1.00 from here | 0 | | silent | |
| 18 | silent | ≤ 0.001 → **0** | | held (**B2**) | 0 | | silent | |
| 19 | silent | 0 | **S1 = 1** | held | 0 | | silent | |
| 20 | silent | 0 | GO current = w₁ = 2.0 → **GO = 1** (**B3**) | held | 0 | | silent | [push +7; S1 on at 19 and GO rises: w₁ += 0.05/(1+w₁/0.2)] |
| 21 | silent | 0 | S3 on | held | **P = T(20)** (500 units) | R2 | silent | [S3 on while GO was on at 20: w₃ −0.05, floor 0] |
| 22 = r0 | silent | 0 | — | held | 0 | **R0: boundary** | sc_对 = Σ_{p∈P(21)} L_p,对 (e.g. 0.08) ≥ 0.02 → **Y(对)**; D(need[对]) on, I off, M level 0 → "对" (**B4**) | [cue 对 → sc = 20 wins; P on at 21 and Y(对) rises: 500 links +0.0002] |
| 23 | own voice "对" heard (1-tick delay) | vetoed (V1) → 0 | — | own voice enters | 0 | R1 | Y(对) held, force 2 → "对" | |
| 24 | | 0 | | | 0 | R2 | force 3 → "对" (**B5**) | |
| 25 | | 0 | | | 0 | R0 | sc from Y(对)(24): L(对→啊) > L(对→END) → **Y(啊)** → "啊" ×3 | [cue 啊; Y(对) on at 24 and Y(啊) rises: L(对→啊) += 0.01/(…)] |
| 28 | | 0 | | | | R0 | L(啊→END) wins → nothing chosen; D off, I on, M off → silent (**B6**) | [END push; Y(啊) on at 27 and END rises: L(啊→END) += 0.01/(…)] |
| 29 | | | | | | R1 = R0(28) ∧ Y(28) = 0 → ring stops | silent | |
| 29–32 | own voice still in the window until t=30 | 0.02–0.1 but vetoed (V1–V4 veto F on t ∈ [24, 32]) | F never turns on, so no S1 and no GO (**no self-reply**) | | | | silent | |

Each behaviour follows directly from these rules:
- **B1 [P]:** a mouth move needs M, which needs D. D needs a Y unit or a cue, and Y is chosen only at R0. R0 needs R2,
  which needs GO with F off, or the ring already running. GO needs an S pulse, and S pulses exist only after F turns
  off. Over 300 lines, F stayed on during every tick of the other's line from its 2nd tick on (min 0.069 > 0.008).
  The other's line can also start while a learned GO fires (dialogue boundaries, PAUSE 4): then F is on at L+5, which
  blocks P and R2 **[P; S: 0 moves in training and test]**.
- **B2 [P, S]:** in silence the whole reserve-weighted FEAT0 input is ≤ 0.001/0.001 = 1, i.e. at most about one
  depleted neuron (≈ 0.3), so a unit that is not held scores ≤ 0.3 + 0.01. A held unit scores 3 − penalty. The penalty
  is ≈ 0 for typical units (mean usage is exactly the target 5%) and still leaves a score ≥ 0.9 for a hub used 8× more
  than the target. So T is frozen from L+2 until the own voice arrives at r0+1. Measured: decode 20/20 at the readout tick, same line after
  a different context 0.88, across lines 0.044, no unit on in more than half of the end states.
- **B3 [P, S]:** the start falls at L+7, deterministically. Training needs S1→GO ≥ 1, i.e. 70 turn pairs (Lemma 3).
- **B4:** the first syllable comes from P = the thought at L+5; later syllables from Y→Y.
- **B5 [P]:** exactly one Y per beat, so exactly its 4 muscles move. Need sets are unique, so the world decodes exactly
  that char, at force 1, 2 and 3 over 3 ticks. Measured: 0 garble.
- **B6 [P]:** the reply ends when END wins, or when no score reaches 0.02, or after at most 15 syllables (breath). No
  self-reply: the own voice in the window [r0+1, m_e+3] is covered by V1–V4 over [r0+2, m_e+5], so F has no offset to
  detect.

---

## 4. Every learned link: how it forms in training

| Link | Target rises at | On at t−1 | Off during the target's hold? | Back-term exposure | Δ per event | Events | Strength reached |
|---|---|---|---|---|---|---|---|
| S1→GO | L+5 (teacher push; later the model's own GO, same tick) | S1, a one-tick pulse at L+4 | yes (S2 at L+5, S3 at L+6) | only S3→GO, held at 0 | 0.05/(1+L/0.2) | 1 per turn | **L ≥ 1 after 70 turns ≈ 48 dialogues; budget 2.0 after 240 [P].** Simulated: 0.99 at 50 dialogues (silent), 1.54 at 100 (speaks), 2.0 by 200 |
| P→Y(first) | r0 = L+7 | P = T(L+5), only at L+6 | yes: P is silent until the next GO | none: 0 hits in 2,000 dialogues **[S]** | 0.0002 × 500 links = 0.1 per event | 1 per turn | speaks once some score ≥ 0.02, from ~50–100 dialogues **[S]**. Content above the shuffled baseline from ~2k dialogues (Table 8) |
| Y(prev)→Y(next) | each boundary r0+3k | Y(prev), held for 3 ticks | yes: released at the boundary | only for patterns like A B A (288 hits in 600 dialogues: the rule weakening back-and-forth links) | 0.01/(…) | ≈5 per turn | a bigram passes 0.02 after 2 occurrences within the decay window (~1,400 dialogues); frequent bigrams are usable after a few hundred dialogues |
| Y(last)→END | r0+3m | Y(last syllable) | yes | none | 0.01/(…) | 1 per turn | END wins for chars that end replies often: stops 100% of free replies in the corpus toy; in the circuit, END or the 0.02 floor stop every reply **[S]** |

On at t−1 but deliberately **not** linked to these targets: T (held, would net 0), R2 (fires before every boundary,
content-free, would learn only the overall prior), D/M (per muscle, Lemma 6), V, and the ear cortex.

Cue margin **[P]**:
- In training, the cued Y scores 4·5 = 20.
- Any other Y scores at most 3·5 + 2.0, because the most it can learn from the single active source family is the
  per-target budget. That is 17 < 20.
- END scores at most 2.73 in training except at the pushed boundary, where it scores 10 + 2.73.
- So teacher-cued speech is exact however much is learned (fixes evidence #9). Simulated: 19/20; the miss is "哈哈哈",
  which the world plays as one long 哈.

---

## 5. B7: stability, interference over ~30k dialogues

- **Activity cannot run away [P]:**
  - fixed: the ear layers, the hippocampus and T (K_T);
  - bounded: P ≤ K_T, and only 1 tick per turn; Y ≤ 1; D ≤ 4; M ≤ 12;
  - single units F, F2, GO, END ≤ 1 each; one travelling pulse in S and in the ring; V ≤ 4.
- **Weights cannot run away [P]:**
  - per-target budgets 2.0 on S→GO, P→Y and Y→Y;
  - END ≤ 2.73 (equilibrium of Lemma 4 with r = 2.14 events per decay period);
  - the concave growth makes every individual weight sublinear.
- **No wholesale forgetting [P]:**
  - the structural behaviours (B1, B3 timing, B5, B6 backstop, exact guided speech) depend on no content weight;
  - S1→GO is refreshed every turn and sits at its budget;
  - content links fade gradually with time constant ≈ 1,400 dialogues (Lemma 4) and are displaced only
    proportionally (budgets);
  - T's codes cannot drift with learning, because T reads layer 0, which no learned or return current reaches. Layer 0
    is also never split or deleted (its split/delete fraction is 0 in `Net3DCfg`), and FEAT0 is frozen at warm-up.
    Only the slow set point (τ 1,000 ticks) moves T's codes.
- **Self-reinforcement in free use (learning on, no teacher):**
  - a back-and-forth cycle A→B→A has every A→B link cancelled by the back term (+η when B starts, −η when A restarts
    while B is still on) **[P]**;
  - longer cycles are limited by the breath reserve (≤ 15 syllables) **[P]**;
  - habit formation over many free replies is **[A]**, untested.
- **Compute [A, by size]:**
  - removes 24k + 2k units and about 2.3M internal PFC edges;
  - adds T/P (10k each, 30 inputs per T unit);
  - adds P→Y and Y→Y tables (≈ 220 MB), which are touched only once per turn or boundary;
  - expected: at least as fast as now (≥ 500 dialogues per 30 min).

---

## 6. STATE §4 evidence, item by item

| # | Failure or evidence | How this design handles it |
|---|---|---|
| 1 | Ear identifies the line at its end plus ~4 ticks | T samples the ear's input layer during exactly that window and then holds it; nothing depends on the afterimage after L+2 **[S]** |
| 2 | Hippocampus 0.95–1.00 | Kept unchanged for later (detail, replay, B8); B2 does not need it |
| 3 | PFC receiving layer 0.15: shared core | Removed from the path; T has no shared core (0 units on in > half of the end states; across 0.044) **[S]** |
| 4 | Thinking layer ≈ chance untrained | T decodes 20/20 untrained, with no learning involved: bottom-up, reserve-weighted input plus a one-tick hold **[S]** |
| 5 | Period-2 alternation in 2-tick loops | One-tick self-hold: consecutive-tick Jaccard 1.00 from L+2 **[S]** |
| 6 | Fresh unit ~2 ticks late; teacher's GO braked; late latch, drift | F threshold set from the measured distribution: off at L+3 for every line; the teacher's GO comes at L+5, when F is already off; T holds from the start, so there is no engagement window **[S]** |
| 7 | Held thought cannot teach (think→go ≈ 0) | Lemma 1. "When" is taught by S1, a one-tick pulse; "what" by P, a one-tick copy; back-term hits on P→Y = 0 **[P, S]** |
| 8 | First-syllable currents (≤ 0.6) below the needed 0.8 | No fixed 2.0 threshold to beat: a relative winner-take-all over Y with a learned-scale floor θ_rel = 0.02 (= 1/5 of one event's total). The mctx(k)→D(k+1) path is removed **[P]** |
| 9 | Guided speech degraded to 61–83% | No learned or return input to D/I/M; cue margin 20 vs ≤ 17 → exact **[P, S]** |
| 10 | Never opened its mouth | Replies to 20/20 probes from 100 dialogues on, at L+7 **[S]** |
| 11 | Budgets stable; featureless neurons muted | Kept; muting also defines FEAT0 (767 of 1920 layer-0 neurons muted) |
| — | hold ≤ across (BRIEF) | Same line in a different context 0.88 vs across 0.044 **[S]** |

---

## 7. Changes from the current code

1. `talk.py`:
   - REPLY_GAP 7;
   - teacher GO push at L+5;
   - cue +5 per D and it forces the boundary;
   - END push after the last syllable;
   - `check()` latency counted from L.
2. `brain.py`:
   - remove the PFC receiving layer, its relays, the thalamus, think↔pfc, and every learned or return path into the
     motor area;
   - add F (FEAT0 input, threshold 0.008, V veto), F2, S1–8, V1–4, T (layer-0 expansion with self-hold 3.0 and set
     point), P, Y, END, and the learned S→GO, P→Y, Y→Y, Y→END.
3. `areas.py` MotorArea:
   - make the ring explicit (R0/R1/R2 plus breath);
   - choose syllable programs at R0;
   - slave D/I/M to the winner (keep the same-tick I/M comparison and the force ramp);
   - remove `d_floor` and free D firing.
4. `net.py`: one `PathCfg` option, a list of targets exempt from the budget (END). The learning rule itself is
   **unchanged**.

---

## 8. Toy simulations

| File | What it checks | Result |
|---|---|---|
| `corpus_stats.py` | baselines | 47,786 turn pairs; first-char entropy 7.7 bits; "always 你" puts 你 in the reply 20.9% of the time (= shuffled); "say the last heard char" 13.8% vs 4.4% shuffled; argmax successor is END for 50.5% of reply tokens |
| `toy_content.py`, `toy_content2.py` | muscle-level readout (Lemma 5, 6) | at chance or ≤ +1.7 pts; outputs 根/箍/佳 |
| `toy_reply.py` + sweeps | syllable-level P→Y, Y→Y, END; exact rule, decay, budgets | see the table below |
| `toy_fresh.py`, `toy_layer0.py` | F current distribution; T on layer 0 | F as in 2.2. T at h=3: decode 1.00, same 0.88, across 0.044; recency share last→8th char 0.25 (+0.30 fade) / 0.13 / 0.10 / 0.07 / 0.06 / 0.04 / 0.04 / 0.02. At h=4.5: broader bag (0.34 / 0.18 / 0.15 / 0.10 …), same 0.68 |
| `toy_thought.py` | T on the cortex (for comparison) | decode 1.00, same 0.49, across 0.034 |
| `toy_circuit.py` | **whole circuit**: real ear replica, protocol, exact rule | see 8.2 |

### 8.1 Content (B4) on the real corpus
Settings: syllable level, η·K_T = 0.1, B_P = 2, η_Y = 0.01, B_Y = 2, END without budget, θ_rel = 0.02, γ = 0.999;
2,000 held-out lines. "First in reply" means the first spoken char appears anywhere in the expected reply; "shuffled"
compares with the reply to a different line.

| Thought model | Turn pairs trained | First in reply / shuffled | All chars in reply / shuffled | Mean length | Stops |
|---|---|---|---|---|---|
| idealised bag (ρ = 0.3) | 3,000 | 0.118 / 0.092 | 0.120 / 0.097 | 1.7 | END 100% |
| idealised bag | 45,786 | 0.136 / 0.096 | 0.120 / 0.088 | 2.0 | END 100% |
| measured profile (h = 3) | 3,000 | 0.113 / 0.091 | 0.121 / 0.096 | 1.8 | END 100% |
| measured profile (h = 3) | 10,000 | 0.079 / 0.065 | 0.095 / 0.081 | 1.8 | END 100% |
| measured profile (h = 3) | 45,786 | 0.123 / 0.090 | 0.111 / 0.090 | 2.1 | END 100% |
| measured profile (h = 4.5) | 3,000 | 0.121 / 0.105 | 0.125 / 0.110 | 1.7 | END 100% |
| measured profile (h = 4.5) | 10,000 | 0.090 / 0.067 | 0.105 / 0.083 | 1.7 | END 100% |
| measured profile (h = 4.5) | 45,786 | 0.148 / 0.111 | 0.126 / 0.099 | 2.05 | END 100% |
| codes with 30% shared core + 20% dropout | 45,786 | 0.144 / 0.104 | 0.101 / 0.086 | 3.3 | END 75% (END budgeted) |

Notes:
- Standard error of a difference ≈ 0.009 (first char) and ≈ 0.0065 (all chars); see section 9 for the significance
  of these gaps.
- Examples (idealised bag, END budgeted): 施华洛拍的吗→不是啊; 好面熟→好看到了; 没有风声→没有点儿.

### 8.2 Whole-circuit toy (`toy_circuit.py`; outputs `circuit600.out`, `circuit2000.out`)
Setup: ear replica with 27.6k units, T 4,000/200, exact rule, protocol of section 2.4; 20 held-out probe lines; test
with learning off.

| Dialogues trained | 0 | 50 | 100 | 200 | 500 | 1000 | 1500 | 2000 |
|---|---|---|---|---|---|---|---|---|
| Mouth moves while the other speaks (test / training) | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 / 0 |
| Spoke after the line | 0/20 | 0/20 | 20/20 | 20/20 | 20/20 | 20/20 | 20/20 | 20/20 |
| Latency (ticks after the line) | – | – | 7 (all) | 7 | 7 | 7 | 7 | 7 |
| Garble ticks | 0 | 0 | 0/111 | 0/180 | 0/135 | 0/84 | 0/108 | 0/78 |
| Second GO after its own reply | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Stops: END / floor / backstop | – | – | 20/0/0 | 10/10/0 | 16/4/0 | 20/0/0 | 20/0/0 | 20/0/0 |
| Guided (teacher-cued) exact | 19/20 | 19/20 | 19/20 | 19/20 | 19/20 | 19/20 | 19/20 | 19/20 |
| S1→GO | 0 | 0.99 | 1.54 | 2.0 | 2.0 | 2.0 | 2.0 | 2.0 |
| Back-term hits P→Y / Y→Y | 0/0 | 0/2 | 0/15 | 0/46 | 0/204 | 0/497 | 0/808 | 0/1175 |
| Mean reply length (syllables) | – | – | 1.9 | 3.0 | 2.3 | 1.4 | 1.8 | 1.3 |

- No self-started speech outside the turn during training (0 at every check).
- The guided miss is "老牛吃嫩草哈哈哈": the world plays 哈哈哈 as one long 哈.
- Example replies:
  - 200: 你在哪里 / 我要;
  - 600: 不是的 / 是的 / 你也可以;
  - 1500: 是啊 / 对啊 / 好 / 谢 (to 哈哈哈妇女节快乐);
  - 2000: 哈 / 有 / 你 / 谢.
- Content counts on 20 probes are too small to judge; use 8.1.
- **Trend:** replies shorten with training (1.3 syllables at 2000 dialogues), because END has no budget while each
  syllable target is capped. See §9 item 7.

---

## 9. Weakest points (honest)

1. **Content is weak [S].**
   - The first syllable is above the shuffled baseline by 1.4–3.3 points (measured profile), roughly 2–4σ on 2,000
     lines; exact first char ≈ 4–5%.
   - Replies are short (≈ 2 syllables) and repetitive (≈ 20 distinct replies to 2,000 lines).
   - Cause: the corpus is barely predictable (first-char entropy 7.7 bits), and a single-layer Hebbian readout of a bag
     of sounds cannot beat it.
   - Echo ("say a heard char") carries a large signal (+9.4 pts as a fixed strategy) but needs a sound→program map.
     Adding a heard-syllable bias β·(recency share) to the first-boundary scores, with β = 0.3, would roughly triple the
     content signal: first syllable 0.138 vs 0.042 shuffled, all characters 0.120 vs 0.057, against +3.6 and +2.1
     points without it (`sweep_echo.py`) **[S, given the map]**. Whether the rule can learn the map is only sketched in
     §11 **[A]**.
2. **Y is localist** (one unit per executed muscle combination). It departs from "not one neuron per char" on the motor
   side. Lemma 6 is the reason; a sparse distributed program code is possible later **[A]**.
3. **Toy scale.** T has 4,000 units (not 10,000), and the toys have no ear-internal learning, no returns and no
   hippocampus. F and T are immune to those by construction (layer 0), but run-time interactions on the GPU are
   unmeasured.
4. **The latency is 7 ticks**, set by the 3-tick ear window. That is ≈ 525 ms at 75 ms/tick, about twice the human gap.
   Turn ends are detected from offsets only: no prediction, no barge-in. The own-voice veto is a fixed 4-tick hangover;
   speech from the other during the model's own speech is not heard as "someone talking".
5. **Repeated syllables** (哈哈) are never learned (no rise), a limitation of the world: no re-onset.
6. **Untested:** the trade-off set by h (bag breadth vs reproducibility). Self-reinforcement over long free use is
   **[A]**.
7. **Replies are short, and END sets the length [S]** (`sweep_endlen.py`, measured T profile, full corpus):

   | END regime | Mean length (syllables) | All chars in reply / shuffled | Stops / robustness |
   |---|---|---|---|
   | no budget (default) | 2.07 | 0.111 / 0.090 | END 100%; robust at γ = 0.9999 |
   | END budget 5 | 3.16 | 0.089 / 0.076 | END 90%, floor 10% |
   | END budget 10 | 2.46 | 0.104 / 0.088 | END 98.5% |
   | stop if sc_END > 0.5·Σ continuation | 3.36 | 0.088 / 0.075 | |
   | stop if sc_END > 1.0·Σ continuation | 10.2 | | backstop in 1061/2000 replies |

   - Longer replies dilute precision, because later syllables come from the bigram chain only.
   - In the circuit, length drifted to ≈ 1.3 syllables at 2,000 dialogues (20 probes, noisy).
   - A budgeted END is γ-dependent (Lemma 5): with budget 2.0, END stopped only 1/2000 replies at γ = 0.9999. So
     "no budget" is the robust default, and an END budget of 5 is the option if longer replies matter more than
     precision.

---

## 10. Parameter summary

| Group | Value | Status |
|---|---|---|
| F | weight 0.001, threshold 0.008 (from measurement), veto −10 | measured |
| S | 8 cells | |
| V | 4 cells | |
| GO | threshold 1; F, speaking −5 | |
| T | K 5%, fan-in 30, h 3.0, set point β 0.3, τ_u 1000 | measured |
| P | threshold 2 | |
| Ring | breath U 0.06, τ 100 → ≤ 15 syllables | |
| Cue / END push | 5 per muscle / 10 | |
| θ_rel | 0.02 | |
| S→GO | η 0.05, B 2 | |
| P→Y | η·K_T = 0.1, B 2 | swept 1/2/4/20 |
| Y→Y | η 0.01, B 2 | |
| Y→END | η 0.01, no budget (option: budget 5 → ≈ 3 syllables) | swept |
| All decays | γ 0.999 per 100 ticks | 0.9999 tested |
| Unchanged | everything else in the ear / mctx / hippocampus | |

---

## 11. Next module (not part of the core; status marked)

**Echo / mirror route: heard syllable → same program.** The corpus toy above shows that a structural bias toward
programs of recently heard syllables roughly triples the content signal **[S, given the map]**. Sketch of how the map
could form with the same rule **[A]**:
- Mirror unit Z_c is created together with Y_c and fires for one tick when Y_c is released (at b+3).
- On at b+2 is the layer-0 code of the model's own voice of c (heard with a 1-tick delay; window slots 2–5 hold c's
  slices 0–3). At b+4 those slots hold other slices, so the link FEAT0 → Z_c passes Lemma 1 (most sources are off at
  b+4).
- The other's voice differs from the own voice in about 30% of cells by ±1 level. In the thermometer code of layer 0
  that flips 1–2 neurons per cell, so hearing c from the other should also drive Z_c.
- Still needed: a held echo trace (latched Z, reset by the next line) feeding Y_c at the first boundary with β ≈ 0.3.
  This adds one latch and one fixed weight per program, and its learnability and false echoes during listening are
  untested.

---

## 12. Biology anchors (from the panel's literature reviews; not re-verified here)
- Offset responses in auditory cortex: Scholl, Gao & Wehr 2010 (cited in 01) — S1.
- Time cells: MacDonald et al. 2011 (03) — S2…S8.
- Output gating of working memory: PBWM; Chatham, Frank & Badre 2014 (01, 05) — P = T ∧ GO.
- Content-free GO via thalamus: Inagaki 2022; Kaufman 2016 (04) — GO.
- Syllable-boundary thalamic input: Moll 2023 (04) — ring R0.
- Corollary discharge suppressing one's own voice: Schneider 2014/2018; Eliades & Wang 2008 (01, 05) — V.
- Speech sound map / motor programs: DIVA, GODIVA (04) — Y.
