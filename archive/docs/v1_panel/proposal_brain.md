# Listen → Latch → Launch: a brain-derived turn-taking speech circuit for concept-brain

*(Partner-speech gate with corollary discharge; an MD plan latch that GO consumes; syllable channels released on a
3-tick beat; a learned final-syllable marker)*

Panel angle: **brain first**. Every element below is taken from what the reviews (01–05, 00_synthesis) say real brains do,
then mapped onto the designer's framework as simply as possible. **The one learning rule is not changed.**

Marks:
- **[P]** = proved by an explicit argument or exact computation;
- **[S]** = simulated in my numpy toys (result given);
- **[A]** = assumed.

The toys keep the real update order (all currents from t−1 states → all fire → learn → reserve update), the exact
`Path` arithmetic, and the real world code (`world/voice.py`), and they train on the real corpus. Directory:
`/tmp/claude-0/-root/e92d3d7d-5823-4f5c-a0ec-4fe302bff9af/scratchpad/arch/brain/` (§10).

---

## 0. Summary

**Five lines**
1. **When** to speak is decided only by content-free structure:
   - N = "partner speech present"; a corollary discharge (CD) keeps the own voice out of N;
   - W = "my answer is given";
   - GO fires once the partner is silent and the MD plan latch is filled.
   No learned weight can open the mouth or keep it open.
2. **What** to say is learned only from sources that are **on at t−1 and off from t+1 through the target's hold**. This is
   the rule's timing law (§3) and the real reason for STATE failure 7. GO silences the plan latch, as the go cue does
   to Inagaki 2022's preparatory state and the first movement does to pre-SMA "sequence cells". The own-voice auditory
   code is a 1-tick transient.
3. **Output:** the 32 per-muscle D become **syllable channels**: disinhibitory neurons, one per learned muscle synergy
   (DIVA's speech-sound map, basal-ganglia action channels). A channel is recruited on the first guided production.
   1-winner selection at each 3-tick beat boundary makes every syllable one exact program lasting 3 ticks, never "~".
4. **Plan content:** a fixed **hippocampus→thinking-layer projection** (vCA1→mPFC) makes the plan line-specific without
   PFC training: decoding of the heard line 0.80 vs 0.05 (chance 0.05). Familiar dialogues then get the right first
   syllable 14–16/20 (2/20 without the projection).
5. **Stopping:** a learned **final-syllable marker** rises *with* the last syllable; that is the only placement the rule
   learns cleanly. A 12-syllable breath limit is the fallback. W gates the plan loop, so the model never restarts after
   its own answer.

**Key changes from the current code** (details in §8)
- **Removed:**
  - the per-muscle D and their instinct path (`dis_thr`);
  - all learned →GO paths, think→motor, per-muscle ear→motor;
  - the pfc/hip⇒motor returns, mctx→motor and the motor-internal learned path;
  - `stop()`.
- **Added:**
  - W (1 unit), CD (4 units), syllable channels (≤4096 recruited slots, fixed programs), the final marker (1) and a breath
    limit;
  - learned paths thal→channels and ear→channels; a fixed hip⇒think projection.
- **Rewired:**
  - N counts active non-muted ear neurons (no reserve weighting);
  - GO, speaking and W inhibit the thalamic latch;
  - GO = plan latched, with vetoes from N and from speaking.
- **Protocol:** the teacher never pushes GO; it guides the next syllable until the model's own boundary takes it, and
  cues REST with the last syllable. Learned budgets must **bind** (§6.4).

**Evidence at a glance**

| claim | status |
|---|---|
| B1, B3, B5, B6, B7 hold for *any* learned weights | [P] + [S] (0 moves in 106 prompts; 200/200 lines launched at free tick 5–6; 0 "~" in 80 replies, every syllable exactly 3 ticks; 40/40 learned stops; 0 restarts; guided 1.00) |
| B2: the plan at GO−1 identifies the heard line | [S] 0.80 (hip⇒think w ≤ 0.15) vs 0.05 without it |
| B4, familiar dialogues | [S] first syllable 14–16/20; overlap 0.36–0.47 vs 0.10–0.12 mismatched (without hip⇒think: 2/20; 0.21 vs 0.19) |
| B4, never-heard lines | **at noise level for any statistical learner on this corpus** [P, computed]: Δ ≤ +0.02 up to 40k dialogues |

---

## 1. Brain derivation: seven stages → seven elements

**A. Listen.**
- Real brains: A1 is sparse and adapting. Speaking suppresses the auditory response to one's own voice through a motor
  corollary discharge (SIS: Houde & Nagarajan 2011; Eliades & Wang 2008), and M2 learns to inhibit auditory cortex for
  self-made sounds (Schneider 2018).
- Need: a "partner talking" state that the own voice cannot trigger.
- Element: **N**, speech presence over the non-muted ear cortex, plus **CD**, a 4-unit delay line from any open
  channel that inhibits N.

**B. Hold.**
- Real brains: MD is necessary in the delay, not in encoding, and amplifies PFC's own structure (Bolkan 2017; Schmitt
  2017). Content beyond working-memory span needs the hippocampus even over short delays (Jeneson & Squire 2012).
  vCA1→mPFC is needed for encoding (Spellman 2015), and sharp-wave-ripple replay refreshes PFC content (Jadhav 2016).
- Need: a line-specific state, built at the end of the line, that is off while input flows.
- Element: hippocampus (existing) → **fixed hip⇒think** → thinking layer → **MD latch** (existing). The latch is
  released by N, GO, speaking and W.

**C. Take the turn.**
- Real brains: the gap between turns has a mode of 0–200 ms (Stivers 2009). Planning starts early and the launch waits
  for the end-of-turn cue (Levinson & Torreira 2015; Bögels 2015). A content-free GO from PPN/MRN via VM thalamus
  reorganises preparation into execution; stimulating it alone launches the prepared act (Inagaki 2022). The trigger
  signal is condition-invariant (Kaufman 2016).
- Need: a launch that does not depend on how strong the content is.
- Element: **GO** = plan latched ∧ partner silent ∧ not speaking; **W** = turn latch (set by GO, reset by N).

**D. First word.**
- Real brains: the basal ganglia keep tonic inhibition and release one channel while suppressing the rest (Mink 1996;
  Hikosaka 2000). In GODIVA the plan feeds a winner-take-all choice layer. Pre-SMA sequence cells **stop at the first
  movement** (Shima & Tanji 2000). DIVA has one speech-sound-map node per learned syllable program (04 §2.2).
- Need: choose one syllable; the plan must go quiet at onset.
- Element: **syllable channels** with 1-WTA at the boundary; learned **thal→channel**; GO ⊣ thal.

**E. Sequence.**
- Real brains: in songbirds an HVC chain keeps time inside syllables and thalamic Uva kicks at syllable onsets;
  perturbing Uva cuts song only at boundaries (Moll 2023; Danish 2017; Long & Fee 2008). Auditory–motor coupling
  prefers about 4.5 syllables/s (Assaneo & Poeppel 2018).
- Need: fixed syllable timing; the next item comes from a transient code of the current one.
- Element: **beat** (the existing 3-tick phase); learned **ear→channel** from the own-voice 1-tick transient.

**F. Articulation.**
- Real brains: motor cortex executes one program's trajectory at a time (Bouchard 2013; Chartier 2018).
- Need: one program per beat and no side route to the muscles.
- Element: M is driven only by the open channel (structural force ramp).

**G. Stop.**
- Real brains: striatal start/stop cells mark the boundaries of a sequence (Jin & Costa 2010; Jin 2014).
- Need: a learned "this one is the last" plus a hard bound.
- Element: **final marker** P_rest, which rises with the last syllable; **breath limit** of 12 syllables [A: the
  breath group is not from the reviews].

**How the designer's framework is kept**
- tonic I against disinhibition decides movement (#4);
- PFC and the other areas decide *what* (#12, #21);
- there is no speech area — GO, W, CD, beat and channels are generic motor and turn structure (#22);
- one learning rule (#9), plus global decay (#3);
- the hippocampus is the fast chain region, and the PFC thought moves with it (#2, #29; now structurally possible);
- no refractory period (#15) — CD inhibits a *detector*, not the speaking neurons;
- "muscles, then inhibition" ends the sentence (#35) — the final marker closes all channels;
- neuron generation (#5, #6) appears as channel recruitment.

---

## 2. Complete specification

### 2.1 Areas (real-model sizes)

Legend: ✚ new, ✎ changed, = unchanged.

| area | neurons | activity | change |
|---|---|---|---|
| = ear | 27.6k slots / 20.2k cortex | 3D net, top-k 40/30/20/10 %; reserve U 0.5, τ 6 | — |
| = motor cortex (mctx) | ≈3.8k | as now | chains/mirrors kept; no motor output |
| ✎ hippocampus | ≈24k mirrors | top 5 %; chain 3.0; internal η 0.5, γ 0.7, B 2 | ✚ output hip⇒think |
| = PFC receiving | ≈24k | top 8 % + set point | — |
| ✎ PFC thinking | 10k | top 5 % + set point | ✚ input from hip |
| ✎ MD thalamus (plan latch) | 2000 | threshold 2, self +2 | ✚ ⊣ GO, SPK, W |
| = relays / returns | as now | | |
| ✎ N | 1 | threshold 0.3 | count, no reserve |
| ✚ CD | 4 | chain | |
| ✚ W | 1 | latch | |
| ✎ GO | 1 | threshold 0.5 | inputs rewired; N and SPK are vetoes |
| = beat | 3 phases | ring (existing counter) | |
| ✚ breath | 12 | chain (counter) | |
| ✎ M (force) | 96 | ramp | no learned or return input |
| = I | 32 | tonic | |
| ✎ syllable channels (the "D") | ≤ 4096 slots | 1-WTA at boundaries, held 3 ticks | replace the 32 per-muscle D |
| ✚ final marker P_rest | 1 | rises with the last syllable | |

### 2.2 Dynamics (every current from t−1 states)

```
N(t)      = [0.001·#{non-muted ear cortex on at t−1} − 10·Σk CDk(t−1) ≥ 0.3]
CD1(t)    = [Σc Pc(t−1) ≥ 1];   CDk(t) = CDk−1(t−1),  k = 2..4
W(t)      = [3·GO(t−1) + 2·W(t−1) − 5·N(t−1) ≥ 2]                                  W(0) = 1
thal_j(t) = [Σ_{i∈grp j} think_i(t−1) + 2·thal_j(t−1) − 10·(N+GO+SPK+W)(t−1) ≥ 2]       5 members per j
GO(t)     = [0.1·Σj thal_j(t−1) − 1000·N(t−1) − 1000·SPK(t−1) ≥ 0.5]                 (vetoes)
boundary(t) ⇔ (GO(t) ∧ ¬SPK(t−1)) ∨ (SPK(t−1) ∧ phase(t−1) = 2)
I_c(t)    = Σj thal_j(t−1)·Wth[j,c] + Σi x_i·ear_i(t−1)·We[i,c] + cue_c(t) + ξ,     ξ ~ U(0, 1e−4)
at a boundary:
    if P_rest(t−1) or syllables = 12:  STOP  (all channels off, SPK := 0)
    else: c* = argmax_{recruited c} I_c(t);  Pc*(t) = 1;  P_rest(t) = [I_rest(t) > ρ·I_c*(t)];
          SPK := 1, phase := 0, syllables += 1
between boundaries: all P (incl. P_rest) hold; phase += 1
I_m(t)    = [1.2 − 2.0·Σ_{c∋m} Pc(t) ≥ 1]               (tonic inhibition, same-tick comparison as now)
M_{m,l}(t)= [m ∈ prog(open channel) ∧ l ≤ phase]         (force ramp 1→2→3 levels)
```

### 2.3 Fixed connections

| from → to | weight | role |
|---|---|---|
| ear cortex (non-muted) ⇒ N | +0.001, no reserve | speech presence; [S] speech 0.45–0.8, silence 0.015–0.02 |
| channels ⇒ CD1; CDk ⇒ CDk+1 | +1 | corollary-discharge delay line |
| CD1..4 ⇒ N | −10 | the own voice is never "the partner" |
| GO ⇒ W +3; W ⇒ W +2; N ⇒ W −5 | | turn latch |
| think ⇒ thal (+1 per member); thal ⇒ thal +2; thal ⇒ think +0.3 | as now | MD loop |
| N, GO, SPK, W ⇒ thal | −10 each | the loop holds only in the delay before *my* answer |
| thal ⇒ GO | +0.1 each | launch |
| N ⇒ GO, SPK ⇒ GO | −1000 (veto) | see §7, item (f) |
| hip ⇒ think | 32 random targets per hip unit, w ~ U(0, 0.15) | vCA1→mPFC |
| channel c ⇒ I_m (m ∈ prog c) | −2.0 | disinhibit the program's muscles |
| teacher muscle cue ⇒ channel c | +4·\|cue ∩ prog c\| | exact program 16 > three-of-four 12 + max learned 2 |
| teacher REST cue ⇒ P_rest | +64 | beats ρ·(16 + 2) |

### 2.4 Learned connections (the one rule)

The rule, unchanged:
- **forward:** src on at t−1 and dst rising at t → w += η/(1 + L/0.2)·x;
- **back:** src on at t and dst on at t−1 → w −= η, floor w0;
- every 100 ticks: decay ×γ, then the per-target budget.

| path | size | η | budget / target (toy E5) |
|---|---|---|---|
| thal → channels + P_rest | 2000 × 4097, full, w0 = 0 | 1e−3 | 1.0 (binds above 1.1 % first-syllable frequency; 0.5 binds above 0.54 %) |
| ear → channels | 20.2k × 96 random slots, w0 = 0 | 1e−4·x | 0.1 (binds above ≈0.9 % successor frequency) |
| ear → P_rest | 20.2k × 1, **full fan-in** like the other global units | 1e−4·x | 0.1 |

Unchanged: ear internal, hip internal, PFC internal, pfc→think, think internal, think→pfc, ear↔mctx.

### 2.5 Channel recruitment (neuron generation, structural)

A guided muscle set with no exact channel takes a free slot with that program (fixed). [P, corpus]:

| dialogues | channels | share of later replies' syllables already covered |
|---|---|---|
| 100 | 324 | 0.67 |
| 1000 | 1143 | 0.91 |
| 10k | 2548 | 0.99 |
| all | 3548 | — |

The toy recruited 324 / 464 / 607 channels at 100 / 200 / 300 dialogues, which matches the corpus count. A channel may
be a k-unit assembly; nothing in the argument changes.

### 2.6 Protocol

- **Training,** per (Q, R) pair (drop the 5,298 unpaired trailing lines):
  1. `listen(Q, pause=0)`.
  2. The teacher holds `cue(r1)` until the model's own boundary takes it. A fallback GO after 8 ticks exists; it was
     used 0 times in 300 E2b dialogues.
  3. The teacher then cues r2 … in turn; the last syllable is cued together with REST.
  4. The model's next boundary ends the turn; then PAUSE 4.
  5. Learning stays on throughout.
- **Test** (as now): `listen(Q, 0)` + 45 free ticks, learning off.
- **Warm-up:** motor clamped, ending with 10 silent ticks.

---

## 3. The timing law of the rule

**[P]** For a target that rises at t0 and is held to t0+2:
- the forward term acts once, at t0, if src(t0−1) = 1;
- the back term removes η at every t ∈ {t0+1, t0+2, t0+3} with src(t) = 1.

So **a source gains +η only if it is on at t0−1 and off at t0+1, t0+2 and t0+3.** It may also be on at t0.

**[S]** T1 (`t1_rule.py`, exact replica, η = 0.01):

| source (target rises at t0, held 3 ticks) | Δw per event | after 100 events |
|---|---|---|
| held thought, on t0−4…t0+6 | 0.0000 | 0.000 |
| held, then silenced by GO (off from t0+1) | **+0.0100** | **0.465** |
| silenced one tick late (off from t0+2) | 0.0000 | — |
| plan latch (on t0−1, t0) | +0.0100 | — |
| ear transient (on t0−1) | +0.0100 | — |
| feature that reappears at t0+2 | 0.0000 | — |

Consequences:
1. A motor link must come either from a **threshold population that GO can silence** or from a **1-tick transient**.
   Top-k layers (thinking layer, hippocampus) never go silent, so their persisting winners always cancel.
2. **A stop target must not rise after the sound has ended.** At that moment every generic "sound present" feature
   switches off exactly as the target rises, which is a perfect offset→onset pair. [S]
   - Toy E2 (stop after the end): stop mass 1.02 vs 0.10 per muscle; it stopped after one syllable in 20/20.
   - T3: STOP 1.00 with length 2.0 at every N.
   - A marker that rises *with* the last syllable keeps those features on at t0+1, so they cancel.

---

## 4. Tick by tick: one familiar line (toy E5, hip⇒think model, after 360 dialogues)

Heard 想打架的样子 (expected reply 哪有想打架). rel 0 = first silent tick after the partner's last sound.

| rel | N | W | thal on | GO | CD on | SPK | phase | event | mouth |
|---|---|---|---|---|---|---|---|---|---|
| −4…−1 | 1 | 0 | 0 | 0 | 0 | 0 | – | | (partner speaking) |
| 0 | 1 | 0 | 0 | 0 | 0 | 0 | – | | (window still holds speech) |
| 1 | 1 | 0 | 0 | 0 | 0 | 0 | – | | |
| 2 | **0** | 0 | 0 | 0 | 0 | 0 | – | | (window silent) |
| 3 | 0 | 0 | **37** | 0 | 0 | 0 | – | | MD latch fills |
| 4 | 0 | 0 | 82 | **1** | 0 | 1 | 0 | **GO** | 哪 (plan → channel) |
| 5 | 0 | **1** | **0** | (1)* | 1 | 1 | 1 | | 哪 |
| 6 | 0 | 1 | 0 | 0 | 2 | 1 | 2 | | 哪 |
| 7 | 0 | 1 | 0 | 0 | 3 | 1 | 0 | NEXT | 个 (ear(6) → channel) |
| 8–9 | 0 | 1 | 0 | 0 | 4 | 1 | 1–2 | | 个 |
| 10–12 | 0 | 1 | 0 | 0 | 4 | 1 | 0–2 | NEXT | 猪 |
| 13–15 | 0 | 1 | 0 | 0 | 4 | 1 | 0–2 | NEXT | 有 (final marker rises with 有) |
| 16 | 0 | 1 | 0 | 0 | 4 | 0 | – | **STOP** | silent |
| 17–20 | 0 | 1 | 0 | 0 | 3, 2, 1, 0 | 0 | – | | own-voice echo hidden by CD |
| 21–29 | 0 | 1 | 0 | 0 | 0 | 0 | – | | silent; no restart (W = 1 gates the latch) |

\* In the toy, GO fired again at tick 5: 0.1·82 − 5·SPK = 3.2 ≥ 0.5. It had no effect, because a boundary needs
¬SPK(t−1). The spec therefore makes N and SPK vetoes (−1000).

### 4.1 Partner speaking (B1)

Numbers:
- About 450–800 non-muted ear neurons are on per tick, so N's current is 0.45–0.8 ≥ 0.3 and N = 1 from tick 1–2.
- W = [2 − 5 ≥ 2] = 0; the thal latch gets −10, so it is off; GO is vetoed.

There is no boundary, so all channels stay off. Each I = [1.2 ≥ 1] is on, every M = 0, and the mouth is still.

**[S]**
- The old reserve-weighted N fell to 0.375 inside 哈哈哈妇女节快乐. With a structural GO this gave barge-ins in 3/6
  probes.
- The count-based N gave 0/6, and 0 moves in all later prompts (106 in total).

### 4.2 End of the line → launch (B2, B3)

**Launch time** [P]:
- The ear window holds 6 slices and advances 2 per tick, so it is silent 3 ticks after the line ends; N falls at rel 2–3.
- P(≥2 of 5 members on) = 1 − 0.95⁵ − 5·0.05·0.95⁴ = 0.0226, so 2000·0.0226 ≈ 45 ± 7 thal units latch the next tick.
- GO follows one tick later.

**[S]**
- Latched units: 46.3 (38–58), 43.5 (35–56), 44.8 (33–58); up to 82 after 2 ticks with hip⇒think.
- GO on free tick 5 or 6: 26/40 and 14/40 lines (E4b); 14/20 and 6/20 (E5).
- So the gap is 4–5 silent ticks, i.e. 300–375 ms.

**B2.** At GO−1, which of 20 lines was heard? Each line was heard after two different preceding lines; Jaccard
nearest-template decoding, chance 0.05 [S]:

| wiring | thal latch | thinking | hippocampus |
|---|---|---|---|
| hip⇒think, w ≤ 0.15 | **0.80** (same line 0.142 vs different 0.047) | 0.80 | 0.85 |
| hip⇒think, w ≤ 0.06 | 0.30 | 0.70 | 0.85 |
| none (current wiring) | 0.05 | 0.00 | — |

### 4.3 First syllable, sequence, articulation, stop (B4–B6)

**GO tick t0.**
- c* = argmax_c Σ_{latched j} Wth[j,c].
- c*'s muscles get I = [1.2 − 2 ≥ 1] = 0, so they move, and M ramps 1 → 2 → 3 levels.
- At t0+1, W = 1 and the thal latch receives −30, so it goes off. In training this makes thal(t0−1) → c*(t0) a clean
  +η event for every latched unit (§3).

**Each next boundary b+3.**
- The own voice of syllable k reaches the ear at b+1…b+3.
- At b+2 the window holds [2 tail slices of k−1 | 4 head slices of k]. This code differs from those at b+1, b+3 and
  b+4, so ear(b+2) → c_{k+1}(b+3) is learnable, and features shared with later ticks cancel.

**Articulation.** One exact program per beat. **[S]** "~" = 0 in 4 × 20 test replies; every syllable lasted exactly
3 ticks (run lengths {3: 240}, {3: 240}, {3: 66}, {3: 72}).

**Stop.**
- P_rest rises with the last syllable; at the next boundary all channels close.
- **[P]** CD silences N from (first channel tick + 2) to (last + 5). The own voice can reach N only from (first + 2) to
  (last + 4), because the window keeps the last slices 3 ticks and N reads t−1. So W stays 1, the latch stays gated,
  and GO is impossible until the partner speaks.
- **[S] T5:** in 40 forced replies of 1–6 syllables, the own voice held the speech count ≥ 0.3 for up to 3 ticks after
  the last syllable (90 tick-instances), and N fired 0 times.
- **[S] E5 v2:** 40/40 replies ended by the learned marker; 0 by breath; 0 restarts.

**Breath.** At most 12 syllables (corpus p99 = max = 12).

**Reply length.**
- ρ = 1 gives replies that are too short: 3.3 syllables against 5.9 expected.
- ρ is one dimensionless ratio, fitted once to the corpus mean. [S] Same trained model, all boundaries logged at test
  (ρ = 10), then truncated offline:

  | ρ | 0.5 | 1 | 2 | **3** | 5 |
  |---|---|---|---|---|---|
  | mean length (expected 5.9) | 2.3 | 3.4 | 4.2 | **4.5** | 12 (never fires) |
  | stopped by the marker | 20/20 | 20/20 | 20/20 | **20/20** | 0/20 |
  | overlap, matched vs mismatched | 0.47 / 0.12 | 0.40 / 0.11 | 0.37 / 0.11 | **0.36 / 0.12** | 0.22 / 0.15 |

  **ρ ≈ 3 is recommended.** Length is still about 1.4 syllables short (mean |error| 2.7), and the marker learns the end
  only roughly.

---

## 5. B1–B7: status

| B | claim | argument | status |
|---|---|---|---|
| B1 | no movement during a partner line, whatever the weights | channels change only at boundaries; a boundary needs GO or SPK; GO is vetoed while N = 1; M has no other input | [P]; [S] 0 moves in 106 prompts (dbg1 6, E2b 20, E5 4×20) |
| B2 | a line-specific state from line end until GO | hippocampus → hip⇒think → MD latch | [S] 0.80; [A] stays ≥ 0.5 while the PFC learns |
| B3 | speaks 4–5 silent ticks after the line, unprompted | N offset ≤ 3 + latch 1 + GO 1; P(latch < 5 units) ≈ 0 | [P]; [S] 200/200 lines (E4b 3×40, E5 4×20); 0 fallback pushes in 300 training dialogues |
| B4 | familiar dialogues: first syllable from the plan, order from the chain | thal→channel and ear→channel with binding budgets | [S] first syllable 14–16/20; LCS/len 0.36–0.39; overlap 0.36–0.47 vs 0.10–0.12 mismatched |
| B5 | one exact syllable per 3 ticks, never "~" | 1-WTA over exact programs + beat hold | [P]; [S] 0 "~" in 80 replies (618 syllables), every run exactly 3 ticks |
| B6 | stops; never restarts by itself | final marker + breath; CD + W | [P]; [S] 40/40 learned stops (E5 v2), 0 restarts in 80 replies, CD 0/40 leaks (T5) |
| B7 | online learning, bounded weights, no runaway, old abilities kept | each learned total ≤ budget; the structure is not learned; guided cue 16 > 12 + 2 | [P]; [S] guided speech 1.00 after training (old model 0.61–0.83) |

---

## 6. Learned links: formation and number of dialogues

### 6.1 thal → channel (first syllable)

- **Event:** once per dialogue, at the GO tick t0.
  - On at t0−1: about 45–80 latched units.
  - Rising at t0: the r1 channel (plus P_rest if |R| = 1).
- **Change:** +1e−3 per synapse, about +0.045 of current per exposure. There is no back term, because the latch is off
  from t0+1.
- **Mass and binding:**
  - Decay per dialogue is 0.999^(48.7/100) = 0.99951, so τ ≈ 2,040 dialogues.
  - Mass without a budget: M = 0.045·f / 4.9e−4 = 92·f, where f is the share of replies that start with that syllable.
  - Budget 1.0 binds for f > 1.1 %.
- **Dialogues needed:** 8 exposures in E5 give 14–16/20 recall. With idealised codes, 1 exposure gives 0.54–0.74 (T3b,
  recently seen pairs).

### 6.2 ear → channel (sequence) and the final marker

- **Event:** every non-first boundary, about 5.2 per reply.
  - On at b+2: about 700 non-muted ear neurons, of which about 16 reach a given channel (fan-out 96 over 4097 slots).
  - Change per event ≈ 16·1e−4·x̄(0.6) ≈ 1e−3 of current.
  - M ≈ 10.6·p_c, so budget 0.1 binds for p_c > 0.9 %.
- **P_rest:**
  - One event per dialogue with full fan-in (≈700 active), so M ≈ 1e−4·700·0.6/4.9e−4 ≈ 86 ≫ 0.1; it is always
    normalised.
  - **[S] Sparse fan-in (16 inputs) never fired the marker:** all replies ran to the breath limit. Full fan-in gave
    20/20 learned stops.

### 6.3 Recall of familiar dialogues in the full circuit [S, E5]

Setup: 20 probe pairs re-heard once per round of 45 dialogues; 8 rounds (360 dialogues); test with learning off.

| | hip⇒think (w ≤ 0.15) | no hip⇒think |
|---|---|---|
| first syllable right | **16/20** (14/20 in the ρ-calibration run) | 2/20 |
| LCS / reply length | 0.36 | 0.16 |
| overlap with expected vs mismatched reply | **0.47 vs 0.10** | 0.21 vs 0.19 |
| ended by the learned marker | 20/20 | 20/20 |
| mean length (expected 5.9) | 3.3 at ρ = 1; 4.5 at ρ = 3 | 3.6 |

Examples (hip⇒think):

| heard | said | expected |
|---|---|---|
| 这是四川吗 | 对啊 | 对啊 |
| 2015求赐一个男朋友 | 是的猪 | 是的 |
| 我开始还以为你回复的是 | 你是的猪 | 你就是个猪 |
| 想打架的样子 | 哪个猪有 | 哪有想打架 |
| 浙江今天总共七千多高三 | 同待 | 同浙快一万了今年算少的 |

Without the projection, almost every reply is 我是的猪 or 你是的猪.

### 6.4 Budgets must bind (a correction to the current settings) [S, T3b]

With syllable channels and idealised codes:
- **Non-binding budget** (γ holds the mass near 0.6 < 1): every reply starts with "你" (9.4 % of first syllables) at
  every N up to 3k, and recall of a pair heard once is 0.10.
- **Binding budget** (0.05): recall is 0.54–0.74 with a generic-free plan, and 0.21–0.25 when 30 % of the plan is
  generic.

Normalisation turns counts into likelihoods; without it, the frequency prior wins.
- **Rule:** B ≈ 0.3 × the un-budgeted steady-state mass of the most frequent target.
- The thinking-layer set point must keep the generic share of the plan small.

### 6.5 What content exists in this corpus at all [P, exact counts]

Never-heard lines (1,500). Δ = matched − mismatched reply overlap; se ≈ 0.008.

| learner (unlimited capacity unless noted) | 1k | 3k | 10k | 40k |
|---|---|---|---|---|
| plan = last char of Q; greedy bigram; learned stop | −0.005 | −0.009 | +0.008 | +0.013 |
| plan = bag of Q's chars | +0.012 | +0.009 | +0.021 | −0.001 |
| syllable channels, idealised plan, binding budget (T3b) | −0.004 | +0.000 | — | — |
| *copy policy: say the heard line back* | | | | **+0.101** (0.142 vs 0.041) |

- Always saying 你 scores 0.209 on the current `overlap` metric, so that metric cannot tell content from frequency.
- The only large content signal is **copying heard characters**: P(r1 ∈ Q) = 0.148 vs 0.057 shuffled.
- A copy policy needs an auditory→motor identity map (hear c → channel c). The forward-only rule learns that map only
  from "hear c, then say c" events, and the protocol excludes those (the designer's "no shadowing").

Therefore B4 must be tested as **recall of familiar dialogues against a mismatched baseline**. Visible generalisation to
new lines would need either a short imitation phase (the dorsal-stream copy map, 04 §2.1–2.2) or a more regular corpus.

---

## 7. STATE §4 failures, one by one

| # | failure | answer |
|---|---|---|
| 1 | ear afterimage (≈4 ticks) | not relied on; the window is silent before GO |
| 2 | hippocampus 0.95–1.0 | now the plan's content source via hip⇒think: [S] 0.05 → 0.80 |
| 3 | PFC receiving layer poor | off the critical path |
| 4 | thinking layer ≈ chance untrained | reproduced [S]: 0.00–0.05 without hip⇒think, 0.80 with it. The set point must keep the generic share small (recall 0.25 vs 0.54, §6.4) |
| 5 | period-2 alternation | no 2-tick loop has to carry content; the latch is a period-1 self-loop that lives 1–2 ticks |
| 6 | fresh lingers; teacher GO braked; drift | N is speech presence and its offset defines the launch. The teacher never pushes GO ([S] 0 fallbacks). The latch fills 1 tick after N falls and GO follows 1 tick later, so there is no drift window |
| 7 | the rule cannot teach "held thought → onset" | the plan is silenced at t0+1 [P; S T1] |
| 8 | first-syllable current < 0.8 | there is no absolute threshold: GO launches, and content only has to be the largest (1-WTA) |
| 9 | guided speech 0.61–0.83 | recognition margin 16 vs ≤ 14; nothing learned reaches M; [S] 1.00 |
| 10 | never opened its mouth | structural GO; [S] it speaks after every line once at least one channel exists (silent before, because it has no motor repertoire yet) |
| 11 | budgets stable | kept, but they must bind (§6.4) |

**New failures found in this panel's toys, now fixed in the design**
- (a) **GO restarting after the model's own answer.** The latch accumulated more than 105 units during long silences.
  Fixed by W ⊣ thal.
- (b) **Barge-in on repeated syllables,** caused by the reserve-weighted N. Fixed by the count-based N.
- (c) **The stop learning generic sound offsets.** Fixed by placing the marker with the last syllable.
- (d) **Muscle-level top-4 release produced random rare chars:** held-out overlap 0.01–0.05, against 0.12–0.19 with
  channels (T3 vs T3b). Fixed by syllable channels.
- (e) **The marker never fired with sparse fan-in.** Fixed by full fan-in.
- (f) **A large latch can outvote a −5 brake on GO.** Fixed by vetoes (−1000).
- (g) **The teacher's REST cue lost to the cued syllable.** Fixed by raising the cue to +64.

---

## 8. Code changes

**`brain/areas.py`**
- MotorArea: channels (slots, `prog` matrix, recruitment on exact cue match). The boundary becomes 1-WTA plus the final
  marker plus the breath limit.
- Remove `dis_thr`, the instinct path, `d_floor`, `slots`, `go_reset`, `stop()` and the learned internal path.

**`brain/brain.py`**
- `ear⇒fresh`: `use_x=False`, thr 0.3.
- New units: CD (4 Units) and W (1 Unit).
- New fixed paths: GO, SPK and W inhibit thal; hip⇒think (k 32, U(0, 0.15)).
- GO inputs: thal +0.1; fresh and SPK −1000.
- New learned paths:
  - thal→channels (η 1e−3, B 0.5–1);
  - ear→channels (k 96, η 1e−4, B 0.1);
  - ear→P_rest (full, η 1e−4, B 0.1).
- Delete think→go, ear→go, think→motor, ear→motor, pfc⇒motor, hip⇒motor, mctx→motor.

**`talk.py`**
- `say()`: hold the cue until the model's boundary; REST with the last syllable; fallback GO after 8 ticks.
- Train on pairs only; clamp the motor during warm-up.
- `check()`:
  - B1 on lines followed by ≥ 45 silent ticks (back-to-back lines with PAUSE 4 collide by design, §9.2);
  - GO-latency histogram;
  - STOP / BREATH / restart counts;
  - recall on re-heard probes against a mismatched baseline.

**GPU cost per tick**
- thal→channels: ≤ 100 active rows × 4097 ≈ 0.4M MACs.
- ear→channels: 0.07M.
- Compare: PFC receiving internal ≈ 0.18M.
- The 8.2M thal→channel weights take 33 MB.

---

## 9. Weak points and first GPU checks

1. **Localist action channels.** They are recruited by a structural, non-synaptic event. This is grounded in DIVA and
   in basal-ganglia action channels, but the designer said "not one neuron per char". Perception stays distributed;
   only the motor *repertoire* is per synergy, and each channel can be an assembly.
2. **Blanket CD.** The model cannot hear the partner while it speaks, so it cannot yield. Back-to-back lines arriving
   within its 4–5-tick latency collide; this happens only in the current check part 1. The later fix is a learned
   negative image (Schneider 2018).
3. **Content.** Never-heard lines sit at the data ceiling (§6.5). Recall size in the full model depends on the plan
   staying line-specific *while the PFC learns* and on a small generic share [A]. This is untested at full scale.
4. **Loops.** Without a plan buffer the ear chain can cycle (有的小有的小 when the marker fails). The marker plus breath
   bound the cycles; competitive-queuing self-inhibition would remove them, but it resembles refractoriness, so it is
   not proposed.
5. **Toy fidelity.** Exact in the toy: ear, N, thal, GO, motor, rule, world. Simplified: the PFC has born weights only
   (no set-point learning); there are no mctx, relays or returns; the hippocampus also learns during tests (the real
   check turns learning off, and the hippocampus still measured 0.95–1.0). Toy runs reach 360 dialogues, not thousands.
6. **Latency.** 300–375 ms against a human mode of about 200 ms; it is set by the 3-tick ear window.

**First GPU checks, in order**
1. Port §8 and run 500 dialogues. Expected counters: B1 0 moves; B3 every line launched; B5 "~" = 0; B6 0 restarts.
2. Plan decoding at GO−1, with and without hip⇒think, at 0 / 500 / 1000 dialogues. Expected ≥ 0.5 with the projection
   and ≈ chance without.
3. Recall on probes re-heard every 50 dialogues, against a mismatched baseline.
4. Fit ρ to the mean reply length (toy optimum ≈ 3).

**Confidence:** B1/B3/B5/B6/B7 0.85; B2 0.7; familiar-dialogue B4 0.6; never-heard B4 (as worded) 0.1 for any design;
overall 0.65.

---

## 10. Files

| file | content |
|---|---|
| `t1_rule.py` | timing law (§3) |
| `corpus_stats.py`, `ceiling.py`, `ceiling_bag.py` (`ceiling_bag.txt`) | corpus facts; ideal-learner ceilings (§6.5) |
| `circuit.py` (`circuit_v1.py` = before the dense marker) | numpy replica: Net3D ear, N, PFC receiving/thinking/thal, optional hippocampus, CD, W, GO, beat, channels, marker, exact learned paths |
| `e1_trace.py`, `dbg1.py` | traces that exposed barge-in and restart |
| `e2_train.py`, `e2b_train.py` (`e2_v2.txt`, `e2b_out.txt`) | early teacher-protocol runs (superseded by E5) |
| `e4_decode.py`, `e4b_decode.py` (`e4_out.txt`, `e4b_006.txt`, `e4b_015.txt`, `e4b_nohip.txt`) | plan specificity |
| `t3_content.py` (`t3_out*.txt`) | muscle-level release fails on content |
| `t3b_syllable.py` (`t3b_*.txt`) | channels, binding budgets, recall vs held-out |
| `t5_cd.py` | corollary-discharge coverage |
| `e5_recall.py` (`e5_*_v1.txt` sparse marker; `e5_hip_v2.txt`, `e5_nohip_v2.txt` final; `e5_hip_rho.txt` ρ calibration) | full-circuit familiar recall, counters, trace |
