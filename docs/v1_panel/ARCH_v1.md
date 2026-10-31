# ARCH_v1 — one integrated architecture for "hear → hold → answer through muscles"

Synthesis of the panel (proposals: learnability, brain, minimal; six adversarial verifier reports; the content route
was designed in parallel in `arch/content/CONTENT.md` and is integrated in §1.8). Labels: **PROVED** = tick algebra or code reading with numbers;
**SIMULATED** = a numpy run (file named); **ASSUMED** = argued, not run. "My toy" = `arch/synth/archv1.py` + `run.py`
(integrated, all kept parts simulated: exact ear and motor-cortex replicas, learned ear recurrence, hippocampus with
fast learning and relay returns, ear↔motor-cortex learned paths, T/P, turn circuit, syllable programs, exact rule incl.
back term, synchronous update order, real corpus, `world/voice.py` copy).

**Integrated-toy results in one paragraph (SIMULATED).** @@HEADLINE@@

---

## 0. The design in eight lines

1. **Hear** (unchanged): ear = 3D net + cortex with its own learned recurrence; hippocampus = fast chain region with
   relay returns into the ear ("hearing in the mind"); motor cortex = proprioception. None of these is on the path
   that decides *when* or *how* to speak, so none of their learning can break those behaviours.
2. **"Someone else is talking" F** reads the *ear input window* (sound energy, before any cortex): silence is exactly 0,
   speech ≥ 22, so no return, replay or learned current can fake or hide speech.
3. **When**: the offset of F gives one content-free **GO** pulse exactly 4 ticks after the line (no learned weight);
   a 4-unit corollary discharge (driven by GO and by speaking) hides the own voice from F with slack on both sides.
4. **Hold (B2)**: the PFC working memory **T** = fixed random expansion of ear layer 0 (immune to learning) with a
   one-tick self-hold; it is frozen by L+3 (3 ticks after the line) and keeps the heard line (decoded 20/20).
5. **What first**: GO opens T's output gate for one tick (**P = T ∧ GO**); the first beat boundary 2 ticks after GO
   chooses a **syllable program** by winner-take-all over two learned P→Y paths — a fast one (binding budget ≈ 5
   events, forgets) and a slow one (≈ 25-event window, no forgetting) — plus the copy route.
6. **How**: one program per 3-tick beat; D/I/M are slaved to it in the same tick (force ramp 1-2-3); nothing learned
   or returned reaches D/I/M → one clean char per beat, never "~"; a repeated syllable is re-articulated.
7. **What next / stop**: a late one-hot copy **Hc** of the program (on the beat's last tick and the boundary tick)
   teaches next-program (Hc→Y) and **END** (Hc→END, no budget); END wins when its learned input beats the
   breath-discounted sum of everything that still wants to be said; a 12-syllable counter is the backstop. All program
   targets are one-tick pulses, so the back term can never touch these links (0 hits).
8. **Teaching**: the teacher only cues the next syllable / clamps END at the model's own boundaries (never GO);
   decision paths learn only when the release had external drive (local third factor). A **content SLOT** (§1.8) takes
   extra content currents under a cap that keeps the teacher's cue the winner; its default occupant is the content
   specialist's **copy route** (auditory mirror of each program → latch of heard chars → replay pulse before every
   boundary → learned copy weight), the only large content signal for never-heard lines.

---

## 1. Complete specification

### 1.1 Areas and sizes

| area | GPU size | toy size | activity rule | status vs current code |
|---|---|---|---|---|
| ear input layer | 960 lines (8 bands × 6 slices × 10 levels × 2) | same | exactly half on (thermometer pairs) | unchanged |
| ear 3D layers 0–3 | 1,920 / 2,880 / 5,760 / 11,520 born (+25 % spare) = 27,600 slots | same (exact replica) | top-k 40/30/20/10 % | unchanged |
| ear cortex (layers 1–3) | 20,160 born; ≈3,115 muted at warm-up | same | as above | unchanged (learned recurrence, reserve U 0.5 τ 6, muting, split/delete) |
| FEAT0 (ear layer 0 minus its silence-pattern neurons) | 1,152 (= 1,920 − 768, every seed) | same | read-only set | **new readout set** (layer 0 receives no learned/return current and is never split) |
| motor cortex mctx (8×4 force map, levels 3, zs 20/40/80) | 5,600 slots, 3,840 cortex | same | top-k 40/30/20 % | kept as proprioception; **no output to the motor area** |
| hippocampus (mirrors of ear cortex + mctx cortex + motor M and D) | 33,328 slots, ≈20,100 alive | same | top 5 % (≈1,007) | kept; **I neurons no longer chained**; plastic always (also in tests) |
| relays hip→ear, hip→mctx | one per cortex neuron | same | threshold 3 (column ±3) | kept (PFC relays removed) |
| F, F2, GO, R1, R2, CD1–CD4 | 9 units | same | threshold | **new** (replace `fresh`) |
| T (PFC working memory) | 10,000, K = 500 | 4,000, K = 200 | top-K | **new** (replaces PFC receiving + thinking + thalamus) |
| P (T's output gate) | 10,000 | 4,000 | AND (threshold 2) | **new** |
| program assembly per slot: Y (selection pulse), Hp (hold), Hc (late copy), Z (auditory mirror), L (heard-latch), E (replay pulse), U (used flag) | 4,096 slots | 2,048 | WTA over Y at boundaries | **new** (replace per-muscle D release); Z/L/E/U = the content route in the SLOT (§1.8) |
| END | 1 | 1 | competes at boundaries | **new** |
| D, I, M | 32, 32, 96 | same | same-tick, slaved to Hp | kept; no learned or return input |
| breath counter n | 1 register (0–12) | same | reset at the first boundary | **new** (replaces breath reserve / `stop()`) |

### 1.2 One tick (synchronous; the order of `Circuit.step`)

1. World: window(t) = Stream(other's char at t, own voice = mouth(t−1), onsets); lines(t) = encode(window(t));
   E(t) = Σ window(t) (= number of "up" input lines on).
2. Currents for every unit from states at t−1 (all paths). Exceptions, as today: the ear's layers are feed-forward
   within the tick from lines(t); the teacher's cue enters the program scores in the same tick; D, I, M are computed
   from the program hold of the same tick.
3. All parts fire.
4. Learning (prev = t−1, now = t). Decision paths only for targets that had external drive (§1.4).
5. Reserve update (ear, mctx: x ← x(1−0.5) if on at t−1, then x += (1−x)/6); set point of T.
6. Clocks: every 100 ticks → per-target budgets are enforced (a bound, not forgetting); every 100 *experienced* ticks
   (E(t) > 0 or a program held) → the forgetting step L ← γ·L on every learned path. Idle time forgets nothing.
   (The toy applied both on the experience clock; in its idle test nothing learned changes except hippocampal
   replay, see §6.)

### 1.3 Fixed units, connections and thresholds (inputs from t−1 unless marked)

| unit | rule | notes / margins |
|---|---|---|
| F (other is talking) | [0.01·E(t−1) − 10·ΣCD(t−1) ≥ 0.1] | silence E = 0 exactly; any 2 slices of the other's speech ≥ 22 (3,000 lines, brain-dyn expH): margin 2.2× above, ∞ below. **PROVED**: on from line tick 1 to L+2, off from L+3 |
| F2 | F(t−1) | |
| GO | [F2(t−1) − 2F(t−1) − 2SPK(t−1) − 2ΣCD(t−1) ≥ 1] = F(t−2)∧¬F(t−1)∧¬SPK∧¬CD | one pulse per offset; no learned input, no teacher push |
| CD1 | [GO(t−1) + SPK(t−1) ≥ 1]; CDk(t) = CD(k−1)(t−1), k = 2..4 | ΣCD > 0 on [GO+1, E+3]; F vetoed on [GO+2, E+4] (E = END boundary) |
| R1 | GO(t−1) | arms the first boundary |
| R2 (pre-boundary clock) | [GO(t−1) + (SPK(t−1) ∧ phase(t−1)=1) ≥ 1] | on exactly the tick before every boundary; read-out clock for P and the SLOT |
| boundary | [R1(t−1) − 2F(t−1) ≥ 1] ∨ [SPK(t−1) ∧ phase(t−1) = 2] | first boundary = GO+2, blocked if the other started talking |
| T_j | top-K of: Σ_{i∈in(j)} s_i(t−1)·x_i(t−1) + 3.0·T_j(t−1) − 0.3·max(0, u_j/(K/N_T) − 1) + ε_j | in(j) = 30 random FEAT0 neurons (weight 1, fixed); ε_j ~ U(0,0.01) fixed (a constant current, not `base`); u_j += (T_j − u_j)/1000 |
| P_j | [T_j(t−1) + GO(t−1) ≥ 2] | K units for exactly one tick (GO+1) |
| Hc_y | [Y_y(t−2) + Y_y(t−3) ≥ 1] | the program's own delay line: on [B+2, B+3] for a program chosen at B |
| SPK | any Hp on | |
| D_m (same tick) | [2.5·Σ_y Hp_y(t)·[m∈set(y)] ≥ 2] | exactly the 4 muscles of the held program |
| I_m (same tick) | [1.2 − 2·D_m(t) ≥ 1] | tonic; = ¬D_m |
| M_{m,l} (same tick) | D_m(t) ∧ l ≤ phase(t) | force ramp 1→2→3 inside the beat |
| own-voice onset (world) | mouth char ≠ none and (muscle set changed or force dropped) | restarts the own-voice slices (re-articulation) |
| chains | ear cortex, mctx cortex, M, D → hippocampal mirror, 3.0, no reserve | I excluded (its mirror taught a context-free prior: minimal-learning C9) |
| returns | hip mirror → relay (1.0; relay reads its column ±3; threshold 3) → ear / mctx cortex +0.5 | "return strength" knob of the designer (#17) |
| teacher | cue: +12 per matching muscle into each program score; END: clamps the stop at that boundary | only read at boundaries (base-route runs used +10 and an additive END push of +100) |

**Boundary rule** (tick t with a boundary; first = no program held at t−1):
```
drive_y = Σ_j (L^PYfast_jy + L^PYslow_jy)·P_j(t−1) + Σ_y' L^CY_y'y·Hc_y'(t−1) + slot_y(t−1)   (learned part)
score_y = drive_y − 0.05·U_y(t−1) + 12·|set(y) ∩ cue(t)|
s_END   = Σ_y' L^CE_y'·Hc_y'(t−1)                                               (0 at the first boundary)
recruit : cue(t) is a 4-muscle set and no committed y has |set(y) ∩ cue| = 4 → commit the next free slot,
          set(y) = cue (fixed Hp→D wiring), learned inputs 0          (local: no existing program got full drive)
stop    : not first and teacher END                                  → 'END' (clamp; END pulse)
          not first and s_END > 0 and s_END ≥ 0.9^n·Σ_y max(0, drive_y) + 12·[a program is cued]  → 'END' (END pulse)
          not first and n ≥ 12                                        → 'breath'
          max score < θ (0.002)                                       → 'floor' ('silent' at the first boundary)
release : y* = argmax score; Y_y*(t) = 1 (pulse); Hp_y* on t..t+2; phase = 0; n += 1; U_y* = 1 (reset at GO)
stop    : Hp off, SPK = 0 (mouth silent in the same tick)
```
Scales (the content specialist's "MEM, ÷10 form", CONTENT.md §4–5): the first-syllable *memory* paths keep the scale
of P→Y below, while the chain (Hc→Y), END, copy (E→Y), the floor θ and the used-flag penalty are 10× smaller, so that a
remembered first syllable outbids an echo (at equal scale the echo wins: familiar recall 0.78 → 0.25). The END
comparison is the summed-drive rule (breath factor 0.9 per syllable). The **base-route control runs** used the
learnability scales (Hc→Y η 0.01 / budget 2, Hc→END η 0.01, θ 0.02), no slow path, no copy route, the simpler "max"
rule (END iff s_END ≥ best score) and no used flag. **Cue margin (PROVED)**: an exact cued program scores ≥ 48 − 0.05;
any other ≤ 36 + 0.5 (fast P→Y) + 1.5 (slow P→Y) + 0.2 (Hc→Y) + C_slot (≤ 7, copy included) = 45.2; at a cued boundary
END needs ≥ 48 (it is ≤ ≈ 0.2, decay-bounded); the teacher's END clamps. Guided speech is exact however much is learned.

### 1.4 Learned paths (all use the one rule of `net.py`: forward η·x/(1+(w−w0)/w_ref) when src on at t−1 and dst rises at t; back w ← max(w−η, w0) when src on at t and dst on at t−1; per-target budget every 100 ticks, γ every 100 experienced ticks)

| path | fan | w_init | η | w_ref | budget / target | γ | back | gate | status |
|---|---|---|---|---|---|---|---|---|---|
| ear→ear | k 96 (cortex) | U(0,0.01) | 0.03·x | 0.2 | 1.0 | 0.999 | on | — | unchanged |
| mctx→mctx | k 96 | U(0,0.01) | 0.03·x | 0.2 | 1.0 | 0.999 | on | — | unchanged |
| ear→mctx / mctx→ear | k 32 | U(0,0.005) | 0.01·x | 0.2 | 1.0 / 0.5 | 0.999 | on | — | unchanged |
| hip→hip | k 96 | U(0,0.01) | 0.5 | 0.2 | 2.0 | 0.7 | on | — | unchanged; plastic always |
| **P→Y fast** (first syllable) | dense N_T × programs | 0 | 0.1/K_T (0.1 per event) | 0.2 | **0.5 (≈5 events, binding)** | 0.999 | on (never hit, PROVED) | external drive | new |
| **P→Y slow** (first syllable; SLOT memory, CONTENT §4) | dense N_T × programs | 0 | 0.06/K_T (0.06 per event) | 0.6 | 1.5 (≈ 25 events) | **1 (no forgetting; interference only)** | on (never hit) | external drive | new |
| **Hc→Y** (next syllable) | dense programs² | 0 | 0.001 (base runs: 0.01) | 0.02 (0.2) | 0.2 (2.0) | 0.999 | on (never hit) | external drive | new |
| **Hc→END** (stop) | programs → 1 | 0 | 0.001 (base runs: 0.01) | 0.02 (0.2) | **none** (bounded by decay: L*(1+L*/w_ref) = η·r/(1−γ) ≲ 0.2) | 0.999 | on (never hit) | external drive | new |
| FEAT0→Z (SLOT: auditory mirror) | 1,152 × programs | 0 | 0.005 | 0.2 | 2.0 | 0.999 | on (sharpens the template) | own voice only (ΣCD > 0) | new (content route) |
| E_c→Y_c (SLOT: copy, diagonal) | 1 per program | 0 | 0.003 | 0.02 | 0.15 (cap) | 0.999 | on (never hit) | external drive | new (content route) |
| other SLOT paths | per route (§1.8) | 0 | — | — | Σ slot contributions ≤ 7 | — | on | external drive | pending (content) |

Gate (third factor, local): a program unit (or END) applies forward learning to its input weights only on a tick
where it rose *and* received teacher drive, and the back term only on the tick after such a rise. In free speech the
gate is closed, so self-produced replies cannot rewrite the decision weights (minimal-learning: ungated drift 你→的/了).

**Why each timing is learnable (PROVED, timing law of the rule).** Targets Y and END are one-tick pulses at a boundary B,
so only sources on at B+1 can be hit by the back term. P is on only at GO+1 = B1−1; Hc of the program chosen at B−3 is
on [B−1, B]; Hc of the program chosen at B is on [B+2, B+3]. None is on at B+1 → the back term never fires on P→Y, Hc→Y,
Hc→END, **including repeats (哈哈)**, because a repeated program's copy restarts only at B+2. My toy counts back-term hits
on these paths: 0 in every run (3 seeds × 2 base runs × 1,691 taught turns; MEM run incl. the slow path and E→Y — see §1.8).

### 1.5 Protocol (training, test, warm-up)

- **Warm-up** (unchanged): 60 random chars, each followed by 4 silent ticks, learning off; mute ear/mctx cortex neurons
  with usage > 0.6; FEAT0 = layer-0 neurons with usage ≤ 0.6 (needs a layer-0 usage count); 10 silent ticks.
- **Training**: for each dialogue, the pairs (u0,u1), (u2,u3), …; a trailing unanswered line is dropped.
  `listen(q)` (3 ticks per char, no gap); `teach(r)`: on every tick the teacher cues the next syllable (its 4 muscles)
  until the model's own boundary takes it; after the last syllable it clamps END at the model's next boundary;
  **no GO push**, no REPLY_GAP; then `PAUSE_AFTER = 5` silent ticks. Learning always on.
- **Test (`check`)**: on a deep copy of the model, learning ON everywhere (the gated decision paths stay closed without a
  teacher). `reply(q)`: listen, then free ticks until 12 ticks after the turn ends (max 60), and at least 45 ticks from
  the end of one line of the other to the next.

### 1.6 Timing table (one taught turn, tick by tick) — from `trace.py` (SIMULATED, matches the algebra)

Taught turn from `trace.out` (seed 0, after 60 training dialogues): the other says "重买" (ticks 0–5, L = 6), the teacher's
reply is "你送". rel = tick − L. E = input-window energy; "T-J" = Jaccard of T with T(L+3). Hc column = Hc(t) = {Y(t−2), Y(t−3)}.

| rel | E | F | GO | ΣCD | R1 | R2 | P | boundary / motor event | held prog., phase, force | mouth | learning (training) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| −1 | 118 | 1 | 0 | 0 | 0 | 0 | 0 | — (no boundary can exist: R1 = SPK = 0) | — | — | ear, mctx, hip, T set point |
| 0 | 79 | 1 | 0 | 0 | 0 | 0 | 0 | — | — | — | |
| 1 | 37 | 1 | 0 | 0 | 0 | 0 | 0 | — | — | — | |
| 2 | **0** | 1 | 0 | 0 | 0 | 0 | 0 | — (T-J = 1.00 from here: the thought is frozen) | — | — | |
| 3 | 0 | **0** | 0 | 0 | 0 | 0 | 0 | — | — | — | |
| 4 | 0 | 0 | **1** | 0 | 0 | 0 | 0 | GO = F(2) ∧ ¬F(3) | — | — | |
| 5 | 0 | 0 | 0 | 1 | 1 | 1 | **200** (= K_T) | P = T(4) ∧ GO(4); CD1 = GO(4); slot pulses would fire here | — | — | |
| 6 = B1 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | **boundary**: scores from P(5) (+ cue 你) → Y(你) pulse | 你, 0, force 1 | **你** | P(5) → Y(你): +0.1 total (gated by the cue) |
| 7 | 33 (own) | 0 (vetoed) | 0 | 2 | 0 | 0 | 0 | hold | 你, 1, force 2 | 你 | |
| 8 | 65 | 0 | 0 | 3 | 0 | 1 | 0 | hold; Hc(你) on (8, 9); T starts to take in the own voice (T-J 0.42) | 你, 2, force 3 | 你 | |
| 9 = B2 | 94 | 0 | 0 | 3 | 0 | 0 | 0 | **boundary**: scores from Hc(8) = 你 (+ cue 送) → Y(送) | 送, 0, force 1 (onset) | **送** | Hc(你) → Y(送) |
| 10 | 93 | 0 | 0 | 4 | 0 | 0 | 0 | hold | 送, 1 | 送 | |
| 11 | 96 | 0 | 0 | 4 | 0 | 1 | 0 | hold; Hc(送) on (11, 12) | 送, 2 | 送 | |
| 12 = B3 | 101 | 0 | 0 | 4 | 0 | 0 | 0 | **boundary**: END score from Hc(11) = 送; the teacher's END ends the turn → **STOP (END)**; D off, I on, M off in the same tick | — | silent | Hc(送) → END |
| 13 | 69 | 0 (vetoed) | 0 | 3 | | | | own-voice tail | — | — | |
| 14 | 34 | 0 (vetoed) | 0 | 2 | | | | own-voice tail | — | — | |
| 15 | 0 | 0 (vetoed; would be 1: E(14) = 34) | 0 | 1 | | | | last tick the own voice could reach F | — | — | |
| 16 | 0 | 0 (vetoed: 1 tick of slack) | 0 | 0 | | | | | — | — | |
| 17 | 0 | 0 | 0 | 0 | | | | F free again; no offset was ever seen → no second GO | | | |
| 18 | | | | | | | | (PAUSE_AFTER = 5 ends; the next line may start) | | | |

Corollary-discharge slack (PROVED from the table): own voice can reach F on [B1+2, E+3] = [8, 15]; F is vetoed on
[GO+2, E+4] = [6, 16] → 2 ticks of slack at the start, 1 at the end. The free turn in `trace.out` is identical up to B1
(GO at rel 4, P = 200 at rel 5, "我" at rel 6) and stops by END at rel 9.

### 1.7 What each behaviour rests on

| B | rests on | proof / evidence |
|---|---|---|
| B1 listening | a mouth move needs D ⇐ Hp ⇐ a boundary ⇐ (GO, which needs F off) or an ongoing turn; F is on from line tick 1 to L+2 for any line | PROVED; SIMULATED: 0 mouth ticks while the other spoke in 10,146 taught turns (2 runs × 3 seeds) and in every test line (1,680 at 1,000 dialogues + stress) |
| B2 holding | T frozen from L+3 to the own voice (≥ L+8); P reads T(GO) | SIMULATED (learning on everywhere): T at the GO tick decodes 20/20 heard lines in every seed at 500 and 1,000 dialogues; same line after different contexts Jaccard 0.79–0.82 vs 0.047–0.056 between lines; hippocampus 0.80–0.90 (0.18 vs 0.14) |
| B3 turn-taking | GO = offset of F, fixed structure | PROVED; SIMULATED: GO exactly once at L+4 and the first syllable at L+6 in 100 % of 10,146 taught turns (the teacher never pushed GO) and in 100 % of test lines; the model spoke after 1,500/1,500 never-heard lines at 1,000 dialogues |
| B4 content | P→Y (familiar recall), Hc→Y chain; never-heard content = SLOT | SIMULATED: base control (no SLOT): familiar lines 20-way top-1 0.57–0.62 (AUC 0.86–0.88), never-heard ≈ chance (0.049–0.070, AUC 0.49–0.50); with the copy route + memory (MEM): @@MEMB4@@ |
| B5 articulation | one exact program per beat; D/I/M slaved; re-articulation onset | PROVED; SIMULATED: 0 "~" ticks; every one of 62,460 taught syllables and every test syllable lasted exactly 3 ticks; guided speech exact in 100 % of turns, incl. 13.5 % of replies with adjacent repeats; taught repeats 哈哈哈/好好/对对对/嗯嗯 4/4 |
| B6 stopping | END (learned, Hc→END), floor, 12-counter; CD hides own voice → no restart | PROVED for termination; SIMULATED: training stops by END at the right boundary in 5,073/5,073 turns (run 2); 0 restarts in all tests, in 30 back-to-back lines and over 900 + 1,000 silent ticks; free replies (base control) stop by END 83.6 %, floor 6.9 %, 12-counter 9.5 % (loops; MEM: @@MEMSTOP@@) |
| B7 stability | every learned path budgeted or decay-bounded; skills have no learned weight; experience clock | PROVED for bounds; SIMULATED: per-program totals P→Y 0.497 ≤ 0.5, Hc→Y 1.984 ≤ 2.0, Hc→END ≤ 0.42 (decay-bounded); 1,000 idle ticks change no decision weight; but replay keeps changing ear→ear / hip→hip weights in silence (max |Δw| ≈ 1.0 / 6.9), so budgets must be enforced on the wall clock (§1.2) |

### 1.8 The content SLOT (interface for the content route)

The content specialist owns what flows through it; ARCH_v1 fixes only the interface:

1. **Read-out clock.** Slot sources are read at boundary B from their state at B−1 (the R2 tick). Any learned slot path
   X→Y must have X on at B−1 and **off at B+1** (targets are one-tick pulses). Persistent states (latches, held
   memories) must be gated by R2 into one-tick pulses (like P = T ∧ GO).
2. **Signals available at B−1**: T (held thought), P (first boundary only), the hippocampus (plastic, any tick), FEAT0
   (immune to learning, also during the reply: the own voice), the ear cortex, Hc (current program), the per-turn
   syllable count n, F/GO/R2.
3. **Combination rule**: additive currents into the same program units (and optionally END) at the boundary; one WTA.
   No slot current may reach D/I/M, GO, F, CD, P, the ring or the counter.
4. **Budget constraint (so the teacher always wins)**: for every program, Σ_slot contributions ≤ C_slot = 7 (enforced by
   per-target budgets × bounded source counts, or by the hard cap the core applies). Then the cued exact program
   (≥ 4·12 − 0.5 = 47.5) beats every other (≤ 3·12 + 0.5 + 2.0 + 7 = 45.5). The teacher's END clamps the stop, so slot
   currents into programs or END can never block or fake a taught stop; at un-cued boundaries END competes normally.
5. **Learning**: same rule, w0 = 0, gated by external drive (the slot must not learn from free speech).
6. **What the content specialist has established** (read from `arch/content/CONTENT.md`, final version, 439 lines, 2026-10-30 20:2x):

**CONTENT.md is final (439 lines). Its route fits this interface without change and is ARCH_v1's default SLOT filler.**
Labels below are the content specialist's (SIMULATED there = front end and decision layer simulated separately, exact
rule, 2–3 seeds); my integrated run is the end-to-end check (last bullet).

   - **Units per program c** (created with the program at recruitment): auditory mirror Z_c (learned FEAT0 → Z_c,
     η 0.005, budget 2.0, w_ref 0.2, γ 0.999; learns only while the own voice can be in the ear — in ARCH_v1 while
     ΣCD > 0; taught by the program's own delay line, which clamps Z_c at B+4 when the own voice of c fills the window at
     B+3; detection = WTA over Z with threshold 0.75·2.0); latch L_c (set by Z_c ∧ F; cleared by the onset of the
     partner's next line and from B+1 after c is released); replay pulse E_c = L_c ∧ R2 on every pre-boundary tick;
     learned E_c → Y_c (η 0.03, cap 1.5, gated); "used" flag U_c (−0.5 on Y_c for the rest of the reply).
   - **Boundary combination**: additive currents, one WTA; END wins iff s_END ≥ 0.9ⁿ·Σ_y max(0, learned drive_y)
     (breath factor reset at GO); the teacher's END clamps the stop.
   - **Memory for familiar lines**: fast P→Y (≈ 5-event budget, γ 0.999) **plus a slow P→Y from the same gate** (η/5,
     ≈ 25-event window, no forgetting, weight ×3); both ≈ 10× the copy path (else the echo outbids recall: 0.78 → 0.25).
     ARCH_v1 uses the equivalent **÷10 form**: P fast (0.1/event, budget 0.5, w_ref 0.2), P slow (0.06/event, 1.5, 0.6,
     no decay); copy, chain and END paths (η, budget, w_ref), θ and the used penalty divided by 10 (§1.3–1.4).
   - **Results (CONTENT §0, §4)**: never-heard (1,500 pairs, 20-way): no copy route 0.057 / AUC 0.504; copy route (MAIN)
     top-1 0.136 / 0.154 / 0.152 at 1k / 3k / 10k dialogues, AUC 0.587–0.602; recommended variant (copy γ 0.9999 + latch
     set point ≤ 10 %) 0.151 / 0.172 / 0.202, AUC 0.583–0.616; MEM scaling costs 0.004–0.013 top-1. Familiar (200
     probes, 2 seeds): first syllable 0.82 after 3 exposures, 0.90–0.91 after 5; retained 0.76–0.81 / 0.54–0.58 /
     0.46–0.50 / 0.39–0.40 at +<1k / +1–3k / +3–6k / +10k dialogues (fast path alone 0.58–0.60 / 0.22–0.25 / 0.02–0.03 /
     0.01–0.04); whole familiar replies top-1 0.45–0.47 (AUC 0.89) after 5 exposures. Replies 4.7–6.8 syllables, 0
     loops, backstop ≤ 3.2 %. Caveat (theirs): mostly echo — it repeats 49–61 % of the heard line's distinct chars
     (humans ≈ 15 %); copy weights are frequency-weighted counts (no local normalisation exists in the rule).
   - **ARCH_v1-specific note (PROVED):** CONTENT.md wires a full E→Y matrix and proves that only the diagonal survives,
     because its program units are *held* 3 ticks (the back term at B+2 cancels every cross term). In ARCH_v1 programs
     are one-tick pulses, so cross terms would *not* cancel; ARCH_v1 therefore wires E_c only to its own program Y_c
     (structural, part of the program's assembly: Y_c, Hp_c, Hc_c, Z_c, L_c, E_c, U_c are created together at
     recruitment). The learned statistic is then identical (copy events only). Cue margin in the ÷10 form: 4·12 − 0.05
     vs 3·12 + 0.5 + 1.5 + 0.2 + 0.15 (+ other slot ≤ 7) — PROVED.
   - **Options listed by CONTENT.md, not adopted as defaults / not in my run**: copy γ 0.9999 and the latch set point
     (≤ 10 %; static form simulated, dynamic set point ASSUMED) — their recommended variant; a sparse "dentate-like"
     code S as the fast path's source (same numbers); splitting very frequent programs (ASSUMED).
   - **In my integrated toy with MEM (all realistic parts, ARCH_v1 timeline), seed 0, 1,000 dialogues + 6 familiar rounds**
     (my 500 never-heard pairs; reference points on this exact set: always-你 top-1 0.051 / AUC 0.496, say the last
     heard char 0.113 / 0.540, echo the whole line 0.240 / 0.646): @@ECHO@@

---

## 2. Every FATAL / SERIOUS issue of the six reports → resolution in ARCH_v1 → evidence

Codes: LD/LL = learnability proposal, dynamics/learning verifier; BD/BL = brain proposal; MD/ML = minimal proposal.

| # | issue (report) | ARCH_v1 resolution | evidence |
|---|---|---|---|
| 1 | **FATAL BD-F1**: "other is talking" counted ear-cortex neurons; returns + learned ear recurrence fill the cortex in silence → never launches, restarts (explains STATE #10) | F reads the ear **input window energy** at t−1 (pre-cortical); returns, recurrence, hippocampal replay, split/delete cannot reach it | PROVED (silence = 0, speech ≥ 22); BD verified the fix; **my toy, with returns + learned ear→ear + learning hippocampus + mctx paths: GO exactly once per heard line at L+4 in 10,146/10,146 taught turns and every test line; F off at L+3 in 100 %; 0 GO during a line; spoke after 1,500/1,500 never-heard lines** |
| 2 | **FATAL LL-F1**: P→Y readout of the real thought gives ≈ chance first syllables on never-heard lines; no accumulation; corpus caps association readouts | P→Y (fast + slow) is the **familiar-line** route; never-heard content is assigned to the content **SLOT**: the copy route (the only large signal in this corpus: echo AUC 0.67, learned copy +0.12–0.15 Δov) | my toy, base route, never-heard: base control at 1,000 dialogues (3 seeds, 500 never-heard pairs): top-1 0.049 / 0.056 / 0.070, AUC 0.490 / 0.500 / 0.503 (chance) (chance as expected); familiar: base control: top-1 0.57 / 0.60 / 0.62, AUC 0.87 / 0.86 / 0.88; first syllable 0.88–0.96 for rare and 0.53–0.60 for top-10 first chars (3 seeds); MEM: @@MEMFAM@@; content specialist (offline, same timing law): copy route lifts never-heard top-1 0.061 → 0.12–0.14, AUC 0.51 → 0.58–0.60 (`arch/content/eval_copy2_s0_1k.out`) |
| 3 | **FATAL ML-F1**: decision readout with counts and non-binding caps from the hippocampus learns no content (prior dominates) | per-program **binding** budget on fast P→Y (0.5 = 5 events) plus a slow P→Y path; source = T (fixed, specific share 0.86 vs hippocampus 0.14); P→Y learns only at the first boundary | ML verifier offline: T + budget 0.94 recall, retention 0.85; my toy familiar set: base control: top-1 0.57 / 0.60 / 0.62, AUC 0.87 / 0.86 / 0.88; first syllable 0.88–0.96 for rare and 0.53–0.60 for top-10 first chars (3 seeds); MEM: @@MEMFAM@@ |
| 4 | **SERIOUS LD-S1**: breath reserve never refills (τ 100 > turn period) → END missed in 44 % of teacher replies | breath = plain **syllable counter reset at the first boundary** (≤ 12), checked *after* END; the teacher's END clamps the stop | PROVED; my toy training: run 1 (old order: counter before END): END 95 % + counter 5 % (= the 12-char replies) at the right boundary; run 2 (END first, clamp): END 5,073/5,073 |
| 5 | **SERIOUS LL-S1**: content memory short and volatile (≈ last 20 contexts per target, 68–89 % of first syllables change with lag) | skills are permanent (no learned weight in GO, F, CD, beat, articulation, counter); the fast P→Y path is a recency memory (≈ 5 contexts per program); a **slow P→Y path** (≈ 25-event window, no forgetting) adds long-term recall; forgetting counts experienced ticks only | PROVED (skills have no learned input); CONTENT §4 (2 seeds): retention at +3–6k dialogues 0.00 (fast only) → 0.43–0.48 (MEM); with the copy route the first syllable changes for only 21–23 % of never-heard lines between 1k and 10k (no-copy design 91–99 %); my toy: weights bounded (P→Y 0.497 / Hc→Y 1.984 per program, Hc→END ≤ 0.42); idle 1,000 ticks: decision weights unchanged; skills unaffected at every checkpoint |
| 6 | **SERIOUS BD-S1**: B2/B4 numbers depended on the hippocampus learning during tests (and a guided pass before the test) | all tests run with learning on everywhere (hippocampus plastic, as in life), on a model copy or after training has ended; no guided pass before tests; the first-syllable source is T, which no learning reaches | my toy B2 (learning on): T decodes 20/20 at GO in all 3 seeds at 500 and 1,000 dialogues (same 0.79–0.82 vs different 0.05); hippocampus 0.80–0.90 |
| 7 | **SERIOUS BD-S2 / BL-S3**: when PFC layers learn, the plan loses specificity and alternates (period 2) | the learning PFC loop (receiving ↔ thinking ↔ thalamus) is **removed from v1**; T has fixed input and a one-tick self-hold (period 1); only a slow set point moves it | LD verifier: T hold 0 changes in 2,896 turns, set-point shift once then 8 % jitter; my toy: T(L+3) vs T(GO) Jaccard Jaccard(T(L+3), T(GO)) = 1.00 (trace); decode 20/20 with all learning on, B2 decode T decodes 20/20 at GO in all 3 seeds at 500 and 1,000 dialogues (same 0.79–0.82 vs different 0.05); hippocampus 0.80–0.90 |
| 8 | **SERIOUS BL-S1**: familiar recall at budget 1.0 ≈ frequency level; reported numbers needed hip learning at test | binding fast budget (≈ 5 events; BL's own fix 0.2–0.5 → 7–13/20 after 2–4 exposures) + slow path; T source; memory paths 10× the copy path; tests ≥ 20 dialogues after the last exposure, learning on | CONTENT §4: first syllable 0.82–0.91 from the 3rd exposure (MEM); my toy, familiar 40 pairs × 6 exposures, tested 50 dialogues later: base control: top-1 0.57 / 0.60 / 0.62, AUC 0.87 / 0.86 / 0.88; first syllable 0.88–0.96 for rare and 0.53–0.60 for top-10 first chars (3 seeds); MEM: @@MEMFAM@@ |
| 9 | **SERIOUS BL-S2**: learned stop marker was a recency effect of ear features; replies looped to the breath limit | END learned from the **one-hot late program copy Hc** (clean timing, no generic ear features), unbudgeted, decay-bounded; 12-counter backstop | PROVED (timing); my toy test stops: test stops (base control, 1,500 lines): END 83.6 %, floor 6.9 %, 12-counter 9.5 % (loops such as 明明明…, no used flag in the control); MEM: @@MEMSTOP@@ |
| 10 | **SERIOUS BL-S4**: no route for never-heard content | the content SLOT (§1.8) with the read-out clock R2 and the cap; content specialist's echo route | see row 2 |
| 11 | **SERIOUS MD-S1**: any pause ≥ 3 ticks = end of turn; no abort; a resumed line is never answered | scope limit stated; the first boundary is blocked if F is on at GO+1 (covers pauses of 3–4 ticks); harness keeps ≥ 45 ticks between the other's lines and PAUSE ≥ 5 after END; proposed (not built): efference-cancelled detector + abort rule (§6) | my toy stress "interrupt": pauses of 2–4 ticks inside the other's turn: 0 mouth ticks during the continuation, answered after it (18/18); pauses of 5–9 ticks: the model talks over the continuation (29–63 mouth ticks per 6 trials) |
| 12 | **SERIOUS ML-S1**: toy content tests ran with frozen weights | all my tests: learning on, familiar lines heard 6 times, tested later | method (run.py) |
| 13 | **SERIOUS ML-S2**: first-syllable path trained at every boundary (6× more next/END events, shared core) | P exists only at GO+1, so P→Y can only learn at the first boundary (structural) | PROVED; my toy: P fires once per turn P = 200 units on exactly one tick per turn (trace); P→Y learns only at first boundaries (structural) |
| 14 | **SERIOUS ML-S3**: back term erodes earlier learning; END not learnable from held/ear sources | all decision targets are one-tick pulses; sources P (1 tick) and Hc (on [B+2, B+3]) are never on at B+1 | PROVED; my toy back-term hits: 0 in every run (3 seeds × 2 base runs × 1,691 taught turns; MEM run incl. the slow path and E→Y — see §1.8) |

MINOR items that ARCH_v1 also closes: idle decay silencing the model (LD-M5, LL-M10 → no learned skill weight + experience clock);
teacher GO misalignment (LD-M3 → no teacher GO); CD zero start margin (BD-M2 → CD driven by GO as well: slack 2 ticks at the
start, 1 at the end); repeated syllables not learnable / merged by the world (BD-M7, BL-M1, MD-M6 → re-articulation onset +
pulse targets + late copy); recruitment oracle (LL-M5, ML-M2, BL-M2 → local rule, §1.3); w_init of decision paths (LL-M6 → 0);
REST/END cue coupled to ρ or to an unbounded learned sum (BD-M1, BL-M4, MD-M1, CONTENT §5.1 → the teacher's END clamps the stop);
zero-tolerance pause after END (MD-M2 → GO needs no rising edge; PAUSE 5); split children of muted parents unmuted (BD-M3 →
change list); check() collisions (LD-M7, MD-S1 → new check()).

---

## 3. The designer's framework: kept, changed, removed

| designer's idea (FRAMEWORK.md §一) | ARCH_v1 | why |
|---|---|---|
| binary neurons; one local rule "earlier → later strengthens, later → earlier weakens" (#9, 2026-10-30) | **kept unchanged** (net.py arithmetic, back term on everywhere; all `back=False` exceptions removed) | the panel proved what the rule can learn (timing law); ARCH_v1 places every learned link where the rule works instead of changing the rule |
| sensory 3D nets, fixed activity, featureless neurons muted, slow split/delete, "near the input splits least" (#5–7) | **kept** (ear and motor cortex unchanged) | they work (ear decodes the line 1.00, STATE #1); F and T read only the input window / layer 0, which split/delete never touch (split_frac[0] = 0) |
| hippocampus = fast chain region, short-term memory (#2, #5) | **kept** (chains 3.0, η 0.5, γ 0.7, budget 2), now **plastic always** (also in checks) | real-time learning; BD-S1 showed it is a within-line learner; it is the natural home of a fast memory in the SLOT |
| returns are normal; return strength is a knob (#16, #17) | **kept for the hippocampus** (relays → ear, mctx, +0.5) | "hearing in the mind" stays; it can no longer break turn-taking because F reads the input window |
| one global weight decay as clean-up (#3) | **kept, clock changed**: decay counts *experienced* ticks | otherwise idle time erases learned content (LD-M5); skills have no learned weights at all |
| motor: tonic inhibition vs disinhibition decides movement; muscles only (#4, 原话 35) | **kept**: I tonic, D disinhibits, M ramps; a stop = programs off → inhibition back | same principle, now with no learned or return current into D/I/M (guided speech fell to 61–83 % because of them, STATE #9) |
| other areas decide what the motor area does; PFC often not needed (#12, #21) | **kept in spirit**: GO, beat and articulation run without the PFC; the PFC (T) only proposes the first syllable once | "大致意图 → 其他区把话说出来" |
| no new "speech area" (#22) | **kept**: GO, CD, beat ring and programs are generic motor/turn structure (corollary discharge, content-free start, syllable channels) | brain anchors: Inagaki 2022, Schneider 2018, Moll 2023, DIVA |
| no refractory period (#15) | **kept**: CD inhibits the *detector* F, never the speaking neurons | |
| don't stack rules (#10) | the turn logic is 9 threshold units with fixed weights; the content paths are three tables with one rule | every unit is in the tick table (§1.6) |
| PFC = thinking; multi-layer; some PFC neurons belong to no area and have no returns (#8, #24, #31, 10-30) | **changed**: v1's PFC is T (working memory with fixed input from the earliest auditory layer and a one-tick self-hold) + its output gate P; the learning PFC loop (receiving ↔ thinking ↔ thalamus) is **removed from v1** | the loop never held line content (STATE #3–#5), learned ≈ 0 into GO (#7) and drifted / alternated when it learned (BD-S2, BL-S3); T decodes 20/20 untrained and cannot drift. **Design debt**: "PFC really thinks" is not realised in v1; a slowly learning PFC can return through the SLOT once a check shows it adds content |
| "fresh" detector over the ear cortex | **changed** to the input-window energy | cortex counts include imagery (BD-F1) |
| motor area selects muscles (per-muscle D, top-4) | **changed** to syllable programs (one recruited unit per 4-muscle set), D/I/M slaved | per-muscle selection turns any learned mixture into an unrelated clean char (Lemma 6). Tension with #8 "not one neuron per char": only on the motor side (an action repertoire; HVC/DIVA analogues); perception stays distributed |
| learned → GO, think→motor, ear→motor, mctx→motor, motor-internal, pfc/hip⇒motor | **removed** | held sources cannot teach onsets (timing law); they interfered with cued speech |
| teacher pushes GO; REPLY_GAP; say() with D cues | **changed**: teacher cues only at the model's own boundaries; END clamp; pairs only | train and test share one timeline (LD-M3, MD-M2) |
| born full of random connections, training modifies them (#23) | **kept** for every area and path that existed; **changed** for the new decision paths (P→Y, Hc→Y, Hc→END, E→Y, FEAT0→Z start at w = 0) | a random born current (≈ 5 per program from 500 P units) would swamp the 0.02 floor and the learned 0.1–2 scale (LL-M6); program units themselves are recruited on use, like split/delete |
| the newest idea: variable sensory counts (net3d `vary_from`) | **compatible**, off by default; must keep `vary_from ≥ 1` (layer 0 fixed) | nothing in turn-taking or articulation reads cortex counts |

---

## 4. Change list vs the current code (`/root/work/concept-brain`, commit 004031a + think→motor)

Everything below is concrete enough to port; the toy (`archv1.py`) is a working reference for each item.

**`brain/net.py`**
1. `Path.learn(gate_now=None, gate_prev=None)`: optional per-target third factor — forward increments only on targets with
   `gate_now`, back term only on targets with `gate_prev` (passed by the owning area). Rule arithmetic unchanged.
2. New `DensePath(n_src, n_dst, eta, w_ref, gamma, budget)` with w0 = 0: `current(src_ids)`, `learn(src_prev_ids,
   src_now_ids, rising_ids, held_prev_ids)`, `decay()` — for P→Y fast and P→Y slow (10,000 × 4,096 each), Hc→Y
   (4,096²), Hc→END (4,096 × 1), FEAT0→Z (1,152 × 4,096) and the diagonal E→Y (4,096).
   Evaluate `current` only on boundary ticks and `learn` only when a gated target rose (GPU: one row-gather + sum).
3. `Circuit.step(experienced: bool)`: split `Path.decay()` into `forget()` (L ← γL) called every 100 *experienced*
   ticks (ear input energy > 0 or a program held) and `normalise()` (per-target budget) called every 100 ticks.
4. `Region`: add a constant `bias` current (for T's tie-breaker ε), because `homeostasis()` overwrites `base`.
5. Decision-path constants (§1.4, MEM ÷10 form): P→Y fast (η 0.1/K_T, w_ref 0.2, budget 0.5, γ 0.999), P→Y slow
   (η 0.06/K_T, w_ref 0.6, budget 1.5, γ 1), Hc→Y (0.001, 0.02, 0.2), Hc→END (0.001, 0.02, none), E→Y (0.003, 0.02,
   cap 0.15), FEAT0→Z (0.005, 0.2, 2.0); all `w_init = 0.0`; remove every `back=False`.

**`brain/net3d.py`** — no change. Assert `vary_from ≥ 1` and `split_frac[0] == 0` (layer 0 must stay fixed: FEAT0, T).

**`brain/areas.py`**
1. `SenseArea`: count layer-0 usage during warm-up; `feat0()` = layer-0 alive neurons with usage ≤ 0.6.
   Bug fix (BD-M3): in `restructure()`, compute and mute featureless neurons *before* `clear_usage()`, and mark split
   children of muted parents as muted (today talk.py's re-mute after restructure mutes nothing).
2. Replace `MotorArea` with a program motor area: `prog` (4,096 × 32 bool), `committed`; state = held program id, phase,
   Y-pulse history (4 ticks, for Hc), counter n, END pulse; `boundary = (R1(t−1) ∧ ¬F(t−1)) ∨ (SPK(t−1) ∧ phase(t−1)=2)`;
   `select()` exactly as §1.3 (drive, used-flag penalty 0.5, cue 12 per muscle, recruitment, END clamp / summed-drive
   END rule with breath factor 0.9ⁿ, 12-counter, floor θ); D/I/M same tick (I = ¬D, M = D ∧ level ≤ phase).
   Delete `dis_thr`, `d_floor`, `slots`, `go_reset`, `push()`, `stop()`, free D firing and the learned motor-internal
   path. `cue(muscles)` → a muscle set read at boundaries only; new `end_clamp()`.
   Content route (SLOT default, per program slot, created at recruitment): Z (auditory mirror; learned FEAT0→Z, gated
   by ΣCD > 0; clamped at B+4 by the program's delay line; WTA detector with threshold 1.5), L (latch: Z ∧ F; cleared
   at the partner's next line onset and after the program's release), E = L ∧ R2 (replay pulse), learned E_c→Y_c
   (diagonal only), U (used flag, reset at GO).
3. New `TurnUnits`: F (reads the input-window energy from Brain.step), F2, GO, R1, R2, CD1–4 (fixed weights §1.3).
4. New `WorkingMemory` T (top-K 5 %, fixed fan-in 30 from FEAT0 × reserve, self 3.0, set point β 0.3 τ 1,000, ε bias)
   and its gate P (= T ∧ GO, one tick).

**`brain/brain.py`**
- Remove: `pfc`, `think`, `thal`, `fresh`; paths `*⇒pfc`, `pfc⇒*中继`, `*中继⇒*` of the PFC, `pfc→think`, `think⇒pfc`,
  `think→pfc`, `think⇒thal`, `thal⇒think`, `thal⇒thal`, `ear⇒fresh`, `fresh⇒go`, `fresh⇒thal`, `think→go`, `ear→go`,
  `think→motor`, `ear→motor`, `mctx→motor`, `pfc⇒motor`, `hip⇒motor`; cfg fields chain_pfc, ret_pfc, pfc_own, pfc,
  think, in_think, think_in, w_fb, thal_group, thal_w, thal_need, fresh_*, ret_motor, to_go, think_mot, ear_mot, mctx_mot.
- Keep: ear, mctx, hippocampus with its chains (ear cortex, mctx cortex, motor **M and D only**), hippocampal relays →
  ear / mctx (ret_hip 0.5), ear↔mctx learned paths, ear and mctx internal paths, warm-up muting.
- Add: TurnUnits, T, P, program motor area, DensePaths P→Y / Hc→Y / Hc→END, an optional `slot` object (§1.8).
- `step(lines, energy, cue=None, end_clamp=False)` → (force, events{release, stop reason, recruit, onset}).
- Compute cost: removes ≈ 36k units and ≈ 3.3M learned edges (PFC receiving internal 24k×96, thinking 10k×96,
  pfc→think 24k×32, think→pfc 10k×32 …) and two relay populations; adds two dense tables touched only at boundaries.

**`world/voice.py`** — `Stream.tick(..., new_self=False)`: restart the own-voice slices on re-articulation (one line;
`arch/synth/voice_copy.py` is the patched copy).

**`talk.py`**
- `Talker.tick`: `energy = heard.sum()` → `brain.step`; own-voice onset = mouth char present and (muscle set changed or
  force dropped) → `new_self` on the next tick.
- `say()` → `teach(reply)`: every tick, cue the next syllable until the model's release takes it; after the last,
  clamp END at the model's next boundary; **no GO push**; `PAUSE_AFTER = 5`; count teacher timeouts (must stay 0).
- `main()`: iterate (other, reply) pairs; drop a trailing unanswered line; delete REPLY_GAP.
- `reply()`: listen (pause 0) + free ticks until 12 ticks after the stop (max 60) and ≥ 45 ticks before the next line;
  syllables counted by onset (repeats count); returns GO tick, first-move tick, stop reason, restarts; **B1 measured here**.
- `check()` (rewrite; runs on `copy.deepcopy(talker)` with learning on, ≈ 3 min on GPU):
  (1) mechanics on 20 probes (B1 moves while the other speaks, GO tick, first move, stop reasons, restarts, '~', run
  lengths); (2) B2: T (and hippocampus) at the GO tick, each probe after two different context lines, Jaccard 1-NN;
  (3) training audit counters since the last check (guided exact, END at the right boundary, back-term hits, own-voice
  leaks, recruitments); (4) content: **20-way discrimination** (char-set F1 of what was said vs the true reply and 19
  others; top-1 chance 0.05, AUC 0.5) on a familiar set (40 pairs re-heard every 100 dialogues, tested ≥ 20 dialogues
  after the last exposure) and on ≥ 500 never-heard pairs; copy rate (said chars in the heard line vs a shuffled line);
  first-syllable accuracy split top-10 vs rare first chars; mean length; stop reasons; seeds. The old `overlap()` is
  kept only as a secondary number (always-"你" scores 0.21 on it).

---

## 5. Pre-training checks (must pass on the GPU implementation before any long run) and milestones

**A. Before any long run: 50 dialogues (≈ 70 turns) + one `check()`; every line must pass, else stop and fix.**

| # | check | pass threshold | toy value (3 seeds, 1,000 dialogues) |
|---|---|---|---|
| 1 | GO: one per heard line, at L+4 | 100 % of training turns; 0 GO during a line | 10,146/10,146 turns; 0 GO during lines |
| 2 | F offset / silence margin | F off at L+3 in 100 %; window energy in silence = 0 exactly; min energy on line ticks ≥ 1 is ≥ 22 | F off at L+3 in 100 %; silence energy 0 by construction; ≥ 22 (brain-dyn expH, 3,000 lines) |
| 3 | first release | at L+6 in 100 % of training turns; teacher timeouts 0 | L+6 in 100 %; 0 timeouts |
| 4 | B1 | 0 mouth ticks while the other speaks (training and check) | 0 in training and all tests |
| 5 | guided speech | said == teacher reply in 100 % of training turns (incl. repeated syllables) | 100 % (10,146 turns; 4/4 taught repeats) |
| 6 | B5 | 0 "~" ticks; 100 % of syllables exactly 3 ticks | 0 "~"; 100 % exactly 3 ticks |
| 7 | B6 in training | 100 % stop by END at the boundary right after the last syllable; 0 GO after the reply; 0 F-on ticks during the own turn (CD coverage) | END 100 % (run 2); 0 restarts; 0 own-voice F ticks |
| 8 | learnability of decision links | back-term hits on P→Y (fast, slow), Hc→Y, Hc→END, E→Y = 0; P = K_T units on exactly one tick per turn | 0 hits; P = 200 on one tick |
| 9 | T hold | Jaccard(T(L+3), T(GO)) = 1.00 in 100 % of turns | 1.00 (trace) |
| 10 | programs | committed = distinct chars the teacher has spoken; pool never full | 1,367 programs = distinct taught chars; never full (2,048 slots) |
| 11 | bounds | per program: P→Y fast ≤ 0.5, slow ≤ 1.5 (+ small overshoot between normalisations); Hc→Y ≤ 0.2 (base scale 2.0); Hc→END ≤ 0.2 (2.0); E→Y ≤ 0.15; ear→ear ≤ 1.0 per target right after normalisation | P→Y 0.497, Hc→Y 1.984, Hc→END 0.42; ear→ear 1.5–1.7 / hip 13–17 measured between normalisations |
| 12 | B2 | T decodes 20/20 at GO; T same-line Jaccard ≥ 0.5 and different-line ≤ 0.1 | 20/20; 0.79–0.82 vs 0.047–0.056 |
| 13 | idle | 1,000 silent ticks: no GO, no mouth move, no change of any decision weight (experience clock) | 0 GO, 0 moves, decision weights unchanged; ear/hip replay weights changed (→ wall-clock budgets) |
| 14 | speed | ≥ 500 dialogues per 30 min (V100), check() excluded | expected faster than now (PFC loop removed) — ASSUMED |

**B. Milestones (realistic, from the toy; never-heard numbers need the SLOT's copy route).**

| when | must hold | content expectation |
|---|---|---|
| 200 dialogues | speaks after ≥ 90 % of heard lines, always at L+6; checks 1–13 still pass | first syllables = a handful of frequent programs (你/我/哈 …) |
| 1,000 dialogues | as above; stops by the 12-counter ≤ 3 % (MEM default) | familiar first syllable (6 exposures, ≥ 20 dialogues later) ≥ 0.7 with MEM (CONTENT §4: 0.82–0.91); never-heard with the base route ≈ chance (my control: top-1 0.049–0.070, AUC 0.49–0.50); familiar in my toy: @@M1@@ |
| 1,000 / 3,000 dialogues **with the copy route in the SLOT** (content specialist's GPU targets, CONTENT.md §6) | as above, plus: echo map — heard char's Z wins at its clean tick ≥ 0.75 / 0.80; latch precision ≥ 0.90, recall ≥ 0.70 | never-heard top-1 ≥ 0.12 / 0.14, AUC ≥ 0.57 / 0.585; copy rate ≥ 0.50 vs ≤ 0.10 shuffled; length 4.5–6.5; backstop ≤ 3 %; repeated bigrams ≤ 1 %; false echoes ≤ 5 %; **control** (E→Y off = the base route) top-1 ≤ 0.07, AUC ≤ 0.52. My integrated echo run: @@ECHO_SHORT@@ |

---

## 6. What remains ASSUMED, and the risks

1. **Toy ≠ GPU code (ASSUMED).** The toy uses exact replicas and the exact rule, but its own RNG draws (e.g. k targets drawn
   with replacement), T at 4,000/200 instead of 10,000/500 (verifiers found the same hold/drift at GPU size), 2,048
   program slots, and no restructure (split/delete). F and T are immune to restructure by construction (they read the
   input window and layer 0, which is never split); its effect on hippocampal content is untested.
2. **Content is still crude (SIMULATED).** Without the SLOT, never-heard lines are at chance (my control runs). With the
   copy route the model mostly echoes heard chars (it repeats ≈ 4× more of the heard line than humans do), and copy
   weights are frequency-weighted counts (no local normalisation exists in the rule — CONTENT §3.4). Familiar recall
   rests on the first syllable only; later syllables come from the generic chain and the copy route (whole-reply
   familiar top-1 saturates ≈ 0.45 in CONTENT §4.3). Very frequent first syllables are forgotten first (per-program
   budgets); the framework's splitting of constantly used neurons is the proposed remedy (untested).
3. **Reply length and loops.** With the simple "max" stop rule (END vs the best continuation) and no used flag (the base
   runs), replies average 4.2 syllables (corpus 6.2) and 9.5 % of free replies loop to the 12-counter (a learned
   repeat such as 天→天 beats END). The adopted default — the content specialist's summed-drive stop with a 0.9ⁿ breath
   factor and the used flag — gives 4.7–5.8 syllables and 0 loops in their sims; in my integrated echo run: @@ECHO_LEN@@.
4. **Scope limit: cannot yield.** A pause of ≥ 5 ticks inside the other's turn is taken as the end of the turn; the model
   then speaks over the continuation and cannot hear it (CD) — stress test: pauses of 2–4 ticks are handled (no overlap, answered after the continuation, 18/18); pauses of 5–9 ticks give overlaps of 29–63 mouth ticks per 6 trials. Proposed (not built): an
   *efference-cancelled* detector F′ = [input energy − predicted own-voice energy ≥ θ], where the prediction is a learned
   inhibitory copy from the held program to FEAT0 (Schneider 2018), plus an abort rule: F′ on while speaking → the next
   boundary forces a stop and leaves the turn open (GO fires again at the next offset).
5. **Latency 6 ticks** (≈ 450 ms at 75 ms/tick; 3 ticks are the ear window) vs a human mode of ≈ 200 ms.
6. **Localist programs.** One unit per taught muscle set; untaught chars cannot be said; 4,096 slots (3,970 chars).
   Recruitment is a structural event (like split/delete), local but new.
7. **World change.** Re-articulation (own-voice phase restart when the force drops) is a change of the physics; without
   it repeated syllables merge into one long sound (13.4 % of replies have an adjacent repeat).
8. **T is not "thinking".** v1's PFC holds a recency-weighted bag of the heard line; abstraction and B8 are not addressed
   (design debt toward the designer's main goal).
9. **Set-point drift of T.** T codes shift once (≈ first 100 dialogues) then jitter ≈ 8 % per checkpoint (LD verifier);
   early P→Y links go stale. Harmless for the fast path; the slow path's long-term retention (0.43–0.48 at +3–6k in
   CONTENT §4) already includes this drift, but it caps how long a memory keyed on T can last.
10. **Hippocampal overshoot.** With η 0.5 the hippocampus's per-target learned input reaches ≈ 13 between two decays
    (the budget is enforced only every 100 ticks) — unchanged from today and off every behaviour path in v1,
    but it matters if the SLOT reads the hippocampus.
11. **Toy run time.** The integrated toy ran at ≈ 12 ms/tick alone and ≈ 30 ms/tick on the shared 4-CPU box.

---

## 7. summary_zh

这次把三个方案、六份挑错报告和"说话内容"专家的结果合成了一个架构 ARCH_v1，并做了一个把所有现有部分都放进去的模拟（真实的耳朵和运动皮层、耳朵自己的联想学习、海马体的快速学习和回送、真实语料），每个种子训练 1000 段对话。

**确认能用的（所有训练和测试里 0 次出错）：** 别人说话时嘴完全不动；别人说完后第 4 拍自动"起动"、第 6 拍开口，每次都一样，不用老师推；每个字干净地说 3 拍，从不出现含糊音，同一个字连说两遍也能分开说；说完会停，停了不会自己再开口；刚听到的那句话被"记住"，100% 能认出是哪一句。@@ZH_RESULT@@

**改了什么、为什么：**
1. "别人在说话"改成直接看耳朵收到的声音响不响，不再数皮层里亮了多少——回送和联想会让皮层在安静时也"听到声音"，模型就永远不开口（这正是以前从没开过口的原因）。
2. 什么时候开口由固定结构决定（别人声音一停就起动），不靠学习——现在的学习规则学不会"一直亮着的想法 → 开口"。
3. 运动区改成"每个字一个动作程序"，每拍只放一个——按单块肌肉挑会把几个字拼成别的字。
4. 前额叶原来那套会学习的回路暂时拿掉，换成一个固定的"记住刚听到什么"的层——那套回路一学习就把想法弄乱、来回跳。
5. "说什么"的连接只在老师示范时改——不然模型会把自己乱说的当成对的。
6. 加了"跟着说听到的字"这条路（耳朵里每个字的声音模板 → 记下这句里听到了哪些字 → 回答时可以说出来），以及快、慢两条记忆，让听过几次的话能记住第一个字。

**还没解决的：** 对没听过的话，内容主要靠"重复听到的字"，比真人重复得多；回答整体还很粗糙；别人中途停一下再接着说时，模型会抢话，还不会让。

---

## 8. Files (all in `arch/synth/`)

| file | what |
|---|---|
| `archv1.py` | the integrated toy: exact Net3D replica (ear, motor cortex), exact `Path` replica (SparsePath, DensePath), hippocampus with relays, F/F2/GO/R1/R2/CD, T/P, programs (Y, Hp, Hc), END, D/I/M, experience-clock decay, slot hooks; `Talker` (world, teacher, free replies) |
| `voice_copy.py` (`voice_orig.py`) | `world/voice.py` + the one-line re-articulation flag (`new_self`) |
| `run.py` | training protocol with per-turn audits, checkpoint tests on a clone (learning on), B2 decoding, 20-way content metrics, stress tests; `python3 run.py SEED N TAG [key=value]` |
| `echo_slot.py` | the content SLOT plug-in drafted from the content specialist's echo route (Z map, latch, R2 replay, diagonal E→Y) |
| `trace.py` → `trace.out` | tick-by-tick trace (timing table §1.6) |
| `agg.py` | aggregates `res_<tag>_s<seed>.json` |
| `run1_base_s{0,1,2}.{log,json,out}` | first set of base-route runs: checkpoints at 0/250/500 and full training audits to 1,000 dialogues; crashed at the model-pickling step (fixed) before the final checkpoint |
| `res_base_s{0,1,2}.{log,json}` | base-route control runs (1,000 dialogues + 6 familiar rounds, 3 seeds; final checkpoint incl. idle test; stress on seed 0) |
| `res_mem_s0.{log,json}` | the adopted default (copy route + fast/slow memory, MEM ÷10 form, summed-drive stop, used flag, cue 12), seed 0, with stress and idle tests |
| `run1_echo_s0.log`, `run2_echoMAIN_s0.*` | interrupted runs of the earlier MAIN content config (audits only) |
| `baselines.py` | reference policies (always-你, echo, last char) on the exact test sets |
| `extract.py`, `assemble.py`, `*.md` section files | digest of results; assembly of this document |
| `smoke.py`, `smoke_echo.py`, `prof.py` | smoke tests / profiling |
