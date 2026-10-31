# ARCH_v2 — design "brainlike": brain-faithful areas and loops (designer 3 of 3)

Folder `arch/v2/brainlike/`.  Labels: **PROVED** (argument with numbers), **SIMULATED** (numpy run; file and numbers),
**CITED** (paper, finding), **ASSUMED**.  All runs numpy, `OMP_NUM_THREADS=1`, on the shared bench v0.3 (copied in
`benchcopy/`, metrics untouched) with two world options: `vocab=` the bench's own 142 chars (so babbling can reach
the vocabulary under random physics) and `eye_layout='retino'` (added in my copy only; innate retinotopy, see the bench
issues).  The agent never receives a muscle cue: every program comes from babbling, every reply from listening.

## 0. Verdict
1. **A brain-faithful division of labour runs end-to-end on the bench without a single muscle cue.**  Babbling (with
   the tutor imitating) gives the motor inventory (96 % of the B/C syllables in the final agent, all self-found); everything the agent says afterwards it learned by listening to other speakers' replies (tutor 'model').
   Mechanics stay exact in every stage: 0 mouth ticks while a partner speaks, latency 6, 100 % 3-tick syllables.
2. **Two capabilities are clearly new; most scene and all memory questions are not yet solved.**  Seed 0, all
   probes of each stage, against the official bar (v1, teacher-forced) and the privileged baselines:
   scene naming "这是什么" 0.51 (bar / prior / n-gram 0.06 / 0.06 / 0.04) and the frame "你要X吗 -> 要X" 0.85, held-out
   fillers 0.86 (n-gram 0.48, held-out ~0); stage totals B 0.33 / C 0.51 / D 0.19 vs bar 0.12 / 0.19 / 0.19 and prior
   0.14 / 0.30 / 0.22; C held-out 0.48 vs n-gram 0.29.  But where / slot / colour are at prior level, taste and ritual
   are below the prior / n-gram, and D (check, name, before, one-shot) is at or below the prior.
3. **The designer's "PFC = recurrent next-input predictor" is the core of the design**, with the loop going through
   the PFC and an error-driven readout: next-syllable prediction while listening 0.82 / 0.82 / 0.77 vs 0.71 / 0.59 /
   0.61 for the best online exact n-gram on the same events (B / C / D).
4. **Holding a thought by a loop works mechanically** (gated 1:1 MD loop: period-2 index 0, held 200 ticks with
   noise, clean replacement) and the content gist variant passes every hold test (decoding 1.00, same line 0.98,
   different lines 0.09) — but it did not improve behaviour, so the final configuration keeps the history-rich gist
   (decoding 0.45).
5. **Lesions: the readout, the dorsal copy route, vision, word<->vision templates, LEX words and the PFC structure
   (DQ, CTX) all reach behaviour; the hippocampus, the memory and visual copy gates and the location-word map are
   passengers in the trained agent** and must be fixed or removed (brief, rule 3).
6. **What made it work were not more layers** but correct signals: own voice from the efference copy (not the CD
   timer), words by mutual predictability, a referent held through the reply and released at a new initiative, a
   turn role from timing, assembly codes for syllables in the mixed layer, per-pathway normalised delta learning.
   Leaky units alone do NOT prevent period 2 (SIMULATED); a within-layer self-hold does.
7. **Open problems**: location words -> attention (slot 0.08, where at prior) and colour; frame choice between
   competing frames (taste below the "好吃" prior); D-stage memory (episodes keyed to drifting codes; a self-caused
   attention loop); motor inventory for late words (the 512-program pool filled with duplicates); ONE seed only (the 3-seed V100
   job timed out, B.10); recast mode and lawful physics implemented but not measured.

## A. Design

### A1. Block diagram
```
 EAR-0 (fixed; cochlear nucleus)  --energy-->  BODY turn units F/GO/CD (innate, ARCH_v1)
   | FEAT0 (1,152 lines, 40 % layer)
   v
 SYL  (STG syllable units; association rule, template form; recruited at theta-clean ticks when nothing matches)
   | syllable events (speaker tag from efference: own vs other)
   +--> ECHO  phonological loop: order rule, one-shot, line-local; frozen copy = "the last partner line"  (DORSAL)
   +--> S_f / S_b  transition statistics: order rule, slow, pre / post normalisation
   |       -> LEX word units = syllables + chunks with mutual high transition probability               (VENTRAL)
   v
 LEX trace (current line) ----------------------+-------------------------------+
                                                 |                               |
 EYE (retinotopic, fixed) -> slot feature maps --+--> A priority map (LIP/FEF):   |
                                                 |    pointing (exogenous) + word templates VL + location words LA
                                                 |    -> held referent (BG gate) -> IT_att (gain-gated pooling)
                                                 v                               v
 PFC  SUP  (L2/3: mixed selectivity over LEX, IT_att, A, MATCH, DQ, DC, CTX, CA1, PREV (assembly code), turn ROLE
            (assembly code), POS; k-WTA 5 %; learned recurrent co-activity assemblies)
      DC   (deep, caudal: current-line gist, leaky tau ~ 20 ticks)
      DQ   (deep, held: BG gate loads DC at the end of a partner line, holds through the reply; leaky + MD loop)
      CTX  (rostral: dialogue topic, leaky over lines, tau ~ 100 ticks)
                                                 |
 HIPPOCAMPUS  EC (DQ + line words + scene pooled + scene per slot + CTX) -> DG (fixed, 2 %) -> CA3 (one-shot
              EC->CA3 + CA3<->CA3 co-activity) -> CA1/content (line syllable order, scene snapshot, gist)
              recall at every partner line end; sleep(): replay into the slow word<->vision templates
                                                 v
 READOUT (premotor/IFG; decision rule): next syllable or END from x = [SUP, DQ, DC, CTX, ROLE, PREV, LEX, ECHOcand,
          POS, IT_att, A, MATCH, DONE (copied word completed), bias] (each pathway normalised) + BG copy gates G_s x source s:
          H (heard line), Vo/Vc/Vn (word evoked by the attended object's shape/colour/number), Vl (location word),
          Vh (new information in the recalled line), Vr (word for the attended slot of the recalled scene)
                                                 v
 BODY: program WTA at the 3-tick beat, END/counter, D/I/M  -> mouth -> own voice -> ear
       programs from babbling (trial sets kept when their own sound is a known syllable); dorsal mirror Mir
```

### A2. The three local rules (exact; code: rules.py; unit tests: test_rules.py, 21/21 pass)
Notation: binary states s(t); onsets on_i(t) = an event of unit i at t; age_i = ticks since i's last onset counted
before t; M = the path's broadcast gate (third factor).
- **R1 ORDER (sequence areas: ECHO, S_f, S_b)** — the designer's rule, onset-based, with an eligibility window:
  e_i(t) = lam^age_i if age_i <= W else 0;  on_j(t): w_ij <- min(w_ij + eta_f M e_i(t), w_max);
  on_i(t): w_ij <- w_ij - eta_b M e_j(t) w_ij (floor 0);  same-tick onsets never link.  Optional normalisation at
  learning time: 'pre' (sum_j w_ij <= B: forward probability P(j|i)) or 'post' (sum_i w_ij <= B: backward P(i|j)).
  ECHO: eta_f = eta_b = 1 (one-shot), W = 4, cleared at each partner-line onset, frozen copy at its end.
  S_f/S_b: eta_f = 0.05, eta_b = 0, W = 4, pre / post normalisation, B = 1.
- **R2 ASSOCIATION (assembly areas: SYL templates, VL word<->vision, LA word<->location, Mir syllable<->program,
  SUP recurrent assemblies, EC->CA3, CA3<->CA3)** — co-activity in either order, held states count:
  w_ij += eta M x_i y_j, then immediately for every target j that learned: if sum_i w_ij > B_j: w_.j *= B_j/sum_i w_ij.
  (Implemented exactly with a per-target scale, O(active) per event.)  Template form (SYL): the winner j:
  w_.j <- (1 - eta M) w_.j + eta M B x/|x|.
  SYL: eta 0.05 (others' voices), 0.025 (own voice), B 1; VL: eta 0.2 under reliable attention (pointing, or a
  held referent / top-down match >= 0.8), 0.02 cross-situational (all slots pooled), B 1 per word and map; LA: eta
  0.2 under reliable attention; Mir: eta 0.5, B 1 per program, M = own-voice detection 3 ticks after release;
  SUP recurrent: eta 0.002, B 1; EC->CA3 and CA3<->CA3: eta 5e-4, B 1 (one-shot, palimpsest ~ 1/(K eta) per unit).
- **R3 DECISION (readout W, copy gates G_s)** — eligibility tag x three-factor, predictive (delta) form, signed:
  tag e = x at the prediction event (the previous syllable event or the line end; life 12 ticks), or a decaying
  trace e <- lam e + x;  p_j = clip(sum_i w_ij e_i + copy_j, 0, 1);
  w_ij += (eta M / max(|e|^2, 1)) e_i (target_j - p_j)  (normalised LMS; negative part = learned inhibition through
  an inhibitory pool, Dale-compliant).  Copy gate s: target 1 if the heard next syllable is source s's candidate,
  0 if s offered a different one.  W: eta 0.3; G: eta 0.3; innate imitation bias G_H(bias) = 0.3.
- **Third factors (broadcast, few)**: M_listen = 1 for syllables / line ends of OTHER speakers (teacher-free
  learning by listening, also overheard dialogue), 0 for the own voice (efference copy: mouth active within 3
  ticks); M_reward = the partner's contingent reaction r (recast +1 if acceptable, play +0.2) applied to the own
  reply's eligibility tags (W, target = what was said); M_att = attention reliability (gates VL/LA);
  M_own = own-voice clean tick (gates Mir and the own-voice template update); M_enc (hippocampus) = partner line end.

### A3. Areas (GPU design size / CPU prototype size)
| area (brain analogue) | GPU | prototype | activity | fan-in | persistence | learns (rule, signal) |
|---|---|---|---|---|---|---|
| EAR-0 (cochlear nucleus) | 1,920 (FEAT0 1,152) | same | 40 % top-k | 4 fixed | 1 tick | no (innate) |
| SYL (STG syllable units) | 4,096 slots (an assembly of ~8 per syllable) | 256 (localist) | 1 winner at a clean tick (cos >= 0.8) | all FEAT0 (1,152) | event | R2 template; recruited when a theta-clean tick matches nothing (cos < 0.75); others' voices + own voice |
| ECHO (phonological loop) | 4,096^2 sparse | 257^2 | — | — | current line; frozen copy of the last partner line | R1 one-shot |
| S_f / S_b (STS statistics) | 4,096^2 sparse | 256^2 | — | — | slow (whole life) | R1 slow, pre / post normalisation |
| LEX (STS/MTG word forms) | 16k (syllables + chunks) | 640 | trace 0.8 per event | 2 syllables per chunk | current line | chunk recruited when P(b|a) and P(a|b) >= 0.75, >= 6 tokens |
| EYE (retinotopic) | real retinotopic map | 3 slots x 240 | sparse | — | — | no (innate retinotopy) |
| A (LIP/FEF/pulvinar priority map) | retinotopic map | 3 | 1 winner | pointing (fixed) + words via VL, LA (learned) | referent held from the partner's line through the reply | via VL/LA |
| IT_att (attended object) | 232 x positions | 232 | 24 bits | gain-gated pooling of the attended slot | event | no (gain gating) |
| VL_o/c/n (ATL hub: word <-> shape/colour/number) | 16k x (3 maps) | 640 x 232 | — | — | slow | R2, M_att |
| LA (word <-> location) | 16k x map | 640 x 3 | — | — | slow | R2, M_att |
| PFC-SUP (L2/3, input/selection) | 20k | 3,000 | 5 % k-WTA | 30 fixed random (mixed selectivity over 11 streams; the previous syllable and the turn role enter as assembly codes) + 3,000 learned recurrent | event | R2 slow (assemblies) |
| PFC-DC (deep, caudal: line gist) | 10k | 1,000 | 5 % | 20 from SUP | leaky 0.85/event (tau ~ 20 ticks); reset at line onset | no |
| PFC-DQ (deep, held line) | 10k | 1,000 | 5 % | 1:1 from DC at the gate | leaky 0.9/tick + MD loop gain 0.3 (holds; noise sd 0.3 tolerated) | no; BG gate |
| PFC-CTX (rostral: topic) | 5k | 500 | 5 % | 20 from DQ | leaky 0.6/line (tau ~ 100 ticks) | no |
| MD thalamus | 10k (1:1 with DQ) | folded into DQ | — | 1:1 | — | no |
| BG gate | few units | event logic | — | turn units | — | innate (update at partner line end, hold during the reply) |
| HIPPO EC | ~30k | 3,928 | — | — | — | — |
| DG | 50k | 2,000 | 2 % | 30 fixed | — | no |
| CA3 | 50k | 2,000 (= DG index) | 2 % | EC->CA3 learned, CA3->CA3 learned | palimpsest ~1/(K eta) episodes per unit | R2 one-shot (eta 5e-4) |
| CA1 / content | — | episode store (index -> line order, scene snapshot, DQ) | — | — | — | one-shot (idealised) |
| READOUT W (premotor/IFG) | 100k x 4,096 | 6,909 x 257 | — | 14 context pathways, each normalised | tag 12 ticks | R3, M_listen, M_reward |
| copy gates G (BG on the dorsal route) | 7 | 7 | — | 6,909 | — | R3 |
| Mir (dorsal mirror Spt-vPMC) | 4,096 x 4,096 | 256 x 512 | — | — | — | R2, M_own |
| BODY (ARCH_v1) | — | same | — | — | — | no (innate) |

### A4. Motor learning and the body (no muscle puppeting)
- **Babbling (stage A).** When the agent is idle and quiet (>= 3 silent ticks) it requests a spontaneous GO
  (p 0.5 per quiet tick; brainstem vocal play).  At the first boundary the mind proposes a TRIAL set (a random
  4-of-32 muscle set, not yet a program), which the body holds for one beat (3 ticks, force 1-2-3).  Three ticks after
  the release the own voice is clean in the ear; if it evokes a known syllable unit (cos >= 0.8; the syllable was
  heard from others before = "the ambient language") the set is committed as a program and the dorsal mirror Mir
  learns syllable <-> program (R2).  Garbles and unknown sounds are dropped.  Linked programs are also practised
  (p 0.2 of babbles), which refines Mir and the own-voice templates.  The tutor's imitation (+0.2) is a reward.
- **Imitation.** The heard line's ECHO offers its syllables through the copy gate G_H (innate bias 0.3, then learned):
  hearing c -> SYL c -> Mir -> program.  This is the dorsal "hear -> repeat" route of ARCH_v1, with order (ECHO is
  an order-rule store instead of v1's unordered latch).
- **Speaking.** At every boundary: p(c) = clip(W x + max_s G_s cand_s) over syllables; program score = max_c p(c)
  Mir[c, y]; END score = p(END).  The innate body does the WTA, the beat, END-vs-best, breath and floor.
- Body kept unchanged (ARCH_v1): F, F2, GO, CD1-4, R1, boundary, phase, program WTA, END rule (END >= best and >=
  floor), breath 12, floor 0.05, D/I/M ramp, re-articulation.  Two additions: an idle-time GO request (babbling) and a
  trial-beat path (babble a set that is not a program).
- **Own-voice tag.** The agent's perception treats a syllable as its own iff the mouth was active within 3 ticks
  (efference copy).  (Bug found: v1's CD timer also runs after every GO even when the turn is then blocked; using CD as
  the own-voice tag deleted the 2nd syllable of every model reply and taught "END after one syllable".)

### A5. Teaching without muscle puppeting
1. Stage A: child-directed statements with pointing, overheard two-speaker routines, free play (the tutor imitates
   babbles, reward +0.2).  The agent never gets a cue; everything it can say comes from babbling.
2. Stages B-D, tutor 'model': the partner asks, a third speaker (or the partner) says the correct reply; the agent
   learns by listening (M_listen) the next-syllable / END readout, the copy gates, word<->vision, word<->location,
   and the episodes.  Overheard dialogue is the same signal.
3. Tutor 'recast' (contingent feedback): the agent answers; reward 1 if acceptable is the third factor on the reply's
   eligibility tags; the recast is then learned by listening exactly like a model reply (DQ still holds the
   question: the BG gate does not reload DQ on the own reply).
4. Teacher forcing ('force') is never used by this agent.

### A6. Developmental stages and pass thresholds (design targets; measured values in part B)
| stage | input | pass threshold before moving on | measured (seed 0) | passed? |
|---|---|---|---|---|
| A hear + babble | CDS with pointing, overheard routines, play | SYL units for >= 90 % of heard chars; programs for >= 90 % of B/C chars; LEX within-word TP >= 3x across-word, chunk precision >= 0.6; 0 mouth ticks while others speak | SYL 67 units (every heard char); programs 93-96 % of B/C chars; TP 0.72 vs 0.012; precision 16/19; 0 overlap | yes |
| B naming with a scene | model tutor | what >= 0.8, slot >= 0.7, where >= 0.7 (seen), held-out >= 0.6 | what 0.64 seen / 0.40 held; where 0.54 / 0.16; slot 0.18 / 0.01 (whole stage; late third ALL 0.57) | no |
| C frames, held-out fillers | model tutor (+ recast) | offer held-out >= 0.7, colour / count >= 0.6, taste >= 0.8, ritual >= 0.8 | offer held-out 0.86; colour 0.17; count 0.36; taste 0.54; ritual 0.43 | offer only |
| D relations, memory | model tutor | check >= 0.8, name >= 0.6, before >= 0.5, one-shot >= 0.5 | check 0.48; name 0.02 (first syllable 0.83); before 0.03; one-shot 0.00 | no |
| E adult corpus interleaved | model tutor | B-D retained >= 90 %; echo-free never-heard above the n-gram baseline | not run | — |
In the brief's spirit (gate the next stage on the previous), I ran B-D on a fixed schedule to measure everything; a
real curriculum would repeat B until its thresholds pass.
Mechanics at every stage: mouth while the partner speaks = 0, syllables of 3 ticks = 100 %, latency 6.

### A7. GPU-scale cost (design sizes)
Synapses: SYL 4.7 M (dense 1,152 x 4,096); ECHO/S sparse (active rows only); VL 16k x 700 = 11 M; SUP 20k x 30 fixed
+ 20k x 1,000 learned = 20 M; DC/DQ/CTX 0.5 M fixed; DG 50k x 100 fixed = 5 M; EC->CA3 50k x 1,000 = 50 M, CA3<->CA3
50k x 1,000 = 50 M; readout 100k x 4,096 = 410 M (dense; only active rows touched: ~2k x 4,096 = 8 M per event);
gates 0.7 M; Mir 16 M.  Total ~ 0.6 G synapses = 2.4 GB float32 (fits a 32 GB V100 with room for traces).
Compute per syllable event (one per ~3 ticks): readout 8 M MAC, SUP recurrent 20 M, CA3 at line ends 50 M; per tick
~10 M MAC on average -> ~1 ms/tick on a V100 with batched sparse kernels -> 10^6 ticks (~ 10k dialogues) in ~ 20 min.
CPU prototype (sizes in the table): 1.2-1.5 ms/tick single core numpy; 800 stage-A dialogues = 52k ticks = 65-80 s.

### A8. The designer's framework: kept / changed
| framework idea | here | why |
|---|---|---|
| binary neurons, fixed active fraction per area, set points | kept (k-WTA everywhere; SYL is a one-winner detector) | |
| "earlier -> later strengthens, later -> earlier weakens" (#9) | kept as R1 in the sequence areas (ECHO, statistics); back term onset-based | the state-based clause is what made held thoughts unlearnable (skeptic S5) |
| "don't strengthen co-active neurons" (#9) | relaxed in association areas only (R2, with immediate normalisation) | assemblies, word<->object binding, one-shot episodes need it (unit tests R2a-b, M1) |
| one global decay as clean-up (#3) | budgets act immediately (R2 normalisation); global down-scaling only in sleep() | delayed budgets overshoot (hippocampus 13-17 vs 2) |
| hippocampus = fast chain region, short-term (#2, #5) | fast one-shot, LONG-lasting episodic index + replay | clinical evidence (H.M.) and the final goal |
| PFC: multi-layer, outputs return as inputs, concept units without returns (#8, #24, 10-30) | SUP <-> DC <-> DQ <-> CTX loops; DQ/CTX/SUP drive nothing outside the PFC except through the readout | |
| "thought range moves slowly" (#29) | DC leaky (tau 20), CTX leaky (tau 100), DQ held by the BG gate | |
| motor = inhibition vs disinhibition, muscles only (#4) | kept (ARCH_v1 body) | |
| random muscle moves + listening = one way to make connections (#12) | babbling is THE way programs are made | |
| split / delete (#5-6) | recruitment of new units on novelty: SYL units, LEX chunks (statistics or fast mapping), programs, episodes | |
| no "speech area" (#22) | kept: the readout is a generic premotor next-event predictor | |
| PFC learns from raw sound (old loop) | changed: the PFC reads learned words, attended objects, held referent, recall | the old loop drifted / alternated |
| not one neuron per char (#8) | tension: SYL and programs are localist in the prototype (an assembly per syllable at GPU scale) | |
| teacher drives the muscles | removed (no cue is ever used) | |

### A9. Fixed (innate) mechanisms — count and plan to learn them
Body (ARCH_v1, kept, 11): F, F2, GO, CD1-4 (one delay line), R1, boundary/phase beat, program WTA, END-vs-best rule,
breath counter, floor, D/I/M ramp, re-articulation, + 3 additions: idle-time GO request (vocal play: always in
stage A, later only when nothing is shown and nobody speaks), trial beat, own-voice keep rule for babbled sets.
Perception (5): EAR-0/FEAT0; SYL one-winner detection threshold; SYL novelty-recruitment threshold; theta sampling
(clean tick = onset + 2 + 3k, reset by an acoustic onset after silence); own-voice tag from the efference copy.
Line structure (3): partner line onset / end from window energy edges (silence onset); turn role from timing
(a line starting <= 8 ticks after another ended is a reply).
Vision (6): retinotopic slot maps; exogenous capture by pointing; attention = argmax with threshold; referent held
by the BG gate and released at a new initiative; gain-gated pooling (IT_att); comparator (word template vs pointed
object, which also vetoes re-binding of a well-learned mismatching word).
PFC (4): fixed random mixed-selectivity fan-in; DC leak + reset at line onset; DQ load at partner line ends + MD hold;
CTX leak per line.
Hippocampus (4): DG fixed expansion; recall + store at partner line ends; episode content = line order + scene
snapshot + DQ (idealised CA1); "new information" filter (recalled syllables not in the current question).
Readout (3): copy-pointer logic (word onset -> first syllable, inside -> next, after -> done flag); per-pathway
normalisation; tag life 12 ticks.
LEX (2): chunk recruitment by mutual transition probability; fast mapping of two adjacent novel syllables (after
2,000 heard syllables).
Total: 41 fixed mechanisms (body 14 incl. 3 babbling additions; perception 5; line structure 3; vision 6; PFC 4; hippocampus 4; readout 3; LEX 2).  Generic (could serve any modality): theta sampling, k-WTA,
leaky integration, gate-at-boundary, gain-gated attention, DG expansion, copy pointer.  Task-flavoured and to be
learned later: the "new information" filter (should become a learned CA1 mismatch signal), the copy-pointer logic
(should become a learned sequence (CA3 chain / ECHO) readout), the comparator (a learned conjunction), the energy-edge
line segmentation (should become learned turn-end prediction), the BG load-at-line-end gate (PBWM: learn it from
reward), fast-mapping (novelty-gated recruitment).

### A10. Risks
1. Localist inventories (SYL units, programs, chunks) are easy to read and to learn in this world (every syllable is
   one fixed 3-tick sound) but will not survive real audio, coarticulation or variable syllable length.  At GPU scale
   SYL should become an assembly code (several units per syllable) with the same template rule.
2. Babbling only covers the vocabulary because the Voice was restricted to the bench's 142 chars (random physics);
   with the 3,970-char Voice a target char needs ~3x10^4 random babbles.  Lawful physics (bench option, pending) is the
   real fix: an inverse model from babbling then generalises to new chars.
3. The copy-pointer logic and the "new information" filter are engineered; they work for 1-2 word replies, not for
   long utterances.  They should become learned sequence readouts (CA3 / ECHO chains).
4. The readout is one delta layer over a wide context; frame words, END and copy gates are linear in the context.
   Compositional generalisation beyond "frame + copied word" is not shown.
5. Attention needs an innate retinotopic eye; the bench's random per-(slot, value) codes have no location structure
   (my copy adds eye_layout='retino'; on the standard eye only the no-vision path is possible).
6. The episode store's CA1 is an idealised index -> record lookup (perfect content, palimpsest only in the index).
7. The agent says its prediction whenever the body opens a turn, e.g. it "rehearses" the predicted next question
   after a model reply; when to speak should be learned (BG value), not fixed.
8. Plasticity in SUP (recurrent assemblies) drifts the context codes on which the readout is keyed (measured in
   the drift probe); replay-based re-alignment would be needed for long lives.

### A11. What the iteration taught (each found by a failing test, then fixed in the design)
1. **Own-voice tag ≠ CD timer.**  v1's corollary-discharge delay line also runs for 4 ticks after every GO, even when
   the first boundary is then blocked by the next speaker.  Using it as "my own voice" deleted the second syllable of
   every model reply (the reply starts 3 ticks after the prompt): stage B stayed at 0.00-0.02 and the readout learned
   "END after one syllable" (SIMULATED, diag: heard 在 _ 间 for 在中间).  Fix: own voice = mouth active within the last
   3 ticks (efference copy).  After the fix the same probes went to 7/8 correct.
2. **Words need mutual predictability.**  Forward transition probability alone recruited frame junk (果在, 车在 from
   X在哪里), which then hijacked visual naming (果 before 苹).  Requiring P(b|a) and P(a|b) >= 0.75 (presynaptic and
   postsynaptic normalisation of the same order rule) leaves 14/17 real words.
3. **The referent must be held through the reply** (BG gate), else location and colour words are bound to nothing:
   attention decayed per syllable and the reply's words arrived after it had faded.
4. **A wide delta readout needs per-pathway normalisation**: ~400 active context units swamped the few decisive ones
   (attention, attended features); normalising each pathway to unit norm (and NLMS) fixed it.
5. **Copy pointers must offer a word at any word onset** (要 + X, 在 + 左边), not only at the reply start.
6. **Leaky units alone do not prevent period 2** in a synchronous 2-layer loop with conflicting contents: the
   potentials meet at the tie and then flip every tick (SIMULATED, M3').  A within-layer self-hold (DQ's 1:1 MD
   loop) does prevent it.  This qualifies the reviews' "leaky units kill period 2".
7. **A recall is an answer only if it partially matches the question** (CA1-style match/mismatch): an unconditioned
   "new information from the recalled episode" source answered where-questions with a recalled "这是什么".

8. **A self-confirming attention loop (found, not yet solved).**  When a reply's own location word can re-steer
   attention, a noisy template confirms itself (after stage A, 右's location template pointed only 0.43 at the right
   slot and 右边 0.72 at the WRONG slot).  "Hold the referent during a reply" + "bind words only under pointing or a
   held referent" fixed the templates (右 0.86, 左 0.71, 中 0.73; stage-A agents A8/A9) but cost behaviour: stage B
   0.18 vs 0.33 over 1,000 dialogues, because statements like "X在左边" no longer teach.  Reverted; the slot questions
   (0.08) are the price.  A graded version (learn less, not never, under self-caused attention) is the next step.
9. **Vocal play must be a single beat.**  Practising a known program through a normal release let the turn continue:
   the next-syllable predictor kept talking into the partner's next line (27 overlapping mouth ticks in 25 B
   dialogues).  As one-beat trials: 0 overlaps in B, C and D while the program inventory keeps growing.
10. **The referent belongs to one exchange.**  A held referent that survived into the next question of the same scene
   bound the next reply word to the previous object (牛奶's template fell from 1.00 after stage A to 0.29-0.37 in B).
   Released at every new initiative (a line that does not follow within 8 ticks): templates stay at 1.00.

### A12. Brain grounding of each choice (CITED; mostly via the five reviews and research/0x, not re-read here)
- Dual streams: dorsal sound -> articulation (repetition), ventral sound -> meaning (Hickok & Poeppel 2007); the copy
  route and the generative route compete for the mouth, as in dual-route repetition models (Ueno et al. 2011).
- Syllable-rate sampling: auditory cortex theta tracks syllables and resets at acoustic edges (Giraud & Poeppel 2012).
- Statistical segmentation by infants at 8 months (Saffran, Aslin & Newport 1996); fast mapping of novel words
  (Carey & Bartlett 1978; Horst & Samuelson 2008 on poor retention without repetition).
- Babbling + contingent feedback shape the syllable inventory (Oller; Goldstein, King & West 2003); the auditory
  target / inverse map is learned from the own voice (DIVA, Guenther 2016).
- Attention: a priority map in LIP/FEF/pulvinar; biased competition and gain modulation select the attended object's
  features (Desimone & Duncan 1995; normalisation model of attention, Reynolds & Heeger 2009); joint attention to a
  pointed object from ~9-12 months.
- Word <-> object binding in a cross-modal hub (ATL, Patterson et al. 2007); cross-situational learning (Smith & Yu 2008).
- PFC: mixed selectivity (Rigotti et al. 2013); BG gating of working memory, update vs maintain (PBWM, O'Reilly &
  Frank 2006); MD thalamus sustains PFC representations (Schmitt et al. 2017); rostro-caudal hierarchy of abstraction
  and timescale (Badre & D'Esposito 2009; Murray et al. 2014); temporal receptive windows lengthen word -> sentence ->
  narrative (Lerner et al. 2011).
- Hippocampus: DG pattern separation, CA3 one-shot autoassociation, CA1 comparison of recall with input (match /
  mismatch novelty; Lisman & Grace 2005); index theory (Teyler & DiScenna 1986); one-shot plasticity (Bittner et al.
  2017); replay in sleep consolidates into cortex (Wilson & McNaughton 1994; Girardeau et al. 2009); complementary
  learning systems (McClelland, McNaughton & O'Reilly 1995).
- Plasticity: co-activity potentiation in either order at high rates (Sjostrom et al. 2001); heterosynaptic
  normalisation at the time of learning (Oh et al. 2015; Zenke & Gerstner 2017); eligibility traces + neuromodulator
  third factor (Yagishita et al. 2014; Gerstner et al. 2018); dendritic prediction error (Urbanczik & Senn 2014).
- Turn timing: infant proto-conversation (Hilbrink et al. 2015); gaps of ~200 ms in adults (Stivers et al. 2009) make
  "a line that follows quickly is a reply" a cue the brain can use.
- Corollary discharge marks the own voice (Schneider et al. 2018) — here from the efference copy of the mouth.

### A13. The designer's idea "the PFC is, in essence, a recurrent network like a language model"
It is already the core of this design, with three refinements the toy results argue for:
1. **The loop goes through the PFC, not only through the input.**  SUP (layer 1-2: mixed selectivity) receives the
   previous syllable (as an assembly code), DC (the line state built from SUP, leaky) and DQ (the held previous line);
   DC is fed back into SUP every syllable — this is the "top state fed back into the network" of the coordinator's
   check (input-only top layer = bigram; looped top layer = context-dependent).
2. **The predictor is trained by listening to everyone**: at every syllable heard from another speaker the readout's
   prediction made at the previous syllable meets its target (R3, M_listen = 1).  Speaking = saying the prediction at
   the own boundaries.  The designer's "G(t-1) -> B(t) Hebbian" link is this readout, but with the delta form
   ("already predicted -> no more strengthening; predicted but absent -> weaken"), because Hebbian counts over
   distributed codes lose to an exact n-gram (coordinator: ~8 points) and to error-driven readouts (compute review).
3. **A recurrent next-token predictor alone cannot answer about the world**: the next syllable of a reply depends on
   the scene (what / where / colour / count), on a heard word that must be copied (offer), and on memory (name,
   before).  So the predictor's input includes the attended object and location, and its output is combined with copy
   sources through learned gates.  Measured below: next-syllable accuracy while listening vs online exact n-grams on
   the identical stream, and the bench's context-dependent tasks.

## B. Prototype results

### B.1 Rule unit tests (`test_rules.py` -> `test_rules.out`): 21/21 PASS (SIMULATED)
R1: one-shot A->B (w 1.00, reverse 0); reverse order never learns; ping-pong A,B,A,B leaves no two-way loop; same-tick
onsets never link; outside the window no link; gate M = 0 blocks; 好不好吃 -> successor(好) = 吃 (the back term erases
好->不); slow + presynaptic normalisation estimates P(next|a) (0.703 for 0.7).  R2: per-target totals <= budget at every
learning tick; 200 one-shot patterns at 2 % complete from a half cue (oldest 40/40, newest 40/40); 30 interleaved slow
concepts with 20 % noise complete 30/30; the template form converges (r 0.995, sum exactly B); M = 0 blocks.  R3: delta
converges; "already predicted -> no more strengthening" (max dw 0.0000); "predicted but absent -> weaken" (1.00 -> 0.50);
M = 0 blocks; signed weights learn an exception (A -> y 1.00, A+B -> y 0.00, w(B) = -0.33); a decaying trace bridges
5/10/20-tick delays with M on (prediction at the target tick 1.00) and learns nothing with M off.
### B.2 Module tests (`test_modules.py` -> `test_modules.out`): 7/7 PASS (SIMULATED)
- M1 assemblies / episodic recall (the agent's EC -> DG -> CA3 path, 300 one-shot episodes, 50 % cue): the right
  episode is retrieved for the oldest 20 and the newest 20 (1.00 / 1.00).
- M2 LEX segmentation on the bench's stage-A stream: within-word transition probability 0.72 (36 words) vs across-word
  0.012 (1,512 pairs); 16 of 19 chunks are lexicon words (什么 牛奶 鸭子 西瓜 汽车 皮球 香蕉 中间 葡萄 苹果 饼干 哪里 早上 好吃 右边 再见;
  junk: 是什 色的 么颜).
- M3 GIST hold: DQ (leaky 0.9 + MD gain 0.3) holds a loaded gist through 40 ticks with selection noise sd 0 and 0.3
  (Jaccard 1.00, J(t,t+1) = J(t,t+2) = 1.00: no period 2); at sd 1.0 it degrades (0.22) without alternating.
- M3' period-2 control: a synchronous two-layer loop with conflicting contents alternates (J(t,t+1) 0.02, J(t,t+2) 1.00);
  **leaky potentials alone (lam 0.9) still end in period 2 once the potentials reach the tie**; a within-layer
  self-hold (the 1:1 MD-type loop) removes it (1.00 / 1.00).
- M4 drift: measured inside the pipeline (Jaccard of the SUP, DQ and DG codes of two fixed probe lines at the end of
  each stage) — see B.6.

### B.3 How stage B moved with each design fix (seed 0; probes = free replies; SIMULATED, `diag*.out`, `pipe_smoke*`)
| version | change (each found by a failing diagnosis) | stage-B free-reply accuracy |
|---|---|---|
| v0 | first agent on the bench (own voice = CD timer) | 0.00-0.02 (500-800 dialogues) |
| v1 | own voice = efference copy; word onsets in copy pointers; whole-word evocation | 7/8 scripted probes right (diag4) |
| v2 | + END-after-copied-word signal, fast mapping, MD gain 0.3 | 0.52 over 500 dialogues (what 0.72, slot 0.72, where 0.00: recalled lines answered where-questions) |
| v3 | + recall counts only if it partially matches the question | prompts spoken as replies ("这是什么") |
| v4 | + turn role from timing, previous syllable as an assembly code in SUP | 0.60 in dialogues 250-500 (what 0.86 / held-out 0.74, where 0.54, slot 0.28) |
| v5 | + referent released at a new initiative | 0.56 in dialogues 250-500 (what 0.87 / held-out 0.84, where 0.43, slot 0.26) |
| **v6 (final)** | + lifelong vocal play as one-beat trials | whole run of 1,000: 0.33; learning in use: first third 0.31 -> last third 0.57 (B.4) |
| v7 (reverted) | + referent held through the reply, words bound only under independent attention | 0.18 over 1,000 (late third 0.40): fixes the location templates (右 0.43 -> 0.86) but removes binding chances; reverted |
| v6 + content gist | DC from words + attended object (B.8) | 0.36 in dialogues 250-500 (hold tests all pass); not adopted |
### B.4 Bench, stages A -> B -> C -> D (seed 0, final configuration v6; `runs/final_s0.json`, `.out`; SIMULATED)
Protocol: `pipeline.py 0 800 1000 1000 800`: stage A 800 dialogues (babbling on), then B, C, D 1,000 / 1,000 / 800
dialogues with tutor 'model' (other2 or the partner says the reply; the agent only listens; probes = 15 % of seen items
+ all held-out items, never followed by help).  Learning stays on throughout; scores are over ALL probes of the stage
(early learning included); "learning in use" = the bench's first vs last third of the stage.

| stage/task | n | acc | seen | held-out | first char | echo-free | parrot |
|---|---|---|---|---|---|---|---|
| B what | 238 | 0.51 | 0.64 | 0.40 | 0.81 | 0.51 | 0.34 |
| B where | 172 | 0.34 | 0.54 | 0.16 | 0.53 | 0.34 | 0.20 |
| B slot | 182 | 0.08 | 0.18 | 0.01 | 0.18 | 0.08 | 0.20 |
| **B ALL** | 592 | **0.33** | 0.47 | 0.21 | 0.53 | 0.33 | 0.25 |
| C offer | 161 | 0.85 | 0.84 | **0.86** | 0.88 | – | 0.92 |
| C taste | 70 | 0.54 | 0.74 | 0.36 | 0.69 | – | 0.79 |
| C count | 112 | 0.36 | 0.37 | 0.34 | 0.41 | 0.38 | 0.43 |
| C colour | 100 | 0.17 | 0.24 | 0.10 | 0.17 | 0.17 | 0.41 |
| C ritual | 61 | 0.43 | 0.43 | – | 0.49 | 0.55 | 0.50 |
| **C ALL** | 504 | **0.51** | 0.53 | **0.48** | 0.56 | 0.29 | 0.64 |
| D check | – | 0.48 | | | | | |
| D name | – | 0.02 | | | | | |
| D before | – | 0.03 | | | | | |
| D one-shot | – | 0.00 | | | | | |
| **D ALL** | – | **0.19** | | | | | |
Learning in use (no cue, no feedback, only listening): B 0.31 -> 0.57 (same items 0.26 -> 0.56); C 0.35 -> 0.70
(same items 0.39 -> 0.68).  Mechanics in every stage: 0 mouth ticks while a partner speaks, 0 overlapping exchanges,
latency median 6, 0 % syllables not 3 ticks, 0 garble, 0 extra turns.  Motor inventory of the final agent (`runs/motor_coverage.out`): 64/67 B/C chars (96 %) have a self-found program
(missing 呀 很 拜); but only 2/9 name chars and 6/67 novel-object chars — they are heard first in stage D, and the
512-slot program pool is full of duplicates (512 programs for 68 chars, 7.5 per char), so lifelong vocal play
stops adding new ones.  This alone caps D:name and D:one-shot.
### B.5 Does every learned part reach behaviour?  Lesions of the trained agent (`runs/final_s0.json`, SIMULATED)
Frozen copies of the final agent (after D), each with one part removed, answer the SAME stream of 150 B/C/D
dialogues (every exchange a probe).  Mean free-reply accuracy over the tasks of each stage (per-task n is 10-40, so
single-task numbers are noisy; the means are the reliable part):
| lesion | B | C | D | mean | reaches behaviour? |
|---|---|---|---|---|---|
| none (intact) | 0.24 | 0.46 | 0.27 | **0.32** | |
| readout W (premotor decision path) | 0.10 | 0.08 | 0.00 | 0.06 | yes, critical |
| ECHO phonological loop (store + its candidate features) | 0.16 | 0.01 | 0.06 | 0.08 | yes, critical (offer, count, ritual) |
| vision (attention, IT, word<->vision) | 0.01 | 0.09 | 0.12 | 0.07 | yes, critical |
| echo copy gate G_H | 0.25 | 0.20 | 0.25 | 0.23 | yes (offer 0.64 -> 0.08, ritual 0.39 -> 0) |
| word <-> vision templates VL | 0.14 | 0.32 | 0.25 | 0.24 | yes (where 0.26 -> 0.03, count 0.88 -> 0.47) |
| held line DQ (structure) | 0.16 | 0.31 | 0.25 | 0.24 | yes (offer 0.64 -> 0.16) |
| topic CTX (structure) | 0.18 | 0.37 | 0.25 | 0.27 | yes |
| LEX chunks (word units) | 0.21 | 0.36 | 0.29 | 0.29 | yes |
| SUP recurrent assemblies (REC) | 0.18 | 0.46 | 0.26 | 0.30 | small (B where 0.26 -> 0.16) |
| SUP as readout input | 0.16 | 0.49 | 0.28 | 0.31 | B only (C even improves) |
| visual copy gates G_Vo/Vc/Vn/Vl | 0.23 | 0.46 | 0.28 | 0.32 | **no — passenger** (the readout names from IT directly) |
| memory copy gates G_Vh/Vr | 0.26 | 0.42 | 0.28 | 0.32 | **no — passenger** (D memory fails anyway) |
| location words LA | 0.25 | 0.45 | 0.28 | 0.33 | **no — passenger** (slot questions fail anyway) |
| hippocampus (recall + store) | 0.25 | 0.41 | 0.28 | 0.32 | **no — passenger** |
| DC as readout input | 0.25 | 0.43 | 0.28 | 0.32 | no (DC acts only through DQ) |
Not lesioned because nothing works without them: SYL templates (perception), Mir (the dorsal mirror: without it no
syllable can be said).  So the brief's rule 3 is met for the dorsal route, the readout, vision/VL, LEX, the PFC
structure (DQ, CTX); it is NOT met for the D-stage memory machinery (hippocampus, Vh/Vr), LA and the visual copy
gates in this trained agent — by the rule they must be fixed or removed.

### B.6 Code drift with learning on (pipeline drift probe: two fixed prompt lines, frozen copies at each stage end)
Jaccard of the codes of the same line at successive stage ends (A->B, B->C, C->D): SUP 0.50/0.52, 0.55/0.60, 0.55/0.66;
DQ 0.35/0.28, 0.39/0.23, 0.28/0.28; DG (hippocampal index) 0.36/0.31, 0.38/0.31, 0.38/0.19.  The readout keeps working
(learning in use rises in every stage) because it is retrained continuously while listening, but anything keyed to
old codes (episodes, slow memories) goes stale — the reason the hippocampus fails here (see the hold test, 0/20 from
words alone) and a case for replay-based re-alignment in sleep (sleep() exists, not used in these runs).

### B.7 The recurrent-predictor comparison (coordinator's request)
Next-syllable prediction while listening (2nd half of each stage; the readout's argmax at every heard syllable vs an
ONLINE exact n-gram over the same within-line syllable stream, START-padded):
| stage | events | brainlike readout | 1-gram ctx | 2-gram ctx | 3-gram ctx |
|---|---|---|---|---|---|
| B | 6,089 | **0.82** | 0.70 | 0.71 | 0.71 |
| C | 6,513 | **0.82** | 0.54 | 0.56 | 0.59 |
| D | 4,597 | **0.77** | 0.53 | 0.59 | 0.61 |
The readout's context includes what the n-gram lacks: the scene (attended object), the previous line (DQ, ECHO) and
the topic — the "loop through the PFC" plus grounding.  On free replies the bench's n-gram baseline (privileged:
it sees the prompt text) gets B 0.16 / C 0.52 (held-out 0.31) / D 0.15 against 0.33 / 0.51 (0.48) / 0.19 here.
### B.8 Holding a thought by a loop (coordinator's checklist; `test_hold.py` -> `test_hold_default.out`, `test_hold_content.out`)
The hold is DQ: the BG gate copies the line gist DC into DQ at the end of a partner line; DQ is held by leaky units
(0.9/tick) + a fixed 1:1 MD loop (gain 0.3) and is NOT plastic; it is replaced at the next partner line end.  20
different partner lines in fixed scenes, presented to a trained, frozen agent with the bench's turn protocol (the
agent answers each line; the next line comes after >= 30 ticks and 8 silent ticks).
| measure (old PFC in brackets) | default gist, stage-A agent | default gist, FINAL agent (after D) | content gist (DC from words + attended object; words as assemblies), stage-A agent |
|---|---|---|---|
| generic-state index: Jaccard between held codes of different lines (old 0.47-0.79; random 0.026) | 0.098 | 0.090 | 0.092 |
| reliability: the same line presented again | 0.31 | 0.34 | 0.98 |
| 20-way decoding after 10 / 50 / 200 silent ticks, noise 0 | 0.45 / 0.45 / 0.45 | 0.50 / 0.50 / 0.50 | 1.00 / 1.00 / 1.00 |
| same with selection noise sd 0.3 | 0.45 / 0.45 / 0.45 | 0.50 / 0.50 / 0.50 | 1.00 / 1.00 / 1.00 |
| replacement by a new partner line (decoded as the new line) / residual overlap with the old | 0.75 / 0.11 | 0.50 / 0.08 | 1.00 / 0.08 |
| period-2 index J(t,t+1) - J(t,t+2) during the hold | 0.000 (no alternation) | 0.000 | 0.000 |
| capacity: DQ holds 1 line; CTX after line n closest to CTX after n-1 (chance 0.05) | 0.53 | 0.53 | 0.58 |
| hippocampus: each of 20 lines recalled from its words alone | 0/20 | 2/20 (of 3,000 episodes) | 0/20 |
| drift with learning on: same line before vs after 200 more dialogues | 0.20 | 0.23 | 0.79 |
How the four old failures are addressed: (1) generic state: the loop is 1:1 and fixed, so it cannot amplify a shared
component; what remains is the content of the gist (default: history-dependent; content gist: reliable);
(2) hold vs update: separated in time by the BG gate (load only at a partner line end, never during the hold);
(3) period 2: the hold loop maps each unit onto itself (M3': leaky units alone are NOT enough); (4) drift: the hold
loop has no plasticity, but the codes it is loaded with drift as SUP's recurrent assemblies learn (0.20 over 200
dialogues for the default gist) — the readout then has to track them.

**But the better hold did not give better behaviour.**  The content-gist agent (A10, trained from scratch with it) scored
0.36 on stage-B probes in dialogues 250-500 (what 0.53, where 0.31, slot 0.17; `runs/diag10.out`), against 0.56-0.60
for the default gist in the comparable checks (`diag6/diag7`).  The default gist carries, besides the line, the
held previous line, the topic and the recall — conjunctive context that the readout uses.  So the final
configuration keeps the default gist; the right design is probably both (a content-held DQ for the question AND a
mixed SUP state for the readout), untested.  Hippocampal recall from the words alone fails in both (0/20): episodes are
indexed by all EC streams equally, so a words-only cue is too weak — the D-stage memory failures start here.

### B.9 Against the official bar and baselines (bench README v1.2 "Reference numbers, 3 seeds", 2,000 dialogues per stage)
Bar = ARCH_v1 MEM trained with teacher forcing (in model / recast mode v1 never speaks: 0 programs).  Mine = seed 0,
1,000 dialogues per stage, tutor 'model' (no cue, no reward), whole stage incl. the early learning phase.
| task | **brainlike** | bar (v1, forced) | prior | n-gram | judgement |
|---|---|---|---|---|---|
| B what (pointing) | **0.51** (held-out 0.40) | 0.06 | 0.06 | 0.04 | real gain: scene naming learned by listening |
| B where | 0.34 (0.16) | 0.32 | 0.31 | 0.32 | at prior level |
| B slot | 0.08 (0.01) | 0.00 | 0.07 | 0.06 | at prior level (location word -> attention fails) |
| C offer (held-out fillers) | **0.85 (0.86)** | 0.04 | 0.07 | 0.48 (held-out ~0) | real gain: frame + ordered copy generalises to never-taught fillers |
| C count | 0.36 (0.34) | 0.10 | 0.31 | 0.30 | slightly above prior |
| C colour | 0.17 (0.10) | 0.08 | 0.16 | 0.17 | at prior level |
| C taste | 0.54 (0.36) | 0.37 | 1.00 | 1.00 | BELOW the prior ("好吃" always) |
| C ritual | 0.43 | 0.62 | 0.20 | 1.00 | below the n-gram |
| D check | 0.48 | 0.46 | 0.48 | 0.31 | at prior level |
| D name | 0.02 (first syllable 0.83) | 0.00 | 0.12 | 0.12 | below prior (second syllable has no program) |
| D before | 0.03 | 0.01 | 0.08 | 0.08 | fails |
| D one-shot | 0.00 | 0.00 | 0.00 | 0.00 | fails |
| **B / C / D ALL** | **0.33 / 0.51 / 0.19** | 0.12 / 0.19 / 0.19 | 0.14 / 0.30 / 0.22 | 0.14 / 0.52 / 0.15 | B and C above the bar; C ties the privileged n-gram overall but wins on held-out (0.48 vs 0.29); D at prior level |
The n-gram and prior are privileged baselines (they read the prompt text and the taught replies directly); mine
hears sound only.  Learning in use (seen items, first -> last third): brainlike B 0.31 -> 0.57, C 0.35 -> 0.70,
D 0.05 -> 0.19; bar B 0.13 -> 0.11, C 0.34 -> 0.23, D 0.14 -> 0.13.  Mechanics are as exact as the bar's.
(My earlier own baseline runs on the same copy, 2,000 dialogues seed 0: B prior 0.15 / n-gram 0.16; C 0.32 / 0.52 held
0.31; D 0.21 / 0.15 — consistent.)

### B.10 Multi-seed run: not available
The 3-seed (+ recast, + lawful-physics) job `gpu/queue/brainlike_main1` ran 40 minutes and timed out with empty logs:
it was submitted with 5 CPU processes before the per-job limit of 4 and the shared 12-core quota (each core ~0.6x a
local core) were announced, so not even stage A + B finished.  All bench numbers above are therefore ONE seed (world
seed 0).  Variance across design versions on that seed is visible in B.3 (stage B 0.33-0.60 across v4-v6 variants);
the seed variance is unmeasured.  Recast mode and lawful physics are implemented (`tutor=recast`,
`BL_PHYSICS=lawful`) but not measured.

## Bench issues (for the coordinator; I did not change any metric)
1. **Eye codes have no location structure.**  Each (slot, value) code is 4 shared + 4 slot-specific units at random
   positions in 1,024, and pointing is 8 random units per slot.  A brain-like attention mechanism (priority map ->
   gain-gated pooling of the attended location) needs an innate retinotopic layout; with the random layout the
   agent cannot know which units belong to which slot, so held-out (task, filler) items that need "the object at
   slot s" can only be solved by per-(slot, value) conjunctions learned task by task.  My copy (benchcopy/world.py)
   adds one option, `eye_layout='retino'`: slot s owns units [240 s, 240 s + 240), the same feature codes are replicated
   in each block (shape 0-159, colour 160-207, number 208-231, pointing 232-239).  Same scenes, held-out sets, metrics.
   (Adopted by the bench in v1.2 as `eye_layout='retino'`; my runs used my own copy of the option.)
2. **Babbling cannot reach the vocabulary with random physics and the 3,970-char Voice.**  A random 4-muscle set
   produces a given char with p ~ 3e-5 (it must share 3-4 muscles and be the first such char in the decoder's order),
   i.e. ~3x10^4 babbles per target char (SIMULATED: only 11 syllables got programs from 165 babbles in my first run).
   I used the bench's `vocab=` option with the bench's own 142 chars (then ~1,100 babbles cover 93 % of the B/C chars).
   Lawful physics (now "checked" in v0.3) is the proper fix; my agent has not been run on it yet.
3. **Trap, not a bug:** v1's corollary-discharge timer runs for 4 ticks after every GO, even when the first boundary is
   then blocked by the model reply (gap_model = 3).  Agents must not use CD as their own-voice tag.
4. In model mode 30 % of the model replies are spoken by the partner itself; speaker identity therefore cannot mark
   "reply" vs "question" — timing can (the model reply starts 3 ticks after the prompt, the next prompt >= 10 later).
5. Process note: my multi-seed V100 job (`gpu/queue/brainlike_main1`) was submitted with 5 CPU processes before the
   coordinator's limit of 4 per job was announced; it timed out without output (B.10).  My final runs used my bench
   copy taken from v0.3 at 02:59 (after the 02:56 redesign of D:before; that task's code is identical to v1.2's).

## Files (all in arch/v2/brainlike/)
| file | what |
|---|---|
| `agent.py` | the brainlike mind: SYL, ECHO, S_f/S_b, LEX, vision (attention, IT, VL, LA, comparator), PFC (SUP, DC, DQ+MD, CTX), hippocampus (EC, DG, CA3, episodes, sleep), readout + copy gates, babbling + mirror + vocal play; a config flag for every ablation (and the tested variants: dc_src, word_assembly, sp_beta) |
| `rules.py` | the three rules R1 (order), R2 (association, exact immediate normalisation), R3 (decision, NLMS delta) |
| `body.py` | ARCH_v1 body (F/GO/CD, beat, WTA, END/breath/floor, D/I/M) + idle-time GO request + trial beat |
| `front.py`, `ear_np_copy.py`, `voice_copy.py` | exact ear layer 0 / FEAT0 replica (copied) |
| `test_rules.py` -> `test_rules.out` | 21 rule unit tests (all pass) |
| `test_modules.py` -> `test_modules.out` | module tests: hippocampal one-shot recall, LEX segmentation, DQ hold, period-2 control (7/7) |
| `test_hold.py` -> `test_hold_default.out`, `test_hold_content.out` | the coordinator's loop-hold checklist on trained agents |
| `bench_run.py` | bench wrapper (vocab, eye layout, physics options via BL_PHYSICS / BL_VOCAB) |
| `pipeline.py` | A -> B -> C -> D on the bench + listening-prediction vs n-gram + drift probe + lesions (frozen copies, fixed probe stream); saves the trained agent |
| `final_tables.py`, `agg.py` | tables from pipeline JSONs |
| `stageA.py`, `cont_run.py`, `diag.py` | stage-A runs, continuation, instrumented diagnosis (per-boundary gates, candidates, predictions) |
| `runs/` | all run outputs: `final_s0.{out,json}` + `final_s0_agent.pkl` (the final run), `final_v7_s0.*` (reverted variant), `diag4-10.out`, `A*.out`, smoke/crash JSONs; GPU multi-seed results in `../gpu/queue/brainlike_main1/result/` |
| `benchcopy/` | the bench v0.3 + one option `eye_layout='retino'` (`benchcopy_v02/` = the earlier copy) |
| `world.py`, `run.py` | my early toy world (before the bench was ready) |
| `design.md`, `cites.md`, `tests_section.md`, `progression.md`, `results_head.md`, `lesions_section.md`, `hold_section.md`, `bars_section.md`, `bench_issues.md`, `verdict.md`, `zh.md`, `assemble.sh` | the parts of `report.md` |

## summary_zh
1. 按"真实大脑的分工"设计并实现了完整架构：背侧"听→复述"（音节单元→语音回路→复制闸门→发音镜像）、腹侧"听→懂"（音节统计→词单元→词与看见的物体、位置的联结）、多层前额叶（浅层混合选择性；深层保持当前句和上一句；更慢的话题层；丘脑式放大回路；基底节闸门：对方说话时更新、回答时保持）、海马体一次记住情节。学习规则只有三类：设计者的先后顺序规则（序列区）、一起亮就加强并当场归一（联想区）、资格痕迹×第三因子（决策通路）。
2. 全程不推肌肉：发音程序全靠咿呀学语自己找到（最终覆盖 B/C 阶段 96% 的音节）；说什么全靠"听别人怎么回答"学会。别人说话时嘴从不动、第 6 拍开口、每个字 3 拍，所有阶段都保持。
3. 共享测试台结果（种子 0，对照官方基准：v1 靠推肌肉训练，B/C/D 为 0.12/0.19/0.19）：真正新出现的能力有两项——指着东西问"这是什么"答对 0.51（v1、先验、n-gram 都≤0.06），"你要X吗→要X"0.85、没教过的 X 上 0.86（n-gram 0.48，没教过的约 0）；三个阶段总分 0.33/0.51/0.19。但"在哪里""左边是什么""什么颜色"只到先验水平，"好不好吃"和寒暄低于先验/n-gram，D 阶段（是不是、记名字、刚才、一次学新词）没有学会。
4. 设计者"前额叶就像语言模型里的循环网络"的想法正是这个架构的核心：边听边预测下一个音节，0.82/0.82/0.77，比同样数据上的 n-gram（0.71/0.59/0.61）好。
5. 用回路"保持想法"：机制本身没问题（没有两拍来回跳、200 拍后还在、新句子能干净替换）；按内容构成的版本保持测试全过（20 选 1 解码 1.00），但行为没变好，所以最终版保留了带上下文的版本（解码 0.45）。
6. 逐个摘除检验：读出层、复述通路、视觉、词-物联结、词单元、前额叶的保持层和话题层都确实影响行为；海马体、记忆复制闸门、视觉复制闸门、位置词联结在训练好的模型里是"乘客"，按规则需要修好或去掉。
7. 迭代中最重要的发现：把 v1 的"自言自语计时器"当成"自己的声音"会删掉每个示范回答的第二个字（这是最初完全学不会的原因）；只靠漏电神经元挡不住两拍来回跳，要靠层内自保持回路。
8. 还没解决：问"左边是什么"和颜色题较弱；D 阶段的记忆；后来才听到的字没有发音程序（程序池被重复程序占满）；只有一个种子的结果（三种子的服务器任务超时，没有输出）；"改说+奖励"模式和有规律的发音物理已实现但没测。
