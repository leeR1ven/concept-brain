# ARCH_v2 — the synthesized architecture (synthesizer's report)

Folder: `v2/synth/` (key outputs in `results/`; the runs wrote to `out/`, not kept). Labels: **PROVED** (argument with numbers), **SIMULATED** (numpy run; file and numbers),
**CITED** (paper, finding), **ASSUMED**. Everything ran locally (≤ 3 processes, `OMP_NUM_THREADS=1`) on the shared bench
v1.2 + life v2.0, copied unchanged into `synth/bench_copy/`.

## 0. Verdict

**ARCH_v2 = tune's recommended configuration** (develop's developmental core: perceptual syllables, babbling inverse map,
recruited conjunctions, signed error-driven readouts, BG copy gates, comparator attention, episodic hippocampus; interleaved
curriculum; awake recall from a 10k-cell store) **+ think's speech intention** (confidence accumulator, learned brake and
eagerness, BG gate into the body's GO, the innate reflexes) **+ a continuous thought** in place of the previous-line
register (it wanders toward favourite topics in long silences) **+ innate retinotopic, slot-relative attention +
new-word programs + recency in recall + a 7th reflex** for self-initiated speech, **− the fixed slow context CTX**
(removed after the lesion study showed it only hurt recall). All nine binding principles hold (§9); no sleep, no reset, local rules only, 0 mechanics errors.

| SIMULATED | tune's best (start) | **ARCH_v2** |
|---|---|---|
| staged J, search seeds 0–2 (SD) | 0.73 (0.05) | **0.83 (0.05)** — better on every seed (+0.08 … +0.11) |
| staged J, held-out world seeds 3–4 | 0.74 (0.01) | **0.82 (0.06)** (+0.13, +0.03) |
| colour / count questions (seeds 0–2) | 0.15 / 0.45 | **0.93 / 0.76** |
| name recall (seeds 0–2 / 3–4) | 0.44 / 0.48 | **0.76 / 0.79** |
| C level after 300 untutored dialogues (drift) | 0.38 | 0.53–0.56 (the loss itself is not smaller) |
| life 10⁶ ticks: free replies / scene questions in new situations | 0.20 / 0.25 | **0.34 / 0.33** |
| life: held-out / social replies / items heard once | 0.07 / 0.06 / 0.30 | **0.10 / 0.32 / 0.36** |
| life: accuracy in the last tenth of life | 0.25 | **0.47** |
| life: self-initiated utterances (appropriate; greeted first) | none | 1,891 (0.19; 347 rewarded greetings) |
| life: favourite topics held after long silences (base rate 0.14) | — | 0.81 |
| mechanics in every run: overlap, non-3-tick syllables, garble, extra turns | 0 | **0** |
| costs | | check 0.73 → 0.63; ritual / taste lower on the held-out seeds (0.82 → 0.73, 0.71 → 0.63) |

**One at a time (§7):** kept — the intention (J −0.03, required by principle 4; in life: initiative, greeting first,
social +0.09), new-word programs (fixes an unbound syllable), the continuous thought (principle 2; equivalent to the
register at the bench's timescale, PROVED), retinotopic attention (J +0.08: colour 0.11 → 0.82, count 0.46 → 0.81),
recency in recall (J +0.05), reflex 7 (0 overlaps by construction), wandering in long silences (favourite topics
0.81 vs a base rate of 0.14; J identical by construction), removing CTX (name 0.32 → 0.85, J unchanged). Rejected — learned local receptive fields in the ear
(no gain; a deeper layer loses information), the line-surprise turn feature (no gain), a residue-carrying thought (−0.05),
surprise-gated entry into the thought (B 0.69 → 0.15: the predicted harm), the naming target (J and life neutral).
Not added — a contingency factor against drift (no timing cue exists in this bench; designer: not yet); longer
eligibility traces (not needed: the thought holds the question until the recast).

**Still open (§12):** free-use drift is reduced, not gone; thinking does not yet improve content (the thought's
operations are fixed); unprompted speech is mostly the partner's line openings; one-shot naming, "before" and life
memory facts (across visits) stay ≈ 0; held-out generalisation ≈ 0.2; babbling still reads world.voice offline; the
learned eagerness V_go and the reward actor barely reach behaviour (V_go is removable).

## 1. Block diagram

```
 obs['ear'] (8x6 window) ──► EAR-0 (fixed 3D layer 0: 1,920 units, fan-in 4; FEAT0 = 1,153)   obs['eye'] retinotopic:
        │ energy E                 │ FEAT0                                              3 slot blocks × 240, same code│
        ▼                          ▼                                                                    ▼
 ┌─ BODY (ARCH_v1, innate) ─┐   syllable clock (phase reset at speech onset, sample onset+2+3k)   scene baseline +
 │ F/F2 "other is talking"  │   Z  perceptual syllables (≤256, vigilance 0.75)  ◄─ own voice      pointing → attended
 │ GO ◄── REL (BG gate)     │      │ own-voice tag = mouth active ≤ 2 ticks ago (efference copy)  block (innate); ATT
 │ CD, R1, R2, 3-tick beat  │      │ babbling memory (3,970 own sounds) → program binding,        (3 slot units)
 │ program WTA, END rule,   │      │   re-tried at every hearing of an unbound syllable              │
 │ breath counter, floor,   │      ▼                                                                   │
 │ D/I/M force ramp ──► 32  │   heard items ──────────────┬───────────────────────────┐               │
 │ muscles                  │      │                      │                           │               │
 └──────────▲───────────────┘      ▼                      ▼                           ▼               ▼
            │              current-line buffer     PFC-1 THOUGHT (continuous)    HIPPOCAMPUS       PRIME (word →
            │              (items + order of the   item assemblies a_i∈[0,1];    DG 6,000 k120     expected eye
            │              line being heard/said)  a line from another speaker   → 10,000 one-shot units), MATCH
            │                      │               flows in at its end (old      cells; recall =   comparator, VR in
            │                      │               content ×0.3: competition);   overlap·σ²;       the slot-relative
            │                      │               own speech does not enter;    awake recall: 8   frame (one map for
            │                      │               P = items ≥ 0.5, order = the  episodes per      every slot), attD
            │                      │               line that entered last        quiet moment       │
            │                      ▼                      ▼                           ▼               ▼
            │              PFC-2 MIX: recruited conjunctions of (P, P1, PL, L, N, FAM, A, F, VR, M, copy candidates)
            │              (PFC-4 slow context CTX: removed — it hurt name recall, §7 #16)
            │                      │
            │                      ▼
            │              GEN readout (MIX → 256 items + END; signed, error-driven)   CRITIC / actor (RPE of the
            │              PFC-3 BG copy gates (prompt / recalled next line / recall     partner's reaction)
            │              completion) × the held candidates
            │                      │ scores of every program at each beat boundary
            │                      ▼
            │              THOUGHT STEPS, one per tick while silent: k1 recall → k2 comparator/attention → k3 VR
            │              confidence e = margin of the best first item (read-only)
            │              SPEECH INTENTION  I ← I + G·e + u,  G = g0 + β(V_go − r̄),  u = 0.15 (reply thoughts)
            │              release REL when I ≥ 1 and P_oth < 0.25 (replies: inside the transition window τ ≤ 7;
            │              self-initiated: only through the learned brake, prior 'floor taken', and after τ ≥ 10)
            └───────────── P_oth "another speaker starts within 10 ticks" (learned brake, time cells τ ≤ 30)
                           V_go  "speaking here is answered with a reward" (learned eagerness)
```

## 2. Every area: size, activity, rule, parameters (prototype; GPU instance in §10)
| area | role | prototype size | activity | learns by (rule, rate) | persistence |
|---|---|---|---|---|---|
| BODY (ARCH_v1) | turn units, beat, program WTA, END, breath, D/I/M force | v1; ≤ 320 programs | — | nothing (innate) | — |
| EAR-0 / FEAT0 | fixed ear layer 0 | 1,920 units, FEAT0 1,153 | top 40 % | — (fixed) | 1 tick |
| syllable clock | one sample per 3 ticks from speech onset + 2 | 1 counter | — | — | until silence |
| Z | perceptual syllables (all voices) | ≤ 256 categories (67 after stage A) | one per sample | R5: vigilance 0.75, template rate 0.1 | — |
| babbling memory | own sound → muscle set | 3,970 sounds (one pass over 35,960 sets) | — | one-shot; binding ≥ 0.85; re-try at hearing | life-long |
| current-line buffer | items + order of the line being heard or said | ≤ 24 items | — | R1 one-shot | the line |
| PFC-1 thought | continuous held content (no register) | 256 item assemblies + 256 × 256 co-holding links | ≤ 12 items ≥ θ 0.5 | entry dynamics innate (line-end gain 1, competition ×0.3, own gain 0); links R11 (wandering in long silences) | until displaced; wanders in long silences |
| PFC-2 MIX | recruited conjunctions (mixed selectivity) | ≈ 48k units after 3,300 dialogues with the retinotopic eye (115k with the flat eye; cap 600k) | ≈ 100–400 per event | R3 recruitment on first co-occurrence | per event |
| GEN | readout MIX → 256 items + END (premotor/cortex) | MIX × 257 | drive | R2, η 0.35 (0.2 in awake recall) | weights |
| PFC-3 gates | BG output gates COPY_prompt / COPY_next / COPY_comp | MIX × 3 | clip(drive) | R2, η 0.5 | per event |
| ATT | 3 attention units | 3 | WTA | pointing (innate) or internal: attD + κ·MATCH > 0.5 | one line |
| attD | word → slot (左边 → left) | 257 × 3 | — | R7 outstar, taught by pointing or a confident comparator | weights |
| PRIME | word → expected eye units, slot-relative | 257 × 240 (one map for all 3 slots; flat-eye variant 257 × 1,024) | — | R7 outstar under pointing only | weights |
| MATCH / VR | comparator per slot; words recognised at the attended slot | computed | — | — | per line |
| HIP | DG pattern separation + one-shot episode cells | DG 6,000 (k 120); 10,000 cells | 2 % (DG) | R4; σ × 0.9995 per stored line; recall overlap·σ² | ≈ 1,400 lines half-life |
| awake recall | quiet-moment rehearsal | 8 episodes per quiet moment | — | R10 (R2 at η 0.2) | — |
| ~~PFC-4 CTX~~ | slow dialogue context → hippocampal key | removed (§7 #16) | — | — | — |
| CRITIC / actor | value of the reply context; RPE | per MIX unit | — | R6, η 0.2 / 0.3 | weights |
| THOUGHT STEPS | recall → comparator → recognition, one per tick | 3 steps | — | — | until the thought closes |
| INTENTION | accumulator I, gain G, urgency u, release | 1 accumulator | — | — (innate dynamics) | one thought |
| P_oth | learned brake: another speaker starts within 10 ticks | ≈ 10⁴–10⁵ keys (features × τ ≤ 30) | — | R8, η 0.2 | weights |
| V_go | learned eagerness: speaking here is rewarded | ≈ 10³–10⁴ keys | — | R8, η 0.2; eligibility ≤ 50 ticks | weights |

## 3. The learning rules (all local; exact)
A synapse sees only its own pre- and postsynaptic activity, its own weight and trace, and at most one broadcast signal.

| rule | where | exact form |
|---|---|---|
| R1 one-shot order | current-line buffer, thought order links, hippocampal content | when item b is detected right after item a in one line: link a→b (and BOS→first); successor query with inhibition of return on positions already used in the reply |
| R2 signed error-driven readout (prediction-error form of the designer's earlier→later rule) | GEN (MIX → items + END), BG copy gates | for the active source set A at the prediction event and target unit j: d_j = Σ_{i∈A} w_ij + (copy routes into j); p_j = clip(d_j, 0, 1); Δw_ij = η·g·(t_j − p_j)/|A| (η 0.35 GEN, 0.5 gates, 0.2 awake recall); t_j = 1 if j's teaching input fired (the heard item's mirror; END at the heard line's end); "predicted → no more strengthening", "predicted but absent → weaken"; the negative part runs through local inhibitory interneurons (Dale) |
| R3 recruitment | MIX | a conjunction unit is created on the first co-occurrence of two context features (pair schema of develop), deleted only if pruning is on (off) |
| R4 one-shot episodic index | hippocampus | key = DG k-WTA (6,000 units, k 120) of (line bag, first, last, scene) — CTX removed; one cell per heard line (the weakest is overwritten when full); σ ← 0.9995·σ per stored line; recall = argmax over cells of overlap·σ^ρ (ρ 2; develop 0.5), overlap ≥ 0.2 |
| R5 perceptual categories | Z | m = |f ∧ T_j|/|T_j|; m ≥ 0.75 → category j and T_j ← 0.9 T_j + 0.1 f; else a new category (vigilance) |
| R6 critic / actor (RPE) | CRITIC, GEN | δ = r − V(first-step context); V += 0.2 δ/|A0|; every own choice (A, y) of that utterance: w_iy += 0.3 δ/|A| |
| R7 outstar | PRIME (word→eye), attD (word→slot) | w_i ← w_i + m_i (t − w_i), m_i = max(1/n_i, 0.02): the conditional frequency, plasticity falling with use; taught only by pointing (PRIME) or pointing / a confident comparator match (attD) |
| R8 PR turn learners | P_oth, V_go | p = clip(b + Σ_k w_k, 0, 1) over binary keys; Δw_k = 0.2 (y − p)/n_keys; P_oth: y = another speaker's line starts within 10 ticks (keys = thought features × time cell τ ≤ 30; prior b = 0 inside the transition window τ ≤ 7 for reply thoughts, else 1); V_go: y = [partner's reaction carries a reward], eligibility held ≤ 50 ticks |
| R9 babbling binding | babbling memory → programs | every producible own sound is stored once (one-shot); a heard category is bound to the babbled set whose own sound matches best, if ≥ 0.85; an unbound category is re-tried at each hearing |
| R10 awake recall (no sleep) | hippocampus → GEN, gates | at a quiet moment (12 silent ticks, not speaking, not in a heard line) 8 stored episodes are drawn uniformly; each is reinstated as the previous line and its successor line is learned as if heard again (R2 at η 0.2); the ongoing thought and context are restored afterwards |
| R11 co-holding links (favourite thoughts) | PFC-1 thought | when a line enters, every pair of items held together: W_ij += 0.02, then each row is renormalised to Σ_j W_ij ≤ 1 (co-activity with immediate normalisation); used only in silences > 200 ticks: every 50 ticks a_i ← a_i (1 − 0.1 f_i), f_i ← f_i + (a_i − f_i)/20 (fatigue), a ← max(a, 0.5 · aW), a ← a / max(a), ≤ 12 items above θ |

## 4. Third factors (broadcast signals)
| signal | source | gates |
|---|---|---|
| M_listen ("heard from another voice") | own-voice tag = mouth active ≤ 2 ticks ago (efference copy) | every listening update (R2 targets), category learning; own speech never teaches the readout |
| reward r → RPE δ = r − V | the partner's reaction (obs reward: recast of an acceptable reply, life greetings/namings) | actor update of the own utterance's choices (R6); V_go target (R8) |
| joint attention | the tutor's pointing (innate orienting) | PRIME and attD learning (R7) |
| comparator confidence | match ≥ 0.5 with a lead ≥ 0.3 | attD learning without pointing; internal attention |
| "another speaker starts" (auditory onset) | body F / line onset | P_oth target (R8), closes the current thought |
| quiet moment | 12 silent ticks | awake recall (R10) |
| novelty (vigilance) | no category match ≥ 0.75 | recruitment of a new category; binding attempt from the babbling memory |

## 5. Innate (fixed) mechanisms, each with its justification and how it could later be learned
| # | mechanism | justification | plan to learn / replace |
|---|---|---|---|
| 1 | body turn units F/F2 ("other is talking", window energy ≥ 10), GO gate, CD, R1/R2, 3-tick beat, program WTA, END-vs-best rule, breath counter, floor 0.3, D/I/M force ramp (ARCH_v1) | the verified 0-error mechanics (principle 8); brainstem/BG-like reflex scaffold (all five reviews) | learned gates may tune or override it later (P_oth already modulates GO) |
| 2 | GO is released by the speech intention (BG gate), not by the v1 timer | principle 4: speech is caused by thinking (think) | — (it is the learned path) |
| 3 | first-syllable now-brake: the first boundary also needs 0.01·E(t) < 0.1 | closes a 2-tick blind window of the GO→R1→boundary pipeline that only a thought-caused GO meets (think run 1: overlaps); a speaker yields when the other starts (Schegloff 2000) | learn its threshold |
| 4 | own voice = mouth active ≤ 2 ticks ago (efference copy), not the GO-driven CD timer | a refused GO still runs CD and deafened the agent (think: B 0.56 → 0.45; brainlike, predictive: the same trap) | none needed (it is the efference copy) |
| 5 | transition window: a reply may start ≤ 7 ticks after another speaker's line | the staged world waits only if the agent starts inside that wait; late starts overlapped (think: 3 and 17 overlap exchanges without it); transition-relevance places (Sacks et al. 1974), ~200 ms gaps (Stivers et al. 2009) | P_oth learns it in reactive worlds; the hard limit can then be relaxed |
| 6 | one utterance per thought; a thought opens at another speaker's line end (reply) or at a scene/pointing onset in silence (scene), closes at the own utterance's end, another speaker's onset, a scene change or after 40 ticks | one selected action per decision (BG); no monologues (0 extra turns) | learned hold/close of the thought (assemblies) |
| 7 | self-initiated speech only through the learned brake with prior "floor taken" (b = 1 outside the window) | initiative is safe only where the partner waits (think: life 0 overlap; staged 1 exchange) | — (the brake is learned; only its prior is innate) |
| 7b | reflex 7: a self-initiated (scene) thought may release only after it has been open ≥ 10 ticks | a scene thought closes when another speaker starts, so in the staged world (partner speaks ≤ 5 ticks after a scene appears or changes) unprompted speech can never overlap (PROVED from world.py); one overlap of think's last design came from this path | learn it as the minimum lapse before self-selection |
| 8 | accumulator: θ_I 1, G = clip(g0 + 2(V_go − r̄), 0, 3), g0 1 (reply) / 0.3 staged, 1.0 life (scene), urgency 0.15 (reply) / 0 staged, 0.02 life (scene) | evidence accumulation with urgency (Cisek et al. 2009; Ratcliff & McKoon 2008); latency becomes a confidence signal | learn the urgency from how long partners wait |
| 9 | time cells τ ≤ 30 as P_oth inputs | interval timing in striatum/hippocampus (MacDonald et al. 2011; Mello et al. 2015) | — |
| 10 | thought steps in a fixed order, one per tick: recall → comparator/attention → visual recognition | develop's operations, now charged one tick each (think) | a learned BG controller of thought operations (open problem 2) |
| 11 | EAR-0 + FEAT0 (fixed 3D layer 0) | the bench's exact front end; learned local receptive fields on top did not help in this world (§7.11) | learned local layers for real audio |
| 12 | syllable clock: phase reset at speech onset, one sample per 3 ticks | theta entrainment to syllables (Giraud & Poeppel 2012) | a learned peak detector |
| 13 | category vigilance 0.75 (new category when nothing matches) | ART-like recruitment; 67/67 categories, purity 1.00 | — |
| 14 | line segmentation by silence | the bench has no pauses inside a line | learned turn-end prediction |
| 15 | babbling: one offline pass over all 35,960 four-muscle sets; best-match binding ≥ 0.85; re-tried at each hearing of an unbound syllable | builds the inverse map without puppeting; **inherited shortcut: it reads world.voice offline** (think's issue 1) | bench `body.babble` with the partner's imitation (`keep='reward'`) in life; lawful physics + a forward/inverse model (DIVA) |
| 16 | retinotopic eye (3 blocks × 240, the same code in every block; bench `eye_layout='retino'`); the attended slot = the block that holds the pointing units; appearance learned in the slot-relative frame | V1–V4 retinotopy, attention as gain on a location (CITED: Reynolds & Heeger 2009); joint attention from ~9–12 months | learned saccade/attention control (priority map) |
| 17 | comparator constants (match_thr 0.6 after ≥ 6 events, conf_min 6, rate_max 0.4, κ 2, match_min 0.5, gap 0.3, att_thr 0.5, rec_thr 0.85) | develop's iteration (§10 there); tune: the defaults sit on a plateau with cliffs on both sides | learned precision weighting |
| 18 | MIX pair schema (which feature groups are conjoined) | additive readouts cannot compute "heard X ↔ seen Y" (develop, PROVED) | recruit conjunctions where errors persist |
| 19 | continuous thought dynamics: another speaker's line enters at its end with gain 1, held content ×0.3 (competition), θ 0.5, ≤ 12 items, own speech gain 0; wandering constants (after 200 silent ticks, every 50: fatigue 0.1 / τ 20, spread 0.5, renormalisation) | principle 2; line-end integration = event-boundary encoding; the readout learns only from the settled state (predictive's map: learning at line onset collapses the loop) | learned input gating (PBWM) |
| 20 | ~~CTX~~ (fixed random slow context, k 24 of 800, λ 0.9 per line): **removed** — lesion and training-time removal show it only hurt name recall (seeds 0–1: 0.32 → 0.85) | — | a learned topic layer (assemblies) when a task needs it |
| 21 | DG expansion (6,000, k 120), recall WTA with overlap·σ², familiarity bins, NEXT / COMPLETION read-outs | index theory (Teyler & DiScenna 1986); one-shot episodes (develop R4) | learned CA3/CA1 |
| 22 | awake recall schedule: a quiet moment = 12 silent ticks; 8 episodes, uniform over the store, η 0.2 | principle 3 (no sleep); tune: uniform beats strength- or recency-weighted (0.46 vs 0.62) | prioritised by prediction error |
| 23 | life only: eye bits that appear in silence join the scene baseline | otherwise the partner's person code and new objects read as pointing (think, tune) | learned figure/ground |

## 6. What changed from the designer's original framework, and why
| framework idea (research/FRAMEWORK.md) | in ARCH_v2 | why (evidence) |
|---|---|---|
| binary neurons; fixed active fraction per area | kept for the ear (top-40 %), DG (k 120), the thought (≤ 12 items); event-driven readouts instead of fixed top-k elsewhere | the readouts are linear in sparse conjunctions; fixed top-k there forced generic states (review A.4) |
| "earlier → later strengthens; later → earlier weakens" (#9, 2026-10-30) | kept as R1 (one-shot order) and in its prediction-error form R2: predicted → no more strengthening, predicted but absent → weaken | counts over distributed codes lose 8 points to error-driven readouts (coordinator); signed R2 gives retention 0.99 vs 0.48 excitatory-only (develop unit test) |
| "don't strengthen co-active neurons" (#9) | relaxed only in the outstar expectations (word ↔ seen object) and in the thought's co-holding links (R11), both normalised immediately | binding a word to the object seen with it needs co-activity (develop R7); favourite thoughts need co-holding links (§7.9) |
| one global weight decay as the clean-up (#3) | not used: every decay value tested was harmful (tune: −0.28 at 0.3, −0.40 at 2 per 1,000 events) | — |
| hippocampus = fast chain region, short-term (#2, #5) | fast one-shot, LONG-lasting episodic index (10,000 cells, σ half-life ≈ 1,400 lines) that teaches the cortex by awake recall | clinical evidence (H.M.); tune: recall + a large store is the only lever against interference and drift (J 0.61 → 0.73) |
| sleep (review decision 4) | NO sleep (designer 2026-10-31): consolidation = awake recall in quiet moments | sleep replay never helped (develop: 0.98 → 0.62 one-shot retention); awake recall doubled once-heard items in life (0.14 → 0.30) |
| PFC: multi-layer, outputs return as inputs (#8, #24) | PFC-1 continuous thought (item assemblies, no register) + PFC-2 conjunctions read at every boundary + PFC-3 BG output gates; the fixed slow context PFC-4 was removed (it hurt memory recall) | the old loop alternated (period 2) or collapsed into one generic state (predictive M6; develop M5); the continuous thought keeps content without either (§7.3) |
| "PFC = recurrent next-input predictor" (IDEAS #1) | the readout predicts every heard item from (thought × current input × scene × recall); speaking = saying the prediction | listening prediction beats exact n-grams (brainlike 0.82 vs 0.70); next-char on the corpus matches n-gram counting (develop 0.228 vs 0.234) |
| hold a thought by a loop (IDEAS #2) | item assemblies hold themselves until new input displaces them; no clear or load at a new line (principle 2) | predictive's map: collapse came from learning at line onset and spare active units, not from the loop |
| favourite thoughts are a feature (IDEAS #7) | in long silences the thought wanders along learned co-holding links toward the most familiar topics | after silences ≥ 2,000 ticks the top-20 frequent items make up 0.82 of the held items (base rate 0.14; 0.24 without wandering) |
| "speech from a general intention; other areas say it" (#21, IDEAS #6) | speech intention (confidence accumulator) + BG gate + learned brake/eagerness; the body executes the syllables | think: honest timing, turn-taking, learned greeting-first; no content gain |
| motor = inhibition vs disinhibition of muscles only (#4) | kept (the ARCH_v1 body) | 0-error mechanics |
| random muscle moves + listening make the connections (#12) | babbling builds the inverse map; every own sound is remembered once; a heard syllable binds the matching babble (no puppeting) | 67/67 programs, 0 wrong (develop); re-trying the binding at every hearing fixes the one category that missed its first match (§7, #2) |
| split / delete of neurons (#5, #6) | recruitment on novelty (categories, conjunction units, episodes); maturation and split tested and switched off | no net gain (tune: split ≈ base, maturation −0.10 to −0.18) |
| "no speech area" (#22) | kept: all routes drive the same program units | — |
| teacher drives the muscles | removed: learning by listening + contingent recasts (reward) | v1 never speaks without the cue (bench) |
| every input is equally strong (2026-10-30) | not addressed | open problem |

## 7. The additions, one at a time (kept or rejected, with numbers)

Method. tune's recommended configuration was reproduced first: my copy gives seeds 0 and 1 bit for bit (J 0.68 / 0.80;
every stage's probe scores and the drift identical to tune's files). Each candidate was then added to the current best
configuration and run on ≥ 2 seeds of the staged J protocol (A 300 → babbling → B 1,000 → BC 1,000 → BCDD 1,000; frozen
B/C/D probes after each stage; drift = C after 300 untutored C dialogues; J = F + 0.5·RET + 0.25·H + 0.25·DR, any
mechanics violation → J = −1). Comparisons are paired by seed (same world, same voices); the seed SD of J is ≈ 0.06, so a
2-seed mean difference below ≈ 0.06 is noise. "Required" = required by a binding principle: kept unless clearly harmful.
All numbers SIMULATED (files in §13).

| # | addition | reference | J per seed (add / ref) | mean ΔJ | what moved | verdict |
|---|---|---|---|---|---|---|
| 0 | tune's recommended configuration, reproduced | tune's own runs | 0.68 / 0.68, 0.80 / 0.80 | 0 | bit-identical | baseline |
| 1 | **speech intention + BG gate + think's 6 reflexes** (incl. the efference own-voice tag) | 0 | 0.67 / 0.68, 0.75 / 0.80 | −0.03 | content neutral (what +0.04, check −0.05, name −0.15 on n ≈ 17); drift s0 0.40 → 0.31; ritual −0.31 on s1 (an unbound 气, fixed by #2) | **kept** (principle 4; within noise). Life: §8.2 |
| 2 | **new-word programs**: an unbound heard syllable is re-matched against the babbling memory at every hearing | 1 | 0.76 / 0.75 (s1; s0 has no unbound syllable: identical) | +0.02 | ritual 0.66 → 0.90, name 0.18 → 0.35 | **kept** |
| 3 | **continuous thought** (PFC-1 without a register) at λ 0.3, no fading | 1 | identical by construction: test_equiv_synth.py, 592/592 records and the same weights | 0 | — | **kept** (principle 2) |
| 4 | turn learners see the line's surprise S instead of think's R | 2 | 0.67 / 0.67 (s0); s1 rejected: 12 overlap ticks in a probe world | 0 (s0) | nothing | **rejected** (no gain; the overlap led to reflex 7) |
| 5 | **retinotopic, slot-relative attention** (bench `eye_layout='retino'`; primes, recognition, comparator and focus keys in the slot-relative frame) | 2 | 0.70 / 0.67, 0.89 / 0.76 | **+0.08** | colour 0.11 → 0.82, count 0.46 → 0.81, drift 0.36 → 0.50; slot −0.04 and where +0.01 (no gain where brainlike hoped); check −0.06; B falls after C on s0 (RET −0.05) | **kept** |
| 6 | **reflex 7**: a self-initiated (scene) thought may release only after it has been open 10 ticks | 5 | in every run from #7 on | 0 | it blocked 3–9 unprompted releases per training run of the final configuration (each a possible overlap); 0 overlaps by construction in the staged world: every scene thought there is closed by the partner's line within 5 ticks (scene lead 5; D change gap 5 after the change; PROVED from world.py) | **kept** (principle 8) |
| 7 | **recency in hippocampal recall** (score overlap·σ², was σ^0.5) | 5 | 0.78 / 0.70, 0.91 / 0.89 | **+0.05** | drift level +0.16 / +0.03, held-out +0.03 / +0.04, check +0.08, slot +0.08 | **kept** |
| 8 | naming target ("say the name of what I attend to": a heard line that contains a word recognised at the attended slot is also learned from the attended object alone, at rate 0.5) | 5 | 0.68 / 0.70, 0.90 / 0.89 | −0.005 | what −0.07, check +0.04; life: namings 11 → 15, nothing else (§7.8) | **rejected** |
| 9 | continuous thought with residue (λ 0.6: the previous line stays above threshold for one more line) | 5 | 0.66 / 0.70 (s0) | −0.05 | what −0.11, check −0.06, ritual +0.07 (carry-over helps rituals, as predictive found) | **rejected** |
| 10 | surprise-gated entry into the thought (gain = max(surprise, 0.2)) | 5 | stopped after stage 2 | — | after stage 2: B 0.15 / C 0.36 vs 0.69 / 0.74 — the predicted harm (§7.3) | **rejected** |
| 11 | learned local receptive fields in the ear's upper layers (module test) | layer 0 | — | — | decoding across voices and noise σ ≤ 2: 1.00 / 1.00 / 0.71 (layers 0/1/2); σ 6: 1.00 / 0.85 / 0.20; selectivity not higher (median chars per unit 21 / 25 / 66) | **rejected** (would be a passenger here) |
| 12 | contingency / turn-role third factor against free-use drift | — | not built | — | the recast and the next question arrive at the same delay in this bench (PROVED, §12 item 1); designer: no special mechanism yet | **not added** |
| 13 | longer eligibility traces (recast mode) | — | not needed | — | the context is held by the thought (the recast ≈ 33 ticks after the reply is learned with the question: 1.000 of 376 cases, §7.3); actor eligibility until the reaction; V_go window 50 ticks | **not needed** |
| 14 | babble memory / new-word programs | — | in the base (develop) + #2 | — | — | kept |
| 15 | **wandering thought in long silences** (favourite thoughts, principle 5) | #1–#7 | J identical by construction | 0 | life: favourites share 0.24 → 0.82 (base rate 0.14); other measures within one-life noise (§7.9) | **kept** (principle 5) |
| 16 | **remove the slow context CTX** (principle 7: the lesion of the #1–#7 agent showed it harmful, §8.3) | #1–#7 | 0.79 / 0.78, 0.90 / 0.91 | +0.00 | **name 0.35 → 0.88, 0.29 → 0.82** (+0.53), D3 +0.07 / +0.05, check −0.05, drift −0.06 / −0.01 | **kept (removed)**; confirmed on seeds 2–4 in §8.1 |

### 7.1 The speech intention on the bench and in life (addition #1)
- Staged J: content is neutral (−0.03, within noise), as think found: develop's whole thought (prompt, recall,
  comparator, recognition) fits into the ticks before the first boundary, so a timer never starved it.
- Life (seed 5, 10⁶ ticks, recast tutor; tune's best vs the same core with the intention; SIMULATED):

| measure | tune's best | + intention (#1, #2) |
|---|---|---|
| free replies: accuracy (n) | 0.20 (2,599) | 0.27 (2,233) |
| scene questions in new situations (n 1,134) | 0.25 | 0.28 |
| held-out (n 300) | 0.07 | 0.08 |
| memory facts and novel objects (n 381) | 0.00 | 0.00 |
| social replies (greet / bye / goodnight) | 0.06 | 0.15 |
| items heard once / 2× / 3–4× / 5–8× / 9+× | 0.30 / 0.27 / 0.40 / 0.46 / 0.56 | 0.33 / 0.32 / 0.41 / 0.49 / 0.63 |
| accuracy, first → last tenth of life | 0.12 → 0.25 | 0.12 → 0.33 |
| initiatives (per 1,000 quiet ticks) | 0 | 2,048 (2.09) |
| appropriate share (partner present) | — | 0.18 (0.33) |
| greeted first (rewarded) / namings | 0 / 0 | 366 / 7 |
| overlap ticks / interruptions / longest utterance | 0 / 0 / 10 syllables | 0 / 0 / 11 syllables |
| stability flags | none | none |
| CPU | 1,298 s | 587 s |

  The free-reply n differs because a greeting the agent makes first is an initiative, not a scored reply (greet n 434 →
  68). Initiatives grow over life (0.10 → 0.20 appropriate after the first fifth). Unprompted speech is still mostly
  the partner's lines (你好 outside the greet window 346, 这是什么 106, the partner's fact lines) and 905 utterances while
  alone — the readout of a prompt-free thought predicts what is heard next (open problem 3).

### 7.3 The continuous thought (principle 2)
- **Equivalence (PROVED + SIMULATED).** With competition λ 0.3 below the threshold θ 0.5, gain 1, no fading and own gain 0,
  every new line from another speaker displaces the held items below threshold, so the readout's context equals the
  previous heard line exactly: test_equiv_synth.py (A 100 + B 300) gives 592/592 identical records and identical weights.
  Every behavioural difference therefore comes from the continuous parameters, which can now be tuned (principle 5).
- **Dynamics on the bench stream** (`ct_probe.py`, A 150 + B 200, seed 0): after every heard line (687) the held set
  equals the new line's items (1.000); 3.74 items held on average; the previous line stays as subthreshold activation
  0.30 (one line later 0.09), not as context; the first heard line after an own reply is learned in exactly the thought
  held at the reply's end (1.000, n 376) — the recast, 25–35 ticks after the reply, is learned with the question; no
  clear, no load, no period 2 (each assembly holds itself; there is no reciprocal synchronous loop).
- **Surprise-gated input (the coordinator's next step).** The mismatch it was meant to fix (predictive: the overheard
  reply flows into its own context while it is heard) cannot arise here: the readout's context for a heard line is the
  thought at the line's onset, and a line enters the thought only at its end (PROVED by construction). Gating the entry
  at the line end instead keeps the question in the thought after a predicted recast, so the next question would be
  learned in the old question's context and weaken the learned answer (predicted but absent → weaken: one event lowers
  the answer's drive by η·p ≈ 0.35 × 0.9 ≈ 0.3, PROVED from R2). Tested in #10.

### 7.5 Retinotopic, slot-relative attention (addition #5)
- **What changed.** The bench's `eye_layout='retino'` gives each slot its own block of 240 units with the same feature
  code in every block (innate retinotopy). The attended slot is the block that holds the pointing units (innate
  orienting); internal attention (word → slot + comparator) is unchanged. Everything the agent learns about appearance
  lives in the slot-relative frame: one prime per word (word → 240 units), recognition and the comparator read the
  attended block through the same weights, and the scene-focus keys F are the attended block's units (slot-invariant);
  the slot itself stays available as A. What is learned at one slot is known at every slot.
- **Numbers (paired by seed, vs think + rebind; SIMULATED).** J 0.67 → 0.70 (s0), 0.76 → 0.89 (s1); colour 0.11 → 0.82,
  count 0.46 → 0.81, C3 +0.19 / +0.27, drift level +0.10 / +0.17; check −0.06 and B falls after C on seed 0 (B1 0.74 →
  B3 0.64: shared slot-relative features now carry two answered tasks, so C's frames interfere with B's).
- **Life (seed 5):** scene questions in new situations 0.28 → 0.33, what 0.27 → 0.51, colour 0.08 → 0.26, count 0.21 →
  0.37; social 0.15 → 0.07 (fewer greetings first: 366 → 230); held-out 0.08 → 0.05 (n ≈ 300).
- **A bug found on the way.** develop's eye parser labels pointing codes in order of appearance; with the retinotopic
  layout that label must be the block index, else the agent attends the wrong block (first run: B 0.30, what 0.06).
  Colour was "not solved" in every earlier design (tune: at the prior 0.14–0.20): the cause was the frame, not the rule.

### 7.7 Recency in hippocampal recall (addition #7)
Recall score overlap·σ^ρ with ρ 2 instead of 0.5 (σ = 0.9995 per stored line: an episode 1,000 lines old counts 0.37
instead of 0.78). Paired (s0, s1): J 0.70 → 0.78, 0.89 → 0.91 (+0.05); drift level +0.16 / +0.03, held-out +0.03 /
+0.04, check +0.08, slot +0.08; name unchanged (brainlike's diagnosis — old 我叫X episodes win — does not apply to
develop's recall, whose strength term already favours recent episodes). **Kept.**

### 7.8 The naming target in life (addition #8)
The life of the "+ recency, reflex 7" column of §8.2, with the naming target on (rate 0.5): namings 11 → 15, appropriate share 0.19 → 0.20; everything else
equal or slightly lower (free 0.37 → 0.36, held-out 0.10 → 0.07, once-heard items 0.41 → 0.35, last tenth 0.51 → 0.45).
Unprompted speech is still the partner's line openings (你好 330, 这是什么 58, 我叫小强 45). Likely cause (ASSUMED, from the key schema): a
prompt-free thought activates few context keys, and the shared direct keys (line start, position 0) carry the marginal
distribution of every heard line's first item, so they outvote the naming evidence. **Rejected** (J −0.005, life no
gain). The fix is a production route that does not go through "predict the next heard line": a copy gate whose
candidate is the word recognised at the attended slot (VR), trained by the partner's reaction (open problem 3).

### 7.9 Favourite thoughts: the thought in long silences (principle 5)
The thought never empties: without new input it keeps what entered last. **Wandering** (`ct_assoc` 0.5, every 50 quiet
ticks once a silence is longer than 200 ticks — beyond any answer window): items held together at a line's entry
strengthen a co-holding link (R2-like co-activity with immediate row normalisation, η 0.02); in a long silence the held
items fatigue (rate 0.1, τ 20 steps), associated items rise (spread = 0.5 × links × activity), the strongest item is
renormalised to 1 (homeostasis) and at most 12 stay above threshold. Measured at the first heard line after every
silence ≥ 2,000 ticks (seed 5 life, 41–44 silences):

| | share of the 20 most frequent items among the held items (vocabulary base rate 0.14) | items held |
|---|---|---|
| no wandering (the thought keeps the last line) | 0.24 | 2.0 |
| **wandering** | **0.82** | 9.2 |

So the designer's "favourite thoughts" emerge from experience (the foodie who always comes back to food): after long
silences the thought has drifted to the most familiar topics. Behaviour in the same life is within one-life noise
(free 0.37 → 0.36, held-out 0.10 → 0.07, social 0.38 → 0.35 with goodnight 0.38 → 0.64, last tenth 0.51 → 0.45; 0
overlaps, no flags). Staged J is unaffected by construction (no staged silence reaches 200 ticks). **Kept, on by
default** (principle 5: a feature to tune, not to forbid); its strength (`ct_assoc`, fatigue) is the knob for how
strongly a mind returns to its favourite topics.

### 7.11 Learned local receptive fields in the ear's upper layers (module test, `ear_local.py`, SIMULATED)
Layers 1–2 on the ear's 8 × 6 (band × slice) map, each unit seeing only the layer below within ±1 (layer 1) or ±2
(layer 2) bands and slices; competitive Hebbian learning (k-WTA 2 of 12 per column, instar with normalisation: local).
Trained on the partner's voice; syllable identity decoded by nearest centroid on the two other voices and on noisy
windows (98 lexicon chars, chance 0.01):

| layer | other2 | self | noise σ 1 / 2 / 4 / 6 | median chars per unit |
|---|---|---|---|---|
| 0 (fixed, FEAT0 1,153) | 1.00 | 1.00 | 1.00 / 1.00 / 1.00 / 1.00 | 21 |
| 1 (576 or 1,152 units) | 1.00 | 1.00 | 1.00 / 1.00 / 0.97–0.99 / 0.85–0.86 | 22–25 |
| 2 | 0.71–0.83 | 0.58–0.85 | 0.53–0.84 / 0.41–0.68 / 0.18–0.36 / 0.13–0.20 | 33–66 |

The fixed layer 0 is already perfect and noise-robust in this world (the bench's noise adds energy by max, it never
removes the signal), so learned layers cannot reach behaviour here and the deeper one loses information. **Rejected for
the bench**; to revisit with real audio (coarticulation, variable syllable length), where combination detectors should
pay. The body's F detector reads the window energy, so it is untouched either way.

## 8. The final configuration

`run_j.py w_eye_layout=retino rebind=1 ct=1 init_min_tau=10 hip_rho=2 ctx=0` (+ tune's recommended keys and think's
final intention keys, the defaults of `run_j.py`); life: `life_synth.py` with the same keys plus think's life keys and
wandering (`ct_assoc=0.5 wander=50 wander_after=200`; staged J is identical with or without it, by construction).
Tag `final_noctx` = ARCH_v2; tag `final` = the same with the slow context CTX still in (the configuration whose lesions
led to removing CTX; also the agent lesioned in §8.3).

### 8.1 Staged mode: J on seeds 0–2 and the held-out seeds 3–4 (SIMULATED)
Mean (seed SD of J) over seeds; J = F + 0.5·RET + 0.25·H + 0.25·DR (tune's `analyze.py`, unchanged); per-task columns are
the final frozen probes after BCDD. tune's rows are its own files (my copy reproduces them bit for bit).

| configuration | seeds | J | F | RET | H | DR | B3 | C3 | D3 | what | slot | where | offer | taste | colour | count | ritual | check | name | before | oneshot | viol |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tune's best | 0,1,2 | 0.73 (0.05) | 0.58 | 0.00 | 0.21 | 0.38 | 0.68 | 0.58 | 0.48 | 0.67 | 0.70 | 0.66 | 0.75 | 0.68 | 0.15 | 0.45 | 0.93 | 0.73 | 0.44 | 0.00 | 0.00 | 0 |
| **ARCH_v2** | 0,1,2 | **0.83 (0.05)** | 0.66 | −0.02 | 0.18 | 0.53 | 0.70 | 0.81 | 0.47 | 0.73 | 0.69 | 0.68 | 0.74 | 0.71 | 0.93 | 0.76 | 0.94 | 0.63 | 0.76 | 0.02 | 0.00 | 0 |
| tune's best, held-out seeds | 3,4 | 0.74 (0.01) | 0.59 | −0.01 | 0.24 | 0.38 | 0.66 | 0.60 | 0.50 | 0.73 | 0.65 | 0.57 | 0.79 | 0.71 | 0.20 | 0.45 | 0.82 | 0.75 | 0.48 | 0.00 | 0.00 | 0 |
| **ARCH_v2, held-out seeds** | 3,4 | **0.82 (0.06)** | 0.65 | −0.05 | 0.22 | 0.56 | 0.70 | 0.79 | 0.45 | 0.73 | 0.66 | 0.69 | 0.87 | 0.63 | 0.82 | 0.82 | 0.73 | 0.61 | 0.79 | 0.00 | 0.00 | 0 |
| ARCH_v2 with CTX still in | 0,1,2 | 0.83 (0.06) | 0.65 | −0.01 | 0.20 | 0.57 | 0.71 | 0.81 | 0.42 | 0.73 | 0.72 | 0.68 | 0.76 | 0.71 | 0.87 | 0.73 | 0.98 | 0.66 | 0.26 | 0.02 | 0.00 | 0 |
| ARCH_v2 with CTX still in, held-out | 3,4 | 0.81 (0.05) | 0.64 | −0.05 | 0.22 | 0.54 | 0.69 | 0.80 | 0.44 | 0.73 | 0.64 | 0.68 | 0.82 | 0.71 | 0.86 | 0.80 | 0.74 | 0.64 | 0.44 | 0.00 | 0.00 | 0 |
| the path: + intention (#1) | 0,1 | 0.71 (0.04) | 0.56 | 0.00 | 0.23 | 0.36 | 0.68 | 0.57 | 0.43 | 0.72 | 0.68 | 0.62 | 0.79 | 0.66 | 0.12 | 0.45 | 0.78 | 0.68 | 0.29 | 0.00 | 0.00 | 0 |
| + new-word programs (#2) | 0,1 | 0.72 (0.05) | 0.57 | 0.00 | 0.23 | 0.35 | 0.67 | 0.59 | 0.45 | 0.73 | 0.66 | 0.61 | 0.82 | 0.55 | 0.11 | 0.46 | 0.90 | 0.69 | 0.38 | 0.00 | 0.00 | 0 |
| + retinotopic attention (#5) | 0,1 | 0.79 (0.09) | 0.63 | −0.02 | 0.20 | 0.50 | 0.69 | 0.80 | 0.41 | 0.77 | 0.63 | 0.63 | 0.76 | 0.64 | 0.82 | 0.81 | 0.97 | 0.62 | 0.32 | 0.02 | 0.00 | 0 |
| + recency (#7) (+ thought, reflex 7) | 0,1 | 0.85 (0.07) | 0.65 | −0.02 | 0.24 | 0.59 | 0.70 | 0.80 | 0.45 | 0.74 | 0.72 | 0.63 | 0.77 | 0.64 | 0.86 | 0.77 | 0.97 | 0.69 | 0.32 | 0.00 | 0.00 | 0 |

- **Paired by seed, ARCH_v2 beats tune's best on all five seeds:** J 0.79 / 0.90 / 0.79 against 0.68 / 0.80 / 0.71
  (seeds 0–2, ΔJ +0.10) and 0.88 / 0.76 against 0.75 / 0.73 on the held-out world seeds 3–4 (ΔJ +0.08). 0 mechanics
  violations in all five runs (≈ 12,000 training replies and ≈ 4,800 probe replies each: no overlap, every syllable 3
  ticks, no garble, no extra turn).
- **Where it comes from:** colour 0.15 / 0.20 → 0.93 / 0.82 and count 0.45 → 0.76 / 0.82 (retinotopic attention), name
  recall 0.44 / 0.48 → 0.76 / 0.79 (CTX removed + recency), C3 0.58 → 0.80, the drift level 0.38 → 0.53 / 0.56.
  **Where it costs:** check 0.73 / 0.75 → 0.63 / 0.61 (retina: the dense shared appearance codes), ritual 0.82 → 0.73 and
  taste 0.71 → 0.63 on the held-out seeds, held-out generalisation H unchanged (0.18–0.22), B falls a little after C
  (RET −0.02 / −0.05).
- **Drift in free use** (C after 300 untutored dialogues, learning on): before → after 0.75–0.83 → 0.47–0.62 (loss
  0.14–0.33) against tune's 0.57–0.62 → 0.32–0.41: the level after drift is higher because the level before is; the
  loss is not smaller. Drift is not solved (open problem 1).

### 8.2 Life mode: one unbroken life of 10⁶ ticks (seed 5, recast tutor; SIMULATED)
Same life (seed 5, recast tutor, ≈ 21 h, 450 visits, 42 nights; nothing reset, learning always on). Columns from left
to right add one step each: tune's best (`tune/out/life/life_rec8big_s5.json`); + the intention (#1, #2); + retinotopic
attention (#5) and the continuous thought; + recency (#7) and reflex 7; + wandering (#15); **ARCH_v2** = that − CTX (#16).

| measure | tune's best | + intention | + retino + thought | + recency, reflex 7 | + wandering | **ARCH_v2** |
|---|---|---|---|---|---|---|
| free replies: accuracy (n) | 0.20 (2,599) | 0.27 (2,233) | 0.27 (2,327) | 0.37 (2,224) | 0.36 (2,223) | **0.34** (2,252) |
| scene questions in new situations (n 1,134) | 0.25 | 0.28 | 0.33 | 0.35 | 0.35 | **0.33** |
| scene questions repeated | 0.79 | 0.75 | 0.85 | 0.90 | 0.85 | **0.89** |
| held-out items (n 263–300) | 0.07 | 0.08 | 0.05 | 0.10 | 0.07 | **0.10** |
| memory facts and novel objects (n ≈ 380) | 0.00 | 0.00 | 0.01 | 0.01 | 0.01 | 0.00 |
| social replies (greet / bye / goodnight) | 0.06 | 0.15 | 0.07 | 0.38 | 0.35 | **0.32** (bye 0.03 → 0.34, goodnight 0.13 → 0.64) |
| what / colour / count / slot | 0.26 / 0.08 / 0.20 / 0.44 | 0.27 / 0.08 / 0.21 / 0.44 | 0.51 / 0.26 / 0.37 / 0.42 | 0.48 / 0.27 / 0.39 / 0.54 | 0.48 / 0.24 / 0.33 / 0.57 | 0.37 / 0.20 / 0.39 / 0.62 |
| items heard 1× / 2× / 3–4× / 5–8× / 9+× | 0.30 / 0.27 / 0.40 / 0.46 / 0.56 | 0.33 / 0.32 / 0.41 / 0.49 / 0.63 | 0.36 / 0.35 / 0.40 / 0.59 / 0.66 | 0.41 / 0.39 / 0.54 / 0.64 / 0.74 | 0.40 / 0.37 / 0.54 / 0.64 / 0.73 | **0.36 / 0.39 / 0.50 / 0.60 / 0.72** |
| accuracy per tenth of life (first … last) | 0.12 … 0.25 | 0.12 … 0.33 | 0.12 … 0.35 | 0.12 … 0.51 | 0.13 … 0.45 | 0.17 0.17 0.30 0.30 0.33 0.32 0.36 0.51 0.47 **0.47** |
| answer rate over life | 0.75–0.89 | 0.75–0.90 | 0.75–0.90 | 0.76–0.92 | 0.77–0.91 | 0.78–0.88 |
| initiatives (per 1,000 quiet ticks) | 0 | 2,048 (2.09) | 1,753 (1.83) | 1,995 (2.04) | 1,958 (2.01) | 1,891 (1.94) |
| appropriate share (partner present) | — | 0.18 (0.33) | 0.14 (0.25) | 0.19 (0.35) | 0.20 (0.35) | 0.19 (0.34) |
| greeted first / namings (rewarded) | 0 / 0 | 366 / 7 | 230 / 10 | 375 / 11 | 376 / 11 | 347 / 18 |
| overlap ticks / interruptions | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | **0 / 0** |
| longest utterance (syllables) | 10 | 11 | 10 | 10 | 10 | 11 |
| stability flags | none | none | none | none | none | **none** |
| favourite topics held after long silences (base 0.14) | — | — | — | 0.24 | 0.82 | **0.81** |
| CPU for the life | 1,298 s | 587 s | 591 s | 565 s | 535 s | 525 s |

- **Against tune's best every group improves except memory** (still ≈ 0): free replies 0.20 → 0.34, scene questions in
  new situations 0.25 → 0.33, held-out 0.07 → 0.10, social 0.06 → 0.32, once-heard items 0.30 → 0.36; accuracy keeps
  rising through life (last tenth 0.25 → 0.47): no drift collapse in an unbroken life, no stability flag, the answer
  rate stays 0.78–0.88. One life per configuration: differences ≤ 0.05 between the last three columns are within noise
  (removing CTX, which lifted staged name recall by +0.44, is neutral in life, whose memory questions span visits).
- **The last step (recency #7 + reflex 7) lifted social replies** (0.07 → 0.38; bye 0.04 → 0.41, mostly 再见 now) and
  slot questions (0.42 → 0.54). The mechanism was not isolated (two changes in one step; one life).
- **Initiative** appears in the first fifth and stays at ≈ 2 per 1,000 quiet ticks; greeting first is learned (347
  rewarded greetings; it is learned by listening — the partner's 你好 after the agent's greeting is learned in the
  scene context — and needs the learned brake: without it there is no unprompted speech at all, §8.3); namings are rare
  (18, mostly "这是X" for novel objects, repeated from the partner's statement); most other unprompted utterances are the
  partner's line openings (你好 outside the greet window, 我叫小强, 这是什么) or speech when alone (822): open problem 3.
- **The innate brakes hold in an unbroken life:** 0 overlap ticks and 0 interruptions in 21 h with ≈ 2,000 self-initiated
  utterances.

### 8.3 Lesions: every learned part must reach behaviour (principle 7; SIMULATED)
Method (`ablate_synth.py`, develop's protocol, brainlike's pooled scoring): the final agent of seed 0 (after B → BC →
BCDD) is copied, one part is switched off, and the copy answers the same probe-only schedule (150 dialogues per stage,
held-out share 0.3; B and C frozen, D with episode storage on). Accuracy is pooled over all B + C + D probes
(893 replies; one SE ≈ 0.016). Training-time lesions (awake recall, the reward actor) are full J runs.

| part removed (test-time lesion of the final seed-0 agent) | pooled accuracy B+C+D (n 893) | drop | most affected | verdict |
|---|---|---|---|---|
| none | 0.652 | — | what 0.62, where 0.69, colour 0.90, count 0.83, ritual 0.91, check 0.66, name 0.45 | |
| GEN (cortex readout) | 0.120 | +0.53 | everything but ritual (0.86, carried by the copy gate) | **reaches behaviour** |
| scene focus F | 0.187 | +0.47 | what 0.00, colour 0.00, count 0.08, check 0.12 | **reaches behaviour** |
| learned attention (word → slot) | 0.348 | +0.30 | where 0.01, colour 0.00, count 0.00 | **reaches behaviour** |
| comparator / primes (word → appearance) | 0.365 | +0.29 | where 0.01, colour / count 0.00, check 0.33, what 0.40 | **reaches behaviour** |
| BG copy gate (prompt) | 0.522 | +0.13 | ritual 0.14, what 0.15 | **reaches behaviour** |
| hippocampal recall routes (NEXT / COMPLETION) | 0.640 | +0.01 | **name 0.45 → 0.10** | reaches behaviour (name; n ≈ 20) |
| whole hippocampus | 0.638 | +0.01 | name 0.45 → 0.10 | reaches behaviour (name) |
| visual recognition VR | 0.640 | +0.01 | what 0.62 → 0.52 | weak (what) |
| learned brake P_oth | 0.647 | +0.00 | in a world with feedback: speech after the partner's recast **0.21 → 0.54** (turn_lesion.py, 216 recasts); life without its learning: **0 initiatives** (the innate prior keeps the floor 'taken'), social 0.35 → 0.09, last tenth 0.45 → 0.33 | **reaches behaviour** (turn-taking and initiative, not content) |
| learned eagerness V_go | 0.652 | 0.00 | none in frozen probes; life without its learning: initiatives 1,958 → 1,992, greeted first 376 → 377, appropriate 0.20 → 0.20, social 0.35 → 0.40 | **passenger** → removable (the gain is then the innate g0). think measured small timing effects and credited greeting first to it; here greeting first is learned just as well without it |
| slow context CTX (innate, not learned) | 0.661 | **−0.01** | **name 0.45 → 0.86** (CTX in the hippocampal key hurts name recall, as in develop) | harmful to name → **removed** (training-time removal, §7 #16: name 0.32 → 0.85, J unchanged) |

Training-time lesions (full J runs, seed 0): awake recall off → J 0.78 → 0.65: the final probes are unchanged (F −0.00) but C collapses in untutored use (drift level 0.57 → 0.00 after 300 dialogues; held-out +0.12): **awake recall is the drift brake** (as tune found); reward actor off → J 0.78 → 0.78 (no effect on the staged tasks, where every reply is also learned from the recast by listening); in life (greetings and goodbyes are rewarded but never modelled) the life without the actor gives social 0.35 → 0.31, once-heard items 0.40 → 0.34, everything else within noise: **weak** — kept as the only route by which reward shapes content.
All test-time lesions keep 0 overlaps and 0 garble. Perception (Z), the babbling memory and the programs are not
lesioned: without them nothing can be heard or said.

### 8.4 world2 smoke run
`world2_smoke.py n=4 days=1.5 ct=1 init_min_tau=10 hip_rho=2 ctx=0 ct_assoc=0.5 wander=50 wander_after=200` (SIMULATED;
it shows the agent runs there, it is not a result): four numpy ARCH_v2 agents as mouth-only bodies
(`PerAgent([MouthOnly(agent), ...])`; world2's 1,600-unit egocentric eye goes through the flat-eye path, because the
3-slot retina of §7.5 does not fit its 25 cells — that is port item P4a), babbling at birth, one unbroken life of 45k
ticks (1.5 days) in 156 s (≈ 0.87 ms per agent-tick on one core). world2's headline: free replies 202 (acc 0.03), scene
0.06, memory 0.02, social 0.00, requests 0.00 (the body cannot move), 1,442 initiatives (20 appropriate, 12 rewarded),
vocabulary 4.5 words per agent, 1,023 agent-to-agent turns, **0 ticks of talking over the partner, 0 interruptions**,
no stability flag. One porting finding: in three of the four agents the perceptual categories hit their cap (256)
within a day, because world2 attenuates sound with distance and mixes several speakers, so the same syllable arrives at
many levels and the vigilance rule recruits a new category for each — the port needs level normalisation (gain
control) before the syllable categories (P4d).

## 9. The nine binding principles: how each is met (and the evidence)
| # | principle | how | evidence |
|---|---|---|---|
| 1 | one unbroken life, learning always on | no state is reset between lines, dialogues, days or nights (`dialogue_reset` off; awake recall restores the ongoing thought and context); life mode = one agent from birth, never frozen | life 10⁶ ticks, flags none (§8.2) |
| 2 | continuous thinking, no hard clear/load | PFC-1 is a continuous thought: lines flow in at their end and push the held content down by competition; nothing is cleared or loaded; own speech does not enter | equivalence + dynamics (§7.3) |
| 3 | no sleep | consolidation = awake recall in quiet moments (8 episodes / 12 silent ticks, uniform over a 10k store); `sleep()` is never called | tune: J 0.61 → 0.73 with recall; life: once-heard items 0.14 → 0.30–0.36 |
| 4 | speech caused by thinking | a learned speech intention (confidence accumulator + learned brake / eagerness) released through a BG gate into the body's GO, with the innate brakes | §7.1; life: 2,000 initiatives, greeting first learned, 0 overlaps |
| 5 | attractors / favourite thoughts are a feature | the thought keeps its content through silence (never idle-empty); optional wandering along learned co-holding links (§7.9) | favourites after long silences (§7.9) |
| 6 | local learning only | every rule in §3 uses pre, post, own weight/trace and at most one broadcast signal | — |
| 7 | every learned part reaches behaviour | lesions of the final agent (§8.3): GEN, scene focus, word→slot attention, comparator/primes, the BG copy gate, the hippocampus (name), awake recall (drift), the learned brake (turn-taking and initiative) reach behaviour; VR and the reward actor are weak; the fixed slow context CTX was harmful and was removed; the learned eagerness V_go is a passenger (removable); additions that did not reach behaviour were not adopted | §8.3 |
| 8 | the body's 0-error mechanics | ARCH_v1 body + 7 reflexes; every run is rejected by J if any overlap / non-3-tick syllable / garble / extra turn occurs | 0 violations in every adopted configuration (§8.1) |
| 9 | one-shot naming not engineered | measured only | 0.00 in every run |

## 10. GPU-scale cost
**Prototype (measured, SIMULATED).** One staged run (A 300 + 3 × 1,000 dialogues + 9 frozen probe sets + drift ≈ 4,500
dialogues ≈ 450k ticks, 373k learning events, 66k awake recalls) takes 1,010–1,040 s on one local core: ≈ 2.3 ms per
tick, numpy, single process (tune's base: 850–880 s). A 10⁶-tick life takes 565 s (most life ticks are silent; tune's
best: 1,298 s; ASSUMED cause: the v1 timer builds a reply context after every line). Per agent at the end of a staged run:
MIX 48k conjunction units with the retinotopic eye (115k with the flat eye); GEN 48k × 257 float32 = 49 MB; gates and
critic 1 MB; hippocampus 10,000 cells × 120 DG indices (4.8 MB) + records; PRIME 257 × 240 (0.25 MB); Z 1,153 × 256
(1.2 MB); babbling memory 3,970 × 1,153 bits (0.6 MB); P_oth / V_go ≈ 10⁴–10⁵ keys.

**GPU instance (PROVED arithmetic).** Two targets.
1. *Rich single agent* (develop's sizes; adult corpus, 4,096 items/programs): MIX ≤ 2M units with GEN as sparse rows
   (top-64 outputs, fp16 values + int16 indices = 256 B per unit) = 0.5 GB; hippocampus 100k cells × 1,000-unit keys
   (int16) = 0.2 GB; PRIME 4,097 × 1,600 fp16 = 13 MB; Z 1,153 × 4,096 fp16 = 9 MB. ≈ 0.8 GB per agent.
2. *Many bodies* (world2: 64 agents on one 32 GB V100, ≤ 10 GB per job): V 512 items; MIX hashed into M = 2¹⁹ slots;
   GEN sparse top-32 rows (128 B per slot) = 67 MB; hippocampus 20k cells × 120 int16 = 4.8 MB; everything else < 10 MB.
   ≈ 80 MB per agent → 64 agents ≈ 5 GB.
Compute per tick for 64 agents: ear layer 0 (64 × 1,920 × 4 MAC), syllable matching at a sample tick (64 × 1,153 × 512 =
38M MAC, one GEMM), conjunction hashing (64 × ~400 keys) and GEN gathers (64 × 400 × 32), hippocampal recall at a line
end (64 × 20k × 120 gathers = 154M), awake recall bursts (8 episodes × ~5 events, amortised over the 12-tick quiet
moment). ≈ 0.2–0.5 G operations per tick in the worst tick, ≪ 1 G on average → about 1–2 ms per tick for all 64
agents on a V100 (kernel-launch bound), i.e. ≥ 30k agent-ticks/s, 40× the numpy prototype per agent and in parallel.
The V100 server's CPU (12 shared cores, ~0.6× a local core) cannot run the numpy agent faster than the local box
(gpu/README); only a torch port uses the GPU.

## 11. Porting plan: torch, GPU-batched, many bodies in world2
world2 v1 (bench/world2, selftest 25/25): 24×24 house and garden, up to 64 agents, 3 caregivers that generate their lines
from the world state, embedded metrics; body = 41 action units (32 mouth + 9 motor: move ×4, turn ×2, take, put/give,
eat); senses ear (N, 8, 6), eye (N, 1600) = egocentric 5×5 view × 64 units per cell (the same code in every cell),
touch 16, taste 8, hunger, reward; batched API `agents.step(obs) -> (N, 41)`; ≈ 118k ticks/s with 16 trivial agents.
Bench agents run unchanged through `PerAgent([MouthOnly(agent), ...])` (§8.4: the numpy agent runs there).

**P0 — freeze the reference.** The numpy SynthAgent and its bench numbers (J, life) are the regression target; record a
fixed-seed trace (per-tick mouth output, detected items, thought confidence, releases) for 2,000 ticks of stage B.

**P1 — all state as batched tensors (B agents).** Body units (F, F2, GO, CD[4], R1, R2, phase, Hp, nsyl, breath,
mouth_ago) as (B,) int/bool; ear layer 0 (B, 1,920) state + reserve (float16); Z templates (B, NF, V) fp16; the thought
a (B, V) + the dominant line's order (B, Lmax) int16; current-line buffer (B, Lmax); hippocampus keys (B, Ncell, k)
int16, σ (B, Ncell), content as fixed-size records (items (B, Ncell, Lmax) int16, scene bit-packed (B, Ncell, n_eye/8),
next (B, Ncell) int32); P_oth / V_go as hashed key tables (B, 2¹⁶) fp16.

**P2 — conjunction units without Python dicts.** unit = hash(group_a, value_a, group_b, value_b) mod M (multiply-shift on
int64), M = 2¹⁹ per agent; a collision makes two conjunctions share a unit (graceful). GEN and gates as sparse top-K rows
(K = 32: values fp16 + output ids int16) with an LRU replacement of the weakest output, or dense (B, M, V+1) fp16 when V
is small. Learning = `scatter_add_` of the NLMS step over the active slots.

**P3 — event-driven control flow → masks.** Every tick each agent computes masks: speech onset, sample tick (clock),
line end, beat boundary, thought step k, quiet moment, reward tick. Each stage runs only on its masked rows with fixed
caps (Lmax 12 items per line, ≤ 600 keys per event, ≤ 8 recalls per quiet moment), so the step is one fixed graph
(CUDA graphs) with no per-agent Python.

**P4 — world2 bridge (the new body).** (a) Eye: world2's view is retinotopic (25 cells × the same 64-unit code) →
attention = one of 25 cells (pointing = bits 62–63 of a cell, the caregiver's gaze target), the slot-relative frame of
§7.5 with 25 positions instead of 3 (the prototype hard-codes 3 slots in attD, the comparator loop and the parser;
the port makes S a parameter); scene focus F = the attended cell's 64 units; the location words (左边 …) become room
and direction words learned by the same word → position outstar. (b) Motor: the 9 motor units become
programs exactly like syllables: motor babbling (random unit, one beat) binds "action → change in the eye/touch"
(take → held bit, move → view shift), and the same GEN readout scores motor programs at the body's boundaries; requests
("去拿X", "把X给我", "过来") are learned from the caregiver's reaction (reward, praise) — the actor/critic already used for
recasts. (c) Hunger and eating reward feed the critic (r = eating reward) and a drive term in the intention gain
(hungry → speak/act sooner): a new third factor, one line. (d) Hearing: world2 attenuates sound with distance and
mixes speakers, and the smoke run's syllable categories hit their cap within a day (§8.4): add level normalisation
(divisive gain control over the window) before the categories, and recruit a category only from clean, loud samples. (e) Many agents hear each other: the own-voice tag (efference)
already separates the own voice; other agents are "other speakers" — learning by listening includes overheard
child–caregiver exchanges, as the bench's overheard routines.

**P5 — validation ladder.** (1) one agent, bench stage B, fp32: mouth trace identical to the numpy reference for 2,000
ticks; (2) J on seeds 0–4 within ±0.03 of the numpy numbers (fp16 tolerance); (3) speed: ≥ 30k agent-ticks/s at 64
agents on the V100 (≤ 10 GB, the runner's limit); (4) world2, 4 / 16 / 64 agents for 10 days each: the world2 reference
table (silent / echo / lookup / namer / oracle) as the floor and ceiling; (5) W3: long multi-agent lives with a population
of parameter variants side by side (natural parameter search: J's sensitive constants — the comparator cliffs, η, the
intention urgency — as the varied genes).

**P6 — scale test (designer, IDEAS #12).** The same architecture at 1× / 4× / 16× units and fan-in in a rich world (world2
with the adult corpus as chat), measuring generalisation (held-out, new situations), knowledge capacity (facts
retained by delay × encounters) and whether one-shot naming emerges.

## 12. Known open problems (in the order I would attack them)
1. **Free-use drift.** Learning by listening still learns the partner's next question as the answer when nobody gives
   feedback. In this bench no timing cue can separate a recast from the next question: both start at
   max(45, stop + 6 or + 13) ticks after the prompt (world.py `_wait`, PROVED), so a contingency window cannot do it here.
   The designer's decision (no special mechanism yet; judge it in life) stands. Awake recall is the brake that exists:
   C after 300 untutored dialogues 0.79–0.81 → 0.50–0.60 with it, 0.77 → 0.01 without it (§8.3). In life, recasts
   continue all life and no drift appears (accuracy rises to the end).
2. **Thinking does not yet improve content.** The thought operations (recall → comparator → recognition) are fixed;
   more time adds nothing (think). Needed: a learned controller of thought operations (BG chooses the next operation,
   rewarded by the confidence gain), which the continuous thought can host.
3. **Initiative says the partner's lines.** A scene thought's readout predicts what is heard next, so unprompted
   speech is mostly 你好 / 这是什么 / the partner's fact lines (appropriate share 0.18–0.20; 0.35 with the partner
   present). A naming target through the same readout did not help (§7.8); the next step is a copy route for the
   recognised word (VR) gated by the partner's reaction.
4. **Memory tasks.** One-shot naming 0.00 (measured, not engineered: principle 9); 刚才S是什么 ≈ 0.0–0.05; life memory
   facts 0.00–0.01. The recall reaches the right episode about half the time, the completion gate never learns to
   copy it (develop follow-up).
5. **Held-out generalisation** stays near 0.2 (H); frame interference between tasks that share frame words grows when
   more tasks are learned (retino: C now learned, B falls after C on seed 0).
6. **Babbling reads world.voice offline** (an inherited privileged shortcut): replace by the body's babbling with the
   partner's imitation in life, then lawful physics with a forward/inverse model.
7. **Turn learners trained with feedback misfire without it.** "I spoke since the last heard line" marks recasts in
   training and every new question in a partner-without-feedback world (C probes: 0.76 → 0.69 spoke rate on later
   exchanges). Replacing it by the line's surprise did not help (§7).
8. **(resolved) The slow context CTX hurt name recall** (lesion of the final agent: 0.45 → 0.86; develop saw the same):
   ASSUMED cause: its bits in the hippocampal key make the 我叫N statement less similar to the 我叫什么 cue. Removed
   (§7 #16): name 0.32 → 0.85 on seeds 0–1, 0.76 / 0.79 on seeds 0–2 / 3–4 (tune 0.44 / 0.48), J unchanged.
9. **Social replies** (greet / bye) are learned only through reward, never modelled in recast mode (think's issue 4).
10. **Inputs of different strength** (designer, 2026-10-30) are not addressed.
11. **world2 hearing:** distance-attenuated, mixed sound makes the vigilance rule recruit categories until the cap
    (§8.4); level normalisation is needed before the port's long lives.

## 13. Files (all in `v2/synth/`)
In the repo, `out/` below is `results/` (pickles and the queue log are not kept; `draft_*.md` are not kept).
- **Synthesized agent:** `core_synth.py` (SynthCore: continuous thought, slot-relative retina, recency, naming target,
  surprise record, wandering), `agent_synth.py` (SynthAgent: think's intention + tune's awake recall + rebind + reflex 7
  + retinotopic attention), on top of copies of tune's `core.py`, `rules.py`, `agent.py` and think's `core_think.py`,
  `agent_think.py`, `body_think.py` (unchanged); `bench_copy/` = bench v1.2 + life v2.0 (byte-identical).
- **Drivers:** `run_j.py` (staged J protocol = tune.py's, plus spoke and overlap records), `life_synth.py` (10⁶-tick
  life, favourite-thought measure), `ablate_synth.py` (test-time lesions), `turn_lesion.py` (turn-taking lesions in a
  world with feedback), `world2_smoke.py` (MouthOnly in world2), `runq.py` / `runq2.py` (job queues; `out/jobs2.txt`).
- **Tests and probes:** `test_equiv_synth.py` (synth v1 mode == tune; CT limit == register), `ct_probe.py` (thought
  dynamics), `ear_local.py` (learned local receptive fields), `dbg_unbound.py`, `dbg_rebind.py`.
- **Analysis:** `analyze.py` (tune's objective, unchanged), `table.py`, `pairs.py` (paired by seed), `mdtable.py`,
  `life_table.py`, `lesion_md.py`; `assemble.py` builds this document from `draft_*.md`.
- **Results:** `out/runs/<tag>_s<seed>.json` (one per staged run: base, think, think_rb, think_S, think_ret (bugged
  attention label, superseded), think_ret2, ret_rec, ret_name, ret_ctres, final, **final_noctx = ARCH_v2**,
  abl_norecall, abl_noactor; ret_ctsurp was stopped after stage 2: `out/ctsurpR_s0_partial.log`),
  `out/life/<tag>_s5.json` + `_log.pkl` (think, ret, final, final_name, final_wander, final_novgo, final_nooth,
  final_noact, **final_noctx = ARCH_v2**), `out/agent_final_s0.pkl` + `_ablate.json`, `out/ear_local_s0.json`,
  `out/world2_smoke.json`, logs `out/*.log`, queue log `out/queue.log`.

## 14. Notes for the coordinator (bench issues; no metric was changed)
1. **Recast vs next question have the same timing in v1.2.** The recast starts at max(45, stop + 6) ticks after the
   prompt and an untutored next question at max(45, stop + 13) (world.py `_wait`), i.e. both ≈ 25–35 ticks after the
   agent's reply. No contingency window can tell them apart in this bench (develop's 10-tick window never fired on
   v1.2; brainlike measured 20–35 ticks). Worth a README line next to the CD trap.
2. **Probe-only worlds are partners without feedback.** The frozen probes and the drift test never recast, so agents
   that learned turn-taking from recast feedback meet a different turn structure there (think's R feature; spoke rate
   on later exchanges of a C dialogue 0.69 vs 0.76 on the first).
3. **eye_layout='retino' with develop-style parsers.** A parser that labels pointing codes by order of appearance
   gets arbitrary slot labels; with the retinotopic layout the attended slot must be the block that holds the pointing
   units (my first retino run had this bug: B 0.30; fixed: B 0.74).
4. **Life with retino** changes the number of scored items slightly (timing depends on the agent): 2,327 vs 2,233 free
   replies in my two seed-5 lives.
5. **Inherited shortcut:** develop's babbling enumerates the muscle sets offline through `world.voice` (think's issue
   1); every comparison here inherits it equally.
6. The GPU runner was not used (numpy agent; the server CPU is slower than the local box).
7. **world2 note:** a bench agent run through `MouthOnly` meets distance-attenuated, mixed sound; vigilance-based
   syllable categories (develop's Z) then fill their cap within a day (3 of 4 agents in the smoke run). Not a world2
   bug — agents need gain control — but worth a line in world2's README for bench agents.

## summary_zh
1. **合成了一个统一的架构 ARCH_v2。** 以 tune 的最佳设置为底（develop 的发育式核心 + 交错课程 + 清醒回忆 + 一万格海马体），
   加上 think 的"由想法引起说话"（说话意图 + 基底节闸门 + 先天刹车反射），把"上一句寄存器"换成连续的想法（安静很久时
   会慢慢走向最熟悉的话题），再加先天的视网膜式注意（按位置分块，学到的东西对每个位置通用）、新字发音补绑、
   回忆偏向最近、一条新反射（主动说话前想法要先开着 10 拍），并去掉了固定的"慢上下文"层（摘除实验显示它只会损害回忆）。
   九条原则全部满足：不重置、不清空想法、不睡觉、说话由想法引起、规则都是局部的、机制 0 错误。
2. **逐个加入、逐个检验**（每项至少 2 个种子，同种子对照）：
   - 说话意图：内容不变（J −0.03，在噪声内），但在"一生"里学会主动先打招呼，不抢话。保留（原则 4）。
   - 新字补绑：修好了一个音节没有发音程序的问题（礼貌用语 0.66→0.90）。保留。
   - 连续想法：在测试台的时间尺度上与旧寄存器逐位等价（已证明），保留（原则 2）。
   - 视网膜式注意：J +0.08；颜色题 0.11→0.82、数量题 0.46→0.81（以前所有设计颜色题都停在先验水平）。保留。
   - 回忆偏向最近：J +0.05。保留。
   - 想法在长时间安静中游走：安静很久后，想法里最常见话题的比例从 0.24 升到 0.81（随机是 0.14），设计者说的
     "爱吃的人总扯到吃"出现了；其他指标不变。保留（原则 5）。
   - 去掉慢上下文层：J 不变，名字回忆 0.32→0.85。保留这个删除。
   - 不采用：耳朵上层学局部感受野（这个世界里没好处，第二层更差）；用"意外度"代替轮次特征（没改善）；
     想法保留上一句（−0.05）；按"意外度"决定新句进入想法（B 从 0.69 掉到 0.15，正是事先推算的害处）；
     "说出我看着的东西"的产出目标（测试台和一生都没提高）；对抗漂移的"及时性"第三因子（这个测试台里改说和下一个问题
     到来的时间完全一样，无法区分；设计者也说暂不加）。
3. **最终结果。** 分阶段测试：J 0.73→0.83（种子 0–2，每个种子都更好），从未用过的种子 3–4 上 0.74→0.82；
   颜色题 0.15→0.93、数量题 0.45→0.76、名字回忆 0.44→0.76；机制 5 次全部 0 错误。
   一生模式（10⁶ 拍，约 21 小时）：自由回答 0.20→0.34，新场景题 0.25→0.33，社交回答 0.06→0.32，
   只经历过一次的内容 0.30→0.36，一生最后十分之一的正确率 0.25→0.47；主动说话约 1,900 次（先打招呼 347 次有奖励），
   0 次抢话，没有稳定性问题。代价："这是X吗"略差（0.73→0.63），新种子上礼貌用语和"好不好吃"略差。
4. **摘除实验**：读出层、场景焦点、注意、视觉比较器、复制闸门、海马体（名字）、清醒回忆（防漂移：去掉后 C 从 0.77
   掉到 0.01）、学到的刹车（去掉后一生里完全不主动说话、复述后插话从 0.21 升到 0.54）都确实影响行为；视觉识别和
   奖励学习作用很弱；学到的"积极性" V_go 是"乘客"，可以去掉；慢上下文层有害，已删除。
5. **还没解决**：无人教时的漂移水平提高了但下降量没变；"多想"还不能提高内容（思考步骤固定，需要学会"下一步想什么"）；
   主动说话多半在说对方的台词；一次学会新名字、"刚才是什么"、跨访问的长期记忆题仍接近 0；没见过的组合泛化约 0.2；
   咿呀学语仍借用了世界的发音表（继承下来的捷径）；world2 里声音随距离变弱，音节类别会不断新建，移植时要加增益控制。
6. **GPU 与移植**：写好了 torch 批量化移植计划（所有状态成张量、组合单元用哈希、事件改成掩码、接 world2 的 25 格视野和
   9 个动作单元，饥饿和吃饭奖励作为新的第三因子）；估计一张 V100 可以同时跑 64 个身体，每个约 80 MB。
   world2 冒烟测试：4 个智能体（只有嘴）连续生活 1.5 天，能回答、主动说话、彼此对答（1,023 次），0 次抢话。
