# ARCH_v3_core, revision 2

**Who and when.** The synthesis member, 2026-11-01. Design and toys only: no training run and no server job.

**What it answers:**
- the four adversarial reviews in `review/`: develop, ml, neuro and skeptic;
- `REVISION_BRIEF.md`, with its 12 designer-spec points;
- `docs/v3design/DESIGNER_SPEC.md`, the designer's 40 mechanisms with their status;
- IDEAS #12–#23;
- the designer's repos (read only): ana (`SIMPLE_CORE.md`, `USER_INTENT.md`), chain-neuron-architecture, char-neuron-network and char-neuron-voice.

**Labels:**
- **PROVED**: code, count or arithmetic.
- **SIMULATED**: a toy run, with its file named.
- **CITED**: a paper, a report or a repo.
- **ASSUMED**: design judgement, not tested.

**New toys** (in `docs/v3design/synth/`). Each ran locally as one process under 40 MB, after `free -m`, reusing the develop reviewer's caregiver world read-only.

| toy | what it tests |
|---|---|
| `age_toy.py` / `.out` / `.json` | three codes for the latest event's age, none of them a dedicated time module |
| `hebb_toy.py` / `.out` / `.json` | the designer's cortical rule against the delta rule, on stability and on correction cost |

---

## 0. Bottom line

1. **The core is rebuilt on the designer's own mechanisms.**
   - **Two mechanisms.**
     - M1 is one area type:
       - binary units;
       - thermometer-pair inputs;
       - a variable active count set by a quantile-tracked global threshold;
       - gain control and fatigue;
       - mutual excitation;
       - two-layer propagation weights.
     - M2 is one plasticity rule:
       - temporal Hebbian learning on neurons that just changed;
       - concave, uncapped growth;
       - one global decay by depth, with birth edges exempt;
       - a plasticity floor of about 1 %;
       - slow split and delete.
   - **Wiring.** Everything else is wiring:
     - 1:1 chain neurons to the PFC and the hippocampus;
     - return paths;
     - a motor area where blocking dominates.
   - **Priors and broadcast.** There are 5 innate priors and one broadcast, δ. δ acts only on motor and value links.
   - **Deleted from r1:**
     - μ;
     - the TIME module;
     - event-boundary loading;
     - τ_ref;
     - the "self" pathway;
     - M4's accumulator and source gates;
     - the NLMS readout;
     - the n_use schedule.
2. **The anchored context survives in a form the spec allows.**
   - The PFC's most recently driven assembly fades by **fatigue handover** under the area's own competition. Which of its units are still winning encodes its age. No time neurons are needed.
   - SIMULATED (`age_toy`, 5 seeds):

     | age code | W1, 3,000 untutored | realistic caregivers, care continues | result |
     |---|---|---|---|
     | handover | 0.99 | 0.68 | equals r1's time bins in every world |
     | literal "number of units still on" (nested sets) | 0.00 by 900 (echo 0.99) | – | collapses |

3. **The designer's learning rule works in this context** (SIMULATED, `hebb_toy`):

   | measure | designer's rule | delta rule |
   |---|---|---|
   | W1, after 3,000 untutored | 1.00 | 0.99 |
   | realistic caregivers: accuracy | 0.69 | 0.68 |
   | realistic caregivers: says partner lines (echo) | **0.01** | 0.33 |
   | realistic caregivers: silent | 0.16 | 0.00 |
   | untouched answers retained | 0.99–1.00 | 1.00 |
   | changed answers right after 4 / 8 / 16 re-teachings | **0.10 / 0.25 / 0.60** | 0.65 / 0.90 / 0.85 |

   - Corrections are 2–4× slower. That is the designer's #22 profile, "learn once, keep".
   - It is risk R1 and question Q2 for the designer.
4. **All five FATAL items are closed by structure, not rules** (§1):

   | FATAL | fix |
   |---|---|
   | release cannot stop | blocking dominates; speech needs a sharp muscle signal (char-neuron-voice) |
   | the world seed changes the physics | a lineage-fixed physics seed |
   | replay collides with the held context | no μ: replay must out-compete the held assembly |
   | tick-level O1 collapses | only neurons that just changed learn; "when" lives in the motor competition |
   | 0 errors after maturity is unreachable | babble and exploration live inside the motor competition; a yield clause; only agent-caused exchange errors count |

5. **World.** The realistic caregiver arms are mandatory: re-ask, move-on and politeness 0.8.
   - With care continuing, the core holds 0.65–0.69.
   - Quick move-ons with care stopped collapse every context, 0.10 at 3,000 (SIMULATED).
   - The residual requirement: tutoring never stops entirely.
6. **Honest numbers.**
   - **Compute:**

     | | CPU per 10⁶-tick life | one full life (L ∝ s) |
     |---|---|---|
     | s = 1 | 1,000–2,000 s | 1,000–2,000 s |
     | s = 4 | 4,000–8,000 s | 16,000–32,000 s, i.e. 3–6 chained jobs |

     - The skeptic measured the tick costs.
     - GPU batching is unproven: gpuport measured 259 ms per tick at batch 64, i.e. 250 agent-ticks per second.
     - The lineage runs on CPU: 30–45 h.
   - **Knobs:** about 146 live numbers in v2.2 become about 70 in r2, of which 30 are genes. This is an estimate; the freeze lint does the census.
7. **The ladder is reordered.**
   - The representation swap comes first: thermometer inputs, sound assemblies, chain neurons, the char-network PFC and the designer's hippocampus.
   - The hippocampus, merge and tick-benchmark tests come next.
   - The lineage comes only after all of them.
8. **Questions for the designer:** Q1–Q6 in §8. Q1 asks whether "age = which units of a fading assembly still win" is acceptable.

---

## 1. Change log: every FATAL and SERIOUS item

| # | reviewer: item (severity) | verdict | how (section) |
|---|---|---|---|
| D1 | develop: 0 errors after maturity is impossible with babble outside M4 (FATAL) | accepted | Babble is spontaneous activity of the muscle-drive units *inside* the blocking competition. Its amplitude follows the motor maps' imprecision, so it fades as babbling learning completes. After maturity, only agent-caused errors in exchanges or addressed windows count (§3.5, §3.9) |
| D2 | develop: P5b is contiguity (SERIOUS) | accepted | 5b = onset × (1 − q̂), with q̂ the PFC's own prediction of that onset. Own voice does not count (§3.4) |
| D3 | develop: W1 / W4 are timing rules moved into the world (SERIOUS) | accepted, modified | The development world uses realistic caregiver statistics (re-ask 30 %, move-on 15 %, jitter, politeness 0.8), plus an impolite sealed partner. SIMULATED: care continuing is required (§4.3) |
| D4 | develop: maturity in the wrong currency (SERIOUS) | accepted | Maturity = 300 world-counted contingent exchanges, capped at 15,000 audible syllables (§3.9) |
| D5 | develop: μ starts replay in answer windows (SERIOUS) | closed by deletion | There is no μ. Replay must out-compete the held assembly (§3.3) |
| D6 | develop: crowds (SERIOUS) | accepted | P3 picks one engaged voice. The brake and the yield read it, and P4b raises η on that stream. Overlap is counted against the engaged partner (§3.4) |
| D7 | develop: #17 values under TD (SERIOUS) | accepted | No per-tick γ; δ's eligibility runs per event. "Upset / hurt" drives the blocking units through innate edges, a Pavlovian brake (§3.4) |
| D8 | develop: disjoint self/other units against multi-syllable lines (SERIOUS) | accepted | Overlapping codes (spec #34): content units are shared and voice units differ. The auditory area completes the syllable sequence (spec #12) (§3.6) |
| D9 | develop: experience per lineage (SERIOUS) | accepted | Siblings get different family mixtures; L grows over the generations; develop's phases become gen 0's world (§5) |
| M1 | ml: release has no stop (FATAL) | closed | Blocking dominates by default. There is no urgency term and no clip (§3.5) |
| M2 | ml: the world seed changes the physics (FATAL) | accepted | A lineage-fixed physics seed (voices, mouth, eye codes); an episode seed per life (§5) |
| M3 | ml: MIX about 20× sparser (SERIOUS) | moot | The RandMix AND units are gone. PFC convergence units take a fan-in ∝ 1/source activity (§3.6) |
| M4 | ml: CA3 capacity about 1,000 (SERIOUS) | accepted | The designer's hippocampus is short-term by design; its horizon is sized, and X3 tests it first (§3.6, §6) |
| M5 | ml: merge prerequisites fail (SERIOUS) | accepted | Activation-matching alignment inside the merge; the sound area ≥ 4 × lexicon × assembly size; the lineage only after the swap; a Darwinian control (§5) |
| M6 | ml: gating generalisation (SERIOUS) | closed | No source gates. The motor competition also reads the content-free pooled size of the held assembly (§3.5) |
| M7 | ml: new kinds (SERIOUS) | partly | X6 (depth) runs first; the feedback-alignment fallback is pre-registered; there is no route list (§6, §7) |
| M8 | ml: scaling is inert (SERIOUS) | accepted | Hippocampal connectivity ∝ s; an experience-only arm; a pre-registered exponent (§4.4) |
| M9 | ml: stability scoped to readouts only (SERIOUS) | accepted | X1+ runs the whole core with all plasticity on (§6) |
| M10 | ml: #22 metaplasticity | closed by the spec | Concave growth **is** #22; its correction cost is measured (§3.8) |
| N1 | neuro: replay against the held context (FATAL) | closed | No μ; the thinking-range competition decides (§3.3) |
| N2 | neuro: CA3 size (SERIOUS) | as M4 | – |
| N3 | neuro: P5b (SERIOUS) | as D2 | – |
| N4 | neuro: M4 credit assignment and ε (SERIOUS) | moot | No M4. δ updates only the links active in the release; exploration is motor noise (§3.4, §3.5) |
| N5 | neuro: M2 blocks corrections (SERIOUS) | changed | There is no n_use floor. The designer's rule corrects 2–4× slower than delta: R1, Q2 (§3.8) |
| N6 | neuro: ungated LATEST, world-fitted segmentation (SERIOUS) | closed | No loading rule and no event boundary. Entry is by competition with an attention gain (§3.2) |
| N7 | neuro: P4 wired to world codes; an innate name (SERIOUS) | accepted | Detectors are defined on sensor physics; the name is learned (§3.4) |
| N8 | neuro: binary forms of holding and of fixed k | closed | Bistable binary assemblies with a variable count; no graded holding (§3.2) |
| S1 | skeptic: O1 tick- vs event-level (FATAL) | closed | Only neurons that just changed learn (per unit, no detector); "when" lives in the motor competition (§3.3, §3.5) |
| S2 | skeptic: the value loop is untested; P5b and probe timing (SERIOUS) | accepted | Contingent 5b; X1+ runs the full loop with real probe timing (§6) |
| S3 | skeptic: maturity cannot be met (SERIOUS) | accepted | As D1, plus a yield clause in P1c (§3.4) |
| S4 | skeptic: compute (SERIOUS) | accepted | §4.2 |
| S5 | skeptic: the representation swap comes last (SERIOUS) | accepted | It is now steps L1–L3 (§5) |
| S6 | skeptic: about 16 hidden items (SERIOUS) | accepted | Each one is listed with its fate (§4.3) |
| S7 | skeptic: §6 numbers ignore the rules (SERIOUS) | accepted | §3.7 uses r2's rules; untested numbers are marked ASSUMED |

---

## 2. Conformance with the designer's specification

All 40 rows of DESIGNER_SPEC. The brief's 12 points map onto rows 1, 4, 5, 7, 22, 17, 24, 25–26, 34, 32 and 33, and IDEAS #12–#23 onto rows 15–16, 22–23 and 36–40.

| # | the designer's mechanism | r2 | how, or why not |
|---|---|---|---|
| 1 | binary neurons | conforms | no graded holding (λ_c 0.3 deleted) |
| 2 | distributed codes, emergent features | conforms | sound assemblies from step L1 |
| 3 | thermometer pairs; input half on | conforms | each ear-window channel becomes N-level pairs; the eye likewise until local3d |
| 4 | strength = number active; global threshold, no quota | conforms | a quantile threshold per area, smoothed (char-net θ_lr). Age handover needs competition: verified with top-2 (`age_toy`); X1+ re-tests it with the threshold |
| 5 | random connections from birth | conforms | – |
| 6 | local (y, z) connectivity | conforms | sensory chains local; the return path only to nearby units (ana) |
| 7 | chain neurons 1:1 to the PFC and the hippocampus; always-on units not sent | conforms | always-on units filtered by cross-input std (chain repo) |
| 8 | return path, strength as a rational ↔ instinct dial | conforms | it is also the restoration path (#14); one gene |
| 9 | char-network PFC | conforms | mutual excitation, spreading activation, quantile threshold, gain control, two-layer weights |
| 10 | PFC as a recurrent predictor | conforms | by temporal Hebbian links, not an error readout |
| 11 | inhibitory partners | **open, Q4** | ana measured no gain: association specificity +0.010 vs +0.013. That test was a global negative-edge form with coarse sampling, not exactly the partner form. Retest for distractor resistance and wrong-answer suppression (X1+ arm) |
| 12 | the PFC gives a rough intention; the areas complete the sequence | conforms | the auditory area's own temporal links (红 → 色) complete the line |
| 13 | direct inter-area links; instinct vs reason | conforms | through the motor competition |
| 14 | temporal Hebbian | conforms | the only rule; tested (`hebb_toy`) |
| 15 | concave, uncapped growth | conforms | Δw = η·a_i·a_j/(1 + w/w_ref)^pow |
| 16 | harder to change with use, never frozen (about 1 %) | conforms | concave growth plus ana's plasticity floor |
| 17 | one global decay by depth | conforms, retest | v2 tune found decay harmful, but it decayed error-driven NLMS readouts (`wd_gen`, `wd_gate`), not concave Hebbian edges that re-strengthen with use. Not the designer's form; retest at L9 |
| 18 | no forgetting, memory or time modules | conforms | no TIME module; one decay; the hippocampus is a region with its own numbers |
| 19 | two-layer weights | conforms | propagation = strength / out-strength / in-strength^α |
| 20 | no phenomenon-specific mechanisms | conforms | 2 mechanisms |
| 21 | dopamine is a tool | conforms | δ only on motor and value links; the cortex learns without it |
| 22 | the designer's hippocampus | conforms | its own chain-fed population, fast Hebbian growth and decay, no episode cells |
| 23 | long-term memory = the cortex via recall | conforms | recall drives the PFC; the PFC's rule learns it; no sleep |
| 24 | split and delete | conforms, slow | local3d found mostly no gain unless capacity is short. Kept on a very long timescale; tested at the scaling step |
| 25 | motor: inhibition dominates by default | conforms | – |
| 26 | blocking against muscle-signal sharpness | conforms | replaces the accumulator |
| 27 | babbling by self-hearing | conforms | online, lawful physics |
| 28 | think in sounds, speak with muscles | conforms | the PFC holds sound assemblies |
| 29 | stop by learned inhibition after a line | conforms | the END symbol is deleted |
| 30 | the model decides when to reply | conforms | – |
| 31 | continuous thought, no close/load | conforms | no loading rule |
| 32 | no event-boundary rules | conforms | learning on neurons that just changed is per unit; there is no detector |
| 33 | no refractoriness | conforms | τ_ref deleted |
| 34 | self/other by overlapping codes | conforms | – |
| 35 | favourites are tuned | conforms | – |
| 36 | excitability = context length | conforms | the PFC threshold target is a gene |
| 37 | local3d vision | deviates, temporary | world codes are thermometer-encoded until local3d (L-late) |
| 38 | instinct = hard-to-change links | conforms | – |
| 39 | evolution and merging | conforms | §5 |
| 40 | whole-model scaling | conforms | §4.4 |

**#22, the brief's choice between ML's "confirmed use" and neuro's two-component synapse.** Neither is adopted.
- The designer's concave growth already is #22: a strong weight is hard to raise. It is lowered only by the global decay, and there is a floor of about 1 %.
- Corrections then come from the new association growing (a rare pair grows noticeably within a few co-occurrences) and from the old one being outgrown and decaying.
- This keeps the designer's goal. Its cost was measured (§3.8).
- Neuro's two-component synapse stays the pre-registered fallback if X1+ shows that corrections are too slow (Q2).

**#23.** The hippocampus stores everything. Repeated input only re-strengthens the same links, with saturating growth. Conforms (M2 in the hippocampal region).

---

## 3. The revised core

### 3.1 On one page

```
 ear window ─► thermometer pairs (half on) ─► auditory L1 … L5 (local, birth-wired, fixed)
 eye ────────► thermometer pairs ───────────► visual L1 … L5      (local3d later)
      every layer ──1:1 chain neurons (always-on units not sent)──► PFC   and ──► hippocampus
 PFC  = char-network: mutual excitation, spreading activation, quantile global threshold (~20–30·s active),
        gain control, fatigue (unequal rates), two-layer weights; holds sound/sight assemblies; thinks
 PFC ──return path (strength = the rational↔instinct dial)──► auditory/visual L2…L5 (inner hearing; restoration)
 hippocampus = own population, chain-fed, fast Hebbian links, fast decay; recall drives the PFC
 motor = muscle-drive units vs blocking units (blocking wins by default); drive from the auditory return path
         (the inverse map learned by babbling) and direct inter-area links; release when sharpness > blocking
 one plasticity rule everywhere; one broadcast δ on motor/value links; 5 innate priors
```

### 3.2 M1, the area (one neuron type, one connection type)

**Definition:**
- **Units.** Binary units with a weighted-sum threshold. Inhibitory units are ordinary units with negative out-weights (ana appendix 5).
- **Global threshold.** Each area has one global threshold θ, tracked as a quantile and smoothed: θ ← (1 − θ_lr)θ + θ_lr·θ\*, where θ\* lets a target count cross (char-net #5).
  - The active count varies around the target, so a strong input lights more units.
  - The thermometer input layers stay half on.
- **Gain control:** drive / (1 + γ_g·Σa).
- **Fatigue.** Spike-frequency adaptation, with unequal rates across units: a log spread of fatigue time constants (a species constant).
- **Persistence.** Mutual excitation within co-active assemblies, which are bistable.
- **Propagation.** Strength is divided by out- and in-strength (two-layer weights, char-net #6).

**Emergent properties, with no rule for them:**
- Content held in the PFC persists until fatigue and competition displace it.
- The units of a held assembly hand activity over, from fast-fatiguing to slow-fatiguing units, at log-spaced ages. **That is the age code** (§3.8).
- New input enters by spreading activation and competition; there is no loading rule.

**Basis.**
- CITED designer: char-neuron-network #5–#8; chain-repo rules 4 and 7; ana kernel #1–#2.
- CITED brain: normalisation (Carandini & Heeger 2012); adaptation and persistent activity (Wang 2001); sequential activation through adaptation in competing populations (Rabinovich et al. 2001).
- Starts from: char-network `brain.py` (threshold, gain), chain-repo `brain.py` (chains, alive IDs), ana's engine.

### 3.3 M2, one plasticity rule

**Definition.**
- **Who takes part.** Only neurons that just changed: rising (off → on) and falling (on → off), each with a short trace of a few ticks.
- **The three cases** (plastic2):
  - falling → rising strengthens: earlier A, later B, so A→B grows;
  - the reverse order weakens A→B;
  - co-rising units strengthen each other.
- **Concave, uncapped growth:** Δw = η_region·a_i·a_j/(1 + w/w_ref)^pow.
- **One global decay.** Every T ticks, plastic edges are multiplied by γ_depth, and edges below a threshold are deleted.
  - Birth edges are exempt.
  - γ_depth follows ana's table: L1 0.90 … PFC 0.999. The hippocampus gets the fastest γ and the largest η.
- **Floor.** About 1 % of plasticity remains for strongly consolidated edges (ana's `plast_min`).
- **Structure, on a very long timescale.**
  - Split: parents in the 40–60 % usage band, 5 % per round, 2 children with 2 inputs each.
  - Delete: never-active and always-on units.
  - Chain neurons follow. Splitting is rarer near the input.
- **The third factor (motor and value links only).** Links into the motor area's drive and blocking units multiply Δw by δ. The eligibility is the co-activity trace of the release.

**Why this solves O1 (skeptic FATAL).**
- Learning happens only when target units rise, so silent ticks teach nothing. There are no "predicted but absent" negatives.
- This is per unit, from the rule's own form: no event detector (spec #32).
- "When to speak" lives in the motor competition.

**Self-reinforcement is avoided for free** (ASSUMED until X1+ tests it).
- Before speaking, the intended assembly is already on in the PFC (speech is caused by thinking).
- Hearing myself therefore makes no PFC content units rise. Only the voice units rise.
- So own lines teach little: the delta rule's "predicted → stop", obtained without a tag.

### 3.4 The broadcast and the priors

**δ, the one broadcast.**
- r = Σ P5 terms minus their slow running mean.
- It acts only on motor and value links (spec #21, "a tool").

**The priors: hard budget of 5 priors and 15 sub-items.**

| prior | sub-items | brain (CITED) | r2 change |
|---|---|---|---|
| **P1 body and brainstem** | (1a) muscles, voice physics, force by unit count (chain repo); (1b) syllable rhythm in the mouth's physics; (1c) brake and yield: innate edges from the engaged voice's presence to the blocking units, so I do not start, and I yield at the next syllable if that voice rises; (1d) spontaneous motor noise, amplitude ∝ motor-map imprecision | Jürgens 2002; Schegloff 2000; Oller 1980 | the yield clause (S3), the engaged source (D6), the fading noise (D1). The ear's sampling oscillator and the corollary-discharge gate are gone: the chains and the forward model do it |
| **P2 wiring plan** | (2a) areas, chain neurons, return paths (not to L1), and their sizes × s; (2b) topography (retinotopy, fovea); (2c) region numbers (η, γ_depth) | Katz & Shatz 1996; O'Leary & Sahara 2008 | no TIME bank, no MIX anchor rule, no M4 source list |
| **P3 orienting** | (3a) attention gain on onsets and novelty; (3b) on the engaged voice and the pointed or gazed place | Johnson et al. 1991; Butterworth & Jarrett 1991 | picks one engaged voice (D6) |
| **P4 coarse social detectors** | (4a) person present; (4b) addressed; (4c) pleased or upset. All read sensor physics: a pitch contour on the voice channel, a configural face on the retina | Johnson & Morton 1991; Fernald 1993 | no world-code wiring, no innate name (N7) |
| **P5 values** (innate edges, plasticity 0, never merged) | (5a) hunger relief; (5b) a response beyond expectation, onset × (1 − q̂); (5c) pleased +, upset −, where "upset" also drives the blocking units directly (a Pavlovian brake) | Watson 1972; Gergely & Watson 1999; Blair 1995; IDEAS #17 | contingency (D2); no per-tick γ (D7) |

### 3.5 Motor: how release starts and stops

- **The competition.** Muscle-drive units compete with a blocking population, and **blocking wins by default** (char-neuron-voice §3).
- **The drive** comes from the auditory return path of what the PFC holds, through the inverse map learned by babbling, and from direct inter-area links.
- **The release.** A line starts only when the drive's sharpness exceeds blocking. Sharpness = the strongest muscle activation × the muscle count; a uniform signal gives 1.
- **The stop.** The line's last syllable's units drive the blocking units through a link learned from line ends (spec #29). There is no END symbol and no refractoriness.
- **What blocking reads** (learned with δ, plus innate edges):
  - the engaged voice (P1c);
  - "upset" (P5c);
  - the content-free pooled size of the held assembly, which gives timing that transfers across questions (M6).
- **Noise.** Exploration and babble are noise inside this competition, so they win only where blocking is weak.

### 3.6 Wiring plan, with sizes at s = 1 (ASSUMED)

| area | units | notes |
|---|---|---|
| auditory L1–L5 | thermometer L1 (half on) + about 4 × 2,000 | local, birth-fixed (ana kernel 2); every layer is chained to the PFC and the hippocampus |
| sound assemblies in the PFC | ≥ 4 × lexicon × assembly size (about 141 × 16 × 4 ≈ 9 k), each a chain follower | birth-match claims; alignment at merge (M5) |
| PFC recurrent pool | about 20 k; target active about 20–30 × s | char-network dynamics; convergence units with fan-in ∝ 1/source activity (M3) |
| visual L1–L5 | world codes → pairs (local3d later) | chained |
| hippocampus | about 10 k chain followers; fast links | horizon sized in X3; connectivity ∝ s (M4, M8) |
| motor | 32 muscles × force levels + blocking units | inverse and forward maps learned by babbling |

### 3.7 The behaviour chains with r2's rules (numbers ASSUMED until X1+)

**(a) Question in noise → hold → answer → learn.**
1. Q's syllables rise in the auditory layers and their PFC chain followers. Mutual excitation holds Q's assembly, and fatigue hands it over across ages.
2. In a format, A is heard at τ ≈ 10. A's units rise, while Q's units active around τ ≈ 10 have risen recently. So Q_τ → A grows by concave increments: 0.10, 0.19, 0.27, … (η 0.1, w_ref 1, pow 1).
3. The next time Q is asked, the propagation from Q_τ lights A in the thinking range. The return path makes A heard inwardly.
4. The inverse map then sharpens A's muscle pattern; the release needs sharpness > blocking.
5. The auditory area's learned 红 → 色 link completes the line. The line-end link drives the blocking units, which stops it.
6. Approval (5c) strengthens the drive links; disapproval strengthens the blocking links for that context. The answer is withheld even before the cortex re-learns it.

**(b) Restoring a misheard syllable.**
- The return path biases the auditory layers toward the held prediction, with the gene "return strength".
- In noise a degraded syllable is completed. A clearly heard different syllable still wins, because the return path is modulation only (chain-repo rule 6: return edges ≤ half the forward ones).
- Experience enters through the strength of the PFC's predictions (#14).

**(c) Greeting.**
- A person's arrival is the newest strong input, so its assembly is held.
- The greeting heard at τ ≈ 30 links arrival_τ → 你好, so the drive comes when the arrival assembly is that old.
- A reply beyond expectation is rewarded (5b). Empty-room objects drive other assemblies, so they do not transfer.

**(d) Not talking over someone.**
- The brake and the yield read the engaged voice, from birth.
- A late start that gets overlapped earns no unpredicted reply and no approval, so δ strengthens blocking at that age. The pooled-size input makes this transfer across questions.

### 3.8 Age without time neurons (the brief's question)

**Result** (SIMULATED, `age_toy`, 5 seeds, 600 formats → 3,000 untutored, 30 % silence):

| age code | W1 | move-on 10 % | realistic mix, care continues |
|---|---|---|---|
| r1 soft bins | 0.99 | 0.10 | 0.65 |
| **handover:** unequal fatigue + the area's competition, so the top units of the held assembly shift at log-spaced ages | **0.99** | 0.10 | **0.68** |
| literal "number of units still on" (nested sets, no competition) | **0.00** by 900, echo 0.99 | 0.00 | 0.56 |

**Reading.**
- Age needs no time module: the held assembly's own units, with unequal fatigue under the area's inhibition, carry it.
- The literal count code fails. Old ages light a subset of the young-age units, so give-ups train the units that answers use.
- **Question Q1 for the designer:** is "unequal fatigue among PFC units" acceptable? It is a distribution of one constant, not a module.

**The designer's rule in the same context** (SIMULATED, `hebb_toy`):
- stability equals the delta rule's;
- echo 0.01 against 0.33;
- corrections 2–4× slower (§0.3).

### 3.9 The maturity rule (decision #20)

- **Maturity** = 300 world-counted contingent exchanges (own utterance → an unpredicted reply), capped at 15,000 audible syllables.
- **After maturity**, these must be 0 in every tenth of life, counted inside exchanges and addressed windows:
  - talk-over starts;
  - failure to yield within one beat;
  - garble in released lines;
  - extra turns.
- Solitary vocal play does not count (D1).

---

## 4. Corrected numbers

1. **Rules as defined.**
   - Concave weights after n co-occurrences: w = √(1 + 0.2n) − 1 = 0.10 (n = 1), 0.41 (n = 5), 1.19 (n = 19), 3.6 (n = 100). PROVED for the toy constants.
   - The char-net curve (CITED) is 0.08 / 0.17 / 0.28 / 0.50 at 1 / 3 / 10 / 100 co-occurrences.
   - r1's walk numbers (0.73, 0.88, "7 events") are withdrawn.
2. **Compute.**
   - Measured by the skeptic on the r1 sizes, per A800 core: silent tick about 1.4 ms, speech tick about 2.6 ms, each replay reactivation about 4.4 ms.
   - Hence s = 1: 1,000–2,000 s per 10⁶ ticks.
   - s = 4: 4,000–8,000 s per 10⁶ ticks; with L ∝ s, **16,000–32,000 s per life** (3–6 chained jobs with checkpoint/resume).
   - GPU (gpuport, CITED): 259 ms per tick at batch 64 (250 agent-ticks per second), 2.56 ms at batch 1. **Batching is unproven**; the lineage stays on CPU.
   - Lineage: P = 12 × 10 generations × 2 lineages at 2–3 jobs per generation ≈ **30–45 h**.
   - The chain-repo engine measured 8.08 ms per step for 52 k neurons and 448 k edges (CITED), consistent with these numbers.
3. **The skeptic's hidden items, with their r2 fate:**

   | items | fate |
   |---|---|
   | the M4 source list (4 items) | deleted |
   | P1c's "last tick" patch | replaced by brake + yield |
   | P2c's person TIME pathway | deleted |
   | P3b pointing bits; P4a, b, c | moved to sensor physics |
   | W1 "fades as the child succeeds" | replaced by a fixed schedule by age |
   | W3 | declared a format |
   | W4 | politeness 0.8 |
   | W7 | declared as the world judging truth |
   | the G8 ceiling | replaced by the return-path ratio (chain-repo rule 6) |

4. **Knob count (ASSUMED estimate).**
   - About 30 genes, 15 species constants, 12 world constants and 11 lineage constants: about 70 in all.
   - That is down from about 146 live numbers in v2.2 (CITED skeptic).
   - The freeze lint gives the true count.
5. **Scaling.**
   - Counts scale × s with fixed fractions. The PFC threshold target scales × s.
   - Hippocampal connectivity scales ∝ s, so its horizon grows (M8).
   - Pre-registered: held-out error falls as s^−α along (s, L, W), with α > 0. An experience-only arm at fixed s separates "data-limited" from "doesn't scale".

---

## 5. Revised migration ladder

The representation swap comes first. The lineage comes only after the merge test.

| step | change | test | pass | fail |
|---|---|---|---|---|
| L0 | world: realistic caregivers (re-ask, move-on, politeness 0.8, jitter, care continuing), W1–W6 off by default, lineage physics seed | selftest; re-baseline v2.2 | bit-identical with the options off | – |
| L1 | thermometer ear, sound assemblies (birth-match claims), chain neurons from every auditory layer | same sound → same code; near/far similarity (ana's foundation check); bench hearing tables | near recall within 0.05 of `h_*` at σ ≤ 1 | any deaf day |
| L2 | **the designer's hippocampus**, module test (X3) | cued recall vs age on a recorded v2.2 stream | ≥ 0.9 at 50 % cue within the horizon | < 0.7: resize first |
| L3 | **merge test** (X4) | two siblings, one wiring, alignment + sign-elected merge | disjoint lessons ≥ 0.9; conflicts → no answer | disjoint < 0.7 |
| L4 | **tick benchmark** (X5) | s = 1, 4; CPU and torch at batch 1 / 16 / 64 | budget confirmed | GPU no faster: CPU only |
| L5 | the char-network PFC + M2 + handover age (X1+, symbolic) | full motor/value loop | §6 | §6 |
| L6 | motor: blocking vs sharpness; online babbling; brake + yield; learned stop | life with lawful physics | ≥ 90 % of lexicon bound by 20 % of life; 0 post-maturity agent errors | < 50 % bound |
| L7 | integrate into the bench agent; delete GEN, RAL, veto, thought list, gates, time cells | staged + life + 10,000 untutored (checkpointed), realistic world | ≥ 0.85 at 3,000 and 10,000 with care continuing; echo ≤ 0.1; J2 within SD of v2.2 on the new world | ≥ 0.15 drift on 3 of 5 seeds |
| L8 | values on sensor physics; contingent 5b; upset → blocking; the #17 tests | world2 | helps, and declines harm, including a never-tried harm | any harmful act after maturity |
| L9 | global decay by depth + slow split/delete (the designer's form; v2's negative test reconsidered) | paired lives | no loss; retention up | loss > SD |
| L10 | development freeze; lineage gen 0–2 with a Darwinian control | §2.4 of r1, fixed | per-generation curves | – |
| L11 | verify the remaining shortcuts (exact quantile, event-driven skip, scalar δ) | equivalence | within tolerance | – |
| L12 | final freeze, one sealed opening | r1 §5 protocol | pre-registered | – |

---

## 6. Revised first experiments

| # | harness | arms | prediction | falsifier | compute |
|---|---|---|---|---|---|
| **X1+** | symbolic: an r2 PFC (threshold, handover age, the designer's rule) with the full motor competition, P5 and δ, real probe timing, multi-syllable answers | O1: event-gated vs tick-level. Age: handover vs bins vs count. Caregivers: W1; re-ask 30 %; move-on 15 % with and without care; politeness 0.8. Replay: takes over vs mixes. Distractors 0.02 per tick. Inhibitory partners on/off. Correction | handover ≥ 0.9 at 3,000 (W1), ≥ 0.6 realistic; tick-level collapses; mixed replay collapses unless it takes over; echo ≤ 0.1; 0 post-maturity extra turns | handover with the threshold < 0.8 in W1, or any post-maturity extra turn | local, < 300 MB, 10–30 min |
| **X2** | (running, by another agent) v2.2 collapse attribution, #22, #23 | – | – | – | – |
| **X3** | the designer's hippocampus on a recorded v2.2 event stream (about 10⁴ events) | horizon (γ, η), k, connectivity | recall ≥ 0.9 at 50 % cue within the horizon; random kicks never reach stored patterns; cued recall does | < 0.7 within the horizon | local, minutes |
| **X4** | merge: siblings with one wiring and a fixed physics seed, sound assemblies | alignment vs none; TIES vs mean | disjoint lessons ≥ 0.9; conflicts → no answer | disjoint < 0.7 with alignment | local, minutes |
| **X5** | tick benchmark of the pure-neuron build | s = 1, 4; numpy; torch at batch 1 / 16 / 64 | CPU in the §4.2 range | GPU per agent-tick not faster than CPU: CPU only | local + 1 job |
| **X6** | depth in space vs time (learning member's E2) | as before | (d) ≥ 0.7 | (c), (d) ≤ (b) + 0.05 while backprop ≥ 0.7 | local, < 1 h |

---

## 7. Open risks, ranked

1. **Correction speed** under the designer's rule: 0.60 after 16 re-teachings, against 0.85 for delta (SIMULATED). It may slow the unlearning of childhood errors (#20). The fixes inside the spec are δ-gated blocking of disapproved outputs and the reverse-order rule; the fallback is the two-component synapse (Q2).
2. **The world must model answers and keep tutoring.** Quick move-ons with care stopped collapse every context (SIMULATED).
3. **Abstraction without backprop** is unchanged (X6).
4. **Holding with handover under a variable-count threshold** is untested beyond top-2 (X1+), as is distractor resistance (#23).
5. **The hippocampus horizon** against long-delay memory tasks. Long-term memory relies on recall-driven cortical learning, which is slow (X3).
6. **More silence instead of echo** with the designer's rule (0.16). It is better than saying partner lines, but it costs answers; urgency is gone by design.
7. **Turn-taking zero after maturity** with blocking vs sharpness is untested (L6, L7).
8. **Compute:** s = 4 lives take 3–6 jobs; GPU batching is unproven; the lineage takes 30–45 h.
9. **Merging distributed assemblies** needs alignment (X4).
10. **Values under evolution** (#17), and **crowds**.

## 8. Questions for the designer

- **Q1.** Is age = "which units of a fading assembly still win" acceptable? It uses unequal fatigue rates, not a time module. The literal count of units still on fails in the toy.
- **Q2.** Corrections are 2–4× slower under your rule, which is the "learn once, keep" profile. Accept that, or allow a fast early weight that is consolidated only at rewarded moments?
- **Q3.** Is one value signal δ, acting only on motor links, within "dopamine is a tool"?
- **Q4.** Retest inhibitory partners in your exact form for distractor resistance and wrong-answer suppression?
- **Q5.** Retest the global decay in your form (Hebbian edges) instead of v2's form (NLMS readouts)?
- **Q6.** The realistic world needs caregivers who keep tutoring (care continuing). Is that acceptable?

---

## summary_zh

1. **第二版按您自己的机制重建核心。** 只剩两个机制：
   - 一种神经元区：二值；输入用温度计成对编码；激活个数由分位数全局阈值决定，可多可少；有增益控制、疲劳和互相激活。
   - 一条学习规则：只有刚变化的神经元参与的时序赫布；越长越难长、没有上限；全脑统一按深度整体降权，出生连接不降；保留约 1% 可塑性；很慢的分裂和删除。
2. 链式神经元把每一层送到前额叶和海马体，前额叶沿回程送回感觉区。海马体是一群自己的神经元，连接长得快、掉得也快。运动区默认阻断占上风，肌肉信号足够锐才开口。
3. 第一版的时间细胞区、事件边界、不应期、"自己"专线、闸门累加器、误差修正读出层全部删掉。
4. **"过了多久"不需要时间神经元：** 刚收到的那团神经元疲劳快慢不一，在竞争下按对数时间轮流占上风；"现在是哪几个在亮"就是时间。
   - 小实验：和第一版的时间分档一样好，3,000 段无人指导后 0.99。
   - 只数"还亮着几个"不行，会崩到 0。
   - 想请您确认：神经元疲劳快慢不一，这算不算可以接受（不是时间模块）。
5. **您的学习规则在这个结构里同样稳定：** 3,000 段后 1.00；在真实的大人行为下，说"对方台词"的比例从 0.33 降到 0.01。代价是改正一个学错的答案要多 2–4 倍的教学次数，这正是"学会一次就不忘"。请您定：接受，还是加"快权重、奖励时才固化"。
6. **审查者的 5 个致命问题都用结构解决：** 阻断默认占上风；一个血统固定"物理/语言"种子；回放必须在竞争中赢过当前想法才进入前额叶；只有刚变化的神经元学习；咿呀和探索都在运动竞争里，长大后只算对话里自己造成的错误。
7. **世界要求：** 大人会重问、会跳过，但要一直持续教；如果大人很快跳过而且不再教，任何结构都会崩。
8. **诚实的数字：**
   - 一生 10⁶ 拍在 CPU 上约 1,000–2,000 秒；s=4 时一生约 4.5–9 小时，要分 3–6 个任务接力；
   - GPU 批量还没有证明更快；
   - 参数从约 146 个降到约 70 个。
9. **迁移顺序改了：** 先换成神经元编码、链式神经元、文字网络式前额叶和您的海马体，再测海马、融合和速度，最后才做代代相传。
