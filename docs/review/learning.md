(Saved by the coordinator from the reviewer's final reply; the harness blocked the reviewer's own write.)

# Whole-architecture review: learning and memory lens

Reviewer folder: `/tmp/claude-0/-root/e92d3d7d-5823-4f5c-a0ec-4fe302bff9af/scratchpad/arch/review/learning/`. Date: 2026-10-31.

**No `report.md` was written.** The harness blocked the write: subagents must return reports as text. This message is the full report. Scripts and outputs are in the folder (§8).

**Labels used throughout:**
- **PROVED**: an argument or arithmetic with numbers, with code lines or counts given.
- **SIMULATED**: a numpy run, either mine (file and numbers given) or another panel member's (file named).
- **CITED**: a paper and its finding. All 52 core citations were checked against abstracts or full text by a helper agent; corrections are applied.
- **ASSUMED**: argued but not tested.

---

## 0. Verdict

1. **The body works. The learner does not.**
   - ARCH_v1 hears, waits, speaks clean syllables and stops, with 0 errors.
   - But all learning that reaches behaviour sits in six tables. They feed one unit per character ("program").
   - Those tables learn only while a teacher pushes the mouth's muscles.
   - The brain-like learners it keeps reach no behaviour (PROVED, §1.1): ear recurrence, the hippocampus, and the motor cortex with its links to the ear.

2. **It stops learning the moment the teacher leaves** (PROVED from `archv1.py`, §1.2).
   - Without a teacher these are all frozen: what to say first, what comes next, when to stop, what to copy.
   - No new character can ever be added.
   - So real-time learning of behaviour in use is exactly zero.

3. **Even with the teacher, it stops improving after about 3k dialogues.**
   - Never-heard content: 20-way top-1 is 0.136 / 0.154 / 0.152 at 1k / 3k / 10k dialogues.
   - First-syllable memory overwrites itself: each program keeps about 5 recent contexts in the fast path and about 25 in the slow path.
   - Reply tails keep changing (85–94 %), and parroting rises with experience (SIMULATED, `arch/content/CONTENT.md`).
   - In a lifetime test (§4.1), the share of taught facts still recalled falls from 0.65 to 0.19 as the lifetime grows from 10k to 300k pairs.
   - Facts whose reply starts with a common character are gone within about 3k pairs.

4. **The differences from the brain that matter most are in the memory system, not in how many layers the PFC has:**
   - **Teaching signal.** Brains learn speech by listening, imitating and getting social feedback, not by having their muscles moved.
   - **Hippocampus.** The real one learns in one shot, keeps memories for days to weeks, and teaches the cortex by replay. Ours forgets in about 2 dialogues (≈ 15 s) and is wired to nothing that behaves.
   - **Cortex.** The real one slowly learns reusable representations, protects consolidated synapses, and assigns new memories to fresh neurons.
   - **Neuromodulators.** In the brain they decide what is learned, when, and how strongly. We have none.

5. **The three most valuable changes, in order** (§5):
   - **(1) Teacher-free learning from heard speech:** "predict what will be heard next, say what you predict", through the auditory mirror Z that already exists.
   - **(2) Memory that accumulates:** a readout unit that has filled up matures, and a fresh one takes over (the designer's own "split" idea). Add a hippocampal index on the content path, with long retention and offline replay.
   - **(3) A slowly maturing learned representation layer.** This is the real PFC / association-cortex upgrade.

6. **Two checks on the real recorded thought codes** (SIMULATED, §4.2–4.3):
   - **Local growth** roughly doubles recall at +3–10k dialogues after the last exposure: from 0.41–0.50 to 0.83–0.88. Learning speed is the same; it needs about 19 % more units.
   - **Learning from what the ear detects** in the heard reply, instead of from the teacher, costs about one exposure.

---

## 1. What learns in ARCH_v1, and does it reach behaviour?

### 1.1 Inventory

PROVED by reading `arch/synth/archv1.py` and `echo_slot.py`. Constants are from ARCH_v1 §1.3–1.4.

| learned thing | constants | reaches behaviour? | what it holds; timescale |
|---|---|---|---|
| ear→ear (k 96) | η 0.03·x, budget 1.0, γ 0.999 | **no**. F reads the input window. T and Z read ear layer 0, which gets no learned or return current (`ein` is masked to layers ≥ 1). | sound-feature transitions; τ ≈ 1.2k dialogues |
| mctx→mctx, ear↔mctx | η 0.03 / 0.01, budget 1 / 0.5 | **no**. mctx has no output to the motor area. | — |
| hip→hip (k 96) | η 0.5, budget 2.0, γ 0.7 | **no**. Hippocampal relays return only to ear/mctx cortex, which nothing on a behaviour path reads. | half-life 194 experienced ticks ≈ 2.3 dialogues ≈ 15 s; about 4 events per target |
| P→Y fast (first syllable) | 0.1 per event, budget 0.5, γ 0.999 | yes | last ≈ 5 contexts per program |
| P→Y slow | 0.06 per event, budget 1.5, γ 1 | yes | last ≈ 25 contexts per program (interference only) |
| Hc→Y (next syllable) | η 0.001, budget 0.2 | yes | context-free, recency-weighted bigram |
| Hc→END | η 0.001, no budget | yes | "this syllable ends replies" |
| FEAT0→Z (auditory mirror) | η 0.005, budget 2.0, gated by own voice | yes (copy route) | sound template of each character the model has said |
| E→Y, diagonal (copy gain) | η 0.003, cap 0.15 | yes | decayed count of "heard c, said c" |
| T set point | τ 1,000 ticks | yes (T code) | homeostasis |
| split/delete in the ear | every 1,000 dialogues | no (F and T are immune by design) | — |

The parts meant to carry long-term learning do not touch behaviour: cortex learning sequences, the hippocampus, and the links between areas. The parts that do touch behaviour are lookup tables keyed on T, a **fixed** random code of the earliest auditory layer. This was a deliberate v1 trade, made after the learning PFC loop drifted: it bought reliability for the body. It is the opposite of what the final goal needs.

### 1.2 What still learns without a teacher (PROVED, `archv1.py`)

- **Decision paths are gated.** They update only when the target had external drive: `g_now = (not c.gate) or Yext`, with `Yext = cue is not None and cue_ov[best] == 4`. With no cue, P→Y fast/slow, Hc→Y, Hc→END and E→Y get no forward increment.
- **Program recruitment needs a cue.** It happens only inside `if cue is not None:`, so the set of things the model can say is frozen in use.
- **Z only refines known characters.** It learns only while ΣCD > 0, taught by the program's own delay line after the model has said c.
- **The gate is logically required in the current design.** A free release is the argmax of the very weights being trained, so learning from it is pure positive feedback; the minimal-learning verifier measured a drift 你 → 的/了 (PROVED; `arch/verify_minimal_learning/report.md` §3).

So the gate is not the bug. The structural problem is that **there is no teaching signal from the world** (P1).

### 1.3 It plateaus (SIMULATED by the panel)

| measure | 1k | 3k | 10k dialogues | source |
|---|---|---|---|---|
| never-heard 20-way top-1, copy route (MAIN) | 0.136 | 0.154 | 0.152 | CONTENT §0, 3 seeds |
| never-heard, no copy route | 0.057 | 0.054 | 0.058 (chance 0.05) | CONTENT §0 |
| parroting (share of the heard line's characters repeated; humans 0.15), copy γ 0.9999 | 0.53 | 0.66 | 0.72 | CONTENT §3.2 |
| whole reply changed on fixed lines, relative to the 1k model | — | 85–94 % between 1.5k and 10k | — | CONTENT §3.3 |
| familiar first syllable at +3–6k / +10k after the last exposure | — | 0.46–0.50 / 0.39–0.43 | — | CONTENT §4.2; reproduced in §4.2 here |
| whole familiar reply, 20-way top-1, after 5 exposures | — | saturates ≈ 0.45 | — | CONTENT §4.3 |

**Corpus fact** (PROVED, `corpus_facts.out`):
- In corpus order, only **8.1 %** of heard lines had been heard verbatim before.
- In only **1.3 %** of pairs does the reply start with a first character already given to that line.
- So verbatim line → reply memory can add at most about 1 % correct first syllables in natural conversation.
- "Familiar recall" is a lab test. What must accumulate over a lifetime is **reusable structure**: words, phrases, response patterns.

---

## 2. Differences from the real brain, and whether each matters

"Matters" means it blocks or limits the final goal. "Harmless" means the simplification can stay.

### 2.1 Structure

**Where learned representations live**
- **Brain:** behaviour runs on learned cortical codes (feature hierarchies, word forms, semantics). Plasticity is everywhere, at different speeds.
- **ARCH_v1:** behaviour reads a **fixed** T, a recency-weighted bag of sounds. The learning areas are off-path (§1.1).
- **Matters most. Fundamental.**
  - No code for meaning can form, so new lines get echo at best.
  - Memories are stored as verbatim contexts, which wastes capacity (§4.1).

**Hippocampus**
- **Brain (CITED):**
  - About 1–2 % of dentate granule cells are active (Arc in ≈ 2 % after exploration; Ramirez-Amaya 2013; Leutgeb 2007).
  - CA3 autoassociation and sequences; it acts as an index into cortex (Teyler & DiScenna 1986).
  - It learns in one shot: a single plateau in a single trial makes a CA1 place field, with a plasticity kernel of about −3…−4 s to +2…+3 s (Bittner 2017).
  - Memories last days to weeks or longer. Dentate LTP is stable for months, up to at least 1 year in vivo (Abraham 2002).
  - Hippocampal lesions 1 day after contextual fear conditioning abolish the memory; lesions at 28 days spare it (Kim & Fanselow 1992).
  - Patient H.M. kept his immediate memory but lost new long-term memory (Scoville & Milner 1957).
  - Replay: hippocampal and cortical replay during sleep are coordinated (Wilson & McNaughton 1994; Ji & Wilson 2007). Suppressing post-training ripples impairs memory (Girardeau 2009); prolonging awake ripples improves it (Fernández-Ruiz 2019).
- **ARCH_v1:**
  - A 1:1 mirror, 5 % active, half-life about 2.3 dialogues.
  - Little pattern separation: same-line Jaccard 0.18–0.19 vs 0.14 between different lines (`res_*_s0.json`).
  - **Wired to no behaviour path.**
- **Matters. Fundamental (wiring and timescale).**
  - There is no persistent one-shot memory, and nothing that could teach a slow learner.
  - The designer's "hippocampus = short-term memory, fast forgetting" conflicts with the evidence: it is a fast-learning *long-term* store. Seconds-scale holding is done by PFC and sensory activity.

**PFC** (the designer's concern)
- **Brain (CITED):**
  - A multi-layer recurrent network that holds information through attractors and short-term synaptic facilitation (τF ≈ 1.5 s, refreshed by brief population spikes; Mongillo 2008).
  - What it holds and when it updates is gated by basal ganglia, trained by dopamine (PBWM; O'Reilly & Frank 2006).
  - It learns task structure (meta-RL; Wang 2018) and matures last, at the end of adolescence (Gogtay 2004).
- **ARCH_v1:** T is a fixed projection with a one-tick self-hold and a freeze at L+3; its output gate is P = T ∧ GO. The learning loop was removed after it drifted and alternated with period 2.
- **Matters. Fundamental.** The binding constraint is learning, not layer count (§6).

**Basal ganglia**
- **Brain:** action channels; dopamine-dependent three-factor plasticity; chunking of action sequences (Graybiel 1998); start/stop signals for chunks (Jin & Costa 2010).
- **ARCH_v1:** the per-beat winner-take-all over programs is a good stand-in for striatal selection. But its weights are learned only under the teacher's cue, there is no dopamine-like signal, and there are no chunks.
- **Matters.** No learning from outcomes (structural). No multi-syllable units (fixable).

**Cerebellum**
- **Brain:** an error-driven forward model. It predicts the sensory results of one's own actions and handles timing. Feedforward speech commands are tuned from auditory error (DIVA; Tourville & Guenther 2011).
- **ARCH_v1:** absent. The corollary discharge is a fixed 4-tick veto, and programs are exact by construction.
- **Harmless now, matters later.** It will be needed for yielding mid-turn, for imitating new sounds, and for cancelling the predicted own voice. Fixable.

**Thalamus**
- **Brain:** relay and gating; it sustains PFC activity.
- **ARCH_v1:** removed. The F, GO and R clocks replace it.
- **Harmless for now.** It returns once the PFC holds learned content.

**Neuromodulatory nuclei**
- **Brain (CITED):**
  - Broadcast signals decide whether, how much and in which direction a Hebbian trace is expressed.
  - Dopamine signals reward-prediction error (Schultz 1997) and song performance error (Gadagkar 2016).
  - ACh: high in waking, it suppresses feedback (encoding); low in slow-wave sleep, it allows consolidation (Hasselmo 2006).
  - ACh signals expected uncertainty and NE unexpected uncertainty (Yu & Dayan 2005).
  - Novelty gates entry into long-term memory (Lisman & Grace 2005; locus coeruleus via D1/D5 receptors, Takeuchi 2016).
- **ARCH_v1:** one local gate, "the target had teacher drive".
- **Matters. Fundamental.** Nothing can be learned from consequences or from novelty.

**Cell types and compartments**
- **Brain (CITED):**
  - Many interneuron classes.
  - Inhibitory plasticity learns a "negative image" that balances excitation (Vogels 2011).
  - Motor input to auditory cortex learns to suppress the sound of one's own movement (Schneider 2018).
  - Two-compartment cells allow a local prediction-error rule (Urbanczik & Senn 2014).
- **ARCH_v1:** binary point units; k-WTA instead of interneurons; no learned inhibition.
- **k-WTA is harmless.** The missing error term and missing negative images matter moderately. Fixable.

**Entorhinal context**
- **Brain:** slowly drifting context lets the hippocampus tell episodes apart.
- **ARCH_v1:** none. Review 03 found it necessary.
- **Matters once the hippocampus is on the content path.** Fixable.

### 2.2 Dynamics

**Learning window**
- **Brain:** STDP windows are about ±20 ms. Seconds-scale learning needs an eligibility trace plus a third factor, or BTSP (CITED: Gerstner 2018; Bittner 2017).
- **ARCH_v1:** 75 ms synchronous ticks. A synapse learns only "A on at t−1, B rising at t"; a source held through the target's onset cancels out (the timing law).
- **Matters. Fundamental for generality.**
  - Every association needs a hand-placed one-tick pulse (P = T∧GO, E = L∧R2, the Hc delay line).
  - So each new behaviour needs a designer-placed gate.
  - Nothing separated by more than one tick can be learned, and no delayed outcome can be learned.

**Turning held states into pulses**
- **Brain (CITED):** theta sequences and replay compress behavioural sequences into single theta cycles, up to 10:1 (Skaggs 1996; O'Keefe & Recce 1993; Foster & Wilson 2007). This is the brain's way of turning slow states into ordered bursts that a timing rule can learn.
- **ARCH_v1:**
  - Pulse readout is used only locally (R2).
  - Replay exists only as uncontrolled activity of the off-path hippocampus in silence: 1,000 idle ticks change hippocampal weights by up to 6.8–6.9 and ear weights by up to 0.8–0.9 (`res_*_s0.json`).
- **Matters.** A generic pulse clock removes the need for per-path gates (PROVED condition, §4.5). Fixable.

**Activity fractions**
- **Brain:** variable in sensory areas, roughly fixed in PFC.
- **ARCH_v1:** fixed top-k.
- **Mostly harmless.** Covered in `00_synthesis.md`.

### 2.3 Learning rules and neuromodulators

**Third factors**
- **Brain (CITED):**
  - Dopamine enlarges spines only when it arrives 0.3–2 s after pairing (Yagishita 2014).
  - Noradrenaline converts an LTP trace that halves by 5 s and is gone by 10 s; serotonin converts an LTD trace gone by 5 s (He 2015).
- **ARCH_v1:** pairing plus a teacher-drive gate; no traces.
- **Matters (P1, P5).** No reinforcement learning is possible.

**Depression below baseline**
- **Brain (CITED):** LTD below baseline exists. Moderate depolarisation without enough drive gives LTD (Artola, Bröcher & Singer 1990), and dendritic-prediction rules act as local delta rules.
- **ARCH_v1:** weights never fall below their birth value, so "predicted but absent" cannot be learned.
- **Matters (P6). Fixable.**
  - Weights are counts, not probabilities: copy gains are frequency-weighted (CONTENT §3.4).
  - Priors dominate.
  - Predictable events use up budget.

**Synaptic consolidation and multiple timescales**
- **Brain (CITED):**
  - Early LTP fades unless captured by a synaptic tag, which decays in < 3 h (Frey & Morris 1997).
  - About 4–5 % of new spines persist for 3–5 months (Yang 2009), and erasing potentiated spines erases the motor memory (Hayashi-Takagi 2015).
  - Bounded single-rate synapses are palimpsests: capacity can be as low as log N (Amit & Fusi 1994), with exponential forgetting (Fusi, Drew & Abbott 2005).
  - Coupled multi-timescale synapses give power-law forgetting and capacity almost linear in N, compared with √N in earlier models (Benna & Fusi 2016).
- **ARCH_v1:** one γ per path, plus two parallel P→Y paths. Per-target budgets make each program a fixed-size palimpsest.
- **Matters for lifetime capacity (§4.1).** Partly built; the per-target budget is the binding limit.

**Normalisation and homeostasis**
- **Brain:** fast per-dendrite heterosynaptic normalisation; slow scaling.
- **ARCH_v1:** per-target budget applied every 100 ticks; a set point on T.
- **Right idea. Fixable.** The hippocampus overshoots its budget of 2 to about 13–15 between normalisations (ARCH_v1 §6.10; `hip_max_target` 14.4–15.2).

**Memory allocation**
- **Brain (CITED):**
  - Neurons that are more excitable at training are recruited into the memory trace (Han 2007; Yiu 2014).
  - Young adult-born dentate cells are more excitable and plastic (Schmidt-Hieber 2004; Ge 2007). Aimone 2011 argues they *reduce* pattern separation in the classical sense.
  - Adult neurogenesis in humans is unresolved (Sorrells 2018 vs Boldrini 2018 and Moreno-Jiménez 2019).
- **ARCH_v1:** only whole programs are recruited, and only at teacher cues. Splitting exists only in the ear.
- **Matters. Fixable.** The designer's split/generate mechanism is the natural cure for saturation if it is applied on the content path (§4.2).

### 2.4 Memory systems, consolidation, sleep

**Complementary learning systems**
- **Brain (CITED):**
  - The hippocampus learns fast; the cortex extracts structure only when learning is gradual and interleaved (McClelland 1995).
  - With a schema in place, the cortex can learn fast (Kumaran 2016). New pairs learned in one trial became hippocampus-independent within about 48 h (Tse 2007).
  - Inside the hippocampus, the monosynaptic pathway can learn statistics while the trisynaptic pathway stores episodes (Schapiro 2017).
- **ARCH_v1:** fast and slow P→Y both learn online from the same gated event. The hippocampus feeds neither, and there is no offline phase.
- **Matters. Fundamental.**
  - No transfer from fast to slow memory.
  - No rehearsal of old memories against new ones.
  - No extraction of structure, so no schema speed-up is possible.

**Sleep**
- **Brain (CITED):**
  - Coordinated replay during sleep.
  - Synapse size falls about 18 % after sleep, in proportion to size. The scaling is selective: large synapses lacking recycling endosomes are spared (de Vivo 2017; Tononi & Cirelli 2014; Klinzing 2019).
- **ARCH_v1:**
  - No sleep.
  - Global decay is the designer's only clean-up.
  - The experience clock means idle time forgets nothing, which is good.
- **Matters.** Replay multiplies experience and protects old memories. Fixable (a protocol phase).

**Forgetting curve**
- **Brain (CITED):** forgetting follows a power function (Wixted & Ebbesen 1991).
- **ARCH_v1:** exponential, τ ≈ 1.2k dialogues on the fast paths; interference only on the slow path.
- **Matters over long lifetimes** (§4.1).

### 2.5 Innate vs learned, development

**What is innate**
- **Brain:** generic machinery: basal ganglia and thalamic loops, oscillators, the dopamine system. Turn timing, speech motor control and vocabulary are learned over years. Proto-conversations appear in the first half-year (Levinson 2016).
- **ARCH_v1:** about 20 task-specific hand-wired mechanisms (F, F2, GO, CD1–4, R1/R2, beat, boundary rule, program assembly, END rule, breath counter, used flag, T/P, Hc, Z clamp, latch, E), with six learned tables.
- **Fine for a first body. Fundamental later.** For AGI each new skill would need a new hand-made gate. Innate parts should be generic: pulse clock, learned gating, recruitment.

**Development**
- **Brain (CITED):**
  - Development is staged: perception tunes first, and non-native contrasts are lost by 10–12 months (Werker & Tees 1984).
  - Babbling comes next, then words.
  - Critical periods start with PV interneuron maturation (Hensch 2005).
  - The PFC matures last (Gogtay 2004).
- **ARCH_v1:** no stages. Warm-up muting, and "near the input splits less", are static versions of a critical period.
- **Matters, because of drift.** Upstream learning makes every downstream readout stale; that is what killed the learning PFC. Per-unit maturation is the brain's answer (§5, O3).

### 2.6 Body, world, training regime

**How speech is learned**
- **Brain (CITED):**
  - 8-month-olds segment words after 2 minutes of listening (Saffran 1996).
  - 9-month-olds learned Mandarin contrasts from live speakers; recordings had no effect (Kuhl 2003).
  - Contingent maternal responses make babbling more mature (Goldstein 2003).
  - A perceptual phase guides later sensorimotor practice (Doupe & Kuhl 1999).
  - No one moves the infant's mouth.
- **ARCH_v1:** the teacher pushes each syllable's muscles and clamps END. The decision tables learn only then.
- **Matters most (P1).** The world gives no muscle cues.

**Input volume and grounding**
- **Brain (CITED):** about 10–20 M words heard by age 3–4 (LENA: ≈ 12,300 adult words per 12-h day, Gilkerson 2017; Hart & Risley 1995 give 13–45 M by age 4, extrapolated and contested). Input is multimodal and grounded.
- **ARCH_v1:** 0.64 M characters (≈ 0.4 M words, ASSUMED 1.5 characters per word) ≈ 2–4 % of that, in one voice with no grounding.
- **Data matters for meaning.**
  - Even ideal surface learners reach only top-1 0.21–0.26 on never-heard lines (`arch/verify_brain_learning/report.md`, ceiling table).
  - Grounding matters later.

**Who is heard**
- **Brain:** many speakers, including overheard conversation.
- **ARCH_v1:** exactly two voices; the model never overhears others.
- **Matters. Fixable (protocol).** Overheard pairs are free training data (O1).

---

## 3. Problems of the whole architecture, ranked by how much they block the final goal

**P1. No teaching signal from the world (fundamental).**
- Behaviour learning in use is zero, and the vocabulary is closed (§1.2, PROVED).
- Free-speech learning would be pure positive feedback, as the minimal verifier showed (PROVED).
- So the fix must be a *new signal*, not removing the gate.

**P2. Nothing learns a representation on the content path (fundamental).**
- T is fixed, and the learning areas are off-path (PROVED).
- Consequences: no meaning, generalisation is only echo, and never-heard performance is flat with experience (SIMULATED, CONTENT §0).

**P3. Memory does not accumulate (fundamental for lifelong learning; fixable by growth).**
- Each program is a palimpsest of about 5 (fast) or 25 (slow) contexts.
- Programs with frequent first characters forget fastest. 你 is 9.4 % of first syllables, so its slow window is 265 pairs ≈ 157 dialogues and its fast window 53 pairs ≈ 31 dialogues (PROVED).
- The share of taught facts still retained falls with lifetime (§4.1, SIMULATED).

**P4. The hippocampus is decorative and forgets in about 15 s (fundamental: wiring and timescale).**
- Half-life ≈ 2.3 dialogues (PROVED).
- Little pattern separation: Jaccard 0.18 vs 0.14 (SIMULATED).
- No route to the decision tables, no replay-based consolidation, and no offline phase.

**P5. The timing law, and no eligibility traces or third factors (fundamental for generality).**
- Only one-tick pulse sources can teach (PROVED, §4.5), so every behaviour needs a hand-placed gate.
- Delayed consequences cannot be learned, so there is no RL.
- The PFC loop never learned (think→think ≈ 0.0001; think→go mean 0.0045; STATE #7).

**P6. Counts, not probabilities (fixable by a local rule term; needs the designer's consent).**
- Copy gains are frequency counts (CONTENT §3.4).
- Priors dominate.
- More experience means more parroting: 0.53 → 0.72 (SIMULATED, CONTENT §3.2).

**P7. Drift breaks readouts (fundamental; must be designed together with P2).**
- Any learning added to T or the PFC invalidates the memories keyed on it.
  - When the plan layers learned, plan decoding fell 0.95 → 0.70 in 200 dialogues (BL-S3, SIMULATED).
  - Ear split/delete drifts codes by about −0.03 per 1k dialogues (`arch/verify_brain_dynamics/report.md` M3, SIMULATED).
- There is no maturation mechanism and no replay re-alignment.

**P8. Only the first syllable depends on what was heard (structural, locally fixable).**
- Later syllables come from a context-free bigram plus the copy route.
- Whole-reply memory saturates at about 0.45, and reply tails keep changing (85–94 %) (SIMULATED, CONTENT).
- The fix needs a line-specific source at every boundary.

**P9. Localist output with no chunking (fixable).**
- No words or phrases exist as units.
- Programs are recruited only on teacher cues (PROVED).

**P10. About 20 task-specific hand-wired gates (fixable now; fundamental for AGI).**
- Turn-taking is rigid. Pauses of 5–9 ticks inside the partner's turn lead to talking over them: 29–75 mouth ticks per 6 trials (SIMULATED, `res_*_s0.json`).
- Latency is 6 ticks ≈ 450 ms. Human gaps are typically about 200 ms, and turns rarely overlap (CITED: Stivers 2009; Levinson 2016).

**P11. The data regime (not architectural; decisive for meaning).**
- The whole corpus is about 2–4 % of a 3–4-year-old's input and about 58 h of conversation at 75 ms per tick (PROVED arithmetic plus CITED).

**P12. Uncontrolled offline plasticity, and budgets enforced late (fixable).**
- Idle replay changes ear and hippocampal weights for no purpose.
- The hippocampus overshoots its budget by about 7× (SIMULATED).

**P13. Growth costs (engineering).**
- Unit count grows linearly with stored episodes unless storage is novelty-gated (§4.2, SIMULATED).

---

## 4. Lifelong-learning evidence

### 4.1 S1: lifetime test on synthetic codes

Files: `lifelong_capacity.py` and `lifelong_capacity2.py` → `lifelong_capacity_s0.out`, `lifelong_capacity_s1.out`, `lifelong_capacity_V1_300k.out`. SIMULATED.

**Setup:**
- First-syllable memory exactly as ARCH_v1's MEM form, i.e. the ÷10-scaled form of the content route (P→Y fast and slow, concave increment, decay and budget per 100 ticks ≈ 2 pairs).
- Context codes are random 200-of-4000 sets. A probe is the same line re-heard with 10 % of units changed (Jaccard ≈ 0.82). Random codes overlap less than real ones, so this is an **upper bound**.
- Targets follow the corpus distribution of reply first characters (2,082 characters).
- Facts are taught 6× each, 100 events apart, with one new fact every 50 events, inside a stream of never-repeated pairs.
- **V0** = ARCH_v1.
- **V1** = V0 plus local growth: when a program's slow unit reaches its budget, it matures (stops learning) and a fresh sub-unit takes over. Readout = fast path + the best of the program's sub-units, i.e. winner-take-all among them.

| lifetime (pairs) | facts taught | V0: share recalled at end | V0: facts with a top-10 first character | V0: other facts | V1: share recalled | V1 slow units |
|---|---|---|---|---|---|---|
| 10k | 190 | 0.65 | 0.20 | 0.96 | 1.00 | 2,330 |
| 30k | 590 | 0.50 | 0.07 | 0.77 | 1.00 | 2,999 |
| 100k | 1,990 | 0.32 (seed 1: 0.33) | 0.02 | 0.49 | 1.00 (seed 1: 1.00) | 5,696 (5,682) |
| 300k | 5,990 | 0.19 | 0.02 | 0.29 | 1.00 (1,496-fact sample) | 13,875 |

- **V0 at 300k, by time since last exposure:** 0.89 (< 1k), 0.65 (1–3k), 0.61 (3–10k), 0.43 (10–30k), 0.25 (30–100k), 0.12 (> 100k).
- **Reading:**
  - The absolute number of facts recalled still creeps up (123 → 1,139), because rare programs have spare room. The share falls toward zero.
  - About 37 % of all facts start with a top-10 character, and those are lost within about 3k pairs.
  - Retention depends on how common the *output* is, not on how important or how repeated the fact is. That is backwards for lifelong learning.
  - Growth fixes it here, at about 40 new units per 1k pairs. Upper bound only, because the codes are random.

### 4.2 S1b: the same comparison on the real recorded thought codes

Files: `realT_growth.py` and `realT_growth2.py` → `realT_growth_s0.out`, `realT_growth_s1.out`, `realT_growth2_s0.out`. SIMULATED.

**Setup:**
- Replays the content specialist's `arch/content/fe_probe_s{0,1}.pkl` read-only: real ear layer 0, the real T at L+5, and 200 probe pairs × 6 exposures inside 14.6k corpus dialogues (22k pairs).
- Uses the same clock as `decide.py`; the readout is the memory paths only.
- Columns: before the k-th exposure (exp1–exp5), then time since the last exposure (+…). Cells give seed 0 / seed 1.

| | exp1 | exp2 | exp3 | exp4 | exp5 | +<1k | +1–3k | +3–6k | +10k | slow units |
|---|---|---|---|---|---|---|---|---|---|---|
| V0 (ARCH_v1 MEM) | 0.38/0.32 | 0.72/0.72 | 0.84/0.84 | 0.90/0.89 | 0.93/0.92 | 0.79/0.76 | 0.58/0.57 | 0.50/0.46 | 0.41/0.43 | 3,054 |
| V1 (local growth) | 0.33/0.26 | 0.72/0.69 | 0.84/0.82 | 0.92/0.91 | 0.93/0.91 | **0.92/0.94** | **0.85/0.90** | **0.83/0.88** | **0.84/0.83** | 3,630/3,641 |

- **V0 reproduces CONTENT §4.2** "P05 + Tslow ×3" (0.38/0.72/0.84/0.90/0.93; 0.79/0.58/0.50), which validates the replay.
- **V1 roughly doubles long-term retention.** Learning speed is unchanged, apart from a slightly lower one-exposure recall.
- **Cost:** about 19 % more units over 14.6k dialogues, ≈ 26 per 1k pairs.
- **Never-heard lines are not inflated:** median top memory score 0.217 (V0) vs 0.222 (V1). Familiar lines at +3–6k score 0.343 vs 0.391. So V1 should not steal the first boundary from the copy route on new lines.
- This is the designer's own "split the always-on" mechanism, applied where saturation happens: on the readout of a full program.

### 4.3 S1c: teacher-free targets

File: `realT_observe.py` → `realT_observe_s0.out`. SIMULATED, seed 0.

**Setup:** the same replay, but the target is what the auditory mirror would detect as the first character of the heard reply. It is correct with probability 0.78 (measured latch recall 0.76–0.81), a wrong program with probability 0.05, otherwise nothing.

| | exp1 | exp2 | exp3 | exp4 | exp5 | +<1k | +1–3k | +3–6k | +10k |
|---|---|---|---|---|---|---|---|---|---|
| V0 teacher-forced | 0.38 | 0.72 | 0.84 | 0.90 | 0.93 | 0.79 | 0.58 | 0.50 | 0.41 |
| V0 from heard reply | 0.30 | 0.58 | 0.77 | 0.90 | 0.91 | 0.77 | 0.56 | 0.45 | 0.36 |
| V1 from heard reply | 0.24 | 0.57 | 0.74 | 0.87 | 0.93 | 0.90 | 0.77 | 0.75 | 0.77 |

- Teacher-free targets cost about one exposure and 2–8 points of retention.
- This assumes the context code is the one for the line heard from another voice. That holds for overheard pairs.
- For "partner answers the model" pairs, the context is the model's own voice. Own vs other voice gives layer-0 Jaccard 0.89 (`arch/verify_learnability_learning/report.md` T12), so transfer is ASSUMED.

### 4.4 Arithmetic (PROVED)

**Clock:** 84,889 ticks per 1,000 dialogues; 1.69 pairs per dialogue (`res_mem_s0.log`).

**Hippocampus:** γ = 0.7 per 100 experienced ticks gives a half-life of ln 0.5 / ln 0.7 = 1.94 steps:
- = 194 ticks ≈ 2.3 dialogues ≈ 15 s at 75 ms per tick;
- the real hippocampus holds new memories for days to weeks (CITED §2.1), which is 10⁴–10⁵× longer.

**Fast P→Y:** τ = 10⁵ ticks ≈ 1,180 dialogues ≈ 2 h of experience.

**Per-program memory windows:**

| first character | share of first syllables | fast window (5 events) | slow window (25 events) |
|---|---|---|---|
| 你 | 0.094 | 53 pairs ≈ 31 dialogues | 265 pairs ≈ 157 dialogues |
| 我 | 0.081 | 62 pairs ≈ 37 dialogues | 309 pairs ≈ 183 dialogues |
| 不 | 0.035 | 143 pairs ≈ 85 dialogues | 716 pairs ≈ 423 dialogues |

By 10k pairs, 60 programs covering 62 % of first syllables are already overwriting themselves (`corpus_facts.out`).

**Pairs available:** 47,786 taught pairs, plus **20,113 pairs in which the partner answers the model's line** (+42 %). Those are usable without a teacher.

**Sparse index capacity:** for an 8,000-unit layer with 40 active, the Willshaw-type binary capacity at half-filled synapses is ln2 · N²/K² ≈ 2.8×10⁴ episodes. This is an order of magnitude only; finite networks reach less (CITED Willshaw 1969).

### 4.5 The timing law has one generic fix (PROVED; checked in `clock_check.py` → `clock_check.out`)

**Condition:**
- Take a held source X read through a periodic one-tick clock of period c (source = X ∧ clock).
- Its back-term hits on a target held for h ticks fall at t ∈ [t0+1, t0+h], and the next clock pulse comes at t0−1+c.
- So the net change is +η per event iff **c ≥ h+2**.

**Exact-rule check, 50 events:**
- h = 1: held source 0.000; c = 2 gives 0.000; c = 3–6 gives 0.292.
- h = 3: c = 2–4 gives 0.000; c = 5–6 gives 0.292.

**Meaning:** R2 is exactly this clock for one-tick program pulses. One generic "pulse readout" structure can replace the per-path gates (P = T∧GO, E = L∧R2) for every held source.

---

## 5. What to optimise, in priority order

### 5.1 Block diagram (v2, learning-centred; the body is kept as-is)

```
 HEARD WORLD: partner · overheard pairs · own voice (tagged by corollary discharge)
      │
      ▼
 EAR 3D net (layer 0 fixed; upper layers mature early) ── FEAT0 ──► Z: auditory mirror (heard c ⇄ program c)
      │ learned recurrence = auditory sequence prediction           taught by babbling / own voice; new
      ▼                                                             perceptual units for recurring unknown sounds
 W: chunk/word layer (NEW): one rule + k-WTA + set point; per-unit maturation; fresh units for unexplained input
      ▼
 T′: PFC context = held code of FEAT0 + W (fast hold as in v1; later a learned, BG-like gate)
      │ read only through a generic pulse clock R (period ≥ h+2)
      ├──► NEXT: T′ → Ẑ "what will be heard next"; learned from every heard next line (target = Z driven by sound)
      ├──► H: hippocampal index (sparse 1–2 %), one-shot T′(×position) → reply program sequence; keeps
      │       10³–10⁴ dialogues; novelty-gated; replays in OFFLINE phases as one-tick pulse sequences
      └──► SLOW readouts (P→Y, chain): learn awake + from H replay; local growth (full units mature, fresh ones join)
                  │                 │                 │
 TURN machinery (F/GO/CD/beat, kept) ─► PROGRAMS Y (WTA per beat) ─► D/I/M ─► mouth ─► own voice ─► ear
                  ▲ inputs at each boundary: Ẑ (via the Z_c ≡ Y_c assembly), H recall, slow readouts, copy route
 Broadcasts: N = novelty/surprise (raises η; decides what H stores) · R = outcome (later; converts eligibility traces)
 OFFLINE phase: H replay → slow readouts and W; downscale fast components only (consolidated weights spared)
```

### 5.2 What learns, from which signal, how fast

| block | learns | teaching signal | speed / forgetting | status |
|---|---|---|---|---|
| Z | sound template per program; new perceptual units | own voice (babbling); recurring unknown sounds | medium | kept; adds recruitment from hearing |
| NEXT (T′→Ẑ) | what is heard next after this context | others' speech: Z fires because of real sound, gated by F | fast + slow paths, with growth | new (same table form as P→Y) |
| H | one episode: context → whole reply sequence | heard or said reply, novelty-gated | one-shot; τ 10³–10⁴ dialogues | changed from a mirror to an index, wired to output |
| slow readouts | consolidated context → program / next syllable | awake events + H replay | slow, no decay, local growth | kept + growth |
| W | chunks and words | the order rule on the auditory stream (transitional statistics) | slow; units mature | new |
| decision gates | choices that depend on outcome | R through eligibility traces | — | new, later |

### 5.3 Changes, in order: what each buys, costs, and how to check it before any long run

**O1. Teacher-free learning from heard speech** (fixes P1; highest priority).

- **What it is:**
  - Learn T′ → Ẑ from every line the model hears from others. The source is T′ read by a pulse 2 ticks after the next line's onset; the target is Z of the first heard character.
  - Learn the chain Zc_a → Z_b (a late copy of Z feeding the next Z) from heard character sequences.
  - At a boundary, a program's score includes the prediction current into its own Z_c.
  - Gate all of this with F, so the target is never the model's own voice. F is 0 during the own turn in 10,146/10,146 turns (PROVED), so there is no self-reinforcing loop.
- **Teacher's new role:** keep its muscle cues only for bootstrapping articulation, like a caregiver modelling sounds. In this toy world the muscle → sound map is arbitrary, so pure babbling search is slow. A random 4-set shares ≥ 3 muscles with a given character only about 0.3 % of the time (PROVED: 113/35,960) (ASSUMED trade-off).
- **Buys:**
  - Real-time learning in use.
  - +42 % more pairs right away (partner answers the model), plus any overheard speech.
  - Matches CITED evidence that comprehension relies on prediction (Heilbron 2022), and that production and comprehension share forward models (Pickering & Garrod 2013).
- **Costs:**
  - One dense table of the P→Y size (T′ 10k × 4k programs ≈ 160 MB in fp32).
  - An onset-locked pulse and a 2-tick delay line.
  - A fixed Z_c ≡ Y_c assembly link.
  - Recruitment of new perceptual units from hearing.
- **Checks:**
  - (a) After 1k teacher dialogues, switch the teacher off. Present 40 new probe pairs 6× as overheard pairs or as "partner answers model" pairs. First-syllable recall must reach ≥ 0.7 after 5 exposures. Today it is 0 by construction; S1c gives 0.91.
  - (b) Run 2k dialogues of free speech with learning on. The share of the top-5 first characters and output entropy must change by < 10 %, i.e. no 你→的/了 drift.
  - (c) The prediction Ẑ on held-out next lines must beat the unigram baseline.

**O2. Memory that accumulates** (fixes P3, P4, P8, P12).

- **What it is:**
  - (a) **Local growth of slow readout units.** A unit that reaches its budget matures and a fresh sub-unit is recruited. To decide which sub-unit learns, use one of two mechanisms (both ASSUMED): encoding-mode excitability for the youngest unit, applied only while a target is driven (like ACh during encoding), or an assembly-level teaching signal.
  - (b) **The hippocampus becomes a sparse index on the content path.**
    - About 1–2 % active (the content specialist's 8000/40 S code already exists).
    - One-shot binding of context × position to the reply's program pulses.
    - Retention τ ≈ 10³–10⁴ dialogues (γ ≈ 0.9988–0.99988 per 100 ticks), with novelty-gated storage.
    - This supplies the "line-specific source at every boundary" that CONTENT §4.3 says whole-reply memory needs.
  - (c) **An offline phase** every few hundred dialogues.
    - H replays its stored episodes as pulse sequences into the slow readouts; the timing law is satisfied by §4.5.
    - Priority goes to novel episodes the slow paths do not yet predict.
    - Downscale fast components only.
- **Buys:**
  - S1b: retention at +3–10k of 0.83–0.88 instead of 0.41–0.50.
  - One-shot whole-reply memory (ASSUMED).
  - Rehearsal of old memories against new ones (CITED: McClelland 1995; van de Ven 2020 for replay in machine learning).
- **Costs:**
  - About 26–40 new units per 1k pairs, which a novelty gate must keep bounded.
  - An 8k-unit H with a dense H → program table (≈ 32 M weights).
  - Replay compute in the offline phases.
- **Checks:**
  - (a) Run `realT_growth.py` V1 (minutes on CPU) inside the full boundary rule with the copy route. Retention must be ≥ 0.8 at +3–6k and +10k, and never-heard top-1 must not fall by more than 0.01.
  - (b) Hear 40 pairs once, test 100 dialogues later. Whole-reply 20-way top-1 must be ≥ 0.5; today it is ≤ 0.23 at k = 1.
  - (c) For once-heard probes, recall must rise across one offline phase with no new exposure.
  - (d) The growth counter must stay at ≤ 50 units per 1k dialogues.

**O3. A slowly maturing learned representation layer** (P2, P7; the real PFC upgrade; research).

- **What it is:**
  - Chunk/word units W over the auditory stream: the order rule plus k-WTA and a set point; recruitment for unexplained input; replay also trains W.
  - T′ is built from W plus FEAT0.
  - Per-unit plasticity falls as the unit consolidates, η_i = η0 / (1 + M_i/M0). This is a critical period at the level of each unit; new things go to fresh units.
- **Buys:**
  - Compression: memories are stored per reusable chunk, not per verbatim line.
  - Generalisation across lines that share words.
  - A schema-like speed-up for later learning (CITED: Tse 2007; Kumaran 2016).
- **Costs:**
  - Drift while units mature, so readouts must re-learn through O2's fast path and replay.
  - It is the least proven part.
- **Checks:**
  - Mature units must keep ≥ 0.9 Jaccard on their preferred inputs across 1k dialogues.
  - T′ drift for fixed lines must be ≤ 5 % per 1k dialogues after maturation.
  - W boundaries must match dictionary words better than a transitional-probability baseline.
  - With the copy route off, T′-based first syllables on never-heard lines must reach ≥ +0.03 top-1 over chance (T today: +0.007).

**O4. Generic timing and third factors** (P5, P10).

- **What it is:**
  - One generic pulse-readout clock for held states (§4.5).
  - Eligibility traces (seconds) on the decision paths.
  - Two broadcast signals: novelty N, which scales η and decides what H stores (CITED: Lisman & Grace 2005; Takeuchi 2016); and outcome R, later from the partner's response, which converts traces (CITED: Yagishita 2014; He 2015; Gadagkar 2016; Goldstein 2003).
- **Buys:** no new hand-placed gate per behaviour, and learning from delayed outcomes, i.e. reinforcement learning.
- **Costs:** one trace per decision-path synapse (about 2× memory) and one broadcast unit.
- **Checks:**
  - Back-term hits must stay at 0 once the hand-placed gates are replaced.
  - Toy: a cue held 6 ticks, with the target 4 ticks after the cue ends, must be learnable with a trace and broadcast, and not without.

**O5. A local prediction-error term on decision paths** (P6).

- **What it is:** depress a unit's active learned inputs when it was strongly predicted but did not fire (ABS-like, a margin form of the delta rule).
- **Buys:** copy weights move toward P(c in reply | c heard), the prior is subtracted, parroting stops rising with experience, and capacity is freed.
- **Costs:**
  - **The designer once deleted a similar rule**, so this needs his consent.
  - A naive "learn only on error" gate cut retention from 0.80–0.95 to 0.15–0.35 (minimal verifier). Use a margin form, not a hard gate.
- **Checks:**
  - The correlation between copy weight and P(c in reply | c heard) must rise from about 0 to > 0.5.
  - Parroting must stay flat from 1k to 10k dialogues.
  - The retention protocol above must not drop.

**O6. A cerebellum-like forward model of the own voice** (P10).

- **What it is:** a learned prediction from the held program to FEAT0. This gives an "efference-cancelled" talking detector F′ and an abort rule (already proposed in ARCH_v1 §6.4), plus learning articulation from auditory error.
- **Buys:** yielding, and imitation of new sounds.
- **Check:** pauses of 5–9 ticks inside the partner's turn must give 0 overlapping mouth ticks (today 29–75).

**O7. The training regime.**
- **Stages:** babbling (Z, programs) → listening to overheard dialogue (W, NEXT) → conversation (teacher optional), with offline phases between.
- **Data:** add a third voice so the model can overhear, and plan for much more data. Even ideal surface learners plateau at top-1 0.21–0.26 on this corpus.
- **Check:** each stage has its own audit; for example, NEXT accuracy on overheard pairs must be above the unigram baseline before conversation starts.

### 5.4 What to keep

- The body.
- The experience clock.
- The idea behind the gate. It becomes "learn only from targets driven by the world".
- Fast plus slow paths.
- Per-target budgets, used as fast heterosynaptic normalisation.
- Programs, and the panel's understanding of the timing law.

---

## 6. The designer's PFC concern, from the learning side

You are right that the PFC is too simple. In v1, T is not a PFC; it is a register.

But adding layers and recurrence to a PFC that learns online with the present rule would repeat the v0 failure, for three learning reasons:
1. **The rule cannot learn held states** (the timing law). The recurrent weights stay at about 0 (think→think ≈ 0.0001; STATE #7).
2. **Any PFC learning drifts its code** and breaks every readout keyed on it: plan decoding went 0.95 → 0.70 in 200 dialogues (BL-S3).
3. **Nothing says what is worth holding.** There is no dopamine/basal-ganglia gate.

What the real PFC adds, and how to build each part:
- **Fast holding through synaptic and attractor dynamics, separate from slow learning** (CITED: Mongillo 2008). Keep T's fast hold, or add a symmetric short-term-facilitation component as review 02 suggested.
- **A gate learned by dopamine** that decides hold vs update (CITED: PBWM, O'Reilly & Frank 2006). This is O4's outcome signal, added later.
- **Knowledge acquired slowly over years** (CITED: Gogtay 2004), from regularities across many episodes and from replay (CITED: Tse 2007; Kumaran 2016). This is O3's maturing W/T′ layer, consolidated by O2's replay.

So the PFC should have two timescales: a fast hold, plus a slow, maturing, multi-layer association code. The "deep layer returns to the upper layer" loop the designer described belongs in the slow code. The concept units "with no return path" the designer described would be mature W/T′ units that drive no area directly (ASSUMED design).

---

## 7. What I am unsure about

- **S1 is an upper bound.** It uses random codes, and the very long lifetimes for V1 were tested only there.
- **S1b is limited.** It covers 14.6k dialogues and the memory-only readout. V1 inside the full boundary rule (copy route, END) is untested.
- **The V1 learning assignment is a mechanism I have not tested** (which sub-unit learns when its program is cued) (ASSUMED).
- **Teacher-free transfer.** The transfer from own-voice context to partner-voice context is untested (ASSUMED). Overheard pairs avoid the issue but need a third voice in the world.
- **O3 is the least certain.** There is no evidence yet that the order rule plus competition gives useful chunk codes at this scale.
- **Biology is approximate here.** The biological timescales come from rodents and primates, and the mapping 1 tick ≈ 75 ms is approximate. Human adult neurogenesis is unresolved, so growth is argued from function, not from human data.

---

## 8. Files

All files are in `/tmp/claude-0/-root/e92d3d7d-5823-4f5c-a0ec-4fe302bff9af/scratchpad/arch/review/learning/`:
- `corpus_facts.py` → `corpus_facts.out`, plus `first_char_freq.npy`
- `lifelong_capacity.py` → `lifelong_capacity_s0.out`, `lifelong_capacity_s1.out`
- `lifelong_capacity2.py` → `lifelong_capacity_V1_300k.out`
- `realT_growth.py` → `realT_growth_s0.out`, `realT_growth_s1.out`
- `realT_growth2.py` → `realT_growth2_s0.out`
- `realT_observe.py` → `realT_observe_s0.out`
- `clock_check.py` → `clock_check.out`

Read-only inputs: `arch/content/fe_probe_s{0,1}.pkl`.

---

## 9. Sources

**Complementary learning systems and consolidation**
- McClelland, McNaughton & O'Reilly 1995, Psychol Rev 102:419 (doi 10.1037/0033-295X.102.3.419)
- Kumaran, Hassabis & McClelland 2016, TICS 20:512 (10.1016/j.tics.2016.05.004)
- Tse et al. 2007, Science 316:76 (10.1126/science.1135935)
- Schapiro et al. 2017, Phil Trans R Soc B (PMC5124075)
- Teyler & DiScenna 1986, Behav Neurosci 100:147
- Kim & Fanselow 1992, Science 256:675
- Abraham et al. 2002, J Neurosci 22:9626
- Scoville & Milner 1957, J Neurol Neurosurg Psychiatry 20:11

**Synaptic memory**
- Benna & Fusi 2016, Nat Neurosci 19:1697 (10.1038/nn.4401)
- Fusi, Drew & Abbott 2005, Neuron 45:599
- Amit & Fusi 1994, Neural Comput 6:957
- Frey & Morris 1997, Nature 385:533
- Yang, Pan & Gan 2009, Nature 462:920
- Hayashi-Takagi et al. 2015, Nature 525:333
- Willshaw, Buneman & Longuet-Higgins 1969, Nature 222:960

**Sleep and replay**
- de Vivo et al. 2017, Science 355:507
- Tononi & Cirelli 2014, Neuron 81:12
- Wilson & McNaughton 1994, Science 265:676
- Ji & Wilson 2007, Nat Neurosci 10:100
- Girardeau et al. 2009, Nat Neurosci 12:1222
- Fernández-Ruiz et al. 2019, Science 364:1082
- Klinzing, Niethard & Born 2019, Nat Neurosci 22:1598
- O'Keefe & Recce 1993, Hippocampus 3:317
- Skaggs et al. 1996, Hippocampus 6:149
- Foster & Wilson 2007, Hippocampus 17:1093

**Three-factor learning and neuromodulation**
- Schultz, Dayan & Montague 1997, Science 275:1593
- Yagishita et al. 2014, Science 345:1616
- He et al. 2015, Neuron 88:528
- Gerstner et al. 2018, Front Neural Circuits 12:53
- Bittner et al. 2017, Science 357:1033
- Hasselmo 2006, Curr Opin Neurobiol 16:710
- Yu & Dayan 2005, Neuron 46:681
- Lisman & Grace 2005, Neuron 46:703
- Takeuchi et al. 2016, Nature 537:357

**PFC and basal ganglia**
- O'Reilly & Frank 2006, Neural Comput 18:283
- Wang et al. 2018, Nat Neurosci 21:860
- Mongillo, Barak & Tsodyks 2008, Science 319:1543
- Gogtay et al. 2004, PNAS 101:8174
- Graybiel 1998, Neurobiol Learn Mem 70:119
- Jin & Costa 2010, Nature 466:457

**Memory allocation and adult neurogenesis**
- Han et al. 2007, Science 316:457
- Yiu et al. 2014, Neuron 83:722
- Aimone, Deng & Gage 2011, Neuron 70:589
- Ge et al. 2007, Neuron 54:559
- Sorrells et al. 2018, Nature 555:377
- Boldrini et al. 2018, Cell Stem Cell 22:589
- Moreno-Jiménez et al. 2019, Nat Med 25:554

**Hippocampal sparsity**
- Leutgeb et al. 2007, Science 315:961
- Ramirez-Amaya et al. 2013, J Neurosci 33:1631

**Representational drift and forgetting**
- Ziv et al. 2013, Nat Neurosci 16:264
- Driscoll et al. 2017, Cell 170:986
- Rule, O'Leary & Harvey 2019, Curr Opin Neurobiol 58:141
- Wixted & Ebbesen 1991, Psychol Sci 2:409

**Rules, inhibition and motor models**
- Vogels et al. 2011, Science 334:1569
- Schneider, Sundararajan & Mooney 2018, Nature 561:391
- Urbanczik & Senn 2014, Neuron 81:521
- Artola, Bröcher & Singer 1990, Nature 347:69
- Tourville & Guenther 2011 (DIVA)

**Language acquisition, prediction and turn-taking**
- Saffran, Aslin & Newport 1996, Science 274:1926
- Kuhl, Tsao & Liu 2003, PNAS 100:9096
- Goldstein, King & West 2003, PNAS 100:8030
- Doupe & Kuhl 1999, Annu Rev Neurosci 22:567
- Werker & Tees 1984, Infant Behav Dev 7:49
- Gadagkar et al. 2016, Science 354:1278
- Hensch 2005, Nat Rev Neurosci 6:877
- Hart & Risley 1995 (book)
- Gilkerson et al. 2017, Am J Speech Lang Pathol 26:248
- Heilbron et al. 2022, PNAS 119:e2201968119
- Pickering & Garrod 2013, BBS 36:329
- Levinson 2016, TICS 20:6
- Stivers et al. 2009, PNAS 106:10587

**Machine learning analogues**
- van de Ven, Siegelmann & Tolias 2020, Nat Commun 11:4069
- Kirkpatrick et al. 2017, PNAS 114:3521

---

## summary_zh

1. 身体部分（听、等、开口、说清楚、会停）已经做好。问题在学习：真正影响说话的学习只有 6 张"查表"，而且只有老师推着嘴说话时才学；耳朵自己的联想、海马体都没接到会影响说话的地方。
2. 老师一走，说什么、怎么接、何时停、学新字全部停止，使用中的实时学习是零。原因不是"门"本身，而是架构里没有来自外界的学习信号。
3. 即使有老师，约 3000 段对话后就不再进步：新句子只会鹦鹉学舌；每个字的记忆只装最近约 5 次（快）和 25 次（慢）；常用字开头的话几千对话就忘光。经历越多，记住的比例越低（模拟：0.65→0.19）。
4. 和真脑最大的差别不在前额叶层数，而在记忆系统：
   - 人靠听、模仿和别人的回应学说话，没人推嘴。
   - 海马体学得快但记得久（几天到几周），会在空闲和睡眠时回放，把内容教给皮层；我们的约 15 秒就忘，而且没接线。
   - 皮层慢慢学出可复用的表征，新记忆交给新神经元。
   - 多巴胺等调质决定学什么。
5. 优先改三件事：
   - ① 不靠老师：从听到的话里学"接下来会听到什么"，说话时说出预测，用现成的"听到的字 ↔ 自己的字"镜像。
   - ② 记忆能累积：一个字的记忆单元满了就"分裂"出新单元接着学（就是你的分裂机制）；海马体改成能记上千段对话、会在空闲期回放的快速记忆。
   - ③ 加一个慢慢成熟的学习表征层（字词块、联合皮层），成熟后少变，新东西交给新单元。
6. 在真实"想法码"上验证：分裂式增长让 3000–10000 段对话后的回忆从 0.41–0.50 提到 0.83–0.88，只多约 19% 单元；不用老师、只用耳朵听到的回答来学，只慢大约一次曝光。
7. 关于前额叶：你说得对，它太简单。但只加层数和回路会重演以前的漂移失败，因为规则学不进持续状态，学习会让下游读出失效，也没有"该记什么"的信号。应该做成两个时间尺度：快的保持（像现在）+ 慢慢成熟的学习层 + 以后由奖励训练的门控。
8. 需要你拍板的三点：
   - "海马体只管短期记忆"与证据不符，它管几天到几周的新记忆。
   - 可能需要一条局部的"预测了却没发生就减弱"规则（你以前删过类似的）。
   - 以后需要多巴胺式的第三因子。