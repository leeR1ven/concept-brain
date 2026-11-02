(Saved by the coordinator from the reviewer's final reply; the harness blocked the reviewer's own write.)

# Whole-architecture review — systems-neuroscience lens (reviewer: neuro)

The harness blocked writing `report.md`, so the full report is below. My simulation scripts and outputs are saved in `/tmp/claude-0/-root/e92d3d7d-5823-4f5c-a0ec-4fe302bff9af/scratchpad/arch/review/neuro/`: `retrieval_ceiling.py/.out`, `semantic_sim.py/.out`, `semantic_retrieval.py/.out` and `assembly_sim.py/.out`.

**Labels:**
- **PROVED** — argument, arithmetic or code reading, with numbers.
- **SIMULATED** — numpy run; file and numbers given. "Mine" means my folder above.
- **CITED** — paper and finding.
- **ASSUMED** — argued but untested.

---

## 0. Verdict

1. **Keep the body.** Turn-taking, syllable timing, articulation and stopping all run with 0 errors (SIMULATED, `arch/synth/res_*`). This covers F, GO, CD, the 3-tick beat, the program winner-take-all (WTA), D/I/M, END and the counter. Biologically this is defensible innate brainstem, basal-ganglia and thalamic structure (CITED):
   - a content-free GO signal launches a prepared action (Inagaki 2022);
   - tonic inhibition plus disinhibition selects actions (Hikosaka 2000);
   - the thalamus gives a kick at each syllable onset (Moll 2023).

2. **In brain terms v1 is a reflex arc.** Every decision reads only two things:
   - the ear's first layer L0 (window energy → F; FEAT0 → T; FEAT0 → echo mirrors Z);
   - the programs' own delay lines (Hc).

   The parts that learn like cortex reach no behaviour at all: ear cortex (≈20k units), motor cortex (≈3.8k) and hippocampus (≈20k alive). PROVED from `Brain.step` in `arch/synth/archv1.py`:
   - program scores = P→Y + Hc→Y + slot (FEAT0 / Z / latch);
   - F reads the input energy;
   - T reads FEAT0;
   - nothing reads ear layers ≥ 1, mctx or the hippocampus.

3. **What v1 calls the PFC is a cerebellum, not a PFC.** T plus P→Y is a fixed random sparse expansion (fan-in 30, 5 % active) with a readout that learns only when a teaching signal is present. That is the Marr–Albus cerebellum / insect mushroom-body design (Marr 1969; Albus 1971; Litwin-Kumar 2017) — CITED, and PROVED for the structure. Such circuits memorise input→output pairs and generalise only through raw input overlap. That is exactly v1's profile: familiar lines 0.57, never-heard lines 0.049–0.070 against chance 0.05 (SIMULATED, `arch/synth`).

4. **v1 has the brain's dorsal route but no ventral route.** The dorsal route (sound → articulation, i.e. repetition) exists; the ventral route (sound → meaning) does not (CITED: Hickok & Poeppel 2007). Hence "早上好你好吗 → 学你好上早吗": repetition without comprehension.

5. **"The PFC has too few layers" is a symptom.** The gaps that actually block the final goal are:
   - (a) no learned representation hierarchy between ear and mouth;
   - (b) a learning rule that cannot form assemblies or bridge delays (SIMULATED, mine);
   - (c) content is learned only while a teacher moves the mouth, and is frozen in free conversation (PROVED);
   - (d) no episodic memory on the behaviour path, and no consolidation;
   - (e) a world where meaning can't be learned or measured at this scale. An ideal similarity memory over 40k pairs gets chance-level content once echo is excluded. Yet the same corpus yields clear word categories to a one-pass local learner (SIMULATED, mine).

6. **Priorities** (detail in §5):
   - ① a learnable "nursery" world, a staged curriculum and meaning tests;
   - ② a per-area learning rule: a co-activity term with an immediate budget in association areas, and an eligibility trace × third factor on decision paths;
   - ③ a lexical layer plus a slow gist layer, which is the real PFC, gated by the existing turn units;
   - ④ production through perception: learn Q→R by listening, then speak through the learned auditory→program map;
   - ⑤ the hippocampus as an episodic index with offline replay;
   - ⑥ novelty and reward signals;
   - ⑦ a forward model of the model's own voice.

---

## 1. What v1 is, mapped onto the brain

| v1 element | brain analogue | comment |
|---|---|---|
| input lines + L0 | cochlea → cochlear nucleus (fixed relay) | L0 receives no learned or return current; `SenseArea.fire` masks extra current to cortex (PROVED) |
| F (window energy), GO, CD | brainstem/midbrain sound presence; thalamic GO (Inagaki 2022); corollary discharge (Schneider 2018) | reflexive and content-free; good |
| T: fixed FEAT0 expansion, 1-tick self-hold | echoic / phonological buffer; built like a cerebellar granular layer or mushroom body | not a PFC: no learned input, no abstraction |
| P→Y fast and slow, Hc→Y, Hc→END (teacher-gated) | corticostriatal or Purkinje readouts with a supervising signal (climbing fibre / phasic dopamine) | memorises; no reward learning |
| programs + WTA on a 3-tick beat | basal-ganglia action channels + brainstem syllable rhythm (4.4 Hz ≈ auditory theta; Giraud & Poeppel 2012) | good; Levelt's "mental syllabary" is a plausible analogue |
| D/I/M | motor nuclei under tonic, pallidal-like inhibition | good |
| copy route: Z → latch → E → Y | dorsal stream (STG → Spt → premotor): repetition | real (toddlers echo too), but only half of language |
| ear cortex layers 1–3 | labelled "A1", but each unit copies one input line (see below) | learns transitions; feeds nothing |
| motor cortex (mctx) | proprioceptive map | no output in v1 |
| hippocampus: 1:1 mirror, η 0.5, γ 0.7 per 100 ticks | CA3-like sequence buffer without DG, CA1 or EC | half-life = 100·ln 0.5 / ln 0.7 = 194 ticks ≈ 15 s (PROVED); no output |
| absent | ventral stream (STS/MTG/ATL), PFC proper, MD thalamus, cerebellum, neuromodulators, other senses | — |

**The ear has no "features" in the neuroscience sense (PROVED from `net3d.py`).**
- Each unit sums three things: its own input line (weight 1), three neighbours (weight ≤ 0.01 each) and a fixed bias drawn from U(0, 0.5).
- The neighbour terms total at most 0.03. They cannot reorder units whose biases differ by more than that.
- So a unit is on when its own line is on and its fixed bias ranks high enough.

The "cortex" is therefore a fixed sparse copy of the spectrogram code. It has no conjunctive, invariant or learned feed-forward features. This is harmless in this toy world, where chars are random codes and the ear decodes the line at 1.00 (STATE #1).

---

## 2. Differences from the real brain

Rating key: **F** = fundamental for the final goal; **L** = matters later; **H** = harmless simplification.

### 2.1 Structure

1. **The ear-to-mouth hierarchy — F.**
   - Brain: A1 → belt/parabelt → STG/STS (word forms) → MTG/ATL (lexical-semantic hub) ↔ IFG/PFC, plus a dorsal repetition stream. Temporal receptive windows lengthen up the hierarchy: word → sentence → narrative (CITED: Hickok & Poeppel 2007; Patterson 2007; Lerner 2011; Hasson 2015).
   - v1: L0 → fixed T → lookup, and L0 → Z → copy.
   - Why it matters: nowhere can a word or a meaning be represented, so B8 (generalisation) is impossible by construction.

2. **Sensory cortex — H in this world, L for real audio.**
   - Brain: learned spectro-temporal receptive fields; sparse and adapting; maps shaped by early experience (CITED: Hromádka 2008; Zhang, Bao & Merzenich 2001).
   - v1: fixed copies of input lines (PROVED, §1).

3. **PFC — F.** See §3.

4. **Hippocampus — F.**
   - Brain: DG pattern separation → CA3 autoassociation (with a symmetric timing window; Mishra 2016) → CA1 → cortex. It encodes episodes in one shot and keeps them for days to years. Patient H.M. kept normal short-term memory but could form no new long-term episodic memories (CITED: Scoville & Milner 1957; Squire 2009).
   - v1: a 1:1 mirror with about 15 s of memory and no output (PROVED). It barely tells lines apart: same-line Jaccard 0.18 vs 0.14 between different lines (SIMULATED, `res_*`).
   - Why it matters: the framework's "hippocampus = short-term memory" is the reverse of the clinical evidence. The final goal needs exactly a fast-learning, long-lasting episodic store.

5. **Basal ganglia — selection H; reward learning L (needed for goal-directed behaviour).**
   - Brain: learns action selection and working-memory gating from dopamine reward-prediction errors (CITED: Schultz 1997; Frank 2005; O'Reilly & Frank 2006).
   - v1: an innate WTA + GO, with supervised readouts.

6. **Thalamus — L.**
   - Brain: MGB relays all hearing, TRN gates it, and MD sustains and amplifies PFC activity (Schmitt 2017).
   - v1: absent (the MD loop was removed).

7. **Cerebellum — L.**
   - Brain: 69 of the brain's 86 billion neurons (Azevedo 2009). It provides forward models, timing and error-driven learning, plus speech fluency (Ackermann 2008) and prediction in language (Moberget & Ivry 2016).
   - v1: none. Ironically, T + P→Y has the cerebellar layout but is used as the "PFC".
   - Why it matters later: predicting the model's own voice (so it can hear while speaking, and yield) and motor learning both need it.

8. **Motor cortex — H for this mouth; L for "a model that can operate machines" (FRAMEWORK #19).**
   - Brain: ventral sensorimotor cortex encodes articulator trajectories and learns skills (Bouchard 2013; Chartier 2018).
   - v1: mctx is a proprioceptive map with no output, and the programs are fixed muscle sets recruited from the teacher's cue.

9. **Cortical microcircuit — mostly H.**
   - Brain: six layers; about 20 % interneurons (PV, SST, VIP); Dale's law; coincidence detection in apical dendrites (Larkum 2013); learned inhibition (Vogels 2011).
   - v1: top-k as a blanket of inhibition, and additive, modulatory return currents.
   - Matters later (L): VIP-like gating of learning and attention, and learned inhibition.

### 2.2 Dynamics

1. **Global synchronous clock (75-ms ticks) — H, with one catch.**
   - The brain runs continuously and asynchronously.
   - Synchronous threshold networks generically fall into 2-cycles, whereas asynchronous symmetric networks settle to fixed points (CITED: Goles-Chacc et al. 1985; Hopfield 1982).
   - So the period-2 alternation of STATE #5 belongs to a known class of artefact. Every new loop must be checked for it. Fixable.

2. **Binary units with exactly one tick of memory — F for any learned holding.**
   - Brain: firing rates are graded, and slow NMDA currents (~100 ms) stabilise working-memory attractors (CITED: Wang 1999; Wang M 2013).
   - Why it matters: v1 has to build every hold and every timing from latches and delay lines (T's self-hold of 3.0, P = T∧GO, the R2 clock, the Hc delay lines, the latch L_c).
   - A per-unit leaky potential is a cheap fix (§5 step 3).

3. **No hierarchy of timescales — F.**
   - Brain: intrinsic timescales lengthen from sensory to prefrontal cortex (CITED: Murray 2014), and higher language areas integrate over whole sentences and narratives (Lerner 2011).
   - v1: every area remembers one tick. A "gist" is by definition a slow variable.

4. **Fixed activity level — H for now.**
   - Brain: sensory cortex is quiet in silence, sparse, and driven by novelty (Hromádka 2008; Ulanovsky 2003).
   - v1: a fixed fraction fires every tick; muting and the output reserve compensate.
   - The variable-count idea in FRAMEWORK §四 points the right way, but it is not a priority.

5. **Syllable rhythm — H, a good match.** The fixed 3-tick beat is 4.4 Hz, close to the 4–8 Hz auditory theta that tracks syllables.

### 2.3 Learning rules and neuromodulators

1. **No co-activity learning — F.**
   - Brain: sustained or high-rate co-firing potentiates in either order (Sjöström 2001); CA3's timing window is symmetric (Mishra 2016); connections between similarly tuned neurons grow with experience (Ko 2011, 2013).
   - v1: held co-activity cancels out (the panel's timing law).
   - SIMULATED (mine, `assembly_sim.out`): with net.py's exact rule, a region trained on 40 concepts forms no assemblies. The learned weight inside a concept is 0.0002 per pair, and reactivation from a half cue is at chance (0.01–0.07 vs 0.05).
   - A co-activity variant with the budget applied immediately works and stays bounded:
     - cue-specific reactivation goes 0.48 → 0.37 over 8 ticks;
     - the concept still holds at 0.34 after 16 ticks (7× chance);
     - it switches to a newly cued concept;
     - every per-target total stays at the budget.

2. **No learning across delays — F.**
   - Brain: eligibility traces plus a third factor bridge 0.3–2 s (Yagishita 2014; He 2015; Gerstner 2018), and behavioural-timescale plasticity spans seconds (Bittner 2017).
   - v1: none. A learned source must be a 1-tick pulse (PROVED by the panel).
   - Why it matters: every learned link needs a hand-built pulse generator. "Held thought → later word" and "act → later feedback" cannot be learned.

3. **Budget enforced only every 100 ticks — fixable, but it must be fixed before adding (1).**
   - Brain: heterosynaptic normalisation happens at the moment of learning (Oh 2015; El-Boustani 2018), and it has to be as fast as Hebbian growth (Zenke & Gerstner 2017).
   - v1: hippocampal totals reach 14–15 against a budget of 2 between normalisations (SIMULATED, `res_*`).
   - In my co-activity run with this 100-tick budget, totals reached 6.5× the budget, and five "super-attractors" captured random starts (SIMULATED).

4. **No neuromodulators — F for AGI, L for milestone 1.**
   - Brain: dopamine (reward error), acetylcholine (encode vs retrieve) and noradrenaline (surprise, reset) decide whether and where Hebbian traces are kept (He 2015; Hasselmo 2006).
   - v1: only the teacher flag (the C11 gate). So there are no goals and no learning from consequences.

5. **Synaptic consolidation — L; the crude version works.**
   - Brain: tag-and-capture and multi-timescale synapses (Frey & Morris 1997; Benna & Fusi 2016).
   - v1: one decay rate γ per path; the fast and slow P→Y pair is a crude two-state version. The slow path keeps 0.39–0.40 at +10k dialogues (CONTENT §4).

6. **Learned inhibition — L.** The brain learns "negative images" of predictable input (Vogels 2011); v1 has none.

### 2.4 Memory systems and consolidation (including sleep and replay)

| memory system | brain | v1 | rating |
|---|---|---|---|
| echoic (2–4 s) | auditory cortex | T + the ear's afterimage; T decodes 20/20 | H |
| working memory | PFC/parietal attractors gated by basal ganglia and MD | T's self-hold | F (no gist-level working memory) |
| episodic | hippocampus: one-shot, long-lasting | a 15-s mirror that nothing uses | F |
| semantic | ATL hub (Patterson 2007) | none | F |
| procedural | basal ganglia, cerebellum | innate programs + the Hc→Y chain | H for now |
| consolidation | hippocampal ripple replay during rest and NREM sleep trains cortex (Wilson & McNaughton 1994; Girardeau 2009; Maingret 2016); sleep down-scales synapses (Tononi & Cirelli 2014); complementary learning systems (McClelland 1995; Kumaran 2016) | no offline phase; uncontrolled replay in silence rewrites ear weights (max \|Δw\| 0.80–0.87 per 1,000 silent ticks; hippocampus 6.8–6.9; SIMULATED, `res_*`) | F for continual learning; the idle drift becomes harmful once cortex matters |

### 2.5 Innate vs learned structure, and development

**Brain (CITED):**
- There are innate scaffolds: tonotopy, laminar circuits, basal-ganglia loops, brainstem rhythm generators, and proto-conversational turn-taking from about 3 months of age (Hilbrink 2015).
- Sensory and association cortex are then tuned by experience.
- The PFC matures last (Huttenlocher & Dabholkar 1997):
  - synaptic density peaks about 3 months after birth in auditory cortex, but only after 15 months in prefrontal cortex;
  - pruning ends around age 12 in auditory cortex but continues into mid-adolescence in PFC.
- Comprehension comes before production (Fenson 1994; Bergelson & Swingley 2012). Babies segment words from transition statistics at 8 months (Saffran 1996), and contingent feedback shapes babbling (Goldstein 2003).

**v1:**
- The body is innate — right.
- The sensory and association representations are fixed — the opposite of biology.
- There are no developmental stages.
- The history of failed PFC loops (STATE #3–5) amounts to building the last-maturing area first, directly on raw sound. **F**: the order of development is itself a design principle.

**Split/delete:**
- Adult neocortex does not add neurons. Adult neurogenesis is confined to the dentate gyrus, and is disputed in humans (Sorrells 2018 vs Boldrini 2018).
- The goal behind split/delete — more and finer units for frequent inputs — is reached in the brain by competitive re-tuning plus homeostasis (Recanzone 1993; Ganguli & Simoncelli 2014).
- As an engineering device for recruiting units it is harmless (**H**).

### 2.6 Body, world and training regime

**The world:**
- Chars are random spectrograms, so there is no phonology and no homophones. Real Mandarin maps roughly 4k chars onto about 1.3k tonal syllables (ASSUMED).
- Muscle sets are random 4-of-32 combinations, with no articulatory structure.
- There is one voice, no noise and no other senses.
- The partner's lines are fixed text, so nothing the model says has any consequence.

Verdict: the random phonology is **H** for now. The lack of contingency and grounding is **F**: there is no pragmatic feedback, and only distributional meaning is learnable.

**The training regime:**
- The teacher puppets the mouth: muscle cues at every boundary, plus the END clamp.
- Decision paths learn only under teacher drive, so in free conversation nothing about what to say is learned (PROVED from the ARCH_v1 §1.4 gate). This is **F** for "learns in real time".
- Each dialogue is heard once, and there is no learning from overheard dialogue — which toddlers do (Akhtar 2001).
- The whole corpus is about 0.64 M chars ≈ 40 h of speech (PROVED: 635,570 chars × 3 ticks × 75 ms). That is a few weeks of what a toddler hears (Hart & Risley 1995: roughly 3–11 M words a year).
- It is adult social-media chat whose replies are barely predictable from the prompt (§6). **F**, but fixable as a regime.

### 2.7 Scale
- v1 has about 10⁵ unit slots and 10⁸ synapse slots, mostly the dense decision tables (PROVED from ARCH_v1 §1.1 and §1.4).
- That is roughly a fruit-fly brain (about 1.4 × 10⁵ neurons; FlyWire, Dorkenwald 2024) and about 10⁶ times fewer neurons than a human brain (8.6 × 10¹⁰; Azevedo 2009).
- Harmless for now, because mechanisms rather than size are the bottleneck (§6). It matters later.

### 2.8 What the framework already gets right (keep)
- Tonic inhibition vs disinhibition decides actions (#4) — the basal-ganglia principle.
- A content-free trigger decides *when* to act (Inagaki 2022).
- A fast hippocampus and a slow cortex (#2, #5) — complementary learning systems.
- An asymmetric "earlier → later" rule for sequences (#9). The songbird HVC model uses almost the same rule (Okubo 2015).
- Global decay + budgets + set point, which match synaptic scaling and sleep down-scaling.
- "External currents choose *which* units fire, not *how many*" — the driver/modulator distinction and normalisation.
- The PFC holds the gist while detail lives elsewhere (#28; Hasson 2015; Baldassano 2017).
- The PFC is not a copy of sensory cortex (#31).
- Concept units with no returns (10-30), which match mixed selectivity (Rigotti 2013).
- A constant PFC activity level (#26; Schmitt 2017).
- No dedicated "speech area" (#22).

---

## 3. The designer's PFC concern, answered

**The concern is right, but "more layers" alone is the wrong fix.** Five properties make a real PFC. v1's T has none of them, and adding layers provides none of them.

| property | real PFC (CITED) | v1's T | consequence |
|---|---|---|---|
| input | apex of the hierarchy: association cortex, hippocampus (vCA1→mPFC), MD thalamus; little direct primary-sensory input (Fuster 2001) | ear L0, the most peripheral code | a PFC on L0 can only hold sounds |
| plasticity | learns categories and rules that transfer to new items (Freedman 2001; Wallis 2001); mixed selectivity (Rigotti 2013); learns slowly, "taught" by striatum and hippocampus (Pasupathy & Miller 2005) | none — fixed | no abstraction |
| dynamics | recurrent layer-2/3 attractors stabilised by NMDA (Wang M 2013); the longest intrinsic timescales (Murray 2014) | 1-tick self-hold | holding must be engineered |
| gating | basal ganglia and MD decide when to update and when to protect (O'Reilly & Frank 2006; Schmitt 2017) | frozen by L+3 via F; output gate P = T∧GO | the turn units already make a ready gate — reuse them |
| output | biases all posterior cortex, top-down, at every step (Miller & Cohen 2001) | picks the first syllable only | no plan for later syllables |

**Why simply adding layers would fail again.** The old receiving ↔ thinking ↔ thalamus loop *was* a multi-layer recurrent PFC. It held no line content (decode 0.15–0.60, STATE #3), learned almost nothing (#4, #7), and drifted and alternated (#5, BD-S2). Its failure now has clear causes:
- its input was raw sound features with a large shared core (00_synthesis F6);
- the learning rule cannot store held co-activity (§2.3, simulated);
- its units had one tick of memory, so it alternated with period 2;
- nothing upstream represented words.

In the brain the PFC is built last, on top of mature lexical and semantic cortex. It should be built last here too.

**What the PFC should become** (full spec in §5 step 3):
- a slow "gist" area of leaky units (time constant τ ≈ 20 ticks);
- fed by a learned lexical layer, by hippocampal recall, and by an efference copy of what the model itself is saying;
- with assemblies learned by co-activity;
- updated while the partner speaks and held during the reply. This is a PBWM-style gate built from the existing F/GO signals, with no new rule;
- read at *every* reply boundary, not only the first.

Two stacked gist layers — one for the line (≈20 ticks) and one for the dialogue (≈100 ticks) — are the "multi-layer" part, like the frontal lobe's rostro-caudal hierarchy (Badre & D'Esposito 2009). Concept units that drive nothing fit naturally here.

Keep T, renamed the "phonological buffer", for verbatim recall. It decodes 20/20 and is immune to learning.

---

## 4. Problems of the whole architecture, ranked

Type: **F** = fundamental (needs a structural change); **X** = fixable (parameters or details).

| # | problem | type | seen yet? | evidence |
|---|---|---|---|---|
| 1 | No learned representation (no ventral stream) between ear and mouth. Cortex and hippocampus are disconnected from behaviour, so nothing can carry meaning and B8 is impossible. | F | yes: never-heard content is at chance without echo (0.049–0.070) | PROVED (code); SIMULATED (res_base) |
| 2 | Content is learned only while the teacher moves the mouth, and is frozen in free conversation. So there is no real-time learning in use, and no learning from listening or from feedback. | F | by design; never run in deployment | PROVED (ARCH_v1 §1.4) |
| 3 | The rule forms no assemblies and bridges no delays. Every new capability needs hand-built pulse circuitry. | F (a local rule extension) | yes (STATE #4, #7) | SIMULATED (mine) + PROVED (panel) |
| 4 | The world and metric cannot show meaning. An ideal similarity memory gets chance-level content once echo is removed, and echo scores best (0.273). Optimisation drifts toward parroting (parrot index 0.49–0.61 vs 0.15 for humans). | X (regime), but it blocks every evaluation | yes | SIMULATED (mine); CONTENT §3 |
| 5 | No episodic store on the behaviour path and no consolidation. Decay and budgets erase frequent items first. Whole-reply recall stays ≤ 0.45, and first-syllable retention is 0.39 at +10k dialogues. | F | partly | CONTENT §4; PROVED |
| 6 | Wrong construction order: the PFC was built directly on raw sound, leading to drift, period-2 alternation and a generic state. | F (plan) | yes (STATE #3–5) | CITED (§2.5) |
| 7 | No value or novelty system: no goals, no learning from consequences, no salience. | F for AGI; not needed for milestone 1 | not yet | — |
| 8 | Representational drift once codes start learning. T already shifts about 8 % per checkpoint through its set point; learned layers will drift more, and long-term links will go stale. | F-ish: needs replay-based co-adaptation and stable lower layers | not yet (T is fixed) | SIMULATED (verify_learnability_dynamics M6); CITED (Driscoll 2017) |
| 9 | Hand-tuned ratios and tick-exact proofs (cue 12, memory ×10 vs copy, breath 0.9ⁿ, θ, used-flag penalty, slot cap 7). Every new route needs re-tuning, which does not scale. | X, but it grows with each module | not yet at scale | PROVED (ARCH_v1 §1.3–1.8) |
| 10 | Uncontrolled replay in silence rewrites ear weights. | X (discipline of offline phases) | yes (idle test) | SIMULATED (res_*) |
| 11 | The budget is enforced only every 100 ticks, so totals overshoot (hippocampus 14–15 vs 2). With co-activity learning this makes super-attractors. | X | yes | SIMULATED (res_*, mine) |
| 12 | Synchronous binary 1-tick units: every loop risks period-2 alternation, and every hold needs a latch. | X (leaky potential) | yes (STATE #5) | CITED (Goles 1985) |
| 13 | Turn-taking is purely reactive. The model cannot hear while speaking (CD blanks F) and cannot yield: pauses of ≥ 5 ticks give 29–75 overlapping mouth ticks per 6 trials. Latency is 450 ms vs about 200 ms in humans. | X (forward model F′) | yes | SIMULATED (res_* stress tests); CITED (Stivers 2009) |
| 14 | Closed vocabulary and no motor learning: programs exist only for chars the teacher has cued, so a newly heard word can never be said. | F for open-ended learning | not yet (closed corpus) | PROVED (recruitment rule) |
| 15 | Robustness: one voice, deterministic sounds, no noise. Whether the Z map transfers to a third voice is untested. | X | not yet | CONTENT §7 |

---

## 5. What to optimise, in priority order

### 5.1 Block diagram of the improved architecture
```
            ┌───────────── VALUE / STATE broadcast (few units; third factor M) ─────────────┐
            │ novelty = LEX prediction error · reward = partner's contingent response ·     │
            │ teacher drive (bootstrap only)                                                │
            ▼                                                                               │
sound → L0 (fixed) ──energy──► F′/F, GO, CD  (innate turn units; F′ = energy − predicted own voice)
         │  └──► T phonological buffer (fixed) ──► verbatim recall P→Y                  (keep)
         ▼
       A1 = ear cortex L1–3 (fixed FF; learned transitions = sound prediction → novelty)
         ▼  learned convergent FF (co-activity/trace rule, immediate budget, set point)
       LEX word-form layer (STG/STS): char units = existing Z mirrors + recruited chunk units;
           Z→Z transitions with the current rule; leaky τ≈3 ticks           ▲ top-down prediction
         ▼                                                                  │ (learned, modulatory)
       GIST = PFC proper: leaky τ≈20 (line) [+ τ≈100 (dialogue)]; recurrent co-activity assemblies;
           set point; concept units; UPDATE while partner speaks / HOLD during reply (F/GO/SPK gate)
         │   ▲                              ▲ ▼
         │   └── HIPPOCAMPUS: DG sparse index (1–2 %) → CA3 co-activity (one-shot) → CA1 → LEX/GIST
         │       recall; slow forgetting; replay in offline "sleep" phases → slow LEX/GIST learning
         ▼
       PLAN: predicted reply in LEX terms (learned from every heard Q→R, incl. others' dialogues)
         ▼  dorsal map LEX → programs (learned from own voice in babbling; Z already does this)
       BG-like program WTA at the 3-tick beat + END/counter (innate) → D/I/M → muscles → own voice → ear
                                   └── efference copy → cerebellum-like forward model → F′
```

**What learns, and how:**
- Sequence areas (ear cortex, motor chains) keep the current rule.
- Association areas (LEX, GIST, CA3) add co-activity/trace learning, with an immediate budget and a set point.
- Plan and decision paths learn with an eligibility trace × the third factor M.
- The hippocampus learns fast and forgets faster than cortex, but over days of experience, not seconds.
- Cortex learns slowly, mostly from replay.

### 5.2 The steps

**Step 1 — A world and tests that can show meaning (prerequisite; cheap).**
- **Build** a "nursery" world on the same ear and mouth physics:
  - about 300 chars and about 30 intents, each with 5–20 paraphrases that share words across intents;
  - replies determined by intent + slot, with some slots stated earlier in the dialogue (an episodic test);
  - held-out paraphrases and held-out slot combinations.
- **Stage the training:**
  - S0: listen only, to others' dialogues;
  - S1: babble — the model's own voice teaches Z;
  - S2: imitate, using the echo route;
  - S3: taught dialogue, as now;
  - S4: free dialogue with contingent feedback from the partner.

  The social corpus then becomes "adolescence".
- **Tests:**
  - paraphrase generalisation on the 20-way metric;
  - category purity of internal codes;
  - one-shot slot recall;
  - retention;
  - parrot index vs humans.
- **Check before use:** run `retrieval_ceiling.py` on the new world. A meaning-based kNN that excludes heard chars should reach ≥ 0.4 top-1, while echo stays ≤ 0.2. If not, the world still cannot tell meaning from echo.
- **Buys:** an evaluation where "thinking" can visibly succeed or fail.
- **Costs:** a generator and metrics (days of work); no GPU.

**Step 2 — A per-area learning rule.** This is local, and it generalises the designer's rule #9 rather than replacing it.
- **(a) Sequence areas:** keep the current rule as it is.
- **(b) Association areas:** potentiate when the source was on at t−1 and the target is on at t, including held activity. Add a post-synaptic trace for temporal pooling (Földiák 1991). Apply the per-target budget immediately, at learning time, and keep the set point on.
- **(c) Decision and plan paths:** an eligibility trace e ← λe + pre(t−1)·rise(t), and Δw = η·M·e.
  - M is the teacher drive now, and reward or novelty later.
  - This generalises the C11 gate.
  - It removes the 1-tick pulse engineering (P = T∧GO, R2 replay pulses) for new links.
- **Buys:**
  - assemblies at 7× chance after 16 ticks (SIMULATED, mine);
  - gist units that accumulate a whole line;
  - learning across seconds.
- **Costs and risks:**
  - one trace float per synapse on gated paths (+320 MB if the dense tables keep traces);
  - attractor capture: 8–10 of 40 concepts captured random starts in my run, so the set point needs tuning;
  - rule #9 is relaxed in association areas only.
- **Checks:**
  - unit tests on scripted spike trains (the panel's rule_exact style);
  - `assembly_sim.py` with pass thresholds: reactivation ≥ 0.4 at tick 8, hold ≥ 0.3 at tick 16, update ≥ 0.4, spontaneous attractors covering ≥ 50 % of concepts (20–25 % today), and per-target totals ≤ budget at every tick;
  - a delay test: a source held for 10 ticks and a target rising 5–20 ticks later, with M on, must learn a weight ≥ 0.5 of the pulse case, and 0 with M off;
  - v1's existing back-term audits stay unchanged.

**Step 3 — LEX + GIST: the ventral stream and the real PFC.**
- **LEX:**
  - its char units are the existing Z mirrors (latch precision 0.93–0.99, CONTENT §2);
  - 2–3-char chunk units are recruited using the split/delete machinery;
  - Z→Z transitions are learned with the current rule. My simulation shows successor statistics give category purity 0.34 vs 0.006 chance (SIMULATED, mine).
- **GIST:**
  - 10k leaky units (λ ≈ 0.95), top 5 % active, with a set point;
  - fan-in of about 64 from LEX, plus recurrent links, all learned by co-activity;
  - the input is open while F is on and until L+3, and blocked from GO to END.
- **Readouts:** GIST→Y and predicted LEX→Y at every boundary, entering the existing SLOT under its cap so the teacher always wins in training.
- **Buys:**
  - a substrate for B8 (generalisation);
  - holding without hand-made latches;
  - a meaningful context for the hippocampus.
- **Costs:** about 20k units and 3M sparse synapses — roughly what the removed PFC loop used (36k units, 3.3M edges).
- **Checks** (offline on recorded streams, then a 50-dialogue smoke run):
  - LEX category purity ≥ 0.25 on the real corpus;
  - GIST Jaccard ≥ 0.5 between paraphrases and ≤ 0.1 between different intents;
  - holding: Jaccard(L+3, end of reply) ≥ 0.8;
  - no period-2 alternation: J(t, t+1) ≥ J(t, t+2);
  - drift: Jaccard ≥ 0.7 across 1k dialogues;
  - all 14 of ARCH_v1's pre-training checks still pass.

**Step 4 — Production through perception (no puppeteer).**
- **Learning:** GIST at the end of Q learns to predict the LEX units active during R, through an eligibility trace across the turn gap.
  - M means "someone is replying", so *every* heard exchange trains it — including dialogues between two other speakers.
  - This is the listener predicting what comes next (Pickering & Garrod 2013).
- **Speaking:** at each boundary, predicted LEX activity drives programs through the dorsal map; the model's own voice re-enters LEX while GIST holds.
- **Babbling can bootstrap the dorsal map without a teacher** (PROVED):
  - 11 % of random 4-muscle sets are exact chars (3,970 / C(32,4) = 3,970 / 35,960);
  - for any set, on average 12.4 chars share 3 of its 4 muscles, and the mouth outputs a char on a 3-of-4 match, so almost every babble is heard as some char.
- **Buys:** real-time learning in use, observational learning, and new words without cues.
- **Risk:** self-reinforcing babble, which M must gate.
- **Checks:**
  - in the nursery world, intents the model has only *observed* between two other speakers must reach ≥ 50 % of the top-1 of intents it was taught;
  - no collapse to a few chars over 10k free replies.

**Step 5 — The hippocampus as an episodic index, with consolidation.**
- **Structure:**
  - **DG:** a fixed convergent expansion from LEX + GIST, 1–2 % active, with a set point. In research/03's simulation, a 1:1 mirror kept a shared core of 0.089, while expansion + set point brought it to 0.017.
  - **CA3:** one-shot co-activity learning.
  - **CA1:** modulatory recall back into LEX and GIST.
- **Forgetting:** a half-life much longer than a dialogue.
- **Offline phases:** every N dialogues, with F off and no teacher, replay recent and old episodes interleaved while LEX and GIST learn slowly. Stop replay-driven learning during waking silence (this fixes problem #10).
- **Buys:** one-shot facts, protection against interference, and slow accumulation of meaning.
- **Checks:**
  - one-shot slot recall ≥ 0.7 after 3 intervening lines, and ≥ 0.5 after 10 dialogues;
  - A→B→A retention at least 0.2 better with replay than without;
  - hippocampal same-line vs different-line Jaccard far above today's 0.18 vs 0.14.

**Step 6 — Novelty and value.**
- **Novelty** = LEX prediction error (an unexpected word). It raises the learning rate and opens the GIST gate, like a noradrenaline-style reset (Chien & Honey 2020).
- **Reward** = the partner's contingent response in stage S4, acting as M on the decision traces. Later it can also teach the GIST gate (PBWM).
- **Checks:** learning speed and retention with vs without each signal, plus an audit for reward hacking.

**Step 7 — A forward model of the own voice, then motor learning.**
- **F′** = window energy − a learned prediction of the own-voice energy from the held program. This is cerebellum-like (Schneider 2018).
- It lets the model hear while speaking, with an abort rule for when the partner resumes.
- **Checks:** pauses of 5–9 ticks → 0 overlapping mouth ticks, and no spurious aborts in 10k turns.

**Order and gating:**
- Do steps 1 and 2 first: they are cheap and unblock everything else.
- Develop steps 3 and 5 offline, in parallel, on recorded streams. Then step 4. Steps 6 and 7 later.
- Every step goes through: unit tests → offline module test → a 50-dialogue integrated smoke run with ARCH_v1's 14 checks → only then a long run. This matches the designer's "先把各个部分结构先做好独立验证，再组合起来".

**Do not change:**
- the body and its audits;
- T as the verbatim buffer;
- the copy route, which becomes the dorsal stream of the new design;
- the panel's tick tables and back-term audits for every new decision path.

---

## 6. My simulations

All runs use numpy with `OMP_NUM_THREADS=1`, each under 3 minutes. They use the same 1,500 never-heard pairs and the same 20-way metric as `arch/content/metric.py`.

**1. Retrieval ceiling** (`retrieval_ceiling.py` → `.out`). Question: how far can generalisation by similar prompts go on this corpus? (SIMULATED)

| policy | top-1 (chance 0.05) | AUC (chance 0.50) |
|---|---|---|
| echo the heard line | 0.273 | 0.666 |
| reply of the most similar training prompt (TF-IDF over chars and bigrams), 1k–40k pairs | 0.080–0.092 | 0.52–0.54 |
| bag of reply chars from the 20 nearest neighbours — mostly echo through the neighbours (copy 0.05–0.12) | 0.089–0.166 | 0.52–0.56 |
| **same bag with the heard chars removed** | **0.055–0.068** | **0.502–0.510** |
| a random training reply | 0.040–0.054 | ≈0.50 |

**2. Semantic structure** (`semantic_sim.py` → `.out`). Question: can a local, one-pass Hebbian learner extract meaning from this corpus? It uses fixed random char codes, co-activity counts and a set point. (SIMULATED)
- **Yes.** Top-5 category purity is 0.278 for a ±2-char context and 0.342 for successor counts — which is what an order rule accumulates. Chance is 0.006, and fixed random codes give 0.005.
- Example neighbours: 我 → 你他她俺; 三 → 两二四五; 红 → 绿颜紫蓝黑; 天 → 今夏明昨.

**3. Semantic retrieval** (`semantic_retrieval.py` → `.out`). Question: does that learned similarity recover reply content beyond echo? (SIMULATED)
- **No.** Top-1 0.057–0.058, AUC 0.504–0.505.

**4. Assembly formation** (`assembly_sim.py` → `.out`). Question: can net.py's rule form assemblies, and is a co-activity variant safe? Setup: 2,000 units, 40 concepts × 12 presentations, top 5 % active, exact weight arithmetic. (SIMULATED)

| variant | per-target totals | half-cue reactivation | hold | update / attractors |
|---|---|---|---|---|
| net.py's exact rule | — | 0.01–0.07 (chance 0.05); inside-concept weight 0.0002 per pair | at chance | at chance |
| forward learning on held targets + back term | — | same as above | at chance | at chance |
| co-activity, budget every 100 ticks | overshoot 6.5× the budget | — | 0.67 → 0.17 within 4 ticks | random starts captured by 5 of 40 concepts |
| **co-activity, budget applied at once** | **≤ 1.000** | **0.48 → 0.37 over 8 ticks** | **0.34 at 16 ticks** | switches to a new concept; 8 of 40 concepts reachable (10 with the set point) |

**Conclusions:**
- The corpus contains learnable meaning: categories emerge.
- But at this scale, almost no reply content depends on meaning. So no small local learner can show B8 on this corpus, and the evaluation world has to change.
- The current rule cannot hold or complete an assembly. Co-activity learning can, provided the budget acts immediately and homeostasis limits capture.

## 7. Where I may be wrong
- **The retrieval ceilings use character-level similarity, raw or learned.** A learner with world knowledge (a human, or a large pretrained model) would get more out of this corpus. My claim is only about small local learners trained on ≤ 40k pairs.
- **`assembly_sim` is small and clean.** Real ear input has large shared cores, and capture depends on parameters. My pass thresholds are first guesses.
- **LEX/GIST is ASSUMED to produce meaning-sensitive codes.** Step 1 and step 3's offline checks can falsify this cheaply, before any long run.
- **The biology comes mostly from rodents, monkeys and songbirds,** and mapping it onto 75-ms binary units is loose. Where biology contradicts a framework rule (#9; "hippocampus = short-term memory"), I give the evidence. But the argument from the final goal, not biology alone, should decide.

---

## summary_zh
1. 身体部分（别人说时不动、轮到时开口、每个字干净地说 3 拍、说完就停）已经可靠，相当于大脑的脑干、基底节、丘脑这些先天结构，保留。
2. 但用大脑来看，v1 是一条反射弧：所有决定只读耳朵第一层（声音能量、T、回声模板）和动作程序自己的延时线；会学习的听觉皮层、运动皮层、海马体，一个也没接到行为上（读代码可证）。
3. 现在叫"前额叶"的 T（固定的随机扩展 + 只在老师示范时学习的读出），结构上其实是小脑、或果蝇的"蘑菇体"：擅长死记听过的句子，但不会抽象。这正是"熟悉的句子 0.57、新句子等于瞎猜"的原因。v1 有大脑里"听→说"的复述通路，却没有"听→懂"的理解通路。
4. 所以，前额叶层数少只是表面现象。关键是耳朵和嘴之间，没有一层层学出来的"字→词→大意"的表示，前额叶没有东西可以想。真实大脑的前额叶最晚成熟，这里也应该最后建。
5. 学习规则：我用模拟验证了，现在的规则学不出"一群一起亮的神经元"（也就是概念、能保持的想法），也跨不过时间间隔。如果在联想区加上"一起亮就加强"，并且预算当场执行、加上设定点，就能形成、保持、切换概念，权重也不会失控。
6. "说什么"只在老师替它动嘴时才学，自由对话时这部分不学，所以现在还不算真正的"实时学习"。
7. 海马体在 v1 里只记大约 15 秒，而且不影响行为。真实的海马体是"一次就记住、能记很久"的情节记忆，靠睡眠时的回放去教皮层。
8. 现在的语料看不出"懂"：我用理想的相似检索、4 万对训练数据，只要不重复对方说过的字，回答内容就等于瞎猜。可同一份语料里，局部规则能学出词类（比如"我"挨着"你、他、她"）。不是学不到，是这种考法看不出来。
9. 建议顺序：
   ① 先造一个"意思可学、可测"的儿童式对话世界，分阶段训练：先听、再咿呀、再模仿、再对话；
   ② 按脑区扩展学习规则：联想区加"一起亮就加强"，决策通路加资格迹和第三因子；
   ③ 加一个词汇层，再加一个变化慢的"大意层"——这才是真正的前额叶，用现有的轮流信号决定什么时候更新、什么时候保持；
   ④ 靠"听"学会怎么回答，再用已学会的"听→说"映射说出来，不再靠老师替它动嘴；
   ⑤ 海马体改做"情节索引"，加离线回放；
   ⑥ 加新奇信号和奖励信号；
   ⑦ 预测并抵消自己的声音，学会让对方先说。
10. 每一步都先离线单独测试，再小跑 50 段对话，过了门槛才做长时间训练。