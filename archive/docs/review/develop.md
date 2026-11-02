(Saved by the coordinator from the reviewer's final reply; the harness blocked the reviewer's own write.)

The harness blocked writing `report.md`, so the full report is below. The scripts and outputs are saved in `/tmp/claude-0/-root/e92d3d7d-5823-4f5c-a0ec-4fe302bff9af/scratchpad/arch/review/develop/`: `corpus_dev.py`/`.out`, `cds_sim.py` with `cds_w0_c0.out`, `cds_w1_c1.out`, `cds_w0_c0_h45.out` (and `.pkl`), and `babble.py`/`.out`.

# Whole-architecture review: the developmental view (reviewer `develop`)

My background is developmental cognitive science and psycholinguistics: how infants and children come to hear, think and talk. I answer the designer's two questions for the whole architecture. My particular focus is whether the training regime can grow understanding at all, and what a realistic developmental path would look like.

Every claim carries a label:
- **PROVED**: an argument with numbers, exact counts, or a direct reading of the code or spec.
- **SIMULATED**: a numpy run; the file and numbers are given.
- **CITED**: a paper and its finding.
- **ASSUMED**: argued but not tested.

All files are in `arch/review/develop/`:
- `corpus_dev.py` → `corpus_dev.out`: a developmental audit of the training corpus.
- `cds_sim.py` → `cds_w0_c0.out`, `cds_w1_c1.out`, `cds_w0_c0_h45.out`: ARCH_v1's content machinery trained on a child-directed curriculum.
- `babble.py` → `babble.out`: babbling coverage and whether articulation maps lawfully onto sound in the world model.

---

## 0. Bottom line

1. **ARCH_v1's body is developmentally sensible; keep it.**
   - The body is the fixed turn-taking circuit (F/GO/CD), the syllable beat, slaved articulation, the END competition, and no mouth movement while listening.
   - Infants have these as early, largely content-free systems before they have language. Proto-conversational turn-taking appears at 3–5 months and canonical babbling around 7 months (CITED: Hilbrink et al. 2015; Oller, in `research/04` §3).
   - Hard-wiring them is a harmless simplification. It is also a good one: it separates "when" from "what".

2. **The current regime cannot grow understanding.** The regime is adult social-media text pairs, played as synthetic sound, heard once, with teacher-forced replies. Four things are missing:
   - **Nothing to refer to.** There are no objects, events or consequences. Meaning (mapping speech to the world and to intentions) cannot be learned from it in principle (CITED: Harnad 1990).
   - **Almost no prompt→reply structure.** The heard line's last char, first char, last two chars, question type or exact identity tells only 0.01–0.03 of the 7.85 bits of the reply's first char. Copying is the only real signal: 14.6 % of reply chars are in the heard line, against 4.3 % for a shuffled line (PROVED by count).
   - **No redundancy.** 99.7 % of (heard, reply) pairs are unique. The 52 most common two-char openings cover 19 % of lines; in child-directed speech, 52 frames cover 51 % of utterances (CITED: Cameron-Faulkner, Lieven & Tomasello 2003).
   - **Too little input.** The whole corpus is ≈ 0.4 M words, i.e. 3–11 weeks of a child's input. The toy's 1,000 dialogues are 0.7–2.4 days of it (PROVED by count; rate CITED from Gilkerson et al. 2017).

3. **Better data alone is not enough.** I trained ARCH_v1's content machinery (exact front end, exact rule, the MEM default) on a clean, repetitive, child-directed curriculum of routines. It still fails constructions that two-year-olds master (SIMULATED, 2 seeds, 2,000 dialogues):
   - Answers like "你要X吗 → 要X" for a never-heard frame–filler combination: **0.00** acceptable. Familiar lines: **0.08 / 0.19**.
   - Naming the object in view: 0.00 from sound alone, 0.15 with a scene input, and **0.55** only when a conjunctive (mixed-selectivity) layer is added.
   - Relational yes/no questions ("这是X吗" about the scene): **0.00** in every variant.

   The causes are structural. The working memory T is a recency-weighted bag dominated by the line's last characters, so frames are confused. The copy latch has no order ("果苹"). Nothing binds the slot of a frame to its filler. The readout is one additive layer over fixed features.

4. **Real-time learning is switched off where it matters.**
   - Decision paths learn only on teacher-driven releases, so in free use nothing about replies is learned (PROVED, ARCH_v1 §1.4).
   - The gate was needed because ungated self-learning drifted (你→的/了 within 200 lines; minimal-learning verifier).
   - Children learn from their own attempts plus contingent social feedback (recasts, responsiveness), from imitation and from overheard exchanges (CITED: Goldstein & Schwade 2008; Chouinard & Clark 2003; Akhtar et al. 2001). No one moves a child's mouth.

5. **The designer's PFC concern is correct but comes second in the order of work.**
   - Early language (first words to item-based frames) develops while the PFC is still very immature (CITED: Huttenlocher & Dabholkar 1997).
   - What it needs first is grounding, a learned lexicon, an ordered working memory of the utterance, binding, and conjunctive codes. My simulation needed a conjunctive layer even for naming.
   - A deep recurrent PFC trained by the present rule on the present input will drift into a generic state again, because there is no recurring latent structure for it to settle on (ASSUMED; consistent with STATE #4–5).

6. **Priority:**
   - (1) a grounded world and a child-directed curriculum;
   - (2) learned representation and structure on the behaviour path;
   - (3) a real learning signal: modelling, imitation and recasts instead of teacher forcing, a contingency/surprise third factor, and an error-driven term;
   - (4) a babbling stage and lawful articulation;
   - (5) sleep-like consolidation;
   - (6) a PFC hierarchy once the curriculum demands it.

   Gate every GPU run on a CPU developmental test bench. `cds_sim.py` already runs 2,000 dialogues in about 80 s.

---

## 1. What I checked

| check | result | label |
|---|---|---|
| Code: `net.py` rule, `net3d.py` (feed-forward fixed, only recurrence and split/delete learn), `voice.py`, `talk.py` (every dialogue heard once; reply teacher-forced) | as summarised in §2 | PROVED |
| ARCH_v1 spec: what learns, what is on the behaviour path, the teacher gate | T reads only fixed ear layer 0. The ear cortex, hippocampus and motor cortex learn but feed no decision. Decision paths learn only under teacher drive. | PROVED |
| Integrated-toy logs `res_base_s0.log`, `res_mem_s0.log` | base control emits memorised whole replies for unrelated lines ("没有加精 → 宿舍", "这么多能行吗 → 学前教育"); MEM echoes with shuffled order ("早上好你好吗 → 学你好上早吗"); never-heard top-1 0.121, familiar 0.425 | PROVED (read) |
| Corpus audit `corpus_dev.out` | §3.1 | PROVED (counts) |
| Curriculum simulation `cds_sim.py` | §3.2 | SIMULATED |
| Babbling in the real world model `babble.out` | 120k random 4-muscle babbles expose 97 % of chars with their exact muscle set. Sound distance and shared muscles are uncorrelated (r = −0.019). | SIMULATED |

---

## 2. How the architecture differs from the brain, and from a developing child

For each difference: does it matter for the final goal (a real-time learner moving toward general intelligence), or is it a harmless simplification?

### 2.1 Body and world

| difference | real child | ours | verdict |
|---|---|---|---|
| **Referents** | vision, touch, objects, faces, gaze, pointing; speech is mostly about the here and now | sound and 32 muscles only | **MATTERS, fundamental.** Without referents only form statistics can be learned (CITED: Harnad 1990). Grounding does not need huge data: 61 h of one child's headcam plus child-directed speech gave word–referent mappings that generalise to new images (CITED: Vong, Wang, Orhan & Lake 2024, *Science*). |
| **Consequences of speaking** | utterances get food, attention and reactions; partners respond contingently | the next line comes from the corpus whatever the model says | **MATTERS.** Contingent responses drive vocal development (CITED: Goldstein, King & West 2003; Goldstein & Schwade 2008; Warlaumont et al. 2014, in `research/04` §3). |
| **Articulation → sound** | the vocal tract is lawful: similar gestures give similar sounds, so babbling builds an inverse model that generalises to new words (DIVA) | each char's sound and its 4-muscle set are independent random draws | **MATTERS** for imitation and vocabulary growth. A heard new char says nothing about how to produce it (PROVED: `voice.py`; SIMULATED: r = −0.019, `babble.out`). |
| **Phonology** | Mandarin has ≈ 400 base syllables (≈ 1,300 with tone) for ≈ 4k chars, with heavy homophony resolved by words and context | 3,970 unrelated holistic sounds, no homophony | Harmless now: hearing is easier than reality. Matters when moving to real audio. |
| **Own vs other voice** | child and adult voices differ greatly in pitch and formants (the "correspondence problem") | own = other + ±1 in 30 % of cells (`voice.py`; STATE.md's "shifted one band" is inaccurate); layer-0 Jaccard 0.89 | Harmless now. Matters later: caregivers solve it by imitating the infant (CITED: Messum & Howard 2015). |
| **Prosody** | stress, final lengthening, pitch at questions and turn ends; used for segmentation and turn prediction | none; turn end is detected only from silence | Moderate, fixable (world change). |
| **Variability** | speakers, noise, coarticulation | each char always identical | Harmless now. |

### 2.2 Training regime

| difference | child | ours | verdict |
|---|---|---|---|
| **Who the speech is for** | child-directed speech (CDS) predicts vocabulary; overheard adult talk does not (CITED: Weisleder & Fernald 2013: 670 to > 12,000 child-directed words/day; overheard speech unrelated to vocabulary at 24 months) | adult social-media talk, which the model "joins" by being puppeted | **MATTERS** |
| **Structure of input** | 51 % of maternal utterances start with one of 52 item-based frames (CITED: Cameron-Faulkner et al. 2003); ≈ 20 % of utterances fall in variation sets (CITED: Küntay & Slobin 1996, secondary source), which help segmentation (CITED: Onnis, Waterfall & Edelman 2008) | top-50 two-char openings cover 19.3 % of lines; same speaker two turns later shares a ≥ 2-char substring in 9.1 %; 47,659 of 47,786 pairs unique (PROVED) | **MATTERS** |
| **Amount** | 2–7 M words/year (CITED: Gilkerson et al. 2017, via the BabyLM call, Warstadt et al. 2023) | whole corpus ≈ 0.42 M words (635,570 chars ÷ ≈ 1.5 chars/word, ASSUMED) = 3–11 weeks; the toy's 1,000 dialogues = 13k words = 0.7–2.4 days | **MATTERS.** Fixable through redundancy: 5 M words of CDS gave a small model grammar knowledge comparable to RoBERTa-base trained on ≈ 6,000× more words (CITED: Huebner et al. 2021, BabyBERTa). |
| **Repetition and spacing** | the same routines daily; spaced exposures help (CITED: Childers & Tomasello 2002) | every dialogue once; in the first 1,000 dialogues the median char type occurs 3 times and only 355 of 1,830 types occur ≥ 10 times (PROVED) | **MATTERS** (fixable) |
| **Teacher forcing** | nobody moves the child's mouth; the child attempts, the adult reformulates (CITED: Chouinard & Clark 2003) | the teacher forces each syllable; decision paths learn only under teacher drive | **MATTERS, fundamental** for real-time learning (see §4 P4) |
| **Order of tasks** | perception, then babbling, then first words with referents, then frames, then grammar and conversation | adult conversation from the first tick | **MATTERS** |
| **Attempt before feedback** | the child answers first and then hears the adult's version (retrieval, then recast) | production is perfect from the start; the model never retrieves on its own in training | Moderate (fixable: a "try-then-recast" protocol, §5) |

### 2.3 Innate vs learned, and developmental order

| item | child | ours | verdict |
|---|---|---|---|
| Turn timing, syllable rhythm | early and largely maturational (CITED: Hilbrink et al. 2015; canonical babbling ≈ 7 months) | hard-wired (F, GO, CD, beat) | **Harmless and good** |
| Syllable inventory | built by babbling, then imitation (DIVA two phases; `research/04` §2.2) | installed by the teacher's cue (recruitment) | **MATTERS.** Untaught chars can never be said (SIMULATED: a never-heard place word 0.00, `cds_*.out`). |
| Imitation | selective (of what is being learned) and itself learned (CITED: Bloom, Hood & Lightbown 1974; Ray & Heyes 2011) | Z map learned from own voice (consistent with the associative account); copy route hard-wired, unordered and context-blind | Partly fine; the copy route **MATTERS** (§3.2) |
| Statistical word segmentation | at 8 months (CITED: Saffran, Aslin & Newport 1996); hippocampal statistical learning in infants (CITED: Ellis et al. 2021) | the ear recurrence can learn "A then B" transition statistics (PROVED from the rule's form; per-target budgets give a backward transitional probability), but it is off the behaviour path and untested on the real ear | **MATTERS** (the first lexical step is unused) |
| Comprehension before production | months ahead (CITED: Fenson et al. 1994; at 6–9 months infants already know common nouns, Bergelson & Swingley 2012) | no comprehension channel; learning only through forced production | **MATTERS** |
| Perceptual tuning | native-language narrowing at 6–12 months (CITED: Werker & Tees 1984; Kuhl 2004) | ear feed-forward fixed | Harmless now (no variability); matters with real audio |
| Growth and pruning (designer's split/delete) | synapses overproduced, then pruned; auditory cortex peaks ≈ 3 months, PFC after 15 months, pruned into mid-adolescence (CITED: Huttenlocher & Dabholkar 1997) | split/delete in the ear; none of it reaches behaviour in v1 (T reads layer 0, which never splits) | Harmless as a mechanism; wasted in v1 |
| PFC maturation | the last region to mature (CITED: Gogtay et al. 2004); executive functions develop over years (CITED: Diamond 2013) | v1's "PFC" is a fixed acoustic hash (T) | See §6 |

### 2.4 Structure

| part | brain | ours (v1) | verdict |
|---|---|---|---|
| **Lexical / conceptual layer** | word forms and semantic representations learned in temporal cortex | none on the behaviour path; all learned decision weights read fixed codes (T, Hc, latch) | **MATTERS, fundamental.** Generalisation is only by acoustic overlap of line endings (PROVED by spec; SIMULATED frame confusion, §3.2). |
| **Working memory of the heard utterance** | item plus order (phonological loop; GODIVA's primacy-ordered plan, `research/04` §2.3) | T is a recency-weighted bag: the last char plus its fade is ≈ 55 % of T (measured by the learnability proposal) | **MATTERS.** Hold 4.5 instead of 3.0 lifts never-heard first syllables from 0.76 to 0.95 (SIMULATED, `cds_w0_c0_h45.out`). |
| **Binding of slot and filler** | hippocampal item–context binding; PFC holds structured representations | none | **MATTERS** (fundamental) |
| **Conjunctive (mixed-selectivity) codes** | the PFC is dominated by mixed selectivity, needed for context-dependent tasks (CITED: Rigotti et al. 2013) | T expands one stream only; the readout adds contributions from separate sources | **MATTERS.** Naming 0.15 (scene alone) → 0.55 with a mixed layer (SIMULATED). A purely additive readout cannot compute "heard X matches seen Y" (PROVED: if a₁+b₁>θ and a₂+b₂>θ then a₁+b₂ and a₂+b₁ cannot both be <θ). |
| **Basal ganglia** | action selection plus dopamine-driven reinforcement of what worked | WTA and GO, with a teacher gate instead of reinforcement | **MATTERS** for learning from one's own actions |
| **Hippocampus** | fast episodes, ordered replay, fast mapping, sleep replay | learns but feeds no behaviour | **MATTERS**: it is the natural ordered short-term store (phonological loop) and episode memory, and it is idle |
| Thalamic loop, cerebellum | holding, timing, forward models | dropped / absent | Harmless now |
| Localist syllable programs | DIVA has one speech-sound-map node per learned syllable | one program per char | Harmless as such; what matters is how programs are created (by teacher, not by babbling or imitation) |

### 2.5 Dynamics
- Binary units, 75 ms synchronous ticks and fixed active fractions are a reasonable level of abstraction: **harmless**.
- The latency of 6 ticks (≈ 450 ms) is slower than adults (≈ 200 ms) but within the child range; young children are slower and get faster over years (CITED: Casillas, Bobb & Clark 2016). **Harmless now.**
- The turn system cannot yield or predict a turn's end. In humans, turn-end projection rests on understanding. **Later.**

### 2.6 Learning rules and neuromodulators
- **One order rule with a timing law.** Held sources cancel, so every learned source must be a gated one-tick pulse (PROVED by the panel). Workable: the same gating trick works for a scene (scene ∧ GO), and biological sampling is pulsed (fixations, attention shifts) (ASSUMED). **Fixable.**
- **No error-driven term.** Children's language learning is strongly prediction-error driven. Implicit negative evidence comes from expectations that fail (CITED: Ramscar et al. 2010; Ramscar, Dye & McCauley 2013), and infants learn more from surprising events (CITED: Stahl & Feigenson 2015).
  - Our rule never depresses a link when the predicted target does not occur, so copy weights are frequency counts (PROVED in CONTENT §3.4).
  - Per-target budgets create frequency artefacts: frequent targets are normalised away and rare ones win. In my curriculum "鸡" became the first syllable for every taste question (SIMULATED, trace in `cds_sim.py`, trace=1).
  - **MATTERS** (fundamental for stable real-time learning).
- **No value or social third factor.** Speech learning is socially gated: a live tutor works where a video does not, and in songbirds dopamine carries the error relative to the tutor's song (CITED: Kuhl, Tsao & Liu 2003; `research/05`). Without a third factor there are only two options, both seen in this project: freeze learning (v1's teacher gate) or let it drift (the ungated 你→的/了 drift). **MATTERS, fundamental.**

### 2.7 Memory and consolidation
- Fast plus slow first-syllable paths form a complementary-learning-systems pair, which is good. Fast mapping followed by poor retention without repetition is also child-like (CITED: Horst & Samuelson 2008). **Consistent.**
- **No sleep or offline consolidation.**
  - In silence the hippocampus replays and changes ear/hippocampal weights without control (ARCH_v1 idle test: |Δw| up to 1.0 / 6.9).
  - Infants' naps consolidate and generalise what they learned about language (CITED: Gómez, Bootzin & Nadel 2006; Friedrich et al. 2015).
  - **Matters moderately** (fixable).

---

## 3. Can the current regime grow understanding and thinking? Evidence

### 3.1 The corpus as a child's input (PROVED by count, `corpus_dev.out`)

| measure | adult corpus | what children get |
|---|---|---|
| size | 32,971 dialogues, 635,570 chars, 3,970 types, 47,786 turn pairs | 2–7 M words per year (CITED) |
| toy run (1,000 dialogues) | 19,396 chars (10,402 heard), 1,830 types, median 3 tokens per type | a day or two of input |
| repetition | 11.1 % of lines repeat an earlier line; 99.7 % of pairs unique; only 112 pairs occur twice or more | routines repeated daily |
| frames | top-50 initial 2-char frames 19.3 %; top-50 final 2-char 11.8 % | 52 frames cover 51 % (CITED) |
| prompt → reply information | first reply char 7.85 bits; given last char / first char / last two chars / question type / exact line: 0.02 / 0.03 / 0.03 / 0.01 / 0.01 bits (held-out, dev-tuned interpolation) | adjacency pairs (greeting–greeting, question–answer, offer–accept) are highly predictable |
| copying | 14.6 % of reply chars occur in the heard line (shuffled 4.3 %) | 2-year-olds' answers often reuse the question's words |
| rituals | 晚安 / 早安 / 谢谢 / 你好 / 生日快乐: 0.8–3.3 per 1,000 pairs each, and the reply fits the ritual in only 19–48 % of cases | daily |
| questions | yes/no questions 5.3 % of pairs; answered with a typical answer char 34 % (other lines 16 %) | ≈ a third of CDS utterances are questions (ASSUMED, typical CDS figure) |

**Reading.**
- Even an ideal statistical learner gets almost nothing from simple features of these lines, apart from copying.
- What children rely on is absent: referents, repetition, frames and contingency.
- The base-control behaviour is what a learner that can only memorise produces on such input: memorised whole replies emitted out of context ("宿舍", "学前教育", "嘘隔壁"), which resemble delayed echolalia.
- With the copy route it produces shuffled immediate echoes ("学你好上早吗").
- In children, echolalia and gestalt phrases are normal transitional phenomena. They resolve through segmentation, recombination and grounding (CITED: Peters 1983, *The Units of Language Acquisition*). Those are exactly the mechanisms v1 lacks.

### 3.2 The same machinery on child-directed input (SIMULATED, `cds_sim.py`)

**Set-up.**
- **Front end:** the content specialist's exact front end, imported read-only from `arch/content/fe.py`: exact ear layer 0 with reserve, FEAT0, T (4000/200, hold 3.0, set point), echo map Z (own-voice gate) and latch.
- **Decision layer:** their exact decision layer (`decide.Decider`): the exact rule with back term, decay and budgets.
- **Reply rule:** the MEM boundary rule (P05 × 10 + Tslow × 30, copy E3, summed-drive stop, used flag), which is ARCH_v1's adopted default.

**Curriculum** (child-directed style; all 63 chars exist in the corpus):
- **Frames:** 你要X吗→要X/我要X; X好不好吃→好吃/X好吃; 你喜欢X吗→喜欢X/喜欢; X在哪里→在这里; 我们去X吧→好去X/好的.
- **Naming:** 这是什么→X/是X, where X is the object in view.
- **Relational check:** 这是X吗→是的/是X if the object in view is X, otherwise 不是.
- **Rituals:** six.
- **Fillers:** 20 objects and 5 places.
- **Held out:** 30 % of each frame's fillers are never heard in that frame (they still appear in other frames), and one place is never heard at all.
- **Scene:** a sparse code for the object in view, gated into a one-tick pulse at the P tick (like P = T ∧ GO). Optional "mixed" layer: top-100 of a fixed random expansion of (T, scene) jointly.
- **Runs:** 2,000 dialogues (≈ 2,900 pairs). Two world/curriculum seeds, plus a T-hold-4.5 variant. Score = the reply is one of the acceptable answers.

| test (2,000 dialogues; seed 0 / seed 1) | A: ARCH_v1 MEM | B: + latch set point .1 | D: no copy | E: B + scene | F: B + scene + mixed |
|---|---|---|---|---|---|
| familiar lines, acceptable (first syllable) | 0.08 / 0.19 (0.85 / 0.85) | 0.12 / 0.35 | 0.23 / 0.23 | 0.15 / 0.31 | 0.15 / 0.31 |
| never-heard frame × filler, acceptable (first syllable) | **0.00 / 0.00** (0.76 / 0.81) | 0.00 / 0.00 | 0.29 / 0.29 (fixed-answer frames only) | 0.00 / 0.00 | 0.00 / 0.00 |
| naming 这是什么 (needs the scene) | 0.00 / 0.00 | 0.05 / 0.05 | 0.05 / 0.05 | 0.15 / 0.15 | **0.55 / 0.55** (first syllable 0.70 / 0.70) |
| relational 这是X吗 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | **0.00 / 0.00** |
| novel place word | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

**Other findings:**
- **Acceptable replies decline with training in A.** Familiar lines go 0.27 → 0.23 → 0.19 → 0.08 at 250 → 500 → 1,000 → 2,000 dialogues (seed 0), as the copy weights of function chars grow.
- **Failure mode 1, run-on.** Example: "你要面条吗 → 要面条你好吃". The leftover latched 你 (copy weight 0.38) keeps the summed drive above END after 条 (0.22); the chain 你→好→吃 then follows the ritual "你好".
- **Failure mode 2, scrambled order.** Example: "我果苹". The latch is a set, so copy order comes from the size of the copy weights.
- **Failure mode 3, frame confusion.** "这是X吗" is answered like "你要X吗" ("我是苹果"), because T is dominated by "X吗".
- **T hold 4.5.** Familiar first syllable 1.00 and never-heard 0.95, but whole replies are still wrong. Naming with the mixed layer reaches 0.70.
- **Relational check.** At best a yes-bias (0.23); 2-year-olds show the same yes-bias (CITED: Fritzley & Lee 2003).
- **Scope of these numbers.** They come from one curriculum and the parameters the panel tuned on the adult corpus. The failure modes follow from the design (unordered latch, recency bag, additive readout, no binding), so I expect them to survive retuning (ASSUMED, but each is PROVED as a structural property).

### 3.3 Verdict
- **Understanding: no.**
  - Nothing in the regime refers to anything (PROVED).
  - The corpus has almost no learnable prompt→reply structure besides copying (PROVED).
  - The machinery cannot learn even the item-based constructions of a 2-year-old from ideal input (SIMULATED).
- **Thinking: no.** The behaviour path holds no learned representation in which a "thought" could form (PROVED by spec).
- **Distributional route.** Text statistics alone can yield meaning-like structure at billions of words (whether that is "understanding" is debated: CITED Bender & Koller 2020 vs Piantadosi & Hill 2022). Our input is 4 × 10⁵ words and the readout is one counting layer, so this route is closed too (PROVED by orders of magnitude).
- **What v1 is, developmentally:** proto-conversation, then rote gestalt memory, then echoing. That is roughly an infant–toddler profile, without the engines that move children beyond it.

---

## 4. Problems of the whole architecture, ranked by how much they block the final goal

| # | problem | kind | seen? | evidence |
|---|---|---|---|---|
| P1 | **No referential world; ungrounded, unrepetitive, adult, undirected input.** Understanding cannot be learned. | fundamental (world and regime) | seen | §3.1 PROVED; CITED |
| P2 | **No learned representation on the behaviour path.** Every decision weight reads a fixed acoustic hash, a program copy or a latch, so generalisation is only by overlap of line endings. The ear cortex, hippocampus and split/delete learn but are unused. | fundamental (architecture) | seen | PROVED (ARCH_v1 §1.4); SIMULATED frame confusion |
| P3 | **No structure:** an ordered memory of the heard utterance, slot–filler binding, and a conjunctive layer are all missing. Frames ("要X") and naming fail even with ideal input; relational questions are impossible for an additive readout. | fundamental (architecture: PFC and hippocampus) | seen (my simulation) | SIMULATED §3.2; PROVED (additivity) |
| P4 | **Learning signal.** Decision learning happens only under teacher forcing and is frozen in free use, so the model learns nothing about replies while it talks. Without a gate it drifts. There is no error-driven term and no social/value third factor. | fundamental (learning rule and regime) | seen | PROVED (spec); verifier drift; CITED |
| P5 | **No path to new words.** Programs are teacher-installed and articulation→sound is arbitrary, so the model cannot learn a word by hearing it, cannot say untaught chars, and imitation cannot generalise. | fundamental for vocabulary growth (world and motor) | seen | SIMULATED (novel word 0.00; r = −0.019) |
| P6 | **Amount and redundancy of input:** the corpus is weeks of a child's input and 99.7 % of pairs are unique. | fixable (curriculum, repetition, interleaving) | seen | PROVED; CITED |
| P7 | **Content-route details:** recency-dominated T; run-on stop against leftover copies; frequency artefacts of per-target budgets. | fixable | seen | SIMULATED |
| P8 | **Grounding meets the timing law:** a held scene is unlearnable and must be gated into pulses (attention shifts). Cross-situational learning will inherit the budget artefacts. | fixable (needs a gate) | not yet seen | PROVED (Lemma 1) |
| P9 | **Learning from own speech, once enabled:** self-reinforcement drift unless a feedback/surprise third factor gates it. | fundamental if ignored | partly seen | verifier SIMULATED |
| P10 | **When the PFC learns again:** generic-state drift returns unless the input has recurring latent structure and the PFC has gating and set points. | fundamental if ignored | seen before (STATE #4–5) | ASSUMED (for the new input) |
| P11 | **No consolidation; uncontrolled replay in silence.** | fixable | seen | ARCH_v1 idle test |
| P12 | **Metric:** the 20-way score rewards echo (whole-line echo scores 0.24–0.27); there is no stage-appropriate test battery. | fixable | seen | PROVED |
| P13 | **Later or harmless now:** program pool filling with babbled non-words (needs pruning); realistic voices will break the own→other Z transfer (Jaccard 0.89 today); no prosody; cannot yield; fixed latency; no homophony. | fixable / later | not yet seen | ASSUMED |

---

## 5. What to optimise, in priority order

### 5.1 The improved overall architecture (block level)
```
 WORLD: partner (speaks, looks/points, recasts, reacts) | scene (objects, events) | own body (mouth; later gaze/hand)
          | sound                                        | sight (onset/attention pulses)          ^
   [EAR 3D net] layer 0 fixed; cortex learns chunks   [EYE 3D net] layer 0 fixed                 |
          \________________________   ____________________/                                      |
                                   v v                                                           |
   [LEXICON/CONCEPT layer] learned (order rule + split/delete): word chunks; word<->object links |
          |                                   |                                                  |
   [HIPPOCAMPUS] <---------------> [PFC-1 utterance working memory: item + order]                |
    fast episodes; ordered replay   [PFC-2 mixed selectivity: context x content x scene]         |
    (= phonological loop);          [PFC-3 slow context/goal: routine, topic (Stage D)]          |
    sleep replay -> slow paths                |                                                  |
          \____________________   ____________/                                                  |
                               v v                                                               |
   [BG: gates (P = WM ∧ GO, scene ∧ GO), WTA over programs]  <-- THIRD FACTOR: partner contingency/
          |                                                       success + surprise (prediction error)
   [ARCH_v1 BODY: F, GO, CD, beat, programs Y, END, D/I/M] ----> MOUTH ----> own voice ----> EAR
           ^ programs created by babbling + imitation (Z map: own program <-> own sound)
```
What stays from ARCH_v1:
- the body;
- the gating trick that turns held states into one-tick pulses;
- the fast and slow memories;
- the one rule, extended by an error term and a third factor.

### 5.2 Changes in priority order (what each buys, what it costs, how to check before any long run)

| # | change | buys | costs | pre-run check (CPU bench, minutes) |
|---|---|---|---|---|
| O1 | **Grounded world plus child-directed curriculum.** A scene channel (a small second Net3D for "sight"); a scripted tutor running here-and-now routines (naming, offers, requests, where/what questions, greetings) with variation sets and spaced repetition; adult corpus only at the last stage, interleaved. | referents; learnable adjacency pairs; redundancy; a meaningful test target | world and tutor code; one more sensory area (a sense, not a "speech area") | extend `cds_sim.py`: naming ≥ 0.8; frames on never-heard fillers ≥ 0.7 |
| O2 | **Representation and structure on the path:** (a) the ear cortex as a lexical layer (test segmentation first); (b) an ordered utterance memory, e.g. hippocampal ordered replay gated to boundaries, replacing the set latch; (c) slot binding, i.e. copy the segment the context selects (the attended object's word), not every heard char; (d) a mixed-selectivity layer over (context, content, scene). | frames with slots in the right order; naming; generalisation by category instead of acoustic ending | learned layers must stay stable (set points, budgets); this is where v1 removed learning, so re-add it one piece at a time | bench: "要X" on never-heard X ≥ 0.7, correct order, no run-on; frame decoding independent of filler; a Saffran-style test of the lexical layer (within-word vs part-word transitional prediction) |
| O3 | **A real learning signal:** (a) replace teacher forcing with modelling plus imitation (the tutor or a third voice says the reply; the auditory mirror drives the program covertly, the same learning event without moving the mouth); (b) "try then recast": the model answers freely, the tutor recasts; (c) a third factor = contingent success plus surprise gating decision-path plasticity; (d) a local error-driven term: when a target's own input predicted it but it did not fire, depress the active inputs (delta-rule form, local to the target). | learning in free conversation (real time); training = deployment; implicit negative evidence; copy probability instead of counts; fewer frequency artefacts | a change to the rule (stated precisely and kept local); a tutor policy | bench: free-speech learning improves without forcing; repeat the 你→的/了 drift test with the gate replaced by the third factor (must not drift); a rare-vs-frequent target retention test |
| O4 | **Babbling stage and lawful articulation:** make each muscle contribute a fixed spectral pattern, so a char's sound is a function of its muscle set (voice.py); babble before dialogue; the tutor mirrors the babble (Messum & Howard). | a full program inventory and Z map without a teacher; generalising imitation; learning words by hearing them | a world change; ≈ 100k babbled syllables (≈ 300k ticks; SIMULATED coverage 97 %, `babble.out`) | coverage: Z winner ≥ 0.8 for all chars; imitation of chars never taught |
| O5 | **Sleep-like consolidation:** an offline phase in which hippocampal replay of the day's episodes trains the slow paths (gated), plus wall-clock budgets. | retention and generalisation; controlled replay | scheduling | CONTENT §4 retention curves with and without sleep |
| O6 | **PFC hierarchy (Stage D):** slow recurrent context/goal layers updated at event boundaries (pulses), gated by the basal ganglia; later inner speech (covert programs → forward model → ear → hippocampus). | relational and multi-step tasks; "thinking" | the largest piece; risk of drift (P10) | tasks: 这是X吗 on novel combinations ≥ 0.8; hidden-object questions; B2 hold |
| O7 | **A stage-appropriate test battery** replaces the echo-rewarding metric as the main target. | correct incentives | small | itself |

### 5.3 The realistic developmental path (stages, with milestones that gate the next stage)

| stage | (child age analogue) | input | milestone that must pass |
|---|---|---|---|
| A. Hearing and babbling | 0–8 months | CDS-like stream without replies; self-babbling; tutor mirrors the babble | lexical layer shows within-word > across-word prediction; Z map covers ≥ 90 % of chars; B1 holds |
| B. Joint attention, first words | 8–15 months | naming routines with scene pulses; contingent turns; recasts | comprehension (choose the named object) ≥ 0.8; naming ≥ 0.8 |
| C. Item-based frames | 15–30 months | offers, requests, where/what, greetings; variation sets | never-heard frame × filler ≥ 0.7; correct order; stop by END |
| D. Relations and memory | 2–4 years | 这是X吗, "球去哪了" after a hidden move, who-said-it, two-step requests | relational on novel combinations ≥ 0.8; hidden-object ≥ 0.8 |
| E. Breadth | later | adult dialogue interleaved with earlier stages | never-heard adult 20-way above echo-only with parroting ≤ 2× human; earlier skills retained ≥ 90 % |

GPU runs are needed only from stage C on. Stages A–C fit on the CPU bench: the front end plus decision layer runs about 2,000 dialogues in 80 s.

---

## 6. The designer's PFC concern

- **Right about the brain.** The PFC is a multi-layer recurrent system:
  - mixed-selective units (CITED: Rigotti et al. 2013);
  - gated updating via the basal ganglia (PBWM, `research/01`, `05`);
  - slower, more abstract levels toward the front;
  - very slow maturation (CITED: Huttenlocher & Dabholkar 1997; Gogtay et al. 2004).
- **But it is not the first bottleneck.** In development, words, naming and item-based frames come before most PFC maturation. They rest on perception, hippocampal learning, the basal ganglia and a small working memory.
- **Our first bottlenecks** are P1–P3: no referents, no lexicon, no ordered memory or binding.
- **A deep recurrent PFC on today's input would fail the same way the old loop did.** That loop drifted into the common "after-line" state, because Hebbian order-learning in a loop amplifies whatever recurs in every dialogue, and in ungrounded, unrepetitive adult pairs the only thing that recurs is the generic state (ASSUMED; consistent with STATE #4–5).
- **Build the PFC in developmental order:**
  1. **Now:** PFC-1, an ordered utterance memory (item plus order, not a recency bag), and PFC-2, a mixed-selectivity layer over (context, content, scene). These are cheap fixed random layers that are learned through readouts. My simulation needed PFC-2 even for naming: 0.15 → 0.55.
  2. **With the third factor (O3):** basal-ganglia gating of what enters and leaves PFC working memory.
  3. **Stage D:** PFC-3, slow context/goal layers. Their units should change at event boundaries (one-tick pulses, satisfying the timing law), be trained by tasks that need them, and be checked for drift on the bench.
  4. **Later:** inner speech as a thinking loop (covert programs, forward model to the ear, hippocampal "hearing in the mind"), which the framework's returns already allow (CITED for the development of inner speech: Alderson-Day & Fernyhough 2015).
- **Concept neurons without returns** (a framework idea) fit PFC-2 and PFC-3.

---

## 7. summary_zh

1. 身体部分（轮流说话、按拍发音、说完会停、听时嘴不动）和婴儿很早就有、与内容无关的本能相符，保留。
2. 现在的训练方式长不出"理解"：语料里没有任何可以指称的东西（没有物体、事件、后果）；成人微博式对话几乎没有规律，听到的句子对回答第一个字只提供 0.01–0.03 比特（共 7.85 比特），只有"重复听到的字"这一条信号；99.7% 的对话对都只出现一次；整个语料只相当于孩子 3–11 周听到的话，训练 1000 段只相当于 1–2 天。
3. 即使换成理想的"妈妈式"语料（固定句型、大量重复、就事论事），现在的内容机制也学不会两岁孩子的句型："你要X吗→要X"在没听过的组合上 0%，熟悉的也只有 8–19%；会出现停不下来（"要面条你好吃"）、顺序打乱（"果苹"）、句型混淆（把"这是X吗"当"你要X吗"答）。
4. 原因在结构：行为通路上没有学来的表征（T 是固定的声学哈希，以句尾为主）；记住听到的话时没有先后顺序；没有"句型+空位"的绑定；读出只是一层加法，算不了"看到的是不是听到的"。
5. 实时学习在关键处是关着的：说什么的连接只在老师推嘴时学，自己说话时什么都不学；孩子是靠自己尝试、大人及时回应和改说、模仿和旁听来学的。
6. 世界设计的问题：字的声音和发音肌肉是随机对应的，咿呀学语无法推广，没教过的字永远说不出；没有语调；没有视觉等其他感觉。
7. 前额叶太简单的担心是对的，但在发育顺序上不是第一位：先要有可指称的世界、词汇层、有顺序的记忆和绑定；在模拟里，命名也要加一层"组合"表征（前额叶式的混合选择性）才从 15% 升到 55%。
8. 优先顺序：①有物体场景的世界+妈妈式分阶段课程；②行为通路上加学来的词汇层、有序记忆、空位绑定、组合层；③用示范/模仿/改说代替推嘴，加"对方回应+意外"的第三因子和误差驱动的学习；④先咿呀学语，并让肌肉和声音有规律地对应；⑤睡眠式巩固；⑥课程需要时再建多层前额叶。
9. 每一步先在 CPU 小测试台上过关（cds_sim.py 两千段对话约 80 秒），再上显卡长跑。