(Saved by the coordinator from the reviewer's final reply; the harness blocked the reviewer's own write.)

# Review round 2: the developmental lens (reviewer `develop`)

The harness blocked `report.md`, so the full report is below. Scripts and outputs are in `arch/v2/review2/develop/`. Labels follow the brief: PROVED, SIMULATED, CITED, ASSUMED. Every run was local, one process at a time, ≤ 270 MB and ≤ 11 min.

## 0. Bottom line

1. **ARCH_v2 has the kit of a one-year-old.** It has syllable categories, an inverse map from babbling, turn-taking with 0 overlaps, joint attention, a held thought and awake recall. It answers practised questions almost perfectly: seen items score 0.95–1.00 in my frozen probes (D1).

2. **One logical flaw drives most of what is still wrong.**
   - Speaking means saying what the agent predicts *another voice* will say next. That prediction does not know whose turn it is.
   - After the agent answers, the next line is the partner's next question. ARCH_v2 learns that question as the answer. It also treats the missing confirmation as negative evidence.
   - Children learn what the *responder* says (role-reversal imitation). A chat language model trained without role tokens behaves exactly like ARCH_v2: it continues with the user's next question.
   - This one flaw is the common cause of free-use drift, of "initiative says the partner's lines", of the turn learners' misfires, and (my run shows) of most of the held-out failure.
   - **A minimal local fix:** a heard line that the agent then answers was a question, so its learning as an answer is cancelled. Paired on seed 0 (nS 600), this raised staged J from 0.70 to 0.93, held-out H from 0.29 to 0.54, and the drift level from 0.28 to 0.67 (loss 0.43 → 0.03). SIMULATED, D5.

3. **Several headline numbers do not mean what they claim.**
   - Life free replies (0.34) score below a responder that echoes or gives one fixed answer per frame. That responder gets 0.51 on the same items (PROVED).
   - H ≈ 0.2 hides a 0.00 on `taste`, where every answer is 好吃.
   - Life memory 0.00 measures a question routine that life never models, not memory.
   - "Favourites 0.81" follows from the form of R11.
   - "No drift in life" holds because life gives recasts for its whole length.

4. **The worlds provide a tutor, not a caregiver.**
   - The bench's content never depends on the child.
   - The partner never follows the child's attention.
   - Life ignores the child's questions and never prompts routines.
   - In world2 the agents mostly talk to each other.
   - Contingent, child-led input is the strongest environmental predictor of early language (CITED: Tamis-LeMonda, Bornstein & Baumwell 2001; Romeo et al. 2018). The agent has no curiosity with which to elicit it.

5. **Ranked changes:**
   1. role-aware learning by listening, plus speaker identity;
   2. comprehension feeds production;
   3. contingent, child-led caregiving plus curiosity-driven asking;
   4. trust-weighted social learning before many bodies;
   5. online, lawful, socially shaped babbling.

## 1. What I ran

Unless stated otherwise, the agent is the saved seed-0 final agent `synth/out/agent_final_s0.pkl`. This is ARCH_v2 with CTX still in; CTX only affects name recall.

| id | files | what | key result |
|---|---|---|---|
| D1 | `heldout_replies.py/.out/.json` | Every reply in frozen B/C/D probes (80 dialogues each, held_frac 0.3). | Seen items ≈ 1.00. Held-out: taste 0/7, offer 0/14, what 0/24 (21 replies were "这是什么"), where 1/17, slot 1/17. |
| D2 | `comprehend.py/.out` | For X在哪里 / X是什么颜色 / 有几个X: the slot the reply thought attends after its comparator step, against X's true slot (150 B + 150 C dialogues). | Attends X 1.00 in every group. Held-out "where": reply 0.00 (n 30). |
| D3 | `vr_what.py/.out` | On 这是什么 + pointing: does visual recognition (VR) at the pointed slot hold the object's name? (Name categories read through the agent's own ear.) | Held-out: VR holds the name 0.73 (n 44), reply correct 0.02. |
| D4 | `drift_replies.py`, `drift_veto.py` (`.out`, `.json`) | run_j's drift protocol on the saved agent, wrong replies classified, with and without the veto. | ok 0.79 → 0.51 without the veto; 0.79 → 0.77 with it. |
| D5 | `veto_j.py`, `veto_j_runs.json`, `veto{0,1,2,3}_s0.log`, `cmp_tasks.py` | The full staged protocol. `run_j.py` is unchanged; ARCH_v2 keys; seed 0; nS 600. The veto is built into a SynthAgent subclass (deepcopy-safe). | See §4.1. |

## 2. Q1: ARCH_v2 against the developing brain

| difference | child / brain | ARCH_v2 | matters for the goal? |
|---|---|---|---|
| **Roles in conversation** | Proto-conversation by 3–4 months (Hilbrink, Gattis & Levinson 2015). Role-reversal imitation in one-year-olds (Carpenter, Tomasello & Striano 2005). I/you learned from overheard dialogue between others (Oshima-Takane 1988). CITED. | One category, "another speaker". No speaker identity. No role for a heard line. Production target = whatever anyone else says next. Own speech does not enter the thought, and hippocampal NEXT links skip own lines. | **Yes, most.** Drift, peers in world2, pronouns in the adult corpus. |
| **"Moving on" means acceptance** | Parents reformulate errors and simply continue after correct utterances (Chouinard & Clark 2003, CITED). | R2 reads the confirmation that does not come as "predicted but absent → weaken". Silent replies after drift: 0.17 → 0.38 (D4). | Yes. |
| **Comprehension feeds production** | Context-bound early words become referential by about 18 months (Bates et al. 1979; Barrett 1995). Pivot schemas ("more X") extend at once to new fillers (Braine 1963). Item-based frames become abstract with type frequency (Tomasello 2003; Bybee 1995). CITED. | The answer is read from recruited frame × filler conjunctions. What comprehension knows (PRIME/VR: word ↔ appearance; attD: word → slot) enters only as conjunction features. D2/D3: the agent knows where X is (1.00) and what it is called (0.73) but cannot say it. | **Yes.** Productivity is what an adult corpus needs. |
| **Generalising cortex vs memorising hippocampus** | Complementary learning systems: a fast, separating hippocampus and a slow, overlapping cortex (McClelland, McNaughton & O'Reilly 1995, CITED). | MIX recruits one unit per new combination, which is a second pattern-separating memory. Awake recall replays into it. IDEAS #5 (things used alike get similar codes) is not built. The designer's 2026-10-31 addendum (random MIX at birth, trained on top, pruned) targets exactly this. | Yes. |
| **Motivation: curiosity and communicative intent** | Infants point to request and to share (Bates, Camaioni & Volterra 1975; Liszkowski et al. 2004). 2–4-year-olds ask many information questions and learn from the answers (Chouinard 2007). Information gaps boost learning (Gruber, Gelman & Ranganath 2014). CITED. | The intention is confidence about what to say. There is no need, no information gap and no goal. Initiatives are the partner's anticipated lines. | **Yes.** An open world schedules no lessons. |
| **Joint attention in both directions** | Caregivers mostly name what the child attends to or holds, and follow-in naming predicts vocabulary (Tomasello & Farrar 1986; Yu & Smith 2012). The child also directs adults' attention (Liszkowski). CITED. | An innate, perfect pointing code for the tutor's focus. The agent has no gaze or point output in bench or life, so follow-in naming cannot happen. | Yes, for an open world. |
| **Social gating and trust** | Infants learn phonetics from live tutors, not video (Kuhl, Tsao & Liu 2003). Preschoolers prefer accurate and adult informants (Koenig & Harris 2005; Jaswal & Neely 2006). CITED. | Every non-own voice teaches equally. | **Yes, for many bodies.** |
| **Speech perception** | Statistical word segmentation by 8 months (Saffran, Aslin & Newport 1996). Speaker normalisation (Kuhl 1979). Prosody marks questions and turn ends. CITED. | One syllable = one item; no word unit. Syllables are exactly 3 ticks with a phase-reset clock. Lines are delimited by silence. No prosody and no level or speaker normalisation; world2 categories hit the 256 cap in a day. | Yes for real audio and world2; no on the bench. |
| **Babbling** | Months of babbling, shaped by caregivers' contingent replies (Goldstein, King & West 2003; Goldstein & Schwade 2008). First words come from the child's own babble repertoire (McCune & Vihman 2001). CITED. | One offline pass over 35,960 muscle sets through `world.voice` gives a complete map of 3,970 sounds before any word. Binding a heard syllable to a babbled sound is Vihman-like (good); the enumeration is not. | Yes for honesty and real bodies; no for bench numbers. |
| **Amount of input** | About 12k adult words per 12-hour day (Gilkerson et al. 2017, CITED). | Life: partner speech fills 44,388 of 1,020,942 ticks (4.3 %), 3,967 heard lines in 21 h, from a 67-syllable vocabulary (PROVED from the life JSON). | Not a flaw now. But one life ≈ one toddler-day of words, and claims should be scaled to that. |

Differences that do not matter for the goal now:
- No sleep. Awake replay exists (Karlsson & Frank 2009), and awake recall does the work here.
- The innate turn reflexes. Infants have proto-conversational timing (round-1 consensus).
- Binary units, innate retinotopy, the 75 ms tick.

## 3. Q2: Problems of the whole architecture

### 3.1 Logically unsound

**(a) The production target does not know whose turn it is.** PROVED from the code path, SIMULATED in D4 and D5.
- The thought keeps the question across the agent's own reply (`ct_own` = 0).
- `end_line` links hippocampal NEXT only between heard lines.
- So the first line another voice says after the agent's reply is learned as the answer to the question. In recast mode that line is the recast (useful). In any world without feedback it is the next question (harmful).
- D4, drift on the saved agent:
  - Replies that are partner lines rose from 0.03 to 0.16 in the last third of drift ("有几个西瓜" and "小猫是什么颜色" given as answers).
  - Silent replies rose from 0.17 to 0.38 after drift.
  - ok fell from 0.79 to 0.51.
- The same cause accounts for:
  - open problem 1 (drift);
  - open problem 3: 703 "nonsense" initiatives with the partner present in life, mostly his openings;
  - open problem 7 (turn learners);
  - open problem 9 (social replies);
  - the held-out fallback: in D1, 21 of 24 held-out 这是什么 were answered "这是什么" (immediate echolalia, the typical toddler response to a question not understood).

**(b) "Moving on" is punished.** "Predicted but absent → weaken" fires each time the partner moves on. One event lowers the answer's drive by about η·p ≈ 0.3 (ARCH_v2 §7.3). Children read moving on as acceptance, not correction (Chouinard & Clark 2003).

**(c) Recall consolidates the confusion.** Awake recall replays NEXT links of the form "question → next question". In D4 the fix also had to cut those links (619 cut).

**(d) The readout is context-bound, and generalisation is set by type frequency.** PROVED for an idealised NLMS readout started at zero. Such a readout converges to the minimum-norm solution.
- Setup: K trained fillers per frame. Each event has S frame-only units and F units tied to the filler.
- Then the drive on a never-heard filler is K·S / (F + K·S).
- Most pair conjunctions touch the filler (its syllables, 8 focus units, VR), so F ≫ S (ASSUMED). With K = 8, the drive on a new filler is a fraction of the trained drive, and filler units from other frames outvote it.
- This is the developmental type-frequency law (Bybee 1995), and it explains why random MIX at birth helps only if it raises S/(F+S).

**(e) The life memory test cannot measure memory.** `life.py` `_topic_memq` always asks as a probe, and no other topic ever answers a fact question. So "我叫什么 → N" is never heard as a question–answer routine in a whole life. Meanwhile 我叫N was heard 9+ times before 62 of the 74 `fact_person` probes, and the score is 0.00. In the staged bench, where the routine is taught, name recall is 0.76.

**(f) A constant blocks novel names.** `core.py` sets `match_n = 6` and `conf_min = 6`, so a word is not recognised before 6 pointed namings. In life, 88 of the 95 `novel` probes come after ≤ 4 encounters. So `novel` = 0.00 is guaranteed, and IDEAS #11's "may emerge later" cannot happen below 6 exposures.

**(g) Favourites are guaranteed by the rule.** R11 is co-occurrence with row normalisation, plus spreading. Spreading along such links concentrates activity on the items with the most co-occurrence mass, that is, the most frequent syllables. So "top-20 frequent share 0.81" follows from the rule. It is the same for any two agents with the same input, and it does not reach speech (ARCH_v2: behaviour within noise). It is not yet the designer's idiosyncratic, behaviour-level "foodie".

### 3.2 Hidden teachers and shortcuts

1. **Babbling (inherited, already acknowledged).** `agent.py` `babble()` calls `voice.decode(m)`, deduplicates by the world's char label, drops garble by that label, and synthesises the sound with `own_pattern(voice, …)`. It is also used in life (`life_synth.py:53`) and in world2 (`world2_smoke.py:45`). This is the only place the agent reads world state it cannot perceive.
2. **The worlds pre-segment perfectly.**
   - No pauses inside a line, and silence between lines.
   - Syllables are exactly 3 ticks, with a clock reset at speech onset.
   - The scene appears 5 ticks before the first line, with 5 blank ticks before it; this opens the "scene thought".
   - Pointing is held in the attended block from the prompt's first tick until the exchange ends.

   Innate mechanisms that rely on these are only valid while the world keeps providing them. world2 does not: voices mix.
3. **The world decides what is appropriate.** Life's "appropriate initiative" is the world's string match, and speech while alone counts as inappropriate. Crib monologues are normal and serve as practice (Weir 1962; Nelson 1989, CITED).
4. **Probes are partners without feedback** (as the synthesizer noted). So drift and probe numbers mix memory with turn structure.

### 3.3 What will not survive an open world, many bodies, real images or an adult corpus

- **Many bodies.** Learning by listening ignores both role and source.
  - In the world2 smoke run (`out/world2_smoke.json`), 1,023 agent-to-agent turns occurred against 1,442 initiatives, of which 20 were appropriate (1.4 %).
  - The older smoke log `w2smoke.log` shows a2a rising 205 → 1,009 → 1,368 per 15k ticks.
  - Peers' echoes of partner lines become most of what an agent hears, and each is learned as "the next line". This is a closed loop of echoes. Prediction: it worsens with the number of agents (§4.4).
- **Pronouns and facts about people.** Without speaker identity, 我/你/他 cannot be resolved. world2 asks 我是谁 and 你叫什么 among three caregivers, and pronouns are the most frequent words in the adult corpus.
- **Adult corpus.** 99.7 % of pairs are unique, so K ≈ 1 per frame. By §3.1(d), a context-bound readout then generalises ≈ 0.
- **Real audio.** Vigilance categories without normalisation already break in world2.
- **Real images.** The slot-relative frame relies on identical codes in every block. A foveated eye breaks that; world2's fovea mode exists but is untested with ARCH_v2.

### 3.4 Numbers that do not mean what they claim

| claim | what it means | evidence |
|---|---|---|
| Life free replies 0.34 (last tenth 0.47) | Below a trivial responder that echoes (greet, bye, goodnight, ritual) or gives one fixed answer per frame (要, 好吃, 是, 在左边…), which scores **0.51** on the same 2,252 items. ARCH_v2 is below it on bye (0.34 vs 1.00), greet (0.07 vs 1.00), goodnight (0.64 vs 1.00), offer (0.54 vs 1.00), taste (0.49 vs 1.00), ritual (0.55 vs 0.86) and where (0.20 vs 0.33). It is clearly above only on slot (0.62 vs 0.08) and what (0.37 vs 0.08). | PROVED from the per-task n and acc in `life/final_noctx_s5.json` and the acceptable sets in the bench README |
| Life "scene questions in new situations" 0.33 | The constant-per-frame responder gets ≈ 0.29 on the scene tasks. | PROVED (same source) |
| Staged H ≈ 0.2 | Pooled final held-out probes, seeds 0–4. Only colour (0.60) and count (0.37) generalise. Taste 0.00 (n 30), where 0.04 (a constant gets 0.33), check 0.24 (a constant gets 0.5), what 0.09, slot 0.05, offer 0.17 (where "要" alone is acceptable). | PROVED from `out/runs/final_noctx_s*.json` |
| Life memory 0.00 | An untaught question routine, not a measure of memory. | §3.1(e) |
| Favourites 0.81 | Follows from the form of R11; not personal, not in behaviour. | §3.1(g) |
| "No drift in life" | Life gives recasts for every taught exchange, all life long. | `life.py` |
| 347 greetings made first | The agent anticipates the partner's 你好. This is correct only because the greeting routine is symmetric; the same mechanism produces 这是什么 and 我叫小强. | life categories: nonsense 703, alone 822, greeting 347, naming 18 |

## 4. Q3: What to change next (ranked)

### 4.1 Role-aware learning by listening, plus speaker identity

**Mechanism.** All of it is local: one trace per synapse and one broadcast signal.
1. The R2 updates made while a heard line is learned in the context of the held thought are kept on an eligibility trace until the line's role is known (within W = 12 ticks of its end). The same applies to the hippocampal NEXT link from the previous line to it.
2. Broadcast signal: "the line got an answer", meaning the agent's own mouth starts within W (efference copy, already used for the own-voice tag). Later this should also include another voice that starts within W.
3. If the line was answered, it was a first pair part (an initiation): its trace as an answer is cancelled. In the full form, the trace is re-applied to an "initiate" role key, so that scene thoughts can still learn to greet and to ask. If it was not answered, it is a response or feedback, and its learning stands; recasts are learned as now.
4. Add a speaker-identity layer: voice categories over the voice-offset part of the ear code, so that roles bind to persons. This is needed for 我/你 and for §4.4.

This is the designer's own pointer in IDEAS #10 ("it also needs a sense of who is answering whom"). It is also what role tokens do for a chat language model.

**Why it is the bottleneck.** It is the common cause of drift, initiative content, turn-learner misfires and the held-out fallback, and it gets worse with every peer added in world2.

**Evidence** (SIMULATED, D5: seed 0, nS 600, paired; `cmp_tasks.py`):

| | J | F | RET | H | drift level after 300 untutored dialogues |
|---|---|---|---|---|---|
| ARCH_v2, nS 600 (VETO 0) | 0.70 | 0.60 | −0.09 | 0.29 | 0.71 → 0.28 |
| **+ veto (VETO 1)** | **0.93** | 0.66 | −0.07 | **0.54** | **0.70 → 0.67** |
| (published ARCH_v2, nS 1000) | 0.79 | 0.63 | −0.04 | 0.21 | 0.81 → 0.52 |

- Per-task held-out, VETO 0 → VETO 1: what 0.25 → 0.75, taste 0.00 → 0.71, offer 0.36 → 0.64, check 0.27 → 0.59, where 0.00 → 0.18, slot 0.18 → 0.18 (n 7–24 each).
- Costs:
  - colour 0.89 → 0.68 (n 28).
  - ritual 0.34 → 0.14 (n 29). Even without the veto, ritual at nS 600 is far below the 0.90 reached at nS 1000, so it is partly under-training.
- Two variants that did not help:
  - Exempting the agent's own repetitions (VETO 2) brought back no gain at stage 1 (B 0.69, held 0.19). The agent's own echoes keep the confusion alive. I stopped it after stage 1.
  - A feedback exemption, "a line containing my whole last utterance is feedback" (VETO 3), was bit-identical to VETO 1. So the ritual cost has another cause, not isolated.
- Mechanics: 0 violations in every run.
- D4, on the already-trained agent: after drift, ok 0.77 instead of 0.51; partner-line replies 0.03 instead of 0.11; silent 0.21 instead of 0.38.

**Falsifiable prediction.**
- On seeds 0–4, J improves by at least the seed SD (0.06) and the drift loss is ≤ 0.05, versus 0.14–0.33 published.
- In life, partner-line "nonsense" initiatives fall by ≥ 50 %, while greeting first stays at about 350 (full form only).
- In world2, a2a echoes no longer rise over days.
- The prediction fails if J does not improve on seeds 1–4.

**Cheap test.**
- `veto_j.py` on seeds 1–4 (11 min each) at nS 1000. That needs about 17 min per run, so split the stages or use nS 800.
- The ritual cost needs one D1-style dump of the trained VETO 1 agent (`save=1`).
- Then the full form (the initiate role key) in one life (seed 5, about 9 min).

### 4.2 Comprehension feeds production

**Mechanism.** Every comprehension map gets its inverse, offered as a copy candidate through a basal-ganglia copy gate keyed only on frame context (prompt items without the filler, position, last item, length):
- the words VR recognises at the attended slot, from PRIME (appearance → word);
- the location word of the attended slot (inverse of attD: slot → word);
- the colour and count words recognised there.

The gate learns by listening, using rule R2: its target is 1 when the heard item is one of the candidates. This is the generic "say what I attend to" route (IDEAS #6, open problem 3). The bench's existing prompt-copy gate shows how such a gate carries a frame over to any filler (ritual and offer).

Add frame-only units, through the designer's random-MIX-at-birth plan or distributional word classes learned by a competitive Hebbian layer over item contexts (frequent frames: Mintz 2003; Redington, Chater & Finch 1998), which raises S in §3.1(d). Test it on top of 4.1; otherwise the role confound hides the effect.

**Why it is the bottleneck.** The knowledge is already present but cannot be said:
- D2: attention to X is 1.00 on held-out "where", yet the reply is 0.00;
- D3: VR holds the name 0.73 on held-out "what", yet the reply is 0.02;
- even with the veto, held-out where and slot are 0.18.

This is the comprehension–production gap of the context-bound stage.

**Falsifiable prediction.**
- On top of 4.1, held-out where and slot rise from 0.18 to ≥ 0.6, and what to ≥ 0.7 (bounded by VR's 0.73).
- Life `novel` > 0 at ≥ 6 encounters.
- Namings among initiatives in life rise from 18.
- Seen items stay within ±0.03.
- The prediction fails if held-out stays ≤ 0.3 while D2/D3 show correct attention and recognition.

**Cheap test.** run_j on seeds 0–1, with D1–D3 (25 s each) as unit tests on the saved agent.

### 4.3 Contingent, child-led caregiving and curiosity-driven asking

**Mechanism, in the world (life; world2 already has part of this):**
- the partner answers the child's 这是什么 by naming a visible object, as world2's `_r_answer` does;
- the agent gets 3 look units (slot gaze) in life, so the partner can name what it looks at (follow-in naming);
- routines are prompted ("跟我说你好"), or the partner models the social reply 30 % of the time;
- the question–answer routine is modelled for half of the facts and only probed for the other half;
- recasts are sometimes expansions.

**Mechanism, in the agent.** An innate curiosity term: the urgency of a scene thought rises when the attended slot holds an object for which VR finds no word (an information gap). The answer then arrives as a recast under joint attention, which the agent can learn by listening.

**Why it is the bottleneck.** In an open world nobody schedules lessons. Children get a large share of their labelled examples by asking and attending (Chouinard 2007; Yu & Smith 2012).

**Falsifiable prediction (world change only, agent unchanged).**
- Life memory facts whose routine is modelled reach > 0.3, while unmodelled facts stay ≈ 0. This separates memory from routine, §3.1(e).
- greet and bye ≥ 0.7.
- With curiosity added: "asking" ≥ 30 % of initiatives with the partner present, and `novel` > 0.

**Cheap test.** A `Life` subclass written in the next designer's folder, plus one life (seed 5, 525 s), compared with `life/final_noctx_s5.json`.

### 4.4 Selective social learning before many bodies

**Mechanism.** The listening third factor M_listen becomes graded: g = trust(speaker).
- Trust is learned by an R8 learner. Its target: the speaker's line was later confirmed (it matched a caregiver's recast or model in the same context, or it was rewarded).
- It starts from an innate prior: adult voices high, peers low (Jaswal & Neely 2006). It needs the speaker identity from 4.1.

**Why it is the bottleneck.** The GPU plan's next step is many bodies, and the smoke run already shows peers dominating what an agent hears.

**Falsifiable prediction.** Without trust, scene accuracy and the share of partner-line intrusions get worse from 4 to 16 MouthOnly agents, in proportion to the peer share of heard lines. With trust they stay at the 4-agent level.

**Cheap test.** `world2_smoke.py` with n = 4 and n = 16 for 1.5 days (≈ 156 s and ≈ 10 min). Log the source of every heard line.

### 4.5 Online, lawful, socially shaped babbling

**Mechanism.**
- Babble through the body in the first hours of life, hearing the own voice only through the ear (bench `body.babble`), with no `world.voice` call.
- Use `physics='lawful'`.
- Imitation-guided search: for a heard syllable with no program, try one-muscle swaps of the best-matching babbled set and keep the improvement (DIVA-like, local).
- The partner imitates babbles: stage A play already does this; life should too (Goldstein & Schwade 2008).

**Why it is the bottleneck.** This is the one real privileged read (§3.2), and it is required for real bodies and audio. It is not a bottleneck for bench numbers.

**Falsifiable prediction (ASSUMED).** ≥ 90 % of the 67 lexicon syllables are bound in fewer babbles than blind enumeration needs (≈ 36k sets), and J is within ±0.03 once binding is complete.

**Cheap test.** A module test on the bench physics only (no agent), in minutes.

## 5. Q4: Are the designer's ideas implemented faithfully?

| idea / principle | status | where it is violated, stretched or under-used |
|---|---|---|
| #1 PFC = next-input predictor | Faithful at the core. | Stretched: used for production without roles. A language model needs role tokens to answer instead of continuing the user's turn (§4.1). |
| #2 hold a thought by a loop; #9 continuous thinking | Implemented. | Under-used. At bench timescales it is exactly equivalent to the old register (PROVED by the synthesizer), so no task tests continuity. Own speech never enters the thought: the agent does not "remember what it said". |
| #3 a words-and-meaning layer between ear and PFC | Not built. | Items are syllables; there is no word unit and no segmentation. |
| #5 middle layers learn so that things used alike get similar codes, plus the 2026-10-31 addendum | Not built. | This is the frame-only capacity whose absence shows in §3.1(d). Compare the random-MIX plan **on top of 4.1**; otherwise role pollution confounds the comparison (held-out what 0.25 → 0.75 came from the veto alone). |
| #6 speech caused by thinking | Mechanically faithful. | Under-used: "say what I attend to" is missing (4.2), and initiatives have no goal (4.3). |
| #7 favourite thoughts | Implemented as a frequency attractor. | Stretched: not personal, not in behaviour. Modulate R11 by value or drive (hunger in world2), and measure favourites in self-initiated speech. Children's intense interests are idiosyncratic and expressed in talk (DeLoache, Simcock & Macari 2007). |
| #8 one unbroken life | Faithful in life and world2. | J's probes and drift test use frozen or copied agents. That is fine as measurement. |
| #10 no sleep | Faithful; awake recall works. | The idea's "good parameters make drift rare" is contradicted by D4/D5: drift is a role problem, not a parameter problem. The idea's own pointer, "who is answering whom", is unbuilt. |
| #11 one-shot naming not urgent | Respected (measured only). | `match_n = 6` forbids it from ever emerging below 6 exposures. |
| #13 world2 | Smoke run only, mouth only. | Under-used: asking answered, requests fulfilled, hunger and praise, i.e. world2's most child-like features, cannot reach a MouthOnly agent. |
| #14 fault tolerance | Holds for sparse codes. | Contradicted at the category stage: vigilance templates are not tolerant to level changes (world2 cap). |
| #17 innate goodwill | Planned. | No affect signal exists in any world: caregivers are never visibly pleased or upset beyond scalar reward. The planned test ("don't take food from someone's hands") needs a distress cue. Helping appears spontaneously around 14–18 months (Warneken & Tomasello 2006). |
| §9 P4 / "no muscle puppeting" | Faithful. | Babbling reads `world.voice` (§3.2). |
| §9 P7 every learned part reaches behaviour | Mostly faithful. | The R11 links and V_go are passengers but stay on. |
| §9 P2 / P3 / P6 / P8 / P9 | Faithful. | The 4.1 rule keeps P6: one trace per synapse and one broadcast signal. |

## summary_zh

1. ARCH_v2 已经有"一岁孩子"的配置：会听音节、会咿呀学语、轮流说话从不抢话、会顺着指点看、想法能保持，练过的问题几乎全对（见过的题 0.95–1.00）。
2. **最关键的问题是"说出我预测下一个人会说的话"不分角色。** 回答之后，对方接着问的下一个问题会被当成"答案"学进去；对方没确认、直接往下说，也被当成"答错了"。孩子学的是"回答的人说什么"（角色互换模仿）；没有角色标记的聊天语言模型也会像 ARCH_v2 一样，接着用户往下问。无人指导时的漂移、主动说话多半是对方的台词、轮次学习失灵，以及大部分"没见过的组合答不出"，都是这一个原因。
3. **我测了一个最小的、局部的修正：** 我听到一句话、接着自己回答了它，说明这句话是问题，就撤销把它当作答案的学习。同种子对照（种子 0，每阶段 600 段对话）：J 0.70 → 0.93，没见过组合 0.29 → 0.54，无人指导 300 段后水平 0.28 → 0.67（下降量 0.43 → 0.03），机制 0 错误。代价是颜色题和礼貌用语下降，礼貌用语的原因还没查清。
4. **有些数字的意思和看上去不一样：**
   - "一生"自由回答 0.34，比"重复对方或每种问法固定答一句"的傻办法（0.51）还低；
   - "没见过组合约 0.2"背后，"好不好吃"答"好吃"这种答案固定的题是 0.00；
   - 一生里记忆题 0.00，是因为世界从来没示范过"我叫什么→名字"的问答，不是记忆差；
   - "爱想的话题 0.81"是规则本身决定的；
   - "一生里不漂移"是因为一生里一直有人纠正。
5. **世界像家教，不像父母：** 内容不随孩子变，不跟着孩子看的东西说，一生模式里孩子问"这是什么"没人理；world2 里智能体主要在彼此对话。
6. **建议按顺序做：**
   - 分角色的"边听边学"，加上能认出说话人；
   - 让"听懂的"能说出来（按问法复制看到的东西的名字和位置）；
   - 世界要跟着孩子回应，孩子要有好奇心、会主动问；
   - 多个体时按说话人的可信度学习；
   - 咿呀学语改为在身体上在线完成、发音规律化，不再读世界的发音表。
7. **设计者的想法：**
   - "PFC = 预测下一个输入"是对的，但用于说话时需要角色；
   - "中间层要学出相似编码"（第 5 条及新补充的随机连接方案）是真正缺的，要在角色修正之后再比较；
   - "爱想的话题"还没有个人特点，也没有体现在说话里；
   - 一次学会新名字被一个常数（至少 6 次指点）挡住了。