# Review round 2 of ARCH_v2: synthesis (coordinator)

Five reviewers answered the same four macro questions independently, each through one lens: neuroscience, learning
theory, development, embodied / open world, and skeptic. Their reports are in `neuro/`, `learning/`, `develop/`,
`embodied/` and `skeptic/`, each with its probe scripts and outputs.

Labels follow the reports: SIMULATED = a reviewer's own run, PROVED = argument or arithmetic from files.

## 1. What everyone agrees on

ARCH_v2 has the kit of a one-year-old:
- syllable categories;
- an inverse map from babbling;
- turn-taking with 0 overlaps;
- joint attention;
- a one-shot episodic index with awake recall.

On practised items it is near perfect: seen items score 0.95–1.00 on frozen probes.

Its headline numbers measure less than the text claimed, and one logical flaw is the common cause of most failures.

## 2. The central flaw: learning does not know whose turn it is (all five reviewers)

**The mechanism (PROVED from the code).**
- "Speaking = saying what I predict the next other voice will say."
- The agent's own speech enters neither the thought (`ct_own` = 0) nor the hippocampal sequence (NEXT skips own lines).
- So the first line another voice says after the agent's reply is learned as the answer to the question, by two
  routes:
  - the cortex: GEN, in the question's context;
  - the hippocampus: the NEXT link, which is then replayed and copied.
- With a recast tutor that line is the correct answer, so in effect the world supplies the label.
- Without a tutor it is the partner's next question. "Moving on" is also read as "predicted but absent → weaken".

**This single cause explains:**
- free-use drift: C 0.79 → 0.51 after 300 untutored dialogues and → 0.31 after 900 (skeptic, SIMULATED);
- "initiatives = the partner's lines";
- the misfires of the turn learners;
- the echo-like fallback;
- **most of the held-out failure.**

**Generalisation exists but is destroyed by learning (learning reviewer, SIMULATED).**
- In a frame × filler probe, ARCH_v2's own core answers never-taught pairs at 1.00 (3/3 seeds) when they are only
  tested on a frozen copy.
- The same pairs asked live, without feedback, as the bench and life do, score 0.00.
- So the bench's H ≈ 0.2 mostly measures drift. The training worlds use `held_frac` 0.15 with learning on.

**Two independent minimal fixes, both local, both SIMULATED.**

| fix | test | result |
|---|---|---|
| **Xcut** (learning): after an own utterance the own reply enters the thought, and the next heard line is not linked as the prompt's NEXT; answers are learned from other speakers' answers | symbolic probe, tutor with 30 % model replies | held-out 0.00 → 0.72 frozen / 0.80 live, seen 1.00 |
| **Answered-line veto** (develop): a heard line that the agent then answers was a question, so its learning as an answer is cancelled (eligibility trace + efference broadcast) | full staged protocol, seed 0, nS 600, paired | see below |

Veto, ARCH_v2 → ARCH_v2 + veto:
- J 0.70 → **0.93**;
- H 0.29 → 0.54;
- drift loss 0.43 → 0.03.
- Held-out by task:
  - what 0.25 → 0.75;
  - taste 0.00 → 0.71;
  - offer 0.36 → 0.64;
  - check 0.27 → 0.59.
- Costs: colour 0.89 → 0.68 and ritual 0.34 → 0.14 (the ritual cost is partly under-training).
- Mechanics: 0 errors.
- Only one seed so far.

Own speech in the thought alone (`ct_own` 0.6) is not enough: drift loss −0.23 vs −0.28 (skeptic).

## 3. Numbers that do not mean what they claimed

| claim | correction | source |
|---|---|---|
| Life free replies 0.34 | A trivial responder (echo the rituals, one fixed answer per frame) scores 0.51 on the same items. Scene questions: 0.43 vs a constant per task type 0.40. Social: 0.32 vs echo 1.00. | develop, skeptic (PROVED) |
| H ≈ 0.2 | Hides 0.00 on taste (where 好吃 is always right) and check 0.24 (below the 0.5 of a constant). Mostly drift (§2). | all |
| Colour / count +0.08 J from retinotopy | Mostly the world's factorised, slot-identical eye code. | skeptic, learning |
| Favourite topics 0.81 vs 0.14 | The fair token baseline is 0.62–0.63. The favourites are function syllables (是 这 什 么 …), follow from the form of R11, and never reach speech. | neuro, skeptic, develop, embodied |
| 1,891 initiatives, 347 greetings first | 81 % alone or nonsense. 626 of 1,337 你好 went to an empty room, with no extinction. "Greeting" = anticipating the partner's 你好. | neuro, embodied |
| Speech intention = confidence | Latency is 6–7 ticks under every lesion; correct 6.01 vs wrong 6.11. A timer in effect. | skeptic |
| Continuous thought | Register-equivalent. Recall saves and restores the thought. Context length cannot be tested on one-line tasks. | all |
| Life memory 0.00 | Life never models the "我叫什么 → N" routine, so this is not a memory measure. `match_n` = 6 also blocks novel names below 6 exposures. | develop |
| 0 overlaps | Guaranteed by reflexes fitted to the bench's timing. A valid regression check, not learned politeness. | skeptic, neuro |

## 4. The ranked plan

Each step comes with the reviewer predictions it must meet.

**Step 0. Honest measurement** (learning, skeptic, develop). Do this before anything else.
- Probe held-out pairs only on frozen copies (`held_frac` 0 in the training worlds), and keep the old protocol for
  comparison.
- Report per-task held-out.
- Show same-run trivial references: echo, a constant per task type, and the random namer in life.
- Report a role-confusion rate.
- Give H more weight in J.
- Keep `gen_probe*.py` and the develop D1–D3 probes as regression tests.

**Step 1. Role-aware learning, self in the loop** (all five).
- Unify Xcut and the veto:
  - own utterances enter the thought (role-tagged) and the hippocampus;
  - a heard line's learning as an answer is held on an eligibility trace until its role is known (answered by me →
    it was a question; contingent on my reply → feedback or recast);
  - the NEXT links follow the real dialogue.
- Speaker identity: voice categories.
- Predictions:
  - J gains at least the seed SD on seeds 0–4;
  - drift loss ≤ 0.05–0.10;
  - frozen H ≥ 0.45;
  - in world2, praise words are not learned as answers.

**Step 2. Comprehension feeds production** (neuro, develop, skeptic).
- A "say what I recognise / attend to" copy route: VR's word at the attended slot, the slot's location word, colour and
  count words.
- Gates keyed on frame context only (`gate_noid`).
- Self-initiated thoughts read the held thought.
- Evidence of the gap: the agent attends to X correctly in 1.00 of held-out "where" items and recognises its name in
  0.73 of held-out "what" items, yet says them correctly 0.00 / 0.02 of the time.
- Prediction: held-out where/slot ≥ 0.6, what ≥ 0.7.

**Step 3. A signed value loop** (neuro, embodied, learning).
- Dips when an expected reward is missing (δ = −V on r = 0).
- A NoGo veto in place of V_go's gain.
- Value in its own slow table, not inside GEN.
- Actor eligibility that expires.
- Innate coarse social detectors: person present / arrived / left / addressed to me; later pleased / upset.
- Prediction: 你好 to an empty room drops from about 65 to ≤ 15 per tenth of life; appropriate initiatives rise from
  0.19 to ≥ 0.4.
- This is the base for IDEAS #17.

**Step 4. A learned meaning layer, plus random connections from birth** (neuro, skeptic, learning, develop; the
designer's 2026-10-31 idea).
- Sparse item codes learned from context, feeding a random-at-birth conjunction layer that is trained on top and pruned
  (the `randmix` agent is testing the layer now).
- Compare on top of Step 1; otherwise the role confound hides the effect.
- Prediction: class-dependent held-out ≥ 0.8; scaling helps only with this layer.

**Step 5. Slow consolidation instead of a 6k-line rehearsal buffer** (learning, skeptic, neuro).
- A slow cortical readout trained mostly by awake replay.
- Recall firms the synapses it used (metaplasticity), which is IDEAS #10's "recall often → firm".
- Cloze-style retrieval practice.
- Prediction: after 3,000 untutored dialogues ≥ 0.5 (ARCH_v2 now ≤ 0.1, ASSUMED).

**Step 6. Open world readiness** (embodied, neuro, develop, skeptic).
- Hearing:
  - an adaptive noise floor;
  - divisive gain control;
  - voice-change boundaries;
  - trust-weighted listening (adult prior, learned reliability).
  Note that at world2 noise σ = 1 the agent hears 0 lines today.
- An action loop:
  - motor babbling in the body;
  - efference remapping;
  - a protected goal slot;
  - drives (hunger) as context;
  - curiosity.
- Remove the world's gifts one by one:
  - online babbling, without `world.voice`;
  - a non-factorised or foveated eye.
- World changes:
  - caregivers follow the child's attention and answer its questions;
  - life models the fact routines;
  - affect cues;
  - taking from someone's hands;
  - negative outcomes.

**GPU port:** keep it to the validated one-agent match plus speed. Bridge to world2 after Steps 1–3. The core's key
schema must stay data-driven.

## 5. Designer's ideas: status after round 2

- **Faithful:** #8 one life, #10 no sleep (as a mechanism), #11, the 0-error mechanics.
- **Stretched:**
  - #1 has the objective and the output rule, but no learned state, embedding or prediction-as-input;
  - #2 and #9 are register-equivalent;
  - #6 decides timing only;
  - #7 holds favourites only as frequency hubs;
  - #10 has nothing that "firms".
- **Not built:** #3 meaning layer, #5 middle layers learn similarity, #14 fault tolerance after Z, #15 context length,
  #16 real images, #17 values.
- **Corrected:** our first reading of the #5 addendum. Frame-only units do exist in ARCH_v2; the collapse comes from
  the learning target.
