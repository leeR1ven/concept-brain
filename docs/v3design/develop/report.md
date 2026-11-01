# v3 design round: the development member's report

**Angle.** Abilities should come from innate drives plus experience, as in a child, not from added modules.

**Scratch folder:** `docs/v3design/develop/`

| file | what it does |
|---|---|
| `count_input.py` / `.out` | counts how much speech the agent hears in the staged J protocol and in one life |
| `cue_count.py` / `.out` | checks whether the life world gives any perceivable cue that separates a question from a correction |
| `role_toy.py` / `.out` | symbolic toy: which listening rule learns answers from which caregiver behaviour, and which one drifts |

How it was run:
- Each script ran locally, one python process at a time, under 100 MB and under 2 minutes.
- I checked `free -m` first.
- No training runs and no server jobs.
- I only read the v2.2 folder (interim results). I did not touch it.

**Labels:**
- **PROVED**: counted from code or a log, or arithmetic.
- **SIMULATED**: my own run; the file is named.
- **CITED**: a paper (author, year) or another project report.
- **ASSUMED**: argued but not tested.

---

## 0. Bottom line

1. **A child's abilities come from a small fixed kit plus years of structured experience.**
   - My proposed kit has 12 items (§1.1):
     - 2 neuromodulator-like broadcasts;
     - 4 innate reward sources;
     - 2 orienting priors;
     - 4 body reflexes.
   - Today's lineage carries about 50 hand-set, ability-specific parts and keys (PROVED by listing):
     - 22 innate items in ARCH_v2 §5;
     - about 11 v2.1 keys (role windows, the say-what-I-recognise routes, the veto, dips, social features);
     - 7 hearing keys;
     - 2 meaning keys;
     - 8 v2.2 keys (the veto variants and the dip scope).
   - Most of these stand in for a missing drive or for missing structure in the world.

2. **The correction problem of v2.1 and v2.2 is mostly a world problem.**
   - **What the world gives today** (PROVED, `cue_count.out`, one life, 10⁶ ticks):
     - 87 % of life's questions contain a question syllable (什么 吗 哪 几 谁 呢). 0 % of corrections do.
     - No correction ever contains 不是.
     - No world has prosody, faces that show talking or gaze, or affect. Social reward arrives as a world-computed correctness number.
     - Life never shows an answer being modelled: 0 exchanges in which the caregiver answers her own question, and 39 overheard question–answer pairs per 10⁶ ticks.
     - The staged J protocol uses the recast tutor, so it also has 0 such exchanges.
   - **How children get answers instead** (CITED):
     - mainly from *formats*, where the adult asks and then answers herself (Ninio & Bruner 1978);
     - from overheard question–answer pairs;
     - from reformulations of their errors (Chouinard & Clark 2003).
     - Approval follows truth, not grammar (Brown & Hanlon 1970), and it comes with recognisable prosody (Fernald 1993).
   - **The toy result** (SIMULATED, `role_toy.out`, 5 seeds). I compared three listening rules:
     - an ARCH_v2-like rule, which learns the first other-voice line after the question as its answer;
     - a v2.1-like rule (role-aware listening, RAL), the same but with v2.1's cancel/commit rule (see §4.1);
     - a "generic adjacency" learner with **no role rule at all**: it learns each heard line as the prediction of the line just before it, whoever said it; its own lines are predicted from the efference copy (the copy of its own motor commands) and teach nothing.

     Results:
     - The generic learner scores 1.00 and still 1.00 after 3,000 untutored exchanges, whenever the world gives 30 % formats or replays corrections as question–answer pairs.
     - In today's recast-only world it learns **nothing (0.00)**.
     - The ARCH_v2-like rule drifts from 1.00 to 0.00. The v2.1-like rule works, but only because of its timing rule.
   - The lineage's worlds never modelled the answer slot, so only a role rule could learn there. That is why role rules kept piling up.

3. **Amount of experience** (PROVED by count, CITED rates):

   | what | heard syllables | roughly equals |
   |---|---|---|
   | staged J protocol | 40,983 (10,827 lines) | 2–3 days of a child's total input |
   | one 10⁶-tick life | 15,457 | about 1 day |
   | one world2 day | about 960 | about 1/20 of a child's day |
   | one child-year | about 6.6 M | about 160 staged protocols, 430 lives or 6,900 world2 days |

   - Repetition per item is not the shortage. Each object name is heard about 451 times in the staged protocol, which is roughly 2–6 weeks of a frequent noun for a child.
   - What is missing is duration, variety, cues and contingency on the child.

4. **What development can replace, and what it cannot.**
   - **Replaceable:**
     - v2.1's role windows;
     - v2.2's six veto keys;
     - turn reflexes 3, 5, 7, 7b and the transition window;
     - the three hand-set social-feature windows;
     - babbling's privileged read of `world.voice`;
     - four of the seven hearing keys;
     - the pointing-only teaching of the word→object maps;
     - the world's correctness reward;
     - later, the fixed order of thought operations.
   - **Not replaceable by experience:** the generic core.
     - the order of lines in the thought;
     - one pool of word candidates feeding speech from any area;
     - conjunctions;
     - a one-shot hippocampal index with awake recall;
     - area-general normalisation and adaptation;
     - the body.
   - Without these, ideal input still fails. Review 1 showed this: on clean child-directed input, v1 scored 0.00 on never-heard frames.

5. **The world changes that matter most** (§3, world code only):
   - formats whose self-answers fade as the child succeeds;
   - corrections in natural forms: the replayed pair, 不是X，是Y, or the plain answer;
   - social reward that is *perceived* (approval prosody, a smile), imperfect and based on truth, instead of a reward number;
   - prosody (the infant-directed register; question, approval and prohibition contours) and face units (talking, gaze at me, pleased, upset);
   - responses to babble, to the child's questions and to its requests;
   - clarification requests and repair games;
   - graded noise and many speakers;
   - local3d images with continuity across viewpoints.

6. **The decisive test.**
   - Freeze the agent code (by hash).
   - Add world levers one at a time. Each ability's curve should rise only after its lever.
   - Lesion each drive. Each should fail in the predicted way (§4.2).

---

## 1. Innate drives and reflexes

### 1.1 The proposed innate budget (fixed; never added to per task)

**Broadcasts (neuromodulator-like):**

| # | item | generic definition (no task, word or modality) | why the brain has it (CITED) | existing code | cost |
|---|---|---|---|---|---|
| B1 | value / reward-prediction error (dopamine-like) | δ = r − V(context), with r the weighted sum of S1–S4. It signs and scales plastic updates and drives basal-ganglia Go/NoGo. | Schultz, Dayan & Montague 1997; Doya 2002 | critic/actor R6, dips, `val_veto` (`think/agent_think.py` ThinkAgent; `v21/agent_v21.py` AgentV21) | exists |
| B2 | precision / surprise (acetylcholine- and noradrenaline-like) | Every area keeps a running mean and spread of its own prediction error. Their ratio sets three things: how much the top-down prediction counts against the input, the learning rate, and when a new unit is recruited. | Yu & Dayan 2005; Doya 2002; Dean, Harper & McAlpine 2005 | pieces only: `hearing/hear.py` SoundGate (μ, ν, ear only), R5 vigilance, comparator confidence | 2 numbers per area plus 1 broadcast; this is the second broadcast the brief asks to justify |

**Innate reward sources (they feed B1):**

| # | item | generic definition | why the brain has it (CITED) | existing code | cost |
|---|---|---|---|---|---|
| S1 | body need | r rises when an internal deficit (hunger) is reduced; the need is also part of the context | hypothalamic hunger drive | world2 `hunger` / `taste` observations; the agent has none (MouthOnly) | 1 input |
| S2 | social contingency | r = +1 when a person-like input (O1) starts within a short window after my own action (efference copy) | Watson 1972; Gergely & Watson 1999; Tronick et al. 1978; Goldstein, King & West 2003; Takahashi, Liao & Ghazanfar 2017 (marmosets: contingent parental feedback speeds vocal development) | V_go's target (SparsePR); today it reads the world's reward number | none new |
| S3 | the partner's affect | coarse, fixed detectors of approval vs disapproval or distress in the voice and face of any person, including third parties; this is IDEAS #17's prosocial core | Fernald 1993 (5-month-olds react to approval and prohibition even in unfamiliar languages); Sagi & Hoffman 1976; Hamlin, Wynn & Bloom 2007 | none; the world's praise reward stands in | 1 detector; needs cues in the world |
| S4 | learning progress (curiosity) | r = the drop in B2's error in the active context, measured as a slow running error minus a fast one, kept per context unit | Oudeyer, Kaplan & Hafner 2007; Oudeyer & Kaplan 2007; Gottlieb et al. 2013; Gruber, Gelman & Ranganath 2014; Bromberg-Martin & Hikosaka 2009; Kidd, Piantadosi & Aslin 2012 | none | 2 traces per context unit |

**Orienting priors:**

| # | item | generic definition | why the brain has it (CITED) | existing code | cost |
|---|---|---|---|---|---|
| O1 | person detector | fixed coarse features: a face- or person-like pattern, a voice-like sound; a person present, arriving or leaving | Johnson et al. 1991; Morton & Johnson 1991; Johnson 2005 | replaces v2.1 `soc_feats` (three hand-set time windows) | − 3 windows |
| O2 | ostensive salience | the infant-directed register, direct gaze and pointing raise B2's attention gain on that stream or place | Csibra & Gergely 2009; Farroni et al. 2002; ManyBabies Consortium 2020 (preference d = 0.35); Triesch et al. 2006 (gaze following can be learned on top) | the hard wire "pointing = attended slot" (SynthCore, `eye_layout='retino'`); world2 pointing units | becomes a prior weight |

**Body reflexes (these exist today and stay):**

| # | item | generic definition | why the brain has it (CITED) | existing code | cost |
|---|---|---|---|---|---|
| R1 | vocal rhythm | the 3-tick syllable beat and breath groups | Jürgens 2002 (brainstem vocal pattern generators); MacNeilage 1998 | bench `body.py` Body / `think/body_think.py` ThinkBody | kept |
| R2 | listening brake | another voice's onset raises the threshold of my own vocal onset; a learned modulation sits on top | Takahashi, Narayanan & Ghazanfar 2013 (turn-taking in marmosets) | body F/F2, with P_oth learned on top | kept; replaces reflexes 3, 5, 7, 7b |
| R3 | efference copy | every motor command is copied to the senses that sense its result, so my own outcomes are predicted and cancelled | Blakemore, Wolpert & Frith 1998; Eliades & Wang 2008 | the own-voice tag (ARCH_v2 §5 #4) and M_listen ("own speech never teaches") | generalised to all motor output |
| R4 | spontaneous action | a low base rate of babbling or moving when idle, raised by S4 | Moulin-Frier, Nguyen & Oudeyer 2014 | bench `body.babble`; world2 `agents.Explore` | kept |

**Left out on purpose:**
- no question detector, no correction rule, no turn window;
- no naming route and no greeting rule;
- no hearing key per problem;
- no meaning-specific cue.

S2 and S3 mention no task or word: they are defined only over person-like input, affect cues and my own efference copy. The four reward weights inside B1 are innate constants and count towards the budget.

**Note.** R3 makes the "own speech never teaches" rule (M_listen) automatic. A prediction that already includes the efference copy has almost zero error on my own voice, so my own voice teaches almost nothing. In babbling the forward model is still imperfect, so some learning from my own voice remains. That is exactly the babbling phase.

### 1.2 Ability by ability

**(a) Turn-taking**
- **Comes from:** R1, R2, S2 and B1, plus the generic time basis (time cells, ARCH_v2 §5 #9).
- **Experience needed:**
  - a partner who answers vocalisations that come in its pauses;
  - who answers nothing it could not hear, i.e. overlapped speech;
  - who waits after addressing the child.
- **Child evidence (CITED):**
  - proto-conversation at 2–3 months (Bateson 1975; Trevarthen 1979);
  - fast responders early, slower at 9 months once content grows (Hilbrink, Gattis & Levinson 2015);
  - infant response latencies follow the adult's: 1 s more adult latency goes with more than 1 s more infant latency (Nguyen, Versyp, Cox & Fusaroli 2022, meta-analysis).
- **In code:**
  - P_oth (SparsePR, `agent_think.py`) already learns "another speaker starts within 10 ticks"; V_go is S2's value.
  - These remove the 7-tick transition window (reflex 5), the "floor taken" prior (reflex 7), the 10-tick rule (reflex 7b) and the first-syllable brake (reflex 3).
  - R2 stays.
- **Falsifier:**
  - Setup: world2 with caregivers who ignore overlapped utterances; the agent has none of those reflexes.
  - Prediction: overlap per utterance falls below 2 % over the life, and the median gap converges to the caregiver's wait.
  - The claim fails if overlap does not fall, or if the agent falls silent.
- **Side benefit (ASSUMED):** in crowds the brake is on 16–52 % of the day and agents go mute (CITED, hearing report). A learned "she is addressing me" prediction (O2 gaze) could release it, as for a child addressed in a crowd.

**(b) Knowing who is speaking, and learning from corrections rather than questions**
- **Comes from:**
  - R3 (no self tag needed);
  - a generic slow state in every area. The designer's local3d result gives identity as a trajectory, s_t = 0.8 s_{t−1} + 0.2 a_t (same person 0.685 vs different 0.534; `TEMPORAL_IDENTITY.md`).
    - Applied to the voice residual it gives speaker identity. v2.1's VoiceCats already reached purity 0.91 and 0.72 from that residual.
    - Applied to faces it gives person identity.
  - co-activity, which binds a voice to the visible talking face;
  - O2 for "addressed to me";
  - S3 for the sign of feedback, and B1.
- **Experience needed:**
  - formats whose self-answers fade as the child succeeds (Ninio & Bruner 1978; Bruner 1983; Snow 1972 on finely tuned input);
  - overheard question–answer pairs (Akhtar, Jipson & Callanan 2001). Speech directed to the child matters more (Weisleder & Fernald 2013; Shneidman & Goldin-Meadow 2012).
  - reformulations that replay the pair, and contrastive 不是X，是Y (Chouinard & Clark 2003; Saxton 1997);
  - approval that tracks truth (Brown & Hanlon 1970);
  - question prosody and a waiting gaze;
  - three or more adult voices plus peers, each with a consistent face.
- **What is learned:** a line's role becomes a learned prediction of *who speaks next*. It is a SparsePR learner on line features: question syllables, contour, gaze, arrival. Its target is what speakers actually do, not what I did afterwards. Answers come from lines that sit right after a question (formats, replayed pairs, overheard pairs). Corrections sign my own answer through S3 and B1 (the R6 actor with δ < 0).
- **Replaces:**
  - RAL's timing windows (W 12, W_other 6, T 60);
  - v2.2's veto keys;
  - v2.1's rejected `ral_q` detector, reborn with a proper target and real cues.
- **Falsifier:** experiments E1 and E2 (§5).

**(c) Repairing misheard words (IDEAS #14 addendum)**
- **Comes from:**
  - the generic predictor, which already predicts the next syllable while listening at 0.82 (CITED, brainlike);
  - B2: the heard syllable is a precision-weighted mix of the bottom-up category match and the prediction. Precision is learned from experience; SoundGate's ν, generalised, is that estimate.
  - S4, which rewards asking "啊?" because the repeat brings information.
- **Experience needed:**
  - mishearing (noise, distance, crowds, a masking burst over one syllable of a predictable line);
  - peers who mispronounce;
  - caregivers who ask for clarification and repeat when asked;
  - new words that sound like known ones, introduced with a new object in view.
- **Child evidence (CITED):**
  - graded sensitivity to mispronunciation at 14 months (Swingley & Aslin 2002; White & Morgan 2008);
  - 2-year-olds treat a new word that differs from a familiar word by one sound as the familiar word, unless context points to a new referent (Swingley 2016);
  - use of sentence context grows into school age (Nittrouer & Boothroyd 1990);
  - speech in noise is immature until about 10 years (Leibold & Buss 2013);
  - "huh?" is universal (Dingemanse, Torreira & Enfield 2013);
  - phonemic restoration (Warren 1970).
- **The meaning report's objection** was that, from context alone, a misheard word looks like a new combination. Putting the scene into the prediction resolves it: no new object in view → restore; a new object in view → learn a new word. That is Swingley 2016's finding.
- **Replaces:** `h_conf` and `h_nocreate` (as precision-gated recruitment), `h_relock` (as the predicted onset) and part of `h_vbound`.
- **Falsifier:** experiment E4.

**(d) Word meaning and categories**
- **Comes from:**
  - generic co-activity (R7 outstar);
  - context-predictive codes in every area (the meaning report's codes are a generic rule, not a meaning module);
  - B2 attention gain, instead of "learn only under pointing";
  - S1, because eating grounds "food";
  - the slow state, which keeps an object's identity across views.
- **Experience needed:**
  - follow-in naming (naming what the child is already looking at or holding);
  - many exemplars and views (local3d);
  - situations where it is ambiguous which object a word names;
  - the same word from every speaker;
  - function: eating, tasting, play;
  - contrastive naming (红色的，不是蓝色的).
- **Child evidence (CITED):**
  - understanding common nouns at 6–9 months (Bergelson & Swingley 2012);
  - learning across ambiguous situations at 12–14 months (Smith & Yu 2008);
  - words promote categories at 3–4 months (Ferry, Hespos & Waxman 2010; Waxman & Markow 1995);
  - the shape bias is learned (Smith et al. 2002);
  - follow-in labelling works best (Tomasello & Farrar 1986); objects that dominate the view get named (Yu & Smith 2012);
  - how distinctive a word's contexts are predicts age of acquisition better than frequency (Roy et al. 2015).
- **Replaces:** the pointing-only gate of PRIME/attD. The meaning DG cue becomes a case of one rule: the hippocampal index reads every association area, each normalised.
- **Falsifier:**
  - Task: world2 "X能不能吃" on foods the agent has eaten but never heard in that frame.
  - Prediction: with S1 above chance; with S1 lesioned at chance.

**(e) Greeting only when someone is there**
- **Comes from:** O1, S2, and B1 with dips (a negative prediction error when an expected response does not come).
- **Experience needed:** greetings modelled on arrival; responses only when someone is present; nothing when alone.
- **Child evidence:** responsiveness to contingent partners, and distress when the partner stops responding (still face; Tronick et al. 1978).
- **Status:** v2.1 Step 3 is already nearly developmental: 5–9 empty-room 你好 per tenth of life (CITED). The hand-set part is `soc_feats`' three windows (300 / 60 / 60 ticks). Replace them with O1 (world2 eye person units 48–61, plus voice onset).
- **Risk:** without social features, v2.1's social replies fell to 0.07 on seed 6 (CITED), so O1 must exist.

**(f) Asking for things**
- **Comes from:** S1 (hunger in the context and as reward), S2, S4 (information gain leads to 这是什么 for unnamed objects) and R4.
- **Experience needed:**
  - early: food after any vocal bout made while hungry and looking at food (a proto-request);
  - then 要X and 饿 → the caregiver fetches. world2 already does this.
  - 这是什么 is answered (world2 `_r_answer`);
  - requests are modelled in caregiver chat.
- **Child evidence (CITED):**
  - pointing to request and to share at 10–12 months (Bates, Camaioni & Volterra 1975);
  - infants point to get information and learn better about what they asked for (Begus, Gliga & Southgate 2014);
  - children's questions are information-seeking and persistent (Chouinard 2007).
- **Status:** no agent code exists today (MouthOnly in world2). This ability is the cleanest "no new code" test.
- **Falsifier:** lesion S4 → asking stays near 0; lesion S1 → requests no longer track hunger.

### 1.3 The drives the brief asked about

- **Infant-directed speech (IDS) preference.**
  - The effect is real but moderate (d = 0.35; ManyBabies Consortium 2020).
  - Its job here is a learning gain on speech addressed to me. In world2 that keeps peer chatter from dominating, which is the trust problem of review 2.
  - Directed speech, not overheard speech, predicts vocabulary (Weisleder & Fernald 2013).
- **Social contingency.**
  - Strongest single predictor in the environment:
    - responsiveness predicts milestones (Tamis-LeMonda, Bornstein & Baumwell 2001);
    - conversational turns matter more than word counts (Romeo et al. 2018; Gilkerson et al. 2018: turns at 18–24 months explain 14–27 % of the variance in outcomes 10 years later);
    - live exposure teaches phonetic contrasts, video does not (Kuhl, Tsao & Liu 2003).
  - Perfect contingency marks one's own body (Watson 1972; Bahrick & Watson 1985); that is R3.
- **Reward from caregiver responses.**
  - It should be computed by the child (S2, S3) from perceived cues, not handed out by the world.
  - Today `reward_ok = 1.0` exactly when the reply is acceptable (bench `world.py`; world2 praise +1). That is a privileged teacher signal (PROVED from code).
- **Curiosity / learning progress.**
  - It accounts for babbling stages and the later shift to imitation (Moulin-Frier et al. 2014), and for asking and "Goldilocks" attention.
  - Use learning *progress*, not raw error, so unlearnable noise stops being attractive (Oudeyer, Kaplan & Hafner 2007).
  - Later it can reward thought operations (ARCH_v2 open problem 2).
- **Prosocial drive (IDEAS #17).** This is S3 applied to anyone, third parties included.
  - Fixed S3 weights are "values that cannot be unlearned"; skills are learned on top.
  - Children help at 14–18 months (Warneken & Tomasello 2006), want the other person helped rather than to do the helping themselves (Hepach, Vaish & Tomasello 2012), and protest harm to third parties by age 3 (Vaish, Missana & Tomasello 2011).
  - This needs affect cues in the world.

---

## 2. The world and caregivers

### 2.1 What the world must provide, against today's worlds (status PROVED from code and READMEs)

| feature | why (CITED) | bench | life | world2 | change |
|---|---|---|---|---|---|
| contingent response to vocal bouts in pauses | Goldstein & Schwade 2008; Gros-Louis et al. 2006 (mothers responded to over 70 % in the lab; home rates are lower); Takahashi et al. 2017 | stage A play only | some initiatives answered | initiatives answered (greeting, naming, calling, asking, requests) | respond to word-like babble early, then fade |
| formats: ask and answer herself, fading with the child's success | Ninio & Bruner 1978; Snow 1972 | a model-reply tutor exists, but the J protocol uses recast (0 such exchanges) | 0 such exchanges; 39 overheard pairs | 30 % of seen items, a fixed rate not tied to the child | per task, falling from about 0.9 to 0.1 as the child succeeds |
| corrections in natural forms | Chouinard & Clark 2003; Saxton 1997 | the plain correct answer, reward 0 | plain | plain correct answer | replayed pair / 不是X，是Y / plain |
| perceived, imperfect approval | Fernald 1993; Brown & Hanlon 1970 | reward number 1.0 exactly when acceptable | same | praise plus a reward number | approval prosody and a smile; no number |
| prosody and faces | Csibra & Gergely 2009; ManyBabies 2020 | none | none | identity and pointing units; no talking, gaze or affect | IDS register, contours, face units |
| follow-in naming | Tomasello & Farrar 1986; Yu & Smith 2012 | pointing | pointing | pointing, fovea | name what is held or in the fovea |
| answering the child's questions and requests | Chouinard 2007; Begus et al. 2014 | no | no | yes | keep; add repeat-on-"啊?" |
| clarification requests | Saxton 1997; Dingemanse et al. 2013 | no | no | no | add |
| noise and mishearing | Leibold & Buss 2013 | σ option | σ option | σ, distance, crowds | a schedule plus masking bursts |
| many speakers | Houston & Jusczyk 2000 | 2 voices | 2 voices | 3 caregivers plus agents | add visitors with new voices and faces |
| caregiver turns of several lines | gives "the speaker went on" examples for role learning | no | rare | some | add comments |
| function and consequence | grounding of food categories | none | none | eat, taste, hunger | use for class tasks |
| real images, continuous across views | IDEAS #16 | factorised codes | factorised | factorised | local3d codes |

### 2.2 How much experience

| | heard lines | heard syllables | about how much child input |
|---|---|---|---|
| staged J protocol (A 300 + B, BC, BCDD 1,000 each; 486,918 ticks), PROVED `count_input.out` | 10,827 | 40,983 | 2–3 days of total input; 4–60 days of child-directed speech only |
| one 10⁶-tick life (seed 5, recast), PROVED | 4,071 | 15,457 | about 1 day |
| world2 day, 4 agents | 94 lines addressed per agent | about 960 audible (12 % of 24,000 ticks, 3 ticks per syllable) | about 1/20 of a day |
| world2 day, 16 agents | 25 lines | about 560 | about 1/30 of a day |
| child, per day | — | about 18,000 | — |
| child-year | — | about 6.6 M | first words at about 12 months |
| child at 24 months | — | about 13 M | about 300 words produced (Fenson et al. 1994) |
| by age 4 | — | 13–45 M words | Hart & Risley 1995 |

How the child rows were built:
- **Per day.** LENA adult word counts are of the order of 10⁴ per 12-hour day (Gilkerson & Richards 2008; Gilkerson et al. 2017, CITED). Child-directed speech alone ranges from 670 to over 12,000 words per 10-hour day (Weisleder & Fernald 2013, CITED). I used 12,000 words × 1.5 syllables per word; the 1.5 is ASSUMED.
- **Per year.** 18,000 syllables a day × 365 ≈ 6.6 M (PROVED arithmetic).
- **Cultural range.** Children of the Tsimane hear under 1 minute of child-directed speech per daylight hour and still learn (Cristia et al. 2019, CITED). Overheard and peer speech carry a lot there.

**Per-item exposure** (PROVED): each of the 12 object names is heard 356–582 times in the staged protocol (mean 451). For a child, a frequent noun at 0.1–0.3 % of input comes 12–36 times a day (ASSUMED rate), so 451 exposures is 2–6 weeks.

**Turns.** The protocol gives about 4,500 free replies. A toddler has hundreds of conversational turns a day (LENA).

**Reading.** The agent is not short of repetitions. It is short of:
- duration and variety: one child-year is about 160 staged protocols, 430 lives or 6,900 world2 days at 4 agents (about 1.7 × 10⁸ ticks);
- cues;
- contingency on its own attention and attempts.

### 2.3 Two analyses

**(1) Lexical cue, PROVED** (`cue_count.out`; life seed 5, 10⁶ ticks, silent agent).
- Questions: 1,776, of which 0.871 contain a question syllable.
- Social openers (你好, 再见 and so on): 852, of which 0.000 do.
- Statements: 0.000. Recasts: 941, of which 0.000 do, and 0 contain 不是.

What follows:
- A learned "my turn next" predictor would find most questions from words alone.
- Greetings need another cue: prosody, gaze, arrival. v2.1's `ral_q` failed exactly there: life social 0.15 / 0.03 because 你好 was not taken as a question (CITED).
- A correction carries no marker at all.

**(2) The role toy, SIMULATED** (`role_toy.py` / `.out`).
- **Setup.** 5 seeds; 40 questions × 12 answers; a bounded delta rule (η 0.2); the agent repeats 80 % of corrections. Each cell gives accuracy after tutored exchanges, then after untutored ones (cumulative).
- **The three learners:**
  - **ARCH_v2-like:** the first other-voice line after the question is its answer; moving on weakens the agent's last answer.
  - **v2.1 RAL-like:** the same, but the step is cancelled when the agent speaks after the line, unless the line carries a reward.
  - **generic adjacency:** no role rule (as defined in §0).

| world (format share, correction form) | ARCH_v2-like: tutored 1,000 → untutored +900 / +3,000 | v2.1 RAL-like | generic adjacency, no rule |
|---|---|---|---|
| 0, plain (today's J protocol and life) | 1.00 → 0.00 / 0.00 | 1.00 → 1.00 / 1.00 | **0.00 → 0.00** |
| 0, replayed pair | 1.00 → 0.00 / 0.00 | 1.00 → 1.00 | **1.00 → 1.00 / 1.00** |
| 0, 不是X是Y only | 0.06 → 0.00 | 0.32 → 0.32 | 0.00 |
| 0.3, plain (world2 "mixed") | 1.00 → 0.00 | 1.00 → 1.00 | **1.00 → 1.00 / 1.00** |
| 0.3, 不是X是Y | 0.98 → 0.00 | 0.99 → 0.99 | **1.00 → 1.00 / 1.00** |

What the toy shows:
- A learner with zero role rules is immune to repeating corrections and to untutored drift.
- That holds **only if the world gives the answer right after the question**: formats, overheard pairs, replayed pairs.
- A 不是X，是Y correction teaches content only with formats. Its job is the sign of the value update, which fits Saxton's contrast account (1997).

**Toy limits:**
- items are identities only; no frame × filler generalisation;
- no noise and no timing;
- speakers are known through the efference copy.

**Interim v2.2 data** (CITED, read-only, in progress: `v22/out/cue_test_base.txt`, `agg_steps.txt`):
- Timing does not separate the two. A 40-tick window catches every correction, but also 33–35 % of questions.
- The agent repeats 20–23 % of corrections, against 8–9 % of questions.
- The reply-veto variants raise the untutored loss at 900 dialogues to 0.44 (base −0.02). After 3,000 untutored dialogues accuracy is 0.008.

So "speak less after reactions" is the wrong lever. Children keep talking, and they learn from adult models.

---

## 3. Concrete world and curriculum proposal (world code only; every option off by default so the references stay bit-identical)

### 3.1 World additions (world2 `care.py`, `lex.py`, `core.py`; bench `physics.py` voice offsets)

**W1. Voice register and contour.**
- The IDS register is the speaker's voice offset +1, with the last syllable lengthened by one beat.
- Contours are offset shifts on the line's last syllable:
  - question rise +1;
  - approval +2 and louder;
  - prohibition −1, louder and short;
  - statement 0.
- This reuses the existing voice-offset machinery, and the ear window shape does not change (ASSUMED rendering).

**W2. Face units in the person code:** talking (on while that person's voice sounds), gaze at the viewer, pleased, upset or crying. That adds 4 units to the 64-unit block, or to the 128-unit fovea block (ASSUMED layout).

**W3. Option `reward='perceptual'`.**
- The `reward` observation keeps only eating.
- Praise becomes an approval contour plus a pleased face. It is based on truth and imperfect: about 0.8 of correct answers approved, about 0.1 of wrong ones (ASSUMED rates; to be set from Brown & Hanlon 1970 and Chouinard & Clark 2003).

**W4. Caregiver behaviours.**
- formats with self-answers fading per task as the child succeeds;
- correction forms: replayed pair 0.3, 不是X，是Y 0.4, plain answer 0.3;
- "嗯？/什么？" after a garbled or non-word answer;
- repeating the last line when the child says "啊？/什么？";
- follow-in naming of the held object or the fovea object;
- waiting with direct gaze after addressing the child;
- multi-line comments;
- responding to babble at about 0.7 early, fading to about 0.3;
- visitors with new voices and faces;
- adult–adult corrections in their own chat.

**W5. Noise schedule.**
- σ and crowd density rise with the phases.
- A masking burst covers one syllable of a predictable line with p 0.05 per line; this is practice for phonemic restoration.

**W6. Games (§3.4) and W7, the local3d eye (§3.2).**

### 3.2 local3d as the eye (IDEAS #16)

**What `threednet/model.py` gives:**
- fixed, local, random wiring (each unit sees itself plus 3 random neighbours) with an exact top-k per layer;
- similar images get similar codes (ρ = 0.989), with no collisions;
- a foveated input that keeps the geometry and only blurs the edge still retrieves 100 % (`FOVEATED.md`);
- identity comes from a slowly integrated state over continuous viewing, not from fixed neurons. Fixed identity neurons were a negative result: 157 people, 0 exclusive neurons (`EXCLUSIVE_NEURONS.md`, `LFW_RIGOROUS.md`).

**Plan:**
- Cache codes offline for each object kind × view × lighting. That needs object image sets with rotation series, such as COIL-100-type data (ASSUMED source; licence to check). Cache faces for the caregivers (LFW, already used in local3d).
- The fovea code is the cached code of the front cell's object at the current relative heading.
- Holding an object while turning gives a rotation series: the continuity the slow state needs.
- Meaning comes from words (IDEAS #16).
- **Cost:** L5 has about 5 M units at 196 × 108. Use 32–48 px inputs, keep top-k, and project randomly to about 1–2k binary units (ASSUMED sizes). World2's speed then stays a table lookup.

### 3.3 Phases, gated by milestones (each adds world features only)

| phase (child analogue) | world features | milestones that open the next phase | heard syllables (world2 days at about 1,000 a day) |
|---|---|---|---|
| P0, newborn (0–6 months) | one near caregiver, IDS, quiet, faces, responds to bouts in pauses (0.7), imitates word-like babbles, feeds on hungry vocalising, peekaboo | babbling covers ≥ 90 % of lexicon syllables without `world.voice` (`physics='lawful'`); more vocalising to a contingent than to a yoked (non-contingent) caregiver; overlap falling | 0.1–0.3 M (100–300) |
| P1, joint attention and first words (6–15 months) | follow-in naming, pointing, formats with high self-answer, give-and-take, greetings on arrival, local3d picture book | the fovea moves to the named object ≥ 0.7 for ≥ 20 kinds (comprehension through active vision; the Looker baseline reaches 0.67); first appropriate namings; empty-room 你好 ≤ 15 per tenth of life | 0.3–1 M |
| P2, answering and asking (15–30 months) | fading formats, correction forms, approval, clarification requests, requests and 这是什么 answered | held-out frames above the best constant; accuracy after a correction higher than before; asking ≥ 30 % of initiatives with a partner present; requests track hunger | 0.5–1.5 M |
| P3, crowds, noise, many speakers (2–3 years) | meals with everyone, σ 0.5–1, visitors, overheard chat with corrections, masking bursts | restoring masked predictable words; "啊?" followed by repeats; learning more from adults than from error-prone peers | 0.5–1.5 M |
| P4, memory and the absent (3 years +) | facts across days, where things were put, 刚才, songs, overheard adult corpus | retention > 0.3 at ≥ 2 encounters | 1 M + |
| P5, values (#17) | requests that would upset someone, third-party distress | declines harmful requests and still helps with ordinary ones | — |

**Total:** about 2.5–5 M syllables. That is about 60–120 staged protocols, or 4–9 child-months of input: 2,500–5,000 world2 days, or 0.6–1.2 × 10⁸ ticks.

### 3.4 New abilities enter only as world levers

| ability | world lever |
|---|---|
| counting | the caregiver counts while pointing in turn |
| colour | contrastive naming (红色的，不是蓝色的) |
| object permanence and "where" | hide-and-find, peekaboo |
| giving and thanking | give-and-take (给你 / 谢谢) |
| sequence memory | songs and routines |
| past events | 刚才 talk after changes |
| helping and refusing | requests, and third-party distress (#17) |
| peer talk | peer play |

The test for each: the ability's curve rises only after its lever is switched on, with the agent's code hash unchanged.

### 3.5 Compute (ASSUMED)

- numpy life speed: 10⁶ ticks ≈ 525 s (CITED, review 2), mostly silent ticks.
- Noisy world2 makes every tick a full tick. At an assumed 500–1,000 ticks/s, 0.6–1.2 × 10⁸ ticks is about 17–67 core-hours per agent.
- The A800 runner caps jobs at 90 min, so lives need checkpointed, chained jobs.
- The GPU port pays only for the scaled model (CITED, gpuport report).
- Age should be measured in heard syllables, not days.

---

## 4. Dedicated mechanisms that development could replace

### 4.1 Mechanism by mechanism

**1. Role-aware listening windows**
- **Today:** `v21/core_v21.py` RoleCore + RecDelta; keys `ral`, `ral_other`; W 12, W_other 6, T 60.
- **What it does:** a heard line's learning as an answer is held on a per-synapse trace. It is cancelled if the agent (or someone else) answers the line within a set number of ticks. It is kept if the line carries a reward after the agent's utterance, or if nobody answers it.
- **Developmental replacement:** R3, plus the order of lines in the thought, plus formats and replayed pairs in the world; S3 signs feedback.
- **Removed:** the cancel/commit timing rules.
- **If the drive or experience is missing:**
  - no formats or replays: no answers are learned (toy 0.00);
  - no R3: own speech teaches itself and drifts.
- **Falsifier:** E1.

**2. v2.2 correction-vs-question vetoes**
- **Today:** `v22/agent_v22.py` AgentV22 keys `veto_R1`, `veto_W`, `veto_rel`, `veto_gate`, `veto_fast`, `veto_eps`.
- **Developmental replacement:** a learned "who speaks next" predictor, plus prosody and gaze in the world, plus corrections in natural forms.
- **Removed:** all 6 keys.
- **If missing:** without cues only timing is left, and it overlaps (33–35 % of questions fall in the correction window). The vetoes silence the agent and stop its learning (loss 0.44 at 900).
- **Falsifier:** E2.

**3. The rejected question detector (`ral_q`)**
- **Developmental replacement:** the same SparsePR learner, with the target "did the speaker wait for me?" and features that include contour and gaze. Multi-line caregiver turns give the negative examples.
- **Removed:** nothing new to remove; it re-uses SparsePR.
- **If missing:** without a cue for greetings, 你好 counts as a non-question (social 0.15 / 0.03).
- **Falsifier:** E2.

**4. Social features**
- **Today:** `soc_feats` in `v21/agent_v21.py` (spoke ≤ 300 ticks ago; appeared or disappeared ≤ 60 ticks ago).
- **Developmental replacement:** O1 plus the generic time basis.
- **Removed:** the three windows.
- **If missing:** social replies collapse (seed 6: 0.07).
- **Falsifier:** E5.

**5. Turn reflexes 3, 5, 7, 7b and the urgency constants**
- **Today:** ThinkAgent, SynthAgent, ThinkBody.
- **Developmental replacement:** R2, plus P_oth and V_go learned under S2, with a partner who ignores overlaps and waits.
- **Removed:** the 7-tick window, the "floor taken" prior, the 10-tick rule and the brake.
- **If missing:** a partner who always waits gives no timing pressure. A partner who answers overlaps teaches overlapping. Early overlaps break the 0-error rule.
- **Falsifier:** E5 and §1.2(a).

**6. Offline babbling through `world.voice`**
- **Today:** develop's R9 babbling memory.
- **Developmental replacement:** R4 + S4 + S2 (the caregiver imitates word-like babbles), with `physics='lawful'`.
- **Removed:** the privileged read.
- **If missing:** with random physics, p ≈ 3 × 10⁻⁵ per babble per character (CITED, bench README), so the lexicon is never covered. With no caregiver imitation there is no shaping.
- **Falsifier:** the P0 milestone.

**7. Hearing keys `h_relock`, `h_vbound`, `h_conf`, `h_nocreate`**
- **Today:** `hearing/agent_h.py` AgentH.
- **Developmental replacement:** predicted onsets; the speaker as a slow state; precision-gated recruitment (B2).
- **Removed:** these 4 as separate parts. `h_gate`, `h_level` and `h_denoise` stay, as instances of the area-general adaptation and normalisation rule, which the brain also has innately.
- **If missing:** with no early noise and no variety of voices, categories do not generalise (v2.1 went deaf at σ 1).
- **Falsifier:** E4.

**8. Pointing-only teaching of PRIME/attD, and "pointing = attended slot"**
- **Developmental replacement:** O2 as a weak prior, plus learned gaze following (Triesch et al. 2006), plus B2 gating co-activity learning.
- **Removed:** the hard wire.
- **If missing:** if nobody names what the child is looking at, the maps learn noise.
- **Falsifier:** the world2 fovea test (Looker 0.67).

**9. The meaning-specific hippocampal cue**
- **Today:** `meaning/mcore.py` MeanCore, `mean_hipcue`.
- **Replacement:** the index reads every association area, each normalised.
- **Removed:** the special cue.
- **Risk:** none; this is an architecture choice, not development.
- **Falsifier:** the meaning class probe.

**10. The world's correctness reward**
- **Today:** bench `reward_ok`, world2 praise +1.
- **Developmental replacement:** S2 and S3 reading perceived cues.
- **Removed:** the privileged teacher.
- **If missing:** if the cues are clean and perfectly tied to correctness, the rule has only moved into the world. Make the cues imperfect and truth-based.
- **Falsifier:** E3.

**11. Fixed order of thought operations (ARCH_v2 §5 #10)**
- **Developmental replacement:** basal-ganglia selection of operations, rewarded by S4 or by gained confidence.
- **Removed:** later.
- **If missing:** without S4, extra thinking time adds nothing (CITED, think report).

**12. `match_n = 6` (blocks one-shot naming)**
- **Developmental replacement:** mutual exclusivity from competition, plus novelty through B2.
- **Removed:** the constant.
- **Falsifier:** one-shot naming in world2.

### 4.2 Drive-lesion predictions (ASSUMED; tested in E5)

| lesion | what is predicted to fail |
|---|---|
| S2 | no turn timing; babble not shaped; greets empty rooms; requests not learned |
| S3 | wrong answers never suppressed; no IDEAS #17 values |
| S4 | no "啊?" repair and no 这是什么 questions; babbling stalls at known sounds |
| S1 | no grounding of edible categories; requests do not track need |
| O1 | social replies collapse |
| O2 | peers' errors learned as much as adults' speech; slower word mapping |
| R3 | own speech teaches itself and drifts |
| B2 | no restoration, or over-restoration; noise categories |

---

## 5. Experiments that would falsify the claims (cheap first; Phase 2 only)

- **E0 (local, minutes).** Extend `role_toy.py` with frame × filler items and mishearing.
- **E1 (life subclass, world change only).**
  - Agent: v2.1 with `ral=0`, own reply in the thought, and no next-line link across its own lines.
  - World: formats fading with success, plus replayed pairs.
  - Prediction: life free replies > 0.62 (the echo-or-constant reference), and drift after 900 untutored dialogues ≤ 0.05, with no role windows.
  - Fail: corrections do need a role rule.
- **E2 (prosody, gaze, a multi-line caregiver).**
  - A learned "who speaks next" gate replaces RAL and the veto keys.
  - Prediction: social ≥ 0.4 with `soc_feats` off, and greetings classed as "my turn" ≥ 0.9.
- **E3 (`reward='perceptual'`).** Prediction: learning curves within 10 % of the reward-number version when the cues are imperfect.
- **E4 (IDEAS #14).**
  - Same agent, generic precision-weighted hearing; one raised in noise, one raised clean.
  - Prediction: the noise-raised one restores a swapped or masked syllable (fault drop ≤ 0.15, against ≥ 0.30 for the clean-raised one). With a new object in view, it learns sound-alike new words no worse.
  - Fail: both restore equally, so experience is not the lever.
- **E5 (drive lesions in world2).** The matrix in §4.2.
- **E6 (decisive).** Frozen agent hash. World levers are switched on one at a time (§3.4). Each ability rises only after its lever, also at 1× and 4× size.

---

## 6. Risks and questions for the designer

1. **Development is slow.** It takes months of input, about 10⁸ ticks.
2. **Rules may move into the world.** That is acceptable only if real caregivers do the same thing (cited) at realistic, imperfect rates.
3. **The 0-mechanics-error principle against early overlaps.** Infants overlap, and the brake R2 keeps overlap low but not at zero while timing is learned. **Question for the designer:** must there be 0 errors from birth, or only after a maturity point?
4. **Over-restoration** (hearing what one expects) has to be tuned through B2.
5. **Curiosity attractors** (a "question explosion") are favourites: tune them, don't forbid them.
6. **One value broadcast mixes four drives.** Their four weights are innate constants. Different weights across agents give temperaments, which also answers the monoculture risk in #17.
7. **The bench changes role.** A developmental agent may score lower on the staged bench early on. Keep the bench as a regression test, and judge on world2 milestones.

---

## 7. Files (all in the scratch folder)

- `count_input.py` / `.out`
- `cue_count.py` / `.out`
- `role_toy.py` / `.out`

---

## summary_zh

1. **结论：孩子的能力来自一套很小的先天装备，加上几年有结构、随孩子变化的经历。**
   - 我建议的先天装备共 12 项：
     - 2 个全脑广播：价值/奖励误差（类多巴胺），精度/意外（类乙酰胆碱、去甲肾上腺素）；
     - 4 个先天奖励来源：饥饿、"我做了事有人回应"、看出对方高兴还是不高兴（也就是设计者 #17 的亲社会本能）、学习进步（好奇心）；
     - 2 个先天朝向：认出人（脸/声音/人来人走），对"冲我说的话"（儿语声调、看着我、指点）更注意；
     - 4 个身体反射：发音节拍、别人一开口就先刹住、自己动作的复制信号（自己的声音被预测掉，不用标签）、闲着时自发地咿呀和乱动。
   - 现在的模型有大约 50 个为某项能力手工加的部件和开关，多半是在替代缺了的驱力或世界里缺的线索。

2. **"把纠正当成问题"主要是世界的问题，不是机制的问题。**
   - 我数了一生的记录：87% 的问题里有疑问字（什么/吗/哪…），纠正里一个都没有，也从来没有"不是"。
   - 所有世界都没有语调、表情、眼神。社交奖励是世界直接算好的"对错分数"。
   - 一生模式里从来没有"大人自问自答"的示范。
   - 孩子学回答，主要靠大人自问自答（Ninio & Bruner）、旁听别人问答、大人把问答重说一遍；"不是X，是Y"和语气只是给对错打正负号。
   - 我的小模型显示：完全不用角色规则、只学"紧跟在上一句后面的话"、自己的话不算老师的学习者，只要世界有 30% 自问自答，或者纠正时把问答重说一遍，就学会（1.00），之后无人教 3,000 次也不忘（1.00）；在今天只有"直接改说答案"的世界里，它什么都学不会（0.00）。
   - 正是因为以前的世界从不示范"问题后面跟答案"，才逼得我们一条条加角色规则。

3. **经历的量：**
   - 整个分阶段测试 ≈ 4.1 万个音节，约等于孩子 2–3 天听到的话；一生 10⁶ 拍 ≈ 1 天；world2 一天 ≈ 孩子一天的 1/20。
   - 孩子一年约 660 万个音节，约等于 160 次分阶段测试。
   - 但每个物名已经听了约 450 次，不缺重复；缺的是时长、多样性、线索，以及大人跟着孩子回应。

4. **发育可以替掉的部件：**
   - v2.1 的角色时间窗、v2.2 的 6 个"少说话"开关、几条轮流说话的反射和时间窗、三个手调的社交特征窗口；
   - 咿呀学语偷看世界发音表的捷径、4 个听觉开关、"只在指点时学词—物"的硬连线、世界直接给的对错分数。
   - 替不掉的是通用核心：想法里句子的先后顺序、任何脑区都能把词送到嘴边的同一个候选池、组合单元、海马的一次记忆和清醒回忆、全脑通用的归一化和适应，还有身体。

5. **世界和课程方案（只改世界代码，用 world2 加 local3d 眼睛）：**
   - 自问自答随孩子进步逐渐减少；纠正用自然形式（重说问答、"不是X，是Y"、直接说答案）；
   - 表扬变成语调和笑脸，而且像真父母一样不完美、看事实对不对；
   - 加儿语声调、疑问/赞许/禁止的语调，以及"在说话、看着我、高兴、难过"的表情单元；
   - 回应咿呀、回答孩子的问题和要求；听不清时问"什么？"，孩子说"啊？"就重说；
   - 噪声和说话的人逐步变多；用真实物体图片，转着看能连成一条轨迹。
   - 分 P0–P5 阶段，按里程碑开下一阶段。新能力只通过新游戏、新物体、新的照料者行为进入世界。

6. **决定性实验：**
   - 冻结智能体代码，一项项加世界里的"杠杆"，每项能力应该在它的杠杆出现后才上升；
   - 逐个去掉驱力，应出现预测的失败；
   - 设计者 #14：在噪声里长大的模型和一直听得很清楚的模型，比谁更会"听错了自己补回来"。

7. **需要设计者决定：** "机制 0 错误"是从出生就要求，还是长大到一定程度后才要求？婴儿早期也会和大人抢话。
