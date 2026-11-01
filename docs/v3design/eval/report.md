# Evaluation design: a sealed test that a fixed set of mechanisms learns new things

**Role:** evaluation member of the v3 design round.

**Labels:**
- **PROVED**: from code, a census script, or arithmetic.
- **CITED**: from a report or a paper.
- **ASSUMED**: a prediction or an unchecked input.
- **SIMULATED**: nothing in this report. There were no training runs in Phase 1.

**Scratch** (`arch/v2/v3design/eval/`):
- `analytic.py`: census of config keys and literals, specificity index, power.
- `budget.py`: compute budget.
- Each runs in about 1 s and under 50 MB.

**Read-only inputs:** the bench (`world.py`, `life.py`, `metrics.py`, `baselines.py`), `world2` (`core.py`, `care.py`), the v2.2 code and its interim `out/agg_steps.txt`, and the reports named in the brief.

## 0. Bottom line

1. **Why the count of mechanisms keeps growing is mainly a measurement problem.**
   - Every part added since ARCH_v2 was designed after its failure was seen, then judged on the same tasks that showed the failure. This applies to RAL, the say routes, the veto and dips, the seven `h_*` keys, the meaning cue in the DG, and v2.2's reply veto.
   - That is test-set reuse at the level of the architecture (Recht et al. 2019; Cawley & Talbot 2010; Dwork et al. 2015).
   - However honest each paired step is, it cannot show generality while the designers can see the test.
   - **The fix is about order in time: freeze the code first, then generate the test.**
2. **The proposed test is "Sealed-Life".**
   - Abilities come as *family specs* (data). A fixed, world-side interpreter runs them inside bench Life and world2.
   - **Three tiers:**
     - **S0**: public development families.
     - **S1-A**: public kinds, with content drawn at opening.
     - **S1-B**: kinds unknown to the designers. They come from random compositions of world-state accessors, drawn with OS entropy only after the freeze, plus optionally a few routines the human designer writes off the box.
   - The test is opened **once per frozen agent hash**, and the result is final.
3. **The code-freeze rule makes "mechanism" checkable.**
   - The agent is a library + a manifest + a typed parameter file.
   - A mechanism change is any of: a change to the library's AST, a change to the manifest's structure, a boolean or mode value, a degenerate parameter, or a per-world parameter override.
   - Lints enforce: no Chinese literals, no imports from the world, no numeric literals in the library.
   - Innate priors have a fixed budget.
   - The agent runs in its own process and sees obs only, behind an audit hook.
   - Caregiver timing is jittered in the sealed worlds, so rules fitted to the bench's timing break visibly.
4. **Measures.**
   - **Γ:** the mean normalised gain on held-out fillers of the S1-B families, on frozen copies, over the best trivial baseline on the same items.
   - **Learning:** shown by a multiple-baseline design, where families are introduced at staggered times in one life.
   - **Scaling:** one global factor *s* for every area, paired with life length L and world richness W, using iso-compute curves (Hoffmann et al. 2022).
   - **Minimality:** each mechanism must be necessary and broad (a specificity index).
   - **Coverage:** J2, life and hearing, checked against v2.2.
5. **Today's stack against this test** (details in §5):
   - **Generic in form:** the learning rules, i.e. delta readout, hippocampal index + awake recall, vigilance categories, instar + pruning, distributional meaning codes, RPE.
   - **Shaped to a task, the world or a metric:** most parts added since v2. PROVED examples:
     - RAL's 6-tick "someone answered" window, against the bench's 3-tick model reply;
     - gateR, with specificity 0.93;
     - the hippocampal recall routes, specificity 1.00;
     - `h_level`, the inverse of world2's attenuation law;
     - `soc_feats`;
     - per-world parameter sets;
     - "counting" that reads a count code from the eye;
     - babbling that reads `world.voice`.
   - **v2.2 (interim) shows the same pattern.** It adds 10 more keys, and its reply veto brings drift back: loss after 900 untutored dialogues goes from −0.02 to 0.45, and retention after 3,000 dialogues is 0.008 (CITED, in progress).
6. **Feasibility.**
   - The harness is about 1,800 lines of new code (ASSUMED estimate); bench and world2 files stay untouched.
   - A full opening (base cell + 3×3 scale grid + W=2 arm + ablations + coverage + the jitter test) costs about **10 A800 jobs, about 8 h wall** at 2 job slots (PROVED arithmetic on CITED timings).
   - Three things can run now, without v3:
     - validating S0 with the trivial references and the RNN;
     - the jitter falsification of RAL;
     - a whole-model scale build of v2.2.

---

## 1. Threat model: how designers cheat by accident (evidence from this project)

| channel | example here | closure (§) |
|---|---|---|
| Designing on the test | Every v2.1, v2.2 and hearing part was built after its failure was seen on the bench or life, then judged there (CITED: v21 §2–4, hearing §1, meaning §1). | test written after the freeze (2.6) |
| Tuning parameters on the test | Comparator constants "sit on a plateau with cliffs" found by tuning on the bench (CITED ARCH_v2 §5 #17). | one opening, pre-registered (2.6, 4.6) |
| Features shaped like the metric | `soc_feats`, "something appeared in silence within 60 ticks", tracks life's arrival schedule behind the empty-room-你好 metric (CITED v21 §4.1). | innate budget + jitter (3.3, 3.4) |
| Rules fitted to world constants | `ral_W_other` 6 ≥ the bench's `gap_model` 3; reflex 7's 10 ticks > `scene_lead` 5 ("PROVED from world.py", ARCH_v2 §5 #7b); transition window τ ≤ 7 fitted to the staged waits (#5). | jittered caregiver timing (3.4) |
| World gifts | Count and colour are eye codes (`Curriculum.vals` has `('cnt', n)`, `('col', c)`, PROVED); world2 has a count field (units 42–44) and fovea class bits. The skeptic found the colour gain comes mostly from the eye code (CITED review2). | gift audit (2.7) |
| Privileged reads | `DevAgent(world, …)` reads `world.seed`, `world.eye` and `world.warmup_windows()`; `babble(voice, n)` calls `voice.decode` / `voice.hear` and builds a `physics.Stream` (PROVED: `agent.py` l. 29, 46–54, 209–252; `agent_h.py` l. 46, 114–125). | process isolation (3.5) |
| Per-world switches | g0 0.3 staged / 1.0 life; urgency 0.15 / 0 / 0.02; "life adds THINK_LIFE + ct_assoc 0.5, wander 50" (CITED ARCH_v2 §5 #8, v21 §7). | one parameter file (3.2) |
| Fake scaling | The meaning round's "4×" enlarged one layer at α/4: −0.11 J (CITED). | global *s* (4.3) |
| Same-model imagination | Designer agents and a sealer agent are the same model, so they imagine the same "new" tasks. | random compositions + human-written families (2.4) |

## 2. Sealed task families

### 2.1 Three tiers

| tier | kinds | content (fillers, words, frames, classes, timing) | who sees it before the opening | what it can show |
|---|---|---|---|---|
| **S0** dev | public (§2.5) | public | everyone | nothing about generality; for development only |
| **S1-A** | public (the brief's kinds: class, counting, order, games, routines, repair) | drawn at opening (fresh words from the 3,970 voice chars, fresh class partitions, fresh frames, jittered timing) | nobody | generalisation to new content |
| **S1-B** | not in the signature list of §2.3 | drawn at opening | nobody | **generalisation to new kinds** (the claim) |

- **This report makes the brief's example kinds public.** They can therefore only be S0 or S1-A. S1-B kinds must be absent from everything any designer has read, including this report.
- **Two gaps are reported.**
  - S0 → S1-A is the content-overfitting gap.
  - S1-A → S1-B is the kind-overfitting gap: Chollet's (2019) "developer-aware generalisation".

### 2.2 A family is data, not code

The spec, which a fixed interpreter runs:
- `world`: `life` | `world2`;
- `introduce_at`: a fraction of life;
- `share`: the share of topics or turns after introduction;
- `builder`: which scene or world state to set up, using the accessor library;
- `routine`: a sequence of existing world operations (say a line by an existing voice, point, set the scene, wait, make an object appear or disappear, mask a syllable, give or take);
- `prompt` and `answer` as functions of world state;
- `fillers` with `held_ratio` 0.3;
- `teaching`: the world's existing tutor rules, i.e. probe 0.15, model / recast, held-out pairs never answered;
- `probes`: 50 / 50 / 100 held-out items on frozen copies at 1/3, 2/3 and the end of life.

**World-side rules** (checked by the implementer's tests):
- **Only existing primitives:** existing voices, existing reward and praise channels, eye codes made by the existing code generator (a new kind gets a new code from the same rule).
- **No new obs channel or unit:** the obs-schema hash must equal the freeze record's.
- **Producible words:** every answer char must be producible. `mouth_vocab` = lexicon ∪ S0 chars ∪ (S1 chars, added at opening). During development a random decoy set of the same size is added, so no mechanism can depend on the size of the producible set.

**Hooks, all without touching bench files** (PROVED from code):
- **Bench life:** `LifeF(Life)` adds `'family'` to `_visit`'s topic list, which is a local list (`life.py` l. 594), so a subclass copy of `_visit` is needed. It also adds a masking source in `_say`, and a privileged `bench_line` hook for the RNN.
- **world2:** `CareMixin._turn` dispatches `getattr(self, '_t_' + kind)` over `p['w_turn']` (`care.py` l. 410–420). `World2F` adds `_t_family` with a weight in `w_turn`.
- **Probes:** `World.run_script` (`world.py` l. 875) runs frozen-copy probes with the same seed's physics and eye codes. The eye codes depend on the seed only (`Curriculum`).

### 2.3 "New in kind" is checkable

Each task gets a **signature** = (input sources, operator chain, output type).

Existing signatures, from `world.py` `TASKS` and world2 `care.py`:

| signature | tasks |
|---|---|
| pointed / slot / held percept → its name | what, slot, holding, who, room |
| name → location word of its percept | where |
| name → attribute word of its percept (colour, count, size) | colour, count, size |
| word vs percept → yes / no | check |
| line → fixed line | ritual, greetings |
| frame + name → copy of the name | offer |
| constant | taste |
| told fact → recall later | name, myname, person, novel, oneshot, where-memory |
| seen scene → recall after a change | before |
| word → motor goal | fetch, give, come |

- S0 adds the signatures of §2.5.
- A candidate S1-B family is **rejected** if its signature equals any listed signature or any S0 signature.
- Equality is decided on the operator chain, with inputs abstracted. Two families of the same kind with different words count as the same signature.

### 2.4 How S1-B is generated, who holds the seal, and the opening

**Programmatic draw (S1-B-G).**
- Inputs: a public *accessor library* of world-state reads (objects and their properties, slots and cells, rooms, persons, holdings, events with times, lines said by whom) and a public *composer*. The composer builds random depth-≤2 compositions of accessors into a prompt → answer function.
- The draw happens **at opening**, with `secrets.token_hex` entropy.
- It keeps the first k_G = 4 compositions that pass the validity checks of §2.7 and the new-kind check of §2.3.
- The list of operators the composer may produce is **not** published in this report. Only the accessors are, and they are plain state reads.

**Human families (S1-B-H, recommended).**
- The designer (human) writes, off the box, 3–4 one-line descriptions of things a 2–3-year-old learns at home with caregivers that the house world can show.
- Nothing is sent until the opening.
- Optional: publish `sha256sum` of the text now, for an audit trail.
- This is the only source not correlated with the designer agents' imagination.

**Fallback (S1-B-A).**
- If the human prefers not to write families, a fresh agent writes them **after the freeze**, from a brief written now ("toddler-at-home routines whose signature is not in §2.3").
- It is weaker because of the shared model; this is flagged in the result.

**The seal is the time order. Timeline:**

| step | what happens | who |
|---|---|---|
| **T0 (now)** | The eval package, the accessor library, the interpreter, S0, the pre-registration (§4.6) and the implementer's brief are frozen and pushed to GitHub. | coordinator |
| **T1 (build)** | Designers use S0 and coverage freely. S1 does not exist yet, except as the human's private text. | designers |
| **T2 (freeze)** | `freeze.py` lints and hashes the agent; `FREEZE.json` is pushed (GitHub timestamp). | coordinator |
| **T3 (opening, once)** | (a) The human sends the family text. (b) A fresh implementer agent implements S1-B-H and runs the S1-B-G draw. This agent has never read the agent code, gets the world API docs only, and may run only the oracle, the trivial references and the RNN. (c) The specs, the draw record and the validation are committed. (d) The pre-registered job wave runs on the A800. The runner refuses to run if the agent hash ≠ FREEZE, or if an opening record already exists for that hash. (e) The results and the analysis script's output are pushed. | human, implementer, coordinator |
| **T4** | S1 becomes public and joins S0. The next version needs a fresh S2. Sealed sets are consumable, as in the reusable-holdout budget (Dwork et al. 2015; Blum & Hardt 2015). | — |

**Re-running after the opening:**
- A crash re-run uses the same hash and seeds.
- A world-side bug fix is logged, and both results are reported.
- Any agent change makes a new version, and the new version needs a new sealed set.

### 2.5 S0 public development families

All families are balanced; every answer needs information from the world.

| id | world | routine | prompt → answer | held-out | trivial level (ASSUMED) |
|---|---|---|---|---|---|
| class | Life | During visits: 宝宝吃X / X可以吃 for edible X; X不能吃 for others. Held-out X are never asked in the question frame. | X能不能吃 → 能 / 能吃 vs 不能 / 不能吃 | 30 % of X (evidence statements only) | 0.5; the bench eye has no class bit (PROVED) |
| count | Life | All 3 slots filled (p_empty 0); N ∈ {1,2,3} copies of X; every count code fixed at 一个, so the eye's count field carries nothing | 有几个X → N | 30 % of X | 0.33 |
| order | Life | Partner present and silent; X appears, Y appears 20–200 ticks later; mention order counterbalanced against appearance order | X和Y谁先来 → X / Y | 30 % of pairs | 0.5 |
| missing (game) | Life | 闭上眼睛 → eye dark for 20 ticks, one of 3 objects removed → 睁开眼睛 | 什么不见了 → X | 30 % of X | about 0.08 |
| count-on (game, new turn structure) | Life | Caregivers count aloud while pointing (一…十). Game 我们来数数: she says n, the child says n+1, she says n+2, … | n → n+1 | starts ≥ 6 never played; heard only in counting aloud | about 0.1; the n-gram over taught pairs fails held-out |
| repair (twins) | Life | Exposed twin: one non-initial syllable per line masked by a noise burst with p 0.1; a clarification (什么?) gets a repeat. Control twin: p 0. | masked B/C questions → their answers | — | the control twin |
| who | world2 | 谁拿着X | → person name | 30 % of (X, person) | about 0.25 |
| go-to-room (routine) | world2 | 去厨房 / 去卧室, …; success = in the room within 400 ticks; praise | motor | 1 room never requested (named elsewhere) | the random actor |
| box (new object) | world2 | A 盒子: putting X in hides it; 打开 shows it | X在哪里 → 在盒子里 | 30 % of X | — |

- **Repair** follows IDEAS #14's addendum: phonemic restoration (Warren 1970) and other-initiated repair (Schegloff, Jefferson & Sacks 1977).
- **Count-on** tests a turn structure where the line after mine is a continuation, neither feedback nor a question.
- **Counting is hard for children** (Wynn 1990, 1992). It is a fair test of slow learning, not a quick win.

### 2.6 Validity checks on every family (run by the implementer before the opening, never with the agent)

1. **Oracle** ≥ 0.98.
2. **Floor:** echo, last, prior, lookup, n-gram and the best constant on the same items stay below 0.9 on held-out items. Families where the floor is ≥ 0.9 are dropped.
3. **kNN baseline** (§6): no more than half the gap above the floor (b + 0.5(1 − b)). Otherwise the family is a lookup in disguise.
4. **Gift audit:** a logistic probe from the obs at prompt end to the answer, trained on the oracle's items, must not reach ≥ 0.9 on held-out fillers. Otherwise the family is a perception lookup, which is allowed only in S0.
5. **Learnable from experience:** the RNN baseline (§6) learns it, or else the family is reported as "unlearned by both".
6. **Probe count:** ≥ 100 held-out probes at the last checkpoint.

## 3. The code-freeze rule

### 3.1 The frozen object

- **The agent consists of three parts:**
  - **L**, the core library: one implementation per rule;
  - **M**, the manifest: areas with counts, activity fractions and projections; rule instances; broadcasts; innate priors with citations;
  - **P**, the parameter file: typed numbers only.
- **The constructor receives only** `(obs_spec, warmup_windows, seed, s)`. Today's `DevAgent(world, …)` would have to change (PROVED, §1).
- **`FREEZE.json`** holds SHA-256 hashes of:
  - L's normalised AST (docstrings and comments stripped);
  - M and P;
  - the obs schema;
  - the eval package;
  - the pre-registration.

### 3.2 Mechanism change versus parameter (checkable)

| change | class | check |
|---|---|---|
| Any change in L's normalised AST, including a numeric literal in L | **mechanism** | AST hash. L may hold no numeric literal other than 0, 1, −1; every other constant lives in P. Lint: `ast` walk. |
| Adding, removing or retyping an area, projection, rule instance, broadcast, gating condition or innate prior in M | **mechanism** | M structure hash (numbers masked) |
| Any boolean or string mode (`vis_rec`, `sleep_mode='strength'`, `dip_scope='pred'`) | **mechanism** | P may hold no bool or str |
| A P value that makes a rule degenerate: a rate of 0, a fraction of 0 or 1, a threshold outside its input's range | **mechanism** | range check per type |
| A per-world value (bench vs life vs world2) | **mechanism** | one P for every world; the agent is not told which world it is in |
| Branching inside L on an area name or a string | **mechanism** | lint: no comparison with string constants inside rule classes |
| A numeric P value inside its declared, non-degenerate range | **parameter** | type ∈ {count (scales with *s*), fraction, rate per event, time constant in ticks, normalised threshold} |
| *s*, L, W | experimental factors | the only inputs that change in sealed runs |

- **What counts as generic (automatic part):** a rule class in L is generic only if M instantiates it in ≥ 2 areas, with the same code and no area-specific branch.
- **A rule used once** must be declared an innate prior, which is budgeted, or it is a module.
- **Reported mechanism count:** |M.rules| + |M.broadcasts| + |M.innate|.

**Census of today's v2.2 stack** (PROVED, `analytic.py`; files `core*.py`, `agent*.py`, `rules.py`, `randlayer.py`, `hear.py`, `agent_h.py`, `mcore.py`, `meanlayer.py`):
- **258 distinct config keys.** Cfg classes alone hold 31 booleans, 100 numbers and 2 string modes.
- **492 numeric literals other than 0, 1, −1 in the code:** core.py 115, randlayer 83, agent_think 43, meanlayer 41, …
- **v2.2 adds 10 more keys:** `veto_R1`, `veto_W`, `veto_rel`, `veto_gate`, `veto_fast`, `veto_eps`, `dip_scope`, `dip_g`, `dip_pthr`, `gain_reply` (`agent_v22.py` docstring).
- **Content-free:** Chinese text appears only in 2 comment lines of `core.py`, and no world module is imported except `physics.Stream`, which is used for babbling. The literal content lint therefore already passes. Content enters through **structure**: the 14 hand-made feature groups and 31–34 pair types of `randlayer.PAIRS` / `core.Core` (e.g. CP / CN / CC, the copy-route candidates; familiarity bins at 20 / 300 exposures).

### 3.3 Innate priors

- **Each prior is an M entry** with four parts:
  - what it computes, in terms of its own input area only, with no word, task or world event;
  - why the brain has it (citation);
  - what could later replace it if learned;
  - its budget slot.
- **The budget B is fixed before the v3 build and never increases.** The synthesiser sets B; ≤ 10 is suggested.
- **Today's stack is far over any such budget:**
  - ARCH_v2 §5 alone lists 23 innate items;
  - add RAL's three role rules, the seven `h_*` keys, the social features and v2.2's veto family;
  - in all, more than 40 hand-set parts (ASSUMED count from the reports).

### 3.4 Features shaped like the metric or the world

- **Static lint.** Every P entry in ticks or counts is listed next to the world constants within ±20 %:
  - bench: `gap_model` 3, `model_wait` 12, `scene_lead` 5, `gap_answer` 45, `reply_cap` 60;
  - life: `greet_wait` 30;
  - world2: the attenuation steps and the 3-tick politeness rule.
- **Each match needs a stated reason:**
  - body physics (the 3-tick syllable, the 32 muscles) is allowed;
  - anything else is flagged.
- **Dynamic test, the binding one:**
  - sealed worlds draw every *caregiver* timing constant from U(0.5×, 2×) of its development value, at opening;
  - caregivers keep the rule "never start while someone in earshot talks", so overlaps stay the agent's own;
  - body physics is not jittered;
  - a feature fitted to a constant then breaks, and the breakage is visible in the jittered-versus-development gap.

### 3.5 Isolation: obs only, by construction

- **Separate process.** The world lives in the parent; the agent lives in a child process that receives obs through a `multiprocessing` Pipe. The eye is sent only when it changes, since it is cached and read-only in the bench.
- **Audit hook in the child** (`sys.addaudithook`, PEP 578):
  - it blocks imports of `world`, `world2`, `lex`, `metrics`, `life`, `physics` and `baselines`;
  - it blocks `open` outside the agent package and its checkpoint folder.
- **Canaries.** A canary GUID sits in every family spec, and the freeze lint greps the agent package for it, as BIG-bench does (Srivastava et al. 2022).
- **Frozen copies** are made by pickling the agent in the child and starting a probe child from the pickle.
- **Cost:** about 1.4 KB per tick and a round trip of about 20–50 µs → under 5 % of a 1,000 s life (ASSUMED pipe latency).
- **v2.2 needs a labelled exemption.** It cannot run under the hook because it babbles through `world.voice`, so in comparisons it is marked "privileged babbling".

## 4. Measures

### 4.1 The normalised score

- **Per family F,** on held-out fillers, on frozen copies:
  - G_F = (acc − b_F) / (1 − b_F);
  - b_F = the maximum over echo, last, prior, best constant chosen on the same items, lookup, n-gram and kNN.
- **Headline:** Γ = the mean of max(G_F, 0) over S1-B. Γ_A is the same over S1-A.
- **taste is out of coverage.** The constant 好吃 is always right, so b = 1 (CITED v21 §1.2).

### 4.2 Learning curves for abilities that come only through the world

**Multiple-baseline design** (Baer, Wolf & Risley 1968):
- in one life, each family is switched on at its own time t_F;
- the order is counterbalanced across seeds, with t_F ∈ [0.1, 0.5] of life;
- every family is probed at every checkpoint;
- families not yet introduced are the age-matched control for the ones already introduced.
- Twin lives are used only for repair, where the exposure is a property of the world.

**Reported per family:**

| measure | what it shows |
|---|---|
| G before introduction | must be < 0.1; otherwise the ability is innate or transferred. Report it as forward transfer (Lopez-Paz & Ranzato 2017). |
| G against n_F | exposures experienced (the world's `n_experienced` counter); fitted as G(n) = G_max(1 − e^(−n/n₀)) |
| G_max, n₅₀ | skill-acquisition efficiency (Chollet 2019) |
| post-introduction contrast against not-yet-introduced families | the learning effect |
| retention of earlier families and coverage tasks at later checkpoints | backward transfer (interference) |

### 4.3 Scaling (with the designer's note of 2026-11-01)

**Model size is one global factor *s*.** Every area grows by *s* together. A sweep of one module is not a scaling test. The meaning round's 4× changed one layer *and* its sparsity (α/4), so each unit trained 4× less often (CITED).

| quantity (today, s = 1) | rule |
|---|---|
| EAR-0 slots 2,400 → FEAT0 ≈ 1,153; Z categories ≤ 256; item / program capacity V 256 (+END), body programs ≤ 320; PRIME / attD rows V+1 | **× s** (counts) |
| random layer about 48k units (`randlayer` c_flat); meaning lines D 512; thought assemblies V, held items ≤ 12; line buffer ≤ 24 | **× s** |
| DG 6,000 units, k 120; CA3 / episode cells 10,000; BG gates and critic (per random-layer unit, automatic); P_oth / V_go keys (derived) | **× s**; k × s |
| active fractions: EAR-0 top 40 %, DG 2 %, random layer α 1 %, meaning 1/32, thought 12/256; vigilance 0.75; recall overlap ≥ 0.2 | **fixed**, so each unit is trained equally often at any *s* |
| fan-in per unit (EAR-0 4, random layer 3, DG 10–12) | **fixed**, so synapses ∝ s. Synapses per cortical neuron are of the same order across mammals (CITED: Braitenberg & Schüz 1998; DeFelipe et al. 2002; Herculano-Houzel 2009). |
| GEN readout | must become **sparse rows with a fixed fan-out** (e.g. 64, as the GPU plan in ARCH_v2 §10 already proposes). Dense N × V grows as s² (PROVED). This is a v3 core decision, before the freeze. |
| learning rates η | **fixed**. R2's step is divided by \|A\|, so the total change per event does not depend on size (PROVED from R2's form). |
| time constants in ticks (noise floor τ 500, time cells ≤ 30, eligibility 50–60, quiet moment 12, fatigue 20); recall episodes per quiet moment (8) | **fixed** (rates and times, not sizes) |
| hippocampal forgetting σ × 0.9995 per stored line (half-life about 1,400 lines at 10,000 cells) | **write it as a fraction of capacity** (half-life 0.14 × cells), so that it scales |
| thresholds that are absolute counts over a population (`conf_min` 6 eye units, `rf.sum() >= 8`) | **write them as fractions**, which is a code change before the freeze. Thresholds counted in exposures (`match_n` 6) stay fixed. |
| sensors and body (ear window 8×6, eye bits, 32 muscles) | fixed; they belong to the world |

**Scale lint (in `freeze.py`):**
- build at s ∈ {0.5, 1, 2};
- assert that every count is multiplied and every fraction and fan-in is unchanged;
- assert that s = 1 reproduces the reference bit for bit.

**Pairing with experience and world:**
- **L** = life length, in units of 10⁶ ticks.
- **W** = world richness:
  - W = 2 doubles the everyday objects (12 + 12 taken from `NOVEL`; their eye codes already exist in `Curriculum`, PROVED), names, caregivers and active families.
- Agent cost per tick ∝ *s*, so compute C ∝ s·L.
- **Iso-compute design** (Hoffmann et al. 2022):
  - for each C, vary (s, L) inversely and find s*(C);
  - fit E = 1 − Γ to E₀ + A·s^(−α) + B·L^(−β);
  - the RNN baseline gets the same grid with H = 256·√s, so its parameters ∝ s.
- **What counts as success:**
  1. α > 0 and β > 0, with CIs that exclude 0;
  2. along the diagonal (0.5, 0.5, 1) → (1, 1, 1) → (2, 2, 2), E falls at every step (paired over seeds);
  3. s* is non-decreasing in C;
  4. the gain from s = 1 → 2 is larger at W = 2 than at W = 1;
  5. no paired loss from s to 2s at the largest L.
- **How failures are read:**

| shape | diagnosis |
|---|---|
| flat in *s*, rising in L | limited by experience; at W = 1 this is fine |
| flat in both | limited by mechanism: **the signature of "this kind needs a new module"**. It is reported per family. |
| falling in *s* | a scale defect: an absolute threshold, or less training per unit (the meaning-round result) |

### 4.4 Minimality ablations

- **Every core mechanism m gets a null:** its rates set to 0, a fixed random replacement for structural parts, or its route removed.
- **Necessity:** removing m drops Γ ∪ coverage by ≥ 0.05 (CI excludes 0) on ≥ 4/5 seeds. Otherwise m is a passenger and must be removed.
- **Breadth:** the specificity index SI_m = (largest family drop) / (sum of drops), together with the number of families hit.
  - A generic mechanism needs SI ≤ 0.6 and ≥ 2 families, S1 families included.
  - SI > 0.8 means a task module in disguise, for example a planted counting module.
- **Computed from ARCH_v2 §8.3's lesions** (CITED values, PROVED arithmetic; only the tasks the report names):

| part | SI | tasks hit |
|---|---|---|
| scene focus F | 0.32 | 4 |
| learned attention | 0.37 | 3 |
| comparator / primes | 0.30 | 5 |
| BG prompt copy gate | 0.62 | 2 |
| hippocampal recall routes | **1.00** | 1 (name) |
| VR | **1.00** | 1 |
| v2.1 gateR (by stage) | **0.93** | — |

- On today's bench, SI is only weakly informative, because every bench task is a variant of "percept → word". It becomes informative with families of different kinds.

### 4.5 Coverage: today's abilities are kept

Measured against v2.2 on the same seeds and machine, with the canonical DG tie rule:
- staged frozen protocol, 5 seeds: J2, per-task held-out G against the best constant, drift after 300 / 900 / 3,000 untutored dialogues;
- life seeds 5 / 6: free replies against echo-or-constant, social, appropriate initiatives, empty-room greetings;
- hearing: the bench at σ 1 and world2 with 2 agents at σ 1;
- **hard constraint:** 0 overlaps, 0 garble, 3-tick syllables.

**Pass:** the mean retention ratio of per-task G is ≥ 0.9, and no task falls below 0.5× of v2.2. Count and colour are also reported with the gifts removed (random eye layout; count by instances).

### 4.6 Pre-registered criteria and statistics (fixed at T0)

**Run conditions:**
- opening seeds 101–105; development uses 0–6;
- the scale grid uses seeds 101–103;
- life 10⁶ ticks with the recast tutor;
- each agent runs with the eye layout it was frozen with.

**A family counts as learned if:**
- G_F ≥ 0.25 at the end; and
- the post-introduction contrast is > 0, with a one-sided 95 % bootstrap CI over seeds and items, Holm-corrected across families.

**Power** (PROVED, `analytic.py`): with 100 held-out probes per family per seed, 5 seeds and a seed SD of 0.05–0.08, the minimal detectable gain is 0.12–0.15 at 80 % power. For Γ over 6 families it is about 0.06.

**Claims:**

| claim | criterion |
|---|---|
| **pass** | ≥ ½ of S1-B families learned at (1,1,1); Γ_v3 > Γ_v2.2 (paired; 5/5 seeds or a CI excluding 0); scaling success shapes 1–2; coverage passes; every core mechanism necessary with SI ≤ 0.6 |
| **weak pass** | ≥ ⅓ of S1-B families learned, otherwise the same |

**Always reported:** Γ_RNN, Γ_A, and the S0 → S1-A → S1-B gaps.

## 5. Which current mechanisms pass and which fail

**Definitions used in the table:**
- **"Generic"** is the brief's definition.
- **"Shaped"** means fitted to one task, the world's constants, or a metric.
- **"Predicted"** results are ASSUMED unless marked otherwise.

| part (code) | generic? | shaped? (evidence) | predicted on the sealed test | verdict |
|---|---|---|---|---|
| R2 signed delta readout (`rules.Delta`; GEN, gates) | yes (Widrow & Hoff 1960; Rescorla & Wagner 1972) | — | needed by everything | **pass** |
| R4 hippocampal index + R10 awake recall (`rules.Hippo`) | yes (Teyler & DiScenna 1986) | the recall *routes* NEXT / COMPLETION are specific (SI 1.00) | memory families | **pass** (rule); routes **fail** |
| R5 vigilance categories (`rules.Templates`, Z) | yes (Carpenter & Grossberg 1987) | — | — | **pass** |
| random layer, instar + use-it-or-lose-it (`randlayer.RandMix`) | rule yes (Rigotti et al. 2013) | the pair schema is hand-listed: 14 groups, 31 types incl. CP / CN / CC copy candidates (PROVED, `randlayer.PAIRS`) | new kinds whose answer depends on a feature pair not listed get no unit | rule **pass**; schema **fail** |
| meaning RI codes (`meanlayer.MeanCodes`) | rule yes (distributional) | learns only from heard lines of others; never from eye, action or replay (PROVED, `mcore` docstring) | class from verbal evidence: likely pass (module 10/10, CITED); class from visual or action evidence: fail | **pass** in form; too narrow in input |
| DG meaning cue, `hipnorm` = D/(k_m·V) (`mcore.MeanCore`) | yes, if restated as "each input group to the DG has equal drive" (divisive normalisation; Carandini & Heeger 2012) | written as a special case | — | **pass** after restating |
| R6 critic / actor + dips δ = −V (v2.1) | yes (Schultz et al. 1997) | dips are heavy on check (held-out 0.80 → 0.43, CITED) | — | **pass** |
| NoGo veto `val_veto` | yes (Frank 2005) if one gate type serves every action | — | — | **pass** |
| R8 P_oth / V_go predictive turn learners | yes | priors (b = 0 inside τ ≤ 7) fitted to the staged waits | jitter breaks the prior | **pass** (rule); prior **fail** |
| R11 co-holding links | yes | — | — | **pass** |
| RAL cancel / commit (`core_v21.RoleCore`: `ral_W` 12, `ral_W_other` 6, reward contingency) | trace + broadcast yes (Frémaux & Gerstner 2016); **the cancel rules no** | `quick = now − t_end <= ral_W_other` (l. 120), against `gap_model` 3 (PROVED); it already failed on life corrections, an unanticipated turn structure (79–87 % cancelled, CITED) | with `gap_model` ≥ 7 the question is no longer cancelled, so role confusion and drift return toward ARCH_v2 (0.14 / 0.54); count-on and other new turn structures are misassigned | **fail** |
| v2.2 reply veto family (`veto_R1`, `veto_W` 40, `veto_rel`, `veto_gate`, …) | no | interim: drift loss900 −0.02 → 0.45 (`vW`, `fin`); after 3,000 untutored dialogues accuracy 0.008 (`fin`) (CITED `out/agg_steps.txt`, in progress) | moves the failure to a different world | **fail** |
| say_rec / say_loc copy routes, gateR (v2.1 Step 2) | no: "say the attended word / the slot's location word" is the what / where answer type | gateR SI 0.93 (PROVED arithmetic) | no route for new answer sources (count, order, missing) | **fail** |
| COPY_prompt / COPY_next / COPY_comp gates (`core.Cfg` booleans) | no: three hand-picked answer sources | prompt gate SI 0.62 | each new source would need a new gate | **fail**; replace with "any held content can be gated to output" |
| thought steps in a fixed order: recall → comparator → VR | no | ARCH_v2 §5 #10, §12 #2 | no new operation sequences | **fail** |
| PRIME / attD taught only by pointing, slot-relative | outstar yes (Grossberg 1974) | the 3-slot frame is the bench's | world2 and fovea need a different frame | rule **pass**; frame **fail** |
| comparator constants (`match_thr`, `conf_min` 6 units, `rate_max`, κ, …) | — | tuned on the bench, plateau with cliffs; `conf_min` is an absolute count | breaks scaling and the eye | **fail** until fractional and frozen |
| `soc_feats` (someone spoke ≤ 300 ticks; appeared or disappeared in silence ≤ 60) | no | metric-shaped (life's arrive schedule ↔ the empty-room-你好 metric) | — | **fail**; replace with coarse innate person-present / addressed detectors (IDEAS #17), budgeted |
| reflex 7 (10 ticks), transition window τ ≤ 7 | no | fitted to `scene_lead` 5 / staged waits ("PROVED from world.py", CITED) | overlaps can appear when timing is jittered | **fail** as written; innate "don't talk over" stays |
| `h_gate`, `h_denoise` (`hear.SoundGate`) | yes, as input adaptation (Dean et al. 2005) if area-general | ear-only thresholds (7ν, 4ν, 600 ticks) | — | **pass** after restating as area-general |
| `h_relock` (syllable clock) | innate prior (Giraud & Poeppel 2012) | — | — | **innate** (budget) |
| `h_level` (retry 1–3 levels louder) | no | "World2's distance subtracts levels, so this undoes distance" (CITED) | — | **fail**; divisive normalisation should give it |
| `h_vbound` (a second rhythm = a new voice) | no | "In world2 every speaker talks at the same rate" (CITED) | breaks when rates differ | **fail** |
| `h_conf`, `h_nocreate` | yes, if one rule "recruit only under confident novelty" covers Z, MIX and the hippocampus | ear-only | — | **pass** after unifying |
| babbling through `world.voice` | — | privileged read (PROVED) | blocked by isolation | **fail**; use body babbling with imitation |
| eye `retino` layout; count / colour codes; world2 fovea class bits | world gifts | count answers have never been computed (PROVED) | count by instances fails: attention focuses one slot, with no aggregation route | count **fail**; retinotopy as a budgeted innate prior **pass** |
| body / turn units, efference copy own-voice tag, joint-attention orienting, time cells, prosocial reward | innate (ARCH_v2 §5 #1, 4, 9, 16; IDEAS #17) | — | — | **innate** (budget) |

**Predicted v2.2 behaviour on the S0 kinds** (ASSUMED, reasoned from code):

| family | prediction | reason |
|---|---|---|
| class (verbal evidence) | pass | — |
| count by instances | ≈ 0 | — |
| order | ≈ 0 | 'before' ≈ 0–0.05, CITED |
| missing | ≈ 0 | — |
| count-on | seen pairs pass, held-out fail | the within-line keys (L, N) differ from the cross-turn keys (P, N) |
| repair | twin Δ ≈ 0 | no part learns from masked lines; restoration was rejected by construction, CITED meaning §1 |
| world2 motor families | 0 | v2.x has no motor learning |

**Prediction for S1-B:**
- families that need a new answer source or a new turn structure: at the trivial level;
- families that are new content in an old kind: pass.

## 6. Baselines

- **Already in the bench** (run through the same world, mouth and metrics):
  - Bench: Silent, echo, last, prior, lookup, n-gram, oracle (`baselines.py`).
  - Life: `RandomNamer`, `life_baseline`.
  - world2: Silent, RandomActor, Echo, Lookup, Namer, Oracle, Looker (`world2.agents`).
  - v2.1's same-run references: the best constant per task on the same items, and echo-or-constant in life.
- **New: kNN.** A privileged baseline that learns online from taught pairs. It answers with the taught reply of the nearest stored (prompt chars ∪ eye bits) by Jaccard. It is a lookup that generalises, so it catches families that are lookups in disguise.
- **New: RNN-BPTT** (the backprop baseline):
  - **Model:** a GRU with H 256, or about 1.5 M parameters; a matched variant with H about 1,024 sits near the core's plastic synapse count.
  - **Input per heard syllable:**
    - the char token;
    - a self / other bit (an efference copy, legitimate);
    - other versus other2 not marked;
    - a silence-bucket token at line ends;
    - a linear projection of the eye vector;
    - the reward.
  - **Training:** strictly online, one pass in life order, truncated BPTT over 64 tokens, Adam. It predicts the next token of *other* voices, and its own tokens are input only: the same "speech = the predicted next input" principle.
  - **Answering:** at every scored answer window, with the bench baselines' privileged timing (`bench_prompt`). It speaks the greedy continuation through a perfect articulator with v1 timing. The world's reactions (praise, recast) therefore depend on its own replies, as for the core.
  - **Variant:** an online + replay buffer (2,000 lines sampled at quiet moments), the analogue of awake recall.
  - **Freezing:** hyperparameters are tuned on S0 only and frozen under the same rules.
  - **Interpretation:** it has privileged perception and turn timing, so it is a *content* reference, not an embodied rival.
    - If it learns a family and the core does not, the core lacks something generic.
    - If neither learns a family, it is data-poor.
  - **Cost:** about 67k tokens per life × about 2.6 M MAC ≈ 170 G MAC, i.e. 1–3 min on CPU (PROVED arithmetic, ASSUMED throughput).
  - **Not included:** an RNN on the raw obs (it would also have to learn segmentation and articulation). That is future work.
- **Innate only:** the core with every rate at 0. It shows what is built in. For v2.x it equals Silent.
- **The module stack:** v2.2, frozen, in the same opening.

## 7. Feasibility: the minimal version now

**Build list** (a new `v3eval/` package; bench and world2 files untouched; about 1,800 lines, ASSUMED):

| file | what it does |
|---|---|
| `freeze.py` | hashes; the lints of §3.2 and §3.4; the scale lint; writes `FREEZE.json` and pushes it |
| `isolate.py` | runs the agent in a child process behind the audit hook; pickled frozen copies; checkpoint / resume across 90-min jobs |
| `families/` | spec schema, interpreter, accessor libraries for Life and world2, the S0 specs, the composer (its operator space stays undisclosed), the implementer's brief |
| `worlds.py` | `LifeF(Life)`: family topics, masking, timing jitter, neutral count codes, mouth-vocabulary extension, the `bench_line` hook. `World2F(World2)`: `_t_family`. |
| `baselines_x.py` | kNN and RNN-BPTT |
| `protocol.py` | the pre-registered run list (agents × seeds × grid × families × checkpoints; multiple-baseline schedule; repair twins) |
| `analysis.py` | G, Γ, curves, n₅₀, contrasts, bootstrap and Holm, the Hoffmann fit and s*, SI and necessity, the coverage table |

**Testing the test before any opening (red team):** the harness must catch three planted cheats, otherwise it is fixed first.
1. **A module adder:** v2.2 plus a hand-written counting module that reads the eye. Expected: S0 count passes, the mechanism count rises by 1, SI ≈ 1, and no S1 gain.
2. **A peeker** that imports `world`. Expected: blocked.
3. **v2.1's RAL under jitter.** Expected: the world-fitted rules break visibly.

**Compute** (`budget.py`):
- **Inputs (CITED, v2.2 interim):**
  - a 10⁶-tick life with the random layer: about 1,000 s on the A800 (`out/life/L_cf_s5.json`);
  - a staged run: 2,633 s, about 5.9 ms per speech-dense tick.
- **Frozen-copy probes:** 6 families × 200 × 90 ticks = 632 s, so base life + probes ≈ 1,630 s.

| item | jobs (4 × 5,400 s) |
|---|---|
| S0 validation (trivial references, kNN, RNN) | 0.1 |
| jitter test on v2.2 (staged, 3 seeds × 3 arms) | 1.1 |
| opening main run: v3 + v2.2, 5 seeds, life A (S1-A) and life B (S1-B) | 1.5 |
| repair control twins | 0.8 |
| scale 3×3 (s, L ∈ {0.5, 1, 2}), W = 1, 3 seeds | 2.4 |
| W = 2 at (1,1), (1,2), (2,2) | 1.3 |
| ablations, 6 mechanisms × 3 seeds | 1.4 |
| coverage (both agents) | 1.5 |
| **total** | **≈ 10.0 jobs ≈ 8 h wall at 2 slots** |

**Limits:**
- The longest single life (s = 2, L = 2) takes 5,264 s, so it needs checkpoint / resume.
- s = 4 and L = 4 are the full version (about 16 base lives per point), split across jobs.
- Peak RSS is about 0.7 GB at s = 1 (CITED), so about 1.4 GB at s = 2.

**Runnable now, without v3:**
- S0 validation and the RNN;
- the jitter falsification test on v2.2;
- v2.2 built by the global-*s* rule at 0.5 / 1 / 2: the first honest whole-model scaling check of the current stack.

## 8. Experiments that would falsify the claims

| claim | it is falsified if |
|---|---|
| The v3 core learns new kinds | Γ_v3 ≤ Γ_v2.2, or < ⅓ of S1-B families learned, or no family rises above its pre-introduction level |
| Fixed mechanisms + scale | E does not fall along the (s, L, W) diagonal, or α ≤ 0 |
| RAL is fitted to the world | J, H and drift are unchanged at `gap_model` 8 with polite caregivers (paired seeds) |
| Repair is learned | the twin contrast is ≤ 0 |
| Mechanism m is needed and generic | the ablation drop is within noise, or SI > 0.8 |
| The test is cheat-proof | any red-team cheat of §7 passes undetected |

## 9. What this removes from today's measurement

- **J as the decision objective.** Replaced by Γ on sealed families plus J2 for coverage.
- **Single-seed variant tables as evidence**, and one-module size sweeps called "scaling".
- **Counting and colour claims that rest on eye codes.**
- **Agent constructors that take `world`.**
- **Per-world parameter sets.**
- **Booleans as "parameters".**

## summary_zh

1. **为什么机制越加越多。** 以前每一种能力，都是先在测试上看到失败，再加一个机制，最后还在同一套测试上打分。这等于"拿考题练习"，所以永远证明不了"通用"。
2. **解决办法是先冻结代码，再出题。**
   - 题目分三层：
     - 公开练习题；
     - 题型公开、内容到开封时才随机生成；
     - 题型连设计者都没见过。第三层由开封时的随机组合，加上您（设计者）私下写的几件"小孩在家跟大人学的事"组成。
   - 每个冻结版本只开封一次，结果作数。
3. **什么算"加了机制"有可检查的规则。**
   - 改了核心代码、改了结构、加了开关或模式、给不同世界用不同参数，都算加机制；只改数值才算调参数。
   - 先天机制有固定名额。
   - 模型放在单独进程里，只能看到感官输入。
   - 开封时大人说话的时间常数随机改变，专门贴合测试台节奏的规则会露馅。
4. **测量四件事。**
   - **新题能不能从经历中学会：** 在一生中分批引入新题，没引入的当对照。
   - **放大：** 整个模型按同一个倍数一起放大，同时加长寿命、丰富世界，看错误率是否随之下降。
   - **最少机制：** 每个机制拿掉都必须有损失，而且影响多种题目，只影响一种的就是变相的专用模块。
   - **已有能力不丢。**
5. **对比基线。** 测试台原有的傻办法，加一个"最近邻记忆"，再加一个用反向传播、经历相同的小循环网络。
6. **现状。** 学习规则本身大多通用。v2 以后加的部件大多是专为某题、某世界节奏或某个指标做的；v2.2 的中间结果已经在"修好一处、坏另一处"。
7. **可行性。** 最小版本约 10 个 A800 任务，约 8 小时。现在就可以先做两件事：用时间扰动检验角色规则，以及把 v2.2 整体放大。