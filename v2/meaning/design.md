# A learned meaning layer (Step 4) — Phase 1 design

Labels: PROVED (argument / arithmetic), CITED (paper or another agent's file), ASSUMED. Nothing here is SIMULATED; the
module test (Phase 2) checks the predictions in §5.

## 0. What has to change, and where

- Every core so far scores 0.25–0.50 on the class probe `gen_probe2.py` (CITED review2/learning, randmix §3.3, v21 §2.2).
  The probe: four frames 你要X吗→要X, 你吃X吗→吃X (foods), 你玩X吗→玩X (toys), X好不好吃→好吃 (food) / 不能吃 (toy);
  2 foods + 2 toys are never heard in the taste frame (heldstream 0) and are asked only of a frozen copy.
- Why identity codes cannot pass it (PROVED from the readout): GEN is NLMS over active conjunction units. For a held
  pair (taste, X) the active units are (a) frame units without X — trained on foods (→好) and toys (→不) alike, so
  they sit near 0.5 / 0.5, and (b) units containing X — trained only in X's other frames (→要, →吃 / 玩). Nothing
  carries "X is like the foods that were taught 好吃", so the answer is a coin flip: 0.25–0.5 observed.
- Even an ORACLE class bit failed in the learning reviewer's version (0.25–0.75, CITED). Correction after the attack
  (§6): the learned K bits there were already a perfect class code (top-8 of the raw counts = idxL(吃) for every food,
  idxL(玩) for every toy; K = Koracle on both seeds). So the code was not the bottleneck. §6 shows what is: the
  hippocampal recall route (SIMULATED).
- randmix §6 (CITED) gives the substrate: the random-at-birth layer takes each item as a sparse line pattern in every
  item bundle (P, P1, PL, L, CP, CN, CC, VR). If two items' patterns overlap, the layer's active units overlap, and
  whatever the readout learned on one transfers to the other. No new keys, no key explosion. So the meaning layer
  only has to supply **learned line patterns**; GEN, gates, critic, RAL and the body stay untouched.

## 1. Constraints every candidate must meet

| constraint | what it means here |
|---|---|
| similarity of use | items used in the same slots of the same frames get overlapping codes (paradigmatic: 苹 ~ 蕉 because both follow 吃), not items that merely co-occur (syntagmatic: 苹 ~ 吃) |
| local | a synapse sees its pre unit, its post unit, its own value/trace, at most one broadcast; normalisation inside one unit or one layer (k-WTA) |
| one life | learning from birth, never reset, never switched off; no sleep phase |
| stable readouts | the code must stop moving once an item is familiar, or readout weights trained on it decay |
| no collapse | codes must not converge to one pattern (then nothing is distinguishable) nor stay all-distinct (then nothing transfers) |
| identity kept | the readout must still produce 苹→果, not 苹→蕉; the copy routes must still copy the exact item |
| lesionable | removing the learned part must change behaviour |

## 2. Four candidate mechanisms and their attacks

Common notation. Meaning lines: a layer of D = 512 binary units (a lexical-semantic area). An item i is a Z category
(one syllable). n_i = how often i has been heard. A heard line ℓ = (i_1 … i_m) with edge tokens BOS / EOS.

### A. Context-indexed Hebbian codes (random indexing with a homeostatic set point) — "RI"

Rule (exact).
- Fixed at birth (random wiring, never learned): every item j projects to k_x = 6 random meaning lines as a LEFT
  neighbour (idxL_j) and to 6 others as a RIGHT neighbour (idxR_j); BOS / EOS have their own indices.
- Learned synapses M[i, ·] from item i's unit onto the meaning lines. When a heard line ends (the PFC-1 buffer still
  holds its order, so both neighbours are known), for every position p with item i = i_p:
  x = idxL(i_{p−1} or BOS) ∪ idxR(i_{p+1} or EOS)  (0/1 over D),
  M[i] ← M[i] + η_i (x − M[i]),  η_i = max(1 / n_i, η_min) — a running mean that matures to a slow EMA.
  (Instar: pre = item i active, post = context drive x_j, the synapse's own value. CITED Kanerva et al. 2000; Sahlgren 2006.)
- Homeostasis: each meaning line keeps its own base rate b_j ← b_j + (x_j − b_j)/τ_b over all item occurrences.
- Code: c_i = the k_m = 16 lines with the largest s_ij = M[i,j] / (b_j + ε), among lines with M[i,j] ≥ m_min
  (k-WTA among the lines one item drives; each line's threshold rises with its own base rate). Incumbent lines get a
  hysteresis factor (1 + h). This is a positive-PMI code (CITED Levy & Goldberg 2014: PPMI ≈ what skip-gram factorises).
- Codes are recomputed in batches (every U heard lines, only items whose M changed).

Why it gives the right similarity (PROVED for the probe's stream, arithmetic in §2.A′). Left and right indices are
separate, so two items share lines only if they have the same neighbour on the same side: 苹 and 蕉 share idxL(吃);
苹 and 吃 share nothing. The division by b_j removes contexts every item has (BOS, EOS, 吗).

Attacks.
- **Drift.** M is a running mean: after n_i occurrences one new occurrence moves it by ≤ 1/n_i. Codes change only
  when a challenger line beats an incumbent by the factor (1 + h). A context shift (new frames at stage C) moves the
  code by Δ ≈ f·n/(n0 + n) of the weight (f = new-context share, n0 = occurrences before the shift), so a familiar
  item's code drifts slowly and stops; η_min bounds the memory window to ≈ 1/η_min occurrences, so it stays plastic
  for life. Residual risk: stage C brings new contexts (要, 个, 好) into object codes after stage B trained readouts
  on them. Bounded by the mixed code (§3): at most the meaning half (16 of 32 lines) can ever move.
- **Collapse.** All codes equal would need every item to rank the same lines highest. A line shared by everything has
  s_ij ≈ M[i,j]/b_j ≈ 1 for all i, while a line specific to a group has s_ij ≫ 1 for that group (PROVED: if a context
  occurs only with group G, b_j = P(G)·M_G,j, so s = 1/P(G)). Specific contexts win; universal ones lose. Opposite
  failure (all distinct, no transfer): items with one obligatory partner (苹 always before 果) get that partner's 6
  lines first — word-specific, which is correct — and the remaining 10 from frame contexts shared with their class.
- **Locality.** Pre/post/own value; b_j is the line's own activity trace; k-WTA = inhibition inside the layer. The
  right neighbour is used at line end, while PFC-1 still holds the line (or with a one-item eligibility trace).
- **One life.** No reset, no phase; codes learn from the first heard line. Frozen probe copies learn nothing.
- **Fault tolerance.** An item misheard as another item of the same class keeps most of the meaning half, so the
  context the readout sees changes less (PROVED, §2.A′: within-class overlap ≈ 14/16 lines vs between-class ≈ 8/16). A
  substitution by a random syllable (another class) gets no help. Prediction: no gain on random substitutions.

### B. Competitive k-WTA over item × context — "CL"

Rule. Meaning units u receive plastic context synapses C[u, ·] from the context lines (x as in A) and plastic item
synapses W[i, u]. On each occurrence: drive_u = C[u]·x − θ_u, the k units with the largest drive win; winners move
C[u] ← C[u] + η(x − C[u]) (instar prototypes, CITED Rumelhart & Zipser 1985); W[i, u] ← W[i, u] + η_i(win_u − W[i, u])
(outstar: P(u wins | i)); θ_u rises with u's own win rate (homeostasis). Code c_i = top-k of W[i].

Attacks.
- **Drift: worse than A.** Two coupled plastic layers: an item's code moves when its own contexts change AND when the
  prototypes C[u] move under other items' contexts. An item not heard for a long time still drifts (its W points at
  units whose meaning has changed). Readouts built on c_i break silently.
- **Collapse: real.** Without θ_u one unit wins every context (classic dead-unit problem); with θ_u, units are forced
  to equal use, which splits frequent classes into arbitrary sub-clusters and merges rare ones.
- **Locality / one life:** fine. **Fault tolerance:** as A.
- **Gain over A:** learned abstraction (a unit can merge 吃- and 要-contexts). Not needed for the probe or the bench.
- Verdict: a backup if A's codes are too literal; not chosen.

### C. Bootstrapped temporal context (TCM-like) — "TC"

Rule. As A, but the context drive is the neighbours' **current learned codes** (plus their fixed indices with weight
α): x = α·idx(nbr) + (1 − α)·c(nbr) (CITED Howard & Kahana 2002).

Attacks.
- **Collapse: proved risk.** With α = 0 the update is c ← average of neighbours' c, i.e. power iteration of the
  co-occurrence graph's random-walk matrix; on a connected graph it converges to one pattern for every item (PROVED:
  the only fixed point of x = Px for a primitive stochastic P is the constant vector). α > 0 anchors it (personalised
  PageRank), but similarity then spreads along chains of co-occurrence.
- **Wrong kind of similarity.** Averaging neighbours' codes makes an item like its neighbours (syntagmatic: 苹 ~ 吃
  ~ 好), unless left/right roles are kept separate. With separate sides it reduces to A plus second-order smoothing.
- **Drift:** feedback loop (my code moves → my neighbours' targets move → …), so codes keep moving after maturity.
- Verdict: rejected for the main line; kept as a module-test variant to show the collapse/association effect.

### D. Codes from hippocampal NEXT statistics during awake recall — "HIP"

Rule. In each awake recall (quiet moment), a replayed episode e and its successor line n: every item of e moves its M
toward the indices of n's items (and n's items toward e's), same instar, rate η_rec.

Attacks.
- **Coverage.** Under v2.1's role-aware listening a question I answered has its NEXT link cut, and recall samples
  only stored, linked episodes (uniform over ≤ 10k). Contexts reach the code late and unevenly.
- **Self-context.** Answers copy the filler (你吃X吗 → 吃X), so X's NEXT-context contains X itself.
- **Information.** The cross-line context of a held pair is exactly the answer we want to predict; for held pairs it
  never exists. What remains (吃 vs 玩 in answers) is what A already gets within the line, earlier and online.
- **Locality / one life:** fine (recall is awake). **Drift:** as A but driven by replay sampling noise.
- Verdict: not chosen; a module-test variant.

### Not chosen: top-down restoration of a misheard item

Using the context drive x (which is, by construction of A, close to the code of the item expected at that position)
to fill in an unexpected item's code would give fault tolerance for free. Attack (PROVED by construction): from the
context alone a misheard item and a novel combination look the same. A toy in the taste frame has never been seen
there; restoration would pull its code towards the foods that were, so it would answer 好吃 — exactly the
class-dependent generalisation we want to keep. The right gating signal is acoustic confidence, but the bench's fault
probe substitutes clearly spoken syllables, so the ear cannot know. Restoration is therefore not part of the design.

### 2.A′ Expected codes on the probe stream (PROVED arithmetic; corrected by the attack)

Frames are drawn uniformly; per exchange the answer is heard with p ≈ 0.85. A seen food then occurs in offer : eat :
taste = 1 : 2 : 1, giving context weights (share of its occurrences) L要 0.28, L吃 0.56, R吗 0.46, R-EOS 0.39,
L-BOS 0.15, R好 0.15. Scores s = M/b (b = the line's base rate): s(L吃) = 7.4; s(L要) = s(R吗) = 3.70 (an exact tie:
both have P(i | c) = 1/12); R好 1.85; EOS 1.27; BOS 0.50. So R好 never enters a code: every filler gets its 6 L吃 (or
L玩) lines plus 10 of the 12 L要 / R吗 lines. Within a class ≈ 14.3 of 16 lines shared, between classes ≈ 8.3 of 16;
a held food shares ≈ 14.3 with seen foods and ≈ 8.3 with toys. The part that separates the classes is 6 of the 32
lines of an item (19 %), not half.

In the bench (CITED world.py templates) foods and toys share every frame except taste, so the learnable class is
"object" (vs colour words, count words, slot words, names); the first syllable's code is its partner syllable (6
lines) plus object-frame lines (要L, 个L, 是L). That is the class the bench's held-out tasks need (offer, taste,
what, where: "a new object in a known frame").

## 3. The chosen design: mixed sound + meaning codes, meaning learned by A

1. **Code of an item in every item bundle of the random layer** = its fixed sound code (16 of 512 sound lines; randmix's
   acoustic fly-hash of the Z template, fixed at first use) ∪ its learned meaning code (16 of 512 meaning lines, A).
   Each item bundle has 1,024 lines (512 sound + 512 meaning), so 32 active lines per item: the same density (1/32) as
   randmix's 16 of 512, hence the same eligibility statistics for the layer's units (fan 3, θ 1, k-WTA).
2. **Why mixed.**
   - Identity: the sound half alone separates every item, so word production (苹→果) and exact copying survive even
     if two items' meaning halves coincide.
   - Drift bound: at most half of an item's lines can ever change; a readout unit driven by sound lines never loses
     its input (PROVED: the sound half is fixed at first use).
   - Transfer: half the lines are shared within a class, so roughly half of an item's active units in a population
     are shared with its class mates (randmix's layer maps line overlap monotonically to unit overlap; CITED test_rl).
3. **Stability** (revised, §6.4). Running mean with maturation (η_i = max(1/n_i, 0.002)); base rates floored at the
   prior rate and slow (τ_b 20,000); an item starts from a fixed random placeholder meaning code and learning swaps
   at most 2 lines per recompute (every U = 20 heard lines), each only past a 1.2× hysteresis; learning swaps start
   at n_i ≥ 3.
3b. **The same meaning code enters the hippocampal DG cue** (revised, §6.1): bag of the line's meaning codes and the
   first item's meaning code, as two extra fixed random fan-in groups of the DG k-WTA.
4. **Where learning happens.** Every heard line of any other speaker (questions included: meaning does not depend on
   role, so RAL's cancel does not touch it). Not the own speech (no self-confirmation), not replay (keeps one source).
5. **What stays.** GEN, gates, critic, RAL, Steps 2–3, hippocampus (its DG cue keeps identities), the body.
6. **Lesion.** At test: meaning off (both routes), layer-only off, held food / toy codes swapped, codes shuffled,
   recall routes off (§6.5).

Parameter count of the new part: D, k_x, k_m, η_min, τ_b, h, U, m_min — fixed, not tuned per task.

## 4. Binding principles

| principle | how it holds |
|---|---|
| one unbroken life, learning always on | M learns from the first heard line to the end; frozen probe copies are copies |
| no sleep | updates at line ends; no offline phase |
| continuous thought | untouched |
| local rules | instar synapse M[i,j] (pre, post, own value); line base rate b_j (own trace); k-WTA inside the layer |
| no puppeting | untouched |
| 0 mechanics errors | measured in every run |
| every learned part reaches behaviour | lesions in §3.6 on the probe and on the staged runs |

## 5. Predictions and falsifiers (module test first)

| test | prediction | falsified if |
|---|---|---|
| class probe, oracle meaning half (all foods share one pattern, all toys another) | held ≥ 0.8 | < 0.8: the readout cannot use line overlap; stop and fix the substrate first |
| class probe, A (RI) on v2.1 + random layer, 5 seeds | held ≥ 0.8; seen ≥ 0.95 | held < 0.6 |
| same, meaning lesioned at test | held back to 0.25–0.5 | no drop: the meaning half is a passenger |
| code purity (filler NN class purity), A | ≥ 0.9 foods/toys in the probe | < 0.7 |
| TC (C) | lower purity or near-collapse (mean pairwise overlap rising) | — (diagnostic) |
| fault: one prompt item → random other item | drop ≈ unchanged (no gain) | — (honest null) |
| fault: filler → another filler of the same class | smaller drop than for a different-class filler | no difference |
| staged, seeds 0–4, paired vs v2.1 + random layer | held-out taste / offer up; F within the seed SD; drift unchanged | F falls by more than the seed SD |
| 4× layer with meaning | no gain beyond the seed SD in this small world (ASSUMED: the bench has 141 syllables and ~4k lines) | — |

## 6. Revision after the attack (independent reviewer) and the first module runs

The reviewer re-derived §2.A′, simulated the random layer with structured codes, and attacked drift and the test plan.
Accepted points and what changed:

1. **The code was never the bottleneck (reviewer, PROVED from the old records) — the recall route is (SIMULATED).**
   v2.1 + random layer + RI codes, seed 0, 900 dialogues: held 0.75 intact; **1.00 with the recall copy routes
   (copy_next, copy_comp) switched off at test**; 0.25 with the meaning half removed. Mechanism (PROVED from the
   core): a held prompt (X好不好吃, X never in taste) recalls the stored episode with the largest DG overlap. The DG cue
   is identity-only (bag, first, last), so that episode is any taste prompt of either class, and its NEXT line
   (好吃 or 不能吃) is offered by the COPY_next gate, which learned to trust recall in this frame (for seen items the
   recalled episode is the same prompt). So recall injects a coin-flip answer.
   **Change: the meaning code also enters the hippocampal DG cue** (two extra input groups: the union of the line's
   meaning codes and the first item's meaning code, fan 10, weight 1). A held prompt then recalls a same-class
   episode, whose answer is class-correct. Seed 0: held 1.00. This is the same layer feeding both the cortex
   (random layer) and the index (DG), as in the ventral stream → hippocampus (ASSUMED analogy).
2. **Arithmetic of §2.A′** corrected above (class-specific part = 6 of 32 lines).
3. **Line overlap → unit overlap** (reviewer simulation, CITED): only ~4 % of ~265 active units differ by class, but
   the NLMS readout puts its weight on them (held 1.00 with 6 class lines, 0.72 with 1). The probe now logs the
   class-only units.
4. **Drift is driven by the ranking of M/b, not by M.** Ranks flip suddenly when a context appears (stage B → C:
   L要, L个 outrank BOS / L是), and b (fast EMA) and M (slow running mean) had mismatched clocks, so new or vanished
   contexts made s spike. Changes:
   - b is floored at the prior rate of a context line (2·k_x / D = 0.023) and is slow (τ_b = 20,000 occurrences);
   - η_min 0.005 → 0.002;
   - **gradual code change**: an item starts from a fixed random placeholder code (16 lines, same density from the
     first sight; reviewer point 9) and at each recompute at most 2 lines are swapped, each only if the challenger
     scores > 1.2 × the weakest incumbent. So a stage switch moves a code by ≤ 2 lines per 20 heard lines.
5. **Lesions made specific** (reviewer point 7): (a) meaning off in both routes; (b) layer-only off; (c) **swap the
   meaning codes of the held foods and held toys** (density unchanged; prediction: held falls well below chance,
   because the answers flip); (d) recall routes off. The oracle is made realistic ("oracle2": 6 class lines + 10
   lines shared by all fillers). Ten seeds; held foods and toys reported separately; replies are near-deterministic
   per filler, so the pooled count is 4 fillers × seeds against a binomial 0.5.
6. **The staged bench cannot validate class meaning** (reviewer point 6, PROVED from world.py): bench taste is
   foods-only with the constant 好吃, and every filler of every task is an object, so no bench held-out pair needs a
   class. Staged gains can come from changed unit statistics. Staged controls therefore: (i) sound + random meaning
   half (rand2: code doubling) and (ii) an oracle of the bench's word classes (oracle2 over the agent's own
   categories of object / colour / count / slot / name syllables, by syllable position). Meaning is credited only for
   what (i) does not give; (ii) bounds what perfect class codes could add on this bench.
7. **Bench lexicon** (reviewer point 8): colour / count / slot syllables share their partner syllable (色, 个, 边), so
   they get near-identical meaning halves; 小 has ~11 partners; 狗/猫/鸟 share idxL(小) with names. Watch the name and
   colour tasks.
8. **Locality** (reviewer point 10): the line-end update equals the online form with a one-item trace (on hearing
   i_p: M[i_p] learns idxL(i_{p−1}), M[i_{p−1}] learns idxR(i_p); EOS at the line end) up to the order of the two
   half-updates within an item; kept at line end for simplicity, stated here.
9. **The DG meaning cue must be scaled to the identity cue (found in the first staged runs; PROVED arithmetic).**
   An item is 1 of V identity lines in the DG's bag group but km of D meaning lines, so with equal weights the
   meaning groups' expected drive per DG unit is km·V/D times the identity bag's: 0.6× in the probe (V = 20), but 8×
   on the bench (V = 256), where meaning then dominated recall (recall of semantically similar but wrong episodes,
   e.g. 中间是什么 recalling 刚才中间是什么; staged J −0.004 over 5 seeds). Final rule (`hipnorm=1`): meaning-group
   weight = hipcue · D / (km · V), i.e. the meaning cue weighs as much as the identity cue (hipcue = 1) at any
   lexicon size. Probe with it: held 1.00 (5 seeds); staged J +0.054 over 5 seeds (report).

**Final configuration** = v2.1 final + randmix c_flat + `mean=ri mean_hipcue=1 mean_hipnorm=1` (D 512, k_x 6, k_m 16,
η_min 0.002, τ_b 20,000, floor 2k_x/D, h 0.2, 2 swaps per recompute, U 20, random placeholder start, n_min 3).
