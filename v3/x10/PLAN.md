# X10 plan: a strict, whole-brain-scaled build of the designer's architecture (s = 1, 4, 16)

Written 2026-11-02, before any model code. Approved by the coordinator 11-02 with three corrections (R3 decay order, M4 wording, H5), applied here. Authority: `docs/ARCHITECTURE.md` (2026-11-01). Where it was unclear I consulted, in order, `v3design/DESIGNER_SPEC.md`, `IDEAS.md`, `designer_msgs_clean.txt`, and then the X1 audit (`arch/v2/audit_x1/AUDIT.md`). Old repos were background only.

Labels: **MEASURED** = from the sizing scripts in `x10/sizing/` (seconds-long, no model). **DOC** = a quotation or rule from ARCHITECTURE.md. **CHOICE** = an open detail I fixed (registered in §3). **EST** = an estimate.

## 0b. Revision 11-02 (designer): the sound area is mostly feature units

This revision supersedes S2, S4, §2.1–2.4 and the s = 16 sizing below. The rest of the plan stands.

**Designer, via the coordinator.** Always-on sensory units have no features and send nothing. Units that are not always on basically all have features and can all send to the PFC. "按理说前额叶应该很大才对". The old form (dom1 plus one global θ per layer) left 92–96 % of A3/A4 dead or always on.

**New form** (`sizing/sound_stats2.py`, form K6; `sound.py`):
- **S2:** each unit takes K = 6 random local inputs from the layer below, weights U(0.5, 1.5). The radius widens with depth: ±1 / 2 / 4 / 8 channels and ±2 / 3s / 4s / 6s level positions, so deeper units cover more channels.
- **S4:** each unit's threshold comes from its **own** birth drive (ARCH §二.1 "按它自己收到的信号来定"). It is the (1 − r_l) quantile of that drive, with r_l = 0.20 / 0.12 / 0.08 / 0.05; the fraction rises with depth. The unit fires at "> quantile". If that never fires, it fires at "≥ quantile" when the resulting rate stays in the band (for example a silence-class unit).
- A0 keeps no chains; the band stays [0.002, 0.90].

**MEASURED, s = 1** (5,000 ticks, World "real" + TV, child silent):

| | A1 | A2 | A3 | A4 |
|---|---|---|---|---|
| units in the chain band | 77 % | 98 % | 100 % | 100 % |
| units with rate 1–30 % | 65 % | 92 % | 97 % | 97 % |
| median rate of chain units | 0.060 | 0.091 | 0.069 | 0.043 |
| layer fraction | 0.117 | 0.122 | 0.083 | 0.046 |

- Total chain units: 8,116 (s = 1); 32,460 (s = 4).
- Codes are much cleaner than with the old form:

| | J same syllable, other voice | J different syllables | J syllable vs silence |
|---|---|---|---|
| A2 | 0.69 | 0.18 | 0.15 |
| A4 | 0.86 | 0.04 | 0.01 |

- Old form, for comparison: A2 0.60 / 0.39 / 0.53.

**Consequence for msg 57 and check a1 (needs a decision).** Lit chain units:

| | silence | speech | ratio |
|---|---|---|---|
| s = 1 | 363 | 1,346 | 0.27 |
| s = 4 | 1,292 | 5,202 | 0.25 |

- Sparse feature units fire for what is rare. Silence is the commonest state (45 % of ticks), so few units can be both sparse and silence-specific. Wider fields made it worse (308 vs 1,560).
- The PFC input in silence is therefore about ¼ of speech: small but not empty (X1 had 0–2).
- Pre-registered a1 asked every layer's silence/speech ratio to be in [0.67, 1.5]; the sound layers and L1 now fail that by construction.
- **Proposal:** a1 keeps A0 exactly half on, PFC count preservation (L2/L1, L3/L2, L4/L3 in [0.67, 1.5] in silence and in speech), and no empty or saturated layer. Silence/speech ratios are reported without a bar.
- msg 57's "always about the same number active" then holds for the input layer only. This is risk R1′ for the designer.

**Sizes:** superseded by the PFC-size presets below.

**Later 11-02 clarifications** (designer, via the coordinator). These supersede the "≥ 80–90 %" target.

*Feature status is a parameter, per area, chosen by the experiment and never learned:*
- `band` / `band_sound` (also per layer) / `band_motor`, `sound_rates` (threshold fractions per depth) and `band_depth_sound` are all per-run knobs (`common.py:config`).
- Every run reports, per area and layer (`brain.py:Brain.make_chain_report`):
  - the share in band, never-on and always-on;
  - the counts among the chained units;
  - the copies sent to PFC L1 and to the hippocampus;
  - lit chain units in silence and in speech.

*The rate cut applies only near the input:*
- Default `band_depth_sound = 1`: the band acts on A1 only, and all of A2–A4 are chained. A deep always-on or never-on unit may reflect this narrow world, not a lack of features.
- Arm "band at every depth" = `band_depth_sound = 4`.
- Motor: force and stop neurons are all chained. The newborn never moves, so force neurons have birth rate 0, and the motor band's low cut defaults to 0. The tonic blocking neurons keep no chains, per §二.7; their rate (1.0) lies outside the band, which is reported.
- PFC L1 ≈ all non-shallow units: 8,195 at s = 1 (A1 in band + A2–A4 + 32 motor). That is ≈ the same as the table above, so the sizes and memory rows stand.

*Construction artifacts (register S2′):*
- Sound units take 6 inputs with continuous random weights and a threshold from their own drive. So no unit needs an impossible coincidence (such as both polarities of one thermometer pair); a unit's drive always varies when any input varies.
- Units dead or always on in the birth stream are reported as world-limited, not removed.

*(a) sweep* (`run_a.py --grid loop | band`):
- **loop grid:** inhibition P0/P1 × loop reciprocal/random × g_loop {0.25, 0.5, 1, 2} × w44 {0, 0.5, 1}. The gains are lower than before: the smoke run showed g_loop 4 drives the newborn into runaway through the return path (A4 86 % on, L1 63 %).
- **band grid:** band_depth {1, 2, 4} × band high cut {0.90, 0.99} × sound threshold fractions {dense, default, sparse}.
- The choice is made by the (a) checks, not by a target share. Stage (c) keeps 1–2 alternative bands as arms (B80 becomes one of them).

**PFC size is a hypothesis to test** (designer 11-02, softened: "你也不一定要把前额叶做的超级大"). Three registered presets come from the band knobs (`common.py:PFC_SIZE`; `sizing/band_sizes.py`). A uniform rate cut would remove A3–A4, whose target rates are 8 % and 5 %. So deeper layers are cut relative to their own target rate r_l, and every layer still sends chains:

| preset | rule | sound copies at s = 1 (A1 / A2 / A3 / A4) | lit silence / speech |
|---|---|---|---|
| small | every layer: rate ≥ 0.95 r_l | 1,804 (183 / 444 / 587 / 590) | 362 / 479 |
| **medium (default)** | A1: band [0.002, 0.90]; deeper: rate ≥ 0.9 r_l | 4,054 (1,670 / 635 / 882 / 867) | 362 / 831 |
| large | A1: band; every deeper unit | 8,150 | 363 / 1,346 |

- The smaller presets are closer to msg 57's constant activity: a high low cut keeps the silence-class units and drops the sparsest speech units.
- The (a) "size" grid crosses the three presets with threshold fractions {dense, default, sparse}, plus the old "band at every depth".
- The winner is chosen by the (a) checks; the runner-up becomes an arm in (c).

**Units and GPU tables** (`sizing/counts.py`; plastic tables as fp32 weight + int32 index; PFC learned links within radius 0.15):

| preset | s | PFC L1 | total units | synapses | tables GB (PFC learned + hippocampus) |
|---|---|---|---|---|---|
| small | 1 / 4 / 8 / 16 | 1,836 / 7,344 / 14,688 / 29,376 | 24k / 96k / 192k / 383k | 4M / 65M / 258M / 1.0G | 0.03 / 0.51 / 2.0 / 8.1 |
| medium | 1 / 4 / 8 / 16 | 4,086 / 16,344 / 32,688 / 65,376 | 35k / 141k / 282k / 563k | 7M / 97M / 381M / 1.5G | 0.05 / 0.75 / 3.0 / 12.0 |
| large | 1 / 4 / 8 / 16 | 8,182 / 32,728 / 65,456 / 130,912 | 56k / 223k / 445k / 891k | 14M / 212M / 839M / 3.3G | 0.10 / 1.66 / 6.6 / 26.6 |

- In every preset the PFC (4 × N1) grows with the rest of the brain, so its share does not shrink with s.
- At s = 16, small and medium fit; large is at the limit (26.6 GB of tables before working memory). The timing pilot measures this.

**G1 construction rule (registered).** The least-squares map's input matrix is chosen among three forms by the map's own quality on the forced codes (top-4 muscles exact, then margin):
- binary at p ≥ 0.5;
- binary at p ≥ 0.8;
- the firing probabilities.

All three reach top-4 exact 1.00, with margins 3.4–4.3. Codes keep shared units: X1f's specificity margin left 9 syllables with no A1 code.

**Timing pilot (MEASURED, A100, 2,000 ticks of overhearing with learning on, one seed per process, default loop g_loop 1 / w44 0.5; `pilot/timing_*.json`):**

| s | small | medium (default) | large | GPU peak (small / medium / large) |
|---|---|---|---|---|
| 1 | 8.5 ms/tick (10k dialogues ≈ 58 min) | 9.5 (≈ 64 min) | 10.8 (≈ 73 min) | 0.15 / 0.18 / 0.42 GB |
| 4 | 18.8 (≈ 127 min) | 36.8 (≈ 249 min); 29.9 with g_loop 0.25, w44 0 | 52.3 (≈ 354 min) | 1.0 / 1.3 / 2.7 GB |
| 8 | – | 107 (≈ 12 h) | 181 (≈ 20 h) | – / 4.5 / 10.1 GB |
| 16 | 165 (≈ 19 h) | 401 (≈ 45 h) | out of memory at 34 GB | 10.0 / 17.7 / – |

- The PFC (drive + learning) is 80–90 % of a tick at s ≥ 4.
- Four s = 1 processes on one GPU raise throughput only 1.24×.
- The sync-free "scan whole tables" path was slower (scatter-add contention), so the default is the sparse path.

**What fits:**
- Inside 90-min jobs today: s = 1 at every size, and s = 4 small at about 3k dialogues per job.
- s = 8 and s = 16 need an optimisation round, and a runner that keeps a checkpoint between jobs (each job folder is wiped).
- Memory alone allows s = 16 small or medium and s = 8 large.

**Activity inflation (a finding for stage a).** In every pilot run activity rose well above the birth calibration within 2,000 learning ticks:
- L4 ≈ 1.9–2.5 × L3;
- Hs 670–5,300 instead of ≈ 50;
- at s = 4 medium with the loop weak and no birth lateral, Hs was still 1,904 and L4/L3 2.5.

So the learned lateral and hippocampal links inflate activity quickly (risk R9). This also inflates the timings.

**Speech basics (smoke only, not stage b).**
- b3 (own-voice loop 0), b4 (echo 0) and b6 (brake 1.00) pass.
- b1 forced syllable said 0.27, with garble 0.84, even with the loop off. The G1 map itself is exact offline (top-4 exact 1.00, margin 3.7), so the break is in the online motor competition or in background A1 drive. Stage (b) starts by locating it (d5).

**Count preservation, form fixed by the smoke run (P3/P4 revised).**
- With the patch neuron at gain 1, a layer's count stays almost fixed whatever the input. Online L2/L1 was 0.40 when L1 grew. With no patch neuron, the count grows faster than the input.
- So each layer's fixed threshold θ_l **and** the patch neuron's gain g_l (from {0, 0.25, 0.5, 0.75, 1}) are now chosen once at birth so that the layer's count follows the layer below tick by tick (least squares over the birth stream). P0 keeps g = 0.
- These are still a fixed threshold plus §四.10's adopted patch neuron; only their values are set by a birth rule.

**Check a1 (proposal, needs the coordinator's decision).**
- Keep: A0 exactly half on; PFC L2/L1, L3/L2, L4/L3 in [0.67, 1.5] in silence and in speech; no empty or saturated layer.
- Report per-layer silence/speech ratios without a bar, because feature units make silence light about ¼ as many sound copies as speech.

## 0. What this build changes, in one paragraph

Every X1 model (X1c–X1g) fed the PFC straight from a one-level on/off copy of the world's 180 abstract sound bits. Only the "on" half had chains. So in silence the PFC received 0–2 lines and died within 3 ticks; the thought-holding loop never closed (audit item 1). X10 instead builds:
- the sound area §二.1–2 describes: a graded thermometer input that is always exactly half on, and four 3-D local layers that get sparser with depth;
- chains from every feature layer, in a firing-rate band that includes the units lit in silence. **MEASURED:** 304 lit chain copies in silence vs 266 ± 27 in speech at s = 1, and 1,156 vs 1,198 ± 55 at s = 4;
- a count-preserving four-layer PFC whose layer sizes equal its input layer;
- a hippocampus;
- a motor area with no hard gates;
- one scale knob s that multiplies every area.

All other mechanisms follow ARCHITECTURE.md and its §四 results. Nothing in §三 is used.

---

## 1. Conformance table (§二.1–§二.12)

"Audit #" refers to the eight X1 deviations listed by the coordinator (2026-11-02) and to AUDIT.md rows.

| § | clause (short) | applies? | X10 implementation | X1 deviation avoided how |
|---|---|---|---|---|
| 2.1 | binary neurons | yes | every unit in every area is 0/1 each tick | (was faithful) |
| 2.1 | inhibitory = ordinary neuron with negative outputs; keep the "held down" role | yes | motor blocking and stop neurons; one non-learning inhibitory neuron per PFC/hippocampal patch (§3 P4, P5); no other inhibitors | patch inhibition is now an explicit neuron, not a subtraction hidden in the threshold code |
| 2.1 | distributed codes, no item neurons | yes | no syllable ID anywhere in the brain; the G1 map and the probes read populations | (was faithful) |
| 2.1 | thermometer on/off pairs; input always exactly half on; nothing heard is also a reading | yes | 180 cochlear channels with **graded** energy (§3 W1); each channel has 3s levels, each level is an on/off pair; asserted half on every tick | X1 had 1 level per binary world bit, so silence was the "off" half; that half had no chains and never reached the PFC (audit 1) |
| 2.1 | deeper sensory layers sparser; per-neuron threshold = fraction of own in-weight sum, rising with depth; **sensory only, not PFC** | yes | sound layers A1–A4 use θ_l·Σw_in with θ non-decreasing in depth (§3 S4). PFC layers use a **fixed** per-layer threshold that does not track in-weights (§3 P3) | X1 used the in-weight-fraction threshold in PFC L2/L3, with homeostasis on top (audit 2) |
| 2.1 | signal strength = number lit (e.g. white light vs black) | partly | the thermometer encodes loudness (TV at 0.4 lights fewer levels than a near voice). The designer's route for "some signals light more" is split/delete (§二.9, msg 69), which is off here | – |
| 2.1 | own vs other voice: similar but different; no labels | yes | physics: same content channels; own voice has its own 6 timbre channels and reaches the ear at content level 0.6 (bone conduction, §3 W2). **MEASURED:** codes of the same syllable in two voices overlap more than different syllables (J 0.60 vs 0.39 at A2) | – |
| 2.2 | 3-D local sparse random net; each unit links to a few nearby units in the layer before (y, z distance only) | yes | A0 sheet = channel (y) × level/polarity (z). A1–A4: local3d's form, 1 dominant input at the same (y, z) plus 3 weak neighbours within \|Δy\| ≤ 1, \|Δz\| ≤ 2; fan-in 4 at every s ("几个") | X1 had no layer beyond the input (audit 7) |
| 2.2 | near-input layers wired at birth, then basically fixed | yes | A1–A4 are all fixed birth wiring (all four count as "near input") | – |
| 2.2 | similar sounds → similar units | yes | checked in (a2); **MEASURED** overlaps above | – |
| 2.2 | vision, fovea, visual motion units | no | not needed: the dialogue world has no picture, and answers depend only on sound | – |
| 2.2 | concept area (own population, local random birth links, same rule, chain pairs to PFC and hippocampus) | arm only | built (sizes §2) but off in the main arms, because §四.14 says to add it only after the PFC answers seen pairs. Arm C+ runs in stage (c2) | was deferred without a plan; now has sizes and a staged arm (audit 7) |
| 2.3 | every layer of every area sends 1:1 chains to the PFC input layer and to the hippocampus | yes | chains from every feature unit of A1–A4, every muscle force neuron and every stop neuron (and concept units in C+), each to one PFC L1 copy and one hippocampal copy | X1 chained only the input layer's "on" half and had no hippocampus (audit 1, 7) |
| 2.3 | each forward chain has a paired return chain to the **same** S | yes | 1:1 for every copy: a sound copy → its sound unit, a muscle copy → its force neuron, a stop copy → its stop neuron. The input layer A0 has neither forward nor return chains (§3 C2) | X1 content copies re-lit no sound neuron, stop copies acted through a mean, and voice copies' returns went nowhere (audit 3) |
| 2.3 | no chain for units on "绝大部分时间" or almost never | yes | chain iff birth firing rate is in [0.002, 0.90] (§3 S5). **MEASURED:** the band keeps PFC input roughly constant; ≤ 0.60 would give 0 in silence | X1 cut at 0.6 and gave never-on units chains |
| 2.3 | chain count follows split/delete | no | not needed: §二.9 is off (slow time scale; §四.5 says it only helps when capacity is short) | – |
| 2.4 | multi-layer recurrent net: input layer L1; **middle layers** (plural) random local; last layer | yes | L1 copies → L2 → L3 (two fixed random local middle layers) → L4 (last) | X1 had one middle layer |
| 2.4 | middle layers learn by split/delete, not weights (§四.11) | yes | L2/L3 weights fixed; split/delete off (as above) | (was faithful) |
| 2.4 | last layer(t) → L1 input(t+1) by Hebbian links, only to nearby L1 units | yes | acquired L4 → L1 links to every L1 copy within radius 0.15 (candidate count ∝ s); post event = onset of the copy's chain-driven input (§3 P7) | (form was faithful; locality kept) |
| 2.4 | loop holds a thought: last layer can re-light L1 | yes | birth L4 → L1 loop (reciprocal, msg 84; §3 P6) plus the learned links. An internally lit L1 copy counts **fully** in L2's input. Closure is a pass criterion (a3) | X1's loop never closed: internal lines counted half (g_int 0.5), an L2 unit needed ≈ 7 lit lines and only 0–2 were lit (audit 1) |
| 2.4 | each PFC layer passes on about as many as it receives | yes | L2, L3, L4 thresholds fixed and calibrated once at birth so count(L_{l+1}) ≈ count(L_l); same layer size as L1 (§3 P2). Checked in (a1) in silence and in speech | X1: L1 ≈ 25 → L2 52–70 → L3 57–81 in speech, and 0 → 0 → 0 in silence (audit 1) |
| 2.4 | maybe no PFC inhibitory neurons; add only on runaway / extinction / scattering (§四.10 order) | yes | (a) tests P0 (none) first. Default P1 = §四.10's adopted minimal form (one non-learning inhibitory neuron per ~33-unit patch, same tick). Decision rule in §6 | X1 stacked patch-mean, homeostasis and a max-relative release; X10 has at most the one adopted form |
| 2.4 | motor-area inhibitory neurons kept | yes | blocking and stop neurons (2.7) | – |
| 2.4 | dense mutual excitation; thinking range; two-layer weights split by out **and** in totals | yes | acquired L4 ↔ L4 and L4 → L1: propagation = g·w/√(out·in), with floors (§3 P9) | X1 divided by out only (a = 1), so every top unit pushed a fixed amount; lateral links saturated to 95–100 % of pairs (audit 4) |
| 2.4 | thinking range size (20–30, or ∝ size: §四.20 open) | yes | follows count preservation: ≈ 20 % of L4, i.e. ∝ s (§3 P11). The fixed-count alternative is risk R3 | X1 fixed 30 as a long-run average; per tick it was 0 in silence and 50–170 in speech |
| 2.4 | rarely co-active units inhibit each other; related units excite each other | yes | effective lateral weight = learned excitation (≥ 0) − the patch inhibitory neuron's share. Related pairs are net positive and rarely co-active pairs net negative, which is the designer's msg-27 method 1 ("联系强的正、不强的负"). Checked by diagnostic d7; partner form = risk R8 | X1: partners off, lateral floored at −birth, and saturated, so there was no net inhibition (audit 7) |
| 2.4 | thinking moves slowly; continuous; no resets; excitability = context length | yes | no resets anywhere (probes restore only neuron states and freeze learning). Knob = birth loop gain and lateral gain relative to chain input (HARVEST 12) | – |
| 2.4 | speech is caused by thought; returning to a topic is a feature | yes | the only route to the mouth is the motor competition fed by inner hearing (through G1) and motor returns; no timers | – |
| 2.5 | PFC output = lighting L1 copies; 1:1 returns re-light S: inner hearing, motor intention | yes | an internally lit copy (driven by L4 or by hippocampal copies) sends its return; the return adds g_ret to S's drive (§3 C1) | X1 inner hearing was read straight from L1 into a motor map |
| 2.5 | return strength tunable; return to the same item weaker than the link to the next item | yes | g_ret is the knob; birth loop per-link strength < learned prediction strength at equilibrium (checked after learning, d3) | – |
| 2.5 | prediction biases hearing (learned) | yes | return drive adds to S's feed-forward drive, so a predicted unit fires on partial input; diagnostic probe with mumbled syllables | X1 had none (audit 7) |
| 2.6 | own population; chains in; fast-growing, fast-decaying Hebbian links; short-term only | yes | hippocampal copies H1 (1:1 with every chain source) → own population Hs (fixed random recode, X3) → learned Hs ↔ Hs and Hs → H1; the fastest decay in the brain | X1 hippocampus off (audit 7) |
| 2.6 | recall lights the areas' units via its own return chains, never the PFC directly | yes | H1 copies lit by Hs → their returns → S → S's normal forward chains reach PFC L1 | X1c read and recalled PFC L1 directly |
| 2.6 | stores everything; concave growth; similar cues → similar recall; no time ring | yes | same rule everywhere; no records, no per-memory cells | – |
| 2.6 | fast memory comes from fast weights, not from internal random links (designer 11-01, X6b) | yes | no birth Hs ↔ Hs recurrence; only the recode fan-in (needed per §五 X3) | – |
| 2.7 | output only through muscles; blocking wins by default; more force neurons → more force | yes | per muscle, s force neurons with graded thresholds; the body reads a muscle as on when ≥ half of its force neurons fire. Half of the blocking neurons are tonic | – |
| 2.7 | speaking is a competition, **no hard threshold**; unsure → silent | yes | each force neuron: G1 drive + motor return − b·(blocking neurons on) − c·(stop neurons on) ≥ θ_m. The non-tonic blocking neurons are recruited by the pooled muscle drive (graded ladder), so a diffuse command recruits more blocking than a sharp one | X1's A2 "3–5 muscles" count gate is removed (audit 6) |
| 2.7 | blocking neurons always on, no chains; stop neurons phasic, with chains | yes | as stated | (was faithful) |
| 2.7 | innate brake reads only the input layer (loudness of nearby voices); returns cannot reach it; stops our speech at the next syllable | yes | birth reflex: the count of lit up-lines in the other voices' A0 channels → stop neurons. G1 is fed from sound layer A1 so that echo and brake have equal latency (§3 M5) | (was faithful) |
| 2.7 | think in sound, speak with muscles; sound → muscle hand-wired in generation 0 (X7) | yes | gen-0 set G1 (§4), in the **plastic, decaying** part | X1's A1/A2 were decay-exempt (audit 6) |
| 2.7 | babbling; within-syllable order; representation collapse; motor association area (§四.15) | no | not needed: overhearing has no babbling, and the X1c body has one muscle set per syllable, with no order or force sequence inside it. Multi-syllable order comes from PFC predictions | – |
| 2.7 | stop after speaking is learned; PFC "stop" via stop copies | yes | stop copies are PFC inputs. The PFC can learn to predict them (they fire under the brake whenever adults talk); no hand-wiring | – |
| 2.8 | held-form temporal Hebbian; co-rising strengthen; fall does nothing | yes | exactly §二.8: A on, just risen, held, or fell ≤ 2 ticks ago when B rises → A→B +. Co-rising → both +. A re-rises while B is on or B fell ≤ 2 ticks ago → A→B − | X1 weakened only while B was still on (minor audit row) |
| 2.8 | always-on shallow sensory units grow no links | yes | they have no chains, so they reach no learned projection | – |
| 2.8 | concave growth, no cap, ≥ 1 % plasticity; weakening and decay slowed by the same factor | yes | c(w) = max(0.01, 1/(1 + w/w_ref)) multiplies growth, weakening and decay | X1 decay was not concave (audit 5) |
| 2.8 | one global decay of acquired links; birth links exempt; deeper = slower; hippocampus fastest | yes | every 200 ticks: hippocampus γ_H = 1 − 0.01/s (fastest), G1 and other motor links 0.9999, PFC L4 → L1 and L4 ↔ L4 0.99995 (slowest). Gen-0 hand wiring is acquired, not birth | X1 decayed the top lateral links fastest in the brain (×0.95 per 100 ticks) (audit 5) |
| 2.8 | time = order; no timers | yes | no timer, refractoriness, fatigue or self-hold anywhere | – |
| 2.8 | reward = tool from innate value | no | not needed: §二.11's method learns by overhearing, where nobody reacts to the child. r = 0 throughout; probes are frozen | – |
| 2.8 | two correction mechanisms (hard-to-change strong links + fast links) | partly | both present (concave rule; hippocampus), but correction is not tested here | – |
| 2.9 | split/delete | no | not needed (above) | – |
| 2.10 | instinct, amygdala | no | not needed: no values in this test; form open (§四.16) | – |
| 2.11 | always alive, no sleep, no reset; children may err | yes | one continuous life per seed; errors counted, not punished | – |
| 2.11 | world teaches; others' and own lines kept apart; no echo training | yes | X1f OverhearWorld (4 adult voices) for training; X1c World (real caregiver) for the live block | – |
| 2.11 | systematic method: 1 sound↔muscle wired; 2 overhear many dialogues; 3 thought → inner hearing → mouth | yes | stages (b) and (c) are exactly steps 1–3 | – |
| 2.12 | gen-0 hand wiring allowed; registered; in the decaying part; test learning only on never-wired content | yes | §4. Main arms wire only G1 (speech basics) | X1 kept hand wiring decay-exempt |
| 2.12 | evolution, fusion, inheritance | no | not needed: one generation | – |
| 2.12 | scale the whole brain together | yes | one knob s multiplies every area (§2) | X1 scaled only PFC L2/L3 with fixed fan-ins, so drive density fell (audit 8) |
| 2.12 | pure neurons; dev equivalents registered | yes | registered equivalents: sparse tables, the "return fires on internal drive" neuron (§3 C1), patch inhibitory neurons computed as means | – |
| 2.12 | sealed new questions | yes | the 10 held-out pairs per seed are never spoken; code is frozen before stage (c) | – |

**No symbolic input reaches the brain.** Each tick the brain receives only the A0 bit vector (1,080·s bits). Reward is 0. World labels are used only by the scorer. Diagnostic hooks (forced inner hearing, oracle paths) exist only in arms labelled DIAGNOSTIC.

**The coordinator's eight audit items, and where X10 avoids each:**

| # | X1 deviation | X10 | checked by |
|---|---|---|---|
| 1 | loop never closed; silence fed 0–2 lines; not count-preserving | deep sound layers + chain band ≤ 0.90 (silence feeds ≈ 23 % of L1, MEASURED); internal lines count fully; thresholds calibrated so that L4 ≈ L3 ≈ L2 ≈ L1 in count | a1, a3 |
| 2 | sensory in-weight-fraction threshold in PFC; homeostasis; stacked patch-mean | fixed per-layer PFC thresholds; no homeostasis; at most §四.10's one adopted inhibitory neuron per patch, chosen by the (a) rule | a1 |
| 3 | returns not 1:1 to S | every copy returns 1:1 to its own sound, force or stop neuron; inner hearing really re-lights sound units | b1, d4 |
| 4 | two-layer weights by out only (a = 1); lateral saturation | divided by √(out·in) with floors; d6 watches the lateral positive-pair fraction | d6 |
| 5 | top lateral decay fastest and not concave | one concave-slowed decay; PFC slowest, hippocampus fastest | code check before (c) |
| 6 | hand-wired speech links decay-exempt; hard 3–5 count gate | G1 in the decaying part; the gate replaced by blocking-neuron competition | b1–b7, d9 |
| 7 | missing: sensory layers, hippocampus, mutual inhibition, prediction biasing hearing, concept area | all built; concept area staged as arm C+ per §四.14 | a1, d7, d8, (c2) |
| 8 | only PFC L2/L3 scaled; fixed fan-ins thinned the drive | every area × s; constant active fraction keeps each unit's active-input count constant; learned fan-ins ∝ s | §2, (c) scale comparison |

---

## 2. Areas and sizes, with one scale knob s

**Rule.** Every count is multiplied by s:
- sensory resolution (thermometer levels per channel);
- units per layer;
- force, blocking and stop neurons;
- PFC layers (= number of chain copies);
- the hippocampal population and its learned fan-outs (X6);
- the concept area.

Fan-ins of fixed birth projections stay constant. The sound area's fan-in is "几个" by DOC, and every layer's active fraction is constant across s, so each unit sees the same expected number of active inputs. Fan-ins of **learned** projections grow ∝ s: every unit within a fixed radius is a potential synapse, so connection density within a neighbourhood stays constant. This addresses audit 8: drive per unit no longer falls as s grows, and associative capacity grows with s (R2).

### 2.1 Areas

| area | layer | s = 1 | s = 4 | s = 16 | form |
|---|---|---|---|---|---|
| **sound** | A0 input | 1,080 | 4,320 | 17,280 | 180 channels × 3s levels × on/off pair; exactly half on |
| | A1–A4 | 4 × 2,160 | 4 × 8,640 | 4 × 34,560 | 2 units per sheet position; 1 dominant + 3 weak local inputs; θ_l·Σw, non-decreasing with depth |
| | feature units with chains (MEASURED s = 1, 4; EST s = 16) | 1,298 | 6,151 | ≈ 29,000 | birth rate in [0.002, 0.90] |
| **motor** | muscle force neurons | 24 | 96 | 384 | s per muscle, graded thresholds |
| | blocking (tonic + recruited) | 8 | 32 | 128 | no chains |
| | stop | 8 | 32 | 128 | chains |
| **PFC** | L1 copies = N1 | 1,330 | 6,279 | ≈ 29,700 | one per chain source (sound + force + stop) |
| | L2, L3 (middle), L4 (last) | 3 × 1,330 | 3 × 6,279 | 3 × 29,700 | same size as L1; fan-in 24 within r 0.1; ≈ 33-unit patches |
| **hippocampus** | copies H1 | 1,322 | 6,247 | ≈ 29,500 | one per chain source |
| | own population Hs | 5,000 | 20,000 | 80,000 | ≈ 50 active (fixed count, X6); recode fan-in 24 |
| **concept** (arm C+) | C1, C2 | 2 × 1,000 | 2 × 4,000 | 2 × 16,000 | X9 two-layer form; ≈ 6 % on |
| **total units** (main, no concept) | | ≈ 21,400 | ≈ 90,400 | ≈ 384,000 | |

### 2.2 Synapses (`sizing/counts.py`)

| | s = 1 | s = 4 | s = 16 |
|---|---|---|---|
| sound (fixed) | 0.03 M | 0.14 M | 0.55 M |
| chains (forward + return, PFC + hippocampus) | 5.3 k | 25 k | 0.12 M |
| PFC birth: feed-forward 24/unit, loop 30/unit, lateral 20/unit | 0.16 M | 0.77 M | 3.6 M |
| PFC learned, potential (L4 → L1 and L4 ↔ L4, radius 0.15; ≈ 94·s^1.1 candidates per unit) | 0.25 M | 5.6 M | 124 M |
| hippocampus: recode 24 + learned Hs ↔ Hs (500·s) + Hs → H1 (200·s) | 3.6 M | 56 M | 898 M |
| motor (G1 ≈ 4 targets per layer-A1 chain unit; brake; blocking) | ≈ 3 k | ≈ 13 k | ≈ 0.1 M |
| **total** | **≈ 4 M** | **≈ 63 M** | **≈ 1.03 G** |
| plastic tables, GPU (fp32 weight + int32 index) | 0.03 GB | 0.5 GB | 8.2 GB |

### 2.3 Why these proportions follow the document

- **PFC L1 = number of chain copies:** §二.3–4 (one copy per chain source).
- **L2 = L3 = L4 = L1 in size:** §二.4 asks each layer to pass on about as many units as it receives. With equal sizes, the active **fraction** is conserved too, which is the simplest case. The PFC is then ≈ 0.55 × the sound area and grows with it, as the designer wants ("PFC proportional to the rest"). The alternative (larger L2–L4 at the same count, i.e. sparser) is risk R3.
- **Hippocampus:** k fixed, fan-outs ∝ s, decay ∝ 1/s, as settled by X6 (§四.9).
- **Motor:** more force neurons per muscle = finer force (§二.7). Blocking and stop neurons scale with it.

### 2.4 What the PFC receives (MEASURED, s = 1 / s = 4; birth, no learning)

| | silence | adult speech |
|---|---|---|
| lit chain copies, s = 1 (of 1,298) | 304 (23 %) | 266 ± 27 (20 %) |
| lit chain copies, s = 4 (of 6,151) | 1,156 (19 %) | 1,198 ± 55 (19.5 %) |
| what they are | the fixed "quiet" code: conjunctions of quiet channels from all four layers | syllable-and-voice-specific units plus the part of the quiet code that the syllable's channels do not silence |
| overlap with the silence code (A2 copies) | – | Jaccard 0.53; different syllables 0.39; same syllable, other voice 0.60 |
| motor copies | 0 | 0 while the child is silent; the 8s stop copies light whenever the brake fires on adult speech |
| on top | internally lit copies: loop, predictions, hippocampal recall | same |

- Copies from layer l arrive l + 1 ticks after the input. So L1 at any tick mixes the current syllable (shallow copies) with the previous ≈ 1 syllable (deep copies). This is an anatomical short trace, not a mechanism.
- The X1 band (≤ 0.60) gives **0** lit copies in silence and 81 (s = 1) in speech. It contradicts msg 57 and reproduces the X1 silence collapse.

---

## 3. Open-choice register

Every detail ARCHITECTURE.md does not fix. Default = the simplest form consistent with it. "→ (a)/(b)" means the value is set by a pre-registered rule in that stage, before stage (c).

**World and physics** (X1c `world.py` and X1f `world6.py`, checked; changes only in `render`):

| id | choice | simplest option | reason |
|---|---|---|---|
| W1 | sound physics | the world's 180 channels (150 content, 10 per syllable; 6 timbre channels per voice) given **graded** energy: loudness × U(0.85, 1) on a syllable's channels, 0.9 × loudness on the voice's timbre channels, + \|N(0, 0.04)\| noise, clipped at 1.5. Mumble and garble as in X1c | §二.1 needs a value to thermometer-code; keeps X1c's syllables, muscles and mouth rule for comparability |
| W2 | own voice at the ear | content channels at 0.6 of an adult's level (bone conduction); own timbre channels as before | §二.1 "similar but different". Also keeps the own-voice G1 drive below release, so the last syllable does not loop through the world (risk R6) |
| W3 | thermometer | 3s levels per channel, thresholds U(0.12, 1.0), sorted, lineage-fixed | §二.1; finer resolution as the brain grows |
| W4 | timing, lexicon, tasks | unchanged: 3 ticks per syllable; 34 syllables (the brief said 37; `world.py` has 34); 32 questions; 22 seen + 10 held-out pairs per seed; 40.6 ticks per overheard dialogue (MEASURED) | comparability |

**Sound area:**

| id | choice | simplest option | reason |
|---|---|---|---|
| S1 | depth | 4 layers | local3d and X7 used four transforms |
| S2 | unit form | local3d dom1: 1 dominant input (weight 1) at the same sheet position, 3 weak neighbours (weight 0.05·U(0.5, 1)) | §二.2 cites local3d's 0.989; HARVEST 13 / X7 rejected equal-weight inputs |
| S3 | density | 2 units per sheet position per layer | the smallest that gives each position a choice of dominant |
| S4 | thresholds | θ_l·Σw_in. θ_1 is the value giving A1 ≈ 25 %. Each deeper θ_l is the smallest value ≥ θ_{l−1} giving ≤ 15 / 10 / 6 %. Fixed after birth | §二.1: "rising with depth, roughly sparser, not strictly fixed" |
| S5 | chain band | birth firing rate in [0.002, 0.90], measured over the birth period | DOC "绝大部分时间都在亮（或几乎从不亮）" read as > 90 % / < 0.2 %; msg 57; MEASURED constant PFC input (§2.4). Risk R1 |
| S6 | learning, split/delete | none in the sound area | §二.2 near-input layers fixed; §二.9 off |

**Chains:**

| id | choice | simplest option | reason |
|---|---|---|---|
| C1 | return trigger and strength | a copy's return fires when its **internal** drive (L4 links, or Hs links for hippocampal copies) crosses the copy's internal threshold. It adds g_ret to S's drive. Default g_ret = 1.05 × S's threshold, so the return alone lights S (inner hearing). Neural equivalent: the return neuron shares the copy's internal inputs (registered program shortcut) | DOC §二.4 "被内部激活…时，会通过回程链让…亮起来". A return driven by every heard copy would re-light S forever through the 1:1 forward chain (risk R6). §二.11 step 3 needs inner hearing in silence |
| C2 | input layer A0 | no chain pair | §二.7 "回程到不了输入层" + §二.3 pairing |
| C3 | latency | forward and return chains are 1-tick relays | – |

**PFC:**

| id | choice | simplest option | reason |
|---|---|---|---|
| P1 | depth | L1, two middle layers, L4 | DOC "中间各层" is plural; two is the smallest plural |
| P2 | sizes and positions | N2 = N3 = N4 = N1. Copies sit at their source's (y, z) sheet position (normalised to the unit square, jitter 0.005); motor copies and L2–L4 units are uniform random | §2.3 |
| P3 | feed-forward and thresholds | 24 random local inputs (r 0.1), weights U(0.5, 1.5) normalised to Σ = 1 per unit; one fixed threshold per layer, calibrated once at birth so that mean count(L_{l+1}) / count(L_l) ≈ 1 over the birth period | §二.4 count preservation; §二.1 forbids the sensory fraction rule here |
| P4 | inhibition | P0 = none; P1 = one non-learning inhibitory neuron per ≈ 33-unit patch (Voronoi), same tick, subtracting g_pm × the mean drive of its members (X1d / X3 form) in L1-internal, L2, L3 and L4. Choice → (a) | §四.10 order and its adopted form |
| P5 | "rarely co-active inhibit, related excite" | effective lateral weight = learned excitation − patch inhibition (no extra mechanism); verified by d7 | msg 27 method 1; DOC §二.4. Partner form = risk R8 |
| P6 | loop at birth | reciprocal (msg 84 "G 返回过来继续激活 A"): each L4 unit returns to the 30 L1 copies with the largest path weight into it, total g_loop. Birth part, non-decaying. Form and g_loop → (a) | §二.4 "回路保持想法"; §三: PFC links random at birth, trained on top |
| P7 | L4 → L1 learning | potential synapses = all L1 copies within r 0.15. Pre = L4 units on or fell ≤ 2 ticks at t − 1. Post event = **onset of the copy's chain-driven input** while the copy was not already on | DOC "和下一刻第一层的**输入**做赫布连接". With this post event, inner hearing (S re-lit by the return → chain) does not confirm its own prediction, and a correctly predicted input is not re-strengthened. Risk R5 |
| P8 | L4 ↔ L4 | birth: 20 random local (r 0.15), total w44, non-decaying. Acquired: all within r 0.15, same rule, post = L4 onset | §四.11 (last layer's internal links learn) |
| P9 | two-layer weights | p = g·w / √(max(out, w_ref) · max(in, w_ref)) on every acquired part. Totals updated incrementally at each weight change, so they are exact every tick | §二.4, §五 X3, HARVEST 6; the floor prevents X6's 1/√10⁻⁶ boost of new links |
| P10 | internal activation of L1 | copy fires internally iff its L4 (birth + acquired) drive ≥ θ_1 (fixed); internally lit copies count fully in L2 | audit 1 |
| P11 | thinking-range size | whatever count preservation gives (≈ 20 % of L4) | §二.4 vs §四.20; risk R3 |
| P12 | gains g31, g44, g_loop, w44, g_pm | relative to θ = 1. g_loop, w44, g_pm → (a). g31 = g44 = 1 → pilot rule in (b): the largest value at which a 1,000-dialogue pilot stays inside (a1)'s bounds | "调参, 不加机制" (§四.6) |

**Learning rule and decay:**

| id | choice | simplest option | reason |
|---|---|---|---|
| R1 | rule | §二.8 held form (table row 2.8); fall window 2 ticks for both signs | DOC; audit minor row |
| R2 | rates | η = 0.02, η⁻ = 0.01, w_ref = 1, pow = 1, floor 0.01; hippocampus η_H = 0.3 (X3) | X1c–X3 values; balance formula (HARVEST 2) checked after the pilot |
| R3 | decay | every 200 ticks: Δw = −(1 − γ)·w·c(w); prune < 10⁻³. γ_H = 1 − 0.01/s; γ_motor (incl. G1) = 0.9999; γ_PFC = 0.99995 (corrected 11-02: the PFC is the deepest, so the slowest) | §二.8 (deeper = slower, hippocampus fastest, concave); X6 (∝ 1/s); X8 (concave decay) |
| R4 | sign | acquired parts of excitatory projections stay ≥ 0 (weakening stops at 0; it cannot cancel birth links) | §二.7 "inhibitory stay inhibitory", by symmetry; the simplest form |
| R5 | reward | none (r = 0) | not needed (§1 row 2.8) |

**Motor:**

| id | choice | simplest option | reason |
|---|---|---|---|
| M1 | force | s force neurons per muscle, thresholds θ_m·(1 + k/s), k = 0 … s − 1; the body counts a muscle as on iff ≥ ⌈s/2⌉ fire | §二.7 |
| M2 | blocking | 8s neurons. 4s tonic (always on). 4s on a ladder of thresholds over the pooled muscle drive. Each inhibits every force neuron by b | §二.7 "抑制占上风", competition, no hard gate; replaces A2 |
| M3 | stop | 8s neurons; inputs: brake (birth), stop-copy returns (1:1), nothing else learned here | §二.7 |
| M4 | brake | count of lit up-lines among the other four voices' A0 timbre channels ≥ n_brake (∝ s) → all stop neurons (birth reflex, non-decaying). Leaving out the own-voice timbre channels is **innate birth wiring** (the lineage knows its own body), not learned. The brake reads only the input layer and is graded by loudness | §二.7 "只看输入层", loudness-graded, so distant TV hardly brakes |
| M5 | G1 source layer | sound layer A1 (shallowest feature layer) only | echo and brake then both take 2 ticks to reach the muscles, so the heard code ends exactly when the brake ends (no echo window); deeper sources would leave 1–3 echo ticks |
| M6 | release scale | G1 scaled so that a full adult-voice code gives 1.5 × the release level against tonic blocking, and own voice (W2) gives < 1×. Checked in (b) | §二.7 |

**Hippocampus:**

| id | choice | simplest option | reason |
|---|---|---|---|
| H1 | sources | copies of every chain source (sound, force, stop; concept in C+) | §二.3, §二.6 |
| H2 | own population | N_H = 5,000·s; fixed random recode fan-in 24 from H1; fixed threshold + same-tick mean-drive inhibition, calibrated for ≈ 50 active | X3 / X6 (§五: "总激活越多越难再往上推"); no quantile quota |
| H3 | recurrence | no birth Hs ↔ Hs; learned Hs ↔ Hs (500·s random candidates) and Hs → H1 (200·s) | designer 11-01; X6b |
| H4 | store vs recall | no switch: learned links always act (DOC has none). Returns use the same g_ret | risk R7 |
| H5 | episodes that start alike | nothing added: H2/H3 stay exactly as written (fixed threshold + same-tick inhibition, no quota, no birth recurrence) | designer 11-02: in a small model, failing to separate them is normal and is left alone |

**Concept area (arm C+ only):** X9's two-layer form.
- C1: 40 random inputs from sound feature units.
- C2: previous C1 through fixed random K, plus current C1 through learned F.
- Both layers send chain pairs to the PFC and the hippocampus.
- It needs a class-specific frame in the world (X9). The current world has none, so C+ runs in stage (c2) with one added frame; it is reported, not pass/fail.

---

## 4. Hand-wiring register (generation 0)

**Birth wiring.** Lineage-fixed, content-free, non-decaying (§二.8 "出生连接不降"). This is not "hand-wired content":
- sound layers A1–A4;
- chain pairs, by the S5 band;
- PFC feed-forward and the reciprocal loop (P6);
- birth lateral L4 links;
- patch inhibitory neurons;
- the hippocampal recode;
- tonic and ladder blocking neurons;
- the brake reflex.

**Generation-0 hand wiring** (§二.11 step 1, §二.12; in the plastic, decaying part):

| set | from → to | how computed | why | decays |
|---|---|---|---|---|
| G1 | chain-bearing units of sound layer A1 that reliably fire for syllable x in the four adult voices (fire in ≥ 50 % of x's ticks, and ≥ 0.3 more often than for any other syllable) → the force neurons of x's 4 muscles | non-negative least squares (ridge 0.3, X1f method) from the measured newborn codes to the muscle sets; scaled per M6; measured on the frozen newborn after the birth period | "模型心里想到一个字的声音，就能本能地说出来" (§二.11.1); babbling cannot learn it (X7) | yes, γ_motor, concave-slowed; retention is checked (b7) |

**DIAGNOSTIC only, used only if (c) fails, to locate the break.** DPC: held, question-specific paths (X1f G0P form) for all seen pairs, placed in the acquired L4 ↔ L4 part, with no Q → A link. It tests whether the readout learns once a path exists.

**Never hand-wired, so all of it can be tested:**
- any question-holding or question-specific path (main arms);
- any Q → A link or answer content, for all 22 seen and 10 held-out pairs;
- multi-syllable order;
- stopping after speaking;
- turn taking and quiet waiting;
- the PFC's stop predictions;
- concept-area content.

No sealed pair is ever spoken during training, and code is frozen before stage (c).

---

## 5. Compute plan (A100 40 GB, 16 CPU cores, torch 2.4, ≤ 90 min per job)

**Life length.**
- 40.6 ticks per overheard dialogue (MEASURED), so 10k dialogues = 406k ticks and 30k = 1.22M ticks.
- Frozen probes: 32 pairs × 2 contexts × ≈ 45 ticks ≈ 3k ticks per checkpoint.

**Estimated speed.** X5 measured ≈ 10 ms per tick at B = 1 for a 55k-unit, 32M-synapse model, launch-bound. These estimates assume ≤ 60 kernels per tick and one host sync (the world). They are **measured first** in a 2,000-tick pilot at each s.

| s | units / synapses | seeds per process (B) | ms per tick (EST) | process ticks/s | 10k-dialogue life |
|---|---|---|---|---|---|
| 1 | 21k / 4M | 8 | 3–5 | 200–330 | 20–35 min |
| 4 | 90k / 63M | 4 | 4–8 | 125–250 | 27–55 min |
| 16 | 384k / 1.0G | 1–2 (≈ 10 GB each) | 10–20 | 50–100 | 70–135 min → 2 chained jobs |

**Implementation rules.**
- **Sparse, never dense N × N.** Fixed birth projections are ELL tables (fixed fan-in, int32 indices): gather → sum → threshold.
- **Learned local projections** (L4 → L1, L4 ↔ L4) are ELL-by-post over the radius candidates:
  - drive = gather of pre states × propagation weight;
  - strengthening touches only rows whose post rose;
  - weakening touches only entries whose pre rose;
  - out/in totals are updated incrementally.
  - At s = 16 these are index lists of rising units, not full-table masks.
- **Hippocampal tables** (Hs ↔ Hs, Hs → H1) are event-driven: about 50 active rows per tick, gathered by pre with index_add.
- **Decay** is one full pass over the acquired tables every 200 ticks.
- **Batching.** Seeds run in lockstep in one process (batch dimension). The worlds stay on CPU (≈ 40 µs per world tick); only A0 bits go up and 24s force bits come down each tick.
- **Probes** freeze learning and restore only neuron states, so they never copy the 8 GB tables.
- **Checkpoints** (fp16 weights) at every 1k dialogues, so an s = 16 life spans 2–3 chained 90-min jobs.

**Job budget (EST): about 22 job-slots.**

| stage | jobs |
|---|---|
| (a) newborn grid at s = 1, plus confirmation at s = 4 and 16 | 2 |
| (b) at s = 1, 4, 16 | 1 |
| (c) main: s = 1 (5 seeds), s = 4 (3), s = 16 (3) | ≈ 9 |
| (c) arms (H-off, R1/R3/R5 arms) | ≈ 6 |
| (c2) C+ | 2 |
| DPC, only if needed | 2 |

At most 2 jobs run at once; one local process at a time, for checks only.

---

## 6. Test protocol and pass criteria

All criteria are fixed now. Every stage uses frozen probes (learning off, mouth on, the brain state restored after each probe), and all world labels stay outside the brain. Decoding = nearest centroid on the binary layer state, trained on half the contexts or noise draws and tested on the other half.

### (a) Newborn checks (no learning)

Each seed has a birth period of 200 overheard dialogues (≈ 8k ticks) with learning off. Chain bands and thresholds are set during it (S4, S5, P3, H2).

- **Grid at s = 1, 2 seeds:**
  - inhibition P0 / P1 (g_pm 1);
  - loop reciprocal / random-local;
  - g_loop ∈ {0.5, 1, 2, 4};
  - w44 ∈ {0, 0.5, 1};
  - 48 configurations.
- **Probe set:**
  - 32 questions × 4 preceding contexts (silence, narration, another question, TV line) × 2 noise draws;
  - questions asked in adult voices;
  - followed by 12 silent ticks.

| check | measure | pass |
|---|---|---|
| a1 activity constant (msg 57, §二.4) | per layer of every area: mean count in silence ÷ in adult speech; PFC count(L_{l+1}) ÷ count(L_l) in silence and in speech; extremes over 8k ticks | every ratio in [0.67, 1.5]; no PFC or Hs layer ever empty or > 50 % on; sound fractions non-increasing with depth |
| a2 same input → same code (§四.19) | PFC L4 and Hs at question end (lag 0): 32-way decoding across contexts; Jaccard same-question-other-context minus other-question-same-frame | L4 decoding ≥ 0.5 (chance 0.031) and margin ≥ 0.1; Hs decoding ≥ 0.5 |
| a3 thought held through the pause (§二.4 "回路保持想法") | L4 32-way decoding at lags 0/3/6/9/12; filler-given-frame decoding at lag 12; motion J(t, t+1) | lag-12 decoding ≥ 0.3, filler-given-frame ≥ 0.3 (chance 0.125), and J(t, t+1) < 0.9 (not frozen) |

**Decision rule.**
- Among the configurations passing a1 and a2, take the one with the highest lag-12 decoding.
- Ties go to P0 over P1 (§四.10 order), then to the smaller g_loop.
- Confirm at s = 4 and 16 with 2 seeds each, using the same relative gains.
- If no configuration passes a3, proceed with the best one and flag it. §四.6 and §四.11 say learned L4 links should then carve paths, which d2 re-measures in (c).

### (b) Generation-0 speech basics (G1 wired; learning on only for b6–b7)

At s = 1, 4 and 16, 3 seeds each.
- "Forced" = a DIAGNOSTIC hook that lights, internally, the PFC copies of a syllable's adult-voice code for 3 ticks per syllable.

| check | pass |
|---|---|
| b1 forced inner hearing of each of 34 syllables → said exactly | ≥ 0.9; garble ≤ 0.05 |
| b2 forced whole answers (all 32 answers, multi-syllable in order) → said exactly | ≥ 0.9 |
| b3 own-voice loop: after b1, the same syllable is said again within 6 ticks | ≤ 0.05 |
| b4 echo: an adult line heard, child said anything within 6 ticks after it ends | ≤ 0.02 per line |
| b5 unsure: forced mixture of two syllables | garble ≤ 0.1 (silence or one clean syllable) |
| b6 brake: forced answer started, adult starts talking | child stops within 1 syllable in ≥ 0.9 |
| b7 newborn while overhearing 1k dialogues (learning on) | utterances ≤ 0.05 per dialogue; b1 re-tested afterwards ≥ 0.85 (G1 retained) |

- g_ret and M6's scale are tuned here only if b1/b3 fail at the defaults. The sweep is g_ret ∈ {0.8, 1.05, 1.3} × G1 scale ∈ {1.2, 1.5, 2}. The rule: pass b1–b6 with the smallest g_ret.
- Then g31/g44 are fixed by the P12 pilot rule.
- **Code is frozen after (b).**

### (c) Systematic training (§二.11): overhear, then answer never-wired questions

**Main arm M:** the architecture as above (hippocampus on, concept off).
- Seeds: s = 1 (5), s = 4 (3), s = 16 (3).
- Sequence: birth period, then G1, then overhearing in OverhearWorld with learning on. Held-out pairs are never spoken.
- Checkpoints at 300, 1k, 3k and 10k dialogues; 30k only where accuracy still rises between 3k and 10k.

**Probe at each checkpoint:**
- a caregiver (C1 or C2, both heard in training) asks each of the 32 questions after a silence or narration context;
- the caregiver then waits 12 ticks;
- the child's utterance is scored as exact, first syllable, silent, wrong or garbled.

**Primary pass (functional):** on the 22 seen pairs, which are never hand-wired. Use the best checkpoint ≤ 10k, seed mean.

The frame baseline is the accuracy of always giving each frame's most common seen answer. MEASURED over 20 task seeds it is 0.50–0.73 (mean 0.58). So raw accuracy says little, and the pass is set on question-specific answers:
- **minority-pair** exact accuracy ≥ 0.5. Minority pairs are those whose answer differs from their frame's most common answer, about 40 % of the seen pairs; frame-level guessing scores 0 on them;
- **and** overall exact accuracy ≥ the seed's frame baseline + 0.1;
- wrong answers ≤ 0.25 of probes;
- garble ≤ 0.1.

**Scale hypothesis (designer):** accuracy at s = 16 > s = 4 > s = 1, paired by seed index. It is reported with the paired differences.

**Secondary, reported only:**
- held-out pairs (needs C+ and class frames, §四.14);
- a live block of 300 real-caregiver exchanges with learning on (accuracy, talk-over, failure to yield, garble);
- retention after 1,000 more dialogues with the probed pairs removed.

**Diagnostics.** Each locates one link of the chain: code → path → prediction → inner hearing → release.

| id | measure | healthy |
|---|---|---|
| d1 | a2 repeated at each checkpoint | L4 lag-0 decoding ≥ 0.5 |
| d2 | a3 repeated (path through the gap) | lag-12 ≥ 0.3 with motion |
| d3 | learned pre-threshold drive into the answer's first-syllable copies vs the best wrong syllable; share of questions whose answer is top-driven; fair specificity (own − one other question) with a 200-permutation null | ratio ≥ 1.5; specificity above the null's 95 % |
| d4 | share of the answer's layer-A1 units lit by return within 6 ticks after the question | ≥ 0.5 when d3 is healthy |
| d5 | G1 drive sharpness at release (top-4 muscles ÷ rest); blocking recruited | – |
| d6 | counts per layer; positive-pair fraction of L4 ↔ L4; hub share of predictions | no layer > 50 %; pairs < 30 %; hub < 0.3 |
| d7 | effective lateral weight (learned − patch) vs co-activity rank | negative for the least co-active quartile, positive for the most |
| d8 | share of S onsets caused by hippocampal returns; M vs H-off | – |
| d9 | b1 re-tested (G1 retention) | ≥ 0.85 |
| d10 | birth-loop per-link push vs learned prediction push (§二.5 "return weaker than next") | learned > loop |

**Arms** (3 seeds unless noted):
- **H-off:** s = 1 and 4; is the hippocampus helping or injecting noise? (R7)
- **B80:** chain band ≤ 0.80, s = 1 (R1).
- **ρ4:** L2–L4 four times larger, same count, so a quarter of the fraction; s = 1 and 4 (R3).
- **POST-any:** post event = any L1 onset, s = 1 (R5).
- **C+:** stage (c2) (§3).
- **DPC:** DIAGNOSTIC, only if M fails at every s. It tells "no path" apart from "no readout".

**If (c) fails,** the report states where the chain breaks, with the d1–d10 numbers, and which R-item it supports.

---

## 7. Risks: where ARCHITECTURE.md may be genuinely under-specified

| # | clause | what is under-specified | how to decide by experiment |
|---|---|---|---|
| R1 | §二.3 "绝大部分时间都在亮…没有链式神经元" vs msg 57 | Where "most of the time" ends decides whether silence reaches the PFC. MEASURED: ≤ 0.60 → silence feeds 0 copies (X1 failure); ≤ 0.90 → constant input, but about half of each syllable's code is the shared "quiet" code (J 0.53 with silence) and ≈ 20 % of L1 is lit | arm B80 vs M at s = 1: compare a1, d3 specificity and accuracy. If both fail at d3 for this reason, ask the designer whether quiet-detector units should carry chains |
| R2 | §二.12 "越大越好" with §二.4 count preservation | With a constant active fraction (≈ 20 %), capacity grows only through the learned fan-in. EST: storable transitions ≈ C (radius candidates): ≈ 100 (s = 1), ≈ 450 (s = 4), ≈ 2,000 (s = 16), against ≈ 300–600 question/answer/frame transitions needed. Prediction: s = 1 fails by capacity, s = 4 is borderline, s = 16 is enough | the scale comparison in (c) is the test; d3 and d6 show whether the failure is crosstalk |
| R3 | §四.20 thinking-range size | count ∝ s (chosen) vs fixed 20–30 with sparser codes. The latter needs larger L2–L4 at the same count, which §二.4 allows (it fixes counts, not sizes) | arm ρ4 at s = 1 and 4 |
| R4 | §二.4 "回路保持想法" | the birth loop's form (reciprocal or random) and gain are not given; §四.6 / X1d found no birth regime in non-conformant builds | the (a) grid; if a3 fails everywhere, d2 in (c) shows whether learning carves paths (§四.11) |
| R5 | §二.4 "和下一刻第一层的**输入**" + §二.8 "B 后亮" | is the post event the copy's onset, or its input's onset? Neuron onsets make internally caused onsets reinforce their own causes (X1d: hub 0.91); input onsets do not | arm POST-any vs M at s = 1: hub share (d6) and accuracy |
| R6 | §二.5 return strength; §二.7 own voice | With 1:1 forward + 1:1 return and binary units, a return that lights S by itself closes a gain-1 loop (S → copy → S). DOC's "returns on internal activation" avoids that. But inner hearing (needed for §二.11 step 3) and own-voice echo then differ only by W2's level (0.6) | (b) b1/b3 with the g_ret × G1-scale sweep; if no margin, report and ask |
| R7 | §二.6 storing vs recalling | DOC has no switch; X6 found no strength ratio works without one. Recall may light sound units at random and reach the mouth or the PFC as noise | H-off arms and d8. §四.19's "listening" signal is a proposal, not used |
| R8 | §二.4 "很少一起亮的神经元互相抑制" / §四.3 | X10 realises it as net effective weights (msg 27 method 1). The designer's partner form is untested and, if partners are on whenever their neuron is off, it reduces to a self-hold (§三) | d7; if d7 fails, a partner arm with partners driven only by learned input |
| R9 | §二.4 count preservation with internal additions | fixed thresholds preserve counts only at the birth operating point; loop and prediction add copies on top, which can run away (X1c) | the a1 bounds and d6 over the life; §四.10 order (P0 → P1) |
| R10 | §二.12 fan-in scaling | birth fan-ins fixed and learned fan-ins ∝ s is my reading; the hippocampus costs memory ∝ s² (8 GB at s = 16) | the s = 16 pilot measures memory and speed; fallback is fan-out ∝ √s for Hs only, registered before use |
| R11 | §二.7 G1 in the decaying part | G1 is unused during overhearing, so it decays slowly. The concave decay should keep it (X8) | b7 / d9 at every checkpoint |
| R12 | §二.1 voices | the world's 6 timbre channels per voice make codes strongly voice-dependent (same syllable in two voices J 0.60 vs different syllables 0.39) | probes use voices heard in training; report a never-heard asker voice separately |

---

## 8. Files

- `x10/PLAN.md`: this plan.
- `x10/sizing/sound_stats.py`: the sound-area statistics behind §2.4 and S5. Run as `python3 -I sound_stats.py <ticks> <s>`; 4 s at s = 1, 18 s at s = 4.
- `x10/sizing/counts.py`: the unit and synapse counts in §2.
- Reused, read-only: `x1g/world.py` and `x1g/world6.py` (34 syllables, 24 muscles, OverhearWorld). Only `render` changes, to W1/W2.

## summary_zh

1. X10 严格按总纲搭：输入层用分档的温度计正反对编码，永远一半亮。后面四层是 local3d 式的局部网络，越往后越稀疏。
2. 每层的特征神经元都有一对一的去程和回程链，送到前额叶和海马体。链的范围按"亮的时间在 0.2% 到 90% 之间"来定，这样安静时也有输入：s=1 时安静 304 个，说话 266 个，大致恒定。
3. 前额叶有四层：输入层、两层中间层、最后一层。各层大小等于输入层，门槛固定，校准到"进多少、出多少"。内部点亮的线算全额，回路必须真的闭合。
4. 海马体、运动区（没有硬门槛，靠阻断神经元的竞争）都在。概念区作为后续一组，按 §四.14 的顺序加。
5. 一个放大旋钮 s 让所有区一起放大：s=1/4/16 时约 2.1 万 / 9 万 / 38 万个神经元，约 400 万 / 6300 万 / 10 亿个突触。
6. 第一代只手动接"声音→肌肉"一组，放在会降权的部分。任何问答、思路、停顿方式都不手接，所以都能拿来测学习。
7. 测试分三步：新生儿检查（安静和说话时亮的数大致相同；同一输入得到同一编码；问题在 12 拍停顿后还能认出），说话基本功，最后旁听大量对话后回答从没手接过的问题。
8. 及格线：每个句式都只答最常见的答案就能对 0.50–0.73，所以要看"答案和句式常见答案不同"的那些题：这些题答对 ≥ 0.5，总答对率也要比这个基线高 0.1。另设 10 个诊断，用来定位断点。
9. 主要风险：链范围让每个字的编码有一半和"安静"共享；前额叶约 20% 亮，容量要靠放大后的入边，预计 s=1 不够、s=16 才够；回程强度与自己声音回声的余量；海马体没有存取开关。
10. GPU 估算：s=1 和 s=4 一次 1 万段旁听约半小时到一小时；s=16 要分成两三个 90 分钟的任务接着跑。

## Addendum 11-02 (stage a, run)

Registered additions:
- **Online birth calibration** (`brain.py:Brain.calibrate_online`). Each configuration's birth stream is replayed with its own birth loop and lateral on, and learning off. Every 50 ticks, each PFC layer's fixed threshold moves toward count(L_l) = count(L_(l−1)), and Hs's toward kh. The thresholds are then frozen.
- **a1** (coordinator 11-02):
  - A0 exactly half on;
  - PFC layer ratios in [0.67, 1.5] in silence and in speech;
  - no PFC or Hs layer empty or over 50 %;
  - silence L1 ≥ 0.2 × speech L1.
- **a1L** (`probes.py:stability`): 300 learning dialogues; every 10-dialogue window has PFC ratios in [0.5, 2] and Hs in [0.5, 2] × kh; no layer over 50 %.
- **Remedy ladder** (`run_a.py`, §四.10 order):
  - R1: learned gains g44 / g_hh / g_h1 at 0.3;
  - R2: the same gains at 0.1;
  - R3: R2 plus η / η_H at a quarter;
  - R4: R3 plus g_ih 2 and patch gain +0.5.
- **Sensitivity sweep** (`run_a.py --mode sens`): 20 one-at-a-time variations. Results: `results/sens/`, tables from `agg_sens.py`.
- **Hippocampal reproducibility:** `probes.py:newborn_codes` also reports the Hs Jaccard margin (same question in another context minus another question of the same frame).

## Addendum 11-02 (coordinator decisions on stage a; pre-registered before the runs)

**a2, registered revision.**
- The hippocampal part of a2 becomes: Hs Jaccard margin (same question in another context − another question of the same frame) ≥ 0.1.
- Hs 32-way decoding is still reported as a number.
- Reason: ARCH §四.19 ("same input → same code") and designer 11-02 ("模型分不开相同的经历很正常，先不管"). By H3 the hippocampus has no birth recurrence, so its lag-0 code holds only the current input.
- `probes.py:newborn_codes` computes the margin in every run.

**Configuration choice (step 2).**
- Candidates: P1, reciprocal loop, g_loop 1.0, medium PFC, default rates, plus one of:
  - (A) w44 0.5 + R2;
  - (B) w44 0 + R2;
  - (C) w44 0 + R3.
- Seeds 0–3 for each: newborn checks (a1, a2, a3), then the **extended a1L**: 1,000 learning dialogues in windows of 10 dialogues.
- A window passes if the PFC ratios are in [0.5, 2], Hs is in [0.5, 2] × kh, and no layer exceeds 50 % in that window.
- **Extended a1L passes** if:
  - ≥ 95 % of windows pass;
  - L4's mean over the last 100 dialogues is ≤ 1.2 × its mean at dialogue 500 (dialogues 490–509);
  - the same holds for Hs.
- Slopes of L4, Hs and A4 over the last 500 dialogues are reported per 100 dialogues.
- **Rule:**
  - Among candidates that pass a1, the revised a2 and the extended a1L on all 4 seeds, take the highest mean lag-12 decoding.
  - If two are within 0.03 of each other, prefer normal learning rates (A or B).
  - The runner-up becomes the (c) arm.

**Large PFC (step 3).**
- Large preset × g_loop {1.5, 2.0}, with P1, reciprocal, w44 0, R2.
- Each run: birth checks plus a1L over 300 dialogues, on seeds 0 and 1.
- It joins (c) as an arm if it passes a1 and a3 with lag 12 ≥ the chosen medium configuration's.

**G1 construction (step 4, approved; ARCH §二.11.1, §二.12).**
- G1's input matrix uses only A1 units that the child's own voice does not light.
- Rule: a unit is cut if, for any of the child's own syllables, it fires in ≥ `g1_own_cut` = 0.2 of that syllable's ticks.
- G1 stays in the plastic, decaying part. There is no new gate, and the brake still reads only the input layer.
- Reported per syllable: A1 code units kept vs before; any syllable below ⅓ is flagged.
- Then the registered sweep g_ret {0.8, 1.05, 1.3} × G1 scale g1_rel {1.2, 1.5, 2.0} on candidate A, with b1–b6.
- The b1–b7 pass criteria are unchanged: the setting that passes b1–b6 with the smallest g_ret, then full (b) on the chosen configuration.

## Addendum 11-02 (designer: no sound → motor connection; learned speech first). Registered before the (b) runs

**G1 is removed.**
- Designer: "听觉区神经元不应该直接连接运动区神经元，而是通过前额叶间接建立联系" (ARCH §二.7, §二.11.1/3, a6de412).
- The motor area now has no input table. A force neuron's only excitatory input is its own 1:1 return chain, with return drive = g_ret_force × (its threshold + tonic blocking).
- Unit check 9 asserts that no table targets force neurons.

**Main (b) arm: learned speech** (designer: "一开始强制模型按规律说话，模型自动就会听到自己的声音，然后把声音和动作联系起来", f86147a).
- Gen-0 scaffold "forced regular speaking" (`scaffold.py`): the teacher program fires syllable x's force neurons on the forced ticks, bypassing the blockers. It is outside the brain and logged.
- No speech wiring; learning on everywhere; brake and own-voice handling unchanged.
- Patterns: 30 forced syllables per syllable in each arm, in random order; syllable 3 ticks, gap 3, pause 12.
  - F1 repetition: x x x.
  - F2 single: x, then a pause (control).
  - F3: an adult says x, then the child is forced to say x x.
- Afterwards, with the scaffold off:
  - b1 by inner hearing (x's sound copies lit internally in PFC L1);
  - an "adult says x, then 12 silent ticks" probe;
  - shares said clear / mumble / garble / other / silent;
  - b2–b7;
  - link diagnostics on L4 → force copies: true/other weight ratio, top-4 exact, margin (X7: about 1.5 needed);
  - the a1L windows during the scaffold, and lag-12 hold afterwards.
- If F1 or F3 passes b1/b2, it replaces hand wiring. Otherwise G1P is used, and the report names the failing link: order, margin, or generalisation from own to adult voice.

**Comparison / fallback: G1P** (`g1p.py`), gen-0 hand wiring inside the PFC.
- From: L4 units that represent x on the frozen newborn. They fire in ≥ 50 % of x's L4 ticks (4–9 after onset) in the four adult voices, and ≥ 0.3 more often than for any other syllable. Syllables with fewer than 10 units are flagged.
- To: x's force copies in PFC L1.
- Fit: NNLS (ridge 0.3), with one gain giving g1p_rel × θ_1.
- Same form as L41: acquired part, two-layer weights, γ_pfc, and the same rule and post event.
- Arms: (i) adult voices only; (ii) also dropping units that the child's own voice lights in ≥ 20 % of x's ticks.
- Registered sweep: g_ret_force {1.1, 1.3, 1.6} × g1p_rel {1.2, 1.5, 2.0}, with b1–b6. Pick the arm and setting that pass b1–b6 with the smallest g_ret_force, then full (b) on the step-2 configuration.

**Echo.** The PFC holds the last heard syllable for about 12 ticks, so the child may repeat what it heard (b4, b7, adult-hold probe). This is reported; no fix is added without the coordinator.
- **G1P selection, registered variant (added after the smoke run, before any full (b) run).** The specified rule (p ≥ 0.5, margin 0.3) finds 0–10 L4 units per syllable at s = 1; 33 of 34 syllables are flagged. A relaxed margin of 0.1 is run next to it. Both are reported; the choice goes by the stage-(b) results.

## Addendum 11-02 (b): diagnosis and arm M, registered before the runs

**diag_g1p** (config A, G1P margin 0.1):
- **(iii) return vs blocking is the first blocker.** Lighting x's force copies directly gives no speech (force neurons 0, silent 1.00). The return drive is 0.91 (g_ret_force 1.3), but the inhibition is 1.05: tonic 0.4 plus about one stop neuron on in silence (the PFC holds stop copies and returns them). A return needs > 1.35.
- **(i)** L4 codes at birth are weakly syllable-specific. Jaccard: different syllables 0.84, same syllable in two voices 0.86, against the pre-onset state 0.66.
- **(ii)** Selective L4 units peak 6–8 ticks after onset (margin 0.3: median 14.5) and are 0 in the first 3 ticks.
- **The G1P link itself works:** during inner hearing, 91 % of x's selected units fire, and the drive into x's own force copies is 3.9 against θ_1 = 1.

**Arm M** (coordinator; ARCH §二.7 "亮的力度神经元越多，力越大", §二.4/§五 proportions):
- m force neurons per muscle (`force_per_muscle`), with m ∈ {1, 8, 32} at s = 1. Thresholds are graded within each muscle's population: θ_m (1 + k/m).
- Each force neuron has its own 1:1 chain copy at a random L1 position.
- The body reads a muscle as on when ≥ ⌈m/2⌉ of its force neurons fire.
- Tonic blocking (4m), ladder blocking (4m) and stop neurons (8m) scale with m, with inhibition normalised by 8m, so the competition keeps its birth balance.
- **Return gain:** g_ret_force = 1.6, from the registered sweep {1.1, 1.3, 1.6}. It is the smallest value that clears tonic blocking plus one or two stop neurons: 1.6 × 0.7 = 1.12 > 0.3 + 0.4 + 0.25. A G1P run at m = 1 with 1.6 isolates the m effect.
- **Runs, seed 0, config A:**
  - G1P at m {1, 8, 32}, margin 0.1;
  - F1 and F3 at m {8, 32}.
- **Reported:**
  - b1–b7 and the speech-link probes;
  - the L41 true/other ratio, top-4 and margin;
  - the number of distinct L4 units with a learned weight > 0.01 onto x's force copies;
  - a1L windows during the scaffold, plus the full activity budget there (first / middle / last window, forced and unforced ticks);
  - lag 12 afterwards.
- Secondary knob, only if M fails: r_learn 0.25 for L4 → L1.

## Addendum 11-02 (a): Hs pre-decision, registered before the clean choose reruns

The choose01 runs had the old G1 wired, which kept 2–5 force neurons firing continuously, so they were rerun with no speech wiring. If the clean reruns still show Hs settling above 2 × kh while the trend is flat:
1. **§四.10 order first.** Add one arm with g_hh = g_h1 = 0.05, all else as the candidate, on seeds 0–1. If it brings Hs within 2 × kh without dropping the revised a2 (Hs J margin) below 0.1 or the mean lag 12 by more than 0.03, adopt it for all candidates.
2. **Only if that fails:** the window bound becomes Hs ≤ 3 × kh. Reason: what matters for one unbroken life is no runaway (a flat trend). This is reported as a revision.

a3 is judged on the clean runs only.

## Addendum 11-02 (a): context test, registered before running

Designer: "增加上下文应该可以把事情记得更久吧。可以测试一下" (more context should help the model remember longer; test it).

**Probe** (`probes.py:context_test`): the a3 probe with lags 0 / 6 / 12 / 24 / 48 ticks after the question, the same 32 questions, and 4 repetitions per question (voices C1 / C2 alternating, fresh noise). Arms:
- **K0:** unrelated preceding context, as in a3 (silence, a narration about a random item, another question, a TV line; one per repetition).
- **K1:** a narration about the same item just before the question (`world.narration(x)`, for example 苹 红 的), then the question.
- **K1c:** a narration of the same form about a different item x′, then the question about x. K1 − K1c isolates the effect of related context from that of more sound.
- **K2:** a richer item inside the question: the item carried by 2 syllables (x plus its colour, from props) in place of the single filler, in the same frame.
- **K3** (hippocampal cue, speaker change) is not run: it needs a second speaker / scene axis that the probe world does not have, so it is not cheap.

**Measures** at each lag:
- 32-way decoding of the question from PFC L4 and from Hs;
- filler-given-frame decoding (8-way) from PFC L4 and from Hs;
- motion J of L4;
- "how long it is remembered": the first lag at which 32-way decoding falls to chance + 0.1 or below.

The decoder trains on 2 repetitions and tests on the other 2, both ways.

**Runs:**
- At birth (learning off) for candidates A, B and C on seeds 0–1 (B and C share a newborn).
- On the learned brain at the end of the extended a1L for B and C on seeds 0–1, run in the same process, so no checkpoint has to survive between jobs.
- The results are reported side by side; they do not enter the (a) decision rule.

## Addendum 11-02 (a): fast-PFC test (main) and a small excitability comparison, registered before running

**Fast PFC.** Designer: "其实让前额叶像海马体一样更容易改变，也能记住前几拍的事情，而且更简单". Compare the synaptic theory of working memory (Mongillo, Barak & Tsodyks 2008).

The arms change only the PFC acquired tables (L4 → L1 and L4 ↔ L4). The rule, two-layer weights, concave factor and single decay are unchanged. Base: candidate C (P1, reciprocal, g_loop 1.0, w44 0, learned gains 0.1, η_H 0.1).

| arm | PFC η (η⁻ = η/2) | γ_pfc |
|---|---|---|
| F0 | 0.005 | 0.99995 |
| F5 | 0.025 | 0.999 |
| F20 | 0.1 (hippocampal level) | 0.99 (hippocampal) |

F5 and F20 knowingly break "deeper layers decay slower" for the PFC; this is the designer's test and is recorded as such. s = 1, seed 0, then seed 1. Measures:
- **a3 probe with learning on** (`probes.question_codes(learn=True)`): the full state, including all plastic weights, is restored between trials, so fast weights act only within a trial.
  - Lags 0 / 6 / 12 / 24 / 48 ticks; 32-way and filler-given-frame decoding.
  - Remembered until: the first lag at which decoding falls to chance + 0.1 or below.
  - Motion J (must stay < 0.9).
  - Input influence: Jaccard of L4 at lag 0 between different questions.
  - The same probe with learning off, as a check that any gain comes from the weights.
- **One-shot.** Each of the 10 held-out pairs is heard once: adult C1 asks, adult C2 answers 3 ticks later. Then 12 silent ticks, then D ∈ {200, 500, 1,000, 2,000} ticks of ordinary overheard dialogue, learning on throughout.
  - Then Q is heard again with learning off, and we check whether A's first-syllable code is the most driven syllable copy in L1 (internal drive, ticks 1–6).
  - Hit rate vs D, against a no-exposure control. The full state is restored between trials.
- **Stability and long-term knowledge.**
  - 5 held-out pairs are each heard 20 times at the start (interleaved).
  - Then a1L runs for 300 learning dialogues: windows, trend and activity budget.
  - The 5 pairs are probed right after teaching and again after the 300 dialogues: is the taught answer still top-driven?

**Excitability comparison** (reduced). Base C, medium preset. The online birth calibration also moves the L1 internal threshold θ_1 until internally lit copies ≈ k × chain-lit copies, for k ∈ {1, 2}. g_loop is raised only if θ_1 reaches its floor; L2–L4 stay count-preserving to L1's total. Reported:
- a1 counts;
- the share of internally lit copies whose feature was lit externally in the last 1–12 ticks (held context) vs others;
- the same decoding / remembered-until / motion / input-influence measures (learning off);
- L4 syllable specificity (Jaccard between syllable codes);
- a1L over 300 dialogues;
- the largest k the medium preset holds with every layer ≤ 50 % on.

## Addendum 11-02 (b, a): b_block sweep and band-overlap check, registered before running

**b_block sweep** (coordinator, inside §二.7: "没把握时结果是不说，而不是说错").
- Motor parameter b_block ∈ {0.8, 1.2, 1.6}, with g_ret_force 1.6.
- A sharp syllable keeps passing (no ladder recruited). A full-ladder mixture is blocked once b_block ≳ 1.0.
- Garble and silent are reported separately.
- M2 (already running) is the b_block 0.8 point. Later (b) runs use the sweep.

**Band-overlap check** (birth only, candidate C base, medium preset).
- First, trace syllable overlap back through the chain: Jaccard between codes of different syllables (same voice), and of the same syllable in two voices.
  - Measured in sound A1–A4 (chained units), PFC L1ext, L2, L3 and L4 (`probes.layer_overlap`).
  - Each population is read in its own latency window after the onset.
- Then vary the chain band's upper cut, keeping medium's depth-relative low cuts:
  - A1 upper cut ∈ {0.9, 0.6, 0.3} with A2 at 0.9;
  - A1 = A2 upper cut ∈ {0.6, 0.3}.
- Reported: syllable overlap at L1 and L4, L1 silence/speech counts (a1 floor: silence ≥ 0.2 × speech), a1, a3 and lag 12.
- Reason: the designer says near-input often-on units carry no features (ARCH §二.3; the band is tunable per area).

**Order:** fast-PFC test (main), band-overlap check, excitability k ∈ {1, 2}, large PFC.

## Addendum 11-02 (b): motor collapse wiring MC (hand-wiring register §4 entry), registered before running

Designer (ARCH §二.7, d15527d): "手动接线只是给他一个说正确话的倾向……可以让说的倾向自己坍缩成正确的肌肉动作。这需要我们提前给运动区肌肉神经元布置一些连接，让其总是能坍缩到正确的肌肉激活顺序。"

**§4 register entry MC** (generation 0, motor area). The links sit in the acquired, decaying part (γ_motor), learn by the same rule, and are inherited.
- **(a) Pattern completion:** excitatory force → force links between the force neurons of different muscles of the same syllable: W(i → j) = Σ_syllables [muscle(i), muscle(j) ∈ syllable] / n_syllables(muscle(i)). Muscles shared by many syllables pull proportionally less.
  - The table goes through the usual two-layer weights. One gain is set so that 2 fully active muscles of a syllable drive each of its other 2 muscles' force neurons at `mc_rel` = 1.2 × their release level (threshold + tonic blocking), median over syllables.
  - Timing: force(t − 1) → force(t), so a pattern settles over a few ticks.
- **(b) Order:** skipped. In the X1c body a syllable is one simultaneous 4-muscle set held 3 ticks, with no within-syllable order.
- **(c) Competition:** the existing blocking ladder, recruited by the pooled drive, is the global inhibition. Pattern excitation plus global inhibition gives winner-take-all among syllable patterns.
  - No gate, no syllable-detector neurons, b_block swept over {0.8, 1.2, 1.6}.
  - Inhibitory neurons between rival patterns would be added only if needed (none yet).
- m = 8 force neurons per muscle as main, m = 32 as a check.

**Collapse test first** (no PFC involved). A diagnostic hook injects drive into the force neurons for 3 ticks (in units of the return drive), then the system runs 6 free ticks. Conditions:
- clean x;
- 2 of x's 4 muscles;
- 60/40 and 50/50 mixtures of x and y, with y sharing 0, 1 or 2 muscles;
- x plus noise (uniform 0–0.3 on all force neurons).

Reported: the share that collapses to the stronger syllable, to the weaker, garble, silent; ticks until the decoded mouth settles; garble on the onset ticks reported separately.

**Then (b) with MC** on top of G1P and of the learned arms F1 / F3, b1–b7 as before.

**Base change (coordinator 11-02, before any fast / band / MC run).**
- Candidate B (P1, reciprocal, g_loop 1.0, w44 0, learned gains 0.1, η 0.02, η_H 0.3) is the tentative base for the fast-PFC test, the band check and (b) with MC.
- Fast arms relative to B: F0 η 0.02 / γ 0.99995; F5 η 0.1 / γ 0.999; F20 η 0.4 / γ 0.99.
- Choose seeds 2–3 run later on the final configuration, as the confirmation before freezing.

**Fast test, trajectory added** (designer: "有没有可能是训练的不够才显得记不住问题").
- For every arm, the a3 probe runs at 0, 100, 300 and 1,000 learning dialogues: lags 0 / 6 / 12 / 24 / 48, filler-given-frame, motion J.
- The probe runs with learning off for F0, and both on and off for F5 / F20.
- It runs in one process (`run_fast.py --mode traj`), so no checkpoint has to survive between jobs.
- a3 at birth is not blocking (§6a: proceed with the best and flag); d2 re-measures the hold in (c).

## Revision 11-02: a3 demoted to a diagnostic; the memory criterion is functional and whole-brain

Designer: "停顿12拍还认得出问题这个测试本就有点问题，你是测试他的记忆能力吗？如果是记忆能力，海马体不是能记住吗？"

- **a3** (PFC hold of a question through a pause, learning off) is now a DIAGNOSTIC of the PFC loop (§二.4 "回路保持想法"). It is reported but is no pass criterion anywhere (stage a, d2). Reason: with learning off the hippocampus stores nothing, and in the architecture remembering across time is the hippocampus's job (X6b: Q → A across the pause 0.94–1.00).
- **Memory criterion, learning on:**
  - (i) One-shot Q → A across the pause, as in the fast test. On rehearing Q, is A's code lit?
    - Reported per route: the PFC route (internal drive from L4 onto A's copies in L1) and the hippocampal route (A's copies lit internally in the hippocampal copy layer H1, whose returns re-light A's sound units).
    - The rehear probe also runs with learning on; the full state is restored between trials.
  - (ii) In (c), the answer after the pause remains the real test.
- **Fast test:** the a3 numbers are also reported with every table learning (hippocampus included) during the probe, with the full state restored between trials. The question is decoded from PFC L1int ∪ L4 and from Hs.

## Reprioritisation 11-02 (designer: "你先让模型能有规律的听到声音作出正确回应吧，保持之类的不用测试的")

- **Dropped as separate tests:** hold / a3 probes (a3 stays only as a number inside existing probes, never a criterion), the excitability k test, the context probes, and the standalone fast-PFC hold test.
  - The fast-PFC seed-0 job had already started and cannot be cancelled (the runner has no cancel), so it finishes and is reported as data only.
- **Focus: hearing speech and responding correctly.**
  1. Speech basics (b): the MC collapse test, then (b) with MC on top of G1P and of the learned arms F1 / F3, with the b_block sweep.
  2. Then (c) per §二.11 on base B, with the registered pass criteria (above the per-frame baseline).
     - Arms: fast PFC F5 (η × 5, γ 0.999), judged by answers; large PFC only if (c) fails at medium.
     - The one-shot Q → A probe at each checkpoint, split by route; answer accuracy at every checkpoint.
  3. The band-overlap check stays as a cheap birth-only diagnostic of speech specificity.
  4. Choose seeds 2–3 run on the final configuration as confirmation before freezing for (c).

## Addendum 11-02 (c): MC revised to pattern pools (hand-wiring register §4, replaces the pairwise MC)

**Why.** The pairwise force → force MC ran away: all 24 muscles came on and stayed on.
- A numpy copy of the motor area (it reproduces the no-MC GPU row exactly) showed that pairwise links cannot tell 34 overlapping 4-of-24 muscle sets apart.
- (i) MC-inh (muscle-level inhibition between rarely co-occurring muscles) and (ii) tagged force sub-populations did not fix it either.
- The collapse test's input was also wrong. A force neuron's return is 0/1 (its PFC copy is lit or not), so a mixture means shares of lit force neurons, not weaker drive. The GPU collapse test now uses 0/1 input.

**MC entry (generation 0, motor area, acquired part, γ_motor, the one rule, inherited).** Per syllable two conjunction groups of mc_nI = 16 neurons, a gen-0 motor synergy. They are not program neurons and run no sequence (X1c syllables are simultaneous).
- Each group neuron samples one force neuron from each of the syllable's 4 muscles (lineage seed) and fires in the same tick when all its sampled force neurons fire. These are fast interneurons; the threshold is 7/8 of its full input drive.
- The excitatory group projects to the syllable's own force neurons. One fully active group gives mc_e = 3 × the median release.
- The inhibitory group projects to the force neurons of the 20 muscles outside the syllable, with mc_i = 4 × the median release. Its links stay inhibitory (≥ 0, subtracted).
- Two-layer weights as everywhere. Gains are calibrated once at birth on one fully active group.
- The groups are driven only by the tendency of the same tick, so the motor area alone is below release, nothing holds after the tendency stops, and completion only sharpens (option (a)).
- Rejected in numpy (they lost 60/40 or 90/70): the within-muscle tie fix (V1); and P2, return drive below release with the pools lifting it (60/40 fell to 0–0.40).

**Numpy table (0/1 input, 34 trials per condition).**

| | clean | 60/40 → stronger | 90/70 → stronger | 80/80 one syllable / garble | 50/50 garble | 2 of 4 | after input stops |
|---|---|---|---|---|---|---|---|
| no MC | 1.00 | 1.00 | 0 | 0 / 1.00 | 1.00 | garble | 0 ticks |
| pools, m = 8 | 1.00 | 1.00 | 0.86 | 0.62 / 0.33 | 0.84 | garble | 0 ticks |
| pools, m = 32 | 1.00 | 0.97 | 0.96 | 0.52 / 0.38 | 0.85 | garble | 0 ticks |

Clean onset garble is 0 everywhere. Ties stay garble (not met: "50/50 mostly silent").

**Next.** GPU collapse test (no MC vs pools, m 8 / 32), then (b) with pools on G1P and F1 / F3. x10_tend reports the mixtures that actually arrive.

## Addendum 11-02 (d): the real tendency carries no syllable; overlap trace first

**x10_tend (base B, m 8, seed 0).** When x is lit in PFC L1 (inner hearing), the returns that reach the force neurons are:
- G1P: 81 % of all force neurons, 23 of 24 muscles over half; x's share 0.87, best rival's 0.97; x ranks first 0.09 (chance 0.03);
- F3: 11 % lit, x first 0.03.

So the bottleneck is upstream of the motor area. (b) with MC is held until a configuration gives a tendency in which x ranks first for ≥ 0.5 of the syllables.

**Overlap trace (birth only, `run_fast.py --mode band`, `probes2.overlap_trace`).**
- Layers: A1–A4 (chained units), L1ext, L2, L3, L4, each read at its latency, per slice and pooled over the syllable's 3 slices.
- Jaccard for different syllables (same voice), the same syllable in two voices, and syllable vs silence; also evoked codes (on in the syllable, off in silence).
- L4 under inner hearing vs heard.
- Variants (ARCH §二.3: near-input often-on units carry no features; deep units are mostly feature units; band and per-layer rates tunable per area):
  - medium (A1 ≤ 0.9);
  - A1 ≤ 0.6 and A1 ≤ 0.3;
  - A1 + A2 ≤ 0.6 and A1 + A2 ≤ 0.3;
  - sound rates × 0.5.
- Each with a1 (including the silence floor) and the a2 numbers.

Then x10_tend2: the tendency tick by tick, plus spontaneous returns.

## Addendum 11-02 (e): P1 (band) and P2 (G1P from heard ∩ inner), approved by the coordinator

**Overlap record.** On medium at birth, the L4 different-syllable Jaccard is 0.32. Other layers, pooled over the 3 slices:

| A1 | A2 | A3 | A4 | L1ext | L2 | L3 | L4 |
|---|---|---|---|---|---|---|---|
| 0.41 | 0.55 | 0.34 | 0.30 | 0.53 | 0.36 | 0.31 | 0.32 |

The "84 %" quoted earlier comes from diag_g1p, a different measurement:
- base C (w44 0 + R3);
- L4 codes = units on in ≥ half of ticks 4–9 after onset, mean over two adult voices;
- the codes include the held pre-onset state, which dominates L4 (the syllable vs pre-onset Jaccard was also high).

The two numbers are not comparable.

**P1 (ARCH §二.3: near-input often-on units carry no features; the band is tunable per area).** Chain band upper cut 0.6 on A1 and A2 (`band_cut_a12=0.6`; lower cuts unchanged).
- At birth the L4 different-syllable Jaccard drops from 0.32 to 0.14, and L4 vs silence from 0.27 to 0.09.
- Inner x vs heard x / heard y: 0.18 / 0.06 (medium 0.31 / 0.21).
- a1 failed only on one Hs-empty tick. It is re-measured on P1 with a1 (50 dialogues) and a 300-dialogue a1L (seed 0). If the empty tick recurs, it is fixed through the existing online Hs threshold calibration.

**P2 (hand-wiring register §4, G1P revised; designer: "前额叶最后一层代表某个声音的神经元").**
- The L4 units that represent x are those on, with margin, both when x is heard (four adult voices) and when the PFC thinks x (inner hearing: x's sound code lit in PFC L1 for 3 ticks, L4 read 3–9 ticks later); `g1p_src='both'`, margin 0.1.
- The gain is set from the inner state: thinking x drives x's own force copies at g1p_rel × θ1 (median).
- Controls: heard-only (old) and inner-only.

**Check.** x10_tend on P1 + P2. Target: x ranked first for ≥ 0.5 of syllables, with a moderate total lit share. F3 runs on P1 too. Only then (b) with MC pools.

**Result x10_P1g / x10_P1f (seed 0; ticks 10–16 = x thought; "nothing forced" = the same ticks with no inner hearing).**

| run | x share / best rival / rivals ≥ half / x first / total lit | nothing forced: total lit | units per syllable (min / median) |
|---|---|---|---|
| base B, G1P heard (margin 0.1) | 0.87 / 0.96 / 32.9 / 0.05 / 0.79 | 0.68 | — |
| P1, G1P heard | 0.83 / 0.91 / 28.8 / 0.18 / 0.67 | 0.00 | 5 / 22 |
| P1, P2 (heard ∩ inner) | 0.66 / 0.72 / 8.8 / 0.39 / 0.35 | 0.00 | 0 / 3 |
| **P1, G1P inner** | 0.68 / 0.51 / 3.0 / **0.83** (1.00 on ticks 10–13) / 0.25 | 0.02 | 15 / 42 |
| P1, F3 (learned) | 0.08 / 0.19 / 0 / 0.02 / 0.09 | 0.08 | — |

- P1 alone removes the push with nothing thought (0.68 → 0.00).
- P1 passes a1 (Hs min 3, no empty tick) and the 300-dialogue a1L.
- Heard ∩ inner leaves too few units (32 of 34 syllables have < 10).
- The inner-only selection meets the target: the units that represent x when the PFC thinks x.
- (b) with MC pools runs on P1 + G1P inner (main), with no-MC and P1 + P2 (both) as comparisons.

**Decision 11-02 (coordinator): new base for speech = P1 + G1P inner** (`band_cut_a12=0.6`, `g1p_src='inner'`, margin 0.1, gain from the inner drive). Basis: ARCH §二.11.1 "模型心里想到一个字的声音，就能本能地说出来"; thinking x is when G1P must fire.
- P2 (heard ∩ inner) was not kept: 32 of 34 syllables had < 10 units (median 3), and x ranked first only 0.39.
- Evoked-only restriction (off in ordinary listening, 0.1 / 0.2) changed little: inner + evoked gives x first 0.82, nothing-thought 0.00. Not adopted.
- Watch b4 echo.

**F3 diagnostic (x10_dF3, `diag_f3.py`).** At the forced muscle onsets of the F3 scaffold on P1:
- Jaccard of L4 with x's inner code, x's heard code, other syllables' inner codes and the ordinary held state, at t0 − 1, t0 and t0 + 1;
- the learned L4 → force-copy links read from x's inner code and from its heard code (true / other, margin, top-4 exact).

Hypothesis: the scaffold teaches from the heard / held state, but speech is triggered by the inner state.

**Learned speech route (F3 / F4): tested, not working in X10 (coordinator 11-02).**
- F4 (one forced x, timed 6 ticks after the adult's x, inside the thought window) did no better than F3. Learned link true/other 1.12, margin 0.50, top-4 0; x first in the tendency 0.09.
- Cause: L4 at speech onsets is large (~25 % on) and mostly shared across syllables, so the one rule learns the shared part. G1P avoids this through its margin selection.
- G1P-inner is the gen-0 speech link: registered, decaying, learning on top. No PFC change now.

**Note for (c).** Answers rely on the same mechanism: learned L4(t) → L1(t + 1) links read from a mostly shared, ~25 %-on L4 (cf. X1e: "同时亮的太多，学到的连接把所有先后顺序平均掉了").
- Each checkpoint reports the learned Q → A link per question (`probes2.qa_link_stats`): L4 after the question drives L41 into L1. Reported: the answer's code drive vs other codes (ratio, ranked first) and the raw weight ratio, for seen and held-out pairs.
- If (c) fails with ratios near 1, the next lever is L4 sparsity / specificity (ARCH §四.20), and F4 is retested on that PFC.
