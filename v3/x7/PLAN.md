# X7 plan (written before any run; not changed after seeing results unless stated below)

Authority: docs/ARCHITECTURE.md (11-01). Items: HARVEST #13, #26, (#28a, #28b if time).
Only the designer's parts. No new module, rule, timer, refractoriness, readout, nothing from §三.

## Part A — item 13: convergent thresholds in the sensory area

### Harness (copied form of local3d `threednet/model.py` + `threshold_activation.py`, and the project's `world/voice.py` sound)
- Sound: project `Voice` physics (F = 8 bands, 10 levels, each syllable 6 time slices; 2 formant peaks per slice at
  levels 5–9, noise floor 0–1). The ear's auditory sheet holds 24 time slices (4 syllables) × 8 bands.
- Input layer A0: thermometer on/off pairs, 10 levels → 20 lines per (band, slice); exactly half on, always (asserted).
- Deeper layers L1–L4: local3d 3-D local net, positions = band × slice, z = (40, 60, 120, 240) neurons per position;
  each neuron reads only nearby units of the layer below (dx, dy ∈ {−1,0,1}, dz ∈ {−2..2} around its own position).
- Threshold (designer form, local3d `threshold_activation`): neuron i fires iff  Σ w·x  >  θ_l · Σ w  (no bias, no
  top-k, no ranking). One θ_l per layer, calibrated once (bisection) so that the mean active fraction on the
  calibration set (one-syllable windows, 40 syllables) is 30 / 15 / 8 / 4 % for L1–L4 (sparser with depth). θ_l then
  rises with depth by itself (reported).
- Weights fixed at birth, no learning (ARCH §二.2 "靠近输入的几层出生时就连好").

### Arms (only fan-in / weight shape differs)
- **dom1 (base, current local3d/net3d form):** 4 inputs: own position weight 1.0 + 3 neighbours at U(0.5,1)·0.01.
- **eq4:** 4 comparable inputs (own position + 3 neighbours), all U(0.5, 1).
- **eq7:** 7 comparable inputs (own position + 6 neighbours), all U(0.5, 1).

### Stimuli (per seed: 40 syllables, Voice seed = net seed)
- silence (all levels 0); one syllable (1 syllable in the 24-slice window, rest silent, random position);
  two syllables; sentence (4 syllables fill the window); two-speaker sentence (max of two sentences, "rich+").
- similar pairs: the same syllable(s) in the other voice vs the own voice (Voice self-deviation, ±1 on ~30 % cells),
  and with a resampled noise floor.

### Metrics, per layer
1. active count for each stimulus class (mean over stimuli); ratio sentence/silence; also the count among
   chain-bearing units only (units on in ≤ 60 % of all stimuli; ARCH §二.3, X1c rate_cut) — secondary.
2. overlap of different sounds: mean Jaccard between different one-syllable windows (different syllable, same slot)
   and between different sentences.
3. similarity preservation: Spearman ρ between input distance (L1 distance of the level grids) and layer Hamming
   distance over all pairs of a 120-stimulus set (40 syllables × {other, own voice, re-noised}, 1-syllable windows)
   — the local3d 0.989 measure; plus J_sim (same syllable other vs own voice) and J_sim / J_diff.
4. recruitment by scattered irrelevant units: on a one-syllable background, switch on k = 1 %, 3 % of L1's active count
   as extra L1 units (random, scattered over the sheet), propagate; count changed units in L2, L3, L4 per added unit.
   Control: the same number of extra units as a coherent local patch (one 3×3 position patch). Reported (c).

### Pass (pre-registered; verdict for eq4 and eq7 separately, against dom1)
- **P1 (counts follow richness, deeper sparser):** at L4 the mean count is ordered silence < 1 syllable < sentence and
  count(sentence) ≥ 2 × count(silence), in every seed; and the mean fraction over all stimulus classes falls
  L1 > L2 > L3 > L4.
- **P2 (different sounds overlap less, similar stay similar):** at L4, J_diff(1-syllable) is lower than dom1's by
  ≥ 0.05 (paired over seeds, same sign in every seed); and Spearman ρ at L4 ≥ 0.95 and ≥ dom1 − 0.03; and
  J_sim/J_diff ≥ dom1's.
- PASS = P1 and P2; PARTIAL = one of them; FAIL = neither. 5 seeds.

## Part B — item 26: does babbling teach "heard sound → the same muscles" under the designer's rule?

### Harness
- X1c body and world (`v3/x1c/world.py` Physics + render, copied unchanged): 34 syllables, each 10 of 150 content
  units; 5 voices × 6 voice units; 24 muscles, each syllable a fixed 4-muscle set; mouth rule clear / mumbled / garble;
  hearing noise (keep 0.95, 0.4 % random units).
- Motor area: X1c `core3._motor` (tonic blocking neurons win by default, divisive gain, release by sharpness
  relative to the max), copied. The birth-given stand-in `Mmap` is REMOVED. The direct link auditory → muscle is a
  learned matrix W (A0 on-units of content + voice, 180 presynaptic; the off-halves are on > 90 % of the time =
  always-on shallow units, no learning, ARCH §二.8).
- No PFC needed for this map (inner hearing = the same content units re-lit by return chains, which drive the muscles
  through the same direct link, as in X1c). State this as a scope limit.
- Babbling = random motor drive on 3–5 random muscles (a babble), which must beat the blocking neurons (ARCH §二.7);
  one babble holds 3 ticks, then rest 3–8 ticks. Quiet room except the child (and, in arm c, the caregiver).
- **The rule (ARCH §二.8 incl. 11-01 held form), applied to W[A→M] every tick, binary units:**
  - B (muscle) rises at t: every A that is on at t−1 (held or not) or fell within 2 ticks, and every A that rises at
    the same tick t, strengthens A→B by η·s(w);
  - A rises at t: every B that was on at t−1 or fell within 2 ticks (B came before A) weakens A→B by η⁻·s(w);
  - a fall weakens nothing. Concave s(w) = max(1/(1+w/w_ref), 0.01); w ≥ 0 (excitatory link); one global decay ×0.998
    every 100 ticks + prune < 1e−3.
  - η⁻/η: primary 1.0 (symmetric: the ARCH gives no asymmetry); sensitivity 0.5 (X1c's convention). Both reported.
- Propagation: two-layer weights (W / √(out·in), mass kept; HARVEST #6), then total mass scaled to the stand-in's
  mass (a gene, same for all arms; as X1d).

### Arms
- **(d) base:** single babbles, own voice heard 1 tick after the muscles (X1c).
- **(a) same tick:** as (d) but the own voice is heard in the same tick as the muscles.
- **(b) reduplicated:** the babble's muscle set is repeated 3× (3 ticks on, 1 tick off) = "bababa"; 1-tick delay.
- **(c) imitation game:** as (d); when the babble is a recognisable syllable s (clear or mumbled) the caregiver
  imitates it with p = 0.5: says s clearly (voice C1, 3 ticks) starting 2 ticks after the babble ends; the child
  then repeats the same muscle set 1–2 ticks after she stops (turn-taking; ASSUMED child behaviour). 1-tick delay.

### Metric (after 500 / 1000 / 2000 babbles; 5 seeds)
- Probe every syllable s (34): fresh motor state, s's content units on for 3 ticks, read the muscles each tick.
  - **inner hearing:** exact content units of s, return gain 1.0 (X1c g_ret_aud);
  - **hearing:** rendered sound of s in voice C1 (noise), direct gain 0.5 (X1c g_dir); innate brake off for the
    probe (the probe measures the map, not turn-taking).
  - outcome per probe from the modal non-silent tick: correct (clear s or mumbled s) / wrong syllable / garble / silent.
- **Primary: share of the 34 syllables said correctly (clear + mumbled).** Secondary: clear-only share, garble share.
- Ceiling reference: the X1c birth stand-in Mmap in the same probe.
- Diagnostic: mean learned W on true pairs (content unit of s → muscle of s) vs other pairs.

### Pass
- An arm passes if its primary share ≥ 0.5 (inner hearing) and beats base (d) by ≥ 0.3 paired, in every seed.
- Base (d) is expected to FAIL (the 1-tick lag makes sound→muscle only weaken). Generation 0 may hand-wire this
  (ARCH §二.12), so a FAIL is acceptable but reported plainly.

## Part C — items 28a, 28b (only if time allows)
- **28a whole-syllable competition:** muscle→muscle links learned by the same rule during babbling (co-rising
  muscles strengthen each other; rarely co-active muscles get learned inhibitory partners, X1c partner form). Arm with
  vs without these links in the motor competition (one extra settling pass). Metric: garble per utterance in the
  inner-hearing probe and in free babbling. Pass: garble falls by ≥ 25 % relative with correct share not lower.
- **28b pace from force neurons rising in turn:** each muscle gets 3 force-grade neurons that rise in turn; the
  syllable-to-syllable link is learned by the same rule from a reduplicated/sequence babble; free running (no timer):
  measure syllable length (ticks) and its spread; compare the two rule forms (held units learn vs only just-changed
  units learn). Pass: syllable length stays within ±1 tick of the taught 3–4 ticks in free running.

## Compute
- Part A and B are small numpy (< 300 MB). Dev checks locally, one process; final multi-seed runs on queue_a800
  (≤ 2 jobs at a time, names x7_…).

## Amendments (stated honestly; written after the seed-0 dev runs of Part B, before the final 5-seed runs)
- B readout gain: the plan scaled the TOTAL learned mass to the stand-in's. In arm (a) the own-voice rows (6 own-voice
  units are on in every babble) took a large share of the mass, so the content rows (the ones the probe uses) became
  weak. Amended main: scale so that the CONTENT rows carry the stand-in's mass. The 'total' form is still reported, and
  an oracle gain sweep (×0.25…×16, best value) is added so a FAIL cannot be blamed on the gain.
- B gain-free map diagnostic added: for each syllable, are the 4 strongest muscles of (content of s → W) exactly its
  4 muscles ("top-4 exact")?
- B sensitivity added: "lawful physics" (the harvest's own smallest test asked for it): a garbled gesture sounds like a
  blend of the syllables it partly forms (each syllable with ≥ 2 of its muscles active gives hit/4 of its content
  units). X1c physics stays the pre-registered main. Babbling runs extended to 5000 (2000 remains the verdict point).

## Part C 28a, details (fixed before running)
- Body X1c, motor area as in B; the sound→muscle map = the X1c stand-in (generation 0 may hand-wire it, §二.12).
- Practice phase, 2000 episodes: the child repeats a random syllable s (inner hearing of s → map → motor) for 3 ticks,
  with motor noise (Gaussian σ 0.3 per muscle per tick) and, with p 0.3, cross-talk from another syllable s' at 0.7.
  Rest 3–8 ticks. During practice, muscle→muscle links learn by the same rule (co-rising strengthen both ways; A rising
  after B weakens A→B), and learned inhibitory partners use X1c's partner form (i rises while j is off → i⊣j grows;
  j rises while i is on → i⊣j shrinks).
- Competition with the links: one settling pass, extra drive g·(E·m − V·m), E and V scaled so the largest row sum is
  1; g = 0.3 (verdict), 0.15 and 0.6 reported.
- Probe: each syllable × 3 draws, 3 ticks, conditions: clean; noise σ 0.3; cross-talk (s + 0.7 s', σ 0.15).
- Metric: garble share per utterance (modal tick), correct share. Pass: garble falls ≥ 25 % relative (noise and
  cross-talk conditions, mean) with correct share not lower, paired over 5 seeds.

## Part C 28b, details (fixed before running)
- Toy (pace.py): per muscle a muscle neuron + 3 force-grade neurons (birth ramp: grade k on after the muscle has been
  on k ticks). Practice: one 4-syllable word (X1c syllables with disjoint muscle sets), 3 ticks per syllable, 300
  repetitions. Links unit→muscle learn by the rule in two forms: held (ARCH 11-01 form, as in Part B) and onset
  ("only just-changed units": A must have risen at t or t−1).
- Free running: the running syllable is held (ASSUMED: by its inner hearing) until the next one starts; the next
  syllable starts when its learned input reaches φ × its input at the taught onset, + Gaussian noise 0.15. φ = 0.6
  (verdict), 0.4 and 0.8 reported. 5 seeds × 200 trials × 3 transitions.
- Metric: syllable length (onset-to-onset ticks), mean, SD, share exactly 3, share within 3 ± 1.
- Pass (per form): mean within 3 ± 0.5 and ≥ 90 % of syllables within 3 ± 1 at φ = 0.6.
