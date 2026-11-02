# State of the project for the architecture-design panel (2026-10-30, late)

Read with: FRAMEWORK.md (the designer's own framework ideas, section 一 is authoritative about the designer's intent;
sections 二/三 are partly outdated — this file supersedes them), 00_synthesis.md (synthesis of five deep literature
reviews 01–05, which motivated many recent mechanisms), and the code at /root/work/concept-brain (brain/net.py,
brain/areas.py, brain/brain.py, brain/net3d.py, talk.py, world/voice.py). The code is the ground truth for what exists.

## 0. Why this panel exists (the designer's request, translated)
"Like AI solving very hard math problems: let many agents work together to settle the architecture first, and only
train once we are sure it is feasible. The current loop — train, find it doesn't work, change parameters, train again —
is too inefficient. First establish a logically feasible architecture, discuss it fully, and train only with enough
confidence. My original architecture is probably roughly right, but some parts may be wrong."
Also: "You don't have to strictly follow all my rules — what matters is reaching the final goal."
So: the designer's framework is the starting point and its spirit should be kept where it works, but any rule may be
changed when the logic or evidence says so. Prefer the simplest mechanism that provably works.

## 1. Final goal and the functional specification
Final goal: an AI architecture that learns in real time (continually, from each experience as it happens, no
backprop/gradients, local learning, binary neurons). The first complete instance: a "brain" that hears speech through
an ear, thinks, and answers only by moving muscles, trained on dialogues (the other says a line, the model answers).

Required behaviours (what "works" means):
- B1 Listening: while the other speaks, the mouth stays still (no interrupting, no echoing).
- B2 Holding what was heard: after the other's line ends, some internal state identifies what was heard and stays
  until the answer is under way (a working-memory "thought").
- B3 Turn-taking: without the teacher, the model starts speaking a few ticks after the other stops.
- B4 Content: what it says is related to what it heard (learned from the dialogue pairs): the first syllable is chosen
  by the held thought; later syllables follow in a learned order.
- B5 Articulation: each syllable comes out cleanly (one char per 3 ticks, not the garble "~" that results when muscles
  of several chars fire together).
- B6 Stopping: it stops at the end of its answer and does not babble on.
- B7 Stability and real-time learning: learning happens online during every dialogue; learned weights do not run away;
  no runaway activity; new learning does not wipe out old abilities wholesale.
- B8 (later) Generalisation: similar heard lines lead to similar answers, not only memorised pairs.
Realistic first milestone: after a few hundred to a few thousand dialogues the model opens its mouth after most heard
lines, says clean chars, and some of those chars belong to the expected answer (above chance). Exact replies to the
noisy corpus are not expected early.

## 2. World and training protocol (fixed unless there is a strong reason)
- Corpus: 32,971 mixed Chinese dialogues (short social-media style), 3,970 distinct chars. Checks use 20 fixed probe
  dialogues.
- Sound: each char = sparse 8-band × 6-slice "spectrogram" (2 loud bands per slice); a char lasts SOUND=3 ticks; the
  ear sees a sliding window; no gap between chars inside a sentence; PAUSE=4 ticks between dialogues. The model's own
  voice is the same spectrogram shifted one band (similar but distinguishable) and is heard by its own ear.
- Muscles: 32 muscles; each char needs a specific set of muscles (world/voice.py `need`); the mouth output per tick is
  decoded from muscle forces; several chars' muscles at once → "~" (garble).
- Training (talk.py main): for each dialogue, the other's line: listen(line, pause=REPLY_GAP=5); the model's line:
  say(reply) — teacher-forced: an external push to the start unit (go) on the reply's first tick, and cue() pushes the
  disinhibitory neurons of the right muscles on each syllable's first tick; everything else (holding the syllable,
  force, switching, stopping) is the motor area's own structure. Learning is on all the time.
- Test (check): reply(prompt) = listen(prompt, pause=0) then 60 free ticks with no teacher.
- Compute: one V100 32 GB; ~35 min per 500 dialogues with 3–4 runs sharing the GPU (a 3000-dialogue run ≈ 3–4 h).

## 3. Current architecture (code = commit 004031a + one uncommitted path, think→motor)
Core (net.py): binary Regions, fixed active fraction (top-k) or threshold; Paths with one learning rule:
  src on at t−1 & dst rising at t → w += eta/(1+(w−w0)/w_ref) × (source reserve x);  "back": src on at t & dst on at
  t−1 → w −= eta (floor w0) — so a source that stays on while the target turns on cancels out (designer's rule);
  PathCfg.back=False disables the back term (used only for →go and think→motor). Global decay every 100 ticks
  (learned part × gamma) and per-destination learned-input budget. Synchronous update: all currents from the
  previous tick's states, then all parts fire, then learning.
Mechanisms added from the literature synthesis (24e28c2 and later):
- Output reserve (short-term depression) on ear / motor-cortex / muscle outputs: x ← x(1−0.5) when firing, recovers
  with tau 6; currents and learning increments are scaled by x.
- Set point (homeostasis) in PFC receiving and thinking layers: penalty for neurons on more than the target fraction.
- Thought feedback: fixed reciprocal thinking→receiving projection (w_fb 0.1) + learned think→pfc.
- Thalamus: 2000 threshold units, each tied to 5 thinking neurons; on if ≥2 of its 5 are on; latches itself; released
  (inhibited, weight 10) by the "fresh" unit.
- "Fresh" unit ("someone is talking"): threshold on the reserve-weighted sum of active ear-cortex neurons; brakes go
  (weight 5) and the thalamus.
- Motor area: per muscle 3 force-level neurons M, 1 inhibitory I (tonic), 1 disinhibitory D; content-free start unit
  go; syllable beat = 3 ticks; at a boundary (go fires while not speaking, or end of a beat while speaking) D's with
  current + go_w(1.2) ≥ dis_thr(2.0) are released (max 4, strongest first) and held for the beat with a force ramp;
  speaking stops when no D passes at a boundary.
- Learned →go from thinking layer and ear (forward-only, budget 2 each); learned think→motor(D) (forward-only,
  budget 1.0, uncommitted, untested).
- Variable sensory count (designer's newest idea, section 四 of FRAMEWORK.md): net3d.vary_from — from layer 2 on,
  the number firing = cap × F/(F + sigma·alive), F = count of neurons whose reserve-weighted feed-forward drive ≥ 1;
  floor 15%; learned/return currents only choose which. Off by default; tested as an option.
- Other: hippocampus = mirror layer with strong 1:1 chains (3.0), fast learning/forgetting; PFC receiving layer =
  mirror layer with weak chains (0.3 × reserve) + own internal connections; returns from PFC/hippocampus to ear and
  motor cortex via relays (open only for concentrated activity); PFC/hippocampus motor-mirror neurons return 1:1 to
  the muscle area (0.3 each); learned ear→mctx, mctx→ear, mctx→motor, motor internal (from M only).

## 4. Evidence so far (measured; use it to test any design)
Decoding = "after a different preceding line, can the state after the pause identify which of 20 sentences was heard"
(Jaccard nearest neighbour; chance 0.05).
1. Ear cortex itself identifies the heard sentence at its end (1.00) and for ~4 silent ticks after (afterimage of
   which neurons were used, via reserve).
2. Hippocampus (strong chains) identifies it after the pause: 0.95–1.00, untrained and throughout training with the
   reserve/set point; without them it decayed to 0.10–0.30 after 500 dialogues.
3. PFC receiving layer identifies it poorly even at the sentence end: 0.15 with thought feedback w_fb=0.1; 0.60 with
   w_fb=0; same-sentence and different-sentence similarity are about equal (a large shared core, plus feedback/own
   recurrence that reflects history, dominate the weak chain input).
4. Thinking layer: untrained ≈ chance in every variant. The old top-k thalamus loop made it a content-free slow
   "clock" (high hold, encodes neither sentence nor context). With the fresh-gated threshold thalamus (no latch,
   reply gap 3): 0.90 after 500 dialogues, 0.70 after 1000 — learning made it content-specific. With the variable
   sensory count: 0.35 at 500.
5. All holding loops (think↔thalamus, think↔PFC) are 2-tick loops; in synchronous updates the thinking layer showed a
   period-2 alternation (jaccard with end state 0.06, 0.82, 0.06, 0.47 …). The thalamus latch removed it (0.7–0.98).
6. The fresh unit stays on ~2 ticks after the other's speech ends (the offset itself is "fresh" input). Before the fix,
   the teacher's go at reply onset was braked, so the model never actually spoke in training and all motor learning was
   ~0. Now the teacher push overrides the brake. With the latch, the thalamus engages only ~2–3 ticks after the end,
   and in that window the thinking layer drifts.
7. The learning rule cannot teach "held thought → target turns on": a held source is on at t−1 and at t+1 around the
   target's onset, so forward and back cancel, and reserve scaling makes forward < back (net negative). Measured:
   think→go stayed ≈0 (mean 0.0045, max 0.06 after 1000 dialogues).
8. First-syllable content: before think→motor, the only content currents into D at the go boundary were the fixed
   PFC/hippocampus motor-mirror returns (≤0.3+0.3=0.6), while ≥0.8 is needed (dis_thr − go_w). mctx→motor learns
   mctx(syllable k) → D(syllable k+1), i.e. not the first syllable.
9. Guided speech (teacher cues only) is 100% correct untrained but 61–83% after 500–1000 dialogues: learned currents
   (returns, mctx→motor, motor internal) interfere with the cued syllables.
10. No run has ever opened its mouth on its own (latest: 1000 dialogues). Mouth while listening: always 0 moves.
11. Learned weights with budgets are stable (no runaway). Featureless (always-on) ear neurons are muted from all paths.

## 5. Constraints for the design
- Binary neurons, local learning only (no gradients), learning online during every dialogue (real time).
- Must run on one V100 at ≥ ~500 dialogues per ~30 min (current sizes: ear cortex ~20k, motor cortex ~3.8k,
  PFC receiving / hippocampus ~24k each, thinking layer 10k, thalamus 2k, 160 muscle-area neurons).
- Keep the designer's framework where it works (sensory 3D nets; hippocampus = fast chain region; PFC as thinking;
  order-based learning; global decay; slow split/delete of neurons; muscles-only output; one simple mechanism
  rather than stacked fixes). Anything may change if the logic requires it — say why.
- Local machine for checks: numpy 1.26, 4 CPUs, no torch, no GPU. Toy simulations must be small (seconds–minutes).
