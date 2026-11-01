# Revision brief: ARCH_v3_core → revision 2

## Inputs
1. **Your document:** `docs/v3design/synth/ARCH_v3_core.md` (the working copy is in `arch/v2/v3design/synth/`).
2. **The four adversarial reviews.** All four say "fix, then build". Each is `report.md` under `docs/v3design/review/<role>/`, with toys and outputs alongside:

   | reviewer | FATAL | SERIOUS (main) |
   |---|---|---|
   | `develop/` | "0 errors after maturity" cannot be met with constant babbling outside M4 | P5b is contiguity, not contingency; W1/W4 are timing rules moved into the world (caregiver toy: realistic move-on and re-ask → 0.01); maturity in the wrong currency; μ starts replay inside answer windows; crowds; #17 values under the TD form (γ horizon 1.5 s); self/other disjoint units vs multi-syllable lines; experience per lineage |
   | `ml/` | speech release cannot stop (G clipped at 0, so urgency always releases); a per-generation world seed changes physics, so inherited weights are invalid | MIX about 20× sparser than assumed; pure-neuron CA3 about 1,000 episodes, flat in s; merge prerequisites fail (assembly agreement 0.2–0.66); gating cannot generalise; scaling inert in s; stability scoped to readouts only |
   | `neuro/` | replay (low μ) collides with the held anchored context (live-anchor replay: 0.99 → 0.00) | CA3 size; P5b non-contingent and harvestable by timing; M4 credit assignment and ε; M2 already blocks corrections (0 of 20 corrected); ungated LATEST loading; P4 wired to world codes and an innate "my name"; binary-neuron forms of graded holding and fixed k |
   | `skeptic/` | O1 at tick level collapses (0.00 by 300 untutored); only event-level learning is stable, and then "when" cannot come from GEN | value loop untested, P5b rewards late answers under the probe timing; the maturity rule cannot be met; compute 4–7× too optimistic and the GPU claim 430–860× off the measured gpuport numbers; the representation swap comes last but merging needs it first; about 16 hidden task- or world-specific items; §6 numbers do not use the M2/M4 rules as defined; knob count honest at about 146 → 100, not 265 → 30 |

## The designer's own specification (binding; it overrides the code, and it overrides your document where they differ)
These were stated by the designer in conversation (2026-10-29 to 2026-11-01). Today's code deviates from several of them, and the code is not the spec.

1. **Neurons are binary** (on/off). Representation is distributed; one character is never one neuron. Feature neurons are those consistently active for an input, and they emerge.
2. **Signal strength = how many neurons an input activates, not a fixed count.** A fixed active count leaves no strong or weak signals. A global threshold sets how much is active, as in the designer's char-neuron network (`gh-ref/char-neuron-network`, read only): a quantile-tracked global threshold keeps about 20–30 units active, but the count varies. The input layer stays about half on (thermometer pairs); the variable count lives in the layers above.
3. **Every network starts full of random connections**, and training modifies them; nothing grows from zero. The PFC is full of mutually exciting connections, and the model may think a lot before speaking.
4. **Chain neurons.** Each area's neurons send dedicated 1:1 copies (chain neurons) to the PFC and to the hippocampus. PFC units are chain-delivered followers of sensory neurons. Always-on, featureless neurons, especially near the input, simply get no chain neuron.
5. **The hippocampus** is its own neuron population. It is connected like the PFC through chain neurons and stores by Hebbian connections that grow and decay faster than the cortex's. It holds short-term memory only and does no logical thinking. There is no "one memory = one new neuron/cell". The designer explicitly replaced an older "time-neuron" hippocampus scheme with this. Long-term memory is the cortex simply learning, slowly, helped by recall.
6. **One global weight-decay mechanism** for clean-up: every T ticks the plastic edges are multiplied by γ, and edges below threshold are deleted; innate edges are untouched. Regions differ only in γ. Do not invent separate clean-up mechanisms.
   - Note: v2's tune found every decay value harmful in that architecture. Reconcile this with #22.
7. **Structural plasticity, on a very long timescale.** Neurons split (parents in the 40–60 % usage band) or are deleted (never active or always on), and their chain neurons and followers follow. Splitting happens rarely near the input and more in deeper layers.
8. **Output only through muscles.** The motor area compares muscle-control neurons against inhibitory neurons: act or not. Most of the time inhibition exceeds the muscle drive, so the muscles stay relaxed. Areas drive the motor area. Instinct and reason compete or cooperate.
9. **Self vs other needs no tag.** Own voice and others' voices activate similar but different neurons: the overlap is the content, and the difference is the speaker.
10. **Do not add rules like "event boundaries"** (said about memory).
    - Wherever v3 relies on onset transients, LATEST loading or event-level learning, say whether it is an emergent property of M1 (adaptation) or a rule. If it is a rule, find the generic form or justify it.
11. **"Later-active neurons inhibit earlier-active ones"** was only a temporary fix. Refractory periods are not to be used.
12. **IDEAS #12–#23** in `docs/IDEAS.en.md` are binding:
    - #12: scale the whole model;
    - #14: mishearing repair is learned;
    - #19: reproduction and merging;
    - #20: errors are allowed in childhood, and 0 is required after maturity;
    - #21: the final product is pure neurons;
    - #22: strong, used connections resist change. The goal: learned once, kept for life. The exact form is ours to test. The ML and neuro reviewers argue that the plain |w| × use form only delays forgetting and blocks corrections; ML recommends counting *confirmed* use, and neuro recommends a two-component early/late synapse with δ-gated capture. Pick and justify a form that keeps the designer's goal and keeps corrections possible.
    - #23: a wide, distractor-robust thinking range; spontaneous recall when alone. **The hippocampus stores everything**: there are no useless memories, and repeated input only re-strengthens the same connections, with saturating growth.

## A question for the designer that the revision must prepare
v3's key structure is the context anchored on "latest event × age", with log time cells. The designer previously dropped a "time-neuron" scheme (it was for the hippocampus). Propose a form of "age" that needs no dedicated time neurons, if one works. For example, the latest event's assembly fades, and the number of its neurons still on encodes how long ago it happened (spec 2: strength = number of active neurons). Alternatively, time cells may emerge from recurrent dynamics. Say whether the toys still pass with that form. If dedicated time cells are truly needed, say so plainly, so that the coordinator can ask the designer.

## What to produce
**Revision 2 of the document**, returned inline in your final reply (the harness blocks report writes). Also save it as `arch/v2/v3design/synth/ARCH_v3_core_r2.md`.

It must contain:
1. **A change log answering every FATAL and SERIOUS item** of all four reviews: accepted (and how), modified, or rejected (and why).
2. **A conformance table:** each of the 12 designer-spec points above against the revised design. Mark it conforms / deviates (why) / open question for the designer.
3. **The revised core, kept as small as possible.** Never fix a review point with a task-specific rule.
4. **Corrected numbers.** Use the rules as defined (n_use, floors, the M4 formula). Use honest compute: take the skeptic's measured tick costs and gpuport's measured GPU throughput. Use the honest knob count.
5. **A revised migration ladder.** Move the representation swap (assemblies, LATEST units, the designer's chain-neuron hippocampus) early, before any merging. Put the CA3/hippocampus, merge and compute tests near the front.
6. **Revised first experiments.** Use X1+ with the full release/value loop and both O1 readings, plus the reviewers' extra arms (move-on and re-ask caregivers, multi-syllable answers, replay reading, distractors). Also: the hippocampus module test, the merge test and the tick benchmark. For each, give the harness, arms, prediction, falsifier and compute.
7. **The remaining open risks**, ranked.
8. **`summary_zh`:** about 15 lines for the designer, in plain Chinese. Include the time-neuron question if it stays open.

## Rules
- Design and toys only: no training runs and no server jobs. Small local toys are fine: `free -m` first, ≤ 1 python process under 700 MB, in `arch/v2/v3design/synth/`.
- An experiment agent (X2, folder `arch/v2/x2/`) is testing v2.2's collapse attribution, the weight-stability idea (#22) and a wider thought (#23) on the real code. Do not touch its folder. Its results will be sent to you if they arrive in time.
