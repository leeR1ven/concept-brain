# Our model (brief for research agents)

A brain-inspired model that learns in real time and can only output through muscles (a simulated mouth that speaks
Chinese characters). It is trained on dialogue: someone says a line, the model replies with its own mouth.
Rules are deliberately few and simple. Everything is binary neurons (on/off) and connections.

Structure:
- Input: every input value is a thermometer code of on/off pairs, exactly one of each pair is on, so the input layer is
  always exactly half active. Silence is also an input pattern.
- Sensory area (hearing): fixed random local 3D network (each neuron = its own position in the layer below, weight 1,
  plus 3 random neighbours with tiny weights, plus a bias). Each layer keeps a FIXED fraction active (top-k): learned or
  top-down input can only change WHICH neurons win, not HOW MANY. Neurons of the area (except the first layer) also have
  plastic connections among themselves (sequence memory: hear the start, the rest comes out = "representation collapse").
- Motor area: a "motor cortex" (same 3D network applied to the current muscle state -> combination features) plus
  muscle neurons: per muscle 3 force-level neurons, 1 inhibitory neuron (tonically on, so muscles are relaxed by default),
  1 disinhibitory neuron (suppresses its inhibitory neuron). A muscle moves only if its disinhibitory neuron gets enough
  summed drive AND its force neurons get enough drive. Other regions connect to the motor cortex.
- "Chain" lines: every feature neuron of hearing/motor cortex/muscles has one dedicated line to one neuron in the PFC and
  one in the hippocampus ("mirror" neurons, 1:1).
- PFC: receiving layer (the mirrors, fixed fraction active ~8%), a thinking layer (PFC-own neurons with no chain in and
  no return out; receives from the receiving layer, recurrent within, feeds back to the receiving layer), and a small
  "thalamus" area reciprocally connected (fixed, group-wise) with the thinking layer to hold a thought across a pause.
- Return lines: PFC/hippocampus -> relay neurons -> hearing / motor cortex. A relay opens only when the PFC mirrors of a
  small local column are concentratedly active (focus), so scattered activity does not return. Return adds current
  (modulates), it does not force a neuron on.
- Hippocampus: its own neurons (mirrors), fast learning and fast forgetting, short-term memory only; chain input is
  strong so it records what actually happens; in silence its free slots replay recent content.
- Direct region-to-region links (hearing <-> motor cortex), born random and weak, learnable.
- Learning (everywhere, only speed differs): if A was on at t-1 and B just turned on at t, A->B strengthens (concave,
  no cap); if B was on at t-1 and A is on at t (also when A stays on), A->B weakens back toward its birth weight.
  Global slow weight decay; plus a per-neuron cap on the total learned input from each connection type (heterosynaptic
  normalization) — added because learned weights grew 10x larger than fixed structure and overrode it.

Current open problems (what we most need help with):
1. PFC must HOLD the gist of what it heard across a ~6-tick pause before replying, but also UPDATE when new content comes.
   With our thalamic loop: weak loop -> thought is overwritten by the (silence) input; strong loop -> thought locks and
   ignores new sentences. Inputs enter PFC too strongly ("what you hear does not have to enter the PFC; it depends on
   what the PFC is thinking" — the designer's view).
2. Speech initiation and sequencing: after hearing a line, the model never starts speaking; motor sequences learned at
   1-tick steps run faster than speech (each syllable lasts 3 ticks); several syllables get released together -> noise.
3. Learning stability with ongoing learning on lots of diverse data (now handled by per-neuron caps — is this how brains
   do it? better ways?).
4. Hippocampus: similar inputs interfere; no offline replay to teach the cortex.
5. Shared ("common") features dominate associations (every sound activates many of the same neurons), so learned
   predictions are not specific.

## Latest result (for the synthesis step)
Thinking layer after hearing a sentence, then a 6-tick pause (Jaccard overlap of active sets), 4 runs x 1000 dialogues:
"hold" = overlap(end of sentence, after pause); "across" = overlap between end states of DIFFERENT sentences.
  loop 0.35, chain 0.15: hold 0.39, across 0.50 | loop 0.35, chain 0.3: hold 0.32, across 0.47
  loop 0.42, chain 0.15: hold 0.73, across 0.79 | loop 0.42, chain 0.3: hold 0.64, across 0.72
In every case hold <= across: what the loop keeps is a generic state shared by all sentences, NOT the content of the
sentence just heard. Weak loop -> content lost; strong loop -> same generic state for every sentence. The thinking
layer's own recurrent connections barely learn (mean learned ~0.0001). The model still never starts speaking.
