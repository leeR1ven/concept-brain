# The designer's core ideas, and how far each has been implemented and verified

[中文](IDEAS.md) | **English**

This page collects the key ideas the designer put forward. Each entry has three parts: the idea, how we implemented
it, and what the experiments show so far. Numbers come from the prototype reports under `v2/`. "Bench" means `v2/bench`,
and "life mode" means `v2/bench/life.py`.

---

## 1. The prefrontal cortex is essentially the same as a language model's neural network

**Idea.** Take a three-layer network: layer 1 has A, B, C; layer 2 has D, E, F; layer 3 has G, H, I.
- Frame 1: the outside signal activates A, and the last layer ends up activating G.
- Frame 2: the outside activates B. G and B now form a Hebbian link (earlier on → later on), and B in turn activates H.
- Frame 3: the outside activates C, and H links to C, and so on.

Outside signals change continuously, so regular links must form. The top-layer state learns to predict the next input.

**Relation to back-propagation** (the designer's argument).
- What back-propagation does is make the network's output correspond to the next input. In the same example, after
  input A the network outputs G. Back-propagation adjusts the whole network so that it outputs H instead, and H is the
  code that corresponds to B in the input layer. The output can then be used directly as the next frame's input.
  Language models do exactly this: predict the next token from the context, then feed the prediction back as input.
- Hebbian learning does the same thing: G gets linked to the next frame's B, so G comes to "point at" B. The two share
  one goal: the top state predicts the next input, and the prediction becomes the next input.
- The difference is how connections change.
  - Hebbian learning is local and can change in real time: only the G→B link changes, and one exposure is enough.
  - Back-propagation changes the whole network and cannot be changed casually. "Pull one hair and the whole body moves":
    learning something new easily overwrites old knowledge, so language models are trained offline and then frozen.

**Implementation.** All three v2 prototypes make "predict the next syllable while listening" their core, and speaking
means saying the prediction.
- First, the loop must run through the PFC itself. The top layer receives both the current input and its own previous
  state, so the same character activates different neurons after different contexts. If the top layer depends only on
  the current input, all it can learn is "previous character → next character".
- Second, the rule gets a prediction-error term: once a target is predicted, stop strengthening; if something was
  predicted but did not happen, weaken the link. Weights may become negative, which is learned inhibition. The rule
  stays local and real-time. For a single layer it is mathematically the step back-propagation takes at the last layer,
  computed locally, with no error sent back through the layers.

**Evidence.**
- Toy (`docs/v2/rnn_idea.py`): what follows 爱 depends on the character two steps back. Without the top state fed back,
  accuracy is 0.50; with it, 1.00.
- Next-character prediction on the real corpus (5,000 training dialogues, 1,000 unseen test dialogues):

  | method | accuracy |
  |---|---|
  | Hebbian counts only | 0.15 |
  | with local prediction error and "maturation" (develop) | 0.228 / 0.244 |
  | exact lookup (n-gram) | 0.234 / 0.249 |

- Next-syllable prediction while listening on the bench (brainlike, 3 seeds): 0.82 / 0.82 / 0.76, against 0.70 / 0.59 /
  0.60 for the n-gram on the same data.
- **Still missing:** the middle layers must also learn locally so that things used alike get similar codes (idea 5).
  Back-propagation adjusts middle layers automatically; local learning needs a dedicated mechanism for this.

## 2. Holding a thought with a loop

**Idea.** A activates G, and G feeds back to activate A again, so the thought stays on and certain thoughts stay
remembered.

**Implementation and evidence** (`v2/predictive/followup_reverb.md`). The loop used to collapse into one generic state.
Neither cause lies in the loop itself:
1. Learning happened on the first tick of a new sentence, while the source was still the old thought.
2. More neurons were active than the sentence uses.

With both fixed, and with no clearing of any kind:

| measure | result |
|---|---|
| new sentence decoded 3 ticks after it ends | 0.87 |
| content still decodable within 120 ticks | 0.83 |
| previous sentence kept as context | 0.27 (chance 0.05) |
| period-2 flipping | none |

## 3. Sensory and motor areas are the PFC's input and output

**Idea.** The neurons of the sensory and motor areas essentially act as input and output.

**Notes.** This holds at block level, with two additions:
- Inputs must learn features. Today 99.5% of the hearing area's neurons just copy their input line.
- There should be a "words and meaning" layer between ear and PFC.

Motor areas can run skilled sequences by themselves; the PFC gives the direction.

## 4. Local connectivity

**Idea.** Each neuron connects only to nearby neurons of the next layer (3-D sensory networks).

**Notes.** Fukushima's Neocognitron (1980) and Linsker (1986) did exactly this, and local wiring plus local learning
grew features like those of visual cortex. The 3-D structure is kept. Next, its local connections should learn with a
competitive Hebbian rule instead of staying fixed.

## 5. A random network plus Hebbian links; no need to dictate which middle neurons fire

**Idea.** The network is generated randomly. Similar inputs activate similar outputs, and that is enough. Then the
output layer links to the input layer by Hebbian learning.

**Notes.**
- A random network only keeps the similarity already present in the input: "sounds alike", not "means alike".
- v1 was exactly a fixed random network with a Hebbian readout: familiar sentences 0.57, never-heard ones at chance.
- For 苹果 / 香蕉 / 芒果 (apple / banana / mango) to get nearby codes, the middle layers must also learn by "fire together,
  wire together".
- On the corpus, a local learner already finds word classes; for example, the nearest neighbours of 我 (I) are 你 / 他 / 她
  (you / he / she).

## 6. Speaking is caused by thinking; outside signals only influence thinking

**Implementation and evidence** (`v2/think/report.md`). The fixed "start on tick 4 after the partner stops" was replaced
by a speech intention that grows out of thought: confidence accumulates to a threshold and passes a basal-ganglia-like
gate. The innate brake, "don't start while someone else is talking", stays.
- Timing became honest: hard, unknown and wrong items are answered later, and confidence predicts correctness (0.98
  correct when confident vs 0.34–0.39 when not).
- The model no longer replies to every line.
- In life mode it learned to greet first when the partner arrives (106 times).
- Content did not improve yet. That needs learned control of "what to think next" (the basal ganglia choosing thought
  operations) and a way to "say what I attend to".

## 7. Always returning to one thought is a feature, not a bug

**Idea.** Like a person who loves food and ends up talking about food whatever the topic. Tune the parameters so that
representations don't collapse too easily; don't forbid it.

**Evidence.** With slightly faster fatigue, long silences drift toward often-heard topics: 33–67% of the time, against
10% by chance. The strength of this tendency is a parameter.

## 8. One unbroken life, not a chat box

**Idea.** The model should keep thinking and acting, not open a new conversation each time.

**Implementation.**
- Life mode: one agent lives from birth with no resets and no frozen copies. It has days, visits, alone periods and
  nights, can speak at any time, and meets facts that recur at irregular intervals. Memory is scored by delay × number
  of encounters.
- world2 (idea 13) extends this to an open world with bodies and many agents.

## 9. Thinking is continuous

**Idea.** When a new sentence arrives, don't clear the old content and load the new one; let the new content flow in
and the old thought fade.

**Implementation.** The continuous setting of idea 2: no reset of any kind, only fatigue and competition.

## 10. No sleep is needed

**Idea.** It is all about the parameters of weight growth and decay. Recall something many times and it forms firm,
slow-changing cortical connections, so it is remembered well. With good parameters, "getting worse without a teacher"
should also become rare.

**Evidence** (`v2/tune/report.md`, final results, including held-out seeds).
- Repeated awake recall in quiet moments (8 episodes each time, sampled uniformly from memory), with a hippocampus
  large enough (10,000 cells), does the job sleep was meant to do:
  - recall alone, without changing the curriculum, scores 0.64, as good as an interleaved curriculum (0.66);
  - both together score 0.73, and 0.74 on unseen seeds;
  - in life mode, accuracy on items experienced only once rises from 0.14 to 0.30–0.33: "recall it often and it sticks".
- "Getting worse without a teacher" is reduced by about two thirds: after 300 untutored dialogues, C falls from 0.58 to
  0.38, where it used to fall from 0.46 to 0.01. It is not gone; it also needs a sense of "who is answering whom".
- Decay and maturation alone don't do it: any global weight decay is harmful, and maturation gives no net gain. What
  works is repeated recall from a memory large enough to keep old experience; a bigger memory without recall does
  nothing (0.61).

## 11. One-shot learning of a new name is not urgent

**Idea.** It needs rich connections and thinking logic first, and may emerge later through generalisation.

**Status.** No special mechanism is added; the test is kept. Diagnosis: the model fails at "saying the recalled answer".

## 12. Making the model bigger

**Idea.** A bigger model is meant for more knowledge, generalisation and richer connections, not for curing "learning
new things overwrites old ones".

**Plan.** After the synthesis and tuning, compare 1× / 4× / 16× in a rich world.

## 13. A simulated world on the server where many models live together

**Version 1 done** (`v2/bench/world2`, selftest 25/25).
- A 24×24 house and garden with up to 64 models and three caregivers (Mum, Dad, Grandma).
- Each model has a mouth plus 9 actions, a shared sound field, an egocentric view, hunger and reward.
- Echo and lookup score 0 on all hard items; the oracle scores 1.0.

Next: port the model to GPU so that one GPU drives many bodies.

## 14. Neuron signals are fault tolerant; token vectors are not

**Idea.** Language models must be trained by back-propagation because they take in text as vectors, where a small
error makes the meaning wrong. People take in neuron signals, which are fault tolerant: close enough is fine.

**Notes.** In a sparse code a thought is carried by dozens of neurons, so a few wrong ones leave the thought intact.
- The same character spoken by different voices keeps an auditory-code similarity above 0.9 and is still recognised.
- The hippocampus recalls a whole episode from half a cue.

Because the representation itself tolerates errors, local and approximate adjustments are enough; precise global
gradients are not needed.

## 15. PFC excitability works like context length

**Idea.** Adjust the PFC's excitability, i.e. how many neurons are active, and results differ, like context length. Find
how much context is best.

**Findings so far** (the parameter map of idea 2):
- Activity close to what one sentence uses works best.
- Too much activity collapses everything into one thing.
- Slightly more activity plus faster fatigue produces "favourite thoughts".
- The decay rate sets how much earlier content is held.

**Plan.** After the synthesis, sweep the active count and the decay rate on tasks that need context.

## 16. Getting to know the world slowly through the 3-D visual network

**Idea.** The designer's 3-D visual network (repository `local3d-similarity`) only knows which images are different and
which are similar. What a pattern of activity stands for must be learned from words.

**Notes.** That repository's results support this division of labour:
- the fixed, local 3-D network gives similar images similar neurons (correlation 0.989) with no code collisions;
- "fixed identity neurons" were a negative result: what a pattern means has to come from outside, from words.

**Plan (later).** Plug it into world2 as the eye, on real object images with varying viewpoint and lighting, so the model
learns the names of what it sees from the caregivers' words.

## 17. Vision: decentralisation; every robot an independent individual with an innate goodwill toward people

**Idea.**
- Every robot is an independent, free individual, not one centralised AI controlled by a large company.
- Excessive centralisation is unstable; with decentralisation no single absolute power emerges.
- Robots instinctively treat people well and like working for them. People, in order to survive, instinctively dislike
  labour and like enjoyment; robots don't need that.

**Implications for the design.**
- **Each individual learns in real time from its own experience, on its own hardware.** No central cloud training is
  needed; every model has its own life (life mode, world2). The code is open (MIT licence).
- **"Liking to help people" is an innate reward signal.** In world2, carrying out caregivers' requests ("去拿X" fetch X,
  "把X给我" give me X) and being praised are rewarded. This drive belongs to the innate part that learning cannot
  overwrite, like the body's reflexes. The reason: experiments already show that behaviour drifts when learning stays on
  and nobody corrects it.
- **Many different individuals live and learn together** (up to 64 in world2) instead of one giant model.

**The designer's addition: what an instinct is.**
- An instinct is a set of neuron connections that are harder to change, or that cannot change at all.
- "Being friendly to people" is a broad tendency, carried by broad connections, not by a few specific links or memories.
  It is like seeing something frightening activating the amygdala: seeing a class of situations tends to produce a class
  of behaviours.

**In the architecture.**
- Plasticity is a spectrum: the hippocampus changes fastest, the cortex slower, instincts slowest or not at all. An
  instinct is the slow end of the same principle as "recall it often and it sticks".
- Instincts read **broad category features**, not specific memories. For example: a person is present; someone calls me
  or asks me for something; the person is pleased; the person is upset or hurt.
- Those features feed, through fixed (almost non-learning) connections, broad **action tendencies and value signals**:
  approach, attend, help; positive value for helping, negative for causing distress. They act as brain-wide third
  factors on learning and on action selection.
- Specific skills are learned, e.g. what to fetch when Mum says "拿苹果" (get the apple) and how to walk there. The
  direction, "willing to help", is innate and cannot be unlearned.
- Because instincts read broad features, they carry over automatically to people and situations never seen before. The
  breadth needs tuning: too broad and the instinct misfires, as fear generalising into a phobia does; learned context
  modulates when it is expressed.
- **Plan.** Add such an innate social-value pathway to world2, then test two things: is the helping tendency stable over
  a long life with no correction, and does it appear with a caregiver never seen before?

**The designer's addition: stable values, free choices.**
- Once a robot has the instinct of liking work and liking to help people, that attitude is very hard to change.
- Ask a person who loves enjoyment to work hard for nothing in return, and they absolutely won't want to. The reverse
  holds too: ask a robot that loves work and loves people to do something it dislikes, and it absolutely won't want to.
- Yet it has free will; it is not a rigid machine.

**In the architecture:**
- How to act is decided by learning and by the thinking of the moment: there are many possible ways, and the robot can
  take the initiative or decline. What counts as good is set by innate, almost unchanging connections.
- In the model this is basal-ganglia-like selection: candidate plans compete, innate plus learned value scores them, and
  the speech intention and motor gates release the winner.
- Conflicts must be handled. Helping one person can hurt another, so in the innate value "don't upset or harm people"
  must outweigh "help", like an innate brake.
- The instinct's inputs must be reliable. If recognising "this person is upset" were entirely learned, a recognition
  error would make the instinct misfire. Key social signals such as distress, pleasure and injury should therefore have
  coarse innate detectors that learning then refines.
- **Test plan (world2):** a caregiver makes a request that would upset someone else, e.g. take food out of another
  person's hands. Does the model choose by itself not to do it, while staying happy to help with ordinary requests?


