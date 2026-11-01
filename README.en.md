# concept-brain

[中文](README.md) | **English**

A research prototype of a brain-inspired AI that **learns in real time, all the time**. There is no offline training
phase and no back-propagation: every connection learns locally while the system lives.

The current instance hears speech and answers through muscles. It has an ear (a 3-D sensory network), a mouth made of
32 muscles with a simple voice physics, a hippocampus, a prefrontal cortex and a motor system. It learns to talk by
listening, babbling and conversing.

> This is research in progress, not a product. The long-term goal is an architecture that keeps learning from the world
> the way a person does.

## Core ideas (from the designer; details and evidence in [docs/IDEAS.en.md](docs/IDEAS.en.md))
1. **The prefrontal cortex is a recurrent next-input predictor, like the recurrent networks of language models.**
   - The top-layer state at one moment forms a Hebbian "earlier → later" link to the input of the next moment, so the
     network learns to predict what comes next; a prediction can be fed back as input.
   - Back-propagation pursues the same goal by adjusting the whole network ("pull one hair and the whole body moves").
   - Local Hebbian learning can change in real time.
2. **Thoughts are held by loops** (A → … → G → A), continuously. A new sentence flows in and the old thought fades;
   nothing is "cleared and reloaded".
3. **Speaking is caused by thinking**, not directly by the external signal; the outside world only influences thinking.
4. **One unbroken life:** the model keeps thinking and acting, unlike a chat box that starts a new conversation each time.
5. **No sleep is needed:** memories become firm through repeated recall, which builds slow-changing connections.
6. **Settling into favourite thoughts is a feature** (like a person who always ends up talking about food); tune it,
   don't forbid it.
7. **Local 3-D sensory networks:** random local wiring keeps "similar inputs → similar neurons". Meaning comes from
   linking these patterns with words.
8. **Sparse neuron codes are fault tolerant** ("close enough is fine"), unlike token vectors, so precise global gradients
   are not needed.
9. **PFC excitability works like context length**, so tune it. The model can **get to know the world slowly** through the
   designer's 3-D visual network, learning what each pattern means from words.

## Status (v2 design phase, see [v2/README.md](v2/README.md))
- **Whole-architecture review.** Five independent reviewers (neuroscience, learning & memory, development, AI theory,
  skeptic) compared the old design with the real brain ([docs/review/SYNTHESIS.md](docs/review/SYNTHESIS.md)). Main
  findings:
  - the learning parts did not reach behaviour;
  - there was no learned representation and no real-time learning without a teacher;
  - the learning rule lacked prediction error, learned inhibition, co-activity learning, eligibility traces and
    broadcast "reward/surprise" signals;
  - the hippocampus should be a one-shot, long-lasting episodic memory.
- **Shared test bench** (`v2/bench`). A grounded micro-world in stages: hear and babble → naming with a scene → sentence
  frames with slots → relations and memory → adult dialogue. It has anti-cheat baselines (echo, lookup, n-gram, prior,
  oracle), a reference agent tick-identical to the old v1, a **life mode** (one unbroken life with visits, nights,
  initiative and recurring facts), and **world2**: an open house and garden where up to 64 models live with scripted
  caregivers and have bodies (mouth + move/turn/take/give/eat).
- **Three independent prototypes**, all learning **without muscle puppeting**: programs come from babbling, and what to
  say is learned by listening.
  - `brainlike`: brain-faithful areas.
  - `develop`: developmental, memory-first.
  - `predictive`: one local prediction-error rule everywhere.

**Results so far** (accuracy; the bar is the old v1 trained by teacher forcing):

| task | brainlike (3 seeds) | develop | old v1 | most frequent answer |
|---|---|---|---|---|
| "这是什么" (what is this?, pointing) | 0.50 ± 0.01 | 0.59 | 0.06 | 0.06 |
| "你要X吗 → 要X" (overall / never-taught X) | 0.81 / 0.78 | 0.86 / 0.44 | 0.04 / – | 0.07–0.09 / ~0 |
| stage totals: naming / frames / memory | 0.34 / 0.52 / 0.17 | 0.54 / 0.62 / 0.48 | 0.12 / 0.19 / 0.19 | 0.14 / 0.30 / 0.22 |

- **Parameter search** (about 150 runs, confirmed on held-out seeds). An interleaved curriculum, awake recall in quiet
  moments and a larger hippocampus raise the overall score from 0.42 to 0.73 (held-out seeds 0.34 → 0.74):
  - forgetting across stages disappears;
  - drift without a teacher is reduced by about two thirds;
  - any global weight decay is harmful.
- **Model size alone fixes none of the failures**; the right signals and rules do.
- **Speech intention from thinking**: replies to hard or unknown items come later, confidence predicts correctness, and
  in life mode the model learned to greet first. Answer content did not improve yet.
- The speech mechanics (never talking over a partner, starting on time, clean 3-tick syllables) stay at 0 errors in every
  run.

- **Unified design ARCH_v2** ([v2/synth/ARCH_v2.md](v2/synth/ARCH_v2.md)). It starts from the tuned setting and adds
  one part at a time, each compared on at least two seeds:
  - kept: the speech intention, a **continuous thought** (a new line pushes the old content down; nothing is cleared;
    long silences drift to favourite topics), innate retinotopic attention, and recency in recall;
  - removed: the fixed slow-context layer, which was harmful.
  - Overall score 0.73 → **0.83** (held-out seeds 0.74 → 0.82); colour / count 0.15 / 0.45 → 0.93 / 0.76.
  - In one unbroken life: free replies 0.20 → 0.34, last tenth of life 0.25 → 0.47, about 1,900 self-initiated
    utterances, 0 overlaps.
  - Still open: drift without a teacher, thinking that improves answers, one-shot naming, generalisation (≈ 0.2).
- **Review round 2** ([docs/review2/SYNTHESIS.md](docs/review2/SYNTHESIS.md)): five independent reviewers discounted
  several headline numbers:
  - life free replies 0.34 are below a trivial responder (0.51);
  - the "favourite topics" are function syllables;
  - the colour gain comes mostly from the world's eye code.
  - One shared root cause: learning by listening ignores whose turn it is, so the line heard after the agent's own
    answer is learned as the answer. A local fix raised J from 0.70 to 0.93 on one seed (drift loss 0.43 → 0.03),
    pending more seeds.

- **ARCH_v2.1** ([v2/v21/report.md](v2/v21/report.md)), Steps 0–3 of the review plan:
  - **Honest measurement.** With held-out pairs never heard in training and probed only on frozen copies, ARCH_v2 has
    H 0.60 (not 0.20) and J 1.00. On held-out items it only ties a constant answer per task (0.55 vs 0.54).
  - **Role-aware listening** (kept). A heard line's learning as an answer waits on an eligibility trace. It is
    cancelled if the line was a question (I or someone else answered it), and kept if it is rewarded feedback after my
    utterance. Results: J 1.00 → 1.23 on every seed, H 0.60 → 0.76, drift loss after 900 untutored dialogues 0.54 →
    0.01, with no ritual or colour cost.
  - **Comprehension feeds production** (kept): held-out what 0.61 → 0.81, where 0.81 → 0.90, slot 0.13 → 0.43.
  - **Signed value loop** (kept for life). In life, empty-room greetings drop from 60–80 to 5–9 per tenth, appropriate
    initiatives rise 0.19 → 0.38, and social replies 0.32 → 0.42. It costs about 0.08 J on the bench.
  - ARCH_v2.1 J 0.996 → 1.18 (5/5 seeds).
  - Open: life answer accuracy is unchanged (0.34 vs 0.62 for a trivial responder). The agent speaks right after most
    corrections, so they get cancelled as "questions".
- **Random connections from birth** ([v2/randmix/report.md](v2/randmix/report.md), IDEAS #5). MIX adds a unit per
  new combination. We replaced it with a random conjunction layer that exists in full from birth, then trained and
  pruned it (use it or lose it).
  - With equal populations per pair type: J 0.83 → 0.95 (4/4 seeds) on the default protocol, 1.00 → 1.09 (5/5) on the
    frozen held-out protocol. Colour drops.
  - Random alone does not help: a fixed random layer with MIX's allocation loses 0.05, and 4×/16× brings nothing.
  - No gain in fault tolerance, drift or meaning-based generalisation.
- **GPU port** ([v2/gpuport/report.md](v2/gpuport/report.md)): a batched torch version of ARCH_v2 reproduces the numpy
  agent bit for bit (185,817 ticks on seed 1; the short and stage-1 protocols on 5 seeds). The only change is that ties in
  the DG k-WTA are broken by unit index.
  - The reference itself depends on the numpy and Python versions. The tie rule alone moves J by up to 0.07 per seed
    (mean 0.825 → 0.807), so single-seed differences under about 0.07 mean nothing.
  - Speed: about 250 agent-ticks/s at 64 agents, far below the 30k target. Today's agent does too little and too
    sequential work per event (awake recall is 70 % of the time). Numpy on server CPUs stays the sweep engine; the GPU
    version waits for the scaled-up model.
- **ARCH_v2.2** ([v2/v22/report.md](v2/v22/report.md)): the canonical tie rule, the flat random layer inside v2.1, and
  learning from corrections in life.
  - Re-baselined ARCH_v2.1: J 1.184 (5 seeds). The `v21/` code turned out to be a rejected variant; v2.2 restores the
    reported one.
  - The random layer: J within noise, generalisation up on every seed, life free replies 0.34 / 0.28 → 0.38 / 0.40.
  - The agent used to speak right after 77–87 % of corrections, so they were cancelled as questions. A reply to a line
    that starts within 3 s of the agent's own speech is now held back while its learned value is low (10 % exploration,
    and only while recent reward is high). Corrections learned 0.13–0.23 → 0.81–0.87; life free replies 0.47 / 0.52,
    and in the last tenth of life 0.65 / 0.64, just above the same-life echo-or-constant reference; staged J 1.205.
  - Retention collapses between 2,100 and 3,000 untutored dialogues (0.16; v2.1 0.54).
  - The design round ([docs/v3design/](docs/v3design/README.md)) classes this veto as a hand rule fitted to the
    bench's timing. v2.2 is the last version that adds a mechanism per ability.
- **Learned meaning layer** ([v2/meaning/report.md](v2/meaning/report.md), Step 4; IDEAS #3, #5, #14). Every syllable
  gets, besides its fixed sound code, a code learned from context with a local rule, so syllables used in the same
  places become alike. The code feeds the random layer and the hippocampal cue.
  - Module test: foods and toys never heard in "X好不好吃" go from 0.50 (chance) to 1.00 on 10/10 seeds, including words
    first heard late in life. Swapping the two classes' codes flips every answer.
  - The old failure was mostly hippocampal recall by identity overlap, not the code; the meaning cue in the DG, weighted
    like the identity cue, fixes it.
  - Bench (v2.1 + random layer, 5 paired seeds): J 1.18 → 1.23, up on every seed, no drift cost. A random-code control
    gets about half the gain, because no bench task needs a class; the bench needs one ("X能不能吃").
  - No fault-tolerance gain; a 4× layer loses 0.11 J.
- **Open-world hearing** ([v2/hearing/report.md](v2/hearing/report.md), first part of Step 6). ARCH_v2.1's ear needed
  exact silence, so in world2 at noise σ 1 it heard 0 lines, and the bench at σ 1 scored J 0. A local front end, behind
  `h_*` keys, fixes this: an adaptive noise floor (every silence trigger becomes relative), a syllable clock that
  re-locks to recognised syllables, a voice-change boundary, level-invariant matching and denoised syllable patterns.
  - The clean bench is unchanged bit for bit (5/5 seeds, J 1.2291 in both arms). On the bench at σ 1, J goes from 0 to
    1.27 (1.27 clean). In world2 at σ 1 the agent hears 330 lines and speaks 193 times (304 / 201 noise-free), with no
    talk-over.
  - In crowds, near lines whose start overlaps another voice are heard with coverage 0.74–0.97 (v2.1: 0.06–0.22).
  - It tracks noise that changes during one life with no reset.
  - Open: the brake stays on in crowds, so agents rarely speak there; far speech fragments in noise; σ 2 works only in
    part; trust-weighted listening needs voice identity.

**Next** (review round 2 plan, Steps 0–6; items 1–4 are ARCH_v2.1, item 5 and the hearing part of item 7 are done):
1. honest measurement;
2. role-aware learning with the self in the loop;
3. comprehension feeds production;
4. a signed value loop;
5. a learned meaning layer on random connections from birth (done; merging into ARCH_v2.2);
6. slow consolidation;
7. open-world hearing (done) and an action loop.

Then the GPU port into world2 and the scaling curves.

**Standalone side experiment** (IDEAS #18, unrelated to the brain architecture): can a back-propagation LLM be
converted into a Hebbian-style model that keeps learning? Only where the rule already is back-propagation, and being
local does not stop forgetting. Sparse feature codes reduce the damage to old knowledge but not the interference between
facts. A frozen network with an entity-keyed sparse episodic store learned 10k facts with no forgetting, and replay
mixed with ordinary text consolidated 98 % of 2k facts into the weights. See [side/llm2hebb/](side/llm2hebb/README.md).

## Run
```bash
pip install numpy
python3 v2/bench/selftest.py            # test bench self-checks (~1–2 min)
python3 v2/bench/world2/selftest.py     # open world self-checks (~40 s)
python3 v2/develop/test_rules.py        # learning-rule unit tests of one prototype
```
The dialogue corpus is not stored here. `python corpus.py` builds it from chatterbot + LCCC + PersonalDialog into
`/root/work/corpus/dialogs.jsonl`. It is needed only for the adult-dialogue stage.

## Layout
- `brain/`, `world/`, `talk.py`: the v1 model (binary neurons, order-based local learning, 3-D sensory nets, voice physics).
- `v2/`: the shared bench, life mode, world2, and the three prototypes plus the tuning and "think" experiments, each with
  `report.md`.
- `docs/`:
  - `IDEAS.md` / `IDEAS.en.md`: the designer's ideas and the evidence;
  - `review/`: the five reviews;
  - `research/`: literature reviews;
  - `v1_panel/`: the v1 design discussion.
- `side/`: standalone experiments that are not part of the brain model (`llm2hebb/`, IDEAS #18).

## License
MIT, see [LICENSE](LICENSE).
