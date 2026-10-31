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

**Next:** synthesize one unified design → port it to GPU so one model drives many bodies → long multi-agent life in
world2, using the designer's 3-D visual network on real images → scaling curves (1× / 4× / 16×).

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

## License
MIT, see [LICENSE](LICENSE).
