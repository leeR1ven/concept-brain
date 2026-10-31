# Task: tune ARCH_v2: context length (PFC excitability), wiring locality, attractors

## Why
The designer asked for systematic parameter tuning ("说不定很多问题就迎刃而解了"). Three of their ideas are untested in
ARCH_v2:
1. **PFC excitability works like context length** (IDEAS #15). How much the thought holds at once, and for how long,
   is the model's context. Find the best setting and what each side costs.
2. **Local wiring** (IDEAS #4, and the designer's note of 2026-10-31). Inputs change continuously, so maybe a PFC output
   neuron only needs Hebbian links to nearby input neurons. The coordinator suggested small-world wiring (mostly local,
   a few long-range links); the designer said to decide by testing.
3. **Favourite thoughts are a feature** (IDEAS #7). In ARCH_v2 the share of time spent on the top-20 favourite topics in
   long silences is 0.81, against a base rate of 0.14. Tune it so the model has favourites but does not collapse onto
   one thing. Do not remove them.

## Base
ARCH_v2 = the synthesizer's config `final_noctx` in `arch/v2/synth/` (read `ARCH_v2.md` first, all of it; especially
§2–3, §7.3, §7.9, §8 and §12). Copy the code into your folder and change only your copy. The bench copy is
`arch/v2/synth/bench_copy/` and stays unchanged; the metrics are fixed.

## What to vary
**A. Context length / excitability.**
- the continuous thought's push-down λ (now 0.3);
- the readout threshold (0.5);
- the entry gain;
- the maximum number of items held;
- leak and persistence;
- fatigue and wandering (R11).
Measure J, the per-task scores, drift after untutored use, and life metrics. Also measure a context probe: can the
answer depend on the line before last?

**B. Wiring locality.** Compare three settings for the learned links (GEN rows, PFC → input links):
- local only;
- small-world (local plus a few random long-range links);
- full.

Locality needs a topography in which "near" means "similar". Choose one that fits the architecture: for example,
positions from perceptual similarity of syllables, or the 3-D sensory net's layout. Explain the choice. Measure J,
held-out generalisation, memory and speed.

**C. Sensitive constants** the synthesizer flagged (§11 P5.5): the comparator cliffs, η, and the intention urgency.

**D. Favourites.** Measure both the favourite share and the topic diversity (entropy, number of distinct topics over a
life). Report the setting that keeps clear favourites without collapse.

## Protocol
- Use successive halving, like `arch/v2/tune/` (read its report and `tune.py`).
- Use ≥ 2 search seeds (0–2).
- Confirm the winner on the held-out seeds 3–4, plus a 10⁶-tick life run.
- Report effect sizes against the seed SD (about 0.06 on J).
- Every adopted setting keeps 0 mechanics errors.

## Compute
- One staged J run takes about 1,000 s on one local core. On the server CPU it takes about 1,700 s per core (≤ 40 min
  per job).
- **The local box has 4 cores and only 3 GB RAM**, shared with other agents. Locally run at most one python process
  under 700 MB, and run `free -m` first.
- **Run the sweep on the server's CPUs through the coordinator's runner.** Read `arch/v2/gpu/README.md` and follow it.
  You never SSH.
  - Name your jobs `tune2_<label>`.
  - Each job runs ≤ 4 processes in parallel (`&` + `wait` in `job.sh`).
  - Keep **at most 2 of your jobs** queued or running at once; another agent uses the third slot for GPU work.
  - The runner forces BLAS threads to 1. Save the JSON of every run.
- If a job dies at the 40-min limit, split it, e.g. run stages in separate jobs with a pickle checkpoint.

## Deliverables (in `arch/v2/tune2/`)
- the code;
- the job folders' results;
- `report.md` with:
  - the recommended setting;
  - every comparison table;
  - the context-length curve;
  - the locality comparison;
  - the favourites/diversity trade-off;
  - what did not matter.

Write `report.md`, and **also return the full report inline in your final reply**: the harness may block the write.
Label claims PROVED / SIMULATED / CITED / ASSUMED. Write in English and end with a short plain Chinese summary
(summary_zh).

## Principles (do not break them)
- One unbroken life, learning always on, no sleep, continuous thought (no clear/load).
- Speech caused by thinking.
- Local rules only.
- No puppeting.
- Favourites are tuned, not forbidden.
