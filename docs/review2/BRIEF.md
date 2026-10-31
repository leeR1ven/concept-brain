# Review round 2: the whole ARCH_v2 architecture

## Goal
The final goal (designer): an AI architecture that **learns in real time, all the time**. It should keep thinking and
acting in an open world, speak because it thinks, and keep learning from the world and from people the way a person
does, with no offline training phase and no back-propagation. The current instance hears speech and answers through
muscles.

ARCH_v2 is the unified design that has just been synthesized. Before we scale it (GPU port, many bodies, a bigger
model), it gets an independent whole-architecture review. Every reviewer answers the **same macro questions** from
their own lens. Do not narrow the topic to one part.

## Questions (answer all four)
1. **ARCH_v2 vs the real brain.** Which important differences remain? Which of them matter for the goal and which do
   not?
2. **Problems of the whole architecture.** Do not stop at the open problems ARCH_v2 lists itself.
   - Is anything logically unsound?
   - Is there a hidden teacher signal or a shortcut (e.g. reading world state the agent could not perceive)?
   - Is there a mechanism that will not survive an open world, many bodies, real images or an adult corpus?
   - Does any number look too good or not mean what it claims?
3. **What should change next to reach the goal?** Rank at most 5 changes. For each give:
   - the mechanism;
   - why it is the bottleneck;
   - a falsifiable prediction;
   - a cheap test on the bench, life mode or world2.
4. **Are the designer's ideas implemented faithfully?** The ideas are in `/root/work/concept-brain/docs/IDEAS.en.md`
   (17 ideas). The binding principles are in ARCH_v2 §9. Where does ARCH_v2 violate, stretch or under-use them?

## Read first
- `arch/v2/synth/ARCH_v2.md`: the design, every result, the lesions and the open problems (75 KB; read all of it).
- `/root/work/concept-brain/docs/IDEAS.en.md`: the designer's ideas and the evidence so far.
- `arch/review/SYNTHESIS.md`: what the first review round concluded (so you can check what was fixed).
- The code in `arch/v2/synth/` (`core_synth.py`, `agent_synth.py`, on top of `core.py`, `rules.py`, `agent_think.py`,
  `core_think.py`, `body_think.py`). The bench is in `arch/v2/synth/bench_copy/`, and world2 is in
  `arch/v2/bench/world2/`.
- Optional: the earlier designs and reports in `arch/v2/{brainlike,develop,predictive,tune,think}/report.md`.

## Decisions already taken by the designer (do not argue against these; work within them)
- PFC = a recurrent next-input predictor (like a language model's RNN), learned by local Hebbian "top state (t−1) →
  input (t)" links. Thoughts are held by loops.
- Thinking is continuous. Nothing is cleared and reloaded when a new line arrives.
- Speech is caused by thinking; outside signals only influence thinking.
- One unbroken life with learning always on. No sleep: memories firm up by repeated recall.
- Favourite thoughts and attractors are a feature. Tune them so they do not collapse; do not forbid them.
- One-shot naming is not urgent.
- Local connectivity and random wiring. Sparse neuron codes are fault tolerant.
- PFC excitability works like context length. Bigger is better once the architecture is sound.
- No muscle puppeting. Programs come from babbling; what to say is learned by listening.
- Innate mechanisms are allowed. The designer sees instinct as connections that are hard or impossible to change.

## Rules
- Write only inside `arch/v2/review2/<your_lens>/`. Everything else is read only. No git, no SSH.
- **Compute is scarce.** The local box has 4 cores and only **3 GB RAM** shared by everyone, and other agents run
  experiments at the same time.
  - At most one local python process at a time, under 15 min and under 600 MB.
  - Run `free -m` first and do not start if less than 700 MB is available.
  - Prefer reasoning from the existing result files: `arch/v2/synth/out/` has every run log and JSON.
  - Do not use the GPU runner.
- Labels: PROVED (argument with numbers), SIMULATED (your run; file and numbers), CITED (paper and finding), ASSUMED.
- Be concrete and quantitative. Point at code lines and result files.
- Write `arch/v2/review2/<your_lens>/report.md`, and **also return the full report inline in your final reply**. The
  harness may block the write. Write in English, under about 5,000 words, and end with a short plain Chinese summary
  (summary_zh).
