# Design round: a fixed set of general mechanisms (the ARCH_v3 core)

## The designer's request (2026-11-01)
> 最好想个办法不用为了让模型实现某项能力而不停的加机制，而是固定数量的机制就能学会任何事情。

"Find a way so that we do not keep adding a mechanism for every new ability. A fixed number of mechanisms should be able
to learn anything."

Earlier the same day the coordinator told the designer that the main risk on the road to general intelligence is that
every new ability so far needed a new hand-designed mechanism, and that the decisive test is whether ability keeps rising
when model and world grow **without new code**. The designer agrees and wants the architecture rebuilt around that.

## Why this round exists
Each step since ARCH_v2 fixed a measured failure by adding a task-specific part:
- **ARCH_v2.1:** role-aware listening (an eligibility trace with hand-set cancel / commit timing rules), say-what-I-recognise
  routes, a signed value loop with social features, dips and a veto.
- **Hearing:** seven front-end keys (noise floor, confident creation, syllable re-lock, voice boundaries, no-create,
  level matching, denoising).
- **Meaning:** learned context codes plus a separate meaning cue in the hippocampal DG.
- **ARCH_v2.2 (in progress):** a correction-vs-question rule and more.

Every one was justified locally, but the count only grows. A model that needs a new module per ability will never be
general.

## Goal
Find a **small, fixed set of generic mechanisms** (the core) with these properties:
- Every ability the current agent has is either:
  - produced by the core from experience, or
  - one of a few explicit, justified innate priors (the brain has some: drives, reflexes, the wiring plan).
- New abilities come only from experience and the world, never from new code.

**Definitions:**
- **Generic.** A mechanism is generic if its definition mentions no task, modality, word or content, and it applies in
  the same form wherever it is used. Examples:
  - "Divisive normalisation in every area" is generic.
  - "A noise floor for the ear" is the same computation, but it is generic only as an instance of the area-general rule.
- **Innate prior.** Allowed, but they must be few, fixed and budgeted, never added per task. Say for each one why the
  brain has it, and cite evidence.

## Binding principles (the designer's; `docs/IDEAS.en.md`)
- The PFC is a recurrent next-input predictor.
- Thoughts are held by loops; thinking is continuous, with no clear/load.
- Sensory and motor areas are the PFC's input and output.
- Local connections and local learning rules:
  - today's rule is pre, post, the synapse's own trace and at most one broadcast signal;
  - you may argue for a few neuromodulator-like broadcasts if the brain has them, but say the cost.
- Speech is caused by thinking.
- One unbroken life with learning always on. No sleep: memories firm up through recall.
- Attractors and favourites are tuned, not forbidden.
- Foveated vision. Neuron codes are fault tolerant.
- Bigger is better once the architecture is sound.
- An innate prosocial drive (IDEAS #17).
- Keep it simple: the designer dislikes piles of rules. Fewer, more general parts beat more, cleverer parts.

## Read first
- `README.en.md`: history of v1 → v2 → v2.1.
- `docs/IDEAS.en.md`: all of it, including #14's addendum of 2026-11-01. That addendum says
  correcting a misheard word may be learned from experience of mishearing, not built in.
- `v2/synth/ARCH_v2.md`: the current architecture, every part.
- Reports:
  - `v2/v21/report.md`;
  - `v2/hearing/report.md`;
  - `v2/meaning/report.md`;
  - `v2/randmix/report.md`.
- `docs/review2/SYNTHESIS.md`: the review-2 plan.
- The v2.2 task, which another agent is running now:
  - its brief is `arch/v2/v22/TASK.md`;
  - its code is in that folder; read only, do not touch it.
- The bench and world:
  - `v2/bench/README.md` (tasks, metrics);
  - `v2/bench/world2/README.md` (the open world with many bodies);
  - the designer's 3-D vision world is the repo `local3d` (IDEAS #16), not built into the agent yet.

## Rules for this round
- **Phase 1, design only.**
  - No training runs and no server jobs.
  - A small local analytic script is fine:
    - at most 1 python process under 700 MB;
    - run `free -m` first, because other agents share the 4-core / 3 GB box.
- **Files.**
  - Write scratch files only into `arch/v2/v3design/<your-role>/` in the working tree.
  - Do not modify the repo or other agents' folders.
- **The report.** The harness blocks report writes, so **return your full report inline in your final reply**.
  - Write it in English.
  - Label claims PROVED / SIMULATED / CITED / ASSUMED.
  - Cite papers with author and year.
  - End with a short plain-Chinese summary (`summary_zh`).
- **Be concrete.** Map every proposed mechanism to the existing code where one exists (file, class). Say what would be
  removed. Name the experiment that would falsify the claim.
