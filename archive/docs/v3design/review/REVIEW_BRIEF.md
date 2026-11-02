# Adversarial review of ARCH_v3_core

## The design under review
`docs/v3design/synth/ARCH_v3_core.md`.

It is one fixed core:
- 4 mechanisms, 2 broadcasts, 2 objectives and 5 innate priors (17 sub-items);
- a 30-gene genome with lineage inheritance (merging of learned weights);
- a maturity rule;
- a neuron-level form for every part;
- a 13-step migration ladder from ARCH_v2.2;
- the first five experiments.

Its inputs are the five member reports in `docs/v3design/` (audit, neuro, learning, develop, eval) and the shared `BRIEF.md` there.

## The designer's binding decisions (2026-11-01)
In `docs/IDEAS.en.md`:
- **#12 addendum:** scale the whole model with one factor, together with experience and the world.
- **#14 addendum:** repairing a misheard word is learned from experience of mishearing.
- **#19 and its addendum:** reproduction and merging. Grown-up models are merged and slightly randomised, and the offspring are born with those connections.
- **#20:** errors are allowed in childhood; 0 mechanics errors are required once grown up.
- **#21:** the final product must be a pure-neuron model. Shortcuts are allowed during development only if they are equivalent to a neuron circuit.

The overriding goal: **a fixed set of mechanisms that learns new abilities from experience, with no new mechanism per ability**, on the way to a real-time-learning general intelligence. It hears, thinks and speaks through muscles, lives one unbroken life without sleep, and later lives in an open world with many models.

## Evidence
- **Code and results:**
  - `v2/`: `synth/` (ARCH_v2), `v21/`, `hearing/`, `meaning/`, `randmix/`, `v22/`;
  - the bench in `v2/bench/`, world2 in `v2/bench/world2/`.
- **v2.2's final report:** `v2/v22/report.md`.
  - Retention collapses to 0.16 at 3,000 untutored dialogues.
  - Its correction cue analysis is in `results/cue_test_base.txt`.

## The one question every reviewer answers
**Is ARCH_v3_core logically sound and buildable?** Specifically:
- Will this fixed core, with the proposed world changes and the lineage loop, actually produce today's abilities and new ones?
- Will it stay stable in one unbroken life?
- Can it end as a pure-neuron model within realistic compute?

Find what is wrong. For each problem give:
- **severity:**
  - FATAL: the design cannot work as stated;
  - SERIOUS: a likely failure, or a large hidden cost;
  - MINOR.
- the evidence (a quote from the document with its section, code, a cited paper, or arithmetic);
- **the smallest fix** that keeps the design general. Never fix a problem with a task-specific rule. That is the failure mode this round exists to end.

**Also check:**
- whether the tick-by-tick behaviour chains (§6) really work with the stated numbers;
- whether any task-specific or world-fitted structure is hidden inside "generic" parts, priors or genes;
- whether the migration order and the first experiments retire the biggest risks first.

## Rules
- **Review only:** no training runs, no server jobs.
- **Toy checks are welcome.** Run small local numpy toy simulations that test a claim. Put them in `arch/v2/v3design/review/<your-role>/` in the working tree (``). Run `free -m` first; at most 1 python process at a time, under 700 MB, because other reviewers share the 4-core / 3 GB box.
- Do not modify the repo or other agents' folders.
- **The report.** The harness blocks report writes, so return your full report inline in your final reply.
  - Write it in English and label claims PROVED / SIMULATED / CITED / ASSUMED.
  - Rank the issues, most severe first.
  - End with an overall verdict: build as is, fix then build, or redesign. Name the 3 fixes that matter most.
  - Finish with a short plain-Chinese summary (`summary_zh`).
