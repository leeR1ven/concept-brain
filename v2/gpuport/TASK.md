# Task: port ARCH_v2 to torch on the GPU, many bodies in world2

## Why
ARCH_v2 (`arch/v2/synth/ARCH_v2.md`) is the unified design. The numpy prototype runs one agent at about 2.3 ms per tick
on one core, and the local box has only 4 cores and 3 GB RAM. Everything the designer wants next needs a fast batched
instance:
- long multi-agent lives in the open world world2;
- parameter sweeps;
- scaling curves (1× / 4× / 16×);
- an innate social-value pathway.

## Plan
Follow ARCH_v2 §10–11 (P0–P5). Change it only where you find a reason, and document why.
- **P0.** Freeze the numpy SynthAgent (config tag `final_noctx`) as the reference. Record a fixed-seed trace (mouth
  output, detected items, confidence, releases) for ≥ 2,000 ticks of stage B.
- **P1–P3.** All state as batched tensors, B agents.
  - Hashed conjunction units with sparse top-K GEN rows (or dense when small), learned with `scatter_add_`.
  - Event-driven control flow becomes masks, so one tick is one fixed graph with no per-agent Python.
- **P5 validation ladder.**
  1. One agent, fp32: the mouth trace matches the numpy reference over the 2,000 ticks. If exact identity is impossible
     because of hashing collisions or float order, explain why and show the behavioural match.
  2. J on seeds 0–4 within ±0.03 of the numpy numbers (ARCH_v2 §8.1).
  3. Speed: aim for ≥ 30k agent-ticks/s at 64 agents, ≤ 10 GB GPU memory.
- **P4 world2 bridge** (after P5.1–2 pass).
  - Eye: slot-relative attention with S = 25 cells instead of 3. world2 also has `eye_mode='fovea'`; support both if
    cheap.
  - Motor: the 9 motor units become programs bound by motor babbling. Requests are learned from caregivers' reactions.
    No puppeting.
  - Hunger and eating reward feed the critic as a third factor.
  - Hearing gain control before the syllable categories (§8.4: categories hit their cap within a day in world2).
  - Overheard speech from other agents is learned like other speakers.
- **P5.4.** world2 runs with 4, 16 and 64 agents for several days each. Report world2's metrics against its reference
  table (silent / echo / lookup / namer / oracle), plus speed and memory.

## Binding principles (ARCH_v2 §9 and `/root/work/concept-brain/docs/IDEAS.en.md`)
- One unbroken life: no resets between lines, dialogues, days or nights. Learning is always on. No sleep (awake recall
  only).
- Thinking is continuous: no clear/load.
- Speech is caused by thinking.
- Local learning rules only.
- Body mechanics stay at 0 errors: no talking over a partner, 3-tick syllables.
- No muscle puppeting.
- Any new mechanism needed for world2 must be minimal, local and documented, with a lesion showing it reaches behaviour.

## Where things are
- Reference code: `arch/v2/synth/` (read only; copy what you need into your folder). The bench copy is
  `arch/v2/synth/bench_copy/`.
- world2: `arch/v2/bench/world2/` (read only; README, API, selftest).
- GPU server: read `arch/v2/gpu/README.md` and follow it exactly. You never SSH. You submit job folders to the
  coordinator's runner.
  - Name your jobs `gpuport_<label>`.
  - Keep **at most 2 of your jobs** queued or running at once; another agent shares the 3 slots.
  - Each job ≤ 40 min; checkpoint and print metrics as you go.
  - The server has torch 2.4 + CUDA and numpy 2.1.2. Local numpy is 1.26.4.
- **Local box:** 4 cores and **3 GB RAM**, shared with other agents. At most one local python process at a time, under
  700 MB. Run `free -m` before starting. Use it for the numpy reference and small checks only.

## Deliverables (in `arch/v2/gpuport/`)
- The torch implementation, e.g. `brain_torch.py` and `agents_torch.py`.
- The tests: trace match, J match, speed.
- The world2 bridge and the multi-agent run drivers.
- `report.md`: what was built, every validation number, deviations from the numpy reference and why, speed and memory
  per agent, what is ready for long runs and sweeps, and open issues.

Write `report.md`, and **also return the full report inline in your final reply**: the harness may block the write.
Label claims PROVED / SIMULATED / CITED / ASSUMED. Write in English and end with a short plain Chinese summary
(summary_zh).

If the work is too big for one go, finish P0–P5.2 cleanly first and report. A correct one-agent GPU match is worth more
than a fast but unvalidated 64-agent run.
