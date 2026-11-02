# Task: open-world hearing (Step 6, first part) — the agent must hear in world2

## The problem (embodied reviewer, `arch/v2/review2/embodied/report.md` §1–3, SIMULATED there)
- In world2 at the bench's own noise level (σ = 1), the real ARCH_v2 agent hears **0 lines** in 15k ticks. It stores
  0 episodes, makes 0 awake recalls and says 0 utterances. Its syllable categories fill their cap of 256.
- Even without noise, 16–64 talking bodies keep the brake on for 33–53 % of the day. They remove every silence of
  200 ticks or more, so triggers that need exact silence never fire.

So the ear and the silence-based triggers were fitted to a quiet two-person bench. The future open world has many
models, noise and overlapping voices (the designer's vision).

## Goal
Make the front end work in world2 without breaking the bench. Use local, brain-like mechanisms. Candidates, from the
SYNTHESIS Step 6:
- an adaptive noise floor;
- divisive gain control (normalisation);
- voice-change boundaries: a line ends when the voice changes, not only at silence;
- relative-silence or energy-drop triggers in place of exact silence.

Also consider speaker separation and trust:
- v2.1 has a perceived voice-category hook (`v21/core_v21.py`, `voice=1`; purity 0.91 partner / 0.72 other);
- trust-weighted listening: adult prior, learned reliability.

## Read first
- `arch/v2/review2/SYNTHESIS.md` (§4 Step 6).
- `arch/v2/review2/embodied/report.md`, with its probes `noise_agent.py` and `silence_probe.py`.
- `arch/v2/bench/world2/` (`README.md`, `DESIGN.md`, `report.md`).
- `arch/v2/synth/ARCH_v2.md` (the ear, the syllable categories Z, and the line and turn triggers).
- `arch/v2/v21/report.md` (the current core).

## Phase 1: design
Write `design.md`: which mechanisms, why they are local, and what they predict. Attack them first:
- Will they still segment the bench's clean lines exactly as now?
- Will categories still form?
- Do they stay stable over one unbroken life?

## Phase 2: test
1. **Ear module alone** (frozen PFC). At world2 noise 0 / 0.5 / 1 and with 2 / 16 / 64 bodies, measure:
   - lines detected vs true lines (precision and recall);
   - syllable category purity;
   - the number of categories;
   - silence-trigger opportunities.
2. **The full v2.1 agent in world2** (`noise_agent.py`-style). Measure lines heard, episodes, recalls and utterances, at
   noise 1 and with crowds.
3. **Regression on the bench.** With the new front end, run v2.1 paired on seeds 0–4, frozen protocol (v21's
   `run_v21.py`). J and J2 must not fall beyond noise. Use the canonical DG tie rule (`gpuport/run_j_canon.py`); the tie
   rule alone moves J by up to 0.07 per seed.

Work on a copy of the v2.1 code in your folder. Another agent (`arch/v2/v22/`) is changing the PFC side; keep your
change to the front end and its triggers, behind config keys, so it merges cleanly.

## Principles (binding)
- One unbroken life.
- No sleep.
- Local rules only.
- The agent reads observations only, never world state.
- No puppeting.
- 0 mechanics errors.
- Do not change the bench's metric code or world2's physics. A world2 setting such as noise level or crowd size is fine.

## Compute
- **Local box:** 4 cores and 3 GB RAM, shared. At most 1 local python process, under 700 MB; run `free -m` first.
- **Server:** the coordinator's runner only (`arch/v2/gpu/README.md`); you never SSH.
  - Use `queue_a800/` for numpy jobs.
  - Name jobs `hearing_<label>`; keep at most 2 of your jobs queued or running.

## Deliverables (in `arch/v2/hearing/`)
- `design.md`;
- the code;
- the results;
- `report.md`, with the tables, what worked, what failed, and the config keys to merge.

**Return the full report inline in your final reply**, because the harness blocks report writes. Label claims PROVED /
SIMULATED / CITED / ASSUMED. Write in English and end with a short plain Chinese summary (summary_zh).
