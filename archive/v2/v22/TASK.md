# Task: ARCH_v2.2 — merge the validated pieces, then fix learning from corrections in life

## Start from
- **ARCH_v2.1**: `arch/v2/v21/` (read `report.md` fully; final config in its §7, including `rec_fix=1`). Copy its code
  into your folder and work on the copy.
- **The random-at-birth conjunction layer**: `arch/v2/randmix/` (read `report.md`; code `randlayer.py`, `rcore.py`). Its
  best form is `c_flat`: equal populations per pair type, instar training, use-it-or-lose-it with regrowth.
- **Noise facts** (`arch/v2/gpuport/report.md` §P0):
  - The DG k-WTA breaks exact ties differently on numpy 1.26 and 2.1.
  - The tie rule alone moves J by up to 0.07 per seed.
  - Compare only paired runs on the same machine and numpy version, judged on 5-seed means.

## Steps, in order. Keep or reject each on paired 5-seed evidence (frozen protocol, J and J2, as v21 reports them)

1. **Canonical tie rule.** Adopt the stable index sort in the DG k-WTA (`gpuport/run_j_canon.py`, `dg`). Re-baseline
   ARCH_v2.1 on seeds 0–4. This is the new reference; every later comparison is paired against it.
2. **Random-at-birth layer.** Replace MIX with randmix's `c_flat` inside the v2.1 core.
   - Fix its colour cost (colour seen 0.86 → 0.74 in randmix). Try:
     - an allocation between flat and MIX's, for example ∝ √c_t;
     - or k_t per population scaled by input dimensionality.
   - Re-check the v2.1 regression tests (`v21/regress/`), including the symbolic `gen_probe_v21.py`.
3. **Learn from corrections in life (the top open problem).** In life the agent speaks right after 79–87 % of the
   partner's corrections, so role-aware listening cancels them as "questions". Life answer accuracy is therefore stuck
   at 0.34, against 0.62 for the echo-or-constant responder.
   - Find a local, perceivable way to tell a correction or recast from a next question.
   - Candidates (v21 §8 lists more; v21 already rejected overlap contingency and a reward-trained question detector):
     - timing relative to my own utterance;
     - less compulsive speech right after a reaction (`u_R1` < 1, together with the veto);
     - a second look: a line that follows my utterance and is followed by my *repeating it* is a correction I took up;
     - the class codes of Step 4 (another agent is designing them now), only as a later option.
   - **Functional target:** life free replies above the same-life echo-or-constant reference, on seeds 5 and 6, without
     losing v2.1's life gains (empty-room greetings ≤ 15 per tenth, social ≥ 0.30, appropriate ≥ 0.38).
4. **Step 3's staged cost.** v2.1's dips cost check: held-out 0.80 → 0.43. Try smaller dip steps, or dips only where
   P_oth predicted a reaction that did not come. Target: keep the life gains with no staged cost beyond noise.
5. **Long untutored use.** Measure retention after 3,000 untutored dialogues on the final config. Today the horizon is
   only measured to 900. Report it; do not build a consolidation mechanism yet.

## Measure (same as v21; the bench's metric code stays unchanged)
- Staged, seeds 0–4, frozen protocol:
  - J and J2;
  - per-task held-out next to the same-run trivial references;
  - drift after 300 / 900 untutored dialogues;
  - role confusion;
  - lesions of every new part.
- Life, seeds 5 and 6 (10⁶ ticks): v21's life metrics and same-life references.
- Units, memory and speed of the new layer.

## Principles (binding; ARCH_v2 §9, `docs/IDEAS.en.md`)
- One unbroken life with learning always on.
- No sleep.
- Continuous thought.
- Speech caused by thinking.
- Local rules only: pre, post, own trace, at most one broadcast signal.
- No puppeting.
- 0 mechanics errors.
- Every learned part reaches behaviour.
- Favourites are tuned, not forbidden.

## Compute
- **Local box:** 4 cores and 3 GB RAM, shared. Run at most 1 local python process, under 700 MB; run `free -m` first.
- **Server:** use the coordinator's runner only. Read `arch/v2/gpu/README.md` and follow it exactly; you never SSH. The
  V100 is retired.
  - Use `queue_a800/` for numpy staged runs (4 processes per job, ≤ 90 min).
  - Name jobs `v22_<label>`. Keep at most 2 of your jobs queued or running; two other agents share the 4 slots.

## Deliverables (in `arch/v2/v22/`)
- the code;
- the result JSONs;
- `report.md`: per step, what was built, paired tables with references, what was kept or rejected, the final ARCH_v2.2
  config, and open problems.

The harness blocks subagent report writes, so **return the full report inline in your final reply**. Label claims
PROVED / SIMULATED / CITED / ASSUMED. Write in English and end with a short plain Chinese summary (summary_zh).

If the work is too big, finish Steps 1–3 cleanly and report.
