# Task: ARCH_v2.1, the first fixes from review round 2 (Steps 0–3)

## Background
Five independent reviewers reviewed ARCH_v2. Read **`arch/v2/review2/SYNTHESIS.md`** first, then the five reports in
`arch/v2/review2/<lens>/report.md`. ARCH_v2 itself is described in `arch/v2/synth/ARCH_v2.md`; its config tag is
`final_noctx`. Copy the code into your folder and work on the copy.

All five reviewers found one root cause: learning by listening does not know whose turn it is. The agent's own speech
enters neither the thought nor the hippocampal sequence, so the line heard after its own answer is learned as the
answer. Two local fixes already exist:
- **Xcut** (learning reviewer: `review2/learning/gen_probe.py`, symbolic): held-out 0.00 → 0.72–0.80.
- **Answered-line veto** (develop reviewer: `review2/develop/veto_j.py`): seed 0, nS 600, J 0.70 → 0.93, drift loss
  0.43 → 0.03. Costs colour and ritual.

## Do these steps in order, one at a time, each kept or rejected on evidence

**Step 0. Honest measurement.** Do this before any change.
- Add a frozen held-out protocol: `held_frac` 0 in the training worlds, so held-out pairs are probed only on frozen
  copies. Keep the old protocol for comparison.
- Report:
  - per-task held-out;
  - a role-confusion rate (replies that are partner lines or echoes);
  - same-run trivial references: echo, a constant answer per task type, and the random namer in life (see the bench's
    baselines and `life_refs.py`).
- Propose a revised objective in which H weighs as much as F, and report both the old J and the new one.
- Do not change the bench's own metric code. Your additions live in your drivers.
- Keep `gen_probe.py`, `gen_probe2.py` and the develop reviewer's D1–D3 probes (`heldout_replies.py`, `comprehend.py`,
  `vr_what.py`) as regression tests.

**Step 1. Role-aware learning with the self in the loop.**
- Unify Xcut and the veto into one local mechanism:
  - own utterances enter the thought with a role tag, and enter the hippocampus, so NEXT links follow the real dialogue;
  - a heard line's learning as an answer waits on an eligibility trace until its role is known:
    - if I answered it, it was a question or initiation; cancel its answer learning, and optionally re-apply it to an
      "initiate" role key;
    - if it is contingent on my reply (overlap with my efference copy, or a reward), it is feedback or a recast; keep it;
  - "moving on" must not be punished as "predicted but absent".
- Speaker identity: add voice categories, if the bench or life gives distinct voices; otherwise make it a hook for
  world2.
- Find the cause of the ritual and colour costs.
- Confirm on seeds 0–4 at the normal nS, plus one 10⁶-tick life.
- Reviewer predictions to test:
  - J gain ≥ the seed SD (0.06);
  - drift loss ≤ 0.05–0.10, also after 900 untutored dialogues;
  - frozen H ≥ 0.45;
  - life nonsense initiatives −50 %.

**Step 2. Comprehension feeds production.**
- Add a "say what I recognise / attend to" copy route: VR's word at the attended slot, the slot's location word, and
  the colour and count words.
- Its gates are keyed only on frame context (`gate_noid`), not on the candidate's identity.
- Self-initiated thoughts read the held thought, so favourite thoughts can reach speech.
- Evidence of the gap: correct attention 1.00 and correct recognition 0.73 on held-out items, yet correct replies only
  0.00–0.02.
- Prediction: on top of Step 1, held-out where/slot ≥ 0.6, what ≥ 0.7; life namings rise from 18.

**Step 3. A signed value loop.**
- Dips: δ = −V when an expected reward or reaction does not come (for example 50 ticks with no reaction).
- A NoGo veto on release instead of V_go's gain.
- The actor and value in their own slow table, not inside GEN.
- Actor eligibility that expires.
- Innate coarse social detectors: person present / arrived / left, and addressed to me, from the eye-change parser in
  life.
- Prediction: 你好 to an empty room drops from about 65 to ≤ 15 per tenth of life; appropriate initiatives rise from
  0.19 to ≥ 0.4; social replies stay ≥ 0.30.

After each step, re-run the probes that the reviewers' claims rest on. If a reviewer's prediction fails, say so plainly.

## Binding principles (ARCH_v2 §9 and `/root/work/concept-brain/docs/IDEAS.en.md`)
- One unbroken life with learning always on.
- No sleep.
- Continuous thought (no clear/load).
- Speech caused by thinking.
- Local rules only: pre, post, own trace, and at most one broadcast signal.
- No puppeting.
- 0 mechanics errors.
- Every learned part reaches behaviour (lesion it).
- Favourite thoughts are tuned, not forbidden.

## Compute
- **Local box:** 4 cores and 3 GB RAM, shared with other agents. Run at most 1 local python process at a time, under
  700 MB. Run `free -m` first.
- **Server CPUs:** use them through the coordinator's runner. Read `arch/v2/gpu/README.md` and follow it exactly; you
  never SSH.
  - Name jobs `v21_<label>`.
  - Keep at most 2 of your jobs queued or running at once. Other agents share the 3 slots, so jobs may wait.
  - Each job ≤ 40 min with ≤ 4 processes. One staged run takes about 28 min on one server core; split stages if
    needed.

## Deliverables (in `arch/v2/v21/`)
- the code;
- the result JSONs;
- `report.md`: per step, what was built, the tables (with trivial references and per-task held-out), the lessons, what
  was kept or rejected, and the final ARCH_v2.1 config.

Write `report.md`, and **also return the full report inline in your final reply**, because the harness may block the
write. Label claims PROVED / SIMULATED / CITED / ASSUMED. Write in English and end with a short plain Chinese summary
(summary_zh).

If the work is too big, finish Steps 0–1 cleanly with 5 seeds and report. Then continue with Steps 2–3 if you can.
