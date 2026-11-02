# Task: a learned meaning layer (Step 4 of the review-2 plan; IDEAS #3, #5, #14)

## The problem
Every version so far fails at **class-based generalisation**: "a new food word should get 好吃 because it is a food". On
the class probe (`gen_probe2.py`) every core scores 0.25–0.5 (chance level):
- ARCH_v2;
- v2.1 with role-aware learning;
- all random-at-birth conjunction layers.

The cause is how words are represented:
- Items are discrete syllable IDs.
- randmix gave each item a fixed sparse code made from its sound (`randmix/randlayer.py`, fly-hash of the Z template).
  Sound gives no meaning overlap: 香蕉 and 苹果 share no more code than 香蕉 and 桌子.
- One misheard syllable costs about 30 % accuracy in every layer, so there is no fault tolerance (IDEAS #14).

The designer's ideas:
- **#3:** a meaning layer.
- **#5:** "the middle layers also learn similarity".
- **#14:** neuron codes are fault tolerant.

## Read first
- `arch/v2/review2/SYNTHESIS.md` §4 (Step 4) and the neuro reviewer's change #4 (`review2/neuro/report.md`).
- `arch/v2/randmix/report.md`, all of it. Its §6 recommends feeding learned item codes into the random layer's input
  lines. The layer is the right substrate: codes enter as line patterns, so keys do not explode.
- `arch/v2/v21/report.md` (role-aware learning; Step 4 must be measured on top of it, otherwise the role confound hides
  the effect).
- `arch/v2/review2/learning/gen_probe2.py` and its v21 / randmix versions.

## Phase 1: design (no training before this is logically sound)
Propose at least 3 mechanisms by which item codes become similar for items used in similar contexts. Use local rules,
learned online in one unbroken life. Examples to consider:
- context-predictive Hebbian codes (an item's code moves towards the codes of what precedes or follows it);
- a competitive / k-WTA layer over (item × context) co-activity;
- codes taken from the hippocampal NEXT/episode statistics during awake recall.

Then attack each one:
- **Representational drift.** When item codes change, every readout built on them breaks. How is this kept stable:
  slow rates, a critical period, mixing a fixed sound part with a learned meaning part, or readout tolerance?
- **Collapse.** Can all codes become the same?
- **Locality** and **one life**.
- Does it give fault tolerance for misheard syllables?

Pick one or combine them, and write the argument down (in `design.md`) before building.

## Phase 2: test
1. **Module test first.** Extend the symbolic class probe (`gen_probe2.py`) to the chosen mechanism. Targets:
   - class-dependent held-out ≥ 0.8 (prediction from SYNTHESIS Step 4);
   - seen items unchanged;
   - fault tolerance: accuracy drop for one similar-sounding / random syllable, against today's 0.27–0.38.
2. **Then staged runs.** Use randmix's RandCore (`c_flat`) on top of the v2.1 core.
   - Another agent (`arch/v2/v22/`) is merging v2.1 with the random layer right now. Until that merge exists, make the
     smallest combination you need in your own folder, and do not touch theirs.
   - Measure paired on seeds 0–4, frozen protocol (v21's `run_v21.py`): J, J2, per-task held-out with references, drift,
     units, and the fault probe (randmix `post.py`).
3. **Scaling check.** Does a bigger random layer help once codes carry meaning (4×)? randmix found no gain without
   meaning. Use torch on the GPU (`queue_a800gpu/`) if it makes the 4× run fit the 90-minute limit; `randmix/randtorch.py`
   is validated.

## Noise
- DG ties make J platform-dependent: up to 0.07 per seed. Use the canonical tie rule (`gpuport/run_j_canon.py`, `dg`).
- Compare only paired runs on the same machine.

## Principles (binding)
- One unbroken life with learning always on.
- No sleep.
- Continuous thought.
- Local rules only.
- No puppeting.
- 0 mechanics errors.
- Every learned part reaches behaviour (lesion it).

## Compute
- **Local box:** 4 cores and 3 GB RAM, shared; at most 1 local python process under 700 MB; run `free -m` first.
- **Server:** the coordinator's runner only (`arch/v2/gpu/README.md`); you never SSH.
  - `queue_a800/` for numpy jobs, `queue_a800gpu/` for torch jobs.
  - Name jobs `meaning_<label>`. Keep at most 2 of your jobs queued or running at once.

## Deliverables (in `arch/v2/meaning/`)
- `design.md`;
- the code;
- the results;
- `report.md` with the tables, what worked and failed, and a recommendation.

**Return the full report inline in your final reply**, because the harness blocks report writes. Use the labels
PROVED / SIMULATED / CITED / ASSUMED. Write in English and end with a short plain Chinese summary (summary_zh).
