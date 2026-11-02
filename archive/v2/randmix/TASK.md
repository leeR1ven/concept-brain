# Task: random connections from birth, trained on top (the designer's idea, 2026-10-31)

## The idea (designer)
"模型可以像小孩一样先随机生成很多连接，在随机连接的基础上训练。而不是在空模型上慢慢加连接。"

In English: like a child, first grow many random connections, then train on top of them. Do not start from an empty
model and add connections one by one.

**Biology.** In the infant cortex, synapses are overproduced and peak in early childhood at about 50 % above the adult
level. They are then pruned by use: connections in use stay and strengthen, unused ones are removed (CITED:
Huttenlocher 1979; Changeux & Danchin 1976).

## Why it matters now
ARCH_v2's conjunction layer MIX does the opposite: it recruits one new unit for every new exact tuple of discrete item
IDs (`MixDict` in `rules.py`, key schema in `core_synth.py` ~135–180). Three independent reviewers traced most of
ARCH_v2's limits to this:
- No fault tolerance. One misheard syllable changes every key it is part of.
- No unit stands for "this frame, whatever the filler". Held-out taste is 0/30 and held-out check is 0.24, below chance.
- The unit count grows without bound. A bigger model only memorises more combinations.
- It is a poor fit for the GPU: a hash dict and collisions.

See `arch/v2/review2/skeptic/report.md` §2.2 and §2.10, and `arch/v2/review2/neuro/report.md` §1 and change #4.

**Caution from history.** v1 was a *fixed* random network with a Hebbian readout. It scored 0.57 on familiar sentences
and was at chance on never-heard ones. So "random" alone is not the claim. The claim is random at birth, then trained on
top, plus pruning.

## What to build and compare
Copy ARCH_v2 (`arch/v2/synth/`, config `final_noctx`; read `ARCH_v2.md` first, all of it) into your folder. Replace
MIX with a layer generated at birth:
- N units, each with a random sparse fan-in from the input groups that MIX keys pair today: held thought items, the last
  item, position or role, scene focus or eye slot, and so on.
- Activity by threshold or k-WTA, at a fixed active fraction.
- GEN and the gates read these units exactly as they read MIX now.

Items are discrete IDs today. Decide how they enter the random layer, for example each item as a sparse random code, or
via Z's acoustic features, so that overlap can exist. Explain the choice.

**Variants:**
- (a) Random and fixed: the v1-like control.
- (b) Random + local training of the fan-in weights: co-activity strengthening with normalisation or competition.
  Aim for inputs used in the same contexts to come to share units. This is IDEAS #5's "the middle layer also learns".
- (c) Like (b), plus pruning of long-unused synapses, optionally with random regrowth ("use it or lose it"; the total
  synapse count is fixed or declining, as in development).
- (d) The ARCH_v2 dict MIX, as the reference.

Sweep a little: N at 1× and 4×, fan-in, and active fraction.

## Measure (same seeds, same bench; the bench copy and metrics stay unchanged)
- J and the per-task scores.
- **Held-out per task:** taste, check, offer, slot, where, what, colour and count. Put the trivial references from the
  same run next to them: echo, and a constant answer per task type.
- Drift after 300 untutored dialogues, and 900 if cheap (see `review2/skeptic/drift_own.py`).
- **Fault tolerance:** replace one syllable of the prompt with a random syllable, or a similar-sounding one, and measure
  the accuracy drop.
- Units and synapses over time, memory, and speed. Also note what this means for the GPU port.
- Seeds 0–1 first. If a variant beats (d) on held-out without losing more than 0.03 on F, confirm it on seeds 3–4.

## Principles (binding)
- One unbroken life with learning always on.
- No sleep.
- Continuous thought.
- Local learning rules only.
- No puppeting.
- 0 mechanics errors.

## Compute
- The local box has 4 cores and **3 GB RAM**, shared with other agents.
- Run at most 2 python processes at once, each under 700 MB. Run `free -m` before each start. Use `nice` and
  `OMP_NUM_THREADS=1`.
- One staged J run takes about 1,000 s.
- You may also use the server CPUs through the coordinator's runner. Read `arch/v2/gpu/README.md` and follow it
  exactly; you never SSH. Name jobs `randmix_<label>` and keep at most 1 of your jobs queued at a time. The slots are
  shared, so it may wait.

## Deliverables (in `arch/v2/randmix/`)
- the code;
- the result JSONs;
- `report.md` with the comparison tables, what worked, what did not, and a recommendation for the next design.

Write `report.md`, and **also return the full report inline in your final reply**, because the harness may block the
write. Label claims PROVED / SIMULATED / CITED / ASSUMED. Write in English and end with a short plain Chinese summary
(summary_zh).
