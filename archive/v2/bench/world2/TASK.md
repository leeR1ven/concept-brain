# world2 task (from the coordinator)
Designer's request: "Can we build a simulated world on the server, like the free real world, let the models live in it
continuously, and put many models in it to learn?"  Build `arch/v2/bench/world2/` (new package). Keep v1.2 and life
mode unchanged; reuse voice physics (lawful option), eye codes, metrics and anti-cheat ideas. Version 1: simple, solid.
- Space: small 2-D house + garden grid (~24×24, rooms); objects with properties (kind, colour, count, size, edible,
  movable), usual places, moved by others, food appears; day/night.
- Bodies (one per model): the existing mouth (32 muscles, same Voice/Stream) + a small motor set (move 4 directions,
  turn/gaze shift, take/put, eat) as extra "muscles" with no built-in meaning; only generic innate reflexes (e.g. random
  exploration when idle), each documented.
- Senses: ear = shared sound field (speakers within range, louder nearer, mixed by max incl. own voice); eye = egocentric
  retinotopic view (e.g. 5×5 cells) coded like the bench's retino eye; taste/touch when eating/holding; hunger as input;
  reward from eating and caregiver praise.
- Caregivers: 2–3 scripted named adults who move and do things (cook, tidy, play), talk mostly about the here-and-now
  (name what they point at, ask questions, make requests like "去拿苹果", "把球给我"), react contingently to agents'
  utterances and actions (recasts, praise, answers, giving food), talk with each other (overheard). Language = micro-world
  lexicon + occasional corpus lines. Scripts generated from world state, not fixed lists.
- Many models: N agents in one world, see/hear each other; caregivers respond to whoever addresses them. API must support
  a BATCHED agent `agents.step(obs_batch) -> actions_batch` (one GPU model drives many bodies) and per-agent loops.
- Measurement (embedded, no resets, scored in life like life mode): naming/QA, comprehension via action (request → right
  action), initiative, vocabulary over time, memory across days (delay × encounters), agent-agent talk, overlaps,
  stability over long runs. References: silent, random actor, echo, scripted oracle; echo/lookup must fail on scene,
  comprehension and memory items.
- Speed: world must not be the bottleneck; vectorise; target ≥ 1e5 world ticks/s with 16 trivial agents on one CPU core.
- Process: short design note (world2/DESIGN.md — a first draft exists), implement, selftest, reference numbers,
  README; no long jobs on the server CPU; coordinator commits.
