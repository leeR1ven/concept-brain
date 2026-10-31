(Saved by the coordinator from the reviewer's final reply; the harness blocked the reviewer's own write.)

# world2: an open house and garden where many models live with scripted caregivers

world2 v1 works end to end in `arch/v2/bench/world2/`. Up to 64 models live without resets in a 24×24 house and garden with three scripted caregivers (妈妈 爸爸 奶奶), on the bench's real-model sound and mouth physics.
- **API:** batched (`agents.step(obs) -> (N, 41)`, one model drives all bodies) and per-agent (`PerAgent`, which gives identical lives).
- **Selftest:** 25/25 PASS. Anti-cheat holds and reference numbers exist.
- **Speed:** 118,000–125,000 ticks/s with 16 trivial agents on one core of the shared box, so the ≥ 100,000 target is met (SIMULATED: `checks/selftest.out`, `checks/refs.out`).
- **Bench untouched:** no bench file was changed; the selftest checks the v1.2 modules and the life-mode files.
- The harness blocked `report.md`, as the brief expected, so this reply is the report.

## The draft, and what I changed
The previous agent left a first-draft DESIGN.md and a 3-line stub. I kept the draft's architecture and built it: grid house and garden, bench mouth plus 9 motor units, shared sound field, retinotopic egocentric eye, state-driven caregivers, in-life records and references. Testing forced these changes (now in DESIGN.md, marked [changed]):
- **Eye:** the block is 64 units, not 48; 48 could not hold the listed features.
- **Movement, so the house cannot seal itself:**
  - Doors are 3 cells wide, the bed sits in a corner, and caregivers rest only on open cells and avoid doorways.
  - Caregivers step over objects and swap places with an idle caregiver in their way.
  - A caregiver blocked for 2 steps by a child squeezes past; the child is nudged one cell and the nudge is logged.
  - Effect (SIMULATED): before, silent agents saw 1,474 failed walks against 409 arrivals in 4 days. Now there are 0–6 failures per 10 days out of 1,900–3,800 walks. Nudges are 0 per day with static agents and 0.1–9 with moving ones.
- **Requests** are made only when the agent's body can carry them out (a path check). Success then measures comprehension, not obstacles: the oracle carries out 99–100 %.
- **Memory probes** only ask about facts told before the current episode, at least 300 ticks ago, and the same item is not asked again within 3,000 ticks. The first version asked the same novel object 4 times in one morning.
- **Food:** after handing food over, a caregiver doesn't ask for it back for 600 ticks. Before, agents that never eat held food forever and requests collapsed.
- **Measures:** "talking over the partner" counts only the caregiver engaged with the agent; overlap with any caregiver is reported separately. Agent-agent turns count only utterances that are not replies to a caregiver.

## What world2 is
**Space.**
- Rooms: kitchen 厨房 (stove, table), bedroom 卧室 (shelf, bed), living room 客厅 (table, sofa), garden 花园 (trees).
- Objects: the bench's 12 everyday objects plus 38 novel kinds. Each instance has a colour, a count of 1–3 and a size, may be edible, and its kind has a usual room.
- Animals wander, there are 3 meals a day, caregivers tidy, agents carry things.
- A day is 24,000 ticks plus a 6,000-tick night (dark, caregivers asleep).

**Body.** The action vector has 41 units.
- 32 mouth units: the bench Voice, mouth rule and onset rule; the own voice is heard one tick later.
- 9 motor units: forward, back, left, right, turn left, turn right, take, put/give, eat. They have no built-in meaning, and an option permutes them per agent.
- An action keeps the body busy for 3 ticks. One optional generic reflex exists: `Explore`, a random twitch after 40 idle ticks.

**Senses** (read-only buffers):
- **ear (N, 8, 6):** one shared sound field, louder nearer and weaker through walls. 妈妈 uses the bench partner's voice, so an agent next to her hears the bench's windows exactly.
- **eye (N, 1600):** egocentric 5×5 view with walls occluding; 25 blocks of 64 units with the same code everywhere (terrain, kind, colour, count, size, held object, person, pointing).
- **touch (16), taste (8), hunger, reward.** Reward comes from eating, praise, imitation, answered initiatives and greetings.

**Caregivers.** Every line is generated from the current state.
- An episode with an agent mixes:
  - naming with pointing;
  - questions about what it sees or holds, and food offers;
  - memory probes: X在哪里, novel names, 我是谁, 你叫什么;
  - requests: 去拿X, 把X给我, 名字过来;
  - one-shot novel objects and facts.
- Each question is a probe, a model (she answers herself) or a recast (praise or the answer). Held-out pairs are never answered.
- Caregivers chat with each other where agents can overhear, 20 % of it adult-corpus lines.
- Contingent reactions to any agent: greeting → 你好; naming a visible object → 对是X; her name → 怎么了 and she comes; 这是什么 → she names it; 要X → she fetches it; hunger → she brings food.

**Measurement:** `evaluate()` reports per-task learning curves; scene (new situations only), held-out, memory (retention by delay × encounters) and social; comprehension per request type; initiative, vocabulary over time, agent-agent turns, overlaps and stability.

## Selftest: 25/25 PASS (SIMULATED, `checks/selftest.out`, 36 s)
**Physics and sound:**
- Voice and mouth rule equal the bench's: 3,970 chars, 0 mismatches on 2,000 random muscle sets.
- An agent next to 妈妈 hears exactly the bench Stream (24 of 24 ticks identical).
- Energy by distance 1/3/5/7/9 cells: 1146 / 802 / 670 / 538 / 0; through a wall at 2 cells: 670.
- Noise σ = 1 gives a silent-window energy of 18.4 (bench ≈ 19).
- Own voice one tick later; a neighbour hears it, far agents don't.

**Body and eye:** motor semantics, 3-tick busy, bump, take / eat / give (reward = hunger before eating, taste lasts 6 ticks); eye codes, pointing, turning, occlusion; nights dark and silent.

**World:** determinism; batched equals `PerAgent`.

**Anti-cheat:**
- Echo and lookup score 0.00 on scene, held-out, memory and requests.
- The oracle scores 1.00 on scene (552), held-out (97), memory (297), social (77) and requests (300).
- The random actor carries out 0.04 of requests.

**Content rules:**
- Held-out pairs are never answered (97 questions, 0 answers heard).
- All 401 object questions point at something in the agent's view.
- Memory delays run from 302 to 112,105 ticks.
- Caregivers say one line at a time, and the oracle never talks over its partner.
- Attention is fair (396–407 questions per agent).
- All 4 agents get rewarded initiatives from all 3 caregivers.
- Agent-agent turns are detected correctly.

**Speed:** 124,660 ticks/s with 16 trivial agents.

## Reference numbers (SIMULATED, `checks/refs.out`, `results/*.json`; 10 days per run, seeds 0–2, 4 agents)
| agent | free acc | scene (new) | held-out | memory | social | requests (fetch / give / come) | appropriate initiatives per agent per day | vocabulary |
|---|---|---|---|---|---|---|---|---|
| silent | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0 | 0 |
| random actor | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.07±0.01 (0.05 / 0.15 / 0.15) | 0.04 | 0 |
| echo | 0.13 | 0.00 | 0.00 | 0.00 | 1.00 | 0.00 | 0 | 2.9 |
| lookup | 0.04 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0 | 1.9 |
| namer | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 7.3±1.0 | 24.7 |
| oracle | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 (1.00 / 1.00 / 1.00) | 0 | 60.8 |

- The oracle scores 1.00 on all 15 tasks. Echo's 1.00 on social is inherent (greetings repeat the prompt), as in life mode.
- Each run has about 640 scene, 70–170 held-out, 400–600 memory and 410–650 request items.
- Input with 4 agents: about 94 caregiver lines addressed per agent per day (150 with the oracle), caregiver speech audible 8–23 % of daytime, about 32 scored questions per agent per day. With 16 agents: 25–38 lines and 10–14 questions per agent per day.
- With 16 agents (seed 0) the pattern holds: oracle 1.00 on every group and 0.99 on requests; echo and lookup 0; random actor 0.05.

## Speed (SIMULATED; trivial agents, 150,000 ticks, best of 2, shared 4-CPU box)
| agents | ticks/s | µs per tick |
|---|---|---|
| 1 | 192,000 | 5.2 |
| 4 | 145,000 | 6.9 |
| 16 | 118,000 (selftest: 125,000) | 8.4 |
| 64 | 73,000 | 13.8 |

Active agents cost more:
- Oracle: 58,000 ticks/s with 16 agents, 70,000 with 4.
- 16 random actors babbling and moving all the time: about 10,000 ticks/s (0.1 ms per tick), still far below any learning agent's step.
- Ear noise > 0 makes every tick a sound tick: about 30,000 ticks/s.

The speed comes from event-driven caregivers and animals, sound computed only while something sounds, eye blocks rewritten only for the (agent, cell) pairs that changed, and one bytes comparison to test the action array.

## Bugs found and fixed during the build (all in my code)
1. **Eye:** cells never touched since start-up showed the first registered cell's code. The selftest caught it; caregivers and the oracle never read the eye.
2. **Scoring:** a question tag overwrote the record's exchange kind, so many scene and where-questions went unscored (scene n rose from 280 to 600 after the fix).
3. **Walking:** the blocked counter added up over the whole walk instead of counting consecutive blocks, so long walks failed.
4. **Closed pockets and doorway congestion** (see the changes above).
5. **Unreachable agent:** the doorway-avoidance goal could leave only a furniture cell, so one agent went unattended for 2 days.
6. **Food give-back loop** (see the changes above).
7. **Chat hold:** the chat partner was released at once instead of held. She then spoke chat lines inside her own episodes, which showed up as the oracle talking over its partner.
8. **Ear noise:** with noise > 0 it piled up inside a shared zero block. Default runs were unaffected.
9. **Initiative category:** utterances that interrupted their caregiver kept the label "naming"; they are now "interrupt".

## For the coordinator
- **No bench bug found.** world2 imports the bench's `physics.py` and `metrics.py` by path under private module names, so it cannot clash with agents' own copies.
- **Comparisons between agents are statistical.** In an open world the content depends on the agents (unlike life mode). A fixed seed with the same agents gives an identical life.
- **Nudges** move an agent one cell without its action. They are rare and logged; agents that integrate their own path should treat them as an outside push.
- **Suggested default for prototypes:** `World2(n_agents=4, seed, tutor='mixed')` for at least 10 days.
  - Bench agents can live in world2 through `PerAgent([MouthOnly(agent), ...])`.
  - Lawful physics, noise, `mouth_vocab` and `motor_shuffle` all run (smoke test: oracle 1.0 under each). The reference numbers use random physics and no noise.

## Known limits
- No furniture words or places yet: "where" answers are rooms.
- Agents can't point or gaze.
- No "让一下" (make way) requests in place of nudges.
- No sleep semantics at night.
- No lawful-physics or noise reference numbers.
- No GPU-batched reference agent.
- The rule that agents use only `obs` can't be enforced by the world.

## Files
All in `arch/v2/bench/world2/`:
- `core.py`: mechanics, event loop, caregiver driver
- `care.py`: caregiver scripts and reactions
- `lex.py`: lexicon, layout, held-out sets, codes
- `agents.py`: wrappers and reference agents
- `metrics.py`
- `selftest.py` → `checks/selftest.out`
- `refs.py` → `results/*.json`, `checks/refs.out`
- `README.md`: usage and full specification
- `DESIGN.md`: the revised design note
- `TASK.md`

## summary_zh
world2 已完成并测试通过：
- **世界：** 24×24 的房子加花园，最多 64 个模型可以一直生活、不重置，有妈妈、爸爸、奶奶三位按脚本行动的看护人。
- **身体：** 嘴和声音沿用基准的真实物理，另有 9 个没有预设含义的动作单元（走、转身、拿、放/给、吃）。
- **感官：** 共享声场（越近越响、隔墙变弱）、以自己为中心的 5×5 视野、触觉、味觉、饥饿和奖励。
- **看护人：** 每句话都按当下状态生成：指物命名、提问、请求（去拿、给我、过来）、记忆题（东西在哪、新东西叫什么、我是谁、你叫什么）、彼此聊天（含成人语料），并回应任何主动说话的模型。
- **接口：** 支持批量和逐个两种调用。
- **自检：** 25/25 通过。
- **防作弊：** 复读和查表在新情境题、留出题、记忆题和请求上都是 0；全知参照全部为 1.0。
- **速度：** 16 个空模型约每秒 12 万拍，达到目标。
- **修复与基准：** 修复了自己代码里的 9 个问题；基准没有发现错误，也没有改动任何基准文件。