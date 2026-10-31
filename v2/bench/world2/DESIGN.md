# world2 — design note (v1: simple and solid) — revised after implementation

Goal: an open, persistent 2-D house + garden where N models live continuously with scripted caregivers, on the bench's
sound/mouth physics, with measurement embedded in life (no resets) and a world fast enough never to be the bottleneck.
This note is the first draft corrected to what was built and tested; **[changed]** marks departures from the draft and
the reason.

## Space
24×24 grid (`lex.LAYOUT`): kitchen 厨房 (top left, stove, table), bedroom 卧室 (top right, shelf, bed in the corner),
living room 客厅 (middle, table, sofa), garden 花园 (bottom, 3 trees). Walls; doors 3 cells wide **[changed: 2-wide
doors were sealed by two children standing in them]**. A cell holds terrain, at most one object (floor or furniture)
and at most one person. Agents are blocked by walls, furniture, objects and persons. **[changed]** Caregivers step
over objects (adults step over toys; objects alone made closed pockets in the small bedroom), swap places with an idle
caregiver in their way, and squeeze past a child who blocks them for 2 steps (the child is nudged one cell; logged as
an `event` 'nudge'; 0 with static agents, 0.1–9 per day with moving ones).
Objects: kind (12 everyday bench objects + 38 novel kinds = the bench's 40 minus 椅子 桌子), colour (6), count (1–3,
animals 1), size (small/big), edible (6 foods + 7 novel foods). Usual rooms: foods → kitchen, 皮球 → living room,
汽车 → bedroom, novel kinds → living room or bedroom (per seed), animals none. Caregivers tidy misplaced objects back
and say where they put them; 小狗 小猫 wander everywhere, 小鸟 鸭子 in the garden; meals (3 a day) put 2–4 food piles on
the kitchen table (uneaten food is cleared at the next meal); agents and caregivers carry things around.
Day 24,000 ticks (30 min at 75 ms/tick), night 6,000: the eye is dark, the caregivers sleep (absent, silent).

## Bodies (one per model)
Action (41,) per agent, > 0 = active: 32 mouth muscles (the bench's Voice: same chars, 4-muscle sets, mouth rule and
onset rule; own voice heard one tick later) + 9 motor units: forward, back, left, right (no turning), turn left,
turn right, take, put/give, eat. No unit has a built-in meaning for the agent (`motor_shuffle=True` permutes them per
agent). When the body is free, the strongest active motor unit starts its action at once and the body is busy for 3
ticks (one beat). take = the object in the cell in front (not animals); put = into the free cell in front, or into
the hands of the person in front (give); eat = one item of the held food. Innate scaffolds: none in the world;
`agents.Explore` is ONE optional generic reflex (a random motor twitch after 40 idle ticks), off unless used.

## Senses (batched, one row per agent; world-owned read-only buffers)
- ear (N, 8, 6): one shared sound field. Sources = every speaking person (caregivers: their char, 3 ticks each;
  agents: their mouth output of the previous tick). A listener hears each source with its levels lowered by
  ceil((d − 2)/2) (d = distance in cells; 0 up to 2 cells), +2 between rooms, nothing beyond 8 cells (6 between
  rooms); sources mix by max; own voice with the bench's `self` offset; 2 slices per tick as the bench Stream.
  妈妈 speaks with the bench partner's voice ('other'), 爸爸 with 'other2', 奶奶 and each agent with own offsets, so an
  agent next to 妈妈 hears exactly the bench's windows (selftest).
- eye (N, 25·64): egocentric 5×5 view (row 0 = own row incl. the own cell, rows 1–4 ahead, columns −2..+2), walls
  occlude (line of sight). One 64-unit block per view cell with the same code in every block (the bench's 'retino'
  idea) **[changed: 64 units instead of 48 to fit all features]**: terrain 12 (one-hot), kind 24 (4 of 24, pairwise
  overlap ≤ 2), colour 6, count 3, size 2, held 1, adult/child 2, identity 12 (3 of 12 per person), pointed-at 2.
  A person's cell also shows the object it holds (+ held bit). Dark at night.
- touch (N, 16): held kind (3 of 12) + held size (2) while holding; bump (1 tick after a blocked move); hand change
  (1 tick). taste (N, 8): 3 of 8 per food for 6 ticks after eating. hunger (N,) in [0, 1]: +1 per 20,000 ticks,
  −0.5 per item eaten. reward (N,): eating (= hunger before), praise / thanks for answers and requests (+1),
  imitation of a naming (+0.5), an appropriate initiative answered (+1), an acceptable greeting reply (+1).

## Caregivers (scripted generators, every line generated from the current state)
妈妈 爸爸 奶奶. Activities (weights 0.6 / 0.14 / 0.12 / 0.14): teach an agent (an episode), tidy, chat with each other,
rest (150–700 ticks on an open cell); 奶奶 cooks at meal times; bedtime: 睡觉了 + 晚安 to the agents in earshot.
Episode with agent i (attention rotates to the least recently attended reachable agent): call it from afar
(名字过来, a 'come' request) or walk up in front of it; 早上好 / 你好 on first contact of the day (social, scored);
2–6 turns separated by 20–150 ticks:
- naming (with pointing): 这是X / 看X / X / 红色的X / 两个X / 大X — an imitation of the name is praised (对, +0.5);
- a question about what i sees or holds (pointing at the object): 这是什么, X是什么颜色, 有几个X, X大不大, 这是X吗,
  你拿着什么, 这是哪里, 这是谁 (another person in view); 你要X吗 when i is hungry (then she hands the food over);
- a memory probe (always a probe): X在哪里 (an object out of view whose current place i saw or was told after its
  last move, ≥ 300 ticks ago), 这是什么 for a novel object named in an earlier episode, 我是谁, 你叫什么;
- a request (comprehension by action, scored by the outcome within 400 ticks): (名字)去拿X (an object i can reach),
  把X给我 (what i holds; not the food of a hungry child), 名字过来;
- a novel object: she brings a never-named kind, puts it where i sees it, points: 这是X (once);
- a fact: 我是妈妈 / 你叫小明 / 这是小红 (pointing at another agent).
Exchange kinds: held-out (task, kind) pairs (≈30 % of each task's everyday kinds, balanced as the bench) are only
probed and never answered by anyone; seen items: 15 % probes (no reaction), 30 % model (she answers herself after 3
ticks; `tutor='mixed'`; 'recast' = 0 %, 'model' = 85 %), else recast (correct → praise + reward; wrong or silent →
the answer). Answer timing = the bench's (≥ 45 ticks, until 6 silent ticks after the agent stops, cut at 60).
Chat (overheard by agents in earshot): X在哪里 → 在R, 名字在哪里 → 在R, 饭好了吗 → 好了/还没有, 这是什么 / X是什么颜色
about objects near them (never held-out pairs), 20 % adult-corpus pairs. The partner is held for the whole chat (an
early version released her at once, so she spoke chat lines inside her own episodes); a caregiver says one line at a
time (selftest invariant).
Contingent reactions to agents (the nearest awake, interruptible caregiver in earshot; words in brackets = category):
a greeting (greeting) → 你好 +1; naming a visible object (naming; once per kind per 3,000 ticks) → 对是X +1; her name
(calling) → 怎么了 +1, she comes and does a turn; 这是什么 (asking) → 这是X (held-out 'what' kinds: 你说呢); 要X / 我要X
(request) → 好 +1, she fetches X and hands it over; 饿 / 要吃 while hungry (hungry) → 好 +1, she brings food; an
unrequested give → 谢谢; anything else → nothing. Talking over the caregiver engaged with you gets no reaction.
**[changed]** Requests are only made when the agent's body can carry them out (path check), so the oracle's success
measures comprehension, not luck with obstacles.

## Many models
N bodies (1–64) in one world; they see and hear each other. Batched API: `world.run(agents, ticks)` with
`agents.step(obs) -> (N, 41)`; `PerAgent([...])` wraps per-agent `step(obs_j) -> (41,)` (selftest: identical lives);
`MouthOnly(bench_agent)` lets a bench agent live in world2 as a body that cannot move.

## Measurement (in life, no resets) and anti-cheat
Records `q` (question / social exchange with its window: reply, acc, held_out, new_situation, n_experienced,
n_asked_before, ticks_since_told for memory), `req`, `init`, `a2a`, `say`, `eat`, `event`, `mon` (per-agent counters
every 10,000 ticks). `evaluate()` pools agents and reports per agent: per-task accuracy and learning curves, scene
(new situations only: this prompt + exact eye input never followed by a taught answer for this agent, so exact lookup
cannot answer), held-out, memory (and retention by delay × encounters), social, comprehension per request type (+
named calls answered by others), initiative (categories, appropriate, reward), vocabulary over time, agent-agent
turns, talking over the partner, latency, imitation, stability per time bin. References: silent, random actor, echo,
lookup (prompt + exact eye input → the reply heard taught), namer, oracle (world state + path planning). Selftest:
echo and lookup 0 on scene, held-out, memory and requests; oracle 1.0 on every question group and ≥ 0.95 (measured
0.997–1.00) on requests; random actor at chance (requests 0.03–0.07).

## Speed
Event-driven: caregivers, animals and clocks are heap events; the sound field is computed only while something sounds
(per-utterance attenuation tables, a sliding ear buffer); eye blocks are rewritten only for the (agent, cell) pairs
that changed (a per-cell viewer map); the action array is tested with one bytes comparison. 16 trivial agents:
≈1.2·10⁵ ticks/s on one (shared) core; 1 agent ≈1.9·10⁵, 4 ≈1.5·10⁵, 64 ≈0.7·10⁵ (README).

## Open (not in v1)
Furniture words and places (放在桌子上), pointing / gaze by agents, caregivers' own gaze, agents' names learned by
agents from each other, make-way requests (让一下) instead of nudging, sleep semantics at night, lawful-physics
reference numbers, a GPU-batched reference agent.
