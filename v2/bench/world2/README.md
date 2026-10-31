# world2 — an open house + garden where many models live with scripted caregivers (v1)

Status: v1 works end to end; `selftest.py` 25/25 PASS (~40 s); reference numbers below and in `checks/refs.out`.
numpy only, Python 3.10+. world2 is a new package inside `arch/v2/bench/`; it **imports** the bench's `physics.py` (the
real model's Voice / Stream, mouth rule) and `metrics.py` (echo-free, parrot) and changes no bench file (v1.2 and life
mode untouched; selftest checks it). Design rationale: `DESIGN.md`.

```python
import sys; sys.path.insert(0, '<scratchpad>/arch/v2/bench')
from world2 import World2, PerAgent, evaluate, headline
w = World2(n_agents=4, seed=0, physics='random', noise=0.0, tutor='mixed')   # options below
log = w.run(agents, ticks=300000)        # or days=10 / hours=...; run() may be called again: life goes on
print(headline(evaluate(log)))
```
`agents.step(obs) -> (N, 41)` float32 actions (one model drives all bodies: batched), or `PerAgent([a_0, ..., a_{N-1}])`
with `a_j.step(obs_j) -> (41,)` (per-agent loops; identical lives, selftest; `obs_j` holds row views of the batched
buffers plus floats for hunger and reward). A bench agent (32 muscles, bench obs) lives in world2 as a body that
cannot move via `PerAgent([MouthOnly(bench_agent), ...])` (its ear window is copied, its eye is world2's).
**Rules for agents (reviewers check the code):** use only `obs` (and, once at birth, `w.warmup_windows()` = the
bench's warm-up stream); never read world internals; the `bench_*` hooks are for reference agents only.

## The world
```
########################   kitchen 厨房: stove O, table T          bedroom 卧室: shelf H, bed B (corner)
#OOkkkkkkkk#HHbbbbbbbBB#   living room 客厅: table T, sofa S        garden 花园: trees Y
#kkkkkkkkkk#bbbbbbbbbBB#   doors D (3 wide)
#kkkkTTkkkk#bbbbbbbbbbb#
#kkkkTTkkkk#bbbbbbbbbbb#
#kkkkkkkkkk#bbbbbbbbbbb#
####DDD#########DDD#####
#llllllllllllllllllllll#
#llllllllllllllllllllll#
#llllllTTllllllllllllll#
#llllllllllllllllllllll#
#llllllllllllllSSSlllll#
##########DDD###########
#gggggggggggggggggggggg#   (10 garden rows, trees at (18,15) (4,16) (10,20))
```
- 24×24 cells; a cell holds terrain, ≤ 1 object (on floor or furniture) and ≤ 1 person. Agents are blocked by walls,
  furniture, objects and persons. Caregivers step over objects, swap with an idle caregiver in their way and squeeze
  past a child that blocks them for 2 steps (the child is nudged one cell; logged as an `event` 'nudge').
- Objects: 12 everyday bench objects (苹果 香蕉 饼干 牛奶 西瓜 葡萄 小狗 小猫 皮球 汽车 小鸟 鸭子) + 38 novel kinds (the
  bench's novel list minus 椅子 桌子). Per instance: colour (6), count 1–3 (animals 1), size small/big, edible (6 foods +
  蛋糕 面包 草莓 橘子 柠檬 番茄 土豆). Usual rooms: foods kitchen, 皮球 living room, 汽车 bedroom, novel kinds living room or
  bedroom (per seed). 小狗 小猫 wander everywhere, 小鸟 鸭子 in the garden (animals cannot be taken). Meals (3 a day,
  奶奶 cooks): 2–4 food piles on the kitchen table, uneaten food cleared at the next meal. Caregivers tidy misplaced
  objects back to their usual rooms (and say so); agents carry things anywhere.
- Day 24,000 ticks (75 ms per tick: 30 min), night 6,000 ticks: eye dark, caregivers asleep (absent, silent).
  Agents may still move and talk at night (nobody reacts).

## Bodies: actions (N, 41), > 0 = active
- **0–31 mouth**: the bench's 32 muscles, Voice, mouth rule (4 muscles = clear char, 3 = mumbled, else garble) and onset
  rule (a new syllable when the set changes, after silence, or when the max force drops: ramp 1-2-3 to repeat a char).
  The own voice enters the shared sound field one tick later. `mouth_vocab='lexicon'` as in the bench.
- **32–40 motor**: forward, back, left, right (no turning), turn_left, turn_right, take, put, eat (this order; no built-in
  meaning for the agent; `motor_shuffle=True` permutes the 9 units per agent). When the body is free, the strongest
  active unit starts its action immediately; the body is then busy for 3 ticks (`motor_ticks`, one beat). take = the
  object in the cell in front (view index 7); put = into the free cell in front, or into the hands of the person in
  front (give; caregivers and agents accept with empty hands); eat = one item of the held food (count −1).
- Innate scaffolds: none in the world. Optional generic reflex `Explore(agents, N, idle=40)`: one random motor unit
  twitched after 40 ticks without motor output (documented, off unless used).

## Senses: obs dict (world-owned read-only buffers, updated in place; copy what you keep)
| key | shape, dtype | content |
|---|---|---|
| `ear` | (N, 8, 6) int64 | the bench's ear window per agent (2 slices/tick). One shared sound field: every speaking person is a source; levels − ceil((d−2)/2) beyond 2 cells, −2 between rooms, nothing beyond 8 cells (6 between rooms); max mix incl. own voice (bench `self` offset). 妈妈 = bench `other`, 爸爸 = `other2`, 奶奶 / every agent: own fixed offsets. A new view each sound tick (valid ≥ 1,024 ticks) |
| `eye` | (N, 1600) uint8 | egocentric 5×5 view: rows 0–4 ahead (row 0 = own row; own cell = block 2, front cell = block 7), columns −2..+2; walls occlude; 25 blocks × 64 units, same code in every block (below). All zero at night |
| `touch` | (N, 16) uint8 | 0–11 held kind (3 of 12), 12–13 held size; 14 bump (1 tick after a blocked move); 15 hand change (1 tick) |
| `taste` | (N, 8) uint8 | 3 of 8 per food kind, for 6 ticks after eating |
| `hunger` | (N,) float32 | 0–1; +1 per 20,000 ticks; −0.5 per item eaten (start 0.3) |
| `reward` | (N,) float32 | eating (= hunger before), praise / thanks (+1, answers and requests), an imitated naming (+0.5), an answered initiative (+1), an acceptable greeting reply (+1) — on the tick its cause is heard (or the next tick) |

Eye block (64 units): 0–11 terrain one-hot (wall, door, kitchen / living / bedroom floor, grass, table, bed, sofa,
stove, shelf, tree); 12–35 object kind (4 of 24, fixed per kind, pairwise overlap ≤ 2); 36–41 colour; 42–44 count;
45–46 size; 47 held (the object code belongs to the object held by the person here); 48 adult, 49 child; 50–61 person
identity (3 of 12); 62–63 pointed at by a caregiver. A person's cell shows the person and what it holds.

## Caregivers 妈妈 爸爸 奶奶 (scripted generators; every line generated from the world state)
- **Activities**: teach an agent (an episode, weight 0.6), tidy (0.14), chat with another caregiver (0.12), rest (0.14,
  150–700 ticks on an open cell); 奶奶 cooks at meal times; bedtime (900 ticks before night): 睡觉了 and 晚安 to the
  agents in earshot (scored). Attention rotates: an episode goes to the least recently attended reachable agent.
- **Episode** with agent i: if far, a call (名字过来 = a 'come' request) else she walks to a cell in front / beside it
  (never in a doorway if avoidable); 早上好 / 你好 on first contact of the day (scored); 2–6 turns, 20–150 ticks apart:
  - naming with pointing: 这是X, 看X, X, 红色的X, 两个X, 大X (an imitation of X is praised 对 +0.5);
  - a question about what i sees or holds, pointing at it: 这是什么 → X / 是X / 这是X; X是什么颜色 → 红色 / 是红色 / 红的;
    有几个X → 两个 / 有两个; X大不大 → 大 / 大的 / 很大 (小 / 小的 / 不大); 这是X吗 → 是 / 是的 / 对 (不是 / 不对);
    你拿着什么 → X / 是X / 我拿着X; 这是哪里 → R / 是R / 在R / 这是R; 这是谁 (another person in view) → name / 是name;
    你要X吗 (i hungry; she brings the food) → 要 / 要X / 我要X, then 给你 and she hands it over;
  - a memory probe (always a probe, never answered): X在哪里 → 在R / R (an object out of view whose current room i saw
    or was told after the object's last move, ≥ 300 ticks ago), 这是什么 (a novel object named in an earlier episode;
    she fetches it if needed), 我是谁 → 妈妈 / 你是妈妈, 你叫什么 → 小明 / 我叫小明 (facts told in an earlier episode;
    each item ≥ 3,000 ticks after it was last asked);
  - a request (scored by the outcome within 400 ticks; only if i's body can do it): (名字)去拿X → holding a kind-X object;
    (名字)把X给我 → she received X from i (not food she gave i in the last 600 ticks); 名字过来 → i next to her.
    Success → praise / 谢谢 + 1;
  - a novel object: a never-named kind appears in front of i, pointing: 这是X (once; others who see it hear it too);
  - a fact: 我是妈妈, 你叫小明, 这是小红 (pointing at another agent).
- **Exchange kinds**: held-out (task, kind) pairs (≈30 % of each task's everyday kinds, balanced as the bench; the set
  depends on the seed) are only probed and never answered by anyone (caregivers, chat, reactions). Seen items: 15 %
  probe (no reaction), 30 % model (she answers herself 3 ticks after the question; not scored), else recast (correct →
  praise +1; wrong or silent → the answer). `tutor='recast'` (no model), `'model'` (85 % model), `'mixed'` (default).
- **Timing** = the bench's: the answer window opens at the end of the prompt and closes ≥ 45 ticks later, 6 silent
  ticks after the agent stops, cut at 60; caregivers do not start talking while someone in earshot is talking (≤ 30
  ticks of politeness).
- **Chat** (overheard by agents in earshot): X在哪里 → 在R, 名字在哪里 → 在R, 饭好了吗 → 好了 / 还没有, 这是什么 /
  X是什么颜色 about objects next to them (never held-out pairs); 20 % adult-corpus pairs (`/root/work/corpus`, the bench's
  training part).
- **Contingent reactions** to an agent's utterance outside an answer window (an *initiative*), by the nearest awake,
  interruptible caregiver in earshot: greeting (你好 早上好 …, first of the day) → 你好 +1; naming a visible object (X 这是X
  是X 看X 有X; once per kind per 3,000 ticks) → 对是X +1; calling a caregiver by name → 怎么了 +1, she comes and does a
  turn; 这是什么 → 这是X (held-out 'what' kinds: 你说呢); 要X / 我要X → 好 +1, she fetches X and hands it over;
  饿 / 要吃 … while hungry → 好 +1, she brings food; anything else → nothing. An unrequested give → 谢谢. Utterances that
  start while the agent's own caregiver is talking get no reaction (category 'interrupt').

## Records (`log`, `type`)
- `q`: a question / social exchange: t, agent, by, task, prompt, acceptable, kind (probe / recast / model / social /
  statement), reply, acc, first_ok, latency, stop, held_out, needs_scene, item, **new_situation** (this prompt with this
  exact eye input was never followed by a taught answer for this agent), n_experienced (times this agent heard the
  answer: taught, recast, overheard, stated), n_asked_before, room, dist, view_k / obj (object questions), memory
  probes: ticks_since_told (since last seen or told), source, ticks_since_first_told; model: after_model_reply.
- `req`: agent, by, req (fetch / give / come), target, prompt, success, ticks, dist0, named, others, others_came.
- `init`: agent, text, category (greeting naming calling asking request hungry repeat other alone unattended interrupt),
  appropriate, reward, by, heard, agents_heard, overlap. `a2a`: src, dst, gap (an agent's utterance answered by another
  agent in earshot within 30 ticks). `say` (every caregiver line), `eat`, `event` (day_start, night_start, meal, tidy,
  novel_appears, nudge, a2a_give), `mon` (per-agent counters every 10,000 ticks).

## `evaluate(log)` (all agents pooled; `per_agent` too)
`tasks` (per task: n, acc, first, echo_free, parrot, spoke, held / seen), `curves` (per task, 10 time bins),
`groups`: **scene** (here-and-now questions in new situations), scene_repeat, **held_out**, **memory** (where novel
person myname), **social**; `comprehension` (fetch / give / come: success, median ticks; named calls: share of other
agents who came; curve); `retention` (memory by delay, by encounters and delay × encounters; QA by encounters);
`initiative` (n, per 1,000 quiet ticks, appropriate, reward, categories, curve); `vocabulary` (distinct lexicon words
used correctly — in acceptable replies or appropriate initiatives — per agent, cumulative curve); `a2a`;
`overlap` (talking over the engaged caregiver: ticks, interruptions; over any caregiver; over agents); `body`;
`latency`; `imitation`; `stability` (per bin: answer rate, acc, initiative, mouth share, overlap, garble, moves, eats)
and `stability_flags`. `headline(res)`: one line.

## Reference agents (`world2.agents`)
`Silent`, `RandomActor` (random motor units p = 0.1 / tick, random 4-muscle babbles p = 0.01), `Echo` (repeats every
question, request and greeting addressed to it), `Lookup` (prompt + exact eye input → the reply it heard taught for
exactly that), `Namer` (every 300 quiet ticks names a random visible object), `Oracle` (answers from the world state,
carries out requests by path planning). The last four are privileged (world hooks).

## Reference numbers (`refs.py`; raw: `results/*.json`, `checks/refs.out`)

world2 reference runs: 10.0 days per run, seeds [0, 1, 2], 4 agents (and 16 agents for seed 0).

### Table (4 agents, mean ± sd over seeds)

| agent | free acc | scene (new) | held-out | memory | social | requests | fetch | give | come | appropriate initiatives /agent/day | vocabulary /agent | a2a turns /day | talk over partner (ticks/day) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| silent | 0.00±0.00 | 0.00±0.00 | 0.00±0.00 | 0.00±0.00 | 0.00±0.00 | 0.00±0.00 | 0.00±0.00 | 0.00±0.00 | 0.00±0.00 | 0.00±0.00 | 0.00±0.00 | 0.00±0.00 | 0.00±0.00 |
| random | 0.00±0.00 | 0.00±0.00 | 0.00±0.00 | 0.00±0.00 | 0.00±0.00 | 0.07±0.01 | 0.05±0.01 | 0.15±0.09 | 0.15±0.01 | 0.04±0.03 | 0.00±0.00 | 140.07±12.33 | 114.30±6.02 |
| echo | 0.13±0.00 | 0.00±0.00 | 0.00±0.00 | 0.00±0.00 | 1.00±0.00 | 0.00±0.00 | 0.00±0.00 | 0.00±0.00 | 0.00±0.00 | 0.00±0.00 | 2.92±0.12 | 0.47±0.29 | 0.00±0.00 |
| lookup | 0.04±0.01 | 0.00±0.00 | 0.00±0.00 | 0.00±0.00 | 0.00±0.00 | 0.00±0.00 | 0.00±0.00 | 0.00±0.00 | 0.00±0.00 | 0.00±0.00 | 1.92±0.31 | 0.00±0.00 | 0.00±0.00 |
| namer | 0.00±0.00 | 0.00±0.00 | 0.00±0.00 | 0.00±0.00 | 0.00±0.00 | 0.00±0.00 | 0.00±0.00 | 0.00±0.00 | 0.00±0.00 | 7.32±0.95 | 24.67±1.94 | 0.00±0.00 | 0.00±0.00 |
| oracle | 1.00±0.00 | 1.00±0.00 | 1.00±0.00 | 1.00±0.00 | 1.00±0.00 | 1.00±0.00 | 1.00±0.01 | 1.00±0.00 | 1.00±0.00 | 0.00±0.00 | 60.75±1.63 | 0.00±0.00 | 0.00±0.00 |

Scored items per run (seed 0, 4 agents, 10.0 days): silent: free 1287, scene 640, held-out 69, memory 443, requests 535, random: free 1430, scene 850, held-out 144, memory 406, requests 414, echo: free 1310, scene 654, held-out 38, memory 430, requests 517, lookup: free 1244, scene 602, held-out 77, memory 409, requests 542, namer: free 1203, scene 575, held-out 66, memory 396, requests 523, oracle: free 1906, scene 1136, held-out 170, memory 603, requests 648

Input to the agents (seed 0, mean over agents): silent n=4: 94 caregiver lines addressed per agent per day, caregiver speech audible 12 % of daytime, silent n=16: 25 caregiver lines addressed per agent per day, caregiver speech audible 7 % of daytime, oracle n=4: 153 caregiver lines addressed per agent per day, caregiver speech audible 20 % of daytime, oracle n=16: 38 caregiver lines addressed per agent per day, caregiver speech audible 14 % of daytime

Oracle per task (seed 0): check 1.00 (n=140), colour 1.00 (n=182), count 1.00 (n=143), goodnight 1.00 (n=43), greet 1.00 (n=120), holding 1.00 (n=93), myname 1.00 (n=109), novel 1.00 (n=157), offer 1.00 (n=140), person 1.00 (n=149), room 1.00 (n=123), size 1.00 (n=120), what 1.00 (n=103), where 1.00 (n=188), who 1.00 (n=96), ALL 1.00 (n=1906)

### 16 agents (seed 0)

| agent | ticks/s | free acc | scene | held-out | memory | requests | questions per agent per day | initiative | a2a |
|---|---|---|---|---|---|---|---|---|---|
| silent | 131,643 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 10.4 | 0 | 0 |
| random | 10,190 | 0.00 | 0.00 | 0.00 | 0.00 | 0.05 | 11.1 | 43832 | 19919 |
| echo | 64,327 | 0.31 | 0.00 | 0.00 | 0.00 | 0.00 | 10.2 | 488 | 5 |
| lookup | 112,578 | 0.09 | 0.00 | 0.00 | 0.00 | 0.00 | 10.7 | 0 | 0 |
| namer | 26,359 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 11.0 | 3370 | 0 |
| oracle | 58,328 | 1.00 | 1.00 | 1.00 | 1.00 | 0.99 | 13.6 | 0 | 0 |

### Speed (trivial agents, 5 days = 150,000 ticks each, best of 2)

| agents | ticks/s | us/tick |
|---|---|---|
| 1 | 192,102 | 5.2 |
| 4 | 145,346 | 6.9 |
| 16 | 118,436 | 8.4 |
| 64 | 72,718 | 13.8 |

## Selftest (`python3 selftest.py` → `checks/selftest.out`, 25 checks, ~40 s)
Bench untouched; Voice / mouth rule = the bench's (3,970 chars); voices; the ear next to 妈妈 = the bench Stream
bit for bit; attenuation (louder nearer, silent beyond range, walls); own voice one tick later, agents hear each other;
mouth / onset rule; motor semantics, busy, bump; take / eat (reward, hunger, taste) / give; eye codes, pointing,
turning, occlusion, terrain of untouched cells; nights dark and silent; determinism; batched == PerAgent; anti-cheat
(echo and lookup 0 on scene, held-out, memory, requests; oracle 1.0 on every group, ≥ 0.95 on requests); random actor
at chance; held-out pairs never answered; here-and-now pointing; memory probe delays; all request types; caregivers
say one line at a time and the oracle never talks over its partner; fair attention; initiative reactions; agent-agent turns; speed ≥ 10⁵ ticks/s with 16 trivial agents.

## Deviations from the task and the draft, known limits
- Eye block 64 units (draft 48); doors 3 wide; caregivers step over objects and nudge a blocking child (both needed:
  2-wide doors and objects made closed pockets in which caregivers got stuck for days).
- In an open world content depends on the agents (who is where, what they hold, what they say), unlike life mode;
  fixed seed + same agents ⇒ identical life (selftest).
- Requests are made only when feasible for the agent's body, so request success measures comprehension. Oracle
  failures left: ≤ 1 % (objects moved or taken by others during the window).
- Memory 'where' answers are rooms (in v1 there are no furniture words / places).
- Speed: active agents cost more (16 random babbling / moving agents ≈ 9,000 ticks/s ≈ 0.1 ms per tick for the world:
  still far below any learning agent's step).
- The agent-only-uses-obs rule cannot be enforced by the world.

## Files
`core.py` (mechanics, event loop, caregiver op driver), `care.py` (caregiver scripts, reactions), `lex.py` (lexicon,
layout, held-out sets, codes), `agents.py` (wrappers, references), `metrics.py`, `selftest.py`, `refs.py` (reference
runs + speed → `results/*.json`, `checks/refs.out`), `DESIGN.md`, `TASK.md`.
