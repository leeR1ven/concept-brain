(Saved by the coordinator from the reviewer's final reply; the harness blocked the reviewer's own write.)

# world2: foveated eye option (`eye_mode='fovea'`), short report

The option is built and tested. The default uniform eye is unchanged bit for bit, and selftest is 32/32 PASS (SIMULATED: `world2/checks/selftest.out`, 64 s).

## What the fovea eye is
`World2(..., eye_mode='fovea')` gives `obs['eye']` of shape (N, 512) instead of 1,600, more detail at the centre and less at the edge:

| part | cells | units per cell | what it shows |
|---|---|---|---|
| fovea | the cell in front, where take/put act | 128 | kind (8 of 60 units, two kinds share ≤ 2; uniform: 4 of 24, ≤ 2 of 4), coarse class, terrain, colour/count/size (2 units per value), held, person identity (5 of 20), pointing |
| medium | the 8 cells around it | 32 | terrain, coarse class (food / animal / toy animal / thing), colour, adult/child, held, pointing |
| outer | the other 16 cells | 8 | obstacle/door/floor, something there, person, adult, motion (3 ticks), pointing |

- Which kind, which person, count and size can only be read in the fovea. Pointing and motion show in every ring, as cues to look; touch still gives the held kind.
- Gaze is the body's heading and position. Turning (3 ticks) or stepping brings a cell into the fovea, so the existing motor units are enough and the action vector stays at 41.
- Only what agents see changes. Caregivers don't use the eye, so an agent that ignores the eye lives exactly the same life in both modes. The selftest checks this; in the reference runs, silent / random / echo / namer / oracle have identical scores in both modes.
- In fovea mode, records of pointed questions add `fovea_at_start`, `fovea_hit`, `fovea_wait` and `fovea_at_close`, and `evaluate()` adds an `active_vision` section.
- New reference agent `Looker`: it reads only `obs` (the documented layout and motor order). When it sees a pointed cell in the periphery, it finds a path within its visible 5×5 frame and turns so the pointed cell is in front of it.

## The default eye is unchanged
- **Regression check:** the selftest now holds md5 signatures of 4 complete lives recorded before any change (oracle, random actor, 16 silent agents, namer): logs, eyes at a mid-day tick, and `evaluate()`. All are bit-identical.
- **Reference runs:** rerunning the uniform references (3 seeds × 10 days) reproduces every v1 table exactly; only the speed columns vary with load.
- **Metrics fix:** this check exposed a v1 bug. `evaluate()`'s vocabulary count could differ between Python processes, because word-length ties followed Python's string hashing. Word order is now deterministic, and all other keys match v1 under 3 hash seeds.

## Fovea checks (all pass)
- **Rings:** the fovea shows full detail, the medium ring shows only class + colour, the outer ring only "something here".
- **Cues:** pointing is visible in every ring. A person in the periphery appears as an adult, with motion on for exactly 3 ticks.
- **Active vision:** turning brings an object into the fovea.
- **Same world:** an eye-independent agent's log is identical across modes, apart from the eye-derived tags.
- **Anti-cheat:** echo and lookup score 0 on scene questions, held-out pairs, memory probes and requests; the oracle scores 1.00 on every group and on requests.
- **Pointing can be followed:** the Looker reaches the pointed cell in 0.66 of cases against 0.15 for a silent agent.
- **Speed:** check is ≥ 90,000 ticks/s, measured about 100,000. The default eye keeps its 100,000 check (121,700 measured).

## Reference numbers in fovea mode
SIMULATED: `checks/refs_fovea.out`, `results_fovea/`; 10 days, seeds 0–2, 4 agents.

- **Question scores are the same as with the uniform eye:**
  - oracle: 1.00 on scene, held-out, memory and social, and 1.00 on requests (0.99 with 16 agents);
  - echo and lookup: 0.00 on scene, held-out, memory and requests;
  - random actor: 0.07 on requests.
- **Active vision.** "Reached" means the pointed cell came into the fovea before the answer window closed:

| agent | in fovea at start | reached | reached, if not at start | median ticks |
|---|---|---|---|---|
| silent (chance) | 0.12 | 0.12 | 0 | – |
| random actor | 0.15 | 0.26 | 0.13 | 22 |
| Looker | 0.30 | 0.67 ± 0.03 | 0.53 | 5 |

- **Speed, fovea mode** (trivial agents): 177,000 / 121,000 / 103,000 / 60,000 ticks/s for 1 / 4 / 16 / 64 agents. Uniform mode on the same run: 192,000 / 129,000 / 117,000 / 72,000.

## Files changed in `arch/v2/bench/world2/`
- `lex.py`: fovea layout and codes, drawn from their own random stream so the uniform codes don't change.
- `core.py`: `eye_mode`, fovea tables, motion marks, `_eyes_fovea`, the pointing watch. Everything is switched off in uniform mode.
- `care.py`: fovea tags on pointed questions.
- `agents.py`: `Looker`.
- `metrics.py`: `active_vision`, deterministic word order.
- `selftest.py`: 7 new checks.
- `refs.py`: `--eye fovea`.
- `README.md`, `DESIGN.md`: documentation of the option.

`report.md` there is your saved copy of my v1 report; I left it untouched.

**Not built, as asked:**
- a separate eye-movement action (gaze is the body's heading);
- caregivers waiting for the child to look;
- the designer's foveated 3-D network (local3d-similarity), noted in the docs as the planned real eye.

## summary_zh
- **新增：** 给 world2 加了“中央凹”视觉选项 eye_mode='fovea'。
- **视野：** 正前方一格看得最细（物体种类、颜色、数量、大小、人是谁、有没有被指），周围 8 格只看大类和颜色，外圈 16 格只知道“有东西 / 有人 / 在动 / 被指着”，总共 512 个单元。
- **主动视觉：** 模型要转身或走动，才能把东西放到正前方看清，现有的动作单元就够用。
- **跟随指点：** 只用眼睛输入的 Looker 能把被指的东西转到视野中央：67%，随机只有 12%。
- **默认不变：** 原来的均匀视觉逐位相同，4 段生活的日志、视野、评分的校验值都对得上。
- **防作弊：** 在新模式下依然成立。
- **速度：** 16 个模型约每秒 10 万拍。
- **自检：** 32/32 通过。
- **顺带修复：** v1 评测里一个随 Python 哈希种子变化的小毛病（词汇计数），现在已固定。