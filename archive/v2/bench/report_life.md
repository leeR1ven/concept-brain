(Saved by the coordinator from the reviewer's final reply; the harness blocked the reviewer's own write.)

Life mode is built and in use. A single agent now lives one unbroken life with no resets and no frozen copies, and every probe is scored when its answer window closes. v1.2's API and results are unchanged, and `selftest.py` passes 38/38 (117 s). READY says v2.0; the full spec and reference numbers are in README.md. Pushing is left to you.

**What I built.** `life.py` is a subclass of the staged world, so physics, held-out sets, eye codes and the exchange runner are all reused. Before touching anything I froze v1.2 in `bench/v1.2/`; the seven core modules are byte-identical to it and the v1.2 fingerprint (ab45fe…) still holds. Usage: `Life(seed, tutor='model'|'recast'|'force', ...).run(agent, hours=… or ticks=…)` (1 h = 48,000 ticks; `run()` can be called again and the life goes on), then `evaluate_life(log)`. References: `RandomNamer`, `life_baseline(name)` and `RefV1`; `life_refs.py` produces `results_life/`. The life's content comes from its own random stream and doesn't depend on the agent; only timing does (checked: the oracle and a silent agent get the same sequence of exchanges).

**The life**
- **Days** hold 8–14 visits separated by alone periods of 100–2,000 ticks (10% run 2,000–6,000). Objects appear, move and disappear by themselves.
- **Nights** are 5,000 ticks with the eye all zero and the ear silent, and have no sleep semantics: the world tells the agent nothing and no metric is about them.
- **Visits:** the partner arrives (an 8-unit "person" code appears in the eye) and waits 30 ticks, so the agent can greet first. Then come a greeting (你好, or 早上好 in the morning), 2–6 topics with silent gaps of 20–300 ticks, and 再见 (晚安 before the night). other2 is present in 30% of visits and chats with the partner in the background.
- **Topics:** B/C/D questions about the current scene (held-out pairs are only ever probed), facts, novel objects, memory questions, 刚才 questions, overheard dialogue, and optionally adult-corpus pairs (`topics='BCDE'`).

**Recurrence (the designer's addendum)**
- **Stable facts, re-told at irregular intervals:** 4 pets with fixed names, the partner's and other2's names, and where things were put. Each object has a usual place, used 70% of the time.
- **Novel objects** come back into view and are sometimes named again; greetings, rituals and the 12 everyday objects recur constantly.
- **Memory questions** are always probes, about facts last told in an earlier visit. Delays range from about 600 to 470,000 ticks.
- **Per-probe logging:** every probe logs `n_experienced` (times its answer was heard or stated before), `n_asked_before`, and for memory items `ticks_since_told`.
- **The key memory metric** is retention by delay × encounters (1, 2, 3–4, 5–8, 9+).

**Initiative.** Any utterance outside an answer window counts.
- **Rewarded:** a greeting in the arrival window, or naming a visible object (once per object per visit), earns a short reply (你好 / 对是X) plus reward.
- **Unrewarded:** nonsense gets no reward and a 30-tick pause; interrupting the partner's question also delays the partner by that pause.
- **Logged only:** talking over a partner is logged as overlap ticks and interruptions, and utterances alone or at night are logged with no reaction.

**Metrics** (`evaluate_life`, all in life and over life time): per-task learning curves; scene / held-out / memory / social groups; retention by delay, by encounters and the two combined; initiatives per 1,000 quiet ticks, share appropriate and reward; overlaps and interruptions; longest monologue; latency; stability per time bin with collapse flags.

**Speed.** A silent tick costs 3.9 µs. While nothing sounds the world keeps a zero window instead of ticking the stream; with `noise > 0` every tick is a full tick (about 30 µs). 10⁶ ticks of life (about 21 h, 42 days, 453 visits) take 5 s with a trivial agent, against a target of a few minutes.

**Reference numbers** (seed 0, one life of 10⁶ ticks, model mode unless noted):

| agent | free acc | scene (new) | held-out | memory | social | initiatives per 1,000 quiet ticks | appropriate | reward | overlap |
|---|---|---|---|---|---|---|---|---|---|
| silent | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0 | – | 0 | 0 |
| random namer (every 200) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 2.31 | 0.21 overall (≈0.8 with partner present) | 454 | 0 |
| echo | 0.53 | 0.00 | 0.00 | 0.00 | 1.00 | 0 | – | 0 | 0 |
| lookup | 0.05 | 0.00 | 0.00 | 0.00 | 0.07 | 0 | – | 0 | 0 |
| oracle | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0 | – | 0 | 0 |
| RefV1, model mode | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0 | – | 0 | 0 |
| RefV1, force mode | 0.46 | 0.07 | 0.06 | 0.00 | 0.84 | 0 | – | 0 | 0 |

- **Anti-cheat holds:** echo and lookup score 0 on scene questions in new situations, on held-out items and on memory items. Echo's 1.00 on greetings and goodbyes is inherent, since those answers repeat the prompt.
- **Repeated situations are scored separately:** in life a scene persists, so lookup can answer a question repeated about an unchanged scene. I tag such questions and report them apart; the headline scene score counts only situations never taught before.
- **The v1 reference in model mode** stays silent for its whole life, because it creates no programs without a teacher.
- **In force mode it learns only the routines:** greet rises from 0.30 to about 0.91, bye from 0.00 to 1.00, rituals from 0.00 to 1.00.
- **Scene and memory stay at or near zero:** scene questions stay around chance, its `check` answers fall from 0.36 to 0.00, and memory is 0.00 at every encounter count from 1 to 9+. It never takes initiative, and no stability flags were raised.

**Selftest life checks (9 new):** v1.2 unchanged; speed; determinism and agent independence; anti-cheat; naming rewards only when the partner is present; nonsense unrewarded and overlap logged; nights dark and silent; memory delays from hundreds to tens of thousands of ticks; recurrence (memory probes after 1 to 5+ encounters, with per-probe logging).

**Known limits**
- The defaults give a sparse life: about 1,800 scored replies per 10⁶ ticks. The visit, alone and night lengths are all parameters.
- At 2.2 ms/tick for `RefV1`, a 10⁶-tick life with corpus topics would take over 30 minutes. Agents are the bottleneck, not the world.
- The appropriateness rule for initiative is narrow (greetings and naming visible objects only).
- The agent-only-uses-obs rule still can't be enforced by the bench.

## summary_zh

已加入"一生"模式：
- **一生不重置：** 一个智能体从出生一直活下去，不重置、不冻结副本，每道题当场评分。
- **生活内容：** 有白天、到访、独处、黑夜（无睡眠含义）、自己变化的场景、背景对话。
- **主动说话：** 智能体可随时开口，命名看得见的东西或打招呼会得到回应和奖励，乱说或打断则没有奖励、对方停顿。
- **反复出现的事实：** 宠物名、人名、东西放在哪、新物体名会不定期再出现，每道题记录之前经历过几次；关键记忆指标是"间隔 × 经历次数"的保持率。
- **速度：** 安静一拍约 4 微秒，100 万拍约 5 秒。
- **防作弊与 v1 基准：** 重复问句和查表在新情境的看图题、没教过的题和记忆题上都是 0；v1 只学会了打招呼和固定应答，看图题接近瞎猜，记忆题为 0，从不主动说话。
- **v1.2 不变：** 原样保留，自检 38/38 通过。

Files are in `arch/v2/bench/`:
- `life.py`
- `life_refs.py`
- `update_readme_life.py`
- `selftest.py`
- `README.md`
- `READY`
- `v1.2/`
- `results_life/`
- `checks/life_refs.log`