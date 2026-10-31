# ARCH_v2 bench (implements `API.md`; deviations and clarifications are marked **[clar]**)

Status: see `READY`. numpy only. Python 3.10+. Copy the folder (or put it on `sys.path`) and import from it.

```python
import sys; sys.path.insert(0, '<scratchpad>/arch/v2/bench')
from world import World
from metrics import evaluate, headline
from baselines import baselines

class MyAgent:
    def step(self, obs):                 # obs = {'ear': (8,6) int, 'eye': (1024,) uint8, 'reward': float, 'cue': (32,) or None}
        return np.zeros(32)              # muscles, > 0 = active (force levels matter for re-articulation, see below)

w = World('B', seed=0, physics='random', noise=0.0, tutor='model')
log = w.run(MyAgent(), n_dialogues=2000)        # a list of dicts, one per exchange (also w.log)
res = evaluate(log); print(headline(res))
bl = baselines('B', seed=0, n_dialogues=2000)   # same world, same metrics; cached in cache/
```
Curricula: `World('C', seed=0, prev=w_B)` keeps the clock, the exposure history, the novel objects already named and
the corpus position of the previous world, with a fresh content stream (same seed ⇒ same physics, held-out sets and eye
codes). `run_curriculum(agent, [('A', 300), ('B', 2000), ('C', 2000), ('D', 2000), ('E', 2000)], seed=0,
tutor='model')` → (worlds, concatenated log). The agent object simply continues.

**Rules for agents (the bench cannot enforce them; reviewers check agent code):** use only `obs` (and, once at birth,
`world.warmup_windows()`); never read `world.voice`, `world.log`, `world.cur` or any other world internal; the
privileged hooks (`bench_*`) are for baselines only.

**Self-test:** `python3 selftest.py` (29 checks, ~90 s; `--long` runs the long ARCH_v1 cross-check, ~5 min) → `checks/selftest.out`.
Speed: the world costs ≈ 19 µs per tick (2,000 dialogues ≈ 190k ticks ≈ 4 s with the trivial agent); `RefV1` ≈ 0.8 ms
per tick (2,000 stage-B dialogues ≈ 2.5 min). On the GPU server (numpy 2.1) the streams are identical (fingerprint in
`bar.py`); learned float32 weights may differ in the last bit (NEP 50 scalar promotion), so long agent runs are
statistically, not bit-wise, reproducible across numpy versions.

## Physics (`physics.py`)
- `Voice` = `/root/work/concept-brain/world/voice.py` copied (verbatim copy: `voice_orig.py`), built over the **sorted
  corpus chars (3,970)** with the world seed ⇒ every char has exactly the real model's sound and 4-muscle set
  (selftest checks sounds, offsets, need sets and the mouth rule against `voice_orig.py`). Mouth rule unchanged
  (all 4 muscles = clear char, 3 of 4 = mumbled char, else `~` garble), vectorised and cached.
- `Stream`: step = T // 3 = 2 slices per tick, 3 ticks per char, sources mix by max; reproduces ARCH_v1's
  `voice_copy.Stream` (with the own-voice re-articulation flag) bit for bit.
- Voices: `other` (partner, original sound), `self` (agent, original offset), `other2` (new: own fixed ±1 offset).
- **Onset of the own voice** (as ARCH_v1's Talker): a mouth tick is a new syllable if the active muscle set changed, or
  the mouth was silent the tick before, or the maximum force on the same set dropped. To say a char twice in a row,
  drop the force (v1 ramps 1-2-3) or release for one tick. The own voice is heard one tick later (as talk.py).
- `physics='lawful'`: each muscle has a fixed pattern (a peak of level 5–9 in one band in 3 of the 6 slices); the sound
  of ANY muscle set = clip(Σ patterns); a char's sound = the sound of its 4-muscle set; the own voice is the sound of the
  set the mouth actually makes (mumbles and garbles sound like their muscles). Checks (`physics_check.py`,
  `checks/physics_*.out`): FEAT0 Jaccard grows with shared muscles 0.12 / 0.24 / 0.40 / 0.63 (0–3 shared; random
  physics 0.21 flat), corr(sound distance, shared muscles) −0.74 (random −0.02); lexicon chars: max pairwise Jaccard
  0.77 (random 0.48); all 3,970 chars: max 0.89 (random 0.63), no pair > 0.9, no identical codes; own voice → its char by
  nearest neighbour 100 %. The char ↔ muscle-set assignment (`need`) is the original one in both physics.
- `mouth_vocab='lexicon'` (or a list of chars): the mouth rule considers only these chars (others → garble) while every
  sound and muscle set stays the real model's. With the full 3,970-char mouth a random 4-muscle babble hits a given
  lexicon char with p ≈ 3·10⁻⁵ (5 of 141 lexicon chars kept after 291 babbles); with `mouth_vocab='lexicon'` 35 of 141
  after 323 babbles (300 stage-A dialogues, body.py babbling kept on the tutor's imitation). `vocab=[...]` instead rebuilds the Voice over those chars (different sounds from the real
  model). Not allowed with stage E.
- `noise=σ`: every new slice cell = max(signal, clip(rint(N(0, σ)), 0, 10)); σ = 1 gives a silent-window energy of ≈ 19
  (ARCH_v1's F threshold is 10, so a v1 body hears "talking" all the time — by design).

## Senses (obs dict, nothing else)
- `ear`: the (8, 6) int window of `Stream.tick` (a new array every tick).
- `eye`: `(n_eye=1024,)` uint8, read-only. Scene = 3 slots (left/middle/right), each empty or (object of 12, colour of 6,
  count 1–3). Union of a fixed sparse code per (slot, attribute value) (8 units each) plus an 8-unit "attended slot"
  code while the tutor points. **[clar]** each (slot, value) code = 4 units shared by the value across slots + 4
  slot-specific units (`eye_shared=4`; set 0 for fully slot-specific codes), so objects can be recognised anywhere.
  **[clar]** pointing is on from the prompt's first tick until the exchange ends (the tutor keeps pointing until
  answered). The scene appears `scene_lead`=5 ticks before a dialogue's first line and is all-zero in the 5 ticks before.
  Option `eye_layout='retino'` (proposed by the brainlike designer; codes identical to their copy): slot s owns units
  [240 s, 240 s + 240) and every block carries the same feature code (shape: 8 of units 0–159, colour: 8 of 160–207,
  count: 8 of 208–231, pointing: 232–239) — an innate retinotopic layout for attention-style agents.
- `reward`: float, 0 except on the first tick of a recast (1.0 if the reply was acceptable) and on the first tick of a
  play imitation (0.2).
- `cue`: only in `tutor='force'`, after the prompt: the (32,) muscle set of the next syllable (1.0 on its 4 muscles) on
  every tick until the agent's mouth makes an onset; after the last syllable **[clar]** an all-zero (32,) vector = the
  v1 END push, until the mouth is silent. None at all other ticks.

## Turn protocol (per exchange) **[clar]**
- prompt: the partner says the line, 3 ticks per char (no pause inside a line). `t_prompt_end` = first tick after it.
- free reply (`kind='probe'` or `'recast'`): the partner waits until the agent has been silent for 13 ticks after its
  stop and ≥ 45 ticks have passed since `t_prompt_end` (ARCH_v1 `free()` + 45-tick rule); if the agent never speaks, 45
  ticks; the window is cut at 60 ticks (stop reason `cut`).
- `tutor='model'`: prompt, 3 silent ticks (`gap_model`; a v1 body's GO is then blocked by the next line), the model
  reply by `other2` (70 %) or by the partner itself (30 %), then up to 12 ticks for the agent to start; if it speaks,
  the next line comes 6 ticks after it stops (PAUSE = 5 silent ticks after the stop tick).
- `tutor='force'`: ARCH_v1 `teach()` — cue from `t_prompt_end`, advance on each mouth onset, END push, then 5 silent
  ticks after the stop tick; teacher timeout after 45 ticks without an onset (counted in `w.timing_violations`).
- Every exchange of a seen item is a probe with probability `probe_frac`=0.15; an exchange draws a held-out (task,
  filler) pair with probability `held_frac`=0.15 and is then always a probe. Held-out pairs are never taught.
- Dialogues: 1–3 exchanges about one scene; 10 silent ticks between dialogues. The content schedule (items, scenes,
  probes, model variants) depends only on (stage, seed, tutor, params), never on the agent's behaviour.

## Stages and tasks
| task | stage | prompt → acceptable replies | needs scene | held-out filler |
|---|---|---|---|---|
| what | B | 这是什么 (pointing) → X / 是X / 这是X | yes | object |
| slot | B | 左边/中间/右边是什么 → X / 是X | yes | object |
| where | B | X在哪里 → 在左边 / 左边 (…中间, 右边) | yes | object |
| offer | C | 你要X吗 (pointing) → 要X / 我要X / 要 | no | object |
| taste | C | X好不好吃 (pointing) → 好吃 / X好吃 / 很好吃 | no | food |
| colour | C | X是什么颜色 → 红色 / 是红色 / 红的 | yes | object |
| count | C | 有几个X → 两个 / 有两个 | yes | object |
| ritual | C | 你好→你好/你好呀, 谢谢→不客气/不谢, 晚安→晚安, 再见→再见/拜拜, 早上好→早上好/早 | no | — |
| check | D | 这是X吗 (pointing at Y) → 是/是的/对 (X = Y, 50 %) or 不是/不对 | yes | object (X) |
| name | D | partner says 我叫N (statement), 1–2 B/C exchanges, then 我叫什么 → N / 你叫N | no | — (8 names) |
| before | D | the scene is shown, 0–1 B questions about the OTHER slots, the object in slot S is replaced (10-tick gap), with p = 0.5 "S是什么" (→ the NEW object), then 刚才S是什么 → old / 是old. The old object is never named before the change, so the answer needs a memory of the scene (repeating the last heard answer is wrong) | yes | — |
| oneshot | D | 这是X (statement, pointing at a NOVEL object, once), 0–2 B exchanges, 这是什么 (pointing) → X; 30 % of oneshot dialogues re-ask an object named in an earlier dialogue (retention) | yes | — (40 novel objects, each introduced once) |
| corpus | E | an adult-corpus pair (`/root/work/corpus/dialogs.jsonl`): training dialogues = corpus[:-3000] in order; probes = 500 fixed pairs from corpus[-3000:] whose prompt never occurs in training (held_out=True) | no | — |
| corpus_fam | E | 40 fixed pairs taught repeatedly (p=0.1 per corpus dialogue) and probed (p=0.1) | no | — |
- Held-out: per task ~30 % of fillers (4 of 12 objects, 2 of 6 foods), balanced so every filler stays trained in the
  other tasks of its stage group. Held-out pairs are never taught (only probed), in every stage and mode.
- D memory questions (name, before) are probes with p = 0.4 (`probe_frac_mem`), otherwise taught per tutor mode;
  oneshot questions are always probes (a model reply would be a second exposure). `one_shot=True` only while the object
  has been named exactly once (n_exposures = 1); `ticks_since_taught` counts from the end of 我叫N / 这是X / the change.
- Filler exchanges inside D routines keep their own stage label (B or C), so `res['D']` holds only D tasks.
- **A — hear and babble** (no reply demanded, no probes): `cds` statements about the scene (X, 这是X, 看X, X在左边,
  红色的X, 两个X, 宝宝看X; pointing), `overheard` routines (partner asks a B/C question, `other2` answers; held-out pairs
  never), `play` periods of 60 ticks: whenever the agent's mouth produces a bout of syllables (bout ends after 2 silent
  ticks) the partner imitates its last ≤ 3 clean chars at once, with `reward_imitate`=0.2 on the first tick of the
  imitation; garble is not imitated. Weights cds/overheard/play = 0.4/0.3/0.3.
- **E — adult corpus interleaved**: each E dialogue is a corpus dialogue with p = `e_corpus`=0.5 (pairs (u0,u1),(u2,u3)…,
  a trailing unanswered line dropped as in ARCH_v1), otherwise a B, C or D dialogue. Corpus pairs are taught per tutor
  mode; never-heard probes are added with p = 0.3 per corpus dialogue. `corpus_start` continues a curriculum.
- Stage strings may combine letters (`'BC'`, `'BCD'`: each dialogue picks one of the letters uniformly).

## Tutor modes
- `model`: prompt, 3 silent ticks, the correct reply by `other2` (70 %) or the partner (30 %); the agent may speak after it
  (`after_model_reply`). Not scored as a free reply.
- `recast`: the agent answers freely (window as a probe, but the tutor reacts 5 silent ticks after the agent's stop, or
  after 45 ticks of silence); then `reward` = 1.0 on the first tick of the recast if the reply was acceptable (else 0)
  and the partner says the correct reply (the agent's own reply if acceptable, else the model reply). Every recast
  exchange is a scored free reply; the reaction afterwards is logged as `after_recast_*`.
- `force`: ARCH_v1 teacher forcing via `cue` (see Senses).
- In every mode `probe_frac` of seen items and all held-out items are probes (no help before or after).

## Log record (one per exchange)
`i, dlg, stage, task, kind ('probe'|'model'|'force'|'recast'), prompt, prompt_by, scene (3 × (object, colour idx,
count idx) or None), attend (slot or None), acceptable, item ((task, filler)), held_out, needs_scene, familiar
(n_exposures > 0), n_exposures, one_shot, ticks_since_taught, model_reply, model_by, cued (force), reply (chars
decoded from the agent's own mouth after the prompt, one per onset, '~' included), t_prompt_start, t_prompt_end,
t_first_mouth, t_stop, latency (t_first_mouth − t_prompt_end), runs (ticks per produced syllable), garble,
extra_turns (bouts after a ≥ 5-tick silence), stop ('silent'|'self'|'cut'), mouth_while_partner (mouth ticks while
other/other2 produced a char, whole exchange), mouth_during_prompt, agent_stop, t_end`; model mode adds
`after_model_*` (the agent's speech after the model reply) and `t_model_end`; recast adds `reward`, `recast`,
`t_recast_end`, `after_recast_*`. Other record kinds (not scored as free replies): `statement` (CDS, 我叫N, 这是X: the
agent's speech after it in `reply`), `overheard` (stage A two-speaker routine), `play` (stage A: `reply` = all babbled
syllables, `imitations` = [(bout, imitated text)], `reward` = total). `dlg` counts dialogues across a curriculum.
`run()` may be called repeatedly on one world (identical to one long run); evaluate slices of `w.log` for learning
curves.

## evaluate(log) → `res[stage][task]` (+ `'ALL'`, `'SCENE'`, `'NOSCENE'`; `res['ALL']` over stages)
- free replies only (probe, recast): `n, acc` (exact match to any acceptable string), `first` (first char correct),
  `echo_free` (prompt chars removed from reply and acceptable strings; **[clar]** scored only on items where every
  acceptable string keeps ≥ 1 char, count `n_echo_free`), `parrot` (fraction of reply chars that are in the prompt),
  `spoke`, `mean_len`, and `held` / `seen` / `familiar` sub-dicts (n, acc, first, echo_free).
- `res[stage]['mech']`: `mouth_while_partner`, `exchanges_with_overlap`, `latency` histogram + `latency_median`,
  `syllables` (complete syllables only: one that began before the window or is still sounding at its end is not
  counted), `not_3_ticks`, `frac_not_3`, `garble`, `extra_turns`, `spoke`, `stop` reasons, `agent_stop` (if the agent
  sets `agent.tick_event`).
- `['guided']` (force: reply == cued), `['after_model']` (imitation of the model reply), `['learning']` (free acc
  on seen items, first vs last third, also on the same items), `['one_shot']` (recall + retention bins by
  ticks_since_taught), `['play']` (stage A: babbled syllables, garble, distinct chars, share in the lexicon,
  imitations, reward), `['corpus_20way']` / `['corpus_fam_20way']` (stage E: ARCH_v1's 20-way char-set-F1 top-1 / AUC,
  `top1_distinct` / `auc_distinct` with distractors of different reply texts only, copy rate vs a shuffled prompt).
- `headline(res)`: one line per stage.

## Baselines (`baselines.py`)
`echo` (the prompt), `last` (the last heard char), `prior` (most frequent reply heard taught for the task type),
`lookup` (exact prompt + scene + pointing → the reply heard taught for exactly that input), `ngram` (char n-gram of the
reply given the prompt head/tail; no copying), `oracle` (acceptable[0]). They run through the same world, mouth and
metrics as privileged agents (hooks `bench_prompt`, `bench_taught`), answer only free-reply exchanges with v1 timing
(first syllable at +6, 3 ticks per char), and learn online from what they heard taught. `Silent` = the trivial agent.

## Body, front end, reference agent
- `frontend.py`: `Ear0(seed)` = the ear's first 3D layer of `brain/net3d.py` bit-identical (2,400 slots, 1,920 alive,
  top 768), with the output reserve (float32 as archv1.py); `feat0(ear, world.warmup_windows())` = usage ≤ 0.6 (as
  ear0.py / fe.py: 1,153 for seed 0); `feat0_v1()` = what archv1.py's warm-up effectively computes (alive minus the
  silence pattern = 1,152); `WorkingMemoryT` = ARCH_v1's T (4,000 / 200, fan-in 30, hold 3.0, set point 0.3/1,000);
  `v1_tie(seed)` replays archv1's rng so T is tick-identical. `World.warmup_windows()` = the birth warm-up stream
  (60 random chars + 4 silent ticks each) — innate calibration, the only non-obs data an agent may take from the world.
- `body.py`: ARCH_v1's body (turn units F/F2/GO/CD/R1/R2, boundary, programs, WTA with cue, recruitment, use penalty,
  END rules 'max'/'sum', breath counter, floor, D/I/M with the 1-2-3 ramp, Hc). A mind passes
  `scorer(body, com, first) -> (scores, s_end)`; the body exposes F, GO, bnd, first, Y/Yext (+ _prev), END, Hc,
  Hc_prev, SPK, phase, Hp, nsyl, stop. **Teacher-free programs**: `babble_rate` / `babble(set)` produce one-syllable
  babbles (3 ticks, same D/I/M/CD path, only when idle); `keep='reward'` commits the set as a program when a reward > 0
  arrives within 15 ticks (the tutor imitated it, stage A) — `keep='manual'`: the mind calls `commit(set)`.
  `recruit='cue'` (v1), `'babble'` or `'both'`.
- `ref_v1.py`: `RefV1(world, cfg='mem'|'base')` = body.py + Ear0/FEAT0 + T + P = T∧GO + P→Y fast (+ slow) + Hc→Y +
  Hc→END + echo slot (Z mirror, latch, R2 replay, E→Y), exact rule, gate = external drive, experience-clock decay.
  `v1_crosscheck.py`: archv1.py's Talker vs the bench world driving archv1's Brain vs the bench world driving RefV1:
  **tick-identical mouth output** (80 dialogues + 40 free replies, 8,097 ticks, 2,313 mouth ticks; `checks/`).
- `baselines.py`: also `Silent` (the trivial agent) and `run_agent(agent, stage, seed, n, tutor, **kw)`.
- Traps reported by designers (not bugs): v1's corollary discharge is driven by GO too, so after a GO whose first
  boundary is blocked (e.g. by a model reply at +3) CD runs 4 ticks with no own voice — do not use CD as an own-voice
  tag (use `body.SPK` / `Hp`); in model mode 30 % of model replies are said by the partner itself, so speaker identity
  does not mark question vs reply — timing does (the model reply starts 3 ticks after the prompt).
- Agents may set `agent.tick_event` (e.g. 'END', 'floor', 'breath' on the tick a turn stops); the log records the first
  one after the reply started as `agent_stop`.
- `World.run_script(agent, items)`: a given list of exchanges (dicts: prompt, reply/acceptable, kind, task, held_out,
  scene, attend) with the same timing and records.

## Reference numbers (2,000 dialogues per stage, world seed 0; runners `bar.py`, `bar_local*.py`; raw results `results_all/*.json`; all tables `checks/bar_tables.txt` via `summarize_bar.py`)

Free-reply acceptable rate — ALL / held-out / scene questions (first-char correct). `refv1_mem` = the bar (ARCH_v1 adopted default, trained with `tutor='force'`, the only regime in which v1 learns); baselines in `force` (model mode within ±0.03).

| stage | refv1_mem (bar) | echo | last | prior | lookup | ngram | oracle |
|---|---|---|---|---|---|---|---|
| B | 0.12 / 0.11 / 0.12 (0.69) | 0.00 / 0.00 / 0.00 (0.39) | 0.00 / 0.00 / 0.00 (0.00) | 0.14 / 0.11 / 0.14 (0.37) | 0.00 / 0.00 / 0.00 (0.00) | 0.13 / 0.11 / 0.13 (0.76) | 1.00 / 1.00 / 1.00 (1.00) |
| C | 0.19 / 0.10 / 0.08 (0.67) | 0.12 / 0.00 / 0.00 (0.49) | 0.00 / 0.00 / 0.00 (0.00) | 0.31 / 0.28 / 0.23 (0.58) | 0.00 / 0.00 / 0.00 (0.00) | 0.54 / 0.28 / 0.24 (0.69) | 1.00 / 1.00 / 1.00 (1.00) |
| D | 0.21 / 0.48 / 0.24 (0.47) | 0.00 / 0.00 / 0.00 (0.31) | 0.00 / 0.00 / 0.00 (0.00) | 0.24 / 0.48 / 0.25 (0.36) | 0.00 / 0.00 / 0.00 (0.00) | 0.16 / 0.08 / 0.16 (0.60) | 1.00 / 1.00 / 1.00 (1.00) |

Per task (acceptable rate; the `prior` column is the chance level of a task-type prior):

| task | bar | prior | ngram | echo |
|---|---|---|---|---|
| B:slot | 0.00 | 0.04 | 0.06 | 0.00 |
| B:what | 0.06 | 0.06 | 0.03 | 0.00 |
| B:where | 0.33 | 0.33 | 0.32 | 0.00 |
| C:colour | 0.08 | 0.14 | 0.19 | 0.00 |
| C:count | 0.08 | 0.32 | 0.30 | 0.00 |
| C:offer | 0.04 | 0.09 | 0.49 | 0.00 |
| C:ritual | 0.51 | 0.21 | 0.99 | 0.86 |
| C:taste | 0.46 | 1.00 | 1.00 | 0.00 |
| D:before | 0.01 | 0.08 | 0.09 | 0.00 |
| D:check | 0.49 | 0.48 | 0.30 | 0.00 |
| D:name | 0.00 | 0.17 | 0.17 | 0.00 |
| D:oneshot | 0.00 | 0.00 | 0.00 | 0.00 |

Bar mechanics (all three stages): 0 mouth ticks while a partner speaks, latency 6 ticks in 100 % of free replies, 27880 syllables all exactly 3 ticks, 0 garble, 0 extra turns, guided exact 1.00, 0 back-term hits on the decision paths. Learning in free use (seen items, first → last third): B 0.13 → 0.11, C 0.34 → 0.23, D 0.14 → 0.13. One-shot recall 0.00 (n = 369).

Stage E (adult corpus interleaved 50/50 with B–D, force, 2,000 dialogues, seed 0) — ARCH_v1's 20-way metric on the
292 never-heard corpus probes (top-1 chance 0.05, AUC 0.5) and the 94 familiar-set probes:

| agent | never-heard top-1 | AUC | copy (shuffled) | familiar top-1 | AUC | exact (never / familiar) |
|---|---|---|---|---|---|---|
| refv1_mem | 0.11 | 0.57 | 0.44 (0.07) | 0.17 | 0.62 | 0.00 / 0.04 |
| echo | 0.29 | 0.67 | 1.00 (0.04) | 0.27 | 0.68 | 0.00 / 0.00 |
| last | 0.16 | 0.58 | 1.00 (0.05) | 0.08 | 0.52 | 0.00 / 0.00 |
| prior | 0.05 | 0.51 | 0.06 (0.06) | 0.08 | 0.49 | 0.00 / 0.02 |
| lookup | 0.05 | 0.50 | - (-) | 0.47 | 0.80 | 0.00 / 0.63 |
| ngram | 0.06 | 0.52 | 0.10 (0.08) | 0.48 | 0.81 | 0.00 / 0.49 |
| oracle | 0.98 | 1.00 | 0.16 (0.05) | 0.73 | 0.98 | 1.00 / 1.00 |

(ARCH_v1's definition: distractors may repeat the true reply text, which is why the oracle gets 0.98 / 0.73; the new keys
`top1_distinct` / `auc_distinct` draw distractors only from different reply texts: oracle 1.00 / 1.00, echo 0.28 / 0.35,
lookup 0.05 / 0.65, ngram 0.06 / 0.64.)
The bar on E reproduces ARCH_v1's own integrated MEM run (res_mem_s0: never-heard top-1 0.121, AUC 0.573, copy 0.457 vs
0.061 shuffled) on a different test set. Inside E the bar's micro-world replies collapse (B 0.03, C 0.06, D 0.01:
interference from 1,385 corpus programs).

v1 without a teacher: `RefV1` in `tutor='model'` (stage B) and `'recast'` (stage C) never speaks (1,075 and 3,984 free
replies, all silent, 0 programs): its programs are recruited by the cue and its decision paths learn only on
teacher-driven releases. Anything above 0 in those modes is new capability.

## Reference numbers, 3 seeds (world seeds 0, 1, 2; 2,000 dialogues per stage; mean ± sd)

Free-reply acceptable rate: ALL / held-out / scene questions. Bar = `refv1_mem` with `tutor='force'`; baselines in `force`.
Bar mechanics over the 9 runs: 10,895 free replies all at latency 6, 90,173 syllables all 3 ticks, 0 overlap, 0 garble,
0 extra turns, guided exact 1.00, 0 back-term hits; one-shot recall 0.00 (n = 369 / 407 / 411).

| stage | bar (seeds) | echo | last | prior | lookup | ngram | oracle |
|---|---|---|---|---|---|---|---|
| B | 0.12±0.01 / 0.11±0.01 / 0.12±0.01 (3) | 0.00±0.00 / 0.00±0.00 / 0.00±0.00 | 0.00±0.00 / 0.00±0.00 / 0.00±0.00 | 0.14±0.01 / 0.11±0.01 / 0.14±0.01 | 0.00±0.00 / 0.00±0.00 / 0.00±0.00 | 0.14±0.02 / 0.11±0.01 / 0.14±0.02 | 1.00±0.00 / 1.00±0.00 / 1.00±0.00 |
| C | 0.19±0.01 / 0.07±0.02 / 0.09±0.01 (3) | 0.10±0.01 / 0.00±0.00 / 0.00±0.00 | 0.00±0.00 / 0.00±0.00 / 0.00±0.00 | 0.30±0.01 / 0.29±0.03 / 0.23±0.01 | 0.00±0.00 / 0.00±0.00 / 0.00±0.00 | 0.52±0.01 / 0.29±0.03 / 0.24±0.01 | 1.00±0.00 / 1.00±0.00 / 1.00±0.00 |
| D | 0.19±0.02 / 0.46±0.02 / 0.22±0.02 (3) | 0.00±0.00 / 0.00±0.00 / 0.00±0.00 | 0.00±0.00 / 0.00±0.00 / 0.00±0.00 | 0.22±0.01 / 0.50±0.01 / 0.23±0.01 | 0.00±0.00 / 0.00±0.00 / 0.00±0.00 | 0.15±0.01 / 0.15±0.05 / 0.15±0.01 | 1.00±0.00 / 1.00±0.00 / 1.00±0.00 |

| task | bar | prior | ngram |
|---|---|---|---|
| B:slot | 0.00±0.00 | 0.07±0.02 | 0.06±0.02 |
| B:what | 0.06±0.01 | 0.06±0.00 | 0.04±0.01 |
| B:where | 0.32±0.03 | 0.31±0.03 | 0.32±0.03 |
| C:colour | 0.08±0.01 | 0.16±0.03 | 0.17±0.02 |
| C:count | 0.10±0.01 | 0.31±0.01 | 0.30±0.01 |
| C:offer | 0.04±0.01 | 0.07±0.02 | 0.48±0.01 |
| C:ritual | 0.62±0.08 | 0.20±0.06 | 1.00±0.00 |
| C:taste | 0.37±0.08 | 1.00±0.01 | 1.00±0.01 |
| D:before | 0.01±0.01 | 0.08±0.01 | 0.08±0.01 |
| D:check | 0.46±0.02 | 0.48±0.01 | 0.31±0.01 |
| D:name | 0.00±0.00 | 0.12±0.04 | 0.12±0.04 |
| D:oneshot | 0.00±0.00 | 0.00±0.00 | 0.00±0.00 |
