# Bench contract (the coordinator's spec; the bench agent implements it and documents any deviation in README.md)

Goal: one shared, fast numpy test bench on which every ARCH_v2 prototype runs, so results are comparable and every
claim is checked against the same baselines. Speed target: a trivial agent runs 2,000 dialogues in under ~3 minutes on
one CPU core.

## Physics: the same world as the real model
- Sound and mouth come from `/root/work/concept-brain/world/voice.py`: `Voice` and `Stream`, with step = T // 3, so each
  char lasts 3 ticks. Speakers mix by taking the maximum.
- Copy the file; do not import it from /root/work.
- Options:
  - `physics='random'`: as today, each char has a random sound and a random 4-of-32 muscle set.
  - `physics='lawful'`: each muscle contributes a fixed spectro-temporal pattern, and a char's sound is a fixed function
    (e.g. a clipped sum) of its muscle set. Similar gestures then sound similar, so babbling can generalise. Keep chars
    distinguishable: check pairwise layer-0 overlap.
  - `noise=σ`: background noise in the ear window.
  - Voices: `other`, `self`, plus a third voice `other2` with its own fixed offset, for overheard dialogue between two
    other speakers.

## Senses given to the agent each tick (obs dict)
- `ear`: the (F, T) int window that `Stream.tick` returns.
- `eye`: a binary scene code `(n_eye,)`.
  - The scene has 3 slots (left / middle / right). Each slot holds an object (about 12 kinds) with a colour (6) and a
    count (1–3), or is empty.
  - Encoding: a fixed sparse random code per (slot, attribute value), unioned, plus an "attended slot" pulse when the
    tutor points or looks.
  - All zeros when there is no scene.
- `reward`: a float. The partner's contingent reaction (e.g. +1 when the agent's reply was acceptable, a small value when
  a babble is imitated back), 0 otherwise. The agent may use it as a third factor or ignore it.
- `cue`: `(32,)` muscle cue or None, only when `tutor='force'` (v1-compatible teacher forcing, for baselines and
  ablations).
- Nothing else. Text, char identities and timing tags are hidden from the agent.

## Agent interface
`agent.step(obs) -> muscles` (a (32,) array; > 0 = active). The world turns the muscles into the own voice through
`Voice.mouth` and plays it back into the ear one tick later, as `talk.py` does. Also provide:
- `bench/frontend.py`: the exact fixed ear layer 0 and FEAT0, as in `arch/content/fe.py` / `arch/synth/archv1.py`.
- `bench/body.py`: ARCH_v1's body as a reusable module. A "mind" supplies a score per program and for END at each
  boundary; the body handles F, GO, CD, beat, WTA, D/I/M and the counter, and exposes its signals (F, GO, boundary flags,
  the current program, own-voice tags) to the mind.
- Program creation must be possible WITHOUT a teacher cue, e.g. from babbling: random muscle sets tried, kept when
  heard as a char. Keep the v1 cue-based recruitment as an option.

Agents may use frontend.py and body.py or their own versions.

## World / tutor
`World(stage, seed=0, physics='random', noise=0.0, tutor='model', vocab=None, params...)`.

Stages (develop and compute reports):
- **A — hear and babble.** Child-directed utterances and overheard two-speaker routines, with no reply demanded. Free
  "play" periods: whatever the agent's mouth produces, the tutor imitates back after a short gap (reward small +).
- **B — naming with a scene.**
  - "这是什么" → object, with the attended-slot pulse.
  - "左边是什么" → object, by slot.
  - "苹果在哪里" → "在左边".
- **C — item-based frames with fillers and held-out combinations.** About 30 % of filler × frame pairs are never
  heard; the fillers appear in other frames.
  - "你要X吗" → "要X" / "我要X"
  - "X好不好吃" → "好吃"
  - "苹果是什么颜色" → "红色"
  - "有几个苹果" → "两个"
  - rituals: 你好 / 谢谢 / 晚安 …
- **D — relations and memory.**
  - "这是X吗" → "是" / "不是".
  - Multi-turn: "我叫N" … later "我叫什么" → "N".
  - A changed scene: "刚才左边是什么".
  - One-shot new object: "这是X" heard once, asked later.
- **E — the adult corpus** (`/root/work/corpus/dialogs.jsonl`), interleaved with earlier stages.

Vocabulary: ~100–300 chars, all present in the corpus, so the same Voice works in stage E.

Tutor modes:
- `model`: after the partner's line, a second speaker (`other2`) or the partner itself says the correct reply. The agent
  overhears it; the agent may or may not speak.
- `recast`: the agent answers freely. The tutor then gives a contingent reaction (`reward`) and says the correct reply
  (a recast).
- `force`: v1 teacher forcing via `cue`.

Turn protocol: ≥ 45 ticks between two lines of the same speaker unless the agent is answering. PAUSE ≥ 5 after an END.
A fixed seed gives identical streams.

## Logging and metrics
- `world.log`: one record per exchange: stage, task, prompt text, scene, which speaker, the agent's produced reply
  (chars decoded from its own mouth output), the acceptable set, tags (held_out, familiar, n_exposures, one_shot,
  ticks_since_taught), and timing (prompt end, first mouth tick, stop tick).
- `evaluate(log)` → a dict, per stage and task:
  - acceptable-reply rate (exact match to any acceptable string; also first-char-correct);
  - split by held-out vs seen;
  - echo-free variant (the reply's chars that appear in the prompt are removed before scoring);
  - parrot index: the fraction of reply chars that are in the prompt;
  - one-shot recall and retention vs `ticks_since_taught`;
  - mechanics: mouth ticks while a partner speaks, latency, chars not 3 ticks long, garble, stop reason, extra turns;
  - learning in free use: the same items early vs late with no `force` cues.
- `baselines(stage, seed)` → the scores of:
  - echo the prompt;
  - say the last heard char;
  - the most frequent acceptable reply of the task type;
  - an exact-lookup table (prompt + scene → a reply heard before);
  - a char n-gram predictor of the reply given the prompt;
  - an oracle (1.0).
  The bench is valid only if echo and lookup fail on held-out items and scene questions, and the oracle passes.

## Deliverables
- `bench/README.md` with usage, `bench/READY` once the API works end-to-end with a trivial agent and the baselines.
- `bench/selftest.py`: reproducible checks of physics, timing, metrics and baselines.
- A v1-equivalent reference agent built from body.py + T + P→Y + echo, with its scores on stages B–D. This is "the bar".
