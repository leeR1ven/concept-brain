(Saved by the coordinator from the agent's final reply; the harness blocked the agent's own write.)

## GPU port of ARCH_v2: report

**Verdict.** The batched torch port reproduces the numpy ARCH_v2 agent (final_noctx) **bit for bit**, wherever I compared it, against the reference with one documented change: exact ties in the DG k-WTA are broken by unit index. This is the one-agent match (P5.1–P5.2), and it is PROVED by exact identity tests. The batched sweep mode (P5.5) works. The speed target (P5.3) is **not met**: at 64 agents I measured about 250 agent-ticks/s against the 30k goal.

The harness blocked writing `report.md`, so this message is the report. Scope followed the coordinator's notes: P0, P1–P3, P5.1–P5.3, then P5.5. P4 and P5.4 are on hold; the V100 box was retired midway, so later jobs ran on the A800 queues.

### P0: frozen reference and traces
- **What I froze (PROVED by md5).** A byte-identical copy of `arch/v2/synth` plus the bench, in `gpuport/ref/`.
- **Seed-0 trace.**
  - Covers stage A (300 dialogues, 18,790 ticks), babbling, then 40 stage-B dialogues: 25,934 ticks in all.
  - Recorded per tick: the observation stream, mouth force, GO, released program, stop reason, the open thought (kind, step, confidence e, intention I), MIX unit count and quiet counter.
  - Also recorded: all 2,467 detections, the awake-recall picks, a digest of every learning event, and the fixed random wiring.
- **Seed-1 trace.** Stage A plus the full stage 1 (1,000 dialogues): 185,817 ticks.
- **Caps measured.** A line has at most 7 items, an event at most 332 conjunction keys; recognised words (VR) ≤ 4, focus bits ≤ 24, thought features ≤ 11, pending brake predictions ≤ 11.

**The reference is itself platform-dependent (SIMULATED).** These findings matter for anyone comparing numbers across machines:
- **DG tie-breaking.** 2.2 % of DG k-WTA calls (22 of 990) have an exact tie at the k-th place, and `np.argpartition` breaks it differently on numpy 1.26 (local) and 2.1 (server). The original reference diverged between the two machines at tick 20,362.
- **Canonical rule fixes it.** With ties broken by index (stable sort), the reference is bit-identical on both platforms. This "canonical" reference is the validation target.
- **The tie rule alone moves J.** It is a measure-zero change, yet J per seed 0–4 goes from 0.794 / 0.903 / 0.791 / 0.877 / 0.758 (mean 0.825) to 0.722 / 0.884 / 0.788 / 0.868 / 0.775 (mean 0.807). Per-seed |ΔJ| reaches 0.072, so the "±0.03 per seed" tolerance is below the reference's own perturbation noise.
- **Python version.** Python 3.12's `sum()` uses Neumaier compensated summation (3.11 does not). This changes the turn learners' output in the last bit.
- **numpy version.** numpy 1.26 adds the copy routes in float64 before rounding; numpy 2 adds them in float32.
- **A quirk in SynthCore.** `_recognize` caches its result per (slot, eye) and invalidates the cache only when the primes learn. The result also depends on `eye_rate` and on `eye_n` crossing 50, so the reference sometimes uses a stale visual recognition. This was the last discrepancy I fixed; the port now replicates the cache exactly. **Worth fixing in v2.1.**

### P1–P3: the batched core
- **Split.**
  - Control plane: numpy arrays with an agent axis B, for body, intention, turn learners, line contexts, continuous thought and record contents.
  - Data plane: torch on the GPU, for ear layer 0, syllable templates, babbling memory, MIX, the readouts, hippocampal keys and strengths, the DG, primes and word→slot attention.
- **Events become masks.** Each handler runs once per tick on the subset of agents that has that event. An `active` mask freezes paused agents completely.
- **MIX.** It sits behind one interface, `HashConj.active(rows, vals, create)`, as the coordinator asked (a randmix projection could slot in).
  - Implementation: a per-agent open-addressing hash table with no collisions. Unit ids follow MixDict's creation order.
  - The schema is data, `GROUPS` and `PAIRS`: an item-code layer, own-speech or role groups are one line each.
- **Readouts.** GEN and the 3 gates form one table of V+4 columns, plus a sink row for padding. Learning is one `index_add_` (or one fused Triton kernel, `fused=True`, also validated exact).
- **Exactness measures.**
  - Drives use a sequential fp32 Triton sum in the reference's unit order, which is numpy's axis-0 order (checked on both numpy versions).
  - float64 is used wherever the reference uses it.
  - Divisions use device tensors: torch computes `x / python_scalar` as `x * (1/s)`. That pitfall caused the first mismatch (overlap 54/120 became 0.45000002, so FAM came out 2 instead of 1).
  - The rng choice algorithm, CPython 3.12's `sum`, and lazy visual recognition with its cache are replicated.
- **Per-agent parameters** for sweeps: `PER_AGENT` lists eta, eta_g, eta_recall, eta_actor, eta_critic, theta_I, nogo, kappa, eta_oth, eta_go.

### P5.1: one agent, fp32 trace match (PROVED, exact identity)
| trace | ticks | result |
|---|---|---|
| seed 0: A + babbling + 40 B dialogues | 25,934 | bit-identical on all 12 per-tick fields, 2,467/2,467 detections, 324/324 recall picks |
| seed 1: A + babbling + full stage 1 (1,000 B dialogues) | 185,817 | bit-identical on every per-tick field, 14,868 detections, 2,920 picks, 130,267 learning events (unit count and unit-id sum each), 3,178 releases, 27,881 units |

- Both runs are open-loop replays of the recorded observation stream on the GPU.
- The seed-0 trace was re-checked after every code change: plain and fused learning paths, on the V100 and on the A800.

### P5.2: J protocol
- **Short full protocol, seeds 0–4.** All three stages, frozen probes and drift, with A 60, 3 × 40 dialogues, probes of 10 and drift 20.
  - Every probe table, online accuracy, drift result, MIX count and hippocampus count is identical to the canonical numpy runs (PROVED).
  - J per seed: 0.3353 / 0.4875 / 0.4484 / 0.3112 / 0.4047 for both.
- **Stage 1 at the real protocol size, seeds 0–4.** A 300 + B 1,000, then frozen B/C/D probes.
  - Probe accuracies (with held-out), online accuracy, MIX units, episodes, recalls, releases and mechanics are identical to canonical numpy (SIMULATED, exact equality).
  - Seeds 0, 2 and 3 matched exactly in the first run. Seeds 1 and 4 first diverged slightly, from the recognition-cache quirk. After the fix, the re-runs of seeds 1, 3 and 4 match exactly.
- **Not done.** Full-length J (3 × 1,000 dialogues) with the torch agent never finished: at about 12 ms per tick under GPU sharing, a seed needs about 3 h, beyond the 90-min job limit. J_torch = J_canon (0.722 / 0.884 / 0.788 / 0.868 / 0.775) follows from identity everywhere it was checked, but it is not separately measured at full length.

### P5.3: speed and memory (SIMULATED; target not met)
| box | B=1 | B=4 | B=16 | B=64 |
|---|---|---|---|---|
| V100 box: ms per tick / agent-ticks per s | 12.7 / 79 | 58 / 69 | 143 / 112 | 264 / 243 |
| A800 GPU queue (shared GPU) | 12.9 / 78 | 46 / 86 | 162 / 99 | 259 / 247 |

- **Best single agent.** Uncontended A800 at B=1: 2.56 ms per tick, about 390 agent-ticks/s. The numpy reference does about 2.3 ms per tick on one local core.
- **GPU memory.** 78–134 MB per agent; 5–6 GB at 64 agents.
- **Where the time goes (profile at B=64).**
  - Awake recall takes 69–70 %. Each quiet moment is 8 dependent replays of about 6 learning steps each, roughly 48 sequential learning steps.
  - The learning-step front end (candidates, keys, hash lookup) costs about 4 ms per batched call on the shared GPU.
  - Per-agent numpy loops (intention, turn learners, contexts) make cost grow almost linearly with B.
- **Conclusion (ASSUMED).** The current agent does too little, and too sequential, work per event for a GPU. To beat CPU cores it would need fused per-event kernels or CUDA graphs plus vectorising the remaining per-agent loops; the 48-step recall chain is inherent to the exact algorithm. For today's model, numpy on the A800 cores (about 10 min per staged run) is the efficient sweep engine. The GPU port pays off for scaled models (P6: millions of MIX units, a 100k-cell hippocampus, V ≈ 4k).

### P5.5: batched sweep mode (`sweep_j.py`)
- **How it works.**
  - One process runs B protocols, each with its own seed and parameters, in lockstep on one brain.
  - Each protocol's worlds run in a thread; a hub batches the agents that are waiting to step.
  - Each protocol owns 3 rows (agent, probe copy, drift copy, via `copy_row`).
  - It writes one `run_j.py`-format JSON per agent.
- **Run (SIMULATED).** 5 seeds, short protocol: 15 rows, 43,380 batched ticks, 2,840 s, 1.4 GB GPU, 0 mechanics violations.
- **Exactness.** Per-agent results equal the B=1 runs exactly, so batching itself is exact. That run used the code before the cache fix; with the fixed core, the B=1 runs equal numpy on all 5 seeds.
- **Throughput.** It is not faster than separate processes.

### Deviations from the reference
- Only the canonical DG tie-break. Everything else reproduces the local numpy 1.26 / Python 3.12 semantics.

### Ready / open
- **Ready.**
  - An exact, inspectable batched core: per-agent seeds and parameters, the conjunction interface, a data-driven schema, row copy and tiling.
  - A trace and J harness for validating any future change against numpy.
- **Hooks are there but not wired:**
  - an item-code layer in front of MIX (a new key group);
  - own-speech and role entries in the thought;
  - signed reward updates on r = 0 ticks (they go through `_reward`).
- **Open.**
  - Speed (above).
  - Full-length torch J.
  - The v2.1 port: no Step-1 report existed yet.
  - The noise-robust event clock for world2: noted, kept out of the bench path, and not built because P4 is on hold.

### Files
All in this directory (`v2/gpuport/` in the repo):
- `brain_torch.py`: the batched brain.
- `kernels.py`: Triton sequential sum, fused learning kernel, hash table.
- `agents_torch.py`: bench adapter.
- `run_j_torch.py`: J protocol for the torch agent.
- `sweep_j.py`: batched sweep mode (P5.5).
- `test_trace.py`: trace replay test (P5.1).
- `bench_speed.py`: speed benchmark (P5.3).
- `compare_j.py`: per-seed J comparison.
- `p0_trace.py`: P0 recorder.
- `run_j_canon.py`: the canonical numpy reference.
- `ref/` (not in the repo): the frozen reference copy, byte-identical to `v2/synth` (with its `bench_copy/`); the repo scripts point there instead.
- `results/`: J runs, trace summaries and logs, speed logs (the scripts write to `out/`; the repo keeps the outputs in `results/`). The per-tick trace arrays (`*.npz`, 5.6 MB) are not in the repo; `p0_trace.py` regenerates them.

## summary_zh
- 我把 ARCH_v2 移植到了 torch 批量版。在所有比对过的地方，它和 numpy 参考逐位一致：
  - seed 0 共 25,934 拍、seed 1 共 185,817 拍逐位相同；
  - 五个种子的短版 J 流程、以及第一阶段完整规模的结果，都与 numpy 完全相同。
- 唯一的改动：DG 平局按编号打破。
- 过程中发现参考版本身依赖平台：numpy 版本、Python 版本都会让结果不同；它的视觉识别缓存还会用到过期结果，建议 v2.1 修正。
- 只改平局规则，单个种子的 J 就会变动最多 0.07，所以“每个种子 ±0.03”的标准比参考本身的噪声还小。
- 批量扫参模式能用，结果和单个跑完全一致。
- 速度没达标：64 个体约每秒 250 拍，目标是 30,000。原因是每个事件的计算太小、而且必须按顺序做，主要是清醒回忆。对现在这个模型，用 A800 的 CPU 跑 numpy 扫参更划算；GPU 版适合放大后的模型。
- 完整长度的 torch J 太慢，没能跑完。world2 和 v2.1 暂缓。