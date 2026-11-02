## X2 status report (interim). The experiments are still running and no attribution result exists yet.

I was told to hand back now, before any arm had finished. Everything below is setup, code checks, calibration, and hypotheses taken from reading the code. **No collapse attribution and no stability verdict can be drawn yet.** Labels: PROVED = from code, counters or exact equality; SIMULATED = my runs; ASSUMED = not yet tested.

### 1. Still running (please collect, or tell me to continue)
- **Local:** one python process, `drift_x2.py tag=lasis`. It is the "as is" drift of v2.2 final (nS 500, seed 0) from my trained checkpoint, to 3,000 dialogues, with diagnostics. It saves a checkpoint at 2,100. It finishes about 30–35 min after 02:37 UTC.
  - Log: `.../arch/v2/x2/loc/lasis.log`
  - Result: `.../arch/v2/x2/out/runs/loc_drift_lasis_s0.json`
- **A800 job `x2_base`** (submitted 02:08, about 75 min): v2.2 final on seeds 0, 1, 2, plus arm **(T)** on seed 0 (`ct_k=24 ct_lam=0.75`).
  - Each run: staged at nS 1000, the relearning probe, then untutored drift to 3,000 with marks at 300/900/1,500/2,100/3,000.
  - Each run also records diagnostics and lesion probes at 2,100 and 3,000, and saves pickles of the trained agent and of the drift copy at 3,000.
  - Folder: `.../arch/v2/gpu/queue_a800/x2_base/`
- **A800 job `x2_stab1`** (submitted 02:36, about 80 min): the stability rule on GEN, same protocol.
  - `stab=0.03` on seeds 0, 1, 2.
  - `stab=0.3` on seed 0.
- **Not yet submitted:**
  - the drift forks: `rm_prune=0`, replay off during untutored use, `ral_hip_own=1`, and no forced silence. Each runs ×3 seeds from the `x2base` trained pickles. The recipe is `fork()` in `mkjobs.sh`; upload the pickles with `submit_x2.py <label> 4 90 <jobs> ck/x2base_s*_tr.pkl.gz`;
  - `stab=0.3` on seeds 1 and 2, and (T) on seeds 1 and 2;
  - one life on seed 5 per stability arm. `life_x2.py` is ready; the reference life is v22's `L_finF_s5`, from the same machine and the same code (bit-identical, see §2);
  - the best combination;
  - the 10,000-dialogue continuation, chained from the `*_d3000.pkl.gz` pickles with `drift_x2.py load=... marks=...,10000`.
- **Aggregate** with `python3 agg_x2.py <result jsons>`.

### 2. What is built (folder `.../arch/v2/x2/`, a copy of the working v22)
- **The stability rule** in `rules.Delta`, per synapse, both signs:
  - `Delta.enable_stab` and `Delta.nudge` hold the rule. The use trace u_ij counts the learning events a synapse took part in, and never decays.
  - Step factor f = max(1/(1 + β·u·|w|/w_ref), f_min).
  - It also applies to the actor's direct writes to GEN.
  - A zero-weight synapse stays fully plastic whatever its use. Unlike tune's per-row maturation, a new output or a fresh unit is therefore not starved.
- **Agent keys:**
  - `stab=β`, `stab_ref=w_ref` (0.01), `stab_min=f_min` (0.05);
  - `stab_g` applies the same rule to the copy gates and gateR;
  - `rm_stab` applies it to the random layer's instar (in `randlayer.py`);
  - `stab_track=1` keeps the traces only.
  - Files: `core_x2.py`, `agent_x2.py`.
- **Drivers:**
  - `run_x2.py` is run_v22 plus `marks=`, `dset=` (settings applied to the drifting copy only), `diag=1`, test-time lesions `les=`, `save_tr`, `save_at` and `relearn=1`.
  - `drift_x2.py` holds the shared drift loop. It resumes from pickles. It reports, per mark: hippocampus composition (live cells and replay pool, tutored-era vs untutored), the age of recall sources, thought outcomes, RAL resolution counts, online reply classes, and the failure shape (partner / silent / echo / other, plus the top wrong replies).
  - It also runs the relearning probe. The taught answer changes for 谢谢→再见, 晚安→早上好, one food 好不好吃→不好吃 and one 你要X吗→不要. It counts the recasts needed until a frozen copy says the new answer.
- **(T)** needs config keys only: `ct_k`, `ct_lam`.
- **(H′)** is coded behind `hip_dup` (default 0) but was never run, because you dropped it. The earlier useful-first (H) arm, which the designer corrected, was never built.
- **Checks (PROVED):**
  - `test_equiv_x2.py` compares against the original v22 folder's AgentV22 on the v2.2 final config, running c_flat plus the gated veto. With every key off, replies, GEN, gates, gateR, hippocampus, critic and the random layer's weights and lines are all bit-identical. This also holds with `stab_track=1`. With `stab=1.0` the runs differ, as expected.
  - At small size, a drift run reloaded from a pickle reproduces the in-process drift exactly.
  - Every smoke run had 0 mechanics violations.

### 3. Findings so far
- **Local training (SIMULATED, nS 500, seed 0, numpy 1.26):** after stage 3, B 0.947, C 0.868, D 0.362, with 0 mechanics violations. Peak memory was about 290 MB. The checkpoint is 14.1 MB gzipped, so pickles should fit the runner's 50 MB copy-back limit.
- **The pruning clock is about 10× faster than v2.2 assumed (PROVED from counters):**
  - The trained agent has 174,942 learning events after about 1,800 dialogues, roughly 97 per dialogue (awake replays included).
  - So `rm_age` 20,000 events is about 200 dialogues, not 2,000.
  - Route (c) therefore does not "coincide with the onset" as v2.2 claimed. The `rm_prune=0` fork will test whether it matters at all.
- **Leading hypothesis (ASSUMED, read from code, being tested):**
  - A hippocampal trace's strength σ decays by 0.9995 per stored line. Recall and replay both require σ > 0.05. So any episode older than about 5,990 stored lines becomes unreachable.
  - An untutored dialogue stores about 2 heard lines, which is about 3,000 dialogues. That matches the 2,100 → 3,000 cliff.
  - Once the tutored Q→answer episodes fall out of the uniform replay pool, awake recall replays only untutored Q→next-question links. These are the links that survive RAL after a silence. It then teaches partner lines as continuations.
  - Predictions:
    - the replay-off and no-silence arms remove most of the collapse;
    - `ral_hip_own=1` reduces it;
    - `rm_prune=0` does little for C;
    - the stability rule delays the collapse but does not stop it: protected answers stay high while a zero-weight partner line still rises at full rate.
  - The diagnostics `hip.live_old` and `pool_old`, and the lesion probes with the NEXT copy route off and with no hippocampal recall, will confirm or refute this.
- **Stability calibration (SIMULATED, on the trained agent with tracking on):**
  - GEN synapses that took part in learning: u median 11 (p90 211). |w| median 1.3e-4 (p90 2.1e-3, max 0.08).
  - For strong synapses (|w| ≥ p90), the median of u·|w| is 0.97.
  - With w_ref = 0.01, the strong synapses' mean step factor is 0.31 at β = 0.03 and 0.08 at β = 0.3. Those are the two strengths now running.
  - The gates have about 8× larger u·|w|. The same β would make them stiffer, with a mean of about 0.09 at β = 0.03.
- **Literature (CITED), as the form's grounding:**
  - metaplasticity: Abraham & Bear 1996, TINS 19:126;
  - cascade / multi-timescale synapses, whose plasticity falls in deeper states: Fusi, Drew & Abbott 2005, Neuron 45:599; Benna & Fusi 2016, Nat Neurosci 19:1697;
  - importance-weighted step size, where u·|w| is a local proxy for importance: EWC, Kirkpatrick et al. 2017, PNAS 114:3521; Synaptic Intelligence, Zenke et al. 2017, ICML;
  - the designer's own char-neuron network already uses concave growth, Δw ∝ (1 − w/w_max)^pow.

### 4. Not yet answered
- C accuracy per arm and seed at each mark, and at 10,000;
- the failure shape on the server;
- which route causes the collapse;
- whether the stability rule delays or stops it, and at what cost (J/J2, per-task held-out, the seed-5 life, relearning exposures);
- (T) against the collapse;
- the recommended parameters for M2.

The first server answers (`x2_base`: three as-is seeds with diagnostics, plus (T) s0) land at about 03:25 UTC. The `x2_stab1` results land at about 04:00.

### summary_zh
- X2 只完成了准备，还没有结论。
- 已做的：
  - 实现了设计者的"权重越强、越常用越难改"规则：按突触计使用次数，步长 = η / (1 + β·使用次数·|权重| / w_ref)，带下限。所有新开关默认关闭，关闭时与 v2.2 逐位一致（已验证）。
  - 实现了衰减实验的各个分支、诊断和"改答案重学"测试。
- 已提交两个 A800 任务：
  - v2.2 原样 3 个种子，加"更宽更持久的思考"(T) 1 个种子；
  - 稳定规则两种强度。
- 本地有一个原样衰减测试正在跑。
- 已发现：随机层的修剪时钟约 200 段对话，不是报告说的 2,000，所以"修剪与崩溃同时发生"的说法不成立。
- 目前最可能的原因（待验证）：海马痕迹在约 3,000 段无人指导对话后全部低于回放门槛，回放只剩"问题→下一个问题"，于是教会了说对方的话。
- (H′) 按设计者更正已撤销，没有运行。